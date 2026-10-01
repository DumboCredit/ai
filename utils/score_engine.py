"""Motor determinista de simulacion de puntaje.

Estima el impacto en puntos de una accion sobre el reporte de credito sin llamar
a ningun LLM: parsea los documentos que ya estan en Chroma a un perfil por buro,
aplica la accion sobre una copia de ese perfil y compara la "salud" ponderada de
los cinco factores FICO antes y despues.

No es FICO ni VantageScore (correr esos modelos exige licencia del buro): es una
estimacion propia con bandas calibradas contra los rangos publicos -- un primer
atraso de 30 dias cuesta 60-110 pts segun el perfil de partida, un hard inquiry
menos de 5. El diseno reproduce el hallazgo central de FICO, que el impacto
depende del perfil de partida, y eso sale solo del modelo: cada factor arranca de
su propia salud, asi que el que ya tiene morosidades pierde poco al sumar otra y
el limpio se desploma.

El motor es puro: no hace I/O, no importa nada del proyecto y trabaja sobre
dataclasses, para poder probarlo contra los reportes guardados sin levantar la
app.
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field
from datetime import date, timedelta

# %% Constantes del modelo %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Pesos FICO publicados.
WEIGHTS = {
    "payment_history": 0.35,
    "utilization": 0.30,
    "credit_age": 0.15,
    "credit_mix": 0.10,
    "inquiries": 0.10,
}

# El dano va mas rapido que la reparacion: misma diferencia de salud pesa mas
# cuando empeora. Calibrado para que un primer atraso de 30 dias en un perfil
# limpio caiga dentro de los 60-110 pts que publica FICO.
SENSITIVITY_DOWN = 1.0
SENSITIVITY_UP = 0.9

SCORE_FLOOR = 300
SCORE_CEILING = 850

# Tipos de cuenta (prefijo del documento o loanType traducido).
_REVOLVING_HINTS = ("tarjeta de credito", "cuenta de cargo", "linea de credito", "creditcard", "chargeaccount")
_MORTGAGE_HINTS = ("hipoteca", "mortgage", "bienes raices")
_INSTALLMENT_HINTS = (
    "prestamo", "prestamos", "installment", "unsecured", "lease", "auto",
    "estudiantil", "educational", "note", "secured",
)

# %% Perfil %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

@dataclass
class Account:
    ref: str
    account_id: str
    creditor: str
    raw_type: str
    kind: str                                     # revolving | installment | mortgage | other
    account_number: str = ""
    tradeline_id: str = ""
    is_open: bool = True
    balance: float = 0.0
    limit: float = 0.0
    late30: int = 0
    late60: int = 0
    late90: int = 0
    is_collection: bool = False
    is_chargeoff: bool = False
    opened_at: date | None = None
    last_activity: date | None = None
    bureaus: set[str] = field(default_factory=set)
    tradeline_bureaus: set[str] = field(default_factory=set)

    @property
    def utilization(self) -> float | None:
        if self.kind != "revolving" or self.limit <= 0:
            return None
        return self.balance / self.limit

    @property
    def is_derogatory(self) -> bool:
        return self.is_collection or self.is_chargeoff or bool(self.late30 or self.late60 or self.late90)


@dataclass
class Inquiry:
    ref: str
    name: str
    bureau: str
    date: date | None = None
    inquiry_id: str = ""            # id del reporte v3, igual entre buros (opcional)


@dataclass
class Profile:
    bureau: str
    score: int | None
    accounts: list[Account] = field(default_factory=list)
    inquiries: list[Inquiry] = field(default_factory=list)
    today: date = field(default_factory=date.today)


# %% Parseo de los documentos de Chroma %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

_NUM = r"(-?[\d.,$]+)"


def _money(raw: str | None) -> float:
    """'$1,234.56' / '1234' / '0.0' -> float. Devuelve 0.0 si no se puede."""
    if not raw:
        return 0.0
    cleaned = raw.replace("$", "").replace(",", "").rstrip(".")
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def _find(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(1) if match else None


def _find_int(pattern: str, text: str) -> int:
    raw = _find(pattern, text)
    return int(_money(raw)) if raw else 0


def _find_date(pattern: str, text: str) -> date | None:
    raw = _find(pattern, text)
    if not raw:
        return None
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", raw)
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


_FIELD_LABELS = (
    "Ciudad del acreedor", "Estado del acreedor", "Codigo postal del acreedor",
    "Direccion del acreedor", "Telefono del acreedor", "ID de la cuenta",
    "En los buros de credito", "En el Buro de Credito", "Buro de Credito",
    "Acreedor/Cuenta", "Numero de cuenta", "Saldo alto",
    "Saldo", "Limite crediticio", "Responsabilidad", "Meses examinados",
    "Recuento de Plazos", "Estado de la cuenta", "Estado de pago de la cuenta",
    "Importe Vencido", "Importe vencido", "Cantidad de Pago Mensual",
    "Cantidad de pago mensual", "Pagos atrasados", "Total de pagos", "Queda el",
    "Porcentaje de utilizacion", "Porcentaje de pagos a tiempo", "La cuenta esta",
    "Fecha de apertura de la cuenta", "Fecha de cierre de la cuenta",
    "Fecha de ultima actividad", "Tipo de fuente de plazo", "Tipo de Fuente de Plazo",
    "Nombre del acreedor",
    # cobranzas v3
    "Estado", "Monto", "Fecha reportada", "Agencia de cobro", "Acreedor original",
)
_NEXT_LABEL = re.compile(r"\s*(?:" + "|".join(re.escape(label) for label in _FIELD_LABELS) + r"):")


def _text_field(doc: str, label: str) -> str | None:
    """Valor de un campo de texto, cortando en el SIGUIENTE campo conocido.

    No se puede cortar en el primer punto: los nombres de acreedor traen puntos
    ("U.S. BANK" se quedaba en "U"), y con el nombre truncado el mismo tradeline
    no se reconocia entre buros.
    """
    match = re.search(re.escape(label) + r":\s*", doc)
    if not match:
        return None
    rest = doc[match.end():]
    end = _NEXT_LABEL.search(rest)
    value = rest[: end.start()] if end else rest
    return value.strip().rstrip(".").strip() or None


def _classify(raw_type: str, has_limit: bool) -> str:
    low = raw_type.lower()
    if any(h in low for h in _MORTGAGE_HINTS):
        return "mortgage"
    if any(h in low for h in _REVOLVING_HINTS):
        return "revolving"
    if any(h in low for h in _INSTALLMENT_HINTS):
        return "installment"
    # Sin pista textual: un limite crediticio real delata una revolvente.
    return "revolving" if has_limit else "other"


def _bureaus_of(doc: str, meta: dict) -> set[str]:
    listed = _find(r"En los buros de credito:\s*(.*?)\.\s*$", doc)
    if listed:
        found = {b.strip() for b in listed.split(",") if b.strip()}
        if found:
            return found
    single = _find(r"Buro de Credito:\s*([A-Za-z ]+)", doc)
    if single:
        return {single.strip()}
    repo = (meta or {}).get("credit_repository")
    return {repo} if repo else set()


def _parse_account(doc: str, meta: dict, ref: str) -> Account:
    raw_type = doc.split(":", 1)[0].strip()
    limit = _money(_find(r"Limite crediticio:\s*" + _NUM, doc))
    balance = _money(_find(r"Saldo:\s*" + _NUM, doc))

    # El campo LATE_COUNT llega en 0 en los reportes reales; la morosidad viene
    # en el estado de la cuenta ("Late30Days", "Collection/Charge-off"). Hay que
    # leer los dos o el motor no ve ni una sola cuenta mala.
    late30 = _find_int(r"Pagos atrasados por 30 dias:\s*" + _NUM, doc)
    late60 = _find_int(r"Pagos atrasados por 60 dias:\s*" + _NUM, doc)
    late90 = _find_int(r"Pagos atrasados por 90 dias:\s*" + _NUM, doc)
    generic_late = _find_int(r"Pagos atrasados:\s*" + _NUM, doc)
    if not (late30 or late60 or late90) and generic_late:
        late30 = generic_late

    status = (_text_field(doc, "Estado de la cuenta")
              or _text_field(doc, "Estado de pago de la cuenta")
              or "")
    status_low = status.lower()
    if "late30" in status_low or "30 dias" in status_low:
        late30 = max(late30, 1)
    if "late60" in status_low or "60 dias" in status_low:
        late60 = max(late60, 1)
    if "late90" in status_low or "late120" in status_low or "90 dias" in status_low:
        late90 = max(late90, 1)

    is_collection = "es un collection" in doc.lower() or "collection" in status_low
    is_chargeoff = "es un charge off" in doc.lower() or "charge-off" in status_low or "chargeoff" in status_low

    low = doc.lower()
    # v3 dice "cerrado" y legacy "cerrada"; la fecha de cierre tambien lo delata.
    closed = "la cuenta esta cerrad" in low or "fecha de cierre de la cuenta" in low
    account_number = _find(r"Numero de cuenta:\s*(\S+?)\.", doc) or ""
    account_id = _find(r"ID de la cuenta:\s*(\S+?)\.", doc) or account_number or ref
    creditor = (_text_field(doc, "Nombre del acreedor")
                or _text_field(doc, "Acreedor/Cuenta")
                or raw_type)

    return Account(
        ref=ref,
        account_id=account_id,
        creditor=creditor.strip(),
        raw_type=raw_type,
        kind=_classify(raw_type, has_limit=limit > 0),
        is_open=not closed,
        balance=balance,
        limit=limit,
        late30=late30,
        late60=late60,
        late90=late90,
        is_collection=is_collection,
        is_chargeoff=is_chargeoff,
        account_number=account_number,
        tradeline_id=str((meta or {}).get("tradeline_id") or ""),
        opened_at=_find_date(r"Fecha de apertura de la cuenta:\s*(\S+)", doc),
        last_activity=_find_date(r"Fecha de ultima actividad:\s*(\S+)", doc),
        bureaus=_bureaus_of(doc, meta),
    )


def _parse_collection(doc: str, meta: dict, ref: str) -> Account:
    """Cobranza v3 (source "Collection") como tradeline derogatorio.

    En legacy la cobranza llega como cuenta con estado "Collection/Charge-off" y
    la lee _parse_account; en v3 viene en su propia seccion del reporte, y sin
    esto el simulador no veia ninguna cobranza de un usuario v3: "elimina mis
    cobranzas" daba 0 puntos.
    """
    status = (_text_field(doc, "Estado") or "").lower()
    account_number = _find(r"Numero de cuenta:\s*(\S+?)\.", doc) or ""
    tradeline_id = str((meta or {}).get("tradeline_id") or "")
    creditor = (_text_field(doc, "Agencia de cobro")
                or _text_field(doc, "Nombre del acreedor")
                or _text_field(doc, "Acreedor original")
                or "Cobranza")
    return Account(
        ref=ref,
        account_id=tradeline_id or account_number or ref,
        creditor=creditor.strip(),
        raw_type="Cobranza (collection)",
        kind="other",
        is_open="cerrad" not in status,
        balance=_money(_find(r"Monto:\s*" + _NUM, doc)),
        is_collection=True,
        account_number=account_number,
        tradeline_id=tradeline_id,
        # La fecha reportada es la que fija la recencia de la cobranza.
        last_activity=_find_date(r"Fecha reportada:\s*(\S+)", doc),
        bureaus=_bureaus_of(doc, meta),
    )


def _tradeline_key(a: Account) -> tuple:
    """Identidad de un tradeline al margen del buro que lo reporta.

    Si el reporte trae un identificador estable (v3 lo manda igual en los tres
    buros, y llega en la metadata como `tradeline_id`), esa es la clave y la
    agrupacion es exacta.

    Sin el, hay que deducirla: la misma cuenta llega hasta cuatro veces (un
    documento consolidado que lista los tres buros y uno por buro), con el
    acreedor escrito distinto ("AMERICAN EXPRESS" vs "AMEX"), el numero enmascarado
    a distinta longitud y hashes diferentes. Lo unico estable ahi es la fecha de
    apertura con los primeros digitos del numero, y se cae a saldo/limite cuando
    falta el numero.
    """
    if a.tradeline_id:
        return ("tid", a.tradeline_id)
    digits = re.sub(r"\D", "", a.account_number)[:5]
    opened = a.opened_at.isoformat() if a.opened_at else ""
    if opened and digits:
        return ("num", opened, digits)
    if opened:
        return ("bal", opened, round(a.balance), round(a.limit))
    creditor = re.sub(r"[^a-z0-9]", "", a.creditor.lower())[:6]
    return ("cred", creditor, round(a.balance), round(a.limit))


def parse_scores(documents: list[str], metadatas: list[dict]) -> dict[str, int]:
    """{buro: puntaje mas reciente}.

    v3 guarda el puntaje vigente en un documento sin fecha (field="score") y el
    historial en documentos "Valor en la fecha ..."; el formato legacy solo trae
    historial. Gana el vigente y, si no hay, el historico mas reciente.
    """
    best: dict[str, tuple[tuple[int, str], int]] = {}
    for doc, meta in zip(documents, metadatas):
        meta = meta or {}
        if meta.get("source") != "CreditScore":
            continue
        bureau = meta.get("credit_repository")
        if not bureau:
            continue
        if meta.get("field") == "score":
            raw, key = _find(r"Puntaje de Credito:\s*(\d+)", doc), (1, "")
        else:
            raw, key = _find(r"Valor en la fecha.*?:\s*(\d+)", doc), (0, str(meta.get("date") or ""))
        if not raw:
            continue
        if bureau not in best or key > best[bureau][0]:
            best[bureau] = (key, int(raw))
    return {bureau: value for bureau, (_, value) in best.items()}


def _merge_equivalent_groups(groups: dict[tuple, list[Account]]) -> None:
    """Segunda pasada: une grupos que son el mismo tradeline con otra fecha o numero.

    Algunos buros reportan la misma cuenta con el numero enmascarado distinto
    ("440066XXXX" vs "XXXX"), asi que la clave primaria no los une y el buro
    terminaba contando dos veces el mismo saldo y el mismo limite. La huella que si
    aguanta es apertura + saldo + limite; el nombre del acreedor no sirve porque
    cada buro lo abrevia a su manera ("BANK OF AMERICA" vs "BK OF AMER"). Se exige
    que los buros de un grupo esten contenidos en los del otro: asi se absorbe el
    documento de un buro en el consolidado, pero dos cuentas distintas que
    casualmente coincidan en las tres cifras y esten en los mismos buros no se unen.
    """
    shapes: dict[tuple, set[tuple]] = {}
    for key, members in groups.items():
        if key[0] == "tid":
            continue  # ya identificado por el reporte: no hay nada que deducir
        for member in members:
            if member.balance <= 0 or member.limit <= 0 or not member.opened_at:
                continue
            shape = (member.opened_at.isoformat(), round(member.balance), round(member.limit))
            shapes.setdefault(shape, set()).add(key)

    for candidate_keys in shapes.values():
        keys = [key for key in candidate_keys if key in groups]
        if len(keys) < 2:
            continue
        keys.sort(key=lambda k: -len({b for m in groups[k] for b in m.bureaus}))
        target = keys[0]
        target_bureaus = {b for m in groups[target] for b in m.bureaus}
        for key in keys[1:]:
            bureaus = {b for m in groups[key] for b in m.bureaus}
            if bureaus <= target_bureaus:
                groups[target].extend(groups.pop(key))


def parse_profiles(
    documents: list[str],
    metadatas: list[dict],
    today: date | None = None,
) -> tuple[dict[str, Profile], list[Account], list[Inquiry]]:
    """Construye un perfil por buro y las listas de tradelines y consultas unicas.

    La misma cuenta, cobranza o consulta llega una vez por buro (y en legacy,
    ademas, en un documento consolidado). Se agrupan para que compartan un ref:
    por el `id` del reporte cuando viene, y por deduccion cuando no.
    """
    today = today or date.today()
    scores = parse_scores(documents, metadatas)

    accounts: list[Account] = []
    inquiries: list[Inquiry] = []
    for doc, meta in zip(documents, metadatas):
        meta = meta or {}
        source = meta.get("source")
        if source == "CreditLiability" and meta.get("field") in (None, "liability"):
            accounts.append(_parse_account(doc, meta, ref=f"A{len(accounts) + 1}"))
        elif source == "Collection":
            accounts.append(_parse_collection(doc, meta, ref=f"A{len(accounts) + 1}"))
        elif source == "CreditInquiry":
            inquiries.append(Inquiry(
                ref=f"I{len(inquiries) + 1}",
                name=(_find(r"Consulta:\s*(.*?);", doc) or "Consulta").strip(),
                bureau=meta.get("credit_repository") or (_find(r"Buro de Credito:\s*([A-Za-z ]+)", doc) or "").strip(),
                date=_find_date(r"Fecha:\s*(\S+)", doc) or _find_date(r"(\d{4}-\d{2}-\d{2})", str(meta.get("date"))),
                inquiry_id=str(meta.get("inquiry_id") or ""),
            ))

    # Si no se agrupan, una accion sobre "la tarjeta de Capital One" solo tocaria
    # el buro cuyo documento eligio el LLM. Ver _tradeline_key.
    groups: dict[tuple, list[Account]] = {}
    for account in accounts:
        groups.setdefault(_tradeline_key(account), []).append(account)

    _merge_equivalent_groups(groups)

    canonical: list[Account] = []
    for index, key in enumerate(sorted(groups, key=lambda k: tuple(str(part) for part in k)), start=1):
        members = groups[key]
        ref = f"A{index}"
        all_bureaus = {b for m in members for b in m.bureaus}
        for member in members:
            member.ref = ref
            member.tradeline_bureaus = all_bureaus
        # Representante del tradeline para mostrarlo una sola vez (al LLM, al front):
        # el documento que lista mas buros y con el nombre de acreedor mas completo.
        canonical.append(max(members, key=lambda m: (len(m.bureaus), len(m.creditor))))
    canonical.sort(key=lambda a: (a.creditor.lower(), a.ref))

    # Con id la agrupacion es exacta; sin el, nombre + fecha, que falla cuando cada
    # buro escribe distinto al acreedor ("WELLSFARGO" vs "WELLS FARGO-PL&L").
    inquiry_groups: dict[tuple, list[Inquiry]] = {}
    for inquiry in inquiries:
        if inquiry.inquiry_id:
            key = ("iid", inquiry.inquiry_id)
        else:
            key = ("name", inquiry.name.lower(), inquiry.date.isoformat() if inquiry.date else "")
        inquiry_groups.setdefault(key, []).append(inquiry)

    def _inquiry_order(key: tuple) -> tuple:
        first = inquiry_groups[key][0]
        return (first.date.isoformat() if first.date else "", first.name.lower(), key)

    canonical_inquiries: list[Inquiry] = []
    for index, key in enumerate(sorted(inquiry_groups, key=_inquiry_order), start=1):
        members = inquiry_groups[key]
        for member in members:
            member.ref = f"I{index}"
        representative = copy.deepcopy(members[0])
        representative.bureau = ", ".join(sorted({m.bureau for m in members if m.bureau}))
        canonical_inquiries.append(representative)

    bureaus = set(scores) | {b for a in accounts for b in a.bureaus} | {i.bureau for i in inquiries if i.bureau}
    bureaus = {b for b in bureaus if b}

    profiles: dict[str, Profile] = {}
    for bureau in sorted(bureaus):
        seen: set[str] = set()
        bureau_accounts = []
        # El documento propio del buro es mas preciso que el consolidado, que
        # promedia los tres; se ordena para que gane el mas especifico.
        candidates = sorted(
            (a for a in accounts if bureau in a.bureaus),
            key=lambda a: (len(a.bureaus), a.ref),
        )
        for account in candidates:
            if account.ref in seen:
                continue
            seen.add(account.ref)
            bureau_accounts.append(copy.deepcopy(account))
        profiles[bureau] = Profile(
            bureau=bureau,
            score=scores.get(bureau),
            accounts=bureau_accounts,
            inquiries=[copy.deepcopy(i) for i in inquiries if i.bureau == bureau],
            today=today,
        )
    return profiles, canonical, canonical_inquiries


# %% Salud por factor %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def _interp(x: float, anchors: list[tuple[float, float]]) -> float:
    if x <= anchors[0][0]:
        return anchors[0][1]
    for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
        if x <= x1:
            if x1 == x0:
                return y1
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return anchors[-1][1]


def _months_between(start: date | None, end: date) -> float:
    if start is None:
        return 0.0
    return max(0.0, (end - start).days / 30.44)


def _recency_weight(months: float) -> float:
    """Una morosidad reciente pesa el doble que una de hace cuatro anos."""
    return _interp(months, [(0, 1.0), (12, 1.0), (24, 0.75), (48, 0.5), (84, 0.3)])


def h_payment_history(p: Profile) -> float:
    """Peor item + cantidad, con decaimiento hiperbolico y peso por recencia.

    Domina el peor item (una coleccion pesa mucho mas que un atraso de 30 dias) y
    la cantidad entra en logaritmo: el salto grande es pasar de "ninguna" a
    "alguna", y de ahi cada item adicional suma poco. La curva nunca toca un piso
    duro; con un piso, los perfiles muy danados (12 cobranzas) quedaban planos y
    limpiarlas salia 0 puntos, que es justo el caso de uso del negocio.
    """
    severities = []
    for a in p.accounts:
        worst = 0.0
        if a.late30:
            worst = max(worst, 1.0)
        if a.late60:
            worst = max(worst, 1.8)
        if a.late90:
            worst = max(worst, 2.6)
        if a.is_collection or a.is_chargeoff:
            worst = max(worst, 4.0)
        if worst <= 0:
            continue
        severities.append(worst * _recency_weight(_months_between(a.last_activity or a.opened_at, p.today)))

    if not severities:
        return 1.0
    units = max(severities) + 0.5 * math.log1p(len(severities) - 1)
    return 0.05 + 0.52 / (1 + 0.28 * (units - 1))


def h_utilization(p: Profile) -> float:
    revolving = [a for a in p.accounts if a.kind == "revolving" and a.limit > 0 and a.is_open]
    total_limit = sum(a.limit for a in revolving)
    if total_limit <= 0:
        return 0.6  # sin revolventes no hay ratio que medir: neutro-bajo
    ratio = sum(a.balance for a in revolving) / total_limit
    health = _interp(ratio, [
        (0.0, 0.95), (0.05, 1.0), (0.09, 0.98), (0.30, 0.74),
        (0.50, 0.55), (0.75, 0.32), (0.90, 0.18), (1.0, 0.08), (1.5, 0.05),
    ])
    maxed = sum(1 for a in revolving if a.balance / a.limit >= 0.9)
    return max(0.03, health - 0.12 * (maxed / len(revolving)))


def h_credit_age(p: Profile) -> float:
    ages = [_months_between(a.opened_at, p.today) for a in p.accounts if a.opened_at]
    if not ages:
        return 0.3
    average = sum(ages) / len(ages)
    oldest = max(ages)
    h_avg = _interp(average, [(0, 0.15), (12, 0.32), (24, 0.50), (48, 0.68), (84, 0.85), (120, 0.95), (180, 1.0)])
    h_old = _interp(oldest, [(0, 0.15), (24, 0.40), (60, 0.65), (120, 0.88), (240, 1.0)])
    return 0.7 * h_avg + 0.3 * h_old


def h_credit_mix(p: Profile) -> float:
    if not p.accounts:
        return 0.2
    health = 0.45
    if any(a.kind == "revolving" for a in p.accounts):
        health += 0.25
    if any(a.kind == "installment" for a in p.accounts):
        health += 0.20
    if any(a.kind == "mortgage" for a in p.accounts):
        health += 0.10
    return min(1.0, health)


def h_inquiries(p: Profile) -> float:
    """Credito nuevo: consultas duras de 12 meses + cuentas abiertas hace poco.

    Sin el segundo termino, abrir tres tarjetas salia *positivo* en perfiles con
    utilizacion alta (el limite nuevo diluye el ratio) y el endpoint terminaba
    recomendando justo lo que hunde a un perfil nuevo.
    """
    recent_inquiries = sum(1 for i in p.inquiries if i.date and _months_between(i.date, p.today) <= 12)
    recent_accounts = sum(1 for a in p.accounts if a.opened_at and _months_between(a.opened_at, p.today) <= 6)
    load = recent_inquiries + 1.5 * recent_accounts
    anchors = [(0, 1.0), (1, 0.90), (2, 0.78), (3, 0.66), (4, 0.56), (5, 0.48), (6, 0.42), (12, 0.15)]
    return _interp(load, anchors)


FACTORS = {
    "payment_history": h_payment_history,
    "utilization": h_utilization,
    "credit_age": h_credit_age,
    "credit_mix": h_credit_mix,
    "inquiries": h_inquiries,
}


def factor_health(p: Profile) -> dict[str, float]:
    return {name: fn(p) for name, fn in FACTORS.items()}


def weighted_health(health: dict[str, float]) -> float:
    return sum(WEIGHTS[name] * value for name, value in health.items())


# %% Acciones %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def _attr(action, name, default=None):
    """Lee el campo de una accion venga como pydantic, dataclass o dict."""
    if isinstance(action, dict):
        value = action.get(name, default)
    else:
        value = getattr(action, name, default)
    return default if value is None else value


def _resolve_accounts(p: Profile, action) -> list[Account]:
    ref = str(_attr(action, "account_ref", "") or "").strip()
    creditor = str(_attr(action, "creditor", "") or "").strip().lower()
    apply_to_all = bool(_attr(action, "apply_to_all", False))
    kind = str(_attr(action, "account_kind", "") or "").strip().lower()

    if apply_to_all:
        pool = p.accounts
        if kind in ("revolving", "installment", "mortgage", "other"):
            pool = [a for a in pool if a.kind == kind]
        action_type = str(getattr(_attr(action, "type", ""), "value", _attr(action, "type", "")))
        if action_type == "remove_account":
            # Sin cuenta concreta, "eliminar cuentas" solo puede significar las
            # derogatorias. Devolver el pool entero borraba el reporte completo y
            # el impacto salia negativo por perdida de mix y antiguedad.
            return [a for a in pool if a.is_collection or a.is_chargeoff]
        return list(pool)
    if ref:
        exact = [a for a in p.accounts if a.ref == ref]
        if exact:
            return exact
    if creditor:
        matches = [a for a in p.accounts if creditor in a.creditor.lower()]
        if matches:
            return matches
    return []


def _default_new_limit(p: Profile) -> float:
    """Limite plausible de una tarjeta nueva cuando el front no especifica uno.

    Con limite 0 la cuenta nueva quedaba fuera del ratio de utilizacion y la
    simulacion solo contaba el castigo por credito nuevo, nunca la dilucion del
    ratio. Un perfil danado recibe limites de tarjeta inicial o asegurada; uno sano,
    algo proporcional a lo que ya maneja.
    """
    derogatory = any(a.is_collection or a.is_chargeoff for a in p.accounts)
    if derogatory or (p.score or 0) < 620:
        return 500.0
    limits = [a.limit for a in p.accounts if a.kind == "revolving" and a.limit > 0]
    if not limits:
        return 1000.0
    return min(5000.0, max(500.0, 0.25 * (sum(limits) / len(limits))))


def apply_actions(p: Profile, actions: list) -> tuple[Profile, list[str]]:
    """Devuelve una copia del perfil con las acciones aplicadas + avisos.

    Una accion sobre una cuenta que ese buro no reporta simplemente no se aplica
    a ese buro: de ahi salen los deltas distintos por buro.
    """
    after = copy.deepcopy(p)
    warnings: list[str] = []

    for action in actions:
        # .value ANTES de str(): con un str-Enum de pydantic, str() devuelve
        # "SimulateActionTypeEnum.REMOVE_ACCOUNT" y ninguna accion hacia match.
        raw_kind = _attr(action, "type", "")
        kind = str(getattr(raw_kind, "value", raw_kind))
        amount = float(_money(str(_attr(action, "amount", 0) or 0)))
        targets = _resolve_accounts(after, action)
        needs_target = kind in (
            "pay_down_balance", "increase_balance", "change_credit_limit",
            "remove_account", "remove_late_payments", "add_late_payment", "close_account",
        )
        if needs_target and not targets:
            if kind == "remove_account" and bool(_attr(action, "apply_to_all", False)):
                warnings.append(f"{after.bureau}: no hay cuentas en cobranza que eliminar")
            else:
                warnings.append(f"{after.bureau}: no se encontro la cuenta para '{kind}'")
            continue

        if kind == "pay_down_balance":
            for a in targets:
                a.balance = max(0.0, a.balance - amount) if amount else 0.0
        elif kind == "increase_balance":
            for a in targets:
                a.balance = a.balance + amount if amount else (a.limit or a.balance)
        elif kind == "max_out_cards":
            pool = targets or [a for a in after.accounts if a.kind == "revolving" and a.limit > 0]
            for a in pool:
                a.balance = a.limit
        elif kind == "change_credit_limit":
            new_limit = float(_money(str(_attr(action, "new_limit", 0) or 0)))
            for a in targets:
                a.limit = new_limit if new_limit else max(0.0, a.limit + amount)
        elif kind == "remove_account":
            ids = {a.account_id for a in targets}
            after.accounts = [a for a in after.accounts if a.account_id not in ids]
        elif kind == "remove_late_payments":
            for a in targets:
                a.late30 = a.late60 = a.late90 = 0
                a.is_collection = a.is_chargeoff = False
        elif kind == "add_late_payment":
            days = int(_attr(action, "days_late", 30) or 30)
            count = max(1, int(_attr(action, "count", 1) or 1))
            for a in targets:
                if days >= 90:
                    a.late90 += count
                elif days >= 60:
                    a.late60 += count
                else:
                    a.late30 += count
                a.last_activity = after.today
        elif kind == "close_account":
            for a in targets:
                a.is_open = False
        elif kind == "open_account":
            account_kind = str(_attr(action, "account_kind", "revolving") or "revolving").lower()
            if account_kind not in ("revolving", "installment", "mortgage", "other"):
                account_kind = "revolving"
            new_limit = float(_money(str(_attr(action, "new_limit", 0) or 0)))
            if account_kind == "revolving" and new_limit <= 0:
                new_limit = _default_new_limit(after)
            after.accounts.append(Account(
                ref=f"NEW{len(after.accounts) + 1}",
                account_id=f"new_{len(after.accounts) + 1}",
                creditor="Cuenta nueva simulada",
                raw_type=account_kind,
                kind=account_kind,
                is_open=True,
                balance=amount,
                limit=new_limit if account_kind == "revolving" else 0.0,
                opened_at=after.today,
                last_activity=after.today,
                bureaus={after.bureau},
            ))
            # Abrir cuenta deja hard inquiry: es parte del costo de la accion.
            after.inquiries.append(Inquiry(
                ref=f"NEWI{len(after.inquiries) + 1}",
                name="Consulta por cuenta nueva simulada",
                bureau=after.bureau,
                date=after.today,
            ))
        elif kind == "remove_inquiry":
            inquiry_ref = str(_attr(action, "inquiry_ref", "") or "").strip()
            count = int(_attr(action, "count", 0) or 0)
            if inquiry_ref:
                before_n = len(after.inquiries)
                after.inquiries = [i for i in after.inquiries if i.ref != inquiry_ref]
                if len(after.inquiries) == before_n:
                    warnings.append(f"{after.bureau}: no reporta la consulta {inquiry_ref}")
            else:
                recent = sorted(
                    [i for i in after.inquiries if i.date and _months_between(i.date, after.today) <= 12],
                    key=lambda i: i.date or date.min,
                    reverse=True,
                )
                drop = {id(i) for i in recent[: (count or len(recent))]}
                after.inquiries = [i for i in after.inquiries if id(i) not in drop]
        elif kind == "wait_months":
            months = max(0, int(_attr(action, "months", 0) or 0))
            after.today = after.today + timedelta(days=round(months * 30.44))
        else:
            warnings.append(f"accion no soportada: {kind}")

    return after, warnings


# %% Simulacion %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

@dataclass
class BureauResult:
    bureau: str
    current_score: int
    estimated_new_score: int
    impact: int
    impact_min: int
    impact_max: int
    factors_before: dict[str, float]
    factors_after: dict[str, float]
    warnings: list[str] = field(default_factory=list)

    @property
    def factor_deltas(self) -> dict[str, float]:
        return {
            name: round(WEIGHTS[name] * (self.factors_after[name] - self.factors_before[name]), 4)
            for name in WEIGHTS
        }

    def top_drivers(self, limit: int = 3) -> list[tuple[str, float]]:
        moved = [(n, d) for n, d in self.factor_deltas.items() if abs(d) > 1e-6]
        return sorted(moved, key=lambda t: -abs(t[1]))[:limit]


def _clamp_score(value: float) -> int:
    return int(round(max(SCORE_FLOOR, min(SCORE_CEILING, value))))


def simulate_bureau(p: Profile, actions: list) -> BureauResult | None:
    """None si ese buro no trae puntaje: sin ancla no hay estimacion honesta."""
    if not p.score:
        return None

    before = factor_health(p)
    after_profile, warnings = apply_actions(p, actions)
    after = factor_health(after_profile)

    delta_health = weighted_health(after) - weighted_health(before)
    if delta_health >= 0:
        span, sensitivity = SCORE_CEILING - p.score, SENSITIVITY_UP
    else:
        span, sensitivity = p.score - SCORE_FLOOR, SENSITIVITY_DOWN
    impact = int(round(delta_health * span * sensitivity))

    new_score = _clamp_score(p.score + impact)
    impact = new_score - p.score  # el clamp manda: nunca prometer fuera de 300-850
    band = max(3, int(round(abs(impact) * 0.22)))

    return BureauResult(
        bureau=p.bureau,
        current_score=p.score,
        estimated_new_score=new_score,
        impact=impact,
        impact_min=impact - band,
        impact_max=impact + band,
        factors_before={k: round(v, 4) for k, v in before.items()},
        factors_after={k: round(v, 4) for k, v in after.items()},
        warnings=warnings,
    )


def simulate(profiles: dict[str, Profile], actions: list) -> list[BureauResult]:
    results = [simulate_bureau(p, actions) for p in profiles.values()]
    return [r for r in results if r is not None]


def feasibility_notes(profiles: dict[str, Profile], actions: list) -> list[dict]:
    """Avisos de viabilidad (bilingues) sobre lo que la simulacion da por hecho.

    El motor puede calcular cualquier escenario; que el usuario pueda ejecutarlo es
    otra cosa. Sin esto, el endpoint respondia que abrir tres tarjetas sube 35
    puntos a un perfil de 490 con doce cobranzas, que jamas seria aprobado.
    """
    notes: list[dict] = []
    reference = next((p for p in profiles.values() if p.score), None)
    if reference is None:
        return notes

    kinds = {str(getattr(_attr(a, "type", ""), "value", _attr(a, "type", ""))) for a in actions}
    derogatory = sum(1 for a in reference.accounts if a.is_collection or a.is_chargeoff)

    if "open_account" in kinds and (reference.score < 620 or derogatory):
        notes.append({
            "es": "Con este perfil (puntaje bajo o cuentas en cobranza) la aprobacion de credito nuevo es poco probable: la mejora estimada solo aplica si te aprueban.",
            "en": "With this profile (low score or accounts in collections) approval for new credit is unlikely: the estimated gain only applies if you are approved.",
        })
    if "remove_account" in kinds or "remove_inquiry" in kinds or "remove_late_payments" in kinds:
        notes.append({
            "es": "Eliminar un item del reporte depende de que el buro acepte la disputa o de un acuerdo con el acreedor; no esta garantizado.",
            "en": "Removing an item depends on the bureau accepting the dispute or on an agreement with the creditor; it is not guaranteed.",
        })

    cash = 0.0
    for action in actions:
        kind = str(getattr(_attr(action, "type", ""), "value", _attr(action, "type", "")))
        if kind != "pay_down_balance":
            continue
        amount = _money(str(_attr(action, "amount", 0) or 0))
        targets = _resolve_accounts(reference, action)
        cash += amount * max(1, len(targets)) if amount else sum(a.balance for a in targets)
    if cash > 0:
        notes.append({
            "es": f"El escenario supone pagar alrededor de ${cash:,.0f} en efectivo.",
            "en": f"The scenario assumes paying about ${cash:,.0f} in cash.",
        })
    if "wait_months" in kinds:
        notes.append({
            "es": "El paso del tiempo supone que nada mas cambia en el reporte durante ese periodo.",
            "en": "The time-passage scenario assumes nothing else in the report changes during that period.",
        })
    notes.append({
        "es": "Los buros tardan entre 30 y 60 dias en reflejar cambios, asi que el efecto no es inmediato.",
        "en": "Bureaus take 30 to 60 days to reflect changes, so the effect is not immediate.",
    })
    return notes


def risk_level(results: list[BureauResult]) -> str:
    """Segun el peor impacto individual, con los cortes documentados."""
    worst = max((abs(r.impact) for r in results), default=0)
    if worst < 10:
        return "low"
    if worst <= 30:
        return "medium"
    if worst <= 60:
        return "high"
    return "critical"


# %% Contexto para el LLM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def accounts_for_llm(accounts: list[Account], limit: int = 40) -> str:
    """Lista compacta para que el LLM ate la accion del usuario a cuentas reales."""
    ordered = sorted(accounts, key=lambda a: (not a.is_derogatory, -(a.balance or 0)))
    lines = []
    for a in ordered[:limit]:
        bits = [f"{a.ref} | {a.creditor} | {a.kind}"]
        if a.limit:
            bits.append(f"saldo {a.balance:.0f}/{a.limit:.0f} ({(a.balance / a.limit) * 100:.0f}%)")
        elif a.balance:
            bits.append(f"saldo {a.balance:.0f}")
        if a.is_collection or a.is_chargeoff:
            bits.append("collection/charge-off")
        lates = a.late30 + a.late60 + a.late90
        if lates:
            bits.append(f"{lates} atraso(s)")
        bits.append("abierta" if a.is_open else "cerrada")
        if a.opened_at:
            bits.append(f"desde {a.opened_at.isoformat()[:7]}")
        reported = a.tradeline_bureaus or a.bureaus
        bits.append("buros: " + ", ".join(sorted(reported)) if reported else "buros: ?")
        lines.append("- " + " | ".join(bits))
    return "\n".join(lines) or "Sin cuentas en el reporte."


def inquiries_for_llm(inquiries: list[Inquiry], limit: int = 15) -> str:
    """Consultas unicas para el prompt: un ref por consulta, con sus buros."""
    ordered = sorted(inquiries, key=lambda i: i.date or date.min, reverse=True)
    lines = [
        f"- {i.ref} | {i.name} | {i.date.isoformat() if i.date else 'sin fecha'} | buros: {i.bureau or '?'}"
        for i in ordered[:limit]
    ]
    return "\n".join(lines) or "Sin consultas en el reporte."
