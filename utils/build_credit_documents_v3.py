"""Construye los Documents (para el vector store) a partir de la estructura nueva
de reporte de crédito (`CreditReportV3`, buró Equifax 3B de dumbo-prod).

El objetivo es producir documentos con el MISMO estilo (texto en español) y las
mismas convenciones de metadata (`source`, `credit_repository`, `user_id`) que el
endpoint clásico /add-user-credit-data, para que el chat, el retriever y los flujos
de generación de cartas encuentren la información sin cambios."""

from datetime import datetime, timezone

from langchain_core.documents import Document

from models import (
    CreditReportV3,
    V3Account,
    V3Collection,
    V3Creditor,
    V3Inquiry,
    V3PersonalInfo,
    V3PublicRecord,
    V3Summary,
)
from utils.get_score_rating import get_score_rating
from utils.get_translation import get_translation

# Traducciones de enums propios de la estructura v3 que no están en translations.json.
_RESPONSABILITY_ES = {
    "INDIVIDUAL": "Individual",
    "JOINT_CONTRACTUAL_LIABILITY": "Responsabilidad contractual conjunta",
    "UNDESIGNATED": "Sin designar",
}
_PUBLIC_RECORD_TYPE_ES = {
    "BANKRUPTCY": "Bancarrota",
    "JUDGMENT": "Fallo judicial",
    "LIEN": "Gravamen",
}
_STATUS_ES = {
    "open": "abierta",
    "closed": "cerrada",
    "OPEN": "abierto",
    "CLOSED": "cerrado",
}
_ACCOUNT_CATEGORY_ES = {
    "creditCards": "Tarjeta de credito",
    "educationalLoans": "Prestamo estudiantil",
    "autoLoans": "Prestamo automotriz",
    "mortagageLoans": "Prestamo hipotecario",
}


def _ts_to_date(ms) -> str | None:
    """Convierte un timestamp en milisegundos a 'yyyy-mm-dd'."""
    if ms is None:
        return None
    try:
        return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    except (ValueError, OSError, OverflowError):
        return str(ms)


def _num(value) -> str:
    """Formatea un número sin '.0' cuando es entero (712.0 -> '712')."""
    try:
        f = float(value)
        return str(int(f)) if f.is_integer() else str(f)
    except (ValueError, TypeError):
        return str(value)


def _money(value) -> str | None:
    if value is None:
        return None
    try:
        return f"${float(value):,.2f}"
    except (ValueError, TypeError):
        return str(value)


def _address_to_str(address) -> str:
    if address is None:
        return ""
    parts = [address.street, address.city, address.state, address.postalCode, address.country]
    return ", ".join(p for p in parts if p)


def _creditor_to_str(creditor: V3Creditor | None) -> str:
    if creditor is None:
        return ""
    out = ""
    if creditor.name:
        out += f"Nombre del acreedor: {creditor.name}. "
    address = _address_to_str(creditor.address)
    if address:
        out += f"Direccion del acreedor: {address}. "
    if creditor.phone:
        out += f"Telefono del acreedor: {creditor.phone}. "
    return out


def _late_months(account: V3Account) -> int:
    """Cuenta los meses marcados como negativos en el historial de pagos."""
    late = 0
    for year in account.paymentHistory or []:
        for month in year.months or []:
            if month.monthType == "NEGATIVE":
                late += 1
    return late


def _format_account(account: V3Account, category_key: str) -> str:
    category = _ACCOUNT_CATEGORY_ES.get(category_key, "Cuenta")
    loan_type = get_translation(account.loanType) if account.loanType else None
    content = f"{category}"
    if loan_type and loan_type != category:
        content += f" ({loan_type})"
    content += ": "

    if account.name:
        content += f"Acreedor/Cuenta: {account.name}. "
    if account.number:
        content += f"Numero de cuenta: {account.number}. "
    if account.status:
        content += f"La cuenta esta {_STATUS_ES.get(account.status, account.status)}. "
    elif account.isOpen is not None:
        content += f"La cuenta esta {'abierta' if account.isOpen else 'cerrada'}. "
    if account.amount is not None:
        content += f"Saldo: {_money(account.amount)}. "
    if account.creditLimit is not None:
        content += f"Limite crediticio: {_money(account.creditLimit)}. "
    if account.highCredit is not None:
        content += f"Saldo alto: {_money(account.highCredit)}. "
    if account.percentage is not None:
        content += f"Porcentaje de utilizacion: {account.percentage}%. "
    if account.monthlyPayment is not None:
        content += f"Cantidad de pago mensual: {_money(account.monthlyPayment)}. "
    if account.pastDueAmount:
        content += f"Importe vencido: {_money(account.pastDueAmount)}. "

    late = _late_months(account)
    content += f"Pagos atrasados: {late}. "
    if account.percentagePaymentsOnTime is not None:
        content += f"Porcentaje de pagos a tiempo: {round(account.percentagePaymentsOnTime)}%. "
    if account.monthsReviewed is not None:
        content += f"Meses examinados: {account.monthsReviewed}. "
    if account.paymentStatus:
        content += f"Estado de pago de la cuenta: {get_translation(account.paymentStatus)}. "
    if account.responsability:
        content += f"Responsabilidad: {_RESPONSABILITY_ES.get(account.responsability, account.responsability)}. "

    opened = _ts_to_date(account.openedAt)
    if opened:
        content += f"Fecha de apertura de la cuenta: {opened}. "
    closed = _ts_to_date(account.closedAt)
    if closed:
        content += f"Fecha de cierre de la cuenta: {closed}. "
    last_activity = _ts_to_date(account.lastActivityAt)
    if last_activity:
        content += f"Fecha de ultima actividad: {last_activity}. "

    content += _creditor_to_str(account.creditor)
    return content.strip()


def _format_summary(summary: V3Summary, bureau: str) -> str:
    fields = [
        ("Total de cuentas", summary.totalAccounts),
        ("Cuentas abiertas", summary.totalOpenAccounts),
        ("Cuentas cerradas", summary.totalClosedAccounts),
        ("Total de cobranzas (collections)", summary.totalCollections),
        ("Total de registros publicos", summary.totalPublicRecords),
        ("Total de consultas (inquiries)", summary.totalInquiries),
        ("Total de tarjetas de credito", summary.totalCreditCards),
        ("Total de hipotecas", summary.totalMortage),
        ("Total de prestamos automotrices", summary.totalAuto),
        ("Total de prestamos estudiantiles", summary.totalEducational),
        ("Otras cuentas", summary.totalOtherAccounts),
    ]
    parts = [f"{label}: {value}" for label, value in fields if value is not None]
    return "Resumen del reporte de credito. " + "; ".join(parts)


def _format_personal_info_docs(user_id: str, info: V3PersonalInfo) -> list[Document]:
    docs: list[Document] = []

    def add(field: str, content: str):
        docs.append(
            Document(
                page_content=content,
                metadata={"field": field, "source": "Personal Info", "user_id": user_id},
                id=field,
            )
        )

    if info.firstName:
        add("FirstName", f"Mi primer nombre es: {info.firstName}")
    if info.middleName:
        add("MiddleName", f"Mi segundo nombre es: {info.middleName}")
    if info.lastName:
        add("LastName", f"Mi apellido es: {info.lastName}")
    if info.nationalIdentifier:
        add("SSN", f"Mi SSN es: {info.nationalIdentifier}")
    if info.dateOfBirth:
        dob = _ts_to_date(info.dateOfBirth) if isinstance(info.dateOfBirth, int) else info.dateOfBirth
        add("BirthDate", f"Mi Fecha de nacimiento es: {dob}")
    address = _address_to_str(info.currentAddress)
    if address:
        add("Address", f"Mis direcciones son: Residiendo Actualmente en: {address}")
    if info.homePhone:
        add("HomePhone", f"Mi telefono de casa es: {info.homePhone}")
    if info.mobilePhone:
        add("MobilePhone", f"Mi telefono movil es: {info.mobilePhone}")
    return docs


def _bureaus_present(report: CreditReportV3) -> list[str]:
    """Buros que aparecen en cualquier sección del reporte, priorizando Equifax."""
    keys: list[str] = []
    for section in (
        report.accounts,
        report.inquiries,
        report.summary,
        report.personalInfo,
        report.creditors,
        report.collections,
        report.publicRecords,
        report.scores,
        report.scoreHistory,
    ):
        for k in section.keys():
            if k not in keys:
                keys.append(k)
    order = {"Equifax": 0, "Experian": 1, "TransUnion": 2}
    keys.sort(key=lambda k: order.get(k, 99))
    return keys


def build_credit_documents_v3(user_id: str, report: CreditReportV3) -> list[Document]:
    documents: list[Document] = []
    bureaus = _bureaus_present(report)

    # --- Personal info: se toma un buró representativo (Equifax de preferencia).
    for bureau in bureaus:
        info = report.personalInfo.get(bureau)
        if info is not None:
            personal_docs = _format_personal_info_docs(user_id, info)
            if personal_docs:
                documents.extend(personal_docs)
                break

    # --- Resumen por buró.
    for bureau, summary in report.summary.items():
        if summary is None:
            continue
        documents.append(
            Document(
                page_content=f"{_format_summary(summary, bureau)} en el Buro de Credito: {bureau}",
                metadata={"source": "CreditSummary", "user_id": user_id, "credit_repository": bureau},
                id=f"summary_{bureau}",
            )
        )

    # --- Puntajes e historial de puntaje por buró.
    for bureau in bureaus:
        score = report.scores.get(bureau)
        if score is not None:
            documents.append(
                Document(
                    page_content=(
                        f"Puntaje de Credito: {_num(score)} en el Buro de Credito: {bureau}, "
                        f"clasificacion: {get_score_rating(score)}"
                    ),
                    metadata={"source": "CreditScore", "field": "score", "user_id": user_id, "credit_repository": bureau},
                    id=f"score_{bureau}",
                )
            )
        for point in report.scoreHistory.get(bureau, []) or []:
            if point.value is None:
                continue
            date = _ts_to_date(point.reportedDate)
            documents.append(
                Document(
                    page_content=(
                        f"Puntaje de Credito: Valor en la fecha {date}: {_num(point.value)} "
                        f"en el Buro de Credito: {bureau}, clasificacion: {get_score_rating(point.value)}"
                    ),
                    metadata={
                        "source": "CreditScore",
                        "field": "history",
                        "date": date,
                        "user_id": user_id,
                        "credit_repository": bureau,
                    },
                    id=f"score_hist_{bureau}_{date}",
                )
            )

    # --- Consultas (inquiries) por buró.
    for bureau, inquiries in report.inquiries.items():
        for inquiry in inquiries or []:
            date = _ts_to_date(inquiry.reportedDate)
            name = (inquiry.creditor.name if inquiry.creditor else None) or "Desconocido"
            documents.append(
                Document(
                    page_content=(
                        f"Consulta: {name}; Tipo de Consulta: {inquiry.type or 'HARD'}; "
                        f"Buro de Credito: {bureau}; Fecha: {date}"
                    ),
                    metadata={
                        "source": "CreditInquiry",
                        "credit_repository": bureau,
                        "date": date,
                        "user_id": user_id,
                    },
                    id=f"inquiry_{bureau}_{name}_{date}",
                )
            )

    # --- Cuentas (liabilities) por buró y por tipo.
    for bureau, by_type in report.accounts.items():
        for category_key in ("creditCards", "educationalLoans", "autoLoans", "mortagageLoans"):
            for account in getattr(by_type, category_key) or []:
                content = f"{_format_account(account, category_key)} En el Buro de Credito: {bureau}."
                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": "CreditLiability",
                            "field": "liability",
                            "user_id": user_id,
                            "credit_repository": bureau,
                        },
                        id=f"account_{bureau}_{category_key}_{account.number or account.name or len(documents)}",
                    )
                )

    # --- Cobranzas (collections) por buró.
    for bureau, collections in report.collections.items():
        for collection in collections or []:
            documents.append(
                Document(
                    page_content=_format_collection(collection, bureau),
                    metadata={
                        "source": "Collection",
                        "field": "collection",
                        "user_id": user_id,
                        "credit_repository": bureau,
                    },
                    id=f"collection_{bureau}_{collection.accountNumber or len(documents)}",
                )
            )

    # --- Registros públicos por buró.
    for bureau, records in report.publicRecords.items():
        for record in records or []:
            documents.append(
                Document(
                    page_content=_format_public_record(record, bureau),
                    metadata={
                        "source": "PublicRecord",
                        "field": "public_record",
                        "user_id": user_id,
                        "credit_repository": bureau,
                    },
                    id=f"public_record_{bureau}_{record.refNumber or len(documents)}",
                )
            )

    return documents


def _format_collection(collection: V3Collection, bureau: str) -> str:
    content = "Cobranza (collection): "
    if collection.accountNumber:
        content += f"Numero de cuenta: {collection.accountNumber}. "
    if collection.status:
        content += f"Estado: {_STATUS_ES.get(collection.status, collection.status)}. "
    if collection.amount is not None:
        content += f"Monto: {_money(collection.amount)}. "
    reported = _ts_to_date(collection.reportedDate)
    if reported:
        content += f"Fecha reportada: {reported}. "
    if collection.agencyClient:
        content += f"Agencia de cobro: {collection.agencyClient.name or ''}. "
        content += _creditor_to_str(collection.agencyClient)
    if collection.originalCreditor and collection.originalCreditor.name:
        content += f"Acreedor original: {collection.originalCreditor.name}. "
    content += f"En el Buro de Credito: {bureau}."
    return content.strip()


def _format_public_record(record: V3PublicRecord, bureau: str) -> str:
    record_type = _PUBLIC_RECORD_TYPE_ES.get(record.type, record.type or "Registro publico")
    content = f"Registro publico ({record_type}): "
    if record.status:
        content += f"Estado: {get_translation(record.status)}. "
    if record.refNumber:
        content += f"Numero de referencia: {record.refNumber}. "
    if record.courtName:
        content += f"Corte: {record.courtName}. "
    if record.amount is not None:
        content += f"Monto: {_money(record.amount)}. "
    if record.assetAmount is not None:
        content += f"Monto de activos: {_money(record.assetAmount)}. "
    filed = _ts_to_date(record.filedDate)
    if filed:
        content += f"Fecha de presentacion: {filed}. "
    reported = _ts_to_date(record.reportedDate)
    if reported:
        content += f"Fecha reportada: {reported}. "
    content += f"En el Buro de Credito: {bureau}."
    return content.strip()
