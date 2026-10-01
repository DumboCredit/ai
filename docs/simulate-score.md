# Endpoints: Simulador de puntaje

`POST /simulate-score` · `POST /get-simulation-options`

Simula el impacto en el puntaje de crédito del usuario ante una acción, descrita en lenguaje natural o ya estructurada. Devuelve el impacto estimado en puntos **por buró** (TransUnion, Equifax, Experian) con su rango, el nivel de riesgo, qué factores FICO se movieron y una explicación bilingüe basada en el reporte real del usuario.

`/get-simulation-options` devuelve lo simulable del reporte (cuentas, cobranzas y consultas con su `ref`), para que el front arme toggles y sliders y mande acciones estructuradas.

> **Requisito:** el usuario debe tener su reporte cargado y con puntajes por buró. Sin puntaje no hay ancla y `/simulate-score` responde `409`.

> **Tiempo de respuesta medido:**
>
> | Llamada | LLM | Latencia | Tokens |
> |---|:---:|---|---|
> | `/get-simulation-options` | no | 0.25 s | 0 |
> | `/simulate-score` con `action` | 2 pasos | 8-9 s | ~3.000 |
> | `/simulate-score` con `actions` | solo narrador | 6 s | ~1.100 |
> | `/simulate-score` con `actions` + `explain: false` | no | 0.12 s | 0 |
>
> Cómo integrarlo en la app: [simulate-score-integration.md](./simulate-score-integration.md).

---

## Cómo se calcula

Los puntos **no los estima un LLM**. El flujo tiene tres etapas separadas a propósito:

1. **Interpretación (LLM).** Traduce el texto del usuario a acciones de un catálogo cerrado, atadas a cuentas reales del reporte por su `ref`. No estima puntos. Se salta si el request trae `actions`.
2. **Motor determinista** ([`utils/score_engine.py`](../utils/score_engine.py)). Aplica esas acciones sobre una copia del perfil de cada buró y calcula el delta comparando la salud ponderada de los cinco factores FICO (historial de pagos 35%, utilización 30%, antigüedad 15%, mezcla 10%, crédito nuevo 10%) antes y después. Las mismas acciones dan siempre el mismo número.
3. **Narrativa (LLM).** Redacta la explicación en español e inglés sobre números que ya están fijos. No puede cambiarlos, y se puede apagar con `explain: false`.

El impacto depende del perfil de partida, igual que en el modelo real: un primer atraso en un perfil limpio cuesta decenas de puntos, y el mismo atraso en un perfil que ya tiene cobranzas cuesta poco. Eso no está cableado, sale de que cada factor arranca de su propia salud.

**No es FICO ni VantageScore.** Correr esos modelos exige licencia del buró. Esto es una estimación propia calibrada contra los rangos públicos (un primer atraso de 30 días: 60-110 pts según el puntaje de partida; un hard inquiry: menos de 5 pts). La respuesta lo dice en `estimate_basis` y hay que mostrarlo en el front.

---

## `POST /get-simulation-options`

Determinista, sin LLM. Devuelve los `ref` que aceptan las acciones de `/simulate-score`.

```json
{ "API_KEY": "tu_api_key", "user_id": "arteaga" }
```

Respuesta real (recortada) de un reporte guardado:

```json
{
  "scores": { "Equifax": 607, "Experian": 604, "TransUnion": 607 },
  "accounts": [
    {
      "ref": "A1", "creditor": "BANK OF AMERICA", "kind": "revolving",
      "is_open": true, "balance": 3615.0, "limit": 7400.0, "utilization": 0.4885,
      "is_collection": false, "is_chargeoff": false, "late_payments": 0,
      "opened_at": "2022-07-23", "bureaus": ["Equifax", "Experian", "TransUnion"]
    },
    {
      "ref": "A3", "creditor": "JPMCB CARD SERVICES", "kind": "revolving",
      "is_open": true, "balance": 11876.0, "limit": 12000.0, "utilization": 0.9897,
      "is_collection": false, "is_chargeoff": false, "late_payments": 0,
      "opened_at": "2025-03-21", "bureaus": ["Equifax", "Experian", "TransUnion"]
    }
  ],
  "inquiries": [
    { "ref": "I2", "name": "CAPITAL ONE", "date": "2024-10-21", "bureaus": ["Equifax", "Experian", "TransUnion"] }
  ]
}
```

| Campo | Descripción |
|---|---|
| `scores` | Puntaje actual por buró. Vacío si el reporte no los trae (en ese caso `/simulate-score` da `409`). |
| `accounts[].ref` | Identificador para `account_ref`. Uno por tradeline, no por documento. |
| `accounts[].kind` | `revolving`, `installment`, `mortgage` u `other`. Las cobranzas v3 vienen como `other`. |
| `accounts[].utilization` | Saldo / límite, solo en revolventes con límite. `null` en el resto. |
| `accounts[].is_collection` / `is_chargeoff` | Ítems que afecta `remove_account` con `apply_to_all`. |
| `accounts[].bureaus` | Burós que reportan ese tradeline. Explica por qué los impactos difieren por buró. |
| `inquiries[].ref` | Identificador para `inquiry_ref`. Uno por consulta, compartido entre burós. |

Los `ref` son estables mientras no se recargue el reporte; después de un `/add-user-credit-data-v3` hay que volver a pedirlos.

---

## `POST /simulate-score`

### Request

#### Modo texto libre

```json
{
  "API_KEY": "tu_api_key",
  "user_id": "usuario_123",
  "action": "dejar que mi cuenta de Capital One se venza 2 meses sin pagar"
}
```

#### Modo estructurado (sin LLM interpretador)

Para cuando el front ya sabe qué simular: togglear un ítem negativo, un slider de pago, un botón de "¿y si elimino esta cobranza?". Los `ref` salen de `/get-simulation-options`.

```json
{
  "API_KEY": "tu_api_key",
  "user_id": "usuario_123",
  "explain": false,
  "actions": [
    { "type": "remove_account", "account_ref": "A9", "note": "toggle cobranza LVNV" }
  ]
}
```

| Campo | Tipo | Requerido | Descripción |
|---|---|:---:|---|
| `API_KEY` | string | sí | API key del backend. Si no coincide → `400`. |
| `user_id` | string | sí | Usuario con crédito previamente cargado. |
| `action` | string | sí\* | Acción en lenguaje natural. Puede ser positiva o negativa. |
| `actions` | array | sí\* | Acciones ya estructuradas. Si viene, no se llama al LLM interpretador. |
| `explain` | bool | no | Default `true`. Con `false` se salta el LLM narrador: `explanation` viene vacía, la respuesta es 100% determinista (0.12 s, 0 tokens) y sirve para recalcular en cada movimiento de un slider. |

\* Se requiere al menos uno de los dos; si faltan ambos → `422`.

### Catálogo de acciones

| `type` | Qué simula | Campos que usa |
|---|---|---|
| `pay_down_balance` | Pagar o abonar a una cuenta | `account_ref`, `amount` (vacío = saldo completo) |
| `increase_balance` | Gastar más / subir el saldo | `account_ref`, `amount` |
| `max_out_cards` | Llevar tarjetas al límite | `account_ref` o `apply_to_all` |
| `change_credit_limit` | Subir o bajar el límite | `account_ref`, `new_limit` o `amount` |
| `remove_account` | Eliminar cuenta, cobranza o charge-off (disputa/acuerdo) | `account_ref`, o `apply_to_all` (solo afecta cobranzas y charge-offs) |
| `remove_inquiry` | Eliminar consultas duras | `inquiry_ref` o `count` |
| `remove_late_payments` | Borrar atrasos de una cuenta (goodwill) | `account_ref` |
| `add_late_payment` | Dejar de pagar | `account_ref`, `days_late` (30/60/90) |
| `open_account` | Abrir cuenta nueva (suma también el inquiry) | `account_kind`, `new_limit` (si falta: 500 en perfil dañado, proporcional en perfil sano), `amount` |
| `close_account` | Cerrar una cuenta abierta | `account_ref` |
| `wait_months` | Dejar pasar el tiempo sin cambios | `months` |

En lugar de `account_ref` se puede mandar `creditor` (nombre del acreedor, coincidencia parcial), útil para enlazar con los resultados de [`/get-litigation-errors`](./get-litigation-errors.md).

Los `ref` (`A1`, `I2`…) identifican **tradelines y consultas, no documentos**: el mismo ítem llega al sistema varias veces (en v3, una vez bajo cada buró; en legacy, hasta cuatro veces entre el documento consolidado y los de cada buró, con el acreedor escrito distinto) y se agrupa antes de simular. Si el payload v3 trae `id` en cuentas, consultas y cobranzas, la agrupación es exacta; si no, se deduce (ver notas de implementación). Una acción sobre un ítem que un buró no reporta no se aplica a ese buró, y queda anotado en `impacts[].notes`.

### Response `200`

Respuesta real sobre un reporte guardado (recortada a un buró y dos factores):

```json
{
  "action": "pagar el 50% del saldo de la tarjeta donde tengo mas balance",
  "interpreted_as": "Pagar $5,938 de la tarjeta A3, que tiene el mayor saldo.",
  "actions": [
    {
      "type": "pay_down_balance",
      "account_ref": "A3",
      "creditor": "JPMCB CARD SERVICES",
      "inquiry_ref": null,
      "apply_to_all": false,
      "account_kind": "revolving",
      "amount": 5938.0,
      "new_limit": null,
      "days_late": null,
      "count": null,
      "months": null,
      "note": "Pagar el 50% del saldo de la tarjeta con mayor balance, A3."
    }
  ],
  "impacts": [
    {
      "bureau": "Equifax",
      "current_score": 607,
      "estimated_new_score": 619,
      "impact": 12,
      "impact_min": 9,
      "impact_max": 15,
      "factors": [
        { "factor": "payment_history", "weight": 0.35, "health_before": 1.0, "health_after": 1.0, "contribution": 0.0 },
        { "factor": "utilization", "weight": 0.3, "health_before": 0.4086, "health_after": 0.591, "contribution": 0.0547 }
      ],
      "notes": []
    }
  ],
  "explanation": "Al pagar $5,938 de la tarjeta A3, que tiene el mayor saldo, se reduce la utilización de crédito, el factor FICO dominante, con un movimiento de +0.055. Por eso, Equifax pasa de 607 a 619, Experian de 604 a 616 y TransUnion de 607 a 619. El escenario supone pagar esos $5,938 en efectivo y los cambios podrían tardar entre 30 y 60 días en reflejarse en los burós.",
  "explanation_en": "By paying $5,938 on card A3, which has the highest balance, credit utilization is reduced, the dominant FICO factor, with a movement of +0.055. As a result, Equifax goes from 607 to 619, Experian from 604 to 616, and TransUnion from 607 to 619. The scenario assumes paying those $5,938 in cash, and the changes could take between 30 and 60 days to appear on the bureaus.",
  "risk_level": "medium",
  "caveats": [
    "El escenario supone pagar alrededor de $5,938 en efectivo.",
    "Los buros tardan entre 30 y 60 dias en reflejar cambios, asi que el efecto no es inmediato."
  ],
  "caveats_en": [
    "The scenario assumes paying about $5,938 in cash.",
    "Bureaus take 30 to 60 days to reflect changes, so the effect is not immediate."
  ],
  "estimate_basis": "Estimación propia de Dumbo basada en los factores FICO y en tu reporte real. No es tu puntaje FICO ni VantageScore oficial: es orientación, no un pronóstico exacto.",
  "estimate_basis_en": "Dumbo's own estimate based on FICO factors and your actual report. It is not your official FICO or VantageScore: it is guidance, not an exact forecast."
}
```

#### `BureauScoreImpact`

| Campo | Tipo | Descripción |
|---|---|---|
| `bureau` | string | `"TransUnion"`, `"Equifax"` o `"Experian"`. |
| `current_score` | int | Puntaje actual en ese buró, tomado del reporte (nunca lo inventa el LLM). |
| `estimated_new_score` | int | Puntaje estimado tras la acción, acotado a 300-850. |
| `impact` | int | Puntos ganados (positivo) o perdidos (negativo). |
| `impact_min` / `impact_max` | int | Extremos pesimista y optimista de la estimación. Es lo que conviene mostrar al usuario, no el punto exacto. |
| `factors` | array | Salud antes/después de cada factor FICO y su contribución al cambio. Sirve para explicar en el front qué movió el puntaje. |
| `notes` | array | Avisos de ese buró, p.ej. que no reporta la cuenta simulada. |

#### `caveats`

Los genera el motor según el escenario, no el LLM: que la aprobación de crédito nuevo es poco probable con puntaje bajo o cobranzas, cuánto efectivo supone el escenario, que una eliminación depende de que el buró acepte la disputa, que el paso del tiempo supone que nada más cambia, y que los burós tardan 30-60 días en reflejar cambios.

#### `risk_level`

| Valor | Rango de impacto |
|---|---|
| `"low"` | Menos de 10 puntos |
| `"medium"` | Entre 10 y 30 puntos |
| `"high"` | Entre 30 y 60 puntos |
| `"critical"` | Más de 60 puntos |

> Se calcula sobre el **peor impacto individual** entre todos los burós, en valor absoluto.

---

## Errores

| Código | Causa |
|---|---|
| `400` | `API_KEY` no coincide. |
| `409` | El reporte del usuario no trae puntajes por buró: sin ancla no se puede estimar. |
| `422` | No se mandó `action` ni `actions`; o la petición no se puede representar con el catálogo (p.ej. "que pasa si me gano la lotería"); o es ambigua (dos tarjetas del mismo acreedor). El `detail` explica qué falta. |
| `500` | El usuario no tiene colección de crédito creada, u otro error interno. |

En lugar de inventar un número, el endpoint prefiere responder `422` explicando por qué no puede simular lo que se le pidió.

---

## Ejemplos de acciones válidas

| Tipo | Ejemplo |
|---|---|
| Negativa | `"dejar que mi cuenta de Capital One se venza 2 meses"` |
| Negativa | `"cerrar mi tarjeta de crédito más antigua"` |
| Negativa | `"abrir 3 tarjetas de crédito nuevas este mes"` |
| Positiva | `"pagar el 50% del saldo de mi tarjeta con mayor balance"` |
| Positiva | `"disputar y eliminar todas mis cuentas en cobranza"` |
| Positiva | `"eliminar el inquiry de Citibank"` |
| Tiempo | `"esperar 12 meses sin hacer nada"` |

### curl

```bash
curl -X POST http://localhost:8080/get-simulation-options \
  -H "Content-Type: application/json" \
  -d '{"API_KEY":"tu_api_key","user_id":"abc123"}'
```

```bash
curl -X POST http://localhost:8080/simulate-score \
  -H "Content-Type: application/json" \
  -d '{"API_KEY":"tu_api_key","user_id":"abc123","action":"pagar el 50% del saldo de mi tarjeta con mayor balance"}'
```

---

## Notas de implementación

- Una sola consulta a Chroma trae `CreditLiability`, `Collection`, `CreditInquiry` y `CreditScore`; el motor parsea los dos formatos de documento (legacy y v3).
- **Cobranzas.** En legacy llegan como cuenta con estado `Collection/Charge-off`; en v3 vienen en su propia sección (`collections`, guardada con `source: "Collection"`) y el motor las convierte en tradelines derogatorios con el monto como saldo y la fecha reportada como recencia. Aparecen en `/get-simulation-options` con `is_collection: true` y se pueden eliminar con `remove_account`.
- La morosidad se lee del **estado de la cuenta** (`Late30Days`, `Collection/Charge-off`), no solo del contador `Pagos atrasados`: en los reportes reales ese contador llega en 0 y de otro modo el motor no vería ninguna cuenta mala.
- El puntaje vigente de v3 (documento sin fecha) gana sobre el historial; en formato legacy se toma el histórico más reciente.
- **Agrupación entre burós.** Cuentas, consultas y cobranzas v3 aceptan un `id` opcional, igual en los tres burós para el mismo ítem. Se guarda en la metadata (`tradeline_id` para cuentas y cobranzas, `inquiry_id` para consultas) y, si está, es la clave: agrupación exacta. Sin `id` se deduce — cuentas por fecha de apertura + primeros dígitos del número con respaldo en saldo + límite, consultas por nombre + fecha, cobranzas por agencia + monto. La deducción falla con dos tarjetas distintas que coinciden en apertura, saldo y límite (se fusionan) y con consultas que cada buró nombra distinto (quedan separadas). Ver [simulate-score-integration.md](./simulate-score-integration.md#manda-el-id-de-cada-cuenta-consulta-y-cobranza).
- Los headers `X-Usage-*` reportan el consumo de tokens de los pasos de LLM, como en el resto de endpoints. Con `explain: false` y `actions` vienen en 0.

---

## Limitaciones conocidas

- **Registros públicos v3 no se leen.** Bancarrotas y gravámenes (`publicRecords`) no entran al motor todavía, aunque son de lo que más pesa en el historial de pagos.
- **Reportes cargados por PDF.** Pueden venir sin puntajes (→ `409`), sin fechas de apertura (la antigüedad y "mi cuenta más antigua" no funcionan) o sin cuentas.
- **Impacto 0 en factores saturados.** Si un factor ya está en su peor nivel (utilización al 100%, muchas cobranzas), una acción negativa más da 0. Es correcto, y la `explanation` lo dice; el front no debe presentarlo como que la acción es inofensiva.
- **La interpretación no es determinista.** El LLM corre con `temperature=1` (los modelos gpt-5.6 solo aceptan eso), así que la misma frase puede traducirse a acciones distintas entre llamadas. Dadas las acciones, el resultado sí es determinista; con `actions` es 100% reproducible.
- **Sin caché.** Ni `/simulate-score` ni `/get-simulation-options` cachean. Si se agrega, no sirve el `_cache_put` del resto de endpoints, que guarda una sola entrada por usuario: un usuario prueba muchos escenarios por sesión.
- **Nombres de buró sin validar.** Si el reporte trae un buró fuera de los tres conocidos, aparece tal cual en la respuesta.
