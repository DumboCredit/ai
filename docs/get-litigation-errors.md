# Endpoint: Detección de errores litigables

`POST /get-litigation-errors`

Analiza el reporte de crédito de un usuario (Equifax, Experian, TransUnion) y detecta **errores litigables** (posibles violaciones de la FCRA). Devuelve la lista de errores encontrados.

> **Requisito previo:** el usuario debe tener su crédito ya cargado mediante `/add-user-credit-data` o `/add-user-credit-data-by-pdf`. Si no, no hay datos que analizar.

---

## Request

```
POST /get-litigation-errors
Content-Type: application/json
```

```json
{
  "API_KEY": "tu_api_key",
  "user_id": "id_del_usuario",
  "reasoning_effort": "medium"
}
```

| Campo | Tipo | Requerido | Descripción |
|---|---|:---:|---|
| `API_KEY` | string | sí | API key del backend. Si no coincide → `400`. |
| `user_id` | string | sí | Usuario con crédito previamente cargado. |
| `reasoning_effort` | `"none"` \| `"low"` \| `"medium"` \| `"high"` | no | Profundidad del análisis. Default: `none`. Recomendado: `medium`. |

---

## Response `200`

Devuelve un **array** de errores. Si no encuentra ninguno, devuelve `[]`.

```json
[
  {
    "error_type": "Doble reporte de la misma cuenta",
    "reason": "La deuda de CAPITAL ONE se reporta activa por el acreedor original y por MIDLAND a la vez.",
    "evidence": "CAPITAL ONE saldo $1,200 (abierta) / MIDLAND FUNDING saldo $1,200 (colección abierta)",
    "name_account": "MIDLAND FUNDING",
    "account_number": "5178XXXX",
    "creditor": "MIDLAND FUNDING LLC",
    "credit_repo": ["Equifax", "TransUnion"]
  }
]
```

| Campo | Tipo | Descripción |
|---|---|---|
| `error_type` | enum | Tipo de error litigable (uno de los 8 valores de abajo). |
| `reason` | string | Por qué es un error litigable. |
| `evidence` | string | Dato textual del reporte que lo sustenta. |
| `name_account` | string \| null | Nombre de la cuenta/acreedor, si aplica. |
| `account_number` | string \| null | Número de cuenta, si aplica. |
| `creditor` | string \| null | Acreedor exacto como aparece en el reporte, si aplica. |
| `credit_repo` | string \| string[] | Buró(s) donde aparece el error. |

### Valores posibles de `error_type`

1. `Reporte posterior a bancarrota sin divulgacion de bancarrota`
2. `Doble reporte de la misma cuenta`
3. `Archivo mezclado`
4. `Multiples SSN con cuentas que no son del consumidor`
5. `Comprador de deuda reporta informacion distinta al acreedor original`
6. `Reporte posterior a la fecha de obsolescencia`
7. `Reporte posterior a disputa sin notacion de disputa`
8. `Consumidor reportado como fallecido estando vivo`

---

## Errores

| Código | Causa |
|---|---|
| `400` | `API_KEY` no coincide. |
| `500` | El usuario no tiene colección de crédito creada, u otro error interno. |

---

## Ejemplos

### curl

```bash
curl -X POST http://localhost:8080/get-litigation-errors \
  -H "Content-Type: application/json" \
  -d '{"API_KEY":"tu_api_key","user_id":"abc123","reasoning_effort":"medium"}'
```

### fetch (JS)

```js
const res = await fetch(`${BASE_URL}/get-litigation-errors`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ API_KEY, user_id, reasoning_effort: "medium" }),
});
const errors = await res.json(); // array; [] si no hay errores litigables
```

---

## Notas

- El endpoint solo **detecta** los errores. Qué se hace con ellos (consentimiento del cliente, asignación de abogado, popup) es lógica de back/front, no de este servicio.
- **Límite conocido:** los tipos *#6 (post-obsolescencia)* y *#8 (fallecido estando vivo)* dependen de campos que el extractor de PDF actual no captura explícitamente (fecha de primera morosidad, marca "deceased"); se infieren de las fechas disponibles en el reporte.
