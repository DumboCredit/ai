# Integración del simulador de puntaje

Guía para integrar el simulador desde la app (front o backend de dumbo-prod). El contrato campo por campo está en [simulate-score.md](./simulate-score.md); acá está **cómo se conecta**, en qué orden, qué mostrar y qué no.

---

## 1. Qué es y qué no es

El simulador estima cuántos puntos movería una acción sobre el reporte real del usuario, **por buró**.

- **No es FICO ni VantageScore.** Correr esos modelos exige licencia del buró. Esto es un modelo propio calibrado contra los rangos públicos de FICO.
- **No es un pronóstico.** Es orientación. La respuesta trae un rango (`impact_min`/`impact_max`) y un texto de encuadre (`estimate_basis`) que **hay que mostrar** en la UI.
- **No decide nada.** No aprueba, no recomienda productos financieros ni promete resultados de disputa.

---

## 2. Las piezas

```
                       ┌──────────────────────────────────────────┐
  POST /simulate-score │ 1. LLM interpretador                     │
  { action: "texto" }  │    texto libre → acciones del catálogo   │
        ─────────────► │    (se salta si mandas `actions`)        │
                       ├──────────────────────────────────────────┤
                       │ 2. Motor determinista                    │
                       │    utils/score_engine.py                 │
                       │    reporte en Chroma → perfil por buró   │
                       │    aplica acciones → delta de 5 factores │
                       │    ESTE PASO DA LOS NÚMEROS              │
                       ├──────────────────────────────────────────┤
                       │ 3. LLM narrador                          │
                       │    explicación es/en sobre números fijos │
                       └──────────────────────────────────────────┘
                                        │
                                        ▼
                         impacts[] por buró + explicación + caveats
```

Lo importante para quien integra: **los puntos los calcula el paso 2, no un LLM.** El mismo input da siempre el mismo número, así que el resultado se puede cachear, comparar entre corridas y mostrar como dato estable. El LLM solo entiende la frase del usuario y luego la redacta.

---

## 3. Precondición

El usuario tiene que tener su reporte cargado y **con puntajes por buró**:

```
POST /add-user-credit-data-v3   (o /add-user-credit-data en formato legacy)
```

Sin puntajes, `/simulate-score` responde `409`: no hay ancla desde donde estimar. No se inventa un puntaje de partida.

### Manda el `id` de cada cuenta, consulta y cobranza

En v3 todo viene anidado por buró (`accounts`, `inquiries` y `collections` son `{ Equifax: ..., Experian: ..., TransUnion: ... }`), así que el mismo ítem llega tres veces y **nada lo liga** salvo su identificador. El campo se llama `id` en las tres secciones, es **opcional** y debe tener el mismo valor en los tres burós para el mismo ítem:

```json
{
  "accounts": {
    "Equifax":    { "creditCards": [{ "id": "TL-777", "name": "BANK OF AMERICA", "amount": 3615, "creditLimit": 7400, ... }] },
    "Experian":   { "creditCards": [{ "id": "TL-777", "name": "BK OF AMER",      "amount": 3615, "creditLimit": 7400, ... }] },
    "TransUnion": { "creditCards": [{ "id": "TL-777", "name": "BOFA",            "amount": 3640, "creditLimit": 7400, ... }] }
  },
  "inquiries": {
    "Equifax":    [{ "id": "INQ-1", "creditor": { "name": "WELLS FARGO-PL&L" }, ... }],
    "Experian":   [{ "id": "INQ-1", "creditor": { "name": "WELLSFARGO" }, ... }],
    "TransUnion": [{ "id": "INQ-1", "creditor": { "name": "WELLS FARGO BANK" }, ... }]
  },
  "collections": {
    "Equifax":    [{ "id": "COL-1", "agencyClient": { "name": "LVNV FUNDING LLC" }, "amount": 673, ... }],
    "Experian":   [{ "id": "COL-1", "agencyClient": { "name": "LVNV FUNDING" },     "amount": 673, ... }]
  }
}
```

**Si viene**, la agrupación es exacta: cada ítem es un solo `ref` con su lista de `bureaus`, cada buró conserva su propio saldo (los 3.640 de TransUnion no contaminan a los otros) y una acción sobre ese `ref` se aplica a todos los burós que lo reportan.

**Si no viene** (o viene en unos ítems y en otros no), todo sigue funcionando: el simulador deduce la agrupación. Pero la deducción falla en dos casos reales, medidos con un reporte de prueba:

| caso | con `id` | sin `id` |
|---|---|---|
| Dos tarjetas distintas abiertas el mismo día, mismo saldo, mismo límite | 2 cuentas | **se fusionan en 1**: el reporte simulado pierde una tarjeta, y "pagar toda la tarjeta" da **+22** en vez de **+14**, porque la utilización baja de 49% a 0% en lugar de a 24% |
| La misma consulta con el acreedor escrito distinto en cada buró | 1 consulta en 3 burós | **3 consultas**, una por buró: eliminarla solo la quita de uno |

Las cobranzas se agrupan bien aun sin `id` (por agencia + monto), pero conviene mandarlo igual.

---

## 4. Los dos endpoints

| Endpoint | LLM | Latencia medida | Tokens | Para qué |
|---|:---:|---|---|---|
| `POST /get-simulation-options` | no | 0.25 s | 0 | Traer todo lo simulable del reporte, con los `ref` que aceptan las acciones |
| `POST /simulate-score` con `action` | 2 llamadas | 8-9 s | ~3.200 | Texto libre |
| `POST /simulate-score` con `actions` | 1 llamada (narrador) | 6 s | ~1.100 | Toggles con explicación |
| `POST /simulate-score` con `actions` + `explain: false` | ninguna | **0.12 s** | 0 | Sliders y toggles en vivo |

Medido sobre los reportes de prueba (13-30 tradelines). El `explain: false` es el que hace viable recalcular en cada interacción de UI: cero LLM, cero tokens, mismos números.

### `/get-simulation-options`

```json
{ "API_KEY": "...", "user_id": "randy" }
```

```json
{
  "scores": { "Equifax": 680, "Experian": 684, "TransUnion": 678 },
  "accounts": [
    {
      "ref": "A9", "creditor": "LVNV FUNDING LLC", "kind": "other",
      "is_open": true, "balance": 673.0, "limit": 0.0, "utilization": null,
      "is_collection": true, "is_chargeoff": false, "late_payments": 0,
      "opened_at": "2024-03-11",
      "bureaus": ["Equifax", "Experian", "TransUnion"]
    },
    {
      "ref": "A12", "creditor": "CAPITAL ONE BANK USA", "kind": "revolving",
      "is_open": true, "balance": 126.0, "limit": 7000.0, "utilization": 0.018,
      "is_collection": false, "is_chargeoff": false, "late_payments": 0,
      "opened_at": "2019-08-02",
      "bureaus": ["Equifax", "Experian", "TransUnion"]
    }
  ],
  "inquiries": [
    { "ref": "I2", "name": "WELLSFARGO", "date": "2024-02-05", "bureaus": ["Experian", "TransUnion"] }
  ]
}
```

Los `ref` (`A9`, `I2`) identifican **tradelines y consultas, no documentos**: la misma cuenta llega al backend hasta cuatro veces (un documento consolidado y uno por buró, con el acreedor escrito distinto) y se agrupa antes de exponerla. Por eso `bureaus` es la lista de burós que reportan *ese* ítem, y es la clave de que los impactos salgan distintos por buró.

Los `ref` son estables mientras no se recargue el reporte. **Después de un `/add-user-credit-data-v3` hay que volver a pedir las opciones**, porque los refs se recalculan.

---

## 5. Tipos (TypeScript)

```ts
type Bureau = "Equifax" | "Experian" | "TransUnion";

type SimulateActionType =
  | "pay_down_balance" | "increase_balance" | "max_out_cards"
  | "change_credit_limit" | "remove_account" | "remove_inquiry"
  | "remove_late_payments" | "add_late_payment" | "open_account"
  | "close_account" | "wait_months";

interface SimulatedAction {
  type: SimulateActionType;
  account_ref?: string;      // "A9" — de /get-simulation-options
  creditor?: string;         // alternativa al ref: nombre del acreedor
  inquiry_ref?: string;      // "I2"
  apply_to_all?: boolean;    // todas las del tipo indicado
  account_kind?: "revolving" | "installment" | "mortgage";
  amount?: number;           // dólares; vacío en pay_down_balance = pagar todo
  new_limit?: number;
  days_late?: 30 | 60 | 90;
  count?: number;
  months?: number;
  note: string;              // requerido: qué representa esta acción
}

interface ScoreFactorDelta {
  factor: "payment_history" | "utilization" | "credit_age" | "credit_mix" | "inquiries";
  weight: number;            // peso FICO: .35 .30 .15 .10 .10
  health_before: number;     // 0-1
  health_after: number;      // 0-1
  contribution: number;      // salud ponderada ganada/perdida por este factor
}

interface BureauScoreImpact {
  bureau: Bureau;
  current_score: number;
  estimated_new_score: number;
  impact: number;            // + gana, − pierde
  impact_min: number;        // mostrar el RANGO, no el punto exacto
  impact_max: number;
  factors: ScoreFactorDelta[];
  notes: string[];           // p.ej. "Experian: no reporta la consulta I1"
}

interface SimulateScoreRequest {
  API_KEY: string;
  user_id: string;
  action?: string;           // texto libre; requerido si no mandas `actions`
  actions?: SimulatedAction[];  // ya estructuradas; se salta el LLM interpretador
  explain?: boolean;         // default true; false = sin LLM, sin explicación
}

interface SimulateScoreResponse {
  action: string;
  interpreted_as: string;    // confirmar al usuario qué se simuló
  actions: SimulatedAction[];
  impacts: BureauScoreImpact[];
  explanation: string;
  explanation_en: string;
  risk_level: "low" | "medium" | "high" | "critical";
  caveats: string[];
  caveats_en: string[];
  estimate_basis: string;    // disclaimer, mostrar siempre
  estimate_basis_en: string;
}
```

---

## 6. Patrones de UI y cómo se llaman

### A. Simulador de texto libre

El diferenciador: ningún simulador del mercado acepta una frase arbitraria. Un input, el usuario escribe, se manda tal cual.

```ts
const res = await fetch(`${API}/simulate-score`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    API_KEY, user_id,
    action: "dejar que mi tarjeta de Capital One se venza 2 meses sin pagar",
  }),
});
```

- Mostrar `interpreted_as` **antes** de los números: es la confirmación de que se entendió bien.
- Si la frase es ambigua (dos tarjetas del mismo banco) o está fuera del catálogo, vuelve `422` con el detalle; conviene mostrarlo como pregunta de aclaración, no como error rojo.
- Sugerir escenarios de arranque en vez de dejar el input vacío: *dejar vencer una cuenta, cerrar la más antigua, pagar la mitad de la tarjeta con más saldo, abrir 3 tarjetas, disputar todas las cobranzas, esperar 12 meses*.

### B. Toggles y sliders sobre el reporte (el patrón de la competencia)

Uno: `/get-simulation-options`. Dos: el usuario togglea o arrastra. Tres: `/simulate-score` con `actions` y `explain: false` — cero LLM, 0.12 s, se puede llamar en cada cambio del slider con debounce. Cuando el usuario se detiene en un escenario, repetir la llamada con `explain: true` para traer la explicación.

```ts
const opts = await getSimulationOptions(user_id);

// toggle: "¿y si elimino esta cobranza?" — en vivo, sin LLM
await simulate({ explain: false, actions: [
  { type: "remove_account", account_ref: "A9", note: "toggle cobranza LVNV" },
]});

// slider: pagar la mitad de una tarjeta
await simulate({ explain: false, actions: [
  { type: "pay_down_balance", account_ref: "A12", amount: 63, note: "slider 50%" },
]});

// el usuario se queda en este escenario: ahora sí, la explicación
await simulate({ explain: true, actions: [
  { type: "pay_down_balance", account_ref: "A12", amount: 63, note: "slider 50%" },
]});

// varias a la vez: el motor las aplica en orden
await simulate({ actions: [
  { type: "remove_account", apply_to_all: true, note: "limpiar cobranzas" },
  { type: "pay_down_balance", apply_to_all: true, account_kind: "revolving", note: "pagar revolvente" },
]});
```

`apply_to_all` en `remove_account` afecta **solo** cuentas en cobranza o charge-off, nunca el reporte completo.

### C. Techo alcanzable y meta de puntaje

Con lo que ya hay se arman las dos features que FICO lanzó en 2026:

```ts
// "Score potential": el techo si todo lo negativo se limpia
await simulate({ actions: [
  { type: "remove_account", apply_to_all: true, note: "techo: sin cobranzas" },
  { type: "remove_late_payments", apply_to_all: true, note: "techo: sin atrasos" },
  { type: "pay_down_balance", apply_to_all: true, account_kind: "revolving", note: "techo: sin saldos" },
]});

// "¿y si no hago nada?": el paso del tiempo solo
await simulate({ actions: [{ type: "wait_months", months: 12, note: "esperar un año" }] });
```

### D. Puente con las disputas

[`/get-litigation-errors`](./get-litigation-errors.md) devuelve `creditor` y `account_number` de cada error litigable. Para simular "¿cuánto gano si esta disputa sale bien?", se puede mandar el `creditor` directamente, sin pasar por las opciones:

```ts
await simulate({ actions: [
  { type: "remove_account", creditor: error.creditor, note: `disputa: ${error.error_type}` },
]});
```

El motor resuelve `creditor` por coincidencia parcial sobre el nombre real del reporte. Si no encuentra la cuenta, el impacto es 0 y aparece el aviso en `impacts[].notes` — hay que revisarlo antes de mostrar un 0 como si fuera un resultado válido.

---

## 7. Qué mostrar

**Sí:**
- El rango `impact_min` a `impact_max`, no el punto exacto. La industria comunica rangos porque el score que jala un prestamista difiere típicamente 30-50 puntos del que muestran estas herramientas.
- Los tres burós por separado. Que difieran no es un bug: cada buró reporta cuentas distintas, y eso sale en `bureaus` y en `impacts[].notes`.
- `estimate_basis` y `caveats`. Los caveats los genera el motor según el escenario: que abrir crédito nuevo con ese perfil probablemente no se apruebe, cuánto efectivo supone el escenario, que una disputa no está garantizada, que los burós tardan 30-60 días.
- `factors`, si quieres un desglose de qué movió el puntaje: `contribution` ya viene ponderado y ordenable.

**No:**
- No presentar `estimated_new_score` como "tu nuevo puntaje será". Es una estimación.
- No ocultar un `impact` de 0. Un 0 puede significar que el factor ya está en su peor nivel (un perfil con utilización al 100% no empeora por maxear otra tarjeta más). La `explanation` lo dice explícitamente; hay que dejarla pasar en lugar de sugerir que la acción es inocua.
- No convertir el resultado en recomendación de producto. "Abrir 3 tarjetas sube X puntos" puede salir positivo en un perfil con utilización alta, y el caveat de aprobación existe justo para eso.

---

## 8. Errores

| Código | Cuándo | Copy sugerido |
|---|---|---|
| `400` | `API_KEY` no coincide | error de integración, no mostrar al usuario |
| `409` | El reporte no trae puntajes por buró | "Necesitamos tu reporte con puntajes para simular" + CTA a cargar reporte |
| `422` | Faltan `action` y `actions`; la petición está fuera del catálogo; o es ambigua | mostrar el `detail` como pregunta: suele decir exactamente qué falta aclarar |
| `500` | El usuario no tiene colección creada u otro error interno | reintentar / soporte |

El endpoint prefiere `422` con explicación antes que devolver un número inventado. Ejemplos reales: *"Hay dos tarjetas de AMERICAN EXPRESS en el reporte: A13 y A10. Se necesita que indiques cuál"*, *"Ganar la lotería no está representado por una acción del catálogo"*.

---

## 9. Rendimiento y consumo

- `/get-simulation-options` no toca el LLM: 0.25 s sobre un reporte de 30 tradelines.
- `/simulate-score` con `actions` + `explain: false` tampoco: 0.12 s y 0 tokens. Es el modo para recalcular en vivo.
- `/simulate-score` con `action` son dos llamadas al LLM (interpretar + narrar): 8-9 s y unos 3.200 tokens. Spinner, y no dispararlo en cada tecla.
- Como el motor es determinista, el resultado se puede cachear en el cliente por `(user_id, hash(actions))` hasta que se recargue el reporte. El backend no cachea nada de esto hoy.
- Los headers `X-Usage-Input-Tokens`, `X-Usage-Output-Tokens`, `X-Usage-Total-Tokens` y `X-Usage-By-Model` traen el consumo, igual que en el resto de endpoints. Con `explain: false` vienen en 0.

---

## 10. Checklist de integración

- [ ] Reporte cargado y con puntajes antes de ofrecer el simulador.
- [ ] Refrescar `/get-simulation-options` después de cada recarga de reporte (los `ref` cambian).
- [ ] Mostrar `interpreted_as` antes de los números en el modo texto libre.
- [ ] Mostrar el rango, los tres burós y `estimate_basis`.
- [ ] Renderizar `caveats` (usar `caveats_en` según el idioma de la app; hay `explanation_en` para lo mismo).
- [ ] Manejar `422` como aclaración, no como fallo.
- [ ] Revisar `impacts[].notes` antes de mostrar un impacto de 0.
- [ ] Debounce en sliders, y `explain: false` mientras el usuario arrastra.
