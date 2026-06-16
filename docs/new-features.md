# DumboAI — Nuevos Features: Plan de Reparación y Simulador de Puntaje

Documentación completa de los 4 endpoints nuevos agregados al sistema.

---

## Base URL

```
http://<host>:8000
```

---

## Autenticación

Todos los endpoints requieren la `API_KEY` del proyecto:
- En endpoints `POST`: campo `API_KEY` dentro del body JSON.
- En endpoints `GET`: query parameter `api_key`.

Respuesta ante key inválida: `400 Bad Request`.

---

## Endpoints

### `POST /add-lesson`

Agrega o actualiza una lección del curso. Si el `lesson_id` ya existe, lo sobreescribe.

**Body:**

```json
{
  "API_KEY": "tu_api_key",
  "lesson": {
    "lesson_id": "lesson_001",
    "title": "¿Qué es el puntaje de crédito?",
    "description": "Aprende cómo se calcula tu puntaje FICO y los 5 factores que lo componen.",
    "level_hint": 1
  }
}
```

| Campo | Tipo | Descripción |
|---|---|---|
| `lesson_id` | `string` | ID único de la lección. Se usa como clave de upsert. |
| `title` | `string` | Título corto visible al usuario. |
| `description` | `string` | Descripción del contenido de la lección. |
| `level_hint` | `int` (1–5) | Nivel del plan al que pertenece esta lección. |

**Respuesta `200`:**

```json
{ "ok": true }
```

---

### `GET /get-lessons`

Retorna todas las lecciones disponibles en el sistema.

**Request:**

```
GET /get-lessons?api_key=tu_api_key
```

**Respuesta `200`:**

```json
[
  {
    "lesson_id": "lesson_001",
    "title": "¿Qué es el puntaje de crédito?",
    "description": "Aprende cómo se calcula tu puntaje FICO y los 5 factores que lo componen.",
    "level_hint": 1
  },
  {
    "lesson_id": "lesson_002",
    "title": "Cómo reducir tu utilización de crédito",
    "description": "Estrategias prácticas para bajar tu ratio de utilización por debajo del 30%.",
    "level_hint": 1
  }
]
```

---

### `POST /generate-plan`

Genera un plan de reparación de crédito personalizado de 5 niveles para un usuario.

La IA analiza el reporte de crédito del usuario almacenado en el sistema y produce tareas semanales concretas (acciones financieras, disputas, lecciones del curso) con el estimado de puntos que ganará cada mes.

> **Requisito:** El usuario debe tener su reporte de crédito cargado previamente en el sistema.

> **Tiempo de respuesta:** entre 15 y 25 segundos.

**Body:**

```json
{
  "API_KEY": "tu_api_key",
  "user_id": "usuario_123"
}
```

**Respuesta `200`:**

```json
{
  "levels": [
    {
      "level": 1,
      "name": "Conoce tu crédito",
      "status": "in_progress",
      "months": [
        {
          "month": 1,
          "title": "Bajar el uso de tu crédito",
          "description": "Reducir tu utilización del 99% al 30% es lo que más puede mover tu puntaje en el corto plazo.",
          "estimated_score_gain": 30,
          "tasks": [
            {
              "week": 1,
              "title": "Pagar $200 a Credit One",
              "task_type": "action",
              "lesson_id": null
            },
            {
              "week": 2,
              "title": "Pagar $150 a Capital One",
              "task_type": "action",
              "lesson_id": null
            },
            {
              "week": 3,
              "title": "No usar las tarjetas esta semana",
              "task_type": "action",
              "lesson_id": null
            },
            {
              "week": 4,
              "title": "¿Qué es el puntaje de crédito?",
              "task_type": "lesson",
              "lesson_id": "lesson_001"
            }
          ]
        }
      ]
    },
    {
      "level": 2,
      "name": "Construyendo tu base",
      "status": "locked",
      "months": []
    },
    {
      "level": 3,
      "name": "Optimizando",
      "status": "locked",
      "months": []
    },
    {
      "level": 4,
      "name": "Ampliando",
      "status": "locked",
      "months": []
    },
    {
      "level": 5,
      "name": "Crédito Pro",
      "status": "locked",
      "months": []
    }
  ]
}
```

#### Referencia de campos — `/generate-plan`

**`LevelPlan`**

| Campo | Tipo | Descripción |
|---|---|---|
| `level` | `int` | Número de nivel, del 1 al 5. |
| `name` | `string` | Nombre fijo del nivel (ver tabla abajo). |
| `status` | `string` | Estado actual del nivel. |
| `months` | `array` | Lista de meses con tareas dentro del nivel. |

**Nombres de niveles (siempre en este orden):**

| `level` | `name` |
|---|---|
| 1 | `"Conoce tu crédito"` |
| 2 | `"Construyendo tu base"` |
| 3 | `"Optimizando"` |
| 4 | `"Ampliando"` |
| 5 | `"Crédito Pro"` |

**Valores de `status`:**

| Valor | Significado |
|---|---|
| `"in_progress"` | Nivel activo, el usuario está trabajando en él. |
| `"completed"` | Nivel completado. |
| `"locked"` | Nivel bloqueado, aún no desbloqueado. |

**`MonthPlan`**

| Campo | Tipo | Descripción |
|---|---|---|
| `month` | `int` | Número de mes dentro del nivel (empieza en 1). |
| `title` | `string` | Tema principal del mes. |
| `description` | `string` | Explicación breve de por qué este mes es importante. |
| `estimated_score_gain` | `int` | Puntos de crédito estimados que el usuario ganará este mes. |
| `tasks` | `array` | Siempre 4 tareas, una por semana. |

**`WeekTask`**

| Campo | Tipo | Descripción |
|---|---|---|
| `week` | `int` | Número de semana dentro del mes (1 al 4). |
| `title` | `string` | Acción concreta que debe hacer el usuario esa semana. |
| `task_type` | `string` | Tipo de tarea (ver tabla abajo). |
| `lesson_id` | `string \| null` | Solo cuando `task_type` es `"lesson"`. Referencia el `lesson_id` de `/get-lessons`. En cualquier otro tipo es `null`. |

**Valores de `task_type`:**

| Valor | Descripción |
|---|---|
| `"action"` | Acción financiera: pago, apertura de cuenta, solicitar aumento de límite, etc. |
| `"dispute"` | Disputar un error en el reporte: inquiry, colección, charge-off, etc. |
| `"lesson"` | Completar una lección del curso. El `lesson_id` indica cuál. |

---

### `POST /simulate-score`

Simula el impacto en el puntaje de crédito del usuario ante una acción específica, descrita en lenguaje natural. Devuelve el impacto estimado en puntos por buró (TransUnion, Equifax, Experian), el nivel de riesgo y una explicación personalizada basada en el reporte real del usuario.

> **Requisito:** El usuario debe tener su reporte de crédito cargado previamente en el sistema.

> **Tiempo de respuesta:** entre 3 y 8 segundos.

**Body:**

```json
{
  "API_KEY": "tu_api_key",
  "user_id": "usuario_123",
  "action": "dejar que mi cuenta de Capital One se venza 2 meses sin pagar"
}
```

| Campo | Tipo | Descripción |
|---|---|---|
| `action` | `string` | Descripción en lenguaje natural de la acción a simular. Puede ser positiva (pagar deuda, abrir cuenta) o negativa (dejar vencer, cerrar cuenta, etc.). |

**Respuesta `200`:**

```json
{
  "action": "dejar que mi cuenta de Capital One se venza 2 meses sin pagar",
  "impacts": [
    {
      "bureau": "TransUnion",
      "current_score": 607,
      "estimated_new_score": 542,
      "impact": -65
    },
    {
      "bureau": "Equifax",
      "current_score": 607,
      "estimated_new_score": 540,
      "impact": -67
    },
    {
      "bureau": "Experian",
      "current_score": 604,
      "estimated_new_score": 539,
      "impact": -65
    }
  ],
  "explanation": "Dejar vencer una cuenta 2 meses impacta directamente el historial de pagos, que representa el 35% del puntaje FICO. Dado que el perfil ya muestra utilización alta, el daño es especialmente severo. Capital One reporta a los 3 burós, por lo que todos se ven afectados por igual.",
  "risk_level": "critical"
}
```

#### Referencia de campos — `/simulate-score`

**`BureauScoreImpact`**

| Campo | Tipo | Descripción |
|---|---|---|
| `bureau` | `string` | Nombre del buró: `"TransUnion"`, `"Equifax"` o `"Experian"`. |
| `current_score` | `int` | Puntaje actual del usuario en ese buró, tomado directamente del reporte. |
| `estimated_new_score` | `int` | Puntaje estimado tras la acción (`current_score + impact`). |
| `impact` | `int` | Puntos ganados (positivo) o perdidos (negativo) por la acción. |

**`risk_level`**

| Valor | Rango de impacto |
|---|---|
| `"low"` | Menos de 10 puntos |
| `"medium"` | Entre 10 y 30 puntos |
| `"high"` | Entre 30 y 60 puntos |
| `"critical"` | Más de 60 puntos |

> El `risk_level` se calcula sobre el **peor impacto individual** entre todos los burós.

**Ejemplos de acciones válidas:**

| Tipo | Ejemplo |
|---|---|
| Negativa | `"dejar que mi cuenta de Capital One se venza 2 meses"` |
| Negativa | `"cerrar mi tarjeta de crédito más antigua"` |
| Negativa | `"abrir 3 tarjetas de crédito nuevas este mes"` |
| Positiva | `"pagar el 50% del saldo de mi tarjeta con mayor balance"` |
| Positiva | `"pagar en su totalidad mi cuenta de Credit One"` |
| Positiva | `"disputar y eliminar el inquiry de Citibank"` |

---

## Errores (todos los endpoints)

| Código | Causa |
|---|---|
| `400` | API key inválida. |
| `500` | El `user_id` no tiene reporte de crédito cargado en el sistema, o error interno. |

---

## Flujo de integración

```
SETUP (una sola vez, operación de admin)
└── POST /add-lesson  ×N  →  cargar todas las lecciones del curso

PANTALLA: Plan del usuario
└── POST /generate-plan  →  5 niveles con tareas semanales y puntos estimados por mes
    └── Si la tarea es tipo "lesson": mostrar contenido usando el lesson_id

PANTALLA: Simulador de puntaje
└── Usuario escribe una acción en lenguaje natural
└── POST /simulate-score  →  impacto por buró + explicación + risk_level

CUANDO EL USUARIO SUBE DE NIVEL
└── POST /generate-plan  →  volver a llamar para actualizar el status de los niveles
```
