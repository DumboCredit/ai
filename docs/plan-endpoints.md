# DumboAI — Plan de Reparación de Crédito

Documentación de los endpoints para generar y gestionar el plan personalizado de 5 niveles.

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

---

## Referencia de campos

### `LevelPlan`

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

---

### `MonthPlan`

| Campo | Tipo | Descripción |
|---|---|---|
| `month` | `int` | Número de mes dentro del nivel (empieza en 1). |
| `title` | `string` | Tema principal del mes. |
| `description` | `string` | Explicación breve de por qué este mes es importante. |
| `estimated_score_gain` | `int` | Puntos de crédito estimados que el usuario ganará este mes. |
| `tasks` | `array` | Siempre 4 tareas, una por semana. |

---

### `WeekTask`

| Campo | Tipo | Descripción |
|---|---|---|
| `week` | `int` | Número de semana dentro del mes (1 al 4). |
| `title` | `string` | Acción concreta que debe hacer el usuario esa semana. |
| `task_type` | `string` | Tipo de tarea (ver tabla abajo). |
| `lesson_id` | `string \| null` | Solo presente cuando `task_type` es `"lesson"`. Referencia el `lesson_id` de `/get-lessons`. En cualquier otro tipo es `null`. |

**Valores de `task_type`:**

| Valor | Descripción |
|---|---|
| `"action"` | Acción financiera: pago, apertura de cuenta, solicitar aumento de límite, etc. |
| `"dispute"` | Disputar un error en el reporte: inquiry, colección, charge-off, etc. |
| `"lesson"` | Completar una lección del curso. El `lesson_id` indica cuál. |

---

## Errores

| Código | Causa |
|---|---|
| `400` | API key inválida. |
| `500` | El `user_id` no tiene reporte de crédito cargado en el sistema, o error interno. |

---

## Flujo de integración

```
1. Cargar lecciones (operación de admin, una sola vez o cuando se añadan nuevas)
   POST /add-lesson  →  { "ok": true }

2. Abrir la pantalla del plan del usuario
   POST /generate-plan  →  CreditPlan con 5 niveles

3. El usuario toca una tarea de tipo "lesson"
   → Usar el lesson_id para mostrar el contenido de la lección
   → Opcionalmente: GET /get-lessons para obtener todos los datos de las lecciones al inicio

4. Cuando el usuario completa un nivel y sube al siguiente
   → Volver a llamar POST /generate-plan
   → El status de los niveles se actualiza según el progreso
```
