"""
Tests for /add-lesson, /get-lessons, /generate-plan endpoints.
Requires the server running at BASE_URL and a user with credit data already loaded.
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"
API_KEY = "pBykKtG13DCmoUuV5U8dEumBUczMJhMVPewahVcu6PtFpR8UzuaX2qJnoxrSaItnZ59GRbf3YrhowTzNC8KmGsTqzBJbv46ocpXI7DmTwweQGE75WklzZJxR8tgyR8Vf"
USER_ID = "juan"

SAMPLE_LESSONS = [
    {
        "lesson_id": "lesson_001",
        "title": "¿Qué es el puntaje de crédito?",
        "description": "Aprende cómo se calcula tu puntaje FICO y los 5 factores que lo componen.",
        "level_hint": 1,
    },
    {
        "lesson_id": "lesson_002",
        "title": "Cómo reducir tu utilización de crédito",
        "description": "Estrategias prácticas para bajar tu ratio de utilización por debajo del 30%.",
        "level_hint": 1,
    },
    {
        "lesson_id": "lesson_003",
        "title": "Disputar errores en tu reporte",
        "description": "Guía paso a paso para identificar y disputar errores ante los burós de crédito.",
        "level_hint": 2,
    },
    {
        "lesson_id": "lesson_004",
        "title": "Cuentas aseguradas: cómo usarlas",
        "description": "Cómo usar una secured credit card para construir historial positivo.",
        "level_hint": 2,
    },
    {
        "lesson_id": "lesson_005",
        "title": "Cómo solicitar un aumento de límite",
        "description": "Cuándo y cómo pedir un aumento de límite sin arriesgar tu puntaje.",
        "level_hint": 3,
    },
]

PASS = "✅"
FAIL = "❌"


def section(title: str):
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


def check(condition: bool, label: str, detail: str = ""):
    status = PASS if condition else FAIL
    line = f"  {status} {label}"
    if detail:
        line += f"  →  {detail}"
    print(line)
    return condition


# ── 1. Seed lessons ────────────────────────────────────────────────────────────

def test_add_lessons():
    section("1. POST /add-lesson  (seed 5 lessons)")
    all_ok = True
    for lesson in SAMPLE_LESSONS:
        resp = requests.post(
            f"{BASE_URL}/add-lesson",
            json={"API_KEY": API_KEY, "lesson": lesson},
            timeout=10,
        )
        ok = resp.status_code == 200 and resp.json().get("ok") is True
        all_ok = all_ok and ok
        check(ok, lesson["lesson_id"], f"status={resp.status_code}")
    return all_ok


def test_add_lesson_wrong_key():
    section("1b. POST /add-lesson  (wrong API key → 400)")
    resp = requests.post(
        f"{BASE_URL}/add-lesson",
        json={"API_KEY": "wrong", "lesson": SAMPLE_LESSONS[0]},
        timeout=10,
    )
    return check(resp.status_code == 400, "rejects bad API key", f"status={resp.status_code}")


# ── 2. Get lessons ─────────────────────────────────────────────────────────────

def test_get_lessons():
    section("2. GET /get-lessons")
    resp = requests.get(
        f"{BASE_URL}/get-lessons",
        params={"api_key": API_KEY},
        timeout=10,
    )
    ok_status = check(resp.status_code == 200, "returns 200", f"status={resp.status_code}")
    if not ok_status:
        return False

    lessons = resp.json()
    ok_count = check(len(lessons) >= 5, f"at least 5 lessons returned", f"got {len(lessons)}")

    ids = {l["lesson_id"] for l in lessons}
    seeded_ids = {l["lesson_id"] for l in SAMPLE_LESSONS}
    ok_ids = check(seeded_ids.issubset(ids), "all seeded lessons present")

    ok_schema = True
    for l in lessons:
        has_all = all(k in l for k in ("lesson_id", "title", "description", "level_hint"))
        if not has_all:
            ok_schema = False
    check(ok_schema, "all lessons have correct schema fields")

    return ok_status and ok_count and ok_ids and ok_schema


# ── 3. Generate plan ───────────────────────────────────────────────────────────

def test_generate_plan():
    section("3. POST /generate-plan")
    payload = {"API_KEY": API_KEY, "user_id": USER_ID}

    t0 = time.time()
    resp = requests.post(f"{BASE_URL}/generate-plan", json=payload, timeout=180)
    elapsed = time.time() - t0

    ok_status = check(resp.status_code == 200, f"returns 200  ({elapsed:.1f}s)", f"status={resp.status_code}")
    if not ok_status:
        print(f"  Response body: {resp.text[:300]}")
        return False

    plan = resp.json()

    # Top-level structure
    ok_levels = check("levels" in plan, "response has 'levels' key")
    if not ok_levels:
        return False

    levels = plan["levels"]
    check(len(levels) == 5, "exactly 5 levels", f"got {len(levels)}")

    EXPECTED_NAMES = [
        "Conoce tu crédito",
        "Construyendo tu base",
        "Optimizando",
        "Ampliando",
        "Crédito Pro",
    ]
    EXPECTED_STATUSES = {"completed", "in_progress", "locked"}

    names_ok = True
    for i, level in enumerate(levels):
        expected_name = EXPECTED_NAMES[i] if i < len(EXPECTED_NAMES) else f"Level {i+1}"
        name_match = level.get("name") == expected_name
        names_ok = names_ok and name_match
        check(
            name_match,
            f"level {level.get('level')} name",
            f"'{level.get('name')}'"
        )

    # Status validity
    status_ok = all(l.get("status") in EXPECTED_STATUSES for l in levels)
    check(status_ok, "all level statuses are valid")

    # Months and weeks
    month_ok = True
    week_ok = True
    score_ok = True
    task_type_ok = True
    lesson_id_ok = True

    valid_task_types = {"action", "dispute", "lesson"}
    seeded_lesson_ids = {l["lesson_id"] for l in SAMPLE_LESSONS}

    for level in levels:
        for month in level.get("months", []):
            if month.get("estimated_score_gain", 0) <= 0:
                score_ok = False
            tasks = month.get("tasks", [])
            if len(tasks) != 4:
                week_ok = False
            for task in tasks:
                if task.get("task_type") not in valid_task_types:
                    task_type_ok = False
                if task.get("task_type") == "lesson":
                    lid = task.get("lesson_id")
                    if lid not in seeded_lesson_ids:
                        lesson_id_ok = False

    check(month_ok, "all levels have at least 1 month")
    check(week_ok, "all months have exactly 4 tasks")
    check(score_ok, "all months have positive estimated_score_gain")
    check(task_type_ok, "all task_type values are valid (action/dispute/lesson)")
    check(lesson_id_ok, "lesson tasks reference valid seeded lesson_ids")

    return True


def test_generate_plan_wrong_key():
    section("3b. POST /generate-plan  (wrong API key → 400)")
    resp = requests.post(
        f"{BASE_URL}/generate-plan",
        json={"API_KEY": "bad", "user_id": USER_ID},
        timeout=10,
    )
    return check(resp.status_code == 400, "rejects bad API key", f"status={resp.status_code}")


def test_generate_plan_unknown_user():
    section("3c. POST /generate-plan  (unknown user → server error or empty plan)")
    resp = requests.post(
        f"{BASE_URL}/generate-plan",
        json={"API_KEY": API_KEY, "user_id": "user_that_does_not_exist_xyz"},
        timeout=60,
    )
    handled = resp.status_code in (200, 400, 422, 500)
    return check(handled, "server handles unknown user gracefully", f"status={resp.status_code}")


# ── Runner ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  DumboAI — Plan Endpoints Test Suite")
    print("=" * 55)

    results = []
    results.append(("add_lessons",           test_add_lessons()))
    results.append(("add_lesson_wrong_key",   test_add_lesson_wrong_key()))
    results.append(("get_lessons",            test_get_lessons()))
    results.append(("generate_plan_wrong_key",test_generate_plan_wrong_key()))
    results.append(("generate_plan_unknown",  test_generate_plan_unknown_user()))
    results.append(("generate_plan",          test_generate_plan()))   # last — slowest

    section("SUMMARY")
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        status = PASS if ok else FAIL
        print(f"  {status}  {name}")
    print(f"\n  {passed}/{len(results)} test groups passed")
    print()
