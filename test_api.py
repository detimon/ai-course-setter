"""
test_api.py - Automatisierter End-to-End Test für das laufende FastAPI-Backend & Frontend.
Verwendet ausschliesslich die Standardbibliothek (urllib.request), keine externen Abhängigkeiten.
"""

import urllib.request
import urllib.error
import json
import sys

BASE_URL = "http://127.0.0.1:8000"

def get(path: str) -> dict:
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=10) as resp:
        assert resp.status == 200, f"Expected 200, got {resp.status}"
        return json.loads(resp.read().decode("utf-8"))

def post(path: str, data: dict) -> dict:
    url = f"{BASE_URL}{path}"
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        assert resp.status == 200, f"Expected 200, got {resp.status}"
        return json.loads(resp.read().decode("utf-8"))

def check_status(path: str) -> int:
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status

def test_api_terrain():
    print("1. Test GET /api/terrain...")
    data = get("/api/terrain")
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) >= 10, "Terrain features should include zones, lines, candidates"
    print(f"   [OK] Geländemodell mit {len(data['features'])} Features erfolgreich empfangen.")

def test_api_generate():
    print("\n2. Test POST /api/generate (Schieberegler-Parameter: Varianz & K-Pfade)...")
    payload = {
        "distance_m": 1050.0,
        "elevation_m": 20.0,
        "num_controls": 4,
        "num_routes": 3,
        "leg_variance": 0.6,
        "min_angle_deg": 60.0,
        "allow_hairpin": False
    }
    data = post("/api/generate", payload)
    assert data["status"] == "success"
    course = data["course"]
    assert len(course["controls"]) == 6, "Expected 6 controls (Start + 4 controls + Finish)"
    assert course["is_geometrically_valid"] is True, "Course should be geometrically valid"
    assert course["leg_variance_cv"] > 0.25, "Variance CV should be active"
    print(f"   [OK] Bahn erfolgreich generiert: {course['total_distance_m']}m, {course['formatted_total_time']}, CV={course['leg_variance_cv']}.")
    print(f"        Teilstrecken: {len(course['legs'])}, Routenwahl-Score: {course['total_route_choice_score']}.")

def test_api_obstacle():
    print("\n3. Test POST /api/obstacle (Interaktives Sperrgebiet & Reaktiver Loop)...")
    payload = {
        "id": "test_barrier",
        "points": [
            [235.0, 115.0], [265.0, 115.0],
            [265.0, 145.0], [235.0, 145.0]
        ],
        "reason": "Test-Sperrung Forstbetrieb"
    }
    data = post("/api/obstacle", payload)
    assert data["status"] == "success"
    print(f"   [OK] Sperrgebiet reaktiv verarbeitet. Neu geroutete Legs: {data['rerouted_legs']}")

def test_api_move_control():
    print("\n4. Test POST /api/move-control (Drag & Drop)...")
    course_data = get("/api/course")["course"]
    assert course_data is not None, "Active course should exist"
    first_ctrl = course_data["controls"][1]  # Erster Zwischenposten der aktiven Bahn
    payload = {
        "control_id": first_ctrl["id"],
        "new_x": first_ctrl["x"] + 10.0,
        "new_y": first_ctrl["y"] + 10.0
    }
    data = post("/api/move-control", payload)
    assert data["status"] == "success"
    print(f"   [OK] Posten {first_ctrl['id']} verschoben. Neue Gesamtlänge: {data['course']['total_distance_m']}m.")

def test_static_assets():
    print("\n5. Test Statische Frontend-Assets...")
    for path in ["/", "/style.css", "/app.js", "/vendor/leaflet.js", "/vendor/leaflet.css"]:
        status = check_status(path)
        assert status == 200, f"Failed to load {path}: {status}"
    print("   [OK] Alle Frontend-Dateien (HTML, CSS, JS, Leaflet) werden fehlerfrei ausgeliefert.")

if __name__ == "__main__":
    print("=" * 64)
    print("   AI COURSE SETTER - E2E BACKEND & FRONTEND TESTS")
    print("=" * 64)
    try:
        test_api_terrain()
        test_api_generate()
        test_api_obstacle()
        test_api_move_control()
        test_static_assets()
        print("\n" + "=" * 64)
        print("   ALLE E2E-TESTS ERFOLGREICH BESTANDEN!")
        print("=" * 64)
    except urllib.error.URLError as e:
        print(f"\n[FEHLER] Konnte Server nicht erreichen: {e}")
        print("Stelle sicher, dass app.py läuft (z.B. python3 app.py)")
        sys.exit(1)
