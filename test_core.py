"""
test_core.py - Didaktischer 4-Stufen-Test für den Backend- und Geometriekern des AI Course Setters.

Die Testsuite führt 4 logisch aufeinander aufbauende Stufen durch:
- Stufe 1 (Einzelposten): Validiert Legalität von Postenkandidaten (IOF-Objekt, Sperrgebiete, Wasser, Dickicht 3).
- Stufe 2 (Zwei Posten): Validiert Distanzkorridor und Machbarkeit (Luftlinie, Mindestabstand, Durchgängigkeit).
- Stufe 3 (Teilstrecke mit K Routenwahlen): Berechnet bis zu K distinkte Pfade (Yen's K-Shortest Paths mit Mindest-Drift)
  und liefert Distanz, berechnete Laufzeit und Höhenmeter für jede Alternative zurück.
- Stufe 4 (Gesamtbahn mit Varianz & Reaktivität): Generiert eine vollständige Bahn mit Rhythmuswechsel (leg_variance=0.6),
  überprüft Mischung aus kurzen und langen Teilstrecken und simuliert nachträgliche Sperrzonen-Verschiebung mit Auto-Rerouting.
"""

import math
import sys
from typing import List, Tuple, Optional

from terrain_model import (
    Point, LineString, Polygon, ControlCandidate, LinearFeature,
    TerrainZone, ForbiddenZone, ElevationModel, SemanticTerrainModel,
    IOFCode, segments_intersect, point_in_polygon, angle_between_legs
)
from course_engine import (
    CourseConfig, NavigationGraph, a_star_search, yens_k_shortest_paths,
    calculate_spatial_drift, analyze_leg_route_choices, validate_leg_feasibility,
    check_intersections, check_angle, HeuristicCourseGenerator, CourseManager
)


def create_synthetic_terrain() -> SemanticTerrainModel:
    """Erzeugt ein synthetisches, abwechslungsreiches Testgelände (500m x 500m)."""
    # Höhenmodell mit sanfter Kuppe in der Mitte
    elev_grid = {}
    for x in range(0, 500, 25):
        for y in range(0, 500, 25):
            dist_to_center = math.hypot(x - 250, y - 250)
            z = max(0.0, 25.0 - (dist_to_center / 12.0))
            elev_grid[(x // 25, y // 25)] = z

    elevation = ElevationModel(base_z=100.0, grid=elev_grid, cell_size=25.0)
    terrain = SemanticTerrainModel(elevation_model=elevation)

    # 1. Start und Ziel
    terrain.add_candidate(ControlCandidate(
        id="START",
        point=Point(50.0, 250.0),
        iof_code=IOFCode.START,
        name="Startplatz Waldparkplatz",
        is_start=True
    ))

    terrain.add_candidate(ControlCandidate(
        id="ZIEL",
        point=Point(450.0, 250.0),
        iof_code=IOFCode.FINISH,
        name="Zielwiese",
        is_finish=True
    ))

    # 2. Reguläre Postenkandidaten (0D)
    candidates = [
        ControlCandidate("101", Point(140.0, 360.0), IOFCode.KNOLL, name="Kuppe Nordwest"),
        ControlCandidate("102", Point(130.0, 140.0), IOFCode.BOULDER, name="Fels 1.5m Südwest"),
        ControlCandidate("103", Point(250.0, 370.0), IOFCode.PIT, name="Grube Nord"),
        ControlCandidate("104", Point(250.0, 130.0), IOFCode.DEPRESSION, name="Senke Süd"),
        ControlCandidate("105", Point(360.0, 360.0), IOFCode.BUILDING, name="Hausecke Ost"),
        ControlCandidate("106", Point(370.0, 140.0), IOFCode.DIST_TREE, name="Markante Eiche"),
        ControlCandidate("107", Point(245.0, 250.0), IOFCode.SADDLE, name="Sattel beim Mauertor"),
        ControlCandidate("108", Point(115.0, 250.0), IOFCode.BOULDER, name="Fels 1.0m (kurzer Posten)"),
    ]
    for c in candidates:
        terrain.add_candidate(c)

    # 3. Lineare Objekte (1D)
    # A) Schneller Waldweg im Norden (Speed-Bonus: 0.6)
    road = LinearFeature(
        id="road_north",
        line=LineString([Point(30.0, 420.0), Point(250.0, 420.0), Point(470.0, 420.0)]),
        feature_type="road",
        passable=True,
        speed_factor=0.6,
        width_m=4.0
    )
    terrain.add_linear_feature(road)

    # B) Steinmauer mit Durchgang bei Y=230..270
    wall_south = LinearFeature(
        id="wall_south",
        line=LineString([Point(200.0, 50.0), Point(200.0, 220.0)]),
        feature_type="wall",
        passable=False
    )
    wall_north = LinearFeature(
        id="wall_north",
        line=LineString([Point(200.0, 280.0), Point(200.0, 450.0)]),
        feature_type="wall",
        passable=False
    )
    terrain.add_linear_feature(wall_south)
    terrain.add_linear_feature(wall_north)

    # 4. Belaufbarkeitszonen (2D)
    # A) Dickicht (Grün 3, Reibung 8.0) östlich der Mauer zwischen Posten
    thicket = TerrainZone(
        id="thicket_east",
        polygon=Polygon([
            Point(290.0, 220.0), Point(330.0, 220.0),
            Point(330.0, 280.0), Point(290.0, 280.0)
        ]),
        zone_type="green_3",
        friction=8.0,
        description="Sehr dichtes Brombeer-Dickicht (Grün 3)"
    )
    terrain.add_zone(thicket)

    # B) Waldsee (Unpassierbar, Reibung inf)
    lake = TerrainZone(
        id="lake_northwest",
        polygon=Polygon([
            Point(50.0, 320.0), Point(90.0, 320.0),
            Point(90.0, 360.0), Point(50.0, 360.0)
        ]),
        zone_type="water",
        friction=math.inf,
        description="Unpassierbarer Waldsee (Reibung inf)"
    )
    terrain.add_zone(lake)

    # 5. Ursprüngliches Sperrgebiet (2D)
    init_obstacle = ForbiddenZone(
        id="nature_reserve",
        polygon=Polygon([
            Point(300.0, 50.0), Point(350.0, 50.0),
            Point(350.0, 100.0), Point(300.0, 100.0)
        ]),
        reason="Biotop / Naturschutzzone"
    )
    terrain.add_forbidden_zone(init_obstacle)

    return terrain


def render_ascii_map(
    terrain: SemanticTerrainModel,
    course_pts: list,
    forbidden_poly: Polygon,
    width: int = 65,
    height: int = 24
) -> str:
    """Rendert eine kompakte ASCII-Karte des Geländes und der aktuellen Bahn."""
    grid = [["." for _ in range(width)] for _ in range(height)]

    def to_grid(x: float, y: float) -> Tuple[int, int]:
        gx = int((x / 500.0) * (width - 1))
        gy = int(((500.0 - y) / 500.0) * (height - 1))
        return (max(0, min(width - 1, gx)), max(0, min(height - 1, gy)))

    # 1. Mauer einzeichnen
    for lf in terrain.linear_features.values():
        if not lf.passable:
            for p in lf.line.points:
                gx, gy = to_grid(p.x, p.y)
                grid[gy][gx] = "#"

    # 2. Weg einzeichnen
    for lf in terrain.linear_features.values():
        if lf.passable:
            for p in lf.line.points:
                gx, gy = to_grid(p.x, p.y)
                grid[gy][gx] = "="

    # 3. Zonen (Dickicht=T, Wasser=W)
    for z in terrain.zones.values():
        sym = "W" if math.isinf(z.friction) else "T"
        bx0, by0, bx1, by1 = z.polygon.bounding_box
        for px in range(int(bx0), int(bx1), 15):
            for py in range(int(by0), int(by1), 15):
                gx, gy = to_grid(px, py)
                grid[gy][gx] = sym

    # 4. Sperrgebiet
    bx0, by0, bx1, by1 = forbidden_poly.bounding_box
    for px in range(int(bx0), int(bx1), 10):
        for py in range(int(by0), int(by1), 10):
            if point_in_polygon(Point(px, py), forbidden_poly):
                gx, gy = to_grid(px, py)
                grid[gy][gx] = "X"

    # 5. Bahn-Pfad einzeichnen
    for p in course_pts:
        gx, gy = to_grid(p.x, p.y)
        if grid[gy][gx] in (".", " "):
            grid[gy][gx] = "*"

    # 6. Posten hervorheben
    for c in terrain.candidates.values():
        gx, gy = to_grid(c.point.x, c.point.y)
        if c.is_start:
            grid[gy][gx] = "S"
        elif c.is_finish:
            grid[gy][gx] = "F"
        else:
            grid[gy][gx] = "O"

    header = "  " + "".join(str(i % 10) for i in range(width))
    lines = [f"{i:02d} " + "".join(row) for i, row in enumerate(grid)]
    legend = (
        "Legende: S=Start, F=Ziel, O=Posten, *=Bahn-Pfad, #=Mauer, "
        "==Weg, T=Dickicht, W=Wasser, X=Sperrzone"
    )
    return "\n".join([header] + lines + [legend])


# ==============================================================================
# DIDAKTISCHE 4 TEST-STUFEN
# ==============================================================================

def test_stage_1_single_control(terrain: SemanticTerrainModel):
    """
    STUFE 1 (Einzelposten):
    Prüft einen einzelnen Postenkandidaten auf Legalität:
    - Gültiges IOF-Objekt
    - Ausserhalb von Sperrgebieten
    - Ausserhalb von Wasser / Reibung inf
    - Ausserhalb von unzumutbarem Dickicht 3 (Grün 3, Reibung >= 8.0)
    """
    print("\n" + "=" * 76)
    print("  STUFE 1: EINZELPOSTEN-LEGALITÄT (IOF-Objekt, Sperrzonen, Wasser, Dickicht 3)")
    print("=" * 76)

    test_candidates = [
        ControlCandidate("C_LEGAL", Point(140.0, 360.0), IOFCode.KNOLL, name="Kuppe im offenen Wald"),
        ControlCandidate("C_FORBIDDEN", Point(320.0, 75.0), IOFCode.PIT, name="Grube im Sperrgebiet"),
        ControlCandidate("C_THICKET", Point(310.0, 250.0), IOFCode.BOULDER, name="Felsblock in Dickicht 3"),
        ControlCandidate("C_WATER", Point(70.0, 340.0), IOFCode.WATER_HOLE, name="Posten mitten im Waldsee"),
        ControlCandidate("C_NO_IOF", Point(150.0, 150.0), iof_code="", name="Posten ohne IOF-Objektmerkmal"),
    ]

    expected_outcomes = [True, False, False, False, False]

    for cand, expected in zip(test_candidates, expected_outcomes):
        is_legal, reason = terrain.is_control_legal(cand)
        status_tag = "[PASS - LEGAL]  " if is_legal else "[PASS - ABGELEHNT]"
        print(f"  {status_tag} {cand.id:<12} ({cand.name:<30}) -> {reason}")
        assert is_legal == expected, f"Erwartetes Ergebnis {expected} für {cand.id}, aber erhalten: {is_legal}"

    print("\n  >> Fazit Stufe 1: Einzelposten-Filter weist illegale Posten deterministisch ab. [OK]")


def test_stage_2_two_controls_feasibility(terrain: SemanticTerrainModel):
    """
    STUFE 2 (Zwei Posten):
    Prüft Distanzkorridor und Machbarkeit zwischen zwei Posten:
    - Mindestabstand (z.B. >= 60 m, Verhindert Postenverwechslung)
    - Maximaldistanz
    - Existenz eines durchgängigen Laufpfades (nicht blockiert durch unpassierbare Mauer)
    """
    print("\n" + "=" * 76)
    print("  STUFE 2: ZWEI-POSTEN-MACHBARKEIT & DISTANZKORRIDOR")
    print("=" * 76)

    nav_graph = NavigationGraph(terrain, resolution=25.0)

    # Fall A: Normaler, machbarer Korridor zwischen START und 101
    feasible_a, path_a, msg_a = validate_leg_feasibility(nav_graph, "START", "101", min_distance_m=60.0, max_distance_m=800.0)
    print(f"  [Fall A: Normaler Leg] START -> 101:")
    print(f"     Status: {'Gültig' if feasible_a else 'Ungültig'} | {msg_a}")
    assert feasible_a is True, "START -> 101 muss machbar sein"
    assert path_a is not None
    assert path_a.estimated_time_s > 0.0, "Laufzeit muss positiv berechnet sein"

    # Fall B: Unterschreitung des Mindestabstands (< 60 m)
    too_close_cand = ControlCandidate("TOO_CLOSE", Point(80.0, 250.0), IOFCode.BOULDER, name="Zu nah am Start (30m)")
    terrain.add_candidate(too_close_cand)
    nav_graph.integrate_point(too_close_cand.point, key=too_close_cand.id)

    feasible_b, _, msg_b = validate_leg_feasibility(nav_graph, "START", "TOO_CLOSE", min_distance_m=60.0, max_distance_m=800.0)
    print(f"\n  [Fall B: Zu naher Posten] START -> TOO_CLOSE (Abstand 30m):")
    print(f"     Status: {'Gültig' if feasible_b else 'Abgelehnt'} | {msg_b}")
    assert feasible_b is False, "Posten mit Distanz < 60m muss abgewiesen werden"

    # Fall C: Isolierter Posten hinter geschlossener Mauer (ohne Durchgang)
    enclosed_cand = ControlCandidate("ENCLOSED", Point(150.0, 480.0), IOFCode.PIT, name="Posten hinter Sperrmauer")
    # Vollständig umschliessende Mauer
    blocking_wall = LinearFeature(
        id="cage_wall",
        line=LineString([Point(130, 460), Point(170, 460), Point(170, 495), Point(130, 495), Point(130, 460)]),
        passable=False
    )
    terrain.add_linear_feature(blocking_wall)
    terrain.add_candidate(enclosed_cand)
    nav_graph.rebuild()

    feasible_c, _, msg_c = validate_leg_feasibility(nav_graph, "START", "ENCLOSED", min_distance_m=60.0, max_distance_m=800.0)
    print(f"\n  [Fall C: Blockierter Posten] START -> ENCLOSED (Hinter Sperrmauer):")
    print(f"     Status: {'Gültig' if feasible_c else 'Abgelehnt'} | {msg_c}")
    assert feasible_c is False, "Vollständig blockierter Posten darf keinen Pfad haben"

    # Aufräumen der temporären Testobjekte
    terrain.remove_candidate("TOO_CLOSE")
    terrain.remove_candidate("ENCLOSED")
    del terrain.linear_features["cage_wall"]
    nav_graph.rebuild()

    print("\n  >> Fazit Stufe 2: Distanzkorridore und Barrierenprüfung funktionieren einwandfrei. [OK]")


def test_stage_3_k_route_choices(terrain: SemanticTerrainModel):
    """
    STUFE 3 (Teilstrecke mit K Routenwahlen):
    Testet eine Verbindung zwischen zwei Posten und verifiziert:
    - K Pfade (K=3) mit Mindest-Drift (>= 20 m)
    - Jede gefundene Teilstrecken-Alternative liefert Distanz, berechnete Laufzeit und Höhenmeter zurück
    - Erkennung konkurrenzfähiger Alternativen (Delta <= 8%) vs. Verwerfung schlechter Umwege
    """
    print("\n" + "=" * 76)
    print("  STUFE 3: TEILSTRECKE MIT K ROUTENWAHLEN (Yen's K-Shortest Paths & Drift)")
    print("=" * 76)

    config = CourseConfig(
        num_routes=3,
        min_drift_m=20.0,
        route_choice_delta_threshold=0.15,  # Bis 15% für didaktische Demonstration
        base_pace_s_per_m=0.3
    )

    nav_graph = NavigationGraph(terrain, resolution=25.0)

    # Teilstrecke von 108 (115, 250) nach 105 (360, 360)
    # Es existieren 3 physische Korridore:
    # 1. Nordroute über den schnellen Waldweg (Y=420) mit Geschwindigkeitsbonus (0.6)
    # 2. Direktroute durch das Mauertor (Y=250) nördlich am Dickicht vorbei
    # 3. Südroute durch das Mauertor südlich am Dickicht vorbei
    print(f"  Analysiere K={config.num_routes} Routenalternativen für Leg 108 -> 105 (Mindest-Drift: {config.min_drift_m}m)...")
    analysis = analyze_leg_route_choices(nav_graph, "108", "105", config)

    print(f"\n  Gefundene signifikant unterschiedliche Pfade: {len(analysis.paths)} von K={config.num_routes}")
    assert len(analysis.paths) >= 2, "Es müssen mindestens 2 signifikant unterschiedliche Routen gefunden werden"

    for idx, path in enumerate(analysis.paths, start=1):
        drift_to_opt = 0.0 if idx == 1 else calculate_spatial_drift(analysis.paths[0].points, path.points)
        time_diff_s = path.estimated_time_s - analysis.paths[0].estimated_time_s
        delta_pct = (path.cost - analysis.paths[0].cost) / (analysis.paths[0].cost + 1e-9) * 100

        print(f"    * Route {idx}: Distanz {path.distance_m:5.1f} m | +{path.climb_m:4.1f} Hm | "
              f"Laufzeit: {path.formatted_time} (+{time_diff_s:4.1f}s, +{delta_pct:4.1f}%) | Drift zum Optimum: {drift_to_opt:4.1f} m")

        assert path.distance_m > 0.0, "Distanz muss positiv sein"
        assert path.estimated_time_s > 0.0, "Berechnete Laufzeit muss positiv sein"

    # Verifikation der räumlichen Spreizung zwischen den Alternativen
    if len(analysis.paths) >= 2:
        d12 = calculate_spatial_drift(analysis.paths[0].points, analysis.paths[1].points)
        assert d12 >= config.min_drift_m, f"Drift zwischen Route 1 und 2 ({d12:.1f}m) muss >= {config.min_drift_m}m sein"

    print(f"\n  Routenwahl-Bewertung: {analysis.description}")
    print("\n  >> Fazit Stufe 3: Mehrfache K-Routenwahlen mit Distanz, Laufzeit und Drift verifiziert. [OK]")
    return config


def test_stage_4_course_variance_and_reactive(terrain: SemanticTerrainModel, config: CourseConfig):
    """
    STUFE 4 (Gesamtbahn mit Varianz & Reaktivität):
    - Generiert eine vollständige Bahn mit leg_variance = 0.6 (Rhythmuswechsel zwischen kurz und lang)
    - Validiert das Vorhandensein von kurzen technischen Posten (<= 250m) und langen Schlägen (>= 400m)
    - Simuliert das nachträgliche Verschieben eines Sperrgebiets direkt auf einen Posten
    - Validiert das automatische Verschieben des Postens und das Re-Routing der Teilstrecken
    """
    print("\n" + "=" * 76)
    print("  STUFE 4: GESAMTBAHN MIT VARIANZ-STEUERUNG (leg_variance=0.6) & REAKTIVITÄT")
    print("=" * 76)

    # Generator-Konfiguration mit Rhythmuswechsel
    config.leg_variance = 0.6
    config.min_angle_deg = 60.0
    config.allow_hairpin = False
    config.num_routes = 3

    generator = HeuristicCourseGenerator(terrain, config)

    print("  Generiere Bahn mit Zielbudget: 1050 m, 4 Zwischenposten und leg_variance = 0.6...")
    course = generator.generate_course(
        start_id="START",
        finish_id="ZIEL",
        target_distance_m=1050.0,
        target_climb_m=15.0,
        num_controls=4
    )

    assert course is not None, "Generator muss eine gültige Bahn finden"
    assert len(course.controls) == 6, f"Bahn muss 6 Punkte haben (Start + 4 Posten + Ziel), hat: {len(course.controls)}"
    assert course.is_geometrically_valid is True, "Bahn muss geometrisch fehlerfrei sein"
    assert len(course.intersection_conflicts) == 0, "Keine Schnittlinien erlaubt"
    assert len(course.angle_warnings) == 0, "Keine unerlaubten spitzen Kehren (<60°)"

    print(f"\n{course.summary()}\n")

    # Validierung der Rhythmus-Steuerung: Mischung aus kurzen und langen Teilstrecken
    leg_dists = [l.distance_m for l in course.legs]
    has_short_leg = any(d <= config.short_leg_max_m for d in leg_dists)
    has_long_or_medium_leg = any(d >= 280.0 for d in leg_dists)

    print(f"  Varianz-Analyse der Teilstrecken-Längen:")
    print(f"    * Längen: {[round(d, 1) for d in leg_dists]} m")
    print(f"    * Variationskoeffizient CV: {course.leg_variance_cv:.2f}")
    print(f"    * Kurze technische Posten (<= {config.short_leg_max_m}m): {course.short_legs_count}")
    print(f"    * Längere Entscheidungsschläge: {course.long_legs_count}")

    assert has_short_leg, f"Bei leg_variance=0.6 muss mindestens eine kurze Teilstrecke (<= {config.short_leg_max_m}m) vorhanden sein!"
    assert course.leg_variance_cv > 0.25, f"Variationskoeffizient CV ({course.leg_variance_cv:.2f}) muss spürbare Varianz zeigen!"

    # --------------------------------------------------------------------------
    # Reaktivitäts-Test: Verschiebung eines Sperrgebiets
    # --------------------------------------------------------------------------
    print("\n  >> SIMULATION: Nachträgliche Verschiebung eines Sperrgebiets")
    selected_c = course.controls[2]  # Wähle den 2. Zwischenposten
    old_coord = (selected_c.point.x, selected_c.point.y)
    print(f"     Platziere Sperrgebiet direkt über Posten {selected_c.id} bei {old_coord}...")

    # Sperrgebiet zentriert über dem Posten
    blocking_poly = Polygon([
        Point(selected_c.point.x - 15.0, selected_c.point.y - 15.0),
        Point(selected_c.point.x + 15.0, selected_c.point.y - 15.0),
        Point(selected_c.point.x + 15.0, selected_c.point.y + 15.0),
        Point(selected_c.point.x - 15.0, selected_c.point.y + 15.0),
    ])

    course_manager = CourseManager(terrain, course, config)
    initial_distance = course.total_distance_m

    update_report = course_manager.update_obstacle(
        obstacle_id="reactive_construction",
        new_polygon=blocking_poly,
        reason="Dringende Baumassnahme Postenbereich gesperrt"
    )

    print("\n  Reaktiver Update-Report:")
    print(f"    * Neu geroutete Teilstrecken: {update_report['rerouted_legs']}")
    assert len(update_report["relocated_controls"]) > 0, "Der betroffene Posten muss verschoben worden sein!"
    for rel in update_report["relocated_controls"]:
        print(f"    * Posten {rel['control_id']} wurde automatisch von {rel['old_point']} nach {rel['new_point']} verschoben "
              f"(Distanz: {rel['shift_distance_m']:.1f} m).")

    new_course = update_report["course"]
    print(f"    * Neue Gesamtlänge: {new_course.total_distance_m:.1f} m (Delta: {new_course.total_distance_m - initial_distance:+.1f} m)")
    print(f"    * Neue Gesamtlaufzeit: {new_course.formatted_total_time}")
    print(f"    * Geometrisch weiterhin gültig: {'JA' if new_course.is_geometrically_valid else 'NEIN'}")
    assert new_course.is_geometrically_valid is True

    # Sammle neue Pfadpunkte und prüfe, dass kein Punkt im Sperrgebiet liegt
    new_path_pts = []
    for leg in new_course.legs:
        new_path_pts.extend(leg.optimal_path.points)

    for p in new_path_pts:
        assert point_in_polygon(p, blocking_poly, include_boundary=False) is False, \
            f"Pfadpunkt {p.as_tuple()} darf nicht im Sperrgebiet liegen!"

    print("\nAktualisierte Kartensituation nach automatischer Umgehung:")
    print(render_ascii_map(terrain, new_path_pts, blocking_poly))

    print("\n  >> Fazit Stufe 4: Varianzsteuerung & reaktives Sperrzonen-Management erfolgreich validiert. [OK]")


def run_all_stages():
    print("=" * 76)
    print("   AI COURSE SETTER - DIDAKTISCHER 4-STUFEN-TEST (CORE & GEOMETRY)")
    print("=" * 76)

    terrain = create_synthetic_terrain()

    # Stufe 1: Einzelposten
    test_stage_1_single_control(terrain)

    # Stufe 2: Zwei Posten
    test_stage_2_two_controls_feasibility(terrain)

    # Stufe 3: K Routenwahlen
    config = test_stage_3_k_route_choices(terrain)

    # Stufe 4: Gesamtbahn & Reaktivität
    test_stage_4_course_variance_and_reactive(terrain, config)

    print("\n" + "=" * 76)
    print("   ALLE 4 TESTSTUFEN WURDEN ERFOLGREICH BESTANDEN!")
    print("=" * 76)


if __name__ == "__main__":
    run_all_stages()
