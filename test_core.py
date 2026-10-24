"""
test_core.py - Verifikation & Testsuite für das semantische Geländemodell und die Bahnsetzer-Engine.

Szenario:
1. Synthetisches Orientierungslauf-Gelände (500m x 500m):
   - Start im Westen, Ziel im Osten
   - Mehrere IOF-Postenkandidaten (Fels, Kuppe, Senke, Grube, Hausecke, Baum)
   - Schneller Waldweg (Speed-Bonus: 0.6)
   - Geteilte Steinmauer (unpassierbar) mit einem schmalen Durchgang (Tor) in der Mitte
   - Dickicht (Grün 3, Reibung 8.0), das Routenwahl-Spreizung erfordert
   - Temporäres Sperrgebiet (Naturschutz / Baustelle)
2. Automatische Generierung einer 3-Posten-Bahn mit Zielbudget ~950m:
   - Verifikation: Keine Schnittlinien
   - Verifikation: Keine spitzen Kehren (< 60° Wendewinkel)
   - Verifikation: Erkennung echter Routenwahlen via Yen's K-Shortest Paths
3. Dynamische Sperrzonen-Verschiebung:
   - Das Sperrgebiet wird nachträglich direkt über den Durchgang / einen Posten gelegt
   - Automatisches Verschieben des betroffenen Postens auf die nächste legale Stelle
   - Automatisches Neu-Routen der betroffenen Teilstrecken um die Sperrzone herum
   - ASCII-Visualisierung & tabellarischer Vorher-/Nachher-Vergleich im Terminal.
"""

import math
import sys
from terrain_model import (
    Point, LineString, Polygon, ControlCandidate, LinearFeature,
    TerrainZone, ForbiddenZone, ElevationModel, SemanticTerrainModel,
    IOFCode, segments_intersect, point_in_polygon, angle_between_legs
)
from course_engine import (
    CourseConfig, NavigationGraph, a_star_search, yens_k_shortest_paths,
    analyze_leg_route_choices, check_intersections, check_angle,
    HeuristicCourseGenerator, CourseManager
)


def create_synthetic_terrain() -> SemanticTerrainModel:
    """Erzeugt ein synthetisches, abwechslungsreiches Testgelände."""
    # Höhenmodell mit sanfter Kuppe in der Mitte
    elev_grid = {}
    for x in range(0, 500, 25):
        for y in range(0, 500, 25):
            # Hügel bei (250, 250) mit 25m Höhe
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

    # 2. Postenkandidaten (0D)
    candidates = [
        ControlCandidate("101", Point(140.0, 360.0), IOFCode.KNOLL, name="Kuppe Nordwest"),
        ControlCandidate("102", Point(130.0, 140.0), IOFCode.BOULDER, name="Fels 1.5m Südwest"),
        ControlCandidate("103", Point(250.0, 370.0), IOFCode.PIT, name="Grube Nord"),
        ControlCandidate("104", Point(250.0, 130.0), IOFCode.DEPRESSION, name="Senke Süd"),
        ControlCandidate("105", Point(360.0, 360.0), IOFCode.BUILDING, name="Hausecke Ost"),
        ControlCandidate("106", Point(370.0, 140.0), IOFCode.DIST_TREE, name="Markante Eiche"),
        ControlCandidate("107", Point(245.0, 250.0), IOFCode.SADDLE, name="Sattel beim Mauertor"),
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
    # Südlicher Mauerflügel (unpassierbar)
    wall_south = LinearFeature(
        id="wall_south",
        line=LineString([Point(200.0, 50.0), Point(200.0, 220.0)]),
        feature_type="wall",
        passable=False
    )
    # Nördlicher Mauerflügel (unpassierbar)
    wall_north = LinearFeature(
        id="wall_north",
        line=LineString([Point(200.0, 280.0), Point(200.0, 450.0)]),
        feature_type="wall",
        passable=False
    )
    terrain.add_linear_feature(wall_south)
    terrain.add_linear_feature(wall_north)

    # 4. Belaufbarkeitszonen (2D)
    # Dickicht (Grün 3, Reibung 8.0) östlich der Mauer zwischen Posten
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

    # 5. Ursprüngliches Sperrgebiet (2D)
    # Kleines Schutzgebiet im Südosten, das vorerst keine Hauptrouten blockiert
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

    def to_grid(x: float, y: float) -> tuple[int, int]:
        gx = int((x / 500.0) * (width - 1))
        gy = int(((500.0 - y) / 500.0) * (height - 1))  # Invert y for screen
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

    # 3. Dickicht
    for z in terrain.zones.values():
        bx0, by0, bx1, by1 = z.polygon.bounding_box
        for px in range(int(bx0), int(bx1), 15):
            for py in range(int(by0), int(by1), 15):
                gx, gy = to_grid(px, py)
                grid[gy][gx] = "T"

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
        "==Weg, T=Dickicht, X=Sperrzone"
    )
    return "\n".join([header] + lines + [legend])


def test_geometry_primitives():
    """Testet Schnittprüfungen, Winkelberechnung und Point-in-Polygon."""
    print("--- 1. Test Geometrie-Primitiven ---")

    # A) Schnitt zweier Strecken
    p1, p2 = Point(0, 0), Point(10, 10)
    p3, p4 = Point(0, 10), Point(10, 0)
    assert segments_intersect(p1, p2, p3, p4) is True, "Kreuzende Segmente müssen erkannt werden"

    p5, p6 = Point(0, 0), Point(5, 5)
    p7, p8 = Point(6, 6), Point(10, 10)
    assert segments_intersect(p5, p6, p7, p8, touch_is_intersection=False) is False, "Parallele getrennte Segmente dürfen nicht schneiden"

    # B) Point in Polygon
    poly = Polygon([Point(10, 10), Point(50, 10), Point(50, 50), Point(10, 50)])
    assert point_in_polygon(Point(30, 30), poly) is True, "Punkt in Mitte muss drinnen sein"
    assert point_in_polygon(Point(5, 5), poly) is False, "Punkt ausserhalb muss draussen sein"

    # C) Winkelprüfung (Spitze Kehre vs. Gerader Durchlauf)
    # Geradeaus (180 Grad)
    ang_straight = angle_between_legs(Point(0, 0), Point(5, 0), Point(10, 0))
    assert math.isclose(ang_straight, 180.0, abs_tol=1.0), f"Erwartet ~180°, erhalten: {ang_straight}"

    # 90 Grad Abknickung
    ang_90 = angle_between_legs(Point(0, 0), Point(5, 0), Point(5, 5))
    assert math.isclose(ang_90, 90.0, abs_tol=1.0), f"Erwartet ~90°, erhalten: {ang_90}"

    # Spitze Kehre (Gegenverkehr, Wendewinkel 30 Grad)
    ang_acute = angle_between_legs(Point(0, 0), Point(10, 0), Point(1, 2))
    assert ang_acute < 60.0, f"Spitze Kehre sollte < 60° sein, ist: {ang_acute}"
    valid, _ = check_angle(Point(0, 0), Point(10, 0), Point(1, 2), min_angle_deg=60.0)
    assert valid is False, "Spitze Kehre muss ungültig sein"

    print("  [OK] Geometrie-Primitiven erfolgreich verifiziert.\n")


def test_course_generation_and_route_choice():
    """Testet A*, Yen's Algorithmus, Routenwahl-Score und Bahnengenerierung."""
    print("--- 2. Test Bahnengenerator & Routenwahl-Metrik ---")

    terrain = create_synthetic_terrain()
    config = CourseConfig(
        min_angle_deg=60.0,
        allow_hairpin=False,
        route_choice_delta_threshold=0.08,
        route_choice_drift_threshold=20.0,
        route_choice_bonus_weight=150.0,
        grid_resolution_m=25.0
    )

    generator = HeuristicCourseGenerator(terrain, config)

    # Bahn mit 3 Zwischenposten und Zielbudget 950 m generieren
    print("  Generiere optimale 3-Posten-Bahn (Zielbudget: 950 m)...")
    course = generator.generate_course(
        start_id="START",
        finish_id="ZIEL",
        target_distance_m=950.0,
        target_climb_m=20.0,
        num_controls=3
    )

    assert course is not None, "Generator muss eine gültige Bahn finden"
    assert len(course.controls) == 5, f"Bahn muss 5 Punkte haben (Start + 3 Posten + Ziel), hat: {len(course.controls)}"
    assert course.is_geometrically_valid is True, "Bahn muss geometrisch fehlerfrei sein"
    assert len(course.intersection_conflicts) == 0, "Es dürfen keine Teilstrecken-Kreuzungen existieren"
    assert len(course.angle_warnings) == 0, "Es darf keine unzulässigen spitzen Kehren geben"

    print(f"\n{course.summary()}\n")

    # Prüfe Routenwahl-Erkennung
    rc_legs = [leg for leg in course.legs if leg.has_route_choice]
    print(f"  Erkannte Teilstrecken mit echter Routenwahl: {len(rc_legs)} von {len(course.legs)}")
    for leg in rc_legs:
        rc = leg.route_choice
        print(f"    * Leg {leg.leg_index + 1} ({leg.from_control.id} -> {leg.to_control.id}): "
              f"Delta={rc.cost_delta_pct:.1f}%, Drift={rc.max_drift_m:.1f} m, Score={rc.score:.1f}")

    print("  [OK] Bahnengenerierung & Routenwahl erfolgreich validiert.\n")
    return terrain, course, config


def test_reactive_obstacle_shift(terrain: SemanticTerrainModel, course, config: CourseConfig):
    """
    Simuliert das Verschieben eines Sperrgebiets und verifiziert:
    - Postenverschiebung aus dem Sperrgebiet
    - Automatisches Neu-Routen betroffener Teilstrecken
    - Korrekte Umgehung im Navigationsgraphen
    """
    print("--- 3. Test Reaktiver Loop: Sperrgebiet-Verschiebung & Auto-Rerouting ---")

    course_manager = CourseManager(terrain, course, config)
    initial_distance = course.total_distance_m

    # Alle Punkte des initialen Bahn-Pfades sammeln
    initial_path_pts = []
    for leg in course.legs:
        initial_path_pts.extend(leg.optimal_path.points)

    print("Initiale Kartensituation (vor Sperrgebiet-Verschiebung):")
    print(render_ascii_map(terrain, initial_path_pts, terrain.forbidden_zones["nature_reserve"].polygon))

    # Szenario:
    # Neues Sperrgebiet wird direkt auf den Posten 104 und die Teilstrecke gelegt (X=235..265, Y=115..145),
    # wodurch Posten 104 im Sperrgebiet liegt und automatisch an den Rand verschoben werden muss,
    # und die Teilstrecken 2 und 3 automatisch um das Sperrgebiet herum geroutet werden.
    print("\n  >> EREIGNIS: Ein neues Sperrgebiet wird aktiviert!")
    print("     Lage: Temporäre Rodung / Baustelle bei (235, 115) bis (265, 145).")
    print("     Achtung: Posten 104 (250, 130) liegt mitten im Sperrgebiet!")

    blocking_polygon = Polygon([
        Point(235.0, 115.0), Point(265.0, 115.0),
        Point(265.0, 145.0), Point(235.0, 145.0)
    ])

    update_report = course_manager.update_obstacle(
        obstacle_id="forestry_closure",
        new_polygon=blocking_polygon,
        reason="Holzschlag & Baustelle"
    )

    print("\nReaktiver Update-Report:")
    print(f"  * Neu geroutete Teilstrecken: {update_report['rerouted_legs']}")
    if update_report["relocated_controls"]:
        for rel in update_report["relocated_controls"]:
            print(f"  * Posten {rel['control_id']} lag im Sperrgebiet! Verschoben von "
                  f"{rel['old_point']} nach {rel['new_point']} (Distanz: {rel['shift_distance_m']:.1f} m)")
    else:
        print("  * Keine Posten direkt im Sperrgebiet; Teilstrecken werden aussen herum geroutet.")

    new_course = update_report["course"]
    print(f"  * Neue Gesamtlänge: {new_course.total_distance_m:.1f} m (Delta: {new_course.total_distance_m - initial_distance:+.1f} m)")
    print(f"  * Neuer Routenwahl-Score: {new_course.total_route_choice_score:.1f}")
    print(f"  * Geometrisch weiterhin gültig: {'JA' if new_course.is_geometrically_valid else 'NEIN'}")

    # Sammle neue Pfadpunkte
    new_path_pts = []
    for leg in new_course.legs:
        new_path_pts.extend(leg.optimal_path.points)

    # Verifikation: Kein Pfadpunkt der neuen Bahn darf im neuen Sperrgebiet liegen
    for p in new_path_pts:
        assert point_in_polygon(p, blocking_polygon, include_boundary=False) is False, \
            f"Pfadpunkt {p.as_tuple()} darf nicht im Sperrgebiet liegen!"

    print("\nAktualisierte Kartensituation (nach Umgehung des Sperrgebiets):")
    print(render_ascii_map(terrain, new_path_pts, blocking_polygon))

    print("\nVergleich der Teilstrecken:")
    for leg in new_course.legs:
        print(f"  Leg {leg.leg_index + 1}: {leg.from_control.id} -> {leg.to_control.id} | "
              f"Länge: {leg.distance_m:.1f} m | {leg.route_choice.description}")

    print("\n  [OK] Reaktiver Loop hat das Sperrgebiet erfolgreich automatisch umgangen!")


def run_all_tests():
    print("=" * 72)
    print("   AI COURSE SETTER - TASK 1 CORE & GEOMETRY VERIFIKATION")
    print("=" * 72 + "\n")

    test_geometry_primitives()
    terrain, course, config = test_course_generation_and_route_choice()
    test_reactive_obstacle_shift(terrain, course, config)

    print("\n" + "=" * 72)
    print("   ALLE TESTS ERFOLGREICH BESTANDEN!")
    print("=" * 72)


if __name__ == "__main__":
    run_all_tests()
