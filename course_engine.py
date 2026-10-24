"""
course_engine.py - Heuristische Postensetzer-Engine & Geometriekern für Orientierungslauf-Bahnen.

Features:
- Dynamischer Navigationsgraph auf Basis des semantischen Geländemodells
- Pfadfindung mit A* und Yen's K-Shortest Paths (vollständig native, hochperformante Implementierung)
- Routenwahl-Metrik: Erkennt gleichwertige Pfade (Delta < 8%) mit räumlicher Spreizung (Drift)
- Geometrische Validierung:
  * check_intersections(): Verhindert kreuzende Verbindungslinien
  * check_angle(): Filtert spitze Kehren / Dog-legs (< 60° Wendewinkel)
- Heuristischer Bahnengenerator (Constraint Search auf Soll-Länge, Höhenmeter & Routenwahl-Score)
- Reaktiver Loop: update_obstacle(polygon) passt den Graphen an, verschiebt betroffene Posten
  und routet Teilstrecken automatisch neu.
- Konfigurierbares CourseConfig-Settings-Dictionary.
"""

from __future__ import annotations
import math
import heapq
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set, Any

from terrain_model import (
    Point, LineString, Polygon, ControlCandidate, LinearFeature,
    TerrainZone, ForbiddenZone, ElevationModel, SemanticTerrainModel,
    segments_intersect, distance_point_to_segment, angle_between_legs,
    line_intersects_polygon, point_in_polygon
)


# ==============================================================================
# 1. KONFIGURATION & SETTINGS
# ==============================================================================

@dataclass
class CourseConfig:
    """Konfigurierbare Parameter der Bahnsetzer-Engine."""
    # Winkelprüfung
    min_angle_deg: float = 60.0               # Mindestwinkel am Posten (zur Vermeidung von Gegenverkehr)
    allow_hairpin: bool = False               # Erlaube spitze Kehren (< min_angle_deg)
    
    # Routenwahl-Analyse
    k_shortest_paths: int = 3                 # Anzahl der zu prüfenden K-kürzesten Pfade (Yen)
    route_choice_delta_threshold: float = 0.08# Maximale Zeit-/Kostendifferenz (8%) für echte Alternativen
    route_choice_drift_threshold: float = 25.0# Minimale räumliche Spreizung (Meter) zweier Pfade
    route_choice_bonus_weight: float = 150.0  # Bonus im Gesamt-Score für Teilstrecken mit Routenwahl
    
    # Strafen & Gewichte
    intersection_penalty: float = 5000.0      # Strafe bei sich kreuzenden Teilstrecken
    hairpin_penalty: float = 2000.0           # Strafe bei zu spitzem Wendewinkel
    length_weight: float = 1.0                # Gewichtung der Längenabweichung
    climb_weight: float = 10.0                # Gewichtung der Höhenmeterabweichung (1 Hm = 10m Distanz)
    
    # Graph-Auflösung
    grid_resolution_m: float = 25.0           # Rasterauflösung für den Navigationsgraphen
    diagonal_connections: bool = True         # 8-Nachbarschaft im Raster

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_angle_deg": self.min_angle_deg,
            "allow_hairpin": self.allow_hairpin,
            "k_shortest_paths": self.k_shortest_paths,
            "route_choice_delta_threshold": self.route_choice_delta_threshold,
            "route_choice_drift_threshold": self.route_choice_drift_threshold,
            "route_choice_bonus_weight": self.route_choice_bonus_weight,
            "intersection_penalty": self.intersection_penalty,
            "hairpin_penalty": self.hairpin_penalty,
            "length_weight": self.length_weight,
            "climb_weight": self.climb_weight,
            "grid_resolution_m": self.grid_resolution_m,
            "diagonal_connections": self.diagonal_connections,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CourseConfig:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ==============================================================================
# 2. NAVIGATIONSGRAPH & A* / YEN'S K-SHORTEST PATHS
# ==============================================================================

@dataclass
class PathResult:
    """Ergebnis einer Pfadsuche."""
    points: List[Point]
    cost: float
    distance_m: float
    climb_m: float
    effective_friction: float = 1.0


class NavigationGraph:
    """
    Räumlicher Navigationsgraph für das Gelände.
    Enthält Knoten (Rasterpunkte, Posten, Pfadknoten) und gewichtete Kanten.
    """

    def __init__(self, terrain: SemanticTerrainModel, resolution: float = 25.0):
        self.terrain = terrain
        self.resolution = resolution
        self.nodes: Dict[str, Point] = {}
        self.adj: Dict[str, List[Tuple[str, float, float, float]]] = {}  # u -> [(v, cost, dist, climb)]
        self.node_spatial_index: Dict[Tuple[int, int], List[str]] = {}
        self._build_graph()

    def _node_key(self, p: Point) -> str:
        return f"node_{round(p.x, 1)}_{round(p.y, 1)}"

    def _add_node(self, p: Point, key: Optional[str] = None) -> str:
        k = key or self._node_key(p)
        if k not in self.nodes:
            self.nodes[k] = p
            self.adj[k] = []
            # Rasterindex
            cell = (int(p.x // self.resolution), int(p.y // self.resolution))
            self.node_spatial_index.setdefault(cell, []).append(k)
        return k

    def _add_edge(self, u: str, v: str) -> None:
        p1 = self.nodes[u]
        p2 = self.nodes[v]
        cost, dist, climb = self.terrain.compute_segment_cost(p1, p2)
        if not math.isinf(cost):
            self.adj[u].append((v, cost, dist, climb))
            # Rückrichtung ebenfalls berechnen (Steigung ist richtungsabhängig!)
            cost_rev, dist_rev, climb_rev = self.terrain.compute_segment_cost(p2, p1)
            if not math.isinf(cost_rev):
                self.adj[v].append((u, cost_rev, dist_rev, climb_rev))

    def _build_graph(self) -> None:
        """Erzeugt das reguläre Grundraster und bindet Posten und Pfade ein."""
        self.nodes.clear()
        self.adj.clear()
        self.node_spatial_index.clear()

        # 1. Bounding Box des Geländes bestimmen
        all_pts: List[Point] = [c.point for c in self.terrain.candidates.values()]
        for lf in self.terrain.linear_features.values():
            all_pts.extend(lf.line.points)
        for z in self.terrain.zones.values():
            all_pts.extend(z.polygon.exterior)
        for fz in self.terrain.forbidden_zones.values():
            all_pts.extend(fz.polygon.exterior)

        if not all_pts:
            all_pts = [Point(0, 0), Point(500, 500)]

        min_x = min(p.x for p in all_pts) - 40.0
        max_x = max(p.x for p in all_pts) + 40.0
        min_y = min(p.y for p in all_pts) - 40.0
        max_y = max(p.y for p in all_pts) + 40.0

        # 2. Rasterknoten anlegen
        grid_nodes: Dict[Tuple[int, int], str] = {}
        step = self.resolution

        nx = int(math.ceil((max_x - min_x) / step)) + 1
        ny = int(math.ceil((max_y - min_y) / step)) + 1

        for ix in range(nx):
            x = min_x + ix * step
            for iy in range(ny):
                y = min_y + iy * step
                pt = Point(x, y, self.terrain.elevation_model.get_elevation(x, y))
                if self.terrain.is_point_allowed(pt):
                    k = self._add_node(pt)
                    grid_nodes[(ix, iy)] = k

        # 3. Kanten zwischen Nachbar-Rasterpunkten verbinden
        for (ix, iy), u in grid_nodes.items():
            neighbors = [(ix + 1, iy), (ix, iy + 1), (ix + 1, iy + 1), (ix + 1, iy - 1)]
            for n_cell in neighbors:
                if n_cell in grid_nodes:
                    v = grid_nodes[n_cell]
                    self._add_edge(u, v)

        # 4. Postenkandidaten und lineare Features integrieren
        for c in self.terrain.candidates.values():
            self.integrate_point(c.point, key=c.id)

        for lf in self.terrain.linear_features.values():
            prev_k = None
            for p in lf.line.points:
                k = self.integrate_point(p)
                if prev_k and prev_k != k:
                    # Direkte Kante entlang des Pfades/Weges mit Speed-Bonus
                    self._add_edge(prev_k, k)
                prev_k = k

        # 5. Ecken von Sperrgebieten als Umgehungsknoten hinzufügen
        for fz in self.terrain.forbidden_zones.values():
            for p in fz.polygon.exterior[:-1]:
                # Etwas nach aussen versetzt
                legal_p = self.terrain.find_nearest_allowed_point(p, max_search_radius=20.0, step=2.0)
                if self.terrain.is_point_allowed(legal_p):
                    self.integrate_point(legal_p)

    def integrate_point(self, point: Point, key: Optional[str] = None, max_connections: int = 6) -> str:
        """Fügt einen Punkt (z.B. Posten) in den Graphen ein und verknüpft ihn mit den nächsten legalen Knoten."""
        k = self._add_node(point, key=key)
        # Finde naheliegende Knoten
        cell_x = int(point.x // self.resolution)
        cell_y = int(point.y // self.resolution)

        nearby_keys: List[str] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nearby_keys.extend(self.node_spatial_index.get((cell_x + dx, cell_y + dy), []))

        # Sortiere nach Abstand
        sorted_keys = sorted(
            [other for other in nearby_keys if other != k],
            key=lambda ok: point.distance_to(self.nodes[ok])
        )

        connected = 0
        for ok in sorted_keys:
            if connected >= max_connections:
                break
            # Prüfe, ob Kante legal ist
            p_other = self.nodes[ok]
            if self.terrain.is_segment_allowed(point, p_other):
                self._add_edge(k, ok)
                connected += 1

        return k

    def rebuild(self) -> None:
        """Baut den Graphen komplett neu auf (z.B. nach Sperrzonen-Änderungen)."""
        self._build_graph()


def a_star_search(
    graph: NavigationGraph,
    start_key: str,
    goal_key: str,
    excluded_edges: Optional[Set[Tuple[str, str]]] = None,
    excluded_nodes: Optional[Set[str]] = None
) -> Optional[PathResult]:
    """
    A*-Algorithmus für den Navigationsgraphen.
    Ermittelt den kostengünstigsten Pfad unter Berücksichtigung von Reibung und Steigung.
    """
    if start_key not in graph.nodes or goal_key not in graph.nodes:
        return None
    if excluded_nodes and (start_key in excluded_nodes or goal_key in excluded_nodes):
        return None

    goal_pt = graph.nodes[goal_key]
    excluded_edges = excluded_edges or set()
    excluded_nodes = excluded_nodes or set()

    # Heuristik: Euklidische Distanz * minimaler denkbarer Reibungsfaktor (0.6)
    def h(k: str) -> float:
        return graph.nodes[k].distance_to(goal_pt) * 0.6

    open_set: List[Tuple[float, float, str]] = []  # (f_score, g_score, node_key)
    heapq.heappush(open_set, (h(start_key), 0.0, start_key))

    came_from: Dict[str, str] = {}
    g_score: Dict[str, float] = {start_key: 0.0}
    dist_accum: Dict[str, float] = {start_key: 0.0}
    climb_accum: Dict[str, float] = {start_key: 0.0}
    closed_set: Set[str] = set()

    while open_set:
        _, current_g, current = heapq.heappop(open_set)

        if current == goal_key:
            # Pfad rekonstruieren
            path_keys = [current]
            curr = current
            while curr in came_from:
                curr = came_from[curr]
                path_keys.append(curr)
            path_keys.reverse()

            pts = [graph.nodes[k] for k in path_keys]
            total_dist = dist_accum[goal_key]
            total_climb = climb_accum[goal_key]
            effective_fric = current_g / (total_dist + 1e-9)

            return PathResult(
                points=pts,
                cost=current_g,
                distance_m=total_dist,
                climb_m=total_climb,
                effective_friction=effective_fric
            )

        if current in closed_set:
            continue
        closed_set.add(current)

        for neighbor, edge_cost, edge_dist, edge_climb in graph.adj.get(current, []):
            if neighbor in excluded_nodes or neighbor in closed_set:
                continue
            if (current, neighbor) in excluded_edges or (neighbor, current) in excluded_edges:
                continue

            tentative_g = current_g + edge_cost
            if tentative_g < g_score.get(neighbor, math.inf):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                dist_accum[neighbor] = dist_accum[current] + edge_dist
                climb_accum[neighbor] = climb_accum[current] + edge_climb
                f = tentative_g + h(neighbor)
                heapq.heappush(open_set, (f, tentative_g, neighbor))

    return None


def yens_k_shortest_paths(
    graph: NavigationGraph,
    start_key: str,
    goal_key: str,
    K: int = 3
) -> List[PathResult]:
    """
    Yen's Algorithmus zur Berechnung der K-kürzesten schleifenfreien Pfade.
    Garantiert deterministische und optimale Alternativrouten.
    """
    first_path = a_star_search(graph, start_key, goal_key)
    if not first_path:
        return []

    # A speichert die gefundenen K-kürzesten Pfade
    A: List[PathResult] = [first_path]
    # B speichert Pfadkandidaten (Prioritätswarteschlange)
    B: List[Tuple[float, int, PathResult]] = []
    seen_paths: Set[str] = {_path_signature(first_path.points)}
    counter = 0

    for k in range(1, K):
        prev_path = A[k - 1].points
        # Wandle Pfadpunkte in Node-Keys um
        prev_keys = [_find_matching_key(graph, p) for p in prev_path]
        if any(key is None for key in prev_keys):
            break

        # Der Spur-Knoten wandert von 0 bis zur vorletzten Position
        for i in range(len(prev_keys) - 1):
            spur_node_key = prev_keys[i]
            root_path_keys = prev_keys[:i + 1]
            root_path_pts = prev_path[:i + 1]

            # 1. Kanten ausschliessen, die von vorherigen Pfaden mit identischem Root-Pfad genutzt wurden
            excluded_edges: Set[Tuple[str, str]] = set()
            for p in A:
                p_keys = [_find_matching_key(graph, pt) for pt in p.points]
                if len(p_keys) > i and p_keys[:i + 1] == root_path_keys:
                    excluded_edges.add((p_keys[i], p_keys[i + 1]))

            # 2. Knoten des Root-Pfads (ausser dem Spur-Knoten selbst) ausschliessen, um Schleifen zu verhindern
            excluded_nodes = set(root_path_keys[:-1])

            # 3. Spur-Pfad von Spur-Knoten zum Ziel berechnen
            spur_result = a_star_search(
                graph, spur_node_key, goal_key,
                excluded_edges=excluded_edges,
                excluded_nodes=excluded_nodes
            )

            if spur_result:
                # Gesamten Pfad zusammensetzen
                total_pts = root_path_pts[:-1] + spur_result.points
                sig = _path_signature(total_pts)
                if sig not in seen_paths:
                    seen_paths.add(sig)

                    # Kosten für den Root-Pfad berechnen
                    root_cost, root_dist, root_climb = _compute_path_metrics(graph.terrain, root_path_pts)
                    total_cost = root_cost + spur_result.cost
                    total_dist = root_dist + spur_result.distance_m
                    total_climb = root_climb + spur_result.climb_m

                    candidate = PathResult(
                        points=total_pts,
                        cost=total_cost,
                        distance_m=total_dist,
                        climb_m=total_climb,
                        effective_friction=total_cost / (total_dist + 1e-9)
                    )
                    counter += 1
                    heapq.heappush(B, (total_cost, counter, candidate))

        if not B:
            break

        _, _, best_candidate = heapq.heappop(B)
        A.append(best_candidate)

    return A


def _find_matching_key(graph: NavigationGraph, pt: Point) -> Optional[str]:
    """Findet den entsprechenden Knotenschlüssel für einen Punkt."""
    for k, p in graph.nodes.items():
        if math.isclose(p.x, pt.x, abs_tol=1e-4) and math.isclose(p.y, pt.y, abs_tol=1e-4):
            return k
    return None


def _path_signature(pts: List[Point]) -> str:
    """Erzeugt eine kompakte Signatur für die Pfad-Identifikation."""
    return "-".join(f"{round(p.x, 1)},{round(p.y, 1)}" for p in pts)


def _compute_path_metrics(terrain: SemanticTerrainModel, pts: List[Point]) -> Tuple[float, float, float]:
    """Berechnet Gesamtkosten, Distanz und Steigung für eine Punktfolge."""
    c, d, cl = 0.0, 0.0, 0.0
    for i in range(len(pts) - 1):
        seg_c, seg_d, seg_cl = terrain.compute_segment_cost(pts[i], pts[i + 1])
        c += seg_c
        d += seg_d
        cl += seg_cl
    return (c, d, cl)


# ==============================================================================
# 3. ROUTENWAHL-METRIK & GEOMETRISCHE VALIDIERUNG
# ==============================================================================

@dataclass
class RouteChoiceAnalysis:
    """Detaillierte Auswertung der Routenwahl auf einer Teilstrecke."""
    has_route_choice: bool
    score: float
    best_cost: float
    alt_cost: float
    cost_delta_pct: float
    max_drift_m: float
    paths: List[PathResult]
    description: str = ""


def calculate_spatial_drift(path1: List[Point], path2: List[Point]) -> float:
    """
    Berechnet die maximale räumliche Spreizung (Drift in Metern) zwischen zwei Pfaden.
    Ermittelt für jeden Punkt von path2 den kürzesten Abstand zur Polygonzug-Linie von path1
    (und umgekehrt).
    """
    def max_dist_to_line(source_pts: List[Point], target_pts: List[Point]) -> float:
        max_d = 0.0
        for p in source_pts:
            min_dist_to_target = math.inf
            for i in range(len(target_pts) - 1):
                d, _ = distance_point_to_segment(p, target_pts[i], target_pts[i + 1])
                if d < min_dist_to_target:
                    min_dist_to_target = d
            if min_dist_to_target > max_d and not math.isinf(min_dist_to_target):
                max_d = min_dist_to_target
        return max_d

    d1 = max_dist_to_line(path1, path2)
    d2 = max_dist_to_line(path2, path1)
    return max(d1, d2)


def analyze_leg_route_choices(
    graph: NavigationGraph,
    start_key: str,
    goal_key: str,
    config: Optional[CourseConfig] = None
) -> RouteChoiceAnalysis:
    """
    Analysiert, ob zwischen zwei Posten eine echte Routenwahlentscheidung vorliegt.
    Bedingungen für echte Routenwahl:
    1. Mindestens 2 alternative Pfade
    2. Zeit-/Kostendifferenz Delta <= config.route_choice_delta_threshold (z.B. 8%)
    3. Räumliche Spreizung (Drift) >= config.route_choice_drift_threshold (z.B. 25 m)
    """
    cfg = config or CourseConfig()
    paths = yens_k_shortest_paths(graph, start_key, goal_key, K=cfg.k_shortest_paths)

    if len(paths) < 2:
        cost = paths[0].cost if paths else 0.0
        return RouteChoiceAnalysis(
            has_route_choice=False,
            score=0.0,
            best_cost=cost,
            alt_cost=cost,
            cost_delta_pct=0.0,
            max_drift_m=0.0,
            paths=paths,
            description="Nur ein dominanter Pfad vorhanden."
        )

    best_p = paths[0]
    alt_p = paths[1]

    delta = (alt_p.cost - best_p.cost) / (best_p.cost + 1e-9)
    drift = calculate_spatial_drift(best_p.points, alt_p.points)

    has_rc = (delta <= cfg.route_choice_delta_threshold) and (drift >= cfg.route_choice_drift_threshold)

    score = 0.0
    if has_rc:
        # Je kleiner die Zeitdifferenz (max. 8%) und je deutlicher der Drift (bis zu 60m), desto höher der Score
        closeness_factor = max(0.0, 1.0 - (delta / cfg.route_choice_delta_threshold))
        drift_factor = min(1.0, drift / 50.0)
        score = cfg.route_choice_bonus_weight * (0.6 * closeness_factor + 0.4 * drift_factor)
        desc = f"Echte Routenwahl! Delta: {delta * 100:.1f}%, Drift: {drift:.1f} m"
    else:
        if delta > cfg.route_choice_delta_threshold:
            desc = f"Keine Routenwahl: Alternative um {delta * 100:.1f}% zu langsam."
        else:
            desc = f"Keine Routenwahl: Pfade laufen fast parallel (Drift: {drift:.1f} m < {cfg.route_choice_drift_threshold} m)."

    return RouteChoiceAnalysis(
        has_route_choice=has_rc,
        score=score,
        best_cost=best_p.cost,
        alt_cost=alt_p.cost,
        cost_delta_pct=delta * 100,
        max_drift_m=drift,
        paths=paths,
        description=desc
    )


def check_intersections(controls: List[ControlCandidate]) -> List[Tuple[int, int]]:
    """
    Prüft, ob sich die geraden Verbindungslinien zwischen aufeinanderfolgenden Posten kreuzen.
    Liefert eine Liste von Tupeln (i, j) mit den Indizes der kollidierenden Teilstrecken.
    (Teilstrecke i verbindet controls[i] -> controls[i+1]).
    Benachbarte Teilstrecken (die sich am gemeinsamen Posten berühren) sind legal.
    """
    conflicts: List[Tuple[int, int]] = []
    n = len(controls) - 1
    for i in range(n):
        p1 = controls[i].point
        p2 = controls[i + 1].point
        for j in range(i + 2, n):
            # i und j sind nicht benachbart
            # Ausnahme: Wenn Start und Ziel identisch sind, dürfen erste und letzte Teilstrecke
            # sich nicht am Start schneiden
            if i == 0 and j == n - 1 and controls[0].id == controls[-1].id:
                continue

            p3 = controls[j].point
            p4 = controls[j + 1].point
            if segments_intersect(p1, p2, p3, p4, touch_is_intersection=False):
                conflicts.append((i, j))
    return conflicts


def check_angle(
    p_prev: Point,
    p_curr: Point,
    p_next: Point,
    min_angle_deg: float = 60.0,
    allow_hairpin: bool = False
) -> Tuple[bool, float]:
    """
    Prüft den Wendewinkel an einem Posten (Dog-Leg / Gegenverkehr).
    alpha >= min_angle_deg (z.B. 60°): Gültig
    alpha < min_angle_deg: Ungültig (spitze Kehre / Gegenverkehr)
    """
    alpha = angle_between_legs(p_prev, p_curr, p_next)
    is_valid = allow_hairpin or (alpha >= min_angle_deg)
    return (is_valid, alpha)


# ==============================================================================
# 4. BAHNEN-DATENSTRUKTUREN & HEURISTISCHER GENERATOR
# ==============================================================================

@dataclass
class CourseLeg:
    """Eine Teilstrecke (Leg) zwischen zwei Posten."""
    leg_index: int
    from_control: ControlCandidate
    to_control: ControlCandidate
    optimal_path: PathResult
    route_choice: RouteChoiceAnalysis
    distance_m: float
    climb_m: float
    cost: float

    @property
    def has_route_choice(self) -> bool:
        return self.route_choice.has_route_choice


@dataclass
class Course:
    """Repräsentiert eine vollständige OL-Bahn von Start bis Ziel."""
    controls: List[ControlCandidate]
    legs: List[CourseLeg]
    total_distance_m: float
    total_climb_m: float
    total_route_choice_score: float
    is_geometrically_valid: bool
    intersection_conflicts: List[Tuple[int, int]] = field(default_factory=list)
    angle_warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        s = [
            f"=== BAHN-ZUSAMMENFASSUNG ===",
            f"Posten: {len(self.controls)} (Start -> {' -> '.join(c.id for c in self.controls[1:-1])} -> Ziel)",
            f"Gesamtlänge: {self.total_distance_m:.1f} m | Höhenmeter: {self.total_climb_m:.1f} Hm",
            f"Routenwahl-Score: {self.total_route_choice_score:.1f}",
            f"Geometrisch gültig: {'JA' if self.is_geometrically_valid else 'NEIN (Konflikte vorhanden)'}",
        ]
        if self.intersection_conflicts:
            s.append(f"  * Schnittfehler bei Teilstrecken: {self.intersection_conflicts}")
        if self.angle_warnings:
            for w in self.angle_warnings:
                s.append(f"  * {w}")
        s.append("Teilstrecken:")
        for leg in self.legs:
            rc_sym = "[RC]" if leg.has_route_choice else "    "
            s.append(
                f"  {rc_sym} Leg {leg.leg_index + 1}: {leg.from_control.id} -> {leg.to_control.id} | "
                f"{leg.distance_m:.1f} m, +{leg.climb_m:.1f} Hm | {leg.route_choice.description}"
            )
        return "\n".join(s)


class HeuristicCourseGenerator:
    """
    Deterministischer Heuristischer Bahnengenerator (Constraint Search).
    Optimiert Postenfolgen auf:
    - Streckenlängen-Budget (km)
    - Höhenmeter-Budget (Hm)
    - Maximierung echter Routenwahlen (Yen's K-Shortest Paths Score)
    - Strikte Vermeidung von Schnittlinien und spitzen Kehren (< 60°)
    """

    def __init__(self, terrain: SemanticTerrainModel, config: Optional[CourseConfig] = None):
        self.terrain = terrain
        self.config = config or CourseConfig()
        self.nav_graph = NavigationGraph(self.terrain, resolution=self.config.grid_resolution_m)

    def generate_course(
        self,
        start_id: str,
        finish_id: str,
        target_distance_m: float,
        target_climb_m: float = 0.0,
        num_controls: int = 3,
        candidate_ids: Optional[List[str]] = None
    ) -> Optional[Course]:
        """
        Sucht die optimale Bahn mit genau num_controls Zwischenposten.
        Filtert ungültige Geometrien frühzeitig per Branch-and-Bound / Constraint Pruning aus.
        """
        start_c = self.terrain.get_candidate(start_id)
        finish_c = self.terrain.get_candidate(finish_id)
        if not start_c or not finish_c:
            return None

        # Verfügbare Kandidaten filtern
        available_cids = candidate_ids or [
            cid for cid, c in self.terrain.candidates.items()
            if cid not in (start_id, finish_id) and not c.disabled and self.terrain.is_point_allowed(c.point)
        ]

        best_course: Optional[Course] = None
        best_fitness = -math.inf

        # Rekursive Suche nach Postenkombinationen
        def search(current_chain: List[ControlCandidate], remaining_cids: List[str]):
            nonlocal best_course, best_fitness

            # Wenn alle Zwischenposten gesetzt sind, Ziel anhängen und evaluieren
            if len(current_chain) == num_controls + 1:
                final_chain = current_chain + [finish_c]

                # 1. Geometrie-Prüfung: Schnitte
                conflicts = check_intersections(final_chain)
                if conflicts:
                    return

                # 2. Winkelprüfung am letzten Zwischenposten zum Ziel
                is_valid, alpha = check_angle(
                    final_chain[-3].point, final_chain[-2].point, final_chain[-1].point,
                    min_angle_deg=self.config.min_angle_deg,
                    allow_hairpin=self.config.allow_hairpin
                )
                if not is_valid:
                    return

                # 3. Vollständige Bahn evaluieren
                course = self._evaluate_course(final_chain)
                fitness = self._calculate_fitness(course, target_distance_m, target_climb_m)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_course = course
                return

            # Nächsten Posten ausprobieren
            for idx, cid in enumerate(remaining_cids):
                cand = self.terrain.get_candidate(cid)
                if not cand or not self.terrain.is_point_allowed(cand.point):
                    continue

                # Constraint 1: Schnittprüfung mit bestehenden Teilstrecken
                test_chain = current_chain + [cand]
                if len(test_chain) >= 4:
                    if check_intersections(test_chain):
                        continue

                # Constraint 2: Winkelprüfung am vorletzten Posten
                if len(test_chain) >= 3:
                    is_valid, alpha = check_angle(
                        test_chain[-3].point, test_chain[-2].point, test_chain[-1].point,
                        min_angle_deg=self.config.min_angle_deg,
                        allow_hairpin=self.config.allow_hairpin
                    )
                    if not is_valid:
                        continue

                # Constraint 3: Längenbudget-Pruning (Untere Schranke via Luftlinie)
                straight_dist = sum(
                    test_chain[i].point.distance_to(test_chain[i + 1].point)
                    for i in range(len(test_chain) - 1)
                ) + cand.point.distance_to(finish_c.point)
                # Falls Luftlinie schon das 1.5-fache des Budgets sprengt, abbrechen
                if straight_dist > target_distance_m * 1.6:
                    continue

                search(test_chain, remaining_cids[:idx] + remaining_cids[idx + 1:])

        search([start_c], list(available_cids))
        return best_course

    def _evaluate_course(self, controls: List[ControlCandidate]) -> Course:
        """Routet alle Teilstrecken einer Postenfolge und berechnet Routenwahlen und Kennzahlen."""
        legs: List[CourseLeg] = []
        total_dist = 0.0
        total_climb = 0.0
        total_rc_score = 0.0

        for i in range(len(controls) - 1):
            c1 = controls[i]
            c2 = controls[i + 1]

            rc_analysis = analyze_leg_route_choices(self.nav_graph, c1.id, c2.id, self.config)
            opt_path = rc_analysis.paths[0] if rc_analysis.paths else PathResult(
                points=[c1.point, c2.point],
                cost=c1.point.distance_to(c2.point),
                distance_m=c1.point.distance_to(c2.point),
                climb_m=0.0
            )

            leg = CourseLeg(
                leg_index=i,
                from_control=c1,
                to_control=c2,
                optimal_path=opt_path,
                route_choice=rc_analysis,
                distance_m=opt_path.distance_m,
                climb_m=opt_path.climb_m,
                cost=opt_path.cost
            )
            legs.append(leg)
            total_dist += leg.distance_m
            total_climb += leg.climb_m
            total_rc_score += rc_analysis.score

        # Winkelwarnungen sammeln
        angle_warnings: List[str] = []
        for i in range(1, len(controls) - 1):
            is_valid, alpha = check_angle(
                controls[i - 1].point, controls[i].point, controls[i + 1].point,
                min_angle_deg=self.config.min_angle_deg,
                allow_hairpin=self.config.allow_hairpin
            )
            if not is_valid:
                angle_warnings.append(
                    f"Spitze Kehre an Posten {controls[i].id}: Wendewinkel nur {alpha:.1f}° (< {self.config.min_angle_deg}°)"
                )

        conflicts = check_intersections(controls)
        is_geom_valid = (len(conflicts) == 0) and (len(angle_warnings) == 0)

        return Course(
            controls=controls,
            legs=legs,
            total_distance_m=total_dist,
            total_climb_m=total_climb,
            total_route_choice_score=total_rc_score,
            is_geometrically_valid=is_geom_valid,
            intersection_conflicts=conflicts,
            angle_warnings=angle_warnings
        )

    def _calculate_fitness(self, course: Course, target_distance_m: float, target_climb_m: float) -> float:
        """Berechnet den Optimierungs-Score einer Bahn."""
        dist_diff = abs(course.total_distance_m - target_distance_m)
        climb_diff = abs(course.total_climb_m - target_climb_m) if target_climb_m > 0 else 0.0

        fitness = (
            - (dist_diff * self.config.length_weight)
            - (climb_diff * self.config.climb_weight)
            + course.total_route_choice_score
        )

        # Qualitätsbonus der Posten
        cand_bonus = sum(c.quality_score * 20.0 for c in course.controls[1:-1])
        fitness += cand_bonus

        # Strafen für Geometriefehler
        if course.intersection_conflicts:
            fitness -= len(course.intersection_conflicts) * self.config.intersection_penalty
        if course.angle_warnings:
            fitness -= len(course.angle_warnings) * self.config.hairpin_penalty

        return fitness


# ==============================================================================
# 5. REAKTIVER LOOP: UPDATE OBSTACLE & POSTEN-VERSCHIEBUNG
# ==============================================================================

class CourseManager:
    """
    Verwaltet eine aktive Bahn und reagiert dynamisch auf Änderungen an Sperrgebieten
    oder manuelles Verschieben von Posten durch den Bahnleger.
    """

    def __init__(self, terrain: SemanticTerrainModel, active_course: Course, config: Optional[CourseConfig] = None):
        self.terrain = terrain
        self.course = active_course
        self.config = config or CourseConfig()
        self.nav_graph = NavigationGraph(self.terrain, resolution=self.config.grid_resolution_m)

    def update_obstacle(
        self,
        obstacle_id: str,
        new_polygon: Polygon,
        reason: str = "temporary_forbidden"
    ) -> Dict[str, Any]:
        """
        Reaktiver Loop:
        1. Sperrgebiet im Geländemodell anlegen/aktualisieren.
        2. Prüfen, ob Posten der aktuellen Bahn in das Sperrgebiet fallen -> Verschieben auf nächsten legalen Punkt.
        3. Navigationsgraphen aktualisieren.
        4. Betroffene Teilstrecken automatisch neu routen.
        5. Aktualisierte Bahn und Report zurückgeben.
        """
        # 1. Sperrgebiet aktualisieren
        fz = self.terrain.update_forbidden_zone(obstacle_id, new_polygon, reason=reason)

        relocated_controls: List[Dict[str, Any]] = []
        # 2. Posten prüfen
        for idx, ctrl in enumerate(self.course.controls):
            if point_in_polygon(ctrl.point, new_polygon, include_boundary=True):
                old_pt = ctrl.point
                new_pt = self.terrain.find_nearest_allowed_point(ctrl.point, max_search_radius=100.0, step=3.0)
                ctrl.point = new_pt
                if ctrl.id in self.terrain.candidates:
                    self.terrain.candidates[ctrl.id].point = new_pt
                relocated_controls.append({
                    "control_id": ctrl.id,
                    "old_point": old_pt.as_tuple(),
                    "new_point": new_pt.as_tuple(),
                    "shift_distance_m": old_pt.distance_to(new_pt)
                })

        # 3. Navigationsgraph neu synchronisieren
        self.nav_graph.rebuild()

        # 4. Teilstrecken neu routen
        rerouted_legs: List[int] = []
        updated_legs: List[CourseLeg] = []
        total_dist = 0.0
        total_climb = 0.0
        total_rc_score = 0.0

        for i in range(len(self.course.controls) - 1):
            c1 = self.course.controls[i]
            c2 = self.course.controls[i + 1]

            # Prüfe, ob alter Pfad der Teilstrecke mit neuem Sperrgebiet kollidiert
            old_leg = self.course.legs[i] if i < len(self.course.legs) else None
            needs_rerouting = True

            rc_analysis = analyze_leg_route_choices(self.nav_graph, c1.id, c2.id, self.config)
            opt_path = rc_analysis.paths[0] if rc_analysis.paths else PathResult(
                points=[c1.point, c2.point],
                cost=c1.point.distance_to(c2.point),
                distance_m=c1.point.distance_to(c2.point),
                climb_m=0.0
            )

            new_leg = CourseLeg(
                leg_index=i,
                from_control=c1,
                to_control=c2,
                optimal_path=opt_path,
                route_choice=rc_analysis,
                distance_m=opt_path.distance_m,
                climb_m=opt_path.climb_m,
                cost=opt_path.cost
            )
            updated_legs.append(new_leg)
            rerouted_legs.append(i + 1)
            total_dist += new_leg.distance_m
            total_climb += new_leg.climb_m
            total_rc_score += rc_analysis.score

        # Bahn aktualisieren
        conflicts = check_intersections(self.course.controls)
        angle_warnings = []
        for i in range(1, len(self.course.controls) - 1):
            is_valid, alpha = check_angle(
                self.course.controls[i - 1].point, self.course.controls[i].point, self.course.controls[i + 1].point,
                min_angle_deg=self.config.min_angle_deg,
                allow_hairpin=self.config.allow_hairpin
            )
            if not is_valid:
                angle_warnings.append(
                    f"Spitze Kehre an Posten {self.course.controls[i].id} ({alpha:.1f}° < {self.config.min_angle_deg}°)"
                )

        self.course.legs = updated_legs
        self.course.total_distance_m = total_dist
        self.course.total_climb_m = total_climb
        self.course.total_route_choice_score = total_rc_score
        self.course.intersection_conflicts = conflicts
        self.course.angle_warnings = angle_warnings
        self.course.is_geometrically_valid = (len(conflicts) == 0 and len(angle_warnings) == 0)

        return {
            "obstacle_id": obstacle_id,
            "relocated_controls": relocated_controls,
            "rerouted_legs": rerouted_legs,
            "new_total_distance_m": total_dist,
            "new_total_climb_m": total_climb,
            "new_route_choice_score": total_rc_score,
            "is_geometrically_valid": self.course.is_geometrically_valid,
            "course": self.course
        }
