"""
terrain_model.py - Universelles Semantisches Geländemodell für die Orientierungslauf-Bahnlegung.

Normalisiert Kartendaten unabhängig vom Ausgangsformat (GeoJSON, OCAD, Raster) in:
- 0D: Postenkandidaten (Felsen, Kuppen, Senken, Wegabzweige, Hausecken etc. nach IOF)
- 1D: Linienobjekte (Pfade, Strassen, Bäche, Mauern, Zäune) mit Passierbarkeit und Geschwindigkeitsbonus
- 2D: Belaufbarkeitszonen (Wald, Grün 1/2/3, Offen, Siedlung) mit Reibungsfaktor
- 2.5D: Höhenmodell für Steigungs- und Höhenmeterberechnung
- Sperrgebiete: Prioritäts-Polygone mit Reibungsfaktor unendlich und reaktivem Verschieben
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union


# ==============================================================================
# 1. GEOMETRIE-PRIMITIVEN & HILFSFUNKTIONEN
# ==============================================================================

@dataclass(frozen=True)
class Point:
    """Repräsentiert einen Punkt im 2D/3D-Raum (Koordinaten in Metern)."""
    x: float
    y: float
    z: float = 0.0

    def distance_to(self, other: Point) -> float:
        """2D euklidische Distanz."""
        return math.hypot(self.x - other.x, self.y - other.y)

    def distance_3d_to(self, other: Point) -> float:
        """3D euklidische Distanz inklusive Höhendifferenz."""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2)

    def as_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def as_tuple_3d(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def __add__(self, other: Point) -> Point:
        return Point(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Point) -> Point:
        return Point(self.x - other.x, self.y - other.y, self.z - other.z)


@dataclass
class LineString:
    """Repräsentiert einen Linienzug aus aufeinanderfolgenden Punkten."""
    points: List[Point]

    @property
    def length(self) -> float:
        if len(self.points) < 2:
            return 0.0
        return sum(self.points[i].distance_to(self.points[i + 1]) for i in range(len(self.points) - 1))

    @property
    def segments(self) -> List[Tuple[Point, Point]]:
        return [(self.points[i], self.points[i + 1]) for i in range(len(self.points) - 1)]


@dataclass
class Polygon:
    """Repräsentiert ein Polygon mit Aussenring und optionalen Löchern."""
    exterior: List[Point]
    holes: List[List[Point]] = field(default_factory=list)

    def __post_init__(self):
        # Aussenring schliessen, falls noch nicht geschlossen
        if len(self.exterior) > 2:
            if not math.isclose(self.exterior[0].x, self.exterior[-1].x, abs_tol=1e-7) or \
               not math.isclose(self.exterior[0].y, self.exterior[-1].y, abs_tol=1e-7):
                self.exterior.append(self.exterior[0])

        closed_holes = []
        for h in self.holes:
            if len(h) > 2:
                h_copy = list(h)
                if not math.isclose(h_copy[0].x, h_copy[-1].x, abs_tol=1e-7) or \
                   not math.isclose(h_copy[0].y, h_copy[-1].y, abs_tol=1e-7):
                    h_copy.append(h_copy[0])
                closed_holes.append(h_copy)
        self.holes = closed_holes

    @property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        """(min_x, min_y, max_x, max_y)"""
        xs = [p.x for p in self.exterior]
        ys = [p.y for p in self.exterior]
        return (min(xs), min(ys), max(xs), max(ys))

    @property
    def centroid(self) -> Point:
        """Berechnet den Schwerpunkt des Aussenrings."""
        pts = self.exterior[:-1] if len(self.exterior) > 1 else self.exterior
        if not pts:
            return Point(0.0, 0.0)
        return Point(sum(p.x for p in pts) / len(pts), sum(p.y for p in pts) / len(pts))


def ccw(a: Point, b: Point, c: Point) -> float:
    """Gibt das Kreuzprodukt der Vektoren (b - a) und (c - a) zurück."""
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)


def point_on_segment(p: Point, a: Point, b: Point, tol: float = 1e-7) -> bool:
    """Prüft, ob Punkt p auf dem Segment [a, b] liegt."""
    if not math.isclose(ccw(a, b, p), 0.0, abs_tol=tol):
        return False
    min_x, max_x = min(a.x, b.x) - tol, max(a.x, b.x) + tol
    min_y, max_y = min(a.y, b.y) - tol, max(a.y, b.y) + tol
    return (min_x <= p.x <= max_x) and (min_y <= p.y <= max_y)


def segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point, touch_is_intersection: bool = True) -> bool:
    """
    Prüft, ob die Liniensegmente [p1, p2] und [p3, p4] einander schneiden.
    touch_is_intersection: Wenn False, gelten gemeinsame Endpunkte oder Randberührungen nicht als Schnitt.
    """
    # Identische Endpunkte
    common_endpoints = (
        (math.isclose(p1.x, p3.x, abs_tol=1e-7) and math.isclose(p1.y, p3.y, abs_tol=1e-7)) or
        (math.isclose(p1.x, p4.x, abs_tol=1e-7) and math.isclose(p1.y, p4.y, abs_tol=1e-7)) or
        (math.isclose(p2.x, p3.x, abs_tol=1e-7) and math.isclose(p2.y, p3.y, abs_tol=1e-7)) or
        (math.isclose(p2.x, p4.x, abs_tol=1e-7) and math.isclose(p2.y, p4.y, abs_tol=1e-7))
    )
    if common_endpoints:
        return touch_is_intersection

    # Bounding Box Vorfilter
    if max(p1.x, p2.x) < min(p3.x, p4.x) - 1e-7 or min(p1.x, p2.x) > max(p3.x, p4.x) + 1e-7 or \
       max(p1.y, p2.y) < min(p3.y, p4.y) - 1e-7 or min(p1.y, p2.y) > max(p3.y, p4.y) + 1e-7:
        return False

    d1 = ccw(p3, p4, p1)
    d2 = ccw(p3, p4, p2)
    d3 = ccw(p1, p2, p3)
    d4 = ccw(p1, p2, p4)

    # Allgemeiner Fall (echter Schnitt)
    if ((d1 > 1e-7 and d2 < -1e-7) or (d1 < -1e-7 and d2 > 1e-7)) and \
       ((d3 > 1e-7 and d4 < -1e-7) or (d3 < -1e-7 and d4 > 1e-7)):
        return True

    if not touch_is_intersection:
        return False

    # Kollineare Fälle / Punkt auf Segment
    if point_on_segment(p1, p3, p4) or point_on_segment(p2, p3, p4) or \
       point_on_segment(p3, p1, p2) or point_on_segment(p4, p1, p2):
        return True

    return False


def find_segment_intersection(p1: Point, p2: Point, p3: Point, p4: Point) -> Optional[Point]:
    """Berechnet den Schnittpunkt zweier Segmente [p1, p2] und [p3, p4], falls existent."""
    if not segments_intersect(p1, p2, p3, p4, touch_is_intersection=True):
        return None

    denom = (p1.x - p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x - p4.x)
    if math.isclose(denom, 0.0, abs_tol=1e-9):
        # Kollinear oder parallel: gebe den gemeinsamen Punkt oder Überlappungsmitte zurück
        pts = [p for p in (p1, p2, p3, p4) if point_on_segment(p, p1, p2) and point_on_segment(p, p3, p4)]
        return pts[0] if pts else None

    t = ((p1.x - p3.x) * (p3.y - p4.y) - (p1.y - p3.y) * (p3.x - p4.x)) / denom
    ix = p1.x + t * (p2.x - p1.x)
    iy = p1.y + t * (p2.y - p1.y)
    iz = p1.z + t * (p2.z - p1.z)
    return Point(ix, iy, iz)


def point_in_ring(p: Point, ring: List[Point], include_boundary: bool = True) -> bool:
    """Ray-Casting Algorithmus zur Prüfung, ob Punkt p innerhalb eines geschlossenen Rings liegt."""
    n = len(ring)
    if n < 3:
        return False

    # Prüfe zuerst, ob der Punkt direkt auf einer Kante liegt
    for i in range(n - 1):
        if point_on_segment(p, ring[i], ring[i + 1]):
            return include_boundary

    inside = False
    j = n - 2
    for i in range(n - 1):
        pi = ring[i]
        pj = ring[j]
        if ((pi.y > p.y) != (pj.y > p.y)):
            intersect_x = (pj.x - pi.x) * (p.y - pi.y) / (pj.y - pi.y + 1e-12) + pi.x
            if p.x < intersect_x:
                inside = not inside
        j = i
    return inside


def point_in_polygon(p: Point, polygon: Polygon, include_boundary: bool = True) -> bool:
    """Prüft, ob Punkt p im Polygon liegt (im Aussenring und nicht in den Löchern)."""
    # Bounding Box Vorabtest
    min_x, min_y, max_x, max_y = polygon.bounding_box
    tol = 1e-7
    if p.x < min_x - tol or p.x > max_x + tol or p.y < min_y - tol or p.y > max_y + tol:
        return False

    if not point_in_ring(p, polygon.exterior, include_boundary=include_boundary):
        return False

    for hole in polygon.holes:
        if point_in_ring(p, hole, include_boundary=not include_boundary):
            return False

    return True


def distance_point_to_segment(p: Point, a: Point, b: Point) -> Tuple[float, Point]:
    """
    Berechnet den kürzesten Abstand von Punkt p zum Segment [a, b]
    und liefert den nächstgelegenen projizierten Punkt auf dem Segment.
    """
    ab_x = b.x - a.x
    ab_y = b.y - a.y
    seg_len_sq = ab_x ** 2 + ab_y ** 2
    if math.isclose(seg_len_sq, 0.0, abs_tol=1e-9):
        return (p.distance_to(a), a)

    # Projektionsfaktor t
    t = ((p.x - a.x) * ab_x + (p.y - a.y) * ab_y) / seg_len_sq
    t = max(0.0, min(1.0, t))
    closest = Point(a.x + t * ab_x, a.y + t * ab_y, a.z + t * (b.z - a.z))
    return (p.distance_to(closest), closest)


def angle_between_legs(p_prev: Point, p_curr: Point, p_next: Point) -> float:
    """
    Berechnet den Wendewinkel am Posten p_curr zwischen Anlauf (p_prev -> p_curr)
    und Ablauf (p_curr -> p_next).

    Winkel alpha zwischen den Vektoren (p_prev - p_curr) und (p_next - p_curr):
    - 180 Grad: Geradeaus (perfekter Durchlauf)
    - 90 Grad: 90-Grad-Rechts-/Linkskurve
    - 0 Grad: 180-Grad-Kehre (Gegenverkehr! Läufer läuft genau denselben Weg zurück)

    Liefert den Winkel in Grad [0, 180].
    """
    v1_x = p_prev.x - p_curr.x
    v1_y = p_prev.y - p_curr.y
    v2_x = p_next.x - p_curr.x
    v2_y = p_next.y - p_curr.y

    len1 = math.hypot(v1_x, v1_y)
    len2 = math.hypot(v2_x, v2_y)

    if math.isclose(len1, 0.0, abs_tol=1e-9) or math.isclose(len2, 0.0, abs_tol=1e-9):
        return 0.0

    dot = (v1_x * v2_x + v1_y * v2_y) / (len1 * len2)
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def line_intersects_polygon(p1: Point, p2: Point, polygon: Polygon) -> bool:
    """Prüft, ob ein Segment [p1, p2] die Ränder des Polygons schneidet oder ganz darin liegt."""
    # 1. Liegt mindestens ein Punkt im Polygon?
    if point_in_polygon(p1, polygon) or point_in_polygon(p2, polygon):
        return True

    # 2. Prüfe Mittelpunkt (falls Segment quer durchgeht ohne Endpunkte darin)
    mid = Point((p1.x + p2.x) * 0.5, (p1.y + p2.y) * 0.5)
    if point_in_polygon(mid, polygon):
        return True

    # 3. Schneidet das Segment eine der Aussenring-Kanten?
    ext = polygon.exterior
    for i in range(len(ext) - 1):
        if segments_intersect(p1, p2, ext[i], ext[i + 1], touch_is_intersection=True):
            return True

    # 4. Löcher
    for hole in polygon.holes:
        for i in range(len(hole) - 1):
            if segments_intersect(p1, p2, hole[i], hole[i + 1], touch_is_intersection=True):
                return True

    return False


# ==============================================================================
# 2. SEMANTISCHE GELÄNDE-OBJEKTE (0D, 1D, 2D, 2.5D)
# ==============================================================================

class IOFCode:
    """IOF Postenmerkmale nach ISOM 2017-2 / Control Descriptions."""
    KNOLL = "101_knoll"             # Kuppe / Hügel
    SADDLE = "103_saddle"           # Sattel
    DEPRESSION = "104_depression"   # Senke
    PIT = "105_pit"                 # Grube
    BOULDER = "201_boulder"         # Fels / Stein
    CLIFF = "203_cliff"             # Felswand
    WATER_HOLE = "301_water_hole"   # Wasserloch
    STREAM_JUNC = "302_stream_junc" # Bachabzweig
    MARSH = "303_marsh"             # Sumpf
    VEG_BOUNDARY = "401_veg_bound"  # Waldgrenze / Vegetationsgrenze
    DIST_TREE = "402_dist_tree"     # Markanter Baum
    ROOT_STOCK = "403_root_stock"   # Wurzelstock
    BUILDING = "501_building"       # Gebäude / Hausecke
    PATH_JUNC = "502_path_junc"     # Wegverzweigung
    WALL_CORNER = "503_wall_corner" # Mauerecke
    FENCE_CORNER = "504_fence_corn" # Zaunecke
    START = "START"                 # Startplatz
    FINISH = "FINISH"               # Ziel


@dataclass
class ControlCandidate:
    """0D: Postenkandidat im Gelände."""
    id: str
    point: Point
    iof_code: str = IOFCode.BOULDER
    name: str = ""
    description: str = ""
    is_start: bool = False
    is_finish: bool = False
    fixed: bool = False             # Fixiert durch den Bahnleger (Co-Pilot)
    disabled: bool = False          # Deaktiviert (z.B. temporär gesperrt)
    quality_score: float = 1.0      # Eignungsbewertung des Merkmals (1.0 = Standard)


@dataclass
class LinearFeature:
    """
    1D: Linienobjekt (Pfade, Strassen, Bäche, Mauern, Zäune).
    speed_factor:
      < 1.0: Geschwindigkeitsbonus (z.B. Weg = 0.7 entspricht 30% Zeitersparnis)
      > 1.0: Bremsend (z.B. dichter Bachlauf)
    passable:
      False: Absolut unpassierbare Mauer / Sperrzaun (Friction = inf)
    """
    id: str
    line: LineString
    feature_type: str = "path"      # 'road', 'path', 'wall', 'fence', 'stream'
    passable: bool = True
    speed_factor: float = 0.7       # Standard-Wegbonus
    width_m: float = 2.0


@dataclass
class TerrainZone:
    """
    2D: Belaufbarkeitszone mit Reibungsfaktor.
    Standardwerte:
      Strasse/Wiese = 0.7 - 0.9
      Weisser Wald = 1.0 (Basis)
      Grün 1 = 1.6
      Grün 2 = 2.5
      Grün 3 = 8.0 - 10.0
      See / Privat = math.inf
    """
    id: str
    polygon: Polygon
    zone_type: str = "forest"
    friction: float = 1.0
    description: str = ""


@dataclass
class ForbiddenZone:
    """
    2D: Sperrgebiet (Naturschutz, Baustelle, Privatgelände).
    Besitzt immer Reibung = infinity und Vorrang vor allen Routen.
    """
    id: str
    polygon: Polygon
    reason: str = "out_of_bounds"
    friction: float = math.inf
    buffer_margin_m: float = 2.0


class ElevationModel:
    """
    2.5D: Digitales Höhenmodell (DHM / Konturlinien).
    Berechnet Höhenwert z(x, y), Gradienten und physiologischen Steigungsaufwand.
    Standard: Naismith / IOF Faustregel (100 m Steigung ~ 1000 m Flachdistanz).
    """
    def __init__(self, base_z: float = 0.0, grid: Optional[Dict[Tuple[int, int], float]] = None, cell_size: float = 20.0):
        self.base_z = base_z
        self.grid = grid or {}
        self.cell_size = cell_size

    def get_elevation(self, x: float, y: float) -> float:
        """Gibt die Höhe z für eine Koordinate zurück (Bilineare Interpolation falls Raster vorhanden)."""
        if not self.grid:
            return self.base_z
        gx = x / self.cell_size
        gy = y / self.cell_size
        ix = int(math.floor(gx))
        iy = int(math.floor(gy))
        fx = gx - ix
        fy = gy - iy

        z00 = self.grid.get((ix, iy), self.base_z)
        z10 = self.grid.get((ix + 1, iy), z00)
        z01 = self.grid.get((ix, iy + 1), z00)
        z11 = self.grid.get((ix + 1, iy + 1), z10)

        # Bilineare Interpolation
        z = (1 - fx) * (1 - fy) * z00 + fx * (1 - fy) * z10 + (1 - fx) * fy * z01 + fx * fy * z11
        return z

    def calculate_climb(self, p1: Point, p2: Point) -> float:
        """Berechnet die positive Steigung (Ascent in Metern) zwischen zwei Punkten."""
        z1 = p1.z if p1.z != 0.0 else self.get_elevation(p1.x, p1.y)
        z2 = p2.z if p2.z != 0.0 else self.get_elevation(p2.x, p2.y)
        return max(0.0, z2 - z1)

    def slope_friction_multiplier(self, p1: Point, p2: Point, climb_penalty_weight: float = 10.0) -> float:
        """
        Gibt einen Zeitfaktor zurück, der Steigungen bestraft.
        10 m Steigung entsprechen nach IOF ca. 100 m zusätzlicher Flachdistanz (Faktor 10.0).
        """
        dist_2d = p1.distance_to(p2)
        if math.isclose(dist_2d, 0.0, abs_tol=1e-7):
            return 1.0
        climb = self.calculate_climb(p1, p2)
        # Effektive äquivalente Flachstrecke = dist_2d + climb * climb_penalty_weight
        # Multiplikator = äquivalente Distanz / reale 2D-Distanz
        return (dist_2d + climb * climb_penalty_weight) / dist_2d


# ==============================================================================
# 3. GESAMTES SEMANTISCHES GELÄNDEMODELL (SEMANTIC TERRAIN MODEL)
# ==============================================================================

class SemanticTerrainModel:
    """
    Zentraler Geländespeicher für 0D, 1D, 2D und 2.5D Daten.
    Bietet Schnittstellen zur Kostenabfrage und reaktiven Sperrzonen-Verwaltung.
    """

    def __init__(self, elevation_model: Optional[ElevationModel] = None):
        self.candidates: Dict[str, ControlCandidate] = {}
        self.linear_features: Dict[str, LinearFeature] = {}
        self.zones: Dict[str, TerrainZone] = {}
        self.forbidden_zones: Dict[str, ForbiddenZone] = {}
        self.elevation_model = elevation_model or ElevationModel()

    # --- 0D Postenkandidaten ---
    def add_candidate(self, candidate: ControlCandidate) -> None:
        # Z-Koordinate aus Höhenmodell befüllen falls 0
        if candidate.point.z == 0.0:
            candidate.point = Point(
                candidate.point.x,
                candidate.point.y,
                self.elevation_model.get_elevation(candidate.point.x, candidate.point.y)
            )
        self.candidates[candidate.id] = candidate

    def get_candidate(self, cid: str) -> Optional[ControlCandidate]:
        return self.candidates.get(cid)

    def remove_candidate(self, cid: str) -> bool:
        return self.candidates.pop(cid, None) is not None

    # --- 1D Lineare Objekte ---
    def add_linear_feature(self, feature: LinearFeature) -> None:
        self.linear_features[feature.id] = feature

    # --- 2D Belaufbarkeitszonen ---
    def add_zone(self, zone: TerrainZone) -> None:
        self.zones[zone.id] = zone

    # --- Sperrgebiete (Obstacle-First) ---
    def add_forbidden_zone(self, fzone: ForbiddenZone) -> None:
        self.forbidden_zones[fzone.id] = fzone

    def update_forbidden_zone(self, zone_id: str, new_polygon: Polygon, reason: Optional[str] = None) -> ForbiddenZone:
        """Aktualisiert oder erstellt ein Sperrgebiet dynamisch."""
        if zone_id in self.forbidden_zones:
            fz = self.forbidden_zones[zone_id]
            fz.polygon = new_polygon
            if reason:
                fz.reason = reason
        else:
            fz = ForbiddenZone(id=zone_id, polygon=new_polygon, reason=reason or "updated_obstacle")
            self.forbidden_zones[zone_id] = fz
        return fz

    def remove_forbidden_zone(self, zone_id: str) -> bool:
        return self.forbidden_zones.pop(zone_id, None) is not None

    # --- Raumabfragen & Reaktivität ---
    def is_point_allowed(self, point: Point) -> bool:
        """Prüft, ob ein Punkt ausserhalb aller Sperrgebiete liegt."""
        for fz in self.forbidden_zones.values():
            if point_in_polygon(point, fz.polygon, include_boundary=True):
                return False
        return True

    def find_nearest_allowed_point(self, point: Point, max_search_radius: float = 100.0, step: float = 5.0) -> Point:
        """
        Falls ein Posten in ein Sperrgebiet geraten ist, projiziert diese Funktion
        den Punkt auf die nächstgelegene legale Koordinate knapp ausserhalb der Sperrzone.
        """
        if self.is_point_allowed(point):
            return point

        best_pt = point
        min_dist = math.inf

        # Projiziere auf die Kanten aller betroffenen Sperrzonen
        for fz in self.forbidden_zones.values():
            if not point_in_polygon(point, fz.polygon, include_boundary=True):
                continue

            ext = fz.polygon.exterior
            for i in range(len(ext) - 1):
                d, proj = distance_point_to_segment(point, ext[i], ext[i + 1])
                # Schiebe den Punkt um einen Sicherheitsabstand ausserhalb der Kante
                dx = proj.x - point.x
                dy = proj.y - point.y
                l = math.hypot(dx, dy)
                if l > 1e-6:
                    norm_x = dx / l
                    norm_y = dy / l
                else:
                    # Kanten-Normalenvektor berechnen
                    edge_dx = ext[i + 1].x - ext[i].x
                    edge_dy = ext[i + 1].y - ext[i].y
                    el = math.hypot(edge_dx, edge_dy)
                    norm_x = -edge_dy / el
                    norm_y = edge_dx / el

                buffer = max(fz.buffer_margin_m, 2.0)
                candidate_pt = Point(proj.x + norm_x * buffer, proj.y + norm_y * buffer)

                # Prüfe, ob candidate_pt tatsächlich ausserhalb liegt
                if self.is_point_allowed(candidate_pt):
                    c_dist = point.distance_to(candidate_pt)
                    if c_dist < min_dist:
                        min_dist = c_dist
                        best_pt = candidate_pt

        # Fallback: Ringsuche in 16 Richtungen
        if not self.is_point_allowed(best_pt):
            r = step
            while r <= max_search_radius:
                for angle_deg in range(0, 360, 22):
                    rad = math.radians(angle_deg)
                    test_pt = Point(point.x + r * math.cos(rad), point.y + r * math.sin(rad))
                    if self.is_point_allowed(test_pt):
                        return test_pt
                r += step

        return best_pt

    def is_segment_allowed(self, p1: Point, p2: Point) -> bool:
        """Prüft, ob ein Segment [p1, p2] frei von Sperrgebieten und unpassierbaren Mauern/Zäunen ist."""
        # 1. Sperrgebiete
        for fz in self.forbidden_zones.values():
            if line_intersects_polygon(p1, p2, fz.polygon):
                return False

        # 2. Unpassierbare Linienobjekte (Mauern, Zäune)
        for lf in self.linear_features.values():
            if not lf.passable:
                for seg_a, seg_b in lf.line.segments:
                    if segments_intersect(p1, p2, seg_a, seg_b, touch_is_intersection=True):
                        return False

        return True

    def get_friction_at_point(self, point: Point) -> float:
        """Ermittelt den Reibungsfaktor an einer Koordinate x, y."""
        if not self.is_point_allowed(point):
            return math.inf

        # Prüfe Zonen (die restriktivste oder spezifischste Reibung)
        current_friction = 1.0  # Standard = Weisser Wald (100%)
        for zone in self.zones.values():
            if point_in_polygon(point, zone.polygon, include_boundary=True):
                # Falls z.B. Grün 3 (8.0) oder Weg (0.7)
                current_friction = zone.friction
                break

        return current_friction

    def compute_segment_cost(self, p1: Point, p2: Point, samples: int = 5) -> Tuple[float, float, float]:
        """
        Berechnet die effektiven Bewegungskosten (Gewichtete Distanz), reale Distanz und Steigung.
        Rückgabe: (cost, distance_m, climb_m)
        Gibt cost = math.inf zurück, falls das Segment unpassierbar ist.
        """
        if not self.is_segment_allowed(p1, p2):
            return (math.inf, p1.distance_to(p2), 0.0)

        dist_2d = p1.distance_to(p2)
        if math.isclose(dist_2d, 0.0, abs_tol=1e-7):
            return (0.0, 0.0, 0.0)

        # 1. Linear-Feature Bonus prüfen (z.B. wenn das Segment auf einem Pfad/Strasse verläuft)
        best_speed_factor = 1.0
        for lf in self.linear_features.values():
            if lf.passable and lf.speed_factor < best_speed_factor:
                # Prüfe, ob p1 und p2 nahe an diesem linearen Feature liegen
                for seg_a, seg_b in lf.line.segments:
                    d1, _ = distance_point_to_segment(p1, seg_a, seg_b)
                    d2, _ = distance_point_to_segment(p2, seg_a, seg_b)
                    if d1 <= lf.width_m and d2 <= lf.width_m:
                        best_speed_factor = min(best_speed_factor, lf.speed_factor)

        # 2. Reibung über Terrainzonen abtasten
        sampled_frictions = []
        for i in range(samples):
            t = (i + 0.5) / samples
            sample_pt = Point(p1.x + t * (p2.x - p1.x), p1.y + t * (p2.y - p1.y))
            fric = self.get_friction_at_point(sample_pt)
            if math.isinf(fric):
                return (math.inf, dist_2d, 0.0)
            sampled_frictions.append(fric)

        avg_friction = sum(sampled_frictions) / len(sampled_frictions)
        effective_friction = avg_friction * best_speed_factor

        # 3. Steigungskosten
        climb = self.elevation_model.calculate_climb(p1, p2)
        # 1 Meter Steigung entspricht 10 Meter flacher Distanz
        climb_equivalent_m = climb * 10.0

        total_cost = (dist_2d * effective_friction) + climb_equivalent_m
        return (total_cost, dist_2d, climb)
