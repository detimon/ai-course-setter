"""
app.py - FastAPI Backend für den interaktiven AI Course Setter.

Stellt REST-Endpunkte bereit für:
- GET  /api/terrain: Liefert das semantische Geländemodell als GeoJSON
- POST /api/generate: Generiert eine optimierte Bahn mit Varianz & K-Routenwahlen
- POST /api/obstacle: Fügt ein Sperrgebiet-Polygon hinzu (reaktiver Update-Loop)
- POST /api/move-control: Manuelles Verschieben von Posten (Echtzeit-Neuberechnung)
- POST /api/reset-obstacles: Setzt Sperrgebiete auf Ausgangszustand zurück
- Liefert das interaktive Web-Frontend aus 'frontend/' aus.
"""

import os
import sys
import math
from typing import List, Dict, Any, Optional

# site-packages einbinden falls lokal installiert
site_packages = os.path.join(os.path.dirname(__file__), "site-packages")
if os.path.exists(site_packages) and site_packages not in sys.path:
    sys.path.insert(0, site_packages)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from terrain_model import (
    Point, LineString, Polygon, ControlCandidate, LinearFeature,
    TerrainZone, ForbiddenZone, SemanticTerrainModel
)
from course_engine import (
    CourseConfig, NavigationGraph, Course, CourseLeg, HeuristicCourseGenerator,
    CourseManager, analyze_leg_route_choices, calculate_spatial_drift
)
from test_core import create_synthetic_terrain


# ==============================================================================
# APP INITIALISIERUNG & GLOBALER STATE
# ==============================================================================

app = FastAPI(title="AI Course Setter API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AppState:
    def __init__(self):
        self.terrain: SemanticTerrainModel = create_synthetic_terrain()
        self.config: CourseConfig = CourseConfig(
            num_routes=3,
            min_drift_m=20.0,
            leg_variance=0.5,
            min_angle_deg=60.0,
            allow_hairpin=False
        )
        self.generator: HeuristicCourseGenerator = HeuristicCourseGenerator(self.terrain, self.config)
        self.active_course: Optional[Course] = None
        self.course_manager: Optional[CourseManager] = None

    def reset_terrain(self):
        self.terrain = create_synthetic_terrain()
        self.generator = HeuristicCourseGenerator(self.terrain, self.config)
        self.active_course = None
        self.course_manager = None

state = AppState()


# ==============================================================================
# PYDANTIC REQUEST MODELLE
# ==============================================================================

class GenerateRequest(BaseModel):
    distance_m: float = Field(default=1050.0, ge=400.0, le=3000.0)
    elevation_m: float = Field(default=20.0, ge=0.0, le=100.0)
    num_controls: int = Field(default=4, ge=2, le=6)
    num_routes: int = Field(default=3, ge=1, le=5)
    leg_variance: float = Field(default=0.5, ge=0.0, le=1.0)
    min_angle_deg: float = Field(default=60.0, ge=0.0, le=120.0)
    allow_hairpin: bool = False


class ObstacleRequest(BaseModel):
    id: Optional[str] = None
    points: List[List[float]]  # [[x1, y1], [x2, y2], ...]
    reason: str = "Benutzer-Sperrgebiet"


class MoveControlRequest(BaseModel):
    control_id: str
    new_x: float
    new_y: float


# ==============================================================================
# SERIALISIERUNGS-HILFSFUNKTIONEN
# ==============================================================================

def serialize_course(course: Course) -> Dict[str, Any]:
    """Konvertiert ein Course-Objekt in ein detailliertes JSON-Format für das Frontend."""
    legs_data = []
    for leg in course.legs:
        alts_data = []
        best_p = leg.route_choice.paths[0] if leg.route_choice.paths else leg.optimal_path

        for idx, p in enumerate(leg.route_choice.paths):
            drift = 0.0 if idx == 0 else calculate_spatial_drift(best_p.points, p.points)
            time_diff = p.estimated_time_s - best_p.estimated_time_s
            delta_pct = (p.cost - best_p.cost) / (best_p.cost + 1e-9) * 100

            alts_data.append({
                "route_id": idx + 1,
                "is_optimal": (idx == 0),
                "distance_m": round(p.distance_m, 1),
                "climb_m": round(p.climb_m, 1),
                "cost": round(p.cost, 1),
                "time_seconds": round(p.estimated_time_s, 1),
                "formatted_time": p.formatted_time,
                "time_diff_s": round(time_diff, 1),
                "delta_pct": round(delta_pct, 1),
                "drift_m": round(drift, 1),
                "points": [[round(pt.x, 1), round(pt.y, 1)] for pt in p.points]
            })

        legs_data.append({
            "leg_index": leg.leg_index,
            "from_control": leg.from_control.id,
            "to_control": leg.to_control.id,
            "direct_line": [
                [round(leg.from_control.point.x, 1), round(leg.from_control.point.y, 1)],
                [round(leg.to_control.point.x, 1), round(leg.to_control.point.y, 1)]
            ],
            "optimal_path": [[round(pt.x, 1), round(pt.y, 1)] for pt in leg.optimal_path.points],
            "has_route_choice": leg.has_route_choice,
            "route_choice_description": leg.route_choice.description,
            "route_choice_score": round(leg.route_choice.score, 1),
            "distance_m": round(leg.distance_m, 1),
            "climb_m": round(leg.climb_m, 1),
            "formatted_time": leg.formatted_time,
            "alternatives": alts_data
        })

    return {
        "controls": [
            {
                "id": c.id,
                "x": round(c.point.x, 1),
                "y": round(c.point.y, 1),
                "z": round(c.point.z, 1),
                "name": c.name,
                "iof_code": c.iof_code,
                "is_start": c.is_start,
                "is_finish": c.is_finish
            }
            for c in course.controls
        ],
        "legs": legs_data,
        "total_distance_m": round(course.total_distance_m, 1),
        "total_climb_m": round(course.total_climb_m, 1),
        "total_time_s": round(course.total_time_s, 1),
        "formatted_total_time": course.formatted_total_time,
        "total_route_choice_score": round(course.total_route_choice_score, 1),
        "leg_variance_cv": round(course.leg_variance_cv, 2),
        "short_legs_count": course.short_legs_count,
        "long_legs_count": course.long_legs_count,
        "is_geometrically_valid": course.is_geometrically_valid,
        "intersection_conflicts": course.intersection_conflicts,
        "angle_warnings": course.angle_warnings
    }


# ==============================================================================
# REST ENDPUNKTE
# ==============================================================================

@app.get("/api/terrain")
def get_terrain() -> Dict[str, Any]:
    """Liefert das semantische Geländemodell im GeoJSON-Format."""
    features = []

    # 1. 2D Belaufbarkeitszonen
    for z in state.terrain.zones.values():
        coords = [[[round(p.x, 1), round(p.y, 1)] for p in z.polygon.exterior]]
        features.append({
            "type": "Feature",
            "id": z.id,
            "geometry": {"type": "Polygon", "coordinates": coords},
            "properties": {
                "layer_type": "zone",
                "zone_type": z.zone_type,
                "friction": z.friction,
                "description": z.description
            }
        })

    # 2. 2D Sperrgebiete
    for fz in state.terrain.forbidden_zones.values():
        coords = [[[round(p.x, 1), round(p.y, 1)] for p in fz.polygon.exterior]]
        features.append({
            "type": "Feature",
            "id": fz.id,
            "geometry": {"type": "Polygon", "coordinates": coords},
            "properties": {
                "layer_type": "forbidden_zone",
                "reason": fz.reason,
                "friction": "inf"
            }
        })

    # 3. 1D Linienobjekte
    for lf in state.terrain.linear_features.values():
        coords = [[round(p.x, 1), round(p.y, 1)] for p in lf.line.points]
        features.append({
            "type": "Feature",
            "id": lf.id,
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {
                "layer_type": "linear",
                "feature_type": lf.feature_type,
                "passable": lf.passable,
                "speed_factor": lf.speed_factor
            }
        })

    # 4. 0D Postenkandidaten
    for c in state.terrain.candidates.values():
        features.append({
            "type": "Feature",
            "id": c.id,
            "geometry": {"type": "Point", "coordinates": [round(c.point.x, 1), round(c.point.y, 1)]},
            "properties": {
                "layer_type": "candidate",
                "name": c.name,
                "iof_code": c.iof_code,
                "is_start": c.is_start,
                "is_finish": c.is_finish,
                "elevation": round(c.point.z, 1)
            }
        })

    return {
        "type": "FeatureCollection",
        "bbox": [0, 0, 500, 500],
        "features": features
    }


@app.post("/api/generate")
def generate_course(req: GenerateRequest) -> Dict[str, Any]:
    """Generiert eine optimierte Bahn anhand der übergebenen Schieberegler."""
    state.config.num_routes = req.num_routes
    state.config.leg_variance = req.leg_variance
    state.config.min_angle_deg = req.min_angle_deg
    state.config.allow_hairpin = req.allow_hairpin

    state.generator = HeuristicCourseGenerator(state.terrain, state.config)
    course = state.generator.generate_course(
        start_id="START",
        finish_id="ZIEL",
        target_distance_m=req.distance_m,
        target_climb_m=req.elevation_m,
        num_controls=req.num_controls
    )

    if not course:
        raise HTTPException(status_code=422, detail="Keine geometrisch gültige Bahn für diese Parameter gefunden.")

    state.active_course = course
    state.course_manager = CourseManager(state.terrain, state.active_course, state.config)

    return {
        "status": "success",
        "message": f"Bahn mit {len(course.controls) - 2} Zwischenposten erfolgreich generiert.",
        "course": serialize_course(course)
    }


@app.post("/api/obstacle")
def add_or_update_obstacle(req: ObstacleRequest) -> Dict[str, Any]:
    """Fügt ein Sperrgebiet hinzu oder verschiebt es und führt den reaktiven Loop aus."""
    if len(req.points) < 3:
        raise HTTPException(status_code=400, detail="Ein Sperrgebiet-Polygon muss mindestens 3 Punkte besitzen.")

    poly_pts = [Point(p[0], p[1]) for p in req.points]
    polygon = Polygon(poly_pts)
    obs_id = req.id or f"user_obs_{len(state.terrain.forbidden_zones) + 1}"

    # Wenn noch keine Bahn generiert wurde, erzeuge eine Default-Bahn
    if not state.active_course:
        default_req = GenerateRequest()
        generate_course(default_req)

    update_report = state.course_manager.update_obstacle(
        obstacle_id=obs_id,
        new_polygon=polygon,
        reason=req.reason
    )

    return {
        "status": "success",
        "obstacle_id": obs_id,
        "relocated_controls": update_report["relocated_controls"],
        "rerouted_legs": update_report["rerouted_legs"],
        "course": serialize_course(state.course_manager.course)
    }


@app.post("/api/move-control")
def move_control(req: MoveControlRequest) -> Dict[str, Any]:
    """Verschiebt einen Posten manuell per Drag-and-Drop und re-routet betroffene Teilstrecken."""
    if not state.active_course or not state.course_manager:
        raise HTTPException(status_code=400, detail="Aktuell ist keine aktive Bahn geladen.")

    target_c = None
    target_idx = -1
    for idx, c in enumerate(state.active_course.controls):
        if c.id == req.control_id:
            target_c = c
            target_idx = idx
            break

    if not target_c:
        raise HTTPException(status_code=404, detail=f"Posten '{req.control_id}' in aktiver Bahn nicht gefunden.")

    new_pt = Point(req.new_x, req.new_y, state.terrain.elevation_model.get_elevation(req.new_x, req.new_y))

    # Prüfe ob neue Koordinate im Sperrgebiet liegt
    if not state.terrain.is_point_allowed(new_pt):
        raise HTTPException(status_code=400, detail="Der Posten kann nicht in ein Sperrgebiet verschoben werden.")

    # Postenposition aktualisieren
    target_c.point = new_pt
    if target_c.id in state.terrain.candidates:
        state.terrain.candidates[target_c.id].point = new_pt

    # Navigationsgraph & Teilstrecken aktualisieren
    state.course_manager.nav_graph.rebuild()
    rerouted_course = state.generator._evaluate_course(state.active_course.controls)
    state.active_course = rerouted_course
    state.course_manager.course = rerouted_course

    return {
        "status": "success",
        "message": f"Posten {req.control_id} nach ({req.new_x:.1f}, {req.new_y:.1f}) verschoben.",
        "course": serialize_course(rerouted_course)
    }


@app.post("/api/reset-obstacles")
def reset_obstacles() -> Dict[str, Any]:
    """Setzt alle Sperrgebiete und Posten auf den Initialzustand des Testgeländes zurück."""
    state.reset_terrain()
    return {"status": "success", "message": "Gelände und Sperrgebiete zurückgesetzt."}


@app.get("/api/course")
def get_current_course() -> Dict[str, Any]:
    """Liefert den aktuellen Bahnstand zurück."""
    if not state.active_course:
        return {"course": None}
    return {"course": serialize_course(state.active_course)}


# ==============================================================================
# STATISCHES FRONTEND MOUNTEN
# ==============================================================================

frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
