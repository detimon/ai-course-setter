/**
 * app.js - Interaktive Leaflet-Karten-Engine für den AI Course Setter.
 * 
 * Behandelt:
 * - L.CRS.Simple Kartenansicht mit OL-Gelände
 * - Schieberegler-Binding für Bahnengenerierung
 * - Draggable Posten mit Live-Rerouting
 * - Interaktives Einzeichnen von Sperrgebieten mit reaktiver Umgehung
 * - K-Routenwahl Visualisierung mit Kontrastfarben & Zeitvergleich
 */

// Globaler Anwendungs-Status
const state = {
  map: null,
  terrainData: null,
  activeCourse: null,
  
  // Layer-Gruppen
  layers: {
    terrain: null,
    obstacles: null,
    courseDirectLines: null,
    courseRoutes: null,
    controls: null,
    drawLayer: null
  },

  // Zeichenmodus für Sperrgebiete
  drawing: {
    active: false,
    points: [],
    tempMarkers: [],
    tempLine: null
  },

  // Routenfarben
  routeColors: [
    "#00e5ff", // Route 1 (Optimal) - Cyan
    "#ff9100", // Route 2 - Orange
    "#76ff03", // Route 3 - Lime
    "#e040fb", // Route 4 - Magenta
    "#ffd600"  // Route 5 - Gelb
  ]
};

// ==============================================================================
// 1. KARTEN-INITIALISIERUNG (L.CRS.Simple für 500x500m Gelände)
// ==============================================================================

function initMap() {
  // L.CRS.Simple: Lat = Y (0..500), Lng = X (0..500)
  state.map = L.map("map", {
    crs: L.CRS.Simple,
    minZoom: -1,
    maxZoom: 3,
    zoomSnap: 0.25,
    attributionControl: false
  });

  const bounds = [[0, 0], [500, 500]];
  state.map.fitBounds(bounds, { padding: [20, 20] });
  state.map.setMaxBounds([[-50, -50], [550, 550]]);

  // Layer-Gruppen anlegen
  state.layers.terrain = L.layerGroup().addTo(state.map);
  state.layers.obstacles = L.layerGroup().addTo(state.map);
  state.layers.courseDirectLines = L.layerGroup().addTo(state.map);
  state.layers.courseRoutes = L.layerGroup().addTo(state.map);
  state.layers.controls = L.layerGroup().addTo(state.map);
  state.layers.drawLayer = L.layerGroup().addTo(state.map);

  // Klick-Event für Sperrgebiet-Zeichnen
  state.map.on("click", onMapClick);
}

// ==============================================================================
// 2. GELÄNDE-RENDERING (IOF-Farbschema)
// ==============================================================================

async function loadTerrain() {
  try {
    const res = await fetch("/api/terrain");
    if (!res.ok) throw new Error("Fehler beim Laden der Geländedaten");
    state.terrainData = await res.json();
    renderTerrain();
  } catch (err) {
    showToast("Fehler: " + err.message, "danger");
  }
}

function renderTerrain() {
  state.layers.terrain.clearLayers();
  state.layers.obstacles.clearLayers();

  if (!state.terrainData || !state.terrainData.features) return;

  // 500x500m Grundgelände (Weisser Wald)
  L.rectangle([[0, 0], [500, 500]], {
    color: "#94a3b8",
    weight: 1,
    fillColor: "#ffffff",
    fillOpacity: 1.0,
    interactive: false
  }).addTo(state.layers.terrain);

  state.terrainData.features.forEach(feat => {
    const props = feat.properties;

    // A) 2D Belaufbarkeitszonen
    if (props.layer_type === "zone") {
      const ring = feat.geometry.coordinates[0];
      const latlngs = ring.map(p => [p[1], p[0]]); // [y, x]

      let fillColor = "#ffffff";
      let fillOpacity = 0.85;
      let strokeColor = "#64748b";

      if (props.zone_type === "open") {
        fillColor = "#fef08a"; // Gelb
      } else if (props.zone_type === "green_1") {
        fillColor = "#bbf7d0"; // Hellgrün
      } else if (props.zone_type === "green_3") {
        fillColor = "#166534"; // Dunkelgrün (Dickicht)
        fillOpacity = 0.95;
      } else if (props.zone_type === "water") {
        fillColor = "#38bdf8"; // Blau
        strokeColor = "#0288d1";
      }

      L.polygon(latlngs, {
        color: strokeColor,
        weight: 1,
        fillColor: fillColor,
        fillOpacity: fillOpacity
      }).bindTooltip(`Zone: ${props.zone_type} (Friction: ${props.friction})`).addTo(state.layers.terrain);
    }

    // B) 2D Sperrgebiete
    else if (props.layer_type === "forbidden_zone") {
      const ring = feat.geometry.coordinates[0];
      const latlngs = ring.map(p => [p[1], p[0]]);

      L.polygon(latlngs, {
        color: "#dc2626",
        weight: 3,
        dashArray: "6, 6",
        fillColor: "#ef4444",
        fillOpacity: 0.45
      }).bindTooltip(`🛑 Sperrgebiet: ${props.reason || 'Gesperrt'}`).addTo(state.layers.obstacles);
    }

    // C) 1D Linienobjekte (Pfade, Mauern)
    else if (props.layer_type === "linear") {
      const latlngs = feat.geometry.coordinates.map(p => [p[1], p[0]]);

      if (!props.passable) {
        // Mauer (Schwarz, dick)
        L.polyline(latlngs, {
          color: "#0f172a",
          weight: 4,
          opacity: 1
        }).bindTooltip("Steinmauer (Unpassierbar)").addTo(state.layers.terrain);
      } else {
        // Weg / Pfad (Braun gestrichelt)
        L.polyline(latlngs, {
          color: "#92400e",
          weight: 3,
          dashArray: "5, 5",
          opacity: 0.9
        }).bindTooltip(`Waldweg (Bonus: ${props.speed_factor})`).addTo(state.layers.terrain);
      }
    }
  });
}

// ==============================================================================
// 3. BAHN-RENDERING & INTERAKTIVE POSTEN (DRAG & DROP)
// ==============================================================================

function renderCourse(courseData) {
  state.activeCourse = courseData;

  state.layers.courseDirectLines.clearLayers();
  state.layers.courseRoutes.clearLayers();
  state.layers.controls.clearLayers();

  if (!courseData || !courseData.controls) return;

  // 1. Teilstrecken & K-Routenwahl-Alternativen zeichnen
  courseData.legs.forEach(leg => {
    // A) Direkte OL-Verbindungslinie (Violett)
    const directPts = leg.direct_line.map(p => [p[1], p[0]]);
    L.polyline(directPts, {
      color: "#c026d3",
      weight: 2,
      opacity: 0.75,
      dashArray: "4, 6"
    }).addTo(state.layers.courseDirectLines);

    // B) K berechnete Routenwahl-Alternativen
    if (leg.alternatives && leg.alternatives.length > 0) {
      leg.alternatives.forEach((alt, altIdx) => {
        const routePts = alt.points.map(p => [p[1], p[0]]);
        const color = state.routeColors[altIdx % state.routeColors.length];
        const isOpt = alt.is_optimal;

        const line = L.polyline(routePts, {
          color: color,
          weight: isOpt ? 4 : 3,
          opacity: isOpt ? 0.95 : 0.75,
          dashArray: isOpt ? null : "6, 4"
        }).addTo(state.layers.courseRoutes);

        // Tooltip mit Zeitvergleich
        const diffText = isOpt ? "OPTIMAL" : `+${alt.time_diff_s}s (+${alt.delta_pct}%)`;
        line.bindTooltip(
          `<strong>Leg ${leg.leg_index + 1} - Route ${alt.route_id}</strong><br>` +
          `Laufzeit: <b>${alt.formatted_time}</b> (${diffText})<br>` +
          `Distanz: ${alt.distance_m} m | Höhenmeter: +${alt.climb_m} Hm<br>` +
          `Spreizung (Drift): ${alt.drift_m} m`
        );

        // Klick auf Route hebt Teilstrecke hervor
        line.on("click", () => selectLeg(leg.leg_index));
      });
    }
  });

  // 2. Posten-Marker setzen (Draggable!)
  courseData.controls.forEach((c, idx) => {
    const latlng = [c.y, c.x];
    let iconClass = "ol-marker-control";
    let htmlContent = `${idx}`;

    if (c.is_start) {
      iconClass = "ol-marker-start";
      htmlContent = "";
    } else if (c.is_finish) {
      iconClass = "ol-marker-finish";
      htmlContent = "";
    }

    const icon = L.divIcon({
      className: iconClass,
      html: htmlContent,
      iconSize: [26, 26],
      iconAnchor: [13, 13]
    });

    const marker = L.marker(latlng, {
      icon: icon,
      draggable: true,
      title: `${c.id}: ${c.name}`
    }).addTo(state.layers.controls);

    marker.bindTooltip(`<b>${c.id}</b>: ${c.name} (${c.iof_code})<br><i>Gedrückt halten zum Verschieben</i>`);

    // Drag-and-Drop: Posten manuell verschieben
    marker.on("dragend", async (e) => {
      const newPos = e.target.getLatLng();
      await moveControl(c.id, newPos.lng, newPos.lat);
    });
  });

  // HUD & Sidebar aktualisieren
  updateHUD(courseData);
  updateLegsList(courseData);
}

// ==============================================================================
// 4. API ACTIONS: GENERIEREN, VERSCHIEBEN, SPERRGEBIETE
// ==============================================================================

async function generateCourse() {
  const btn = document.getElementById("btn-generate");
  btn.disabled = true;
  btn.innerHTML = `<span class="btn-icon">⏳</span> Berechne...`;

  const req = {
    distance_m: parseFloat(document.getElementById("input-distance").value),
    elevation_m: parseFloat(document.getElementById("input-elevation").value),
    num_controls: parseInt(document.getElementById("input-controls").value),
    num_routes: parseInt(document.getElementById("input-routes").value),
    leg_variance: parseFloat(document.getElementById("input-variance").value),
    allow_hairpin: document.getElementById("input-hairpin").checked
  };

  try {
    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req)
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Fehler bei der Bahnengenerierung");

    renderCourse(data.course);
    showToast(data.message, "success");
  } catch (err) {
    showToast(err.message, "danger");
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<span class="btn-icon">⚡</span> Bahn generieren / optimieren`;
  }
}

async function moveControl(controlId, newX, newY) {
  try {
    const res = await fetch("/api/move-control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ control_id: controlId, new_x: newX, new_y: newY })
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Verschieben fehlgeschlagen");

    renderCourse(data.course);
    showToast(data.message, "success");
  } catch (err) {
    showToast(err.message, "danger");
    // Bei Fehler altes Gelände neu laden
    if (state.activeCourse) renderCourse(state.activeCourse);
  }
}

async function resetObstacles() {
  try {
    const res = await fetch("/api/reset-obstacles", { method: "POST" });
    const data = await res.json();
    showToast(data.message, "success");
    await loadTerrain();
    if (state.activeCourse) await generateCourse();
  } catch (err) {
    showToast("Fehler beim Zurücksetzen: " + err.message, "danger");
  }
}

// ==============================================================================
// 5. INTERAKTIVES SPERRGEBIET-ZEICHNEN
// ==============================================================================

function toggleObstacleDrawMode() {
  const btn = document.getElementById("btn-draw-obstacle");
  const help = document.getElementById("draw-help");

  state.drawing.active = !state.drawing.active;

  if (state.drawing.active) {
    btn.innerHTML = `<span class="btn-icon">✓</span> Fertigstellen`;
    btn.classList.add("btn-primary");
    btn.classList.remove("btn-danger");
    help.classList.remove("hidden");
    state.map.getContainer().style.cursor = "crosshair";
    state.drawing.points = [];
    state.layers.drawLayer.clearLayers();
  } else {
    finishObstacleDrawing();
  }
}

function onMapClick(e) {
  if (!state.drawing.active) return;

  const lat = e.latlng.lat;
  const lng = e.latlng.lng;
  const pt = [lng, lat]; // [x, y] in Metern

  state.drawing.points.push(pt);

  // Punkt-Marker auf Karte
  const marker = L.circleMarker([lat, lng], {
    radius: 5,
    color: "#dc2626",
    fillColor: "#ef4444",
    fillOpacity: 1
  }).addTo(state.layers.drawLayer);

  // Beim Klick auf den ersten Punkt -> sofort beenden
  if (state.drawing.points.length > 2 && marker === state.drawing.tempMarkers[0]) {
    finishObstacleDrawing();
    return;
  }

  state.drawing.tempMarkers.push(marker);

  // Vorschau-Linie
  if (state.drawing.points.length > 1) {
    if (state.drawing.tempLine) state.layers.drawLayer.removeLayer(state.drawing.tempLine);
    const linePts = state.drawing.points.map(p => [p[1], p[0]]);
    state.drawing.tempLine = L.polyline(linePts, { color: "#dc2626", dashArray: "4, 4" }).addTo(state.layers.drawLayer);
  }
}

async function finishObstacleDrawing() {
  const btn = document.getElementById("btn-draw-obstacle");
  const help = document.getElementById("draw-help");

  state.drawing.active = false;
  btn.innerHTML = `<span class="btn-icon">🛑</span> Sperrgebiet zeichnen`;
  btn.classList.add("btn-danger");
  btn.classList.remove("btn-primary");
  help.classList.add("hidden");
  state.map.getContainer().style.cursor = "";

  const pts = state.drawing.points;
  state.layers.drawLayer.clearLayers();

  if (pts.length < 3) {
    showToast("Sperrgebiet abgebrochen (mindestens 3 Punkte erforderlich).", "warning");
    return;
  }

  try {
    const res = await fetch("/api/obstacle", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        points: pts,
        reason: "Benutzer-Sperrgebiet"
      })
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Fehler beim Anlegen des Sperrgebiets");

    await loadTerrain();
    renderCourse(data.course);

    let msg = `Sperrgebiet angelegt! `;
    if (data.relocated_controls && data.relocated_controls.length > 0) {
      msg += `Posten ${data.relocated_controls.map(c => c.control_id).join(', ')} automatisch evakuiert!`;
    } else {
      msg += `Betroffene Teilstrecken automatisch umgeleitet.`;
    }
    showToast(msg, "success");
  } catch (err) {
    showToast("Fehler: " + err.message, "danger");
  }
}

// ==============================================================================
// 6. HUD & SIDEBAR UPDATES
// ==============================================================================

function updateHUD(course) {
  document.getElementById("stat-dist").textContent = `${course.total_distance_m} m`;
  document.getElementById("stat-climb").textContent = `+${course.total_climb_m} Hm`;
  document.getElementById("stat-time").textContent = course.formatted_total_time;
  document.getElementById("stat-cv").textContent = `CV ${course.leg_variance_cv} (${course.short_legs_count} kurz / ${course.long_legs_count} lang)`;

  const valEl = document.getElementById("stat-validity");
  if (course.is_geometrically_valid) {
    valEl.textContent = "Gültig (0 Fehler)";
    valEl.className = "badge badge-success";
  } else {
    valEl.textContent = "Geometrie-Konflikt";
    valEl.className = "badge badge-danger";
  }

  document.getElementById("stat-rc-score").textContent = `Routenwahl-Score: ${course.total_route_choice_score}`;
}

function updateLegsList(course) {
  const container = document.getElementById("legs-list");
  container.innerHTML = "";

  course.legs.forEach(leg => {
    const card = document.createElement("div");
    card.className = "leg-card";
    card.id = `leg-card-${leg.leg_index}`;

    const rcBadge = leg.has_route_choice
      ? `<span class="leg-route-badge">Echte Routenwahl</span>`
      : `<span style="font-size: 0.65rem; color: #64748b;">Dominanter Pfad</span>`;

    let altsHtml = "";
    if (leg.alternatives && leg.alternatives.length > 0) {
      altsHtml = `<div class="leg-alternatives-box">`;
      leg.alternatives.forEach((alt, idx) => {
        const color = state.routeColors[idx % state.routeColors.length];
        const diffText = alt.is_optimal ? "Bestzeit" : `+${alt.time_diff_s}s`;
        altsHtml += `
          <div class="alt-row">
            <span><span class="alt-pill" style="background:${color};"></span>Route ${alt.route_id} (${alt.formatted_time})</span>
            <span>${alt.distance_m}m | <b>${diffText}</b></span>
          </div>
        `;
      });
      altsHtml += `</div>`;
    }

    card.innerHTML = `
      <div class="leg-header">
        <span>Leg ${leg.leg_index + 1}: Posten ${leg.from_control} → ${leg.to_control}</span>
        ${rcBadge}
      </div>
      <div class="leg-metrics">
        <span>📏 ${leg.distance_m} m</span>
        <span>⛰️ +${leg.climb_m} Hm</span>
        <span>⏱️ ${leg.formatted_time}</span>
      </div>
      ${altsHtml}
    `;

    card.addEventListener("click", () => selectLeg(leg.leg_index));
    container.appendChild(card);
  });
}

function selectLeg(legIndex) {
  document.querySelectorAll(".leg-card").forEach(c => c.style.borderColor = "var(--border-color)");
  const card = document.getElementById(`leg-card-${legIndex}`);
  if (card) {
    card.style.borderColor = "#3b82f6";
    card.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  // Karte auf Teilstrecke zoomen/zentrieren
  const leg = state.activeCourse.legs[legIndex];
  if (leg && leg.optimal_path.length > 0) {
    const pts = leg.optimal_path.map(p => [p[1], p[0]]);
    const poly = L.polyline(pts);
    state.map.fitBounds(poly.getBounds(), { padding: [60, 60] });
  }
}

function showToast(message, type = "info") {
  const toast = document.getElementById("toast");
  toast.textContent = message;
  toast.className = `toast toast-${type}`;
  toast.classList.remove("hidden");
  setTimeout(() => toast.classList.add("hidden"), 3500);
}

// ==============================================================================
// 7. EVENT LISTENER BINDING
// ==============================================================================

function bindEvents() {
  // Slider Value Badges
  const bindSlider = (sliderId, badgeId, suffix = "") => {
    const slider = document.getElementById(sliderId);
    const badge = document.getElementById(badgeId);
    slider.addEventListener("input", () => {
      badge.textContent = `${slider.value} ${suffix}`;
    });
  };

  bindSlider("input-distance", "val-distance", "m");
  bindSlider("input-elevation", "val-elevation", "Hm");
  bindSlider("input-controls", "val-controls", "");
  bindSlider("input-routes", "val-routes", "");

  document.getElementById("input-variance").addEventListener("input", (e) => {
    const val = parseFloat(e.target.value);
    let label = `${val.toFixed(2)}`;
    if (val <= 0.1) label += " (Gleichmässig)";
    else if (val >= 0.5) label += " (Kurz & Lang)";
    else label += " (Ausgewogen)";
    document.getElementById("val-variance").textContent = label;
  });

  // Buttons
  document.getElementById("btn-generate").addEventListener("click", generateCourse);
  document.getElementById("btn-draw-obstacle").addEventListener("click", toggleObstacleDrawMode);
  document.getElementById("btn-reset-obstacles").addEventListener("click", resetObstacles);
}

// Initialisierung bei DOMContentLoaded
document.addEventListener("DOMContentLoaded", async () => {
  initMap();
  bindEvents();
  await loadTerrain();
  await generateCourse();
});
