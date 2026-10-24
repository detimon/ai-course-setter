# AI Course Setter (Orientierungslauf-Bahnlegung)

Intelligente, regelbasierte und reaktive Web-Applikation zur Bahnlegung im Orientierungslauf (OL) mit universellem semantischem Geländemodell und interaktiver Karte.

---

## 1. Vision & Philosophie

Die Bahnlegung im Orientierungslauf ist eine Kunst, die präzise Kartengeometrie, sportphysiologische Anforderungen und kognitive Herausforderungen (Routenwahlen) miteinander verbindet. 

Klassische ML-Modelle ("Black Box") sind für diesen Zweck ungeeignet, da:
- Bahnleger die exakten Gründe für eine Routenführung nachvollziehen und anpassen wollen.
- Sportliche und sicherheitsrelevante Regeln (keine spitzen Kehren, keine kreuzenden Verbindungslinien, absolute Sperrgebiete) strikt und ohne Ausnahmen garantiert werden müssen.
- Geländedaten in unterschiedlichsten Formaten vorliegen (GeoJSON, OCAD/LibOCAD, Raster-Farbauszüge).

**AI Course Setter** setzt daher auf:
1. **Universelles semantisches Geländemodell (0D, 1D, 2D, 2.5D)** zur Abstraktion aller Kartenquellen.
2. **Sperrgebiete zuerst ("Obstacle-First") & reaktives Verhalten**: Sofortige automatische Neuführung bei Sperrungen oder Konflikten.
3. **Heuristische postensetzende Engine**: Deterministische Optimierung auf Streckenlänge, Höhenmeter und echten Routenwahl-Score (Yen's K-Shortest Paths via A*).
4. **Voll konfigurierbare Bewertungsmetriken**: Alle Faktoren (Winkel, Reibungen, Schwellwerte) als dynamisches Parameterset.
5. **Co-Pilot Workflow**: Der Bahnleger behält die volle Kontrolle. Posten können verschoben oder fixiert werden; das System aktualisiert Teilstrecken und Statistiken in Echtzeit.

---

## 2. Datenmodell: Universelles Semantisches Geländemodell

Kartendaten beliebiger Herkunft (Vektor, CAD, Raster) werden in vier geometrisch-semantische Dimensionen normalisiert:

```
+-----------------------------------------------------------------------------------+
|                        UNIVERSAL SEMANTIC TERRAIN MODEL                           |
+-----------------------------------------------------------------------------------+
| 0D: Points (Postenkandidaten)                                                     |
|     - ID, Koordinate (x, y), Höhe z                                               |
|     - IOF-Merkmalscode (z.B. 101 Kuppe, 201 Fels, 301 Senke, 501 Hausecke)        |
|     - Status: Start, Posten, Ziel, Fixiert / Gesperrt                             |
+-----------------------------------------------------------------------------------+
| 1D: Linear Features (Pfade, Strassen, Mauern, Zäune, Bäche)                        |
|     - Linienzug LineString[(x1,y1), (x2,y2), ...]                                 |
|     - Typ & Passierbarkeit (passable: True/False)                                 |
|     - Geschwindigkeits- bzw. Reibungsfaktor (z.B. Pfad = 0.7 Bonus, Bach = 1.3)   |
+-----------------------------------------------------------------------------------+
| 2D: Zones (Belaufbarkeitszonen & Reibungsflächen)                                 |
|     - Polygon(Exterior, Holes)                                                    |
|     - Zonenkategorie (Offen=0.9, Wald=1.0, Grün1=1.6, Grün2=2.5, Grün3=10.0, ...) |
|     - Sperrzonen / Out-of-Bounds (Naturschutz, Baustelle, Privat): Reibung = inf  |
+-----------------------------------------------------------------------------------+
| 2.5D: Elevation Model (Höhenmodell)                                               |
|     - Kontinuierliche z-Höhenabfrage z(x, y)                                      |
|     - Gradientenberechnung für Steigungsstrafen (z.B. 1 Hm climb = 10 m Distanz)  |
+-----------------------------------------------------------------------------------+
```

### IOF-Reibungstabelle (Default Settings)
| Kategorie | Terrain-Typ | Reibungsfaktor (Friction) | Beschreibung |
|---|---|---|---|
| **Paved / Road** | Strasse / Fester Weg | `0.6 - 0.7` | Maximaler Geschwindigkeitsbonus |
| **Path / Trail** | Waldpfad / Pfadspur | `0.75 - 0.85` | Schnelles Vorankommen ohne Hindernisse |
| **Open Land** | Offenes Gelände / Wiese | `0.9` | Gute Belaufbarkeit |
| **Normal Forest** | Weißer Wald (100%) | `1.0` | Referenzwert (100% Laufgeschwindigkeit) |
| **Rough Open** | Heideland / Halboffen | `1.2 - 1.4` | Leicht erschwerte Belaufbarkeit |
| **Green 1** | Schwer belaufbar (Grün 1) | `1.6` | Verlangsamtes Laufen (Unterholz) |
| **Green 2** | Sehr schwer belaufbar (Grün 2) | `2.5 - 3.5` | Stark bremsendes Dickicht |
| **Green 3 / Fight** | Kaum belaufbar (Grün 3) | `8.0 - 10.0` | Extremes Dickicht, Durchkämpfen |
| **Forbidden / Water** | See, Oliv, Privat, Sperrzone | `Infinity (inf)` | Absolut unpassierbar |

---

## 3. Architektur & Engine-Komponenten

```
                   +---------------------------+
                   |  Map Input (OCAD / GeoJSON |
                   |  / Raster-Segmentation)   |
                   +-------------+-------------+
                                 |
                                 v
                   +---------------------------+
                   |   terrain_model.py        |
                   | - Geometry Primitives     |
                   | - Semantic Terrain Model  |
                   | - Forbidden Zone Manager  |
                   | - Spatial & Cost Query    |
                   +-------------+-------------+
                                 |
                                 v
                   +---------------------------+
                   |   course_engine.py        |
                   | - Navigation Graph        |
                   | - A* & Yen's K-Shortest   |
                   | - Route Choice Metric     |
                   | - Geometry Constraints    |
                   | - Heuristic Generator     |
                   | - Reactive Update Loop    |
                   +-------------+-------------+
                                 |
        +------------------------+------------------------+
        v                                                 v
+-----------------------------+             +-----------------------------+
| CLI / Tests (test_core.py)  |             | Future UI (FastAPI/React)   |
| - Verifikation aller Regeln |             | - Interaktive Karte         |
| - Terminal-Visualisierung   |             | - Co-Pilot & Drag-and-Drop  |
+-----------------------------+             +-----------------------------+
```

### Kernregeln der Postensetzer-Engine

1. **Routenwahl-Metrik (Drift & Zeitdifferenz)**:
   - Berechnet die K kürzesten Pfade zwischen zwei Posten mittels **Yen's K-Shortest Paths** auf Basis von A*.
   - Eine Teilstrecke besitzt *echte Routenwahl*, wenn mindestens zwei Routen vorliegen mit:
     $$\Delta = \frac{\text{Kosten}_{\text{Pfad 2}} - \text{Kosten}_{\text{Pfad 1}}}{\text{Kosten}_{\text{Pfad 1}}} \le 8\%$$
     und einer maximalen räumlichen Spreizung (Drift) $\ge \text{Schwellwert}$ (z.B. 25–40 m).
   - Derartige Teilstrecken erhalten einen signifikanten **Route-Choice-Bonus** im Scoring.

2. **Geometrische Gültigkeit**:
   - **Kreuzungsverbot**: Gerade Verbindungslinien zwischen Posten dürfen einander nicht schneiden:
     $$\forall (L_i, L_j) \text{ mit } |i - j| > 1: L_i \cap L_j = \emptyset$$
   - **Winkelprüfung (Spitze Kehren / Dog-legs)**:
     Winkel $\alpha$ an Posten $P_i$ zwischen Anlauf $(P_{i-1} \to P_i)$ und Ablauf $(P_i \to P_{i+1})$ muss mindestens $\ge 60^\circ$ betragen (außer `allow_hairpin=True`), um Kollisionen und Gegenverkehr zu verhindern.

3. **Reaktives Verhalten (Sperrgebiete zuerst)**:
   - Sperrzonen besitzen Priorität $\infty$.
   - Bei Anlage oder Verschiebung einer Sperrzone werden betroffene Posten automatisch auf die nächstgelegene legale Koordinate projiziert.
   - Alle Teilstrecken, deren Pfad die Sperrzone schneidet, werden sofort und transparent neu geroutet.

---

## 4. Roadmap

### Phase 1: Backend- & Geometriekern (Aktueller Meilenstein)
- [x] Initialisierung Projektstruktur und Dokumentation (`PROJECT.md`)
- [x] `terrain_model.py`: Semantische Datentypen (0D, 1D, 2D, 2.5D), Point-in-Polygon, Segment-Intersections, Reibungsberechnung
- [x] `course_engine.py`: Navigationsgraph, A*, Yen's K-Shortest Paths, Routenwahl-Score, Winkel- & Schnittfilter, Heuristischer Bahnengenerator, Reaktives Update
- [x] `test_core.py`: Vollständige Verifikation mit synthetischem Gelände, Postengenerierung und Sperrgebiet-Umgehung

### Phase 2: Kartendaten-Importeure
- [ ] GeoJSON Parser für Vektordaten
- [ ] OCAD / LibOCAD Importer für IOF-Kartensymbole (ISOM 2017-2 / ISSprOM 2019-2)
- [ ] Raster-Segmentierer (Farbzerlegung für gescannte Karten / PDFs)
- [ ] Digitales Geländemodell (DHM / GeoTIFF / SRTM) Integration

### Phase 3: Interaktives Web-Frontend
- [ ] Backend-API mit FastAPI / WebSockets für Echtzeit-Interaktion
- [ ] MapLibre / Leaflet Kartenkomponente mit OL-Farbschema
- [ ] Interaktiver Co-Pilot: Posten per Drag & Drop verschieben, fixieren, neu routen
- [ ] Sperrzonen-Zeichentool (Polygon-Editor direkt auf der Karte)
- [ ] Live-HUD mit Längenbudget, Höhenmetern, Routenwahl-Statistiken

### Phase 4: Standard-Export & Interoperabilität
- [ ] IOF XML 3.0 Export (Standardformat für OCAD, Condes, Purple Pen)
- [ ] Postenbeschreibungen nach IOF-Standard (Symbole & Text)
- [ ] PDF-Druckexport für Bahnen und Postennetze

### Phase 5: Fortgeschrittene Bahnlegungs-Features
- [ ] Mehretappen- & Staffelbahnlegung (Gabelungen, Farsta, MOP)
- [ ] Automatische Kategorien-Skalierung (z.B. D14, H21, D60) aus einem Postennetz
- [ ] KI-gestütztes Tuning der Reibungsfaktoren anhand echter GPS-Tracks (Strava / Livelox)

---

## 5. Status & Changelog

- **2026-10-24**: Phase 1 gestartet. Geometrie- und Routingkern mit nativer, robuster Implementierung von Yen's Algorithmus, A*, Kollisionsprüfung und Sperrzonen-Reaktivität konzipiert.
