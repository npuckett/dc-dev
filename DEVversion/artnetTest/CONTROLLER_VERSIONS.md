# Art-Net Controller Prototypes — Version Summary

These prototypes were built to explore different approaches to driving the 12 LED panels (4 units × 3 panels each) via Art-Net/DMX before the final production system was developed.

All controllers target `10.42.0.200` on Universe 0, driving 12 DMX channels (1 per panel).

---

## 1. artnetTest.py — Direct Channel Control
**Location:** `DMXtest/artnetTest.py` (294 lines)
**UI:** Tkinter — 12 individual sliders + master controls
**Purpose:** First controller. Direct manual DMX control with no animation logic. Used to verify Art-Net connectivity, find safe DMX ranges, and test individual panels.

- 12 sliders (one per channel), organized by unit
- Master slider, All ON/50%/Low, Fade In/Out buttons
- Chase effect (sequential panel flash)
- Random Fade mode (panels drift between random values 1–50)
- Discovered hardware limits: `MAX_DMX = 212` (10V from 12V decoder), `MIN_DMX = 1` (panels won't reliably turn off at 0)

**Key constant:** Established the 1–50 practical brightness range used by all subsequent versions.

---

## 2. simpleWaveController.py — Linear Wave
**Location:** `DEVversion/artnetTest/simpleWaveController.py` (282 lines)
**UI:** Tkinter — 4 sliders (origin, speed, width, noise)
**Purpose:** First animation. A sine wave sweeps across X. All 3 panels in a unit share the same base value.

- Wave origin, speed, and width controls
- Per-panel Perlin noise layer for organic variation
- Front panels (1 & 2): capped at 1–50 DMX
- Back panels (3): separate range 1–200 DMX
- No person tracking — purely parametric animation

---

## 3. radialGradientController.py — Radial Gradient
**Location:** `DEVversion/artnetTest/radialGradientController.py` (304 lines)
**UI:** Tkinter — 6 sliders (origin, speed, phase, min/max DMX, noise)
**Purpose:** Expanding radial gradient from an origin point. First use of smooth parameter interpolation (lerp).

- Animated gradient pulsing outward from a movable origin
- All parameters lerp toward targets (~2s transition) instead of snapping
- Per-unit noise seeds for variation
- Greyscale mapping with adjustable min/max DMX range
- Units treated as 1D positions (no per-panel differentiation)

---

## 4. radialPulseController.py — 3D Point Light
**Location:** `DEVversion/artnetTest/radialPulseController.py` (475 lines)
**UI:** Tkinter — 9 sliders (X/Y/Z origin, pulse speed/amplitude, falloff, distance, noise, DMX max)
**Purpose:** First 3D point light model. Each panel has a collector point; brightness = f(distance). This became the foundation for the production system.

- 3D panel positions in meters (physically accurate: 60cm spacing, angled panels)
- Point light at (x, y, z) with configurable falloff exponent
- Inverse-distance brightness: `brightness = 1 / (1 + distance^falloff)`
- Pulse animation modulates light intensity over time
- Per-panel noise for organic feel
- `PulseConfig` dataclass — first use of structured config

**Key innovation:** The collector-point + distance-falloff model carried directly into production.

---

## 5. vectorController.py — Directional Vector Sweep
**Location:** `DEVversion/artnetTest/vectorController.py` (617 lines)
**UI:** Tkinter — 8 sliders (speed, vector X/Y, wave speed/width, Z delay, noise, min brightness) + panel grid display
**Purpose:** Movement vectors sweep across the panel grid. Brightness follows a traveling wave in an arbitrary direction.

- Full 3D panel positions (x, y, z) with front/back differentiation
- Configurable movement vector (dx, dy) — wave can travel diagonally
- Z-delay parameter: back panels respond with time offset (depth effect)
- Tkinter canvas shows real-time panel brightness as colored rectangles
- Front (1–50) and back (1–200) DMX ranges separated
- `VectorConfig` dataclass with master speed control

---

## 6. springController.py — Spring Physics (CLI)
**Location:** `DEVversion/artnetTest/springController.py` (949 lines)
**UI:** Terminal only — keyboard input (1–4 to simulate person, SPACE to exit, Q to quit)
**Purpose:** First physics-based controller. 4×2 spring grid (front panels) + 4×1 independent strip (back panels). External forces propagate through coupled springs.

- `SpringNode` with position, velocity, rest position, damping
- Neighbor coupling: adjacent springs pull each other
- Active zone input applies force at nearest column
- Dwell time increases force magnitude over time
- Time-of-day awareness: rush hours, evening mode
- `NarrativeState` system: Ambient → Acknowledge → Engage → Deeper Engagement → Peak
- Engagement memory (acknowledgment boost decays over 5 min)
- Back panels respond independently with their own wave behavior
- `--simulate` and `--debug` CLI flags

**Key innovation:** The spring-coupled physics and narrative state machine became the conceptual basis for the production behavior system.

---

## 7. springControllerGUI.py — Spring Physics + GUI
**Location:** `DEVversion/artnetTest/springControllerGUI.py` (855 lines)
**UI:** Tkinter — simulation controls + real-time panel display + narrative status
**Purpose:** Interactive GUI version of the spring controller for testing and demonstration.

- Same spring physics as springController.py
- Input simulation panel: sliders for active population, position, dwell time, passive population
- Real-time panel brightness display (colored rectangles)
- Narrative status display showing current state/action/mood
- Spring value readout
- Auto mode with simulated person entry/exit
- Time-of-day display

---

## 8. springControllerGUI_v2.py — Enhanced Spring + Tuning
**Location:** `DEVversion/artnetTest/springControllerGUI_v2.py` (1,218 lines)
**UI:** Tkinter — extended with spring visualization canvas + tuning controls
**Purpose:** Enhanced version with visual spring debugging, more panel contrast, and live physics tuning.

- Row differentiation: top vs bottom panels respond differently (delay, amplitude, phase offset)
- Much stronger forces (`active_force_base`: 15 → 40) for visible response
- Sharper position falloff for left/right contrast
- Lower coupling (0.3 → 0.05 horizontal, 0.02 vertical) for panel independence
- Spring visualization: canvas shows spring nodes with connecting lines
- Per-column variation for organic feel
- Bigger ambient waves (`amplitude`: 6.0)
- Live tuning sliders for spring stiffness, damping, coupling, forces

---

## 9. pointLightController3D.py — 3D Visualization (PyVista)
**Location:** `DEVversion/artnetTest/pointLightController3D.py` (611 lines)
**UI:** PyVista 3D window — interactive 3D scene with panels, trackzone, point light
**Purpose:** First full 3D visualization. Panels rendered as rectangles in 3D space with real-time brightness from a point light. Used PyVista for rendering.

- 3D panel geometry: 12 rectangles with correct positions, angles, normals
- Point light rendered as a sphere
- Trackzone visualized as wireframe box
- Simulated person walking through trackzone
- Wandering behavior for idle animation
- Linear falloff: `brightness = max(0, 1 - distance / falloff_radius)`
- All dimensions in centimeters (accurate to physical install)
- Art-Net output alongside visualization
- Interactive camera orbit/zoom

**Key innovation:** Established the centimeter-based coordinate system and panel geometry used in production.

---

## 10. pointLightController3D_pygame.py — 3D Viz (Pygame/OpenGL)
**Location:** `DEVversion/artnetTest/pointLightController3D_pygame.py` (961 lines)
**UI:** Pygame + OpenGL — 3D view with embedded GUI panel
**Purpose:** Replaced PyVista with Pygame/OpenGL for better compatibility and integrated GUI controls. This is the direct ancestor of the production controller's visualization.

- Same 3D panel model as PyVista version
- Pygame/OpenGL rendering: panels as quads, light as sphere, trackzone as wireframe
- Embedded GUI: sliders for falloff radius, brightness, speed, pulse
- Mouse drag to rotate camera, scroll to zoom
- Keyboard controls: arrow keys move light, P toggles person, SPACE toggles wander
- Simulated person with walk-through animation
- Art-Net output with visualization-only fallback

**Key innovation:** The Pygame/OpenGL rendering approach and embedded slider GUI carried directly into the production `lightController_osc.py`.

---

## 11. waveFieldController.py — Perlin Wave Field
**Location:** `DEVversion/artnetTest/waveFieldController.py` (1,036 lines)
**UI:** Tkinter — wave parameters + ripple simulation + real-time panel display
**Purpose:** Liquid-like always-moving wave field. Unlike springs which settle to equilibrium, this never stops. People inject ripples that spread and interact.

- Custom Perlin noise implementation (1D + 2D)
- Layered wave system: base drift + slow waves + fast shimmer
- Ripple injection: simulated person creates expanding ripple waves
- Ripples have amplitude, speed, decay, and wavelength
- Time-of-day presets: morning (calm), afternoon (active), evening (dramatic), night (ambient)
- Per-panel 3D positions with front/back separation
- Panel brightness = base waves + ripple overlay + noise
- Auto-ripple mode for demonstration
- DMX front (1–50) vs back (1–212) ranges

---

## Evolution Path

```
artnetTest.py          Direct sliders — verify hardware
       │
simpleWaveController   1D wave — first animation
       │
radialGradientController  Radial pulse — lerp transitions
       │
radialPulseController  3D point light — distance falloff  ─────┐
       │                                                        │
vectorController       Directional sweep — vector movement      │
       │                                                        │
springController       Spring physics — force propagation       │
       │                                                        │
springControllerGUI    + interactive GUI                         │
       │                                                        │
springControllerGUI_v2 + tuning + visualization                 │
       │                                                        │
pointLightController3D   3D scene (PyVista) ◄───────────────────┘
       │
pointLightController3D_pygame  3D scene (Pygame/OpenGL)
       │
waveFieldController    Perlin wave field (explored, not adopted)
       │
       ▼
   PRODUCTION (IO/lightController_osc.py)
   Point light model + Pygame/OpenGL viz
   + behavior system from spring concepts
```

## Screenshots
See `/UIs/` folder for interface screenshots of each controller.
