# Focus of the update
This final major update of the software follows a week of observation as to the effects of the self-tuning meta behaviors and pushing the bound of the interaction further based on 1:1 interaction with the system. This update will serve for the last week of the project operating, but will very likely require incremental updates during this time so make these changes in a way that allows that to happen simple. Do your best, but assume it will need to be tuned.
## Primary Changes
### Re-working the Autotuning
While the interaction budget and meta tuner have worked to continuously adjust the values, what has occurred is too much of a revert to the middle. The range of interactions / movements is lower and everything is now too muted.
- Look at the Reports to more clearly see the patterns of interaction over the week
- More heavily weight these times of day
- Have different goals for the budget at different times of day to create more variation

### Clearer and Faster Switch to Engaged visitors
As you can see from the reports, the vast majority of the inputs to this system are people just passing it on the sidewalk in the passive zone. Currently when a person moves into the active zone the system is much too slow to respond, making it unclear that it is working. 
- Clearer and more immediate acknowledgement of a person entering.
- Even if there are others there always acknowledge a change
- Clear progression and intensity over time
- potential new gestures (see ### New Gestures / Movements)
- More active an playful in this mode generally to separate from the others

### New Gestures / Movements
As described in other sections the animations/movement have become too similar and muted. While the goal is still to avoid this becoming just loud blinking lights, the intensity and range does need to be expanded. This isn't about replacing the behaviour system, just expanding it.
- Up the max intensity to 100. Keep in mind that the most visible change range is still 1-50, but use this upper intensity too. 
- More Active engagement of the size / shape of the falloff
- - Expand the use of pulsing the falloff size over time combined with other movements, not just the large gesture that lights all the panels. Consider how this moving gradient can be used in multiple ways/gestures.
- - Scaling the fall-off in X,Y,Z individually (with a main focus on X,Y). The panels are arranged linearly and changing the shape of the falloff object will allow for new kinds of effects.
- - Rotating the Falloff object. This could be used more extensively in engaged mode.
- - The scaling and rotating should have clear inertia to set it back to being round

# Important Notes
- New Features cannot break the basic input / output of the system
- - Consider how it will work with the viewer, socket server, database, reports, etc
- Always consider the smooth animation of the point object

---

# V5 Implementation Log

## Status: Complete — Ready for Live Testing
All changes implemented in `v5Dev/lightController_osc_v5.py` and `v5Dev/light_behavior_v5.py`. Both files pass syntax verification and import checks. Merge to production after live camera testing.

---

## 1. Autotuning Rework

### Time-of-Day Profiles (replaces periodic 6-hour resets)
- Static `home_values` replaced with 6 interpolated time profiles: midnight, 6am, 10am, 2pm, 6pm, 10pm
- New `_get_current_home_values()` method smoothly blends between profiles based on current time
- Each profile shifts personality, brightness, speed, pulse, exploration, etc. to match expected activity patterns
- Morning = energetic/exploratory, Afternoon = sociable, Evening = bright/dramatic, Night = quiet/contemplative

### Reduced Mean Reversion
- `_reversion_base`: 0.02 → 0.01
- `_reversion_progressive`: 0.06 → 0.03
- Output multipliers (brightness, speed, pulse) get **half** the reversion strength via `_output_reversion_scale = 0.5`
- Curiosity nudges now bias toward time-interpolated home values instead of static ones

### Removed
- 6-hourly periodic reset blends (`_reset_hours`, `_reset_blend`, `_last_reset_hour`) — no longer needed with continuous time-of-day interpolation

---

## 2. Faster / Clearer Engaged Mode

### Mode Parameters
- ENGAGED `brightness_max`: 30 → 55
- ENGAGED `brightness_min`: 8 → 12
- ENGAGED `move_speed`: 25 → 40
- ENGAGED `follow_smoothing`: 0.03 → 0.06
- CROWD `brightness_max`: 45 → 80
- CROWD `brightness_min`: 12 → 15

### Faster Transitions
- IDLE → ENGAGED: 0.8s → 0.4s
- FLOW → ENGAGED: 0.8s → 0.4s

### Entry Pulse (first person enters)
- Brightness boost: 25 → 50
- Duration: 0.8s → 1.5s
- Falloff shape contracts to 60% during pulse for focused beam effect

### Re-Entry Pulse (new person while already engaged)
- Smaller acknowledgment: boost 30, duration 0.5s
- Fires when someone enters while already in ENGAGED/CROWD mode

### Phase Transition Pulse
- Micro-pulse when dwell phase changes (greet → engage → bond)
- Boost 10, duration 0.3s

### Dwell Rewards (stronger progression)
- Greet phase: +8 brightness (was +5)
- Engage phase: ramps to +20 (was +10), +3 per 4s (was +2 per 5s)
- Bond phase: +25 brightness (was +15)
- Phase transitions detected via `last_dwell_phase` state field

---

## 3. New Gestures & Movements

### Anisotropic Falloff Shape System
- `PointLight` now has `falloff_scale` (3-axis np.array) and `falloff_rotation` (Y-axis float)
- Spring animation toward targets with ~2s settle time (`falloff_spring_speed = 2.0`)
- Inertia drift back to neutral when no gesture active (`falloff_inertia_speed = 1.5`)
- `PanelSystem.calculate_brightness()` transforms diff vector by inverse rotation (XZ plane) + inverse scale → ellipsoidal falloff
- `brightness_max` default raised to 100

### New Gesture Types
- **SWEEP**: Slow X-axis translation with stretched falloff (`scale_x: 2.5`), dramatic wide beam sweep
- **FOCUS**: No position change, contracts shape (`scale_x: 0.5, scale_z: 0.5`) + brightness boost (+20), intense spotlight effect

### Existing Gesture Falloff Modifiers
- LEAN: `scale_z: 1.5` (stretches toward viewer)
- SWAY: `scale_x: 1.8` (widens side to side)
- ORBIT: `rotation: 0.4` (rotates ellipse while circling)
- SETTLE: `scale_x: 0.7, scale_z: 0.7` (contracts during settle)

### Gesture Timing (tightened intervals)
- Greet phase: 6-12s (was 8-15s)
- Engage phase: 8-16s (was 10-20s)
- Bond phase: 12-25s (was 15-30s)

### Visualization
- `draw_ellipsoid_wireframe()` replaces sphere wireframe in OpenGL render
- Uses `glScalef` and `glRotatef` to show actual falloff shape

### WebSocket / Viewer
- Broadcasts `falloff_scale` (array) and `falloff_rotation` (float) per light in state updates

---

## Files Modified
| File | Lines | Key Changes |
|------|-------|-------------|
| `lightController_osc_v5.py` | ~5550 | PointLight fields, PanelSystem ellipsoid, AutoTuning profiles, main loop application, ellipsoid viz, WS broadcast |
| `light_behavior_v5.py` | ~3760 | SWEEP/FOCUS gestures, MODE_PARAMS, transitions, entry/re-entry/phase pulses, dwell rewards, falloff shape pipeline, status texts |

## Design Decisions
- **Spring-back**: ~2s medium inertia for falloff shape returning to neutral
- **Rotation**: Y-axis only (rotation in XZ ground plane matches linear panel arrangement)
- **Budget approach**: Constant budget with time-varying home values (not traffic-matching budget)
- **Both files copied**: Controller and behavior both in `v5Dev/` for isolated testing

---
---

# Version History — All IO Versions

## V1 — Foundation (Jan 28, 2026)

**Backup:** `v1_backup_20260128_152853/`

The initial production deployment. ~2,600 lines across controller + behavior.

- **4 modes**: IDLE, ENGAGED, CROWD, FLOW
- **10 gestures**: acknowledge, curious, welcome, bored, farewell, surprised, thinking, hesitant, playful, bloom
- **MetaParameters**: 6 personality sliders (responsiveness, energy, attention_span, sociability, exploration, memory) + global multipliers
- **Infrastructure**: pygame/OpenGL 3D viz, Art-Net DMX, WebSocket viewer, OSC input (port 7000), SQLite tracking database
- **Zones**: Active 475×205 cm, passive 650×330 cm
- Camera tracker: YOLO + RTSP, single-instance file locking

| File | Lines |
|------|-------|
| lightController_osc.py | 1,597 |
| light_behavior.py | 1,011 |
| camera_tracker_osc.py | 1,010 |
| pedestrian_simulator.py | 737 |

---

## V2 — Behavior Overhaul & Trend System (Late Jan 2026)

**Folder:** `V2Dev/`

Massive behavior expansion — behavior file tripled in size. Added multi-timescale awareness and a desperation/aggression system.

- **IdleTrends**: Multi-timescale trend data (1min, 5min, 30min, 1hr) with activity anticipation, flow momentum, energy level
- **AggressionState**: "Desperation meter" that rises without engagement, falls on conversion. Time-of-day caps (financial district patterns)
- **TimePeriod enum**: LATE_NIGHT, MORNING, AFTERNOON, EVENING — first time-of-day awareness
- **Dwell phases**: CURIOUS (0–3s) → ENGAGED (3–8s) → REWARDED (8–15s) → DEEP (15s+)
- **Zone-based detection in tracker**: `ZoneChecker` class, zone confidence weighting
- **New files**: `world_coordinates.json` as single source of truth, separate calibration, slider settings
- **Design philosophy**: "Convert passersby into engagers" — anticipatory positioning, peripheral pulse, invitation gestures

| File | Lines |
|------|-------|
| lightController_v2.py | 3,582 |
| light_behavior_v2.py | 3,115 |
| camera_tracker_osc_v2.py | 1,292 |
| v2behavior.md | 874 |

---

## V2.5 — Tracker Refactor (Feb 9, 2026)

**Folder:** `V2_5Dev/`

Focused refactor of the camera tracker for speed and simplicity. No behavior changes.

- **Parameters reduced**: 9 → 3 live + 2 config (confidence, fusion distance, smoothing)
- **Zone sorting removed from tracker**: Delegated entirely to lightController (was dead code)
- **`Tracker` class** replaces monolithic `main()` — structured from ~300 lines
- **Performance fixes**: Eliminated double world-coordinate transform, RTSP buffer flushing via `grab()`, reduced GPU→CPU transfers
- **Cyclist merging removed**: Minimal value for sidewalk installation
- **Result**: ~15–20% faster per frame, codebase reduced to ~700 lines target

| File | Lines |
|------|-------|
| camera_tracker_osc.py | 1,009 |
| CODE_REVIEW.md | formal review |
| TUNING_GUIDE.md | 3-slider reference |
| CALIBRATION_REFERENCE.md | coordinate system reference |

---

## V3 — Full Modular Refactor (Feb 3, 2026)

**Folder:** `V3Dev/`

Complete architectural decomposition of the monolith into 6 subpackages. Not adopted for production.

- **Decomposed ~8,000 lines** into: `config/`, `tracking/`, `behavior/`, `visualization/`, `network/`, `display/`
- **New `Application` class** as central coordinator with `AppState` dataclass
- **CLI entry point**: `run.py` with `--headless`, `--test`, `--verbose` flags
- **Unit tests** added: `test_behavior.py`, `test_network.py`, `test_visualization.py`
- **Mock implementations**: `MockDMXOutput`, `MockWebSocketBroadcaster` for testing
- **OSC port changed**: 7000 → 7777
- **BehaviorMode expanded**: IDLE, ACTIVE, FOLLOWING, AMBIENT, PULSE, WAVE
- **Identified issues**: "Parameter zombie" problem from auto-tuning, zone definitions duplicated in 3+ places, over-engineered chart rendering (~1,200 lines of drawing code)

**Note:** This clean architecture was designed but not adopted. Production continued as the incrementally patched monolith.

---

## V4 — Camera Tracker FPS Optimization (Feb 2026)

**Folder:** `V4Dev/` (top-level, not inside IO/)

Performance-focused tracker development based on V2.5 (not V3's modular approach).

- **Stage 1**: Per-stage timing instrumentation with percentile reporting (`--benchmark-interval 60`)
- **Stage 2**: Batched YOLO inference, vectorized projection, reduced GPU→CPU transfers
- **Same OSC contract**: `/tracker/person/<id>`, `/tracker/count`
- Controller (4,698 lines) imports from production `IO/light_behavior.py` and `IO/tracking_database.py`

| File | Lines |
|------|-------|
| camera_tracker_v4.py | 1,207 |
| lightController_osc.py | 4,698 |

---

## V5 — Final Behavior Expansion (Feb 18–25, 2026)

**Folder:** `v5Dev/`

Final major update for the project's last week. Three primary changes based on a week of observation.

1. **Autotuning rework**: Static home values → 6 time-of-day profiles with smooth interpolation. Reduced mean reversion. Removed periodic 6-hour resets.
2. **Faster engaged mode**: IDLE→ENGAGED 0.8s→0.4s, brightness 30→55, entry pulse boost 25→50 with focused beam, re-entry and phase transition pulses.
3. **Anisotropic falloff**: 3-axis falloff scale + Y-axis rotation with spring animation. Two new gestures (SWEEP, FOCUS). Existing gestures enhanced with falloff modifiers. Max intensity raised to 100.

| File | Lines |
|------|-------|
| lightController_osc_v5.py | ~5,550 |
| light_behavior_v5.py | ~3,760 |

---

## Production (Current IO/ root)

The live deployment files. Structurally a V2 monolith with V3/V4 features layered in — **not** the modular V3 architecture.

| File | Lines | Notes |
|------|-------|-------|
| lightController_osc.py | 5,296 | Docstring says "V3" but structure is V2 monolith with patches |
| light_behavior.py | 3,649 | V2 IdleTrends/AggressionState + accumulated patches |
| tracking_database.py | 1,538 | SQLite with zones, flow analysis, velocity tracking |
| camera_tracker_osc.py | — | Production tracker |
| pedestrian_simulator.py | — | Headless testing tool |

---

## Growth Summary

| Version | Date | Controller | Behavior | Key Innovation |
|---------|------|-----------|----------|----------------|
| V1 | Jan 28 | 1,597 | 1,011 | Foundation — 4 modes, 10 gestures, OSC+Art-Net+WebSocket |
| V2 | Late Jan | 3,582 | 3,115 | Multi-timescale trends, aggression, dwell phases |
| V2.5 | Feb 9 | — | — | Tracker refactor: 9→3 params, 15-20% faster |
| V3 | Feb 3 | 638 (app) | decomposed | Full modular refactor (not adopted for production) |
| V4 | Feb | 4,698 | shared | FPS instrumentation, batched YOLO |
| V5 | Feb 18-25 | ~5,550 | ~3,760 | Anisotropic falloff, time-of-day autotuning, new gestures |
| Prod | Current | 5,296 | 3,649 | V2 monolith + V3/V4 patches; V5 staged for merge |
