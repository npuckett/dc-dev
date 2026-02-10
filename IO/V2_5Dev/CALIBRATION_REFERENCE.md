# Calibration Reference — Cameras, Markers & Coordinate System

Cross-referenced from `camera_calibration.py`, `lightController_osc.py`, and `world_coordinates.json`.

---

## Coordinate System

| Axis | Direction | Notes |
|------|-----------|-------|
| **Origin (0,0,0)** | Back right corner of Panel Unit 0, at floor level | Where the rightmost panel meets the storefront floor |
| **X** | Negative = left (toward Unit 3) | Panels span X = 0 to X = −300 |
| **Y** | Positive = up | Floor = 0, street is below |
| **Z** | Positive = forward (away from panels, into tracking zone) | Cameras face this direction |

### Reference Y Levels

| Level | Y (cm) | Description |
|-------|--------|-------------|
| Storefront floor | **0** | Origin reference plane |
| Camera mount | **−15** | 15 cm below floor (see discrepancy note below) |
| Street / sidewalk | **−66** | Where people walk, 66 cm below floor |

---

## Camera Positions

Both cameras are mounted at the front edge of the active tracking zone (Z = 78), looking outward into the street.

| | Camera 1 (RIGHT) | Camera 2 (LEFT) |
|---|---|---|
| **Position** | (−30, −15, 78) | (−270, −15, 78) |
| **Aligned with** | Unit 0 center | Unit 3 center |
| **Color in viz** | Red | Blue |
| **IP** | 10.42.0.75 | 10.42.0.172 |
| **Model** | Reolink RLC-520A | Reolink RLC-520A |
| **FOV** | 80° H × 48° V | 80° H × 48° V |
| **Looks toward** | (−150, −66, 180) — center of active zone | (−150, −66, 180) — center of active zone |

### Camera Angles (Euler XYZ, degrees)

| | Pitch (tilt down) | Yaw (horizontal rotation) | Roll |
|---|---|---|---|
| **Camera 1** | 22° down | −25° (angled left toward center) | 0° |
| **Camera 2** | 22° down | +25° (angled right toward center) | 0° |

Both cameras aim inward to create an overlap zone around X = −100 to X = −200 for stereo matching. All files now agree on pitch = 22° and yaw = ±25°.

---

## Calibration Markers

7 ArUco markers (DICT_4X4_50), each 15 cm × 15 cm.

### Marker Map (bird's eye, looking down from above)

```
           X=-270        X=-150        X=-30
            (Left)       (Center)      (Right)
             |              |             |
  Z=168  ---[2]----------[1]----------[0]---   Front row
             |              |             |
  Z=219  ---[4]----------[6]----------[3]---   Back row
             |              |             |
             |              |             |
  Z=628     ...............[5]..............    Subway wall (VERTICAL)
```

### Marker Positions

| ID | Position (X, Y, Z) | Orientation | Description | Visible to |
|----|---------------------|-------------|-------------|------------|
| 0 | (−30, −66, 168) | Flat on ground | Right front | Camera 1 only |
| 1 | (−150, −66, 168) | Flat on ground | Center front (**SHARED**) | Both cameras |
| 2 | (−270, −66, 168) | Flat on ground | Left front | Camera 2 only |
| 3 | (−30, −66, 219) | Flat on ground | Right back | Camera 1 only |
| 4 | (−270, −66, 219) | Flat on ground | Left back | Camera 2 only |
| 5 | (−150, −15, 628) | **Vertical** on wall | Subway wall (~550 cm from cameras) | Both cameras |
| 6 | (−150, −66, 219) | Flat on ground | Center back (**SHARED**) | Both cameras |

- Front row is 90 cm forward from the tracking zone edge (Z = 78 + 90 = 168)
- Back row is 51 cm behind front row (Z = 168 + 51 = 219)
- Marker 5 is on the subway wall, ~550 cm from camera line (Z = 78 + 550 = 628)

### Per-Camera Visibility

| Camera 1 (Right) | Camera 2 (Left) |
|---|---|
| 0, 1, 3, 5, 6 | 1, 2, 4, 5, 6 |

Shared markers: **1, 5, 6** — these provide the common reference frame between cameras. Minimum 3 markers required per camera for solvePnP.

---

## Tracking Zones

Defined in `lightController_osc.py` (the controller owns zone classification, not the tracker):

### Active Zone (people engaging with the installation)

| Dimension | Value | Computed bounds |
|-----------|-------|-----------------|
| Width | 260 cm | X: −280 to −20 |
| Depth | 205 cm | Z: 78 to 283 |
| Height | 300 cm | Y: −66 to 234 |
| Center X | −150 | Centered on panels |

### Passive Zone (sidewalk passersby)

| Dimension | Value | Computed bounds |
|-----------|-------|-----------------|
| Width | 400 cm | X: −350 to 50 |
| Depth | 270 cm | Z: 283 to 553 |
| Height | 300 cm | Y: −66 to 234 |

---

## Discrepancies — RESOLVED

All four discrepancies identified during the initial cross-file audit have been fixed:

| # | Issue | Fix Applied |
|---|-------|-------------|
| 1 | `CAMERA_LEDGE_Y` was −16 in both `.py` files | Changed to **−15** in `camera_calibration.py` and `lightController_osc.py` |
| 2 | Marker 5 Z was 578 in `lightController_osc.py` and `world_coordinates.json` | Changed to **628** (matching the `camera_calibration.py` fallback and physical measurement). Distance updated 500→550 cm. |
| 3 | Tracking zone bounds in `world_coordinates.json` were from an old coordinate convention | Updated to match `lightController_osc.py`: active X [−280,−20], Z [78,283]; passive X [−350,50], Z [283,553] |
| 4 | Camera pitch was 10° in `world_coordinates.json` | Changed to **22°** to match `lightController_osc.py` |

All three files (`camera_calibration.py`, `lightController_osc.py`, `world_coordinates.json`) are now in agreement on all coordinate values.
