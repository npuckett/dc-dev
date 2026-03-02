# Camera Angle Configuration

## Reolink RLC-520A Specifications
- **Horizontal FOV:** 80°
- **Vertical FOV:** 48°
- **Resolution:** 5MP (2560×1920)
- **Lens:** f=4.0mm

## Camera Positions (Fixed)
| Camera | Position (X, Y, Z) | Description |
|--------|-------------------|-------------|
| Camera 1 | (-30, -15, 78) | Right camera, aligned with Unit 0 |
| Camera 2 | (-270, -15, 78) | Left camera, aligned with Unit 3 |

## Marker Visibility
| Camera | Visible Markers |
|--------|-----------------|
| Camera 1 | 0 (red), 1 (green), 3 (yellow), 5 (cyan), 6 (orange) |
| Camera 2 | 1 (green), 2 (blue), 4 (magenta), 5 (cyan), 6 (orange) |

**Shared markers:** 1, 5, 6 (visible to both cameras)

---

## Camera 1 (Right Camera) Rotation

| Axis | Angle | Direction | Description |
|------|-------|-----------|-------------|
| **Pitch (X)** | **22°** | Down | Tilts camera down for better ground coverage |
| **Yaw (Y)** | **-25°** | Left | Rotates camera 25° left toward center (sees marker 0 on right edge) |
| **Roll (Z)** | **0°** | Level | No tilt, horizon stays level |

### Euler Angles (XYZ order): `(22, -25, 0)`

**Physical Setup:** Mount camera level, then:
1. Rotate the camera 25° to the LEFT (looking from behind)
2. Tilt it DOWN by 22°

---

## Camera 2 (Left Camera) Rotation

| Axis | Angle | Direction | Description |
|------|-------|-----------|-------------|
| **Pitch (X)** | **22°** | Down | Tilts camera down for better ground coverage |
| **Yaw (Y)** | **+25°** | Right | Rotates camera 25° right toward center (sees marker 2 on left edge) |
| **Roll (Z)** | **0°** | Level | No tilt, horizon stays level |

### Euler Angles (XYZ order): `(22, +25, 0)`

**Physical Setup:** Mount camera level, then:
1. Rotate the camera 25° to the RIGHT (looking from behind)
2. Tilt it DOWN by 22°

---

## Why These Angles?

The angles were optimized to ensure each camera can see:
1. **Ground-level markers** (1, 6) - requires steeper pitch (22°) for better floor coverage
2. **Its own side marker** (0 for Camera 1, 2 for Camera 2) - requires 25° yaw
3. **The center markers** (1, 6) - shared by both cameras
4. **Overlap in the center** for stereo depth calculation

With 80° horizontal FOV and 25° yaw, each camera covers from its own side to past center.
With 22° pitch, cameras are angled down enough to track people on the ground while still seeing the tracking zone.

---

## Angle Calculations

### Yaw (Y-axis rotation) - Horizontal aiming
```
Camera 1: X=-30, Target X=-150
  Δx = -150 - (-30) = -120cm (left)
  Δz = 180 - 78 = 102cm (forward)
  Yaw = atan2(-120, 102) = -50°
  
Camera 2: X=-270, Target X=-150
  Δx = -150 - (-270) = +120cm (right)
  Δz = 180 - 78 = 102cm (forward)
  Yaw = atan2(120, 102) = +50°
```

### Pitch (X-axis rotation) - Vertical aiming
```
Camera Y = -15, Target Y = -66
  Δy = -66 - (-15) = -51cm (down)
  Δz = 180 - 78 = 102cm (forward)
  Distance (horizontal) = sqrt(120² + 102²) = 157cm
  Pitch = atan2(-51, 157) = -18°
```

---

## Coverage Analysis

With 80° HFOV and 50° yaw angle:
- Each camera covers 40° on each side of its aim direction
- Camera 1 covers from -90° to -10° (relative to forward)
- Camera 2 covers from +10° to +90° (relative to forward)
- **Overlap zone:** Both cameras see X=-200 to X=-100 (center 100cm)

### Active Zone Coverage
- Active zone width: 260cm (X: -280 to -20)
- With angled cameras: Full coverage with 100cm stereo overlap
- Stereo depth calculation reliable in overlap zone

---

## Implementation Notes

1. **Calibration markers** should be visible from both cameras' angled perspectives
2. **ArUco detection** works best when markers are within ±45° of camera optical axis
3. **Marker 5** at Z=578 is at the edge of reliable detection (~500cm distance)
4. Consider adding markers at intermediate Z positions for better depth calibration
