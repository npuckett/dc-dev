# World Coordinates Data Format

## Overview

The `world_coordinates.json` file defines all physical objects and their positions in the installation's coordinate system. This file serves as the **single source of truth** for:

- Camera calibration
- Tracking system configuration
- Visualization systems
- Any future components that need spatial awareness

## Coordinate System

| Axis | Direction | Notes |
|------|-----------|-------|
| **X** | Negative = Left | X=0 is right edge of Unit 0, X=-300 is left edge of Unit 3 |
| **Y** | Positive = Up | Y=0 is floor level, Y=-66 is street level |
| **Z** | Positive = Forward | Z=0 is back of panels, Z increases into tracking zone |

**Origin (0, 0, 0)**: Back right corner of Panel Unit 0, at floor level.

```
     Y (up)
     │
     │    
     │   ┌─────────────────────────────────────┐
     │   │  Unit 3    Unit 2    Unit 1   Unit 0│ ← Panels at Z=0
     └───┼─────────────────────────────────────┼────── X (negative = left)
         │           Tracking Zone             │
         │                                     │
         └─────────────────────────────────────┘
                          │
                          Z (forward into space)
```

## File Structure

### `version`
Format version string. Increment when making breaking changes.

### `units`
Measurement units for all coordinates (always "centimeters").

### `coordinate_system`
Human-readable description of axis directions and origin location.

---

### `panels`

Physical light panel units.

```json
{
  "unit_spacing": 80,        // Distance between unit centers
  "panel_width": 60,         // Width of each panel
  "panel_depth": 60,         // Depth of panel structure
  "units": {
    "0": {
      "center": [-30, 0, 0], // 3D center position
      "right_edge": 0,       // X coordinate of right edge
      "left_edge": -60,      // X coordinate of left edge
      "description": "..."
    }
  },
  "subpanels": {
    "local_positions": {     // Position relative to unit center
      "1": {"y": 120, "z": -23, "angle_deg": -30}  // Top panel
    }
  }
}
```

**Panel numbering**: Unit 0 is rightmost, Unit 3 is leftmost.  
**Subpanel numbering**: 1=top, 2=middle, 3=bottom.

---

### `cameras`

Camera positions, orientation, and calibration intrinsics.

```json
{
  "camera_1": {
    "position": [-30, -15, 78],
    "description": "Right camera - aligned with Unit 0 center",
    "coverage": "Right half of tracking zone",
    "mounting": "15cm below floor level, fixed position",
    "rotation": {
      "euler_deg": {"pitch": 18, "yaw": -50, "roll": 0},
      "target_point": [-150, -66, 180]
    },
    "intrinsics": {
      "focal_length_px": [1740.8, 1740.8],
      "principal_point": [1024, 768],
      "image_size": [2048, 1536],
      "dist_coeffs": [0, 0, 0, 0, 0]
    },
    "fov": {"horizontal": 80, "vertical": 48}
  }
}
```

#### Rotation Fields

| Field | Description |
|-------|-------------|
| `euler_deg.pitch` | Tilt angle in degrees (positive = down toward street) |
| `euler_deg.yaw` | Horizontal rotation (negative = left, positive = right) |
| `euler_deg.roll` | Tilt around optical axis (0 = level) |
| `target_point` | World coordinate the camera aims at |

#### Intrinsics Fields (for OpenCV)

| Field | Description |
|-------|-------------|
| `focal_length_px` | [fx, fy] focal length in pixels |
| `principal_point` | [cx, cy] image center in pixels |
| `image_size` | [width, height] of camera image |
| `dist_coeffs` | Lens distortion coefficients [k1, k2, p1, p2, k3] |

---

### `calibration_markers`

ArUco markers used for camera calibration.

```json
{
  "marker_size": 15,
  "marker_type": "ArUco",
  "corner_offsets": {
    "description": "Local corner positions relative to marker center",
    "horizontal": {
      "note": "Marker lying flat, facing up (normal = +Y)",
      "corners": [
        [-7.5, 0, -7.5],
        [7.5, 0, -7.5],
        [7.5, 0, 7.5],
        [-7.5, 0, 7.5]
      ],
      "normal": [0, 1, 0]
    },
    "vertical": {
      "note": "Marker on wall, facing toward cameras (normal = -Z)",
      "corners": [
        [-7.5, 7.5, 0],
        [7.5, 7.5, 0],
        [7.5, -7.5, 0],
        [-7.5, -7.5, 0]
      ],
      "normal": [0, 0, -1]
    }
  },
  "markers": {
    "0": {
      "position": [-30, -66, 168],
      "orientation": "horizontal",
      "description": "Right front",
      "visible_to": ["camera_1"]
    }
  }
}
```

#### Marker Corner Order

ArUco markers have 4 corners detected in a consistent order. The `corner_offsets` define local offsets from the marker center to each corner:

```
      Corner 0          Corner 1
         ┌──────────────────┐
         │                  │
         │     MARKER       │
         │     CENTER       │
         │                  │
         └──────────────────┘
      Corner 3          Corner 2
```

#### Computing World Corner Positions

To get the world coordinates of marker corners for calibration:

```python
marker = world['calibration_markers']['markers']['0']
center = marker['position']  # [-30, -66, 168]
orientation = marker['orientation']  # "horizontal"
offsets = world['calibration_markers']['corner_offsets'][orientation]['corners']

corners_world = []
for offset in offsets:
    corner = [center[i] + offset[i] for i in range(3)]
    corners_world.append(corner)
```

**Marker ID convention**:
- 0, 1, 2: Front row (left to right when facing panels)
- 3, 4: Back row outer markers
- 5: Vertical marker on far wall
- 6: Back row center marker

---

### `tracking_zones`

Defined tracking areas.

```json
{
  "active": {
    "description": "...",
    "bounds": {
      "x": [-387.5, 87.5],    // [min, max]
      "y": [-66, 234],
      "z": [78, 283]
    },
    "dimensions": {"width": 475, "height": 300, "depth": 205},
    "center": [-150, -66, 180.5]
  }
}
```

---

### `light_behavior`

Constraints for light movement.

```json
{
  "wander_box": {
    "bounds": {
      "x": [-280, -20],
      "y": [0, 150],
      "z": [-28, 32]
    }
  }
}
```

---

### `reference_levels`

Named Y-axis reference heights.

| Name | Y Value | Description |
|------|---------|-------------|
| `floor` | 0 | Storefront floor level |
| `street` | -66 | Street/sidewalk level |
| `camera_ledge` | -15 | Camera mounting height |

---

## Adding New Objects

When adding new objects to the coordinate system:

1. **Choose the appropriate section** or create a new top-level key
2. **Use consistent position format**: `[X, Y, Z]` array in centimeters
3. **Include a description** for human readability
4. **Add relevant metadata** (visibility, orientation, etc.)
5. **Update version** if making structural changes

### Example: Adding a New Sensor

```json
{
  "sensors": {
    "motion_sensor_1": {
      "position": [-150, 100, 50],
      "type": "PIR",
      "detection_cone": {
        "direction": [0, 0, 1],
        "angle_deg": 45
      },
      "description": "Ceiling-mounted motion sensor"
    }
  }
}
```

---

## Usage in Code

### Python

```python
import json

with open('world_coordinates.json', 'r') as f:
    world = json.load(f)

# Get camera position
cam1_pos = world['cameras']['camera_1']['position']

# Get marker positions for calibration
markers = world['calibration_markers']['markers']
for marker_id, data in markers.items():
    pos = data['position']
    print(f"Marker {marker_id}: {pos}")
```

### Calibration System

The calibration system should:

1. Load `world_coordinates.json`
2. Use `calibration_markers` as ground truth positions
3. Match detected ArUco markers to known positions
4. Compute camera intrinsics/extrinsics
5. Transform pixel coordinates → world coordinates using this reference

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.1 | 2026-01-27 | Added camera intrinsics, rotation data, marker corner offsets |
| 2.0 | 2026-01-26 | New origin at back right corner of Unit 0 |
