# Calibration System Audit - V2 Coordinate System

**Date:** January 26, 2026  
**Scope:** Audit of `camera_tracker_cuda_v2.py` against the V2 coordinate system defined in `world_coordinates.json` and `lightController_v2.py`

---

## Executive Summary

The current calibration code (`camera_tracker_cuda_v2.py`) uses an **OLD coordinate system** that is fundamentally incompatible with the V2 coordinate system. There are significant mismatches in:

1. **Origin location** - Old system uses different origin than V2
2. **Marker positions** - All marker coordinates are wrong
3. **Camera position expectations** - Initial guesses use old coordinate assumptions
4. **Synthesized view rendering** - Pixel offsets don't match new origin

**Severity: HIGH** - The calibration will produce completely incorrect world coordinates.

---

## Coordinate System Comparison

### V2 System (CORRECT - from `world_coordinates.json`)

| Property | Value |
|----------|-------|
| **Origin (0,0,0)** | Back right corner of Panel Unit 0, at floor level |
| **X-axis** | Negative toward Unit 3 (left when facing panels) |
| **Y-axis** | Positive upward |
| **Z-axis** | Positive forward into tracking zone (away from panels) |
| **Floor level** | Y = 0 |
| **Street level** | Y = -66 |
| **Camera height** | Y = -15 |

### OLD System (in `camera_tracker_cuda_v2.py`)

| Property | Value | Issue |
|----------|-------|-------|
| **Origin** | Appears to be left of panels | **WRONG** - should be right edge of Panel 0 |
| **X-axis** | Positive toward right | **INVERTED** from V2 |
| **Panel range** | X = 0 to X = 240 | **WRONG** - V2 uses X = 0 to X = -300 |
| **Marker X coords** | X = -40 to X = 280 | **WRONG** - completely different coordinate system |

---

## Detailed Findings

### 1. CRITICAL: Marker World Positions Mismatch

The hardcoded marker positions in `CalibrationMode.marker_world_positions_3d` (lines 780-800) are completely wrong:

| Marker | OLD Position (in code) | V2 CORRECT Position | Delta |
|--------|------------------------|---------------------|-------|
| 0 | (-40, -66, 90) | **(-30, -66, 168)** | X: +10, Z: -78 |
| 1 | (120, -66, 90) | **(-150, -66, 168)** | X: -270, Z: -78 |
| 2 | (280, -66, 90) | **(-270, -66, 168)** | X: -550, Z: -78 |
| 3 | (-40, -66, 141) | **(-30, -66, 219)** | X: +10, Z: -78 |
| 4 | (280, -66, 141) | **(-270, -66, 219)** | X: -550, Z: -78 |
| 5 | (120, -16, 550) | **(-150, -15, 628)** | X: -270, Y: +1, Z: -78 |
| 6 | (120, -66, 141) | **(-150, -66, 219)** | X: -270, Z: -78 |

**Key Issues:**
- The X-axis is essentially flipped (positive vs negative direction)
- The Z values are shifted by ~78cm (the Z-offset of the tracking zone)
- The old code treats X=0 as the left side; V2 treats X=0 as the right side

### 2. CRITICAL: Camera Position Initial Guess

In `compute_3d_pose()` (lines 960-970), the initial camera position guess uses wrong coordinates:

```python
# OLD (WRONG):
if 'Camera 1' in camera_name:
    init_camera_pos = np.array([0.0, -16.0, 0.0])    # Camera 1 at X=0
    init_target = np.array([120.0, -66.0, 200.0])    # Looking at X=120
else:
    init_camera_pos = np.array([240.0, -16.0, 0.0])  # Camera 2 at X=240
    init_target = np.array([120.0, -66.0, 200.0])
```

**V2 CORRECT:**
```python
# Camera 1 (Right camera) at X=-30, Z=78
init_camera_pos = np.array([-30.0, -15.0, 78.0])
# Camera 2 (Left camera) at X=-270, Z=78
init_camera_pos = np.array([-270.0, -15.0, 78.0])
```

### 3. Synthesized View Rendering Offset

In `world_to_synth_pixels()` (lines 367-377), the pixel offset is wrong:

```python
# OLD (WRONG):
offset_x = 150  # pixels from left edge for X=0
offset_y = 50   # pixels from top edge for Z=0
```

With V2 coordinates (X from 0 to -300, Z from 78 to 600+), this will:
- Place X=0 (right edge of panels) at pixel 150
- Place all panels at negative pixel values (off-screen left)
- Z=0 would be above the panels, not at the tracking zone

### 4. Camera Assignment Comments (Misleading)

The comments describe Camera 1 as "left" and Camera 2 as "right" in several places:

```python
# Lines 752-753:
# - Camera 1 sees: Marker 0, 1, 3 (left + shared)
# - Camera 2 sees: Marker 1, 2, 4 (right + shared)
```

But in the V2 system:
- **Camera 1 is RIGHT** (at X=-30, aligned with Unit 0)
- **Camera 2 is LEFT** (at X=-270, aligned with Unit 3)

### 5. Floor Plan Grid Generation

The `load_or_create_background()` method (lines 450-470) draws a grid assuming:
- Panels at X=0 to X=240
- Z=0 at the panel line

But V2 uses:
- Panels at X=0 to X=-300
- Z=0 at the panel back edge, tracking zone at Z=78+

### 6. Validation Logic Incorrectly Configured

The code validates that camera Z should be positive (lines 1073-1075):

```python
if camera_pos[2] < 0:
    print("⚠️ WARNING: Camera Z still negative...")
```

In V2, cameras ARE at positive Z (Z=78), so this is correct, but the expectation comment says "cameras at Z=0 looking toward positive Z" which contradicts the actual V2 camera Z position.

---

## Marker Placement Analysis

### Current Physical Layout (from V2)

```
                        PANELS (at Z=0)
        Unit 3         Unit 2         Unit 1         Unit 0
    ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
    │  X=-270  │   │  X=-190  │   │  X=-110  │   │  X=-30   │  ← ORIGIN (0,0,0)
    └──────────┘   └──────────┘   └──────────┘   └──────────┘     at right corner
         ↓              ↓              ↓              ↓
       CAM 2         (center)       (center)        CAM 1      ← Z = 78 (camera line)
         
    ═══════════════════════════════════════════════════════════  Z = 78 (front of active zone)
         │                                              │
         │              ACTIVE TRACKING ZONE            │
         │                                              │
    ─────────────────────────────────────────────────────────  Z = 168 (front marker row)
       [M2]           [M1-SHARED]                     [M0]
         │                                              │
    ─────────────────────────────────────────────────────────  Z = 219 (back marker row)
       [M4]           [M6-SHARED]                     [M3]
         │                                              │
    ═══════════════════════════════════════════════════════════  Z = 283 (back of active zone)
         │                                              │
         │              PASSIVE TRACKING ZONE           │
         │                                              │
         │                                              │
    ─────────────────────────────────────────────────────────  Z ≈ 628
                      [M5-VERTICAL]
                    (on subway wall)
```

### Marker Visibility

| Marker | Position (X, Y, Z) | Visible To | Purpose |
|--------|-------------------|------------|---------|
| 0 | (-30, -66, 168) | Camera 1 only | Right front reference |
| 1 | (-150, -66, 168) | **BOTH** | **SHARED center front** |
| 2 | (-270, -66, 168) | Camera 2 only | Left front reference |
| 3 | (-30, -66, 219) | Camera 1 only | Right back (depth) |
| 4 | (-270, -66, 219) | Camera 2 only | Left back (depth) |
| 5 | (-150, -15, 628) | **BOTH** | **SHARED vertical (far)** |
| 6 | (-150, -66, 219) | **BOTH** | **SHARED center back** |

### Potential Marker Placement Issues

1. **Marker 5 Distance:** At Z=628, this is very far from the cameras (550cm from camera line). This may cause:
   - Lower detection reliability
   - Higher measurement error due to distance
   - Potential occlusion by people in tracking zone

2. **Vertical Marker Orientation:** Marker 5 is vertical (on a wall). The current code handles this with a `vertical_markers` set, but the corner calculation must account for the orientation.

3. **Limited View Overlap:** Markers 0, 3 are only visible to Camera 1; Markers 2, 4 are only visible to Camera 2. This means:
   - Each camera relies on only 4-5 markers
   - If SHARED marker 1 is occluded, calibration fails
   - No redundancy for the outer markers

4. **Coplanarity Issues:** Markers 0-4 and 6 are all on the same Y plane (Y=-66). For solvePnP:
   - Coplanar points can cause pose ambiguity (two valid solutions)
   - Marker 5 at Y=-15 breaks coplanarity, which helps
   - Current code correctly uses IPPE for coplanar cases

---

## Strategy for V2 Calibration

### Phase 1: Update Coordinate System (Immediate)

1. **Update `marker_world_positions_3d`** to match `world_coordinates.json`:
   ```python
   self.marker_world_positions_3d = {
       0: (-30.0, STREET_LEVEL_Y, 168.0),   # Right front
       1: (-150.0, STREET_LEVEL_Y, 168.0),  # Center front - SHARED
       2: (-270.0, STREET_LEVEL_Y, 168.0),  # Left front
       3: (-30.0, STREET_LEVEL_Y, 219.0),   # Right back
       4: (-270.0, STREET_LEVEL_Y, 219.0),  # Left back
       5: (-150.0, CAMERA_LEDGE_Y, 628.0),  # Subway wall - SHARED
       6: (-150.0, STREET_LEVEL_Y, 219.0),  # Center back - SHARED
   }
   ```

2. **Update camera visibility mapping**:
   ```python
   self.camera_marker_visibility = {
       'Camera 1': [0, 1, 3, 5, 6],  # Right camera (X=-30) sees right + shared
       'Camera 2': [1, 2, 4, 5, 6],  # Left camera (X=-270) sees left + shared
   }
   ```

3. **Update initial camera position guess**:
   ```python
   if 'Camera 1' in camera_name:
       init_camera_pos = np.array([-30.0, -15.0, 78.0])
       init_target = np.array([-150.0, -66.0, 200.0])
   else:
       init_camera_pos = np.array([-270.0, -15.0, 78.0])
       init_target = np.array([-150.0, -66.0, 200.0])
   ```

4. **Update synthesized view offsets**:
   ```python
   def world_to_synth_pixels(self, world_x, world_z):
       # V2 coordinates: X from 0 to -300, Z from 78 to 600+
       # Put X=0 on the RIGHT side of the view
       offset_x = self.width - 50  # X=0 at right edge
       offset_y = 10   # Z=78 near top
       
       # X goes negative (left), so subtract
       px = int(offset_x + world_x / SYNTH_CM_PER_PIXEL)
       # Z goes positive (down in bird's eye)
       py = int(offset_y + (world_z - 78) / SYNTH_CM_PER_PIXEL)
       return px, py
   ```

### Phase 2: Load Coordinates from JSON (Recommended)

Instead of hardcoding, load marker positions from `world_coordinates.json`:

```python
def load_marker_positions(self):
    """Load marker positions from world_coordinates.json"""
    json_path = os.path.join(os.path.dirname(__file__), 'world_coordinates.json')
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    self.marker_world_positions_3d = {}
    for marker_id, marker_data in data['calibration_markers']['markers'].items():
        pos = marker_data['position']
        self.marker_world_positions_3d[int(marker_id)] = tuple(pos)
    
    self.marker_size_cm = data['calibration_markers']['marker_size']
```

This ensures calibration and visualization always use the same coordinates.

### Phase 3: Improve Calibration Quality

1. **Add more shared markers:** Currently only markers 1, 5, 6 are shared. Consider adding a fourth shared marker in the center of the passive zone for better overlap validation.

2. **Use marker corners for solvePnP:** The current code uses only marker centers. Using all 4 corners per marker gives 4× more correspondences and handles marker rotation better.

3. **Add reprojection visualization:** After calibration, draw projected marker positions on the camera view to visually verify accuracy.

4. **Implement cross-validation:** After calibrating both cameras, project the same 3D point through both and verify they converge.

### Phase 4: Marker Placement Optimization

Consider adjusting marker placement:

| Current | Proposed | Reason |
|---------|----------|--------|
| Front row at Z=168 | Keep | Good distance from cameras |
| Back row at Z=219 | Keep | Provides depth reference |
| Marker 5 at Z=628 | Move to Z=400 | Closer = better detection |
| All floor markers flat | Add 2 vertical markers on panels | Break coplanarity further |

---

## Implementation Checklist

### Immediate Fixes (Required)

- [ ] Update `marker_world_positions_3d` to V2 coordinates
- [ ] Update `camera_marker_visibility` with correct camera names
- [ ] Update initial camera guess positions
- [ ] Fix `world_to_synth_pixels()` offsets
- [ ] Update grid drawing in `load_or_create_background()`
- [ ] Update comments about camera left/right assignments
- [ ] Update validation logic for camera Z position

### Recommended Enhancements

- [ ] Load marker positions from `world_coordinates.json`
- [ ] Add reprojection error visualization
- [ ] Implement calibration cross-validation
- [ ] Add calibration quality metrics to saved file
- [ ] Create visual debugging mode showing marker positions in 3D

### Optional Improvements

- [ ] Add additional shared marker in passive zone
- [ ] Add vertical markers on panel faces
- [ ] Implement automatic marker detection quality scoring
- [ ] Add calibration history/versioning

---

## Conclusion

The calibration system requires significant updates to work with the V2 coordinate system. The primary issues are:

1. **All marker positions are wrong** - different origin and axis direction
2. **Camera assignments are labeled backwards** - Camera 1 is RIGHT, not left
3. **Synthesized view rendering is incompatible** - offsets assume old coordinate system
4. **Initial pose guess will fail** - uses completely wrong camera positions

The recommended approach is to:
1. First update all hardcoded coordinates to match V2
2. Then refactor to load from `world_coordinates.json`
3. Finally, add validation and visualization to verify calibration quality

This will establish a single source of truth for coordinates and ensure consistency between calibration and visualization.

---

## Appendix: Camera Specifications & Placement Optimization

### Reolink RLC-520A Specifications

| Property | Value |
|----------|-------|
| **Lens** | f=4.0mm fixed, F=2.0 |
| **Horizontal FOV** | 80° |
| **Vertical FOV** | 48° |
| **Resolution** | 2560 × 1920 (5MP) at 30fps |
| **Aspect Ratio** | 4:3 |
| **Night Vision** | 30m (100ft) with 18 IR LEDs |
| **Sensor** | 1/2.7" CMOS |

### FOV Coverage Calculations

With an 80° horizontal FOV, the coverage width at a given distance is:

```
Width = 2 × Distance × tan(FOV/2)
Width = 2 × Distance × tan(40°)
Width ≈ 1.68 × Distance
```

| Distance from Camera | Coverage Width | Notes |
|---------------------|----------------|-------|
| 100cm (1m) | 168cm | Too narrow |
| 200cm (2m) | 336cm | Covers active zone front |
| 300cm (3m) | 504cm | Good for active zone |
| 400cm (4m) | 672cm | Covers passive zone |
| 550cm (5.5m) | 924cm | Maximum useful range |

### Current Camera Position Analysis

**Current Setup (from world_coordinates.json):**
- Camera 1 (RIGHT): Position (-30, -15, 78) - At Z=78, the front edge of active zone
- Camera 2 (LEFT): Position (-270, -15, 78) - At Z=78, the front edge of active zone
- Cameras are 240cm apart (X = -30 to X = -270)

**Problem: Cameras at Z=78 (front of active zone)**

If cameras are AT the front of the active zone (Z=78), they're looking INTO the zone, which means:
- Active zone (Z=78 to Z=283) is 205cm deep → width coverage ~345cm at the back
- Passive zone starts at Z=283, extends to Z=613 (330cm deep)
- At Z=613 (far edge of passive), coverage width would be ~900cm

**Issue:** With cameras pointing INTO the tracking zones, subjects close to the cameras will be very large in frame (making tracking difficult) and the horizontal coverage overlaps incorrectly.

### Recommended Camera Repositioning

**Option A: Move Cameras Forward (Recommended)**

Move cameras to Z=0 (at the panel line) pointing toward the street:

```
Camera positions: Z = 0 (at panels)
                      ↓
                  PANELS (Z=0)
    ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
    │ Unit 3   │   │ Unit 2   │   │ Unit 1   │   │ Unit 0   │
    └────┬─────┘   └──────────┘   └──────────┘   └────┬─────┘
       CAM 2                                        CAM 1
       X=-270                                       X=-30
         ↘                                           ↙
           ═══════════════════════════════════════════  Z=78 (active front)
                         ACTIVE ZONE
           ───────────────────────────────────────────  Z=168 (marker row)
           ───────────────────────────────────────────  Z=219 (marker row)  
           ═══════════════════════════════════════════  Z=283 (active back)
                         PASSIVE ZONE
           ───────────────────────────────────────────  Z=613 (passive back)
```

**Coverage from Z=0:**

| Zone | Distance | Width at Edge | Panels Covered |
|------|----------|---------------|----------------|
| Active Front (Z=78) | 78cm | 131cm | ~1.6 units |
| Marker Row 1 (Z=168) | 168cm | 282cm | ~3.5 units |
| Active Back (Z=283) | 283cm | 475cm | All 4 units |
| Passive Back (Z=613) | 613cm | 1030cm | All + extra |

**Overlap Zone:** With 80° FOV, both cameras would overlap significantly in the center, providing:
- Stereo triangulation for better depth estimation
- Redundancy if one camera loses a target
- Better accuracy in the shared zone (X = -90 to X = -210)

### Optimized Tracking Zone Dimensions

Given camera coverage, I recommend **narrowing the active zone** for better accuracy:

**Current Active Zone:**
- X: -387.5 to +87.5 (475cm wide) ← Too wide!
- Z: 78 to 283 (205cm deep)

**Recommended Active Zone:**
- X: **-280 to -20** (260cm wide) - Matches panel spread with margin
- Z: 78 to 283 (205cm deep) - Keep depth
- This ensures both cameras have good coverage of the entire zone

**Passive Zone (can remain wider):**
- X: -350 to +50 (400cm wide)
- Z: 283 to 550 (267cm deep) - Slightly shorter for better detection

### Updated world_coordinates.json Tracking Zones

```json
"tracking_zones": {
  "active": {
    "description": "Primary tracking - people engaging with installation",
    "bounds": {
      "x": [-280, -20],
      "y": [-66, 234],
      "z": [78, 283]
    },
    "dimensions": {"width": 260, "height": 300, "depth": 205}
  },
  "passive": {
    "description": "Secondary tracking - people passing by",
    "bounds": {
      "x": [-350, 50],
      "y": [-66, 234],
      "z": [283, 550]
    },
    "dimensions": {"width": 400, "height": 300, "depth": 267}
  }
}
```

### Marker Placement Optimization

**Current Marker Issues:**

1. **Marker 5 too far (Z=628):** At 6+ meters, detection will be unreliable
2. **No markers at active zone boundaries:** Hard to verify edge accuracy
3. **All floor markers at same Y:** Coplanarity can cause solvePnP ambiguity

**Recommended Marker Layout:**

```
                    PANELS (Z=0)
    ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
    │ Unit 3   │   │ Unit 2   │   │ Unit 1   │   │ Unit 0   │  ← ORIGIN (0,0,0)
    └──────────┘   └──────────┘   └──────────┘   └──────────┘
         │              │              │              │
       CAM 2                                        CAM 1      ← Z = 0 (recommended)
         │                                            │
    ═════╪════════════════════════════════════════════╪═════  Z = 78 (active front)
         │                                            │
    ─────┼───────────[M7]──────────────[M8]───────────┼─────  Z = 120 (NEW marker row)
         │                                            │
    ─────[M2]──────────[M1]──────────────────────────[M0]───  Z = 168 (front marker row)
         │                                            │
    ─────[M4]──────────[M6]──────────────────────────[M3]───  Z = 219 (back marker row)
         │                                            │
    ═════╪════════════════════════════════════════════╪═════  Z = 283 (active back)
         │                                            │
         │              PASSIVE ZONE                  │
         │                                            │
    ─────────────────────[M5]─────────────────────────────   Z = 400 (moved closer!)
                       (vertical, on wall)
```

**New Marker Positions:**

| Marker | Position (X, Y, Z) | Purpose | Visible To |
|--------|-------------------|---------|------------|
| 0 | (-30, -66, 168) | Right front | Cam 1 |
| 1 | (-150, -66, 168) | Center front (SHARED) | Both |
| 2 | (-270, -66, 168) | Left front | Cam 2 |
| 3 | (-30, -66, 219) | Right back | Cam 1 |
| 4 | (-270, -66, 219) | Left back | Cam 2 |
| 5 | (-150, -15, **400**) | Vertical on wall (moved closer!) | Both |
| 6 | (-150, -66, 219) | Center back (SHARED) | Both |
| 7* | (-200, -66, 120) | NEW: Left-center near active front | Both |
| 8* | (-100, -66, 120) | NEW: Right-center near active front | Both |

*Markers 7 and 8 are optional additions for improved coverage

### Camera Angle Recommendations

With cameras at Z=0, pointing toward the street:

**Camera 1 (RIGHT, X=-30):**
- Aim toward X=-150, Z=200 (center of active zone)
- Horizontal angle: ~27° left of straight ahead
- Coverage: X=-180 to X=+120 at active zone center

**Camera 2 (LEFT, X=-270):**
- Aim toward X=-150, Z=200 (center of active zone)
- Horizontal angle: ~27° right of straight ahead
- Coverage: X=-420 to X=-120 at active zone center

**Overlap Region:** X=-180 to X=-120 (60cm wide at Z=200)
- This overlap enables stereo matching and track handoff

### Summary of Recommendations

1. **Move cameras to Z=0** (at panel line) instead of Z=78
2. **Narrow active zone X** from 475cm to 260cm (-280 to -20)
3. **Move marker 5 closer** from Z=628 to Z=400
4. **Optionally add markers 7, 8** at Z=120 for better near-zone calibration
5. **Update camera initial guess** in solvePnP to Z=0 instead of Z=78
6. **Aim cameras toward zone center** (X=-150, Z=200)

### Implementation Priority

1. **HIGH:** Update marker 5 position in world_coordinates.json (Z=628 → Z=400)
2. **HIGH:** Update camera positions if physically moved (Z=78 → Z=0)
3. **MEDIUM:** Narrow active zone X bounds for better accuracy
4. **LOW:** Add additional markers 7, 8 for improved calibration

