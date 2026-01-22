# Floor Plan Creation Guide for Multi-Camera 3D Tracking

## Overview

The synthesized bird's-eye view uses **3D calibration** with ArUco markers at known world positions. This guide walks through creating an accurate floor plan and positioning markers for proper camera pose estimation.

---

## Step 1: Understand the 3D Coordinate System

### World Coordinate System
```
        Z+ (height/up)
        │
        │
        │
        └───────────► Y+ (depth into room)
       /
      /
     /
    X+ (width/right)
```

- **Origin (0,0,0)**: Your chosen reference point (e.g., a corner)
- **X-axis**: Increases to the right
- **Y-axis**: Increases forward/into the room
- **Z-axis**: Increases upward (height above floor)
- **Floor plane**: Z = 0

### Synthesized View (Bird's Eye)
The bird's eye view shows X-Y positions projected onto the floor (Z=0):
```
Canvas Size:    800 × 600 pixels
Scale:          0.05 meters/pixel (5cm per pixel)
Coverage:       40m × 30m real-world area
```

### Conversion (meters to pixels in synth view)
```
Pixels = Meters × 20
Meters = Pixels × 0.05
```

---

## Step 2: Measure Your Physical Space (3D)

### What to Measure

1. **Room dimensions**
   - Length (Y), Width (X), Height (Z)
   - Note any irregular shapes

2. **Camera positions (in meters)**
   - X: distance from origin along width
   - Y: distance from origin along depth
   - Z: mounting height above floor
   - Viewing direction

3. **Marker positions (in meters)**
   - Each marker needs (X, Y, Z) coordinates
   - Z=0 for floor markers
   - Z>0 for wall/elevated markers

### Example Measurement

```
                    Y=0 (reference wall)
    ┌─────────────────────────────────────────────┐
    │ ●(0,0,0) Origin                             │
    │                                             │
    │    [CAM 1]                                  │
    │    X=2, Y=1.5, Z=3 (mounted at 3m high)     │
    │                                             │
  X=0                                          X=10m
    │                                             │
    │                           [CAM 2]           │
    │                           X=8, Y=6, Z=2.5   │
    └─────────────────────────────────────────────┘
                    Y=10m
```

---

## Step 3: Plan 3D Marker Positions

### Default Marker Layout (editable in code)

| Marker ID | X (m) | Y (m) | Z (m) | Location |
|-----------|-------|-------|-------|----------|
| 0 | 2.0 | 2.0 | 0.0 | Floor, near origin |
| 1 | 8.0 | 2.0 | 0.0 | Floor, far X |
| 2 | 2.0 | 8.0 | 0.0 | Floor, far Y |
| 3 | 8.0 | 8.0 | 0.0 | Floor, opposite corner |
| 4 | 5.0 | 5.0 | 0.0 | Floor, center |
| 5 | 2.0 | 5.0 | 1.5 | Wall, 1.5m high |
| 6 | 8.0 | 5.0 | 1.5 | Wall, 1.5m high |
| 7 | 5.0 | 2.0 | 1.0 | Stand, 1m high |
| 8 | 5.0 | 8.0 | 1.0 | Stand, 1m high |

### Update Code with Your Positions

Edit `camera_tracker_cuda.py`, find `CalibrationMode.__init__`:

```python
self.marker_world_positions_3d = {
    0: (2.0, 2.0, 0.0),      # Floor marker
    1: (8.0, 2.0, 0.0),      # Floor marker
    2: (2.0, 8.0, 0.0),      # Floor marker
    3: (8.0, 8.0, 0.0),      # Floor marker
    4: (5.0, 5.0, 0.0),      # Floor center
    5: (2.0, 5.0, 1.5),      # Wall at 1.5m height
    6: (8.0, 5.0, 1.5),      # Wall at 1.5m height
    # Add your measured positions...
}
```

### Marker Placement Strategy

**Mix of heights is recommended:**
- 4+ floor markers (Z=0) for ground plane reference
- 2+ elevated markers for better depth estimation
- Spread markers across the camera's field of view

**Why 3D is better than 2D:**
- Works with cameras at any angle
- Handles markers at different heights
- Computes true camera position and orientation
- More accurate floor projection for person tracking

---

## Step 4: Create Floor Plan Image (Optional)

The system auto-generates a grid with meter labels if no `floor_plan.png` exists.

### To create a custom floor plan:

1. **Image size**: 800 × 600 pixels
2. **Scale**: 1 meter = 20 pixels
3. Draw room outline, doors, furniture
4. Mark camera positions
5. Save as `floor_plan.png` in `gettingStarted/`

---

## Step 5: Calibration Process

### Generate Markers

```bash
cd /home/nick/Documents/Github/dc-dev
python -c "from gettingStarted.camera_tracker_cuda import CalibrationMode; CalibrationMode.generate_markers()"
```

Or manually:
```bash
python -c "import cv2; d=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50); [cv2.imwrite(f'marker_{i}.png', cv2.aruco.generateImageMarker(d,i,400)) for i in range(9)]"
```

### Print Requirements
- **Size**: Minimum 15cm × 15cm (20cm recommended)
- **Paper**: Matte (non-glossy)
- **Mounting**: Flat on rigid backing

### Calibration Steps

1. **Place markers** at measured 3D positions
2. **Run tracker**: `python gettingStarted/camera_tracker_cuda.py`
3. **Press C** to enter calibration mode
4. **Press 1** to select Camera 1
5. **Press SPACE** to detect markers
6. **Press ENTER** to compute 3D pose (need 4+ markers)
7. **Repeat** for Camera 2 (press 2, SPACE, ENTER)
8. **Press S** to save calibration
9. **Press ESC**, then **T** to view synthesized output

### Calibration Output

After successful calibration, you'll see:
- **Reprojection error**: Should be < 5 pixels (lower is better)
- **Camera position**: (X, Y, Z) in meters from origin

---

## Step 6: Camera Intrinsics (Advanced)

The system estimates camera intrinsics (focal length, etc.) based on resolution.

For higher accuracy:
1. Perform proper camera calibration with a checkerboard
2. Update the `camera_matrix` and `dist_coeffs` in calibration JSON

Default estimation assumes:
- ~65° horizontal field of view
- Minimal lens distortion
- Principal point at image center

---

## Quick Reference Card

```
┌────────────────────────────────────────────────────────┐
│           3D CALIBRATION QUICK REFERENCE               │
├────────────────────────────────────────────────────────┤
│                                                        │
│  COORDINATE SYSTEM:                                    │
│    X = right, Y = forward, Z = up                      │
│    Floor plane: Z = 0                                  │
│                                                        │
│  SYNTH VIEW SCALE:                                     │
│    1 pixel = 5cm = 0.05m                               │
│    Canvas: 800×600 px = 40×30 meters                   │
│                                                        │
│  MARKER REQUIREMENTS:                                  │
│    • Minimum 4 markers per camera                      │
│    • Mix of floor (Z=0) and elevated (Z>0)             │
│    • Print size: 15cm+ (20cm recommended)              │
│    • Measure positions in METERS (x, y, z)             │
│                                                        │
│  CALIBRATION KEYS:                                     │
│    C = Enter calibration mode                          │
│    1-9 = Select camera                                 │
│    SPACE = Detect markers                              │
│    ENTER = Compute 3D pose                             │
│    S = Save calibration                                │
│    ESC = Exit calibration                              │
│    T = Show synthesized view                           │
│                                                        │
│  GOOD CALIBRATION:                                     │
│    • Reprojection error < 5 pixels                     │
│    • Camera position looks reasonable                  │
│    • People appear at correct floor positions          │
│                                                        │
└────────────────────────────────────────────────────────┘
```

---

## Troubleshooting

### "Need 4+ markers"
- Ensure at least 4 markers are visible and detected
- Check that marker IDs match `marker_world_positions_3d`
- Improve lighting, print larger markers

### High Reprojection Error (>10px)
- Re-measure marker positions carefully
- Ensure markers are flat (not curved)
- Check marker size in code matches printed size
- Use more markers for over-determination

### Camera Position Seems Wrong
- Verify origin is consistent
- Check coordinate axis orientation
- Confirm all measurements are in meters

### People Appear in Wrong Location
- The system uses foot position (bottom of bounding box)
- Works best when people are standing on floor (Z=0)
- Re-calibrate if cameras have moved
