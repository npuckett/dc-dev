# Floor Plan Creation Guide for Multi-Camera Tracking

## Overview

The synthesized bird's-eye view requires a floor plan that maps your physical space to pixel coordinates. This guide walks through creating an accurate floor plan and positioning ArUco markers for calibration.

---

## Step 1: Understand the Coordinate System

### Current Default Settings
```
Canvas Size:    800 × 600 pixels
Scale:          0.05 meters/pixel (5cm per pixel)
Coverage:       40m × 30m real-world area
```

### Coordinate Origin
```
    (0,0) ────────────────────────────► X+ (pixels / meters)
      │
      │         YOUR FLOOR PLAN
      │
      │
      ▼
     Y+ (pixels / meters)
```

- **Origin (0,0)**: Top-left corner
- **X-axis**: Increases to the right
- **Y-axis**: Increases downward (standard image coordinates)

### Conversion Formulas
```
Pixels = Meters × 20
Meters = Pixels × 0.05
```

### Reference Table
| Meters | Pixels |
|--------|--------|
| 0.5m   | 10px   |
| 1m     | 20px   |
| 2m     | 40px   |
| 5m     | 100px  |
| 10m    | 200px  |
| 15m    | 300px  |
| 20m    | 400px  |
| 30m    | 600px  |
| 40m    | 800px  |

---

## Step 2: Measure Your Physical Space

### What to Measure

1. **Room dimensions**
   - Total length and width
   - Note any irregular shapes

2. **Camera positions**
   - Distance from a reference corner/wall
   - Height (for documentation, not used in 2D plan)
   - Viewing direction/angle

3. **Key landmarks**
   - Doors, pillars, furniture
   - Areas of interest for tracking
   - Obstacles

4. **Planned marker positions**
   - Measure where you'll place each ArUco marker
   - Need 4+ markers visible per camera

### Recommended Approach

```
                    WALL A (reference wall)
    ┌─────────────────────────────────────────────┐
    │ •(0,0)                                      │
    │   ↓ measure from here                       │
    │                                             │
  W │        [CAM 1]                              │ W
  A │           ↘                                 │ A
  L │              viewing                        │ L
  L │              direction     [CAM 2]          │ L
    │                               ↙             │
  D │                                             │ B
    │                                             │
    │                                             │
    └─────────────────────────────────────────────┘
                    WALL C
```

**Pick a reference corner** (e.g., where Wall A meets Wall D) as your origin (0,0).

---

## Step 3: Create the Floor Plan Image

### Option A: Simple Grid (No Image Editing Required)

The system auto-generates a grid if no `floor_plan.png` exists. Just configure marker positions in code.

### Option B: Create Custom Floor Plan

**Requirements:**
- Image size: 800 × 600 pixels (or match your `SYNTH_VIEW_WIDTH` × `SYNTH_VIEW_HEIGHT`)
- Format: PNG recommended
- Filename: `floor_plan.png` in `gettingStarted/` folder

**Using GIMP, Photoshop, or similar:**

1. Create new image: 800 × 600 pixels
2. Fill with dark background (#282828 recommended)
3. Draw room outline using the conversion: `meters × 20 = pixels`
4. Add reference features (doors, walls, etc.)
5. Mark camera positions with icons/labels
6. Add a scale bar for reference
7. Save as `floor_plan.png`

**Example layout for a 15m × 12m room:**
```
Room: 15m × 12m = 300px × 240px

If centered in 800×600 canvas:
- Room starts at pixel (250, 180)
- Room ends at pixel (550, 420)
```

### Option C: Scale from Architectural Drawing

If you have an existing floor plan:

1. Determine the real-world dimensions
2. Calculate required scale: `target_pixels / real_meters = scale`
3. Resize image to fit 800×600 while maintaining aspect ratio
4. Note the final meters-per-pixel for configuration

---

## Step 4: Plan Camera Positions

### Document Each Camera

| Camera | Physical Position | Pixel Position | Mounting Height | View Direction |
|--------|------------------|----------------|-----------------|----------------|
| Cam 1  | (2m, 1.5m)       | (40, 30)       | 3m              | Southeast      |
| Cam 2  | (12m, 8m)        | (240, 160)     | 2.5m            | Northwest      |

### Camera Coverage Visualization

Sketch approximate camera field-of-view on your floor plan:

```
        [CAM 1] ←── camera position
           /\
          /  \
         /    \
        /      \
       /________\  ←── approximate coverage area
```

### Overlap Zones

Identify areas where cameras overlap - these are ideal for:
- Cross-camera track handoff
- Verification of calibration accuracy
- Placing shared calibration markers

---

## Step 5: Plan Marker Positions

### Marker Placement Strategy

**Minimum:** 4 markers per camera
**Recommended:** 6-8 markers for better accuracy

**Placement rules:**
1. Spread markers across the camera's field of view
2. Avoid clustering markers in one area
3. Place at floor level (or known height with compensation)
4. Ensure markers are flat and not curved
5. Avoid reflective surfaces behind markers

### Example Marker Layout

```
    ┌─────────────────────────────────────────────┐
    │                                             │
    │   [0]                              [1]      │
    │                                             │
    │              [4]                            │
    │                        (center)             │
    │                                             │
    │   [2]                              [3]      │
    │                                             │
    └─────────────────────────────────────────────┘

Marker positions (example for 20m × 15m room):
- Marker 0: (2m, 2m)    → pixels (40, 40)
- Marker 1: (18m, 2m)   → pixels (360, 40)
- Marker 2: (2m, 13m)   → pixels (40, 260)
- Marker 3: (18m, 13m)  → pixels (360, 260)
- Marker 4: (10m, 7.5m) → pixels (200, 150)
```

### Update Code with Marker Positions

Edit `camera_tracker_cuda.py`, find `CalibrationMode.__init__`:

```python
self.marker_world_positions = {
    0: (40, 40),      # 2m, 2m from origin
    1: (360, 40),     # 18m, 2m
    2: (40, 260),     # 2m, 13m
    3: (360, 260),    # 18m, 13m
    4: (200, 150),    # 10m, 7.5m (center)
}
```

---

## Step 6: Print and Place Markers

### Printing Markers

1. The markers were generated in the workspace root:
   - `marker_0.png` through `marker_4.png`

2. Print at **minimum 15cm × 15cm** (20cm+ recommended for ceiling cameras)

3. Use matte paper (avoid glossy/reflective)

4. Mount on rigid backing (cardboard, foamcore)

### Physical Placement

1. Lay markers flat on floor at planned positions
2. Use tape measure from your reference origin
3. Ensure markers are:
   - Horizontal (not tilted)
   - Right-side up
   - Clearly visible to cameras
   - Not occluded by furniture/people

### Record Actual Positions

After placement, verify and record actual positions:

| Marker ID | Planned Position | Actual Position | Pixel Coords |
|-----------|-----------------|-----------------|--------------|
| 0         | (2m, 2m)        | (2.1m, 1.95m)   | (42, 39)     |
| 1         | (18m, 2m)       | (17.9m, 2.05m)  | (358, 41)    |
| ...       | ...             | ...             | ...          |

---

## Step 7: Adjust Scale (Optional)

### For Smaller Spaces

If your space is smaller than 40m × 30m, you may want finer resolution:

```python
# Example: 10m × 8m room with higher detail
SYNTH_VIEW_WIDTH = 800
SYNTH_VIEW_HEIGHT = 640
SYNTH_METERS_PER_PIXEL = 0.0125  # 1.25cm per pixel

# Now: 800px = 10m, 640px = 8m
# Conversion: pixels = meters × 80
```

### For Larger Spaces

```python
# Example: 80m × 60m area
SYNTH_VIEW_WIDTH = 800
SYNTH_VIEW_HEIGHT = 600
SYNTH_METERS_PER_PIXEL = 0.1  # 10cm per pixel

# Now: 800px = 80m, 600px = 60m
# Conversion: pixels = meters × 10
```

---

## Quick Reference Card

```
┌────────────────────────────────────────────────────────┐
│              CALIBRATION QUICK REFERENCE               │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Default Scale: 1 pixel = 5cm = 0.05m                  │
│  Canvas: 800×600 px = 40×30 meters                     │
│                                                        │
│  CONVERSIONS:                                          │
│    Meters → Pixels: multiply by 20                     │
│    Pixels → Meters: multiply by 0.05                   │
│                                                        │
│  MARKER REQUIREMENTS:                                  │
│    • Minimum 4 markers per camera                      │
│    • Print size: 15cm+ (20cm recommended)              │
│    • Place on floor, spread across view                │
│    • Use matte (non-glossy) paper                      │
│                                                        │
│  CALIBRATION KEYS:                                     │
│    C = Enter calibration mode                          │
│    1-9 = Select camera                                 │
│    SPACE = Capture markers                             │
│    ENTER = Compute homography                          │
│    S = Save calibration                                │
│    ESC = Exit calibration                              │
│                                                        │
└────────────────────────────────────────────────────────┘
```

---

## Troubleshooting

### Markers Not Detected
- Increase print size
- Improve lighting (avoid harsh shadows)
- Check for glare/reflections
- Ensure marker is flat and not warped

### Poor Position Accuracy
- Re-measure marker positions
- Use more markers (6-8)
- Ensure markers are truly on floor plane
- Check for lens distortion (fisheye cameras need correction)

### Coordinate Mismatch
- Verify origin is consistent between physical and digital
- Check X/Y axis orientation matches
- Confirm scale factor is correct
