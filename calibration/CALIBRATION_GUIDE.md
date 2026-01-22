# Multi-Camera Calibration Guide

## Overview

The calibration system uses **ArUco fiducial markers** to map each camera's view onto a shared "world" coordinate system, enabling a **synthesized bird's-eye view** that shows all tracked people across all cameras in a single unified display.

## System Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Camera 1   │    │  Camera 2   │    │  Camera N   │
│  (RTSP)     │    │  (RTSP)     │    │  (RTSP)     │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                  │
       ▼                  ▼                  ▼
┌──────────────────────────────────────────────────┐
│              YOLO Person Detection               │
│           (GPU-accelerated tracking)             │
└──────────────────────────────────────────────────┘
       │                  │                  │
       ▼                  ▼                  ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ Homography H₁│   │ Homography H₂│   │ Homography Hₙ│
│ (from calib) │   │ (from calib) │   │ (from calib) │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │                  │                  │
       └─────────────┬────┴────┬─────────────┘
                     ▼         ▼
          ┌──────────────────────────┐
          │   Synthesized Bird's     │
          │      Eye View            │
          │  (unified floor plan)    │
          └──────────────────────────┘
```

## View Modes

| Key | Mode | Description |
|-----|------|-------------|
| `1-9` | Individual | View single camera fullscreen |
| `S` | Side-by-Side | All cameras in a row |
| `G` | Grid | 2x2 or 3x3 grid layout |
| `T` | Synthesized | Bird's eye view + camera thumbnails |
| `C` | Calibration | Interactive marker detection |
| `Q` | Quit | Exit the application |

## Calibration Process

### Step 1: Print ArUco Markers

Generate and print the calibration markers:

```python
from camera_tracker_cuda import CalibrationMode
CalibrationMode.generate_markers()  # Creates aruco_markers/ folder
```

Or run in Python:
```bash
python -c "import cv2; d=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50); [cv2.imwrite(f'marker_{i}.png', cv2.aruco.generateImageMarker(d,i,400)) for i in range(7)]"
```

Print markers 0-6 at a **minimum size of 15cm x 15cm** for reliable detection.

### Marker Layout (7 Markers)

```
                        ┌─────────────────────────────────────┐
                        │        STOREFRONT (panels)          │
                        │     X=0                    X=240    │
                        └─────────────────────────────────────┘
                                        │
                                        │ Z=0 (panels)
                                        │
     ═══════════════════════════════════╪═══════════════════════════════════
                                        │ Z=78 (active zone starts)
                                        │
           ┌─[0]─┐              ┌─[1]─┐              ┌─[2]─┐
           │     │              │SHARED│             │     │
           └─────┘              └─────┘              └─────┘
           X=-40                X=120                X=280
                                                              Z=90 (FRONT ROW)
                                        │
           ┌─[3]─┐              ┌─[6]─┐              ┌─[4]─┐
           │     │              │SHARED│             │     │
           └─────┘              └─────┘              └─────┘
           X=-40                X=120                X=280
                                                              Z=141 (BACK ROW)
                                        │
     ═══════════════════════════════════╪═══════════════════════════════════
                                        │ Z=283 (passive zone starts)
                                        │
                                        │
                                ┌──[5]──┐
                                │VERTICAL│
                                │ SHARED │
                                └────────┘
                                X=120, Z=550
                                (subway wall)
```

**Marker Assignments:**

| ID | Position | Description | Visible To |
|----|----------|-------------|------------|
| 0 | (-40, -66, 90) | Left front | Camera 1 |
| 1 | (120, -66, 90) | Center front | **Both** |
| 2 | (280, -66, 90) | Right front | Camera 2 |
| 3 | (-40, -66, 141) | Left back | Camera 1 |
| 4 | (280, -66, 141) | Right back | Camera 2 |
| 5 | (120, -16, 550) | Subway wall (VERTICAL) | **Both** |
| 6 | (120, -66, 141) | Center back | **Both** |

**Note:** Y=-66 is street level (66cm below storefront floor). Marker 5 is vertical on the subway entrance wall.

### Step 2: Define Marker World Positions

The marker positions are pre-configured in `camera_tracker_cuda.py`. The default configuration matches the table above.

### Step 3: Place Markers in Physical Space

1. Place flat markers (0-4, 6) on the street at the specified X,Z positions
2. Mount marker 5 vertically on the subway entrance wall at camera height
3. Ensure **4+ markers are visible** to each camera for reliable calibration
4. Markers 1 and 6 (center column) must be visible from **both** cameras

### Step 4: Run Calibration

1. Start the tracker: `python camera_tracker_cuda.py`
2. Press `C` to enter calibration mode
3. Press `1` to select Camera 1
4. Press `SPACE` to detect and capture visible markers
5. When 4+ markers detected, press `ENTER` to compute homography
6. Repeat for Camera 2 (press `2`, then `SPACE`, then `ENTER`)
7. Press `S` (in calibration mode) to save calibration to `camera_calibration.json`
8. Press `ESC` to exit calibration, then `T` to view synthesized output

## Calibration File Format

`camera_calibration.json`:
```json
{
  "homographies": {
    "Camera 1": [
      [h11, h12, h13],
      [h21, h22, h23],
      [h31, h32, h33]
    ],
    "Camera 2": [ ... ]
  },
  "synth_width": 800,
  "synth_height": 600,
  "meters_per_pixel": 0.05
}
```

## Adding More Cameras

1. Edit the `CAMERAS` list in `camera_tracker_cuda.py`:

```python
CAMERAS = [
    {'name': 'Camera 1', 'url': 'rtsp://...', 'fps': 30, 'enabled': True},
    {'name': 'Camera 2', 'url': 'rtsp://...', 'fps': 30, 'enabled': True},
    {'name': 'Camera 3', 'url': 'rtsp://...', 'fps': 30, 'enabled': True},
    # Add as many as needed
]
```

2. Ensure cameras share overlapping areas or visible markers
3. Run calibration for each new camera

## Optional: Custom Floor Plan

1. Create a `floor_plan.png` image (same dimensions as SYNTH_VIEW_WIDTH x SYNTH_VIEW_HEIGHT)
2. Place it in the `gettingStarted/` folder
3. The synthesized view will use it as background

## Tuning the Synthesized View

| Setting | Default | Description |
|---------|---------|-------------|
| `SYNTH_VIEW_WIDTH` | 800 | Width of bird's eye view in pixels |
| `SYNTH_VIEW_HEIGHT` | 600 | Height of bird's eye view |
| `SYNTH_METERS_PER_PIXEL` | 0.05 | Scale (0.05 = 5cm/pixel = 40m x 30m coverage) |

## Troubleshooting

### Markers Not Detected
- Ensure adequate lighting
- Print markers larger (20cm+ recommended)
- Check markers aren't reflective or glossy
- ArUco dictionary is DICT_4X4_50 (IDs 0-49)

### Poor Tracking in Synthesized View
- Calibrate with more markers (6-8 recommended)
- Ensure markers are placed accurately
- Recalibrate if cameras moved

### People Appear in Wrong Location
- Homography uses foot position (bottom of bounding box)
- Works best when people are standing on the calibrated floor plane
- Elevated positions will appear offset

## Future Enhancements

1. **Cross-camera track ID fusion**: Merge tracks when person moves between camera views
2. **Height compensation**: Adjust for non-floor marker placement  
3. **Live marker refinement**: Continuously refine homography from visible markers
4. **Zone detection**: Define regions in bird's eye for counting/alerts
5. **OSC output**: Send unified track positions to external systems
