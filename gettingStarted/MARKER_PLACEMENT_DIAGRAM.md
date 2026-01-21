# 5-Marker Calibration Plan

## Coordinate System (centimeters)

- **X**: Left-right along panel array (panels centered at X=120)
- **Y**: Height (floor = 0)
- **Z**: Depth from panels (Z=0 at panels, increasing outward)

## Marker Coordinates

| Marker # | Location Description | Coordinates (cm) | Camera |
|----------|---------------------|------------------|--------|
| **0** | Left front - on Z=90 tile line, left seam | X = -40, Y = 0, Z = 90 | Cam 1 |
| **1** | Center front - on Z=90 tile line, centered | X = 120, Y = 0, Z = 90 | **SHARED** |
| **2** | Right front - on Z=90 tile line, right seam | X = 280, Y = 0, Z = 90 | Cam 2 |
| **3** | Left back - one tile further from marker 0 | X = -40, Y = 0, Z = 141 | Cam 1 |
| **4** | Right back - one tile further from marker 2 | X = 280, Y = 0, Z = 141 | Cam 2 |

## Top-Down View (X-Z Plane)

```
    Z (depth from panels, cm)
    ▲
    │
    │              PANELS (Z = 0)
    │    ════════════════════════════════════════
    │         Unit 0    Unit 1    Unit 2    Unit 3
    │           │         │         │         │
    │           ▼         ▼         ▼         ▼
    │    ┌─────────────────────────────────────────┐
    │    │            STOREFRONT FLOOR             │
    │    │                                         │
  90┤────│──[0]────────────[1]────────────[2]──────│────  ← FRONT ROW (tile line)
    │    │   ★              ★★             ★       │
    │    │  (-40)         (120)          (280)     │
    │    │                                         │
    │    │                                         │
 141┤────│──[3]────────────────────────────[4]─────│────  ← BACK ROW (tile line)
    │    │   ★                              ★      │
    │    │  (-40)                         (280)    │
    │    │                                         │
    │    │                                         │
    │    └─────────────────────────────────────────┘
    │                    SNOW LINE
    │
────┼──────────────────────────────────────────────────►  X (cm)
        -40      0      60     120     180    240    280
         │              │       │              │      │
         │              │       │              │      │
     LEFT SEAM      UNIT 1   CENTER        UNIT 3  RIGHT SEAM
```

## Camera View Zones

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│              PANELS (at Z = 0)                                   │
│    ══════════════════════════════════════════                    │
│                                                                  │
│                                                                  │
│   ╔═══════════════════════╦═══════════════════════╗              │
│   ║                       ║                       ║              │
│   ║      CAMERA 1         ║       CAMERA 2        ║              │
│   ║       ZONE            ║        ZONE           ║              │
│   ║                       ║                       ║              │
│   ║   ┌───┐          ┌────┴────┐           ┌───┐  ║   Z = 90     │
│   ║   │ 0 │          │    1    │           │ 2 │  ║   (front)    │
│   ║   └───┘          │ SHARED  │           └───┘  ║              │
│   ║  X=-40           └────┬────┘          X=280   ║              │
│   ║                   X=120                       ║              │
│   ║                       ║                       ║              │
│   ║   ┌───┐               ║                ┌───┐  ║   Z = 141    │
│   ║   │ 3 │               ║                │ 4 │  ║   (back)     │
│   ║   └───┘               ║                └───┘  ║              │
│   ║  X=-40                ║               X=280   ║              │
│   ║                       ║                       ║              │
│   ╚═══════════════════════╩═══════════════════════╝              │
│                                                                  │
│   Camera 1 sees: 0, 1, 3          Camera 2 sees: 1, 2, 4         │
│   (left + shared)                 (right + shared)               │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Side View (looking from left, X-Z plane at Y=0)

```
         PANELS                   FRONT ROW              BACK ROW
           │                         │                      │
           │                         │                      │
           ▼                         ▼                      ▼
    ═══════════════════─────────[0][1][2]─────────────[3]──[4]────
           │                                                      
         Z = 0                    Z = 90                 Z = 141   
                                                                   
                              51 cm tile                51 cm tile
```

## Placement Instructions

### Step 1: Identify Tile Lines
- **Front row**: Z = 90 cm from panels
- **Back row**: Z = 141 cm from panels (one 51cm tile further)

### Step 2: Place Front Row (Z = 90)
- **Marker 0** at X = -40 cm (left tile seam)
- **Marker 1** at X = 120 cm (centered on panels) — **SHARED**
- **Marker 2** at X = 280 cm (right tile seam)

### Step 3: Place Back Row (Z = 141)
- **Marker 3** at X = -40 cm (directly behind marker 0)
- **Marker 4** at X = 280 cm (directly behind marker 2)

### Placement Tips
1. **Lay markers flat** on the ground, face up toward cameras
2. **Tape corners** so they don't shift
3. **Center of marker** = the coordinate position
4. Use the tile lines as guides for Z positioning

## Calibration Process

```bash
# Start the tracker
python camera_tracker_cuda.py
```

1. **Press C** to enter calibration mode
2. **Press 1** to select Camera 1, then **SPACE** to detect markers (should see 0, 1, 3)
3. **Press 2** to select Camera 2, then **SPACE** to detect markers (should see 1, 2, 4)
4. **Press ENTER** to compute camera pose
5. **Press S** to save calibration

## Calibration Checklist

```
Camera 1 (Left Side):
  [ ] Marker 0 @ (-40, 0, 90) detected
  [ ] Marker 1 @ (120, 0, 90) SHARED detected
  [ ] Marker 3 @ (-40, 0, 141) detected
  [ ] 3D pose computed successfully

Camera 2 (Right Side):
  [ ] Marker 1 @ (120, 0, 90) SHARED detected
  [ ] Marker 2 @ (280, 0, 90) detected
  [ ] Marker 4 @ (280, 0, 141) detected
  [ ] 3D pose computed successfully
```
