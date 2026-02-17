# Drop Ceiling — Camera to DMX Diagrams (Runtime-Focused)

A plain-language diagram set that traces the live system from camera input to Art-Net DMX output.

This file is intentionally separate from `BEHAVIOR_DIAGRAMS.md`.
It is based on the runtime stack centered on:
- `IO/camera_tracker_osc.py`
- `IO/lightController_osc.py`
- `IO/light_behavior.py`
- `IO/systemd/light-controller.service`
- `IO/camera_calibration.py`

> Audience: coding-literate readers who want clear system understanding without heavy jargon.

---

## 1) End-to-End System Overview

```mermaid
flowchart LR
    subgraph CAM["Camera Inputs"]
        C1["Camera 1 RTSP\n10.42.0.75:555"]
        C2["Camera 2 RTSP\n10.42.0.172:555"]
    end

    subgraph TRACKER["Tracker Process\n(camera_tracker_osc.py)"]
        T1["Capture + YOLO\n25 FPS target"]
        T2["Pixel -> Floor World\nusing camera calibration"]
        T3["Cross-camera fusion\n+ temporal smoothing"]
        T4["OSC output\n/tracker/person/<id> x z\n/tracker/count n"]
        T1 --> T2 --> T3 --> T4
    end

    subgraph CTRL["Light Controller\n(lightController_osc.py)"]
        L1["OSC receive\n0.0.0.0:7000"]
        L2["Zone classify\nactive / passive"]
        L3["BehaviorSystem\nmode + gestures + params"]
        L4["Point light -> 12 panel intensities"]
        L5["Art-Net send\nuniverse 0"]
        L1 --> L2 --> L3 --> L4 --> L5
    end

    subgraph OUT["Outputs"]
        D1["DMX Node\n10.42.0.200"]
        D2["Physical LED panels\n12 channels"]
        V1["WebSocket viewer\n:8765"]
        F1["Tailscale Funnel\nremote access path"]
        DB["Tracking DB\ntrends + learning logs"]
        D1 --> D2
        V1 --> F1
    end

    C1 --> TRACKER
    C2 --> TRACKER
    TRACKER -->|"OSC UDP"| CTRL
    CTRL --> D1
    CTRL --> V1
    CTRL --> DB
```

---

## 2) Runtime Services and Startup Order

```mermaid
flowchart TB
    BOOT["System boot"] --> NET["network.target + network-online.target"]

    NET --> CTS["camera-tracker.service\n(camera_tracker_osc.py)"]
    CTS --> LCS["light-controller.service\n(lightController_osc.py)"]

    LCS --> RESTART["Restart=always\nRestartSec=5\nStartLimit: 5 in 60s"]
    LCS --> ENV["Runtime env\nDISPLAY=:0\nSDL_VIDEODRIVER=x11\nvenv python"]

    LCS --> PIPE["Live pipeline\nOSC in -> behavior -> Art-Net out"]
```

---

## 3) Data Contract Between Tracker and Controller

```mermaid
sequenceDiagram
    participant Cam as RTSP Cameras
    participant Trk as camera_tracker_osc.py
    participant Ctl as lightController_osc.py
    participant Beh as light_behavior.py
    participant Dmx as Art-Net / DMX Node

    loop ~25 times per second
        Cam->>Trk: New frames
        Trk->>Trk: Detect people + project to world x,z
        Trk->>Ctl: OSC /tracker/person/<id> x z
        Trk->>Ctl: OSC /tracker/count n
        Ctl->>Ctl: Update tracked people and zones
        Ctl->>Beh: update(active_count, passive_count, positions, trends)
        Beh-->>Ctl: behavior params dict
        Ctl->>Dmx: 12 DMX values (1..255)
    end
```

---

## 4) Tracker Pipeline (Inside `camera_tracker_osc.py`)

```mermaid
flowchart TB
    A["RobustCamera threads\nRTSP capture + reconnect"] --> B["Main loop\nframe sync + pacing"]
    B --> C["YOLO person detection\nmodel: yolo11n.pt"]
    C --> D["Per box: use bottom-center foot point"]
    D --> E["CalibrationManager.bbox_to_floor\nproject pixel -> world floor"]
    E --> F["TrackingFusion._fuse\nmerge close cross-camera detections"]
    F --> G["TrackingFusion._match_and_smooth\ntrack IDs + EMA smoothing"]
    G --> H["OSC send\n/tracker/person/<id> x z\n/tracker/count"]

    I["Live tuning settings\nconfidence: 0.10..0.95\nfusion_dist: 50..500 cm\nsmoothing: 0.01..0.50\nmax_lost_frames: 15..150"] --> C
    I --> F
    I --> G
```

---

## 5) Calibration Geometry (Pixel to World Floor)

```mermaid
flowchart LR
    P["Input: bounding box\n(x1,y1,x2,y2)"] --> F["Foot point\n((x1+x2)/2, y2)"]
    F --> U["Undistort point\nusing K + dist_coeffs"]
    U --> R["Build camera ray\nK^-1 * [u,v,1]"]
    R --> W["Rotate into world\nusing R^T from rvec"]
    W --> I["Intersect with floor plane\ny = -66 cm"]
    I --> O["Output world position\n(x,z) in centimeters"]

    C["Calibration file\nIO/camera_calibration.json\n(rvec, tvec, K, dist)"] --> U
    C --> W
```

**Plain-language note**
- The tracker finds each person in image pixels.
- Calibration converts that image point into a real floor location in centimeters.
- That floor location is what behavior uses.

---

## 6) Controller Frame Loop (Inside `lightController_osc.py`)

```mermaid
sequenceDiagram
    participant OSC as OSC Handler
    participant PM as TrackedPersonManager
    participant BS as BehaviorSystem
    participant PS as PanelSystem
    participant AN as Art-Net
    participant DB as Tracking DB
    participant WS as WebSocket

    loop Render/update loop (~30 FPS)
        OSC->>PM: latest tracked persons (id,x,z)
        PM->>PM: classify active/passive zones
        PM->>BS: counts + dwell + positions
        BS->>BS: mode logic + parameter pipeline
        BS-->>PS: move_speed, pulse_speed, falloff_radius, brightness range, smoothing
        PS->>PS: compute per-panel brightness
        PS->>AN: send 12 DMX bytes
        BS->>DB: behavior and trend records
        PM->>DB: tracking events
        BS->>WS: live state for viewer
    end
```

---

## 7) Behavior Mode State Machine (Runtime Values)

```mermaid
stateDiagram-v2
    [*] --> IDLE

    IDLE: IDLE\nmove 20 cm/s\nbright 3..15\npulse 4000 ms\nfalloff 80 cm
    ENGAGED: ENGAGED\nmove 25 cm/s\nbright 8..30\npulse 2500 ms\nfalloff 50 cm
    CROWD: CROWD\nmove 60 cm/s\nbright 12..45\npulse 1500 ms\nfalloff 40 cm
    FLOW: FLOW\nmove 25 cm/s\nbright 5..20\npulse 3000 ms\nfalloff 70 cm

    IDLE --> ENGAGED: active>=1\nstickiness 0s\ntransition 0.8s
    IDLE --> FLOW: passive_rate>=3/min for 15s\ntransition 2.0s

    ENGAGED --> CROWD: active>=2 for 3s\ntransition 0.5s
    ENGAGED --> IDLE: active=0 for 5s\ntransition 3.0s

    CROWD --> ENGAGED: crowd thins for 5s\ntransition 2.0s
    CROWD --> IDLE: no one for 5s\ntransition 4.0s

    FLOW --> ENGAGED: active>=1\nstickiness 0s\ntransition 0.8s
    FLOW --> IDLE: low passive traffic 10s\ntransition 3.0s

    note right of IDLE
      Global guard:
      min mode duration = 8s
    end note
```

---

## 8) Behavior Parameter Pipeline (Frame-by-Frame)

```mermaid
flowchart TB
    A["1. Start from mode base values"] --> B["2. Blend transitions if mode is switching"]
    B --> C["3. Apply engaged breathing overlay\n(phase + depth)"]
    C --> D["4. Apply proximity response\nnear panels: brighter/slower/tighter"]
    D --> E["5. Apply dwell rewards\nnotice/greet/engage/bond"]
    E --> F["6. Apply active count influence\n(single vs multi-person)"]
    F --> G["7. Apply flow / trend influence\nmostly in idle/flow"]
    G --> H["8. Apply gesture overlays\nwelcome, nod, sway, orbit, bloom..."]
    H --> I["9. Apply meta personality sliders"]
    I --> J["10. Apply global multipliers"]
    J --> K["11. Clamp safe ranges + return params dict"]

    K --> OUT["Output params\nbrightness_min/max\npulse_speed\nfalloff_radius\nmove_speed\nfollow_smoothing\nwander_interval"]
```

---

## 9) Meta Parameters -> Actual Light Behavior

```mermaid
flowchart LR
    subgraph P["Personality sliders (0.0..1.0)"]
        R["responsiveness"]
        E["energy"]
        A["attention_span"]
        S["sociability"]
        X["exploration"]
        M["memory"]
    end

    subgraph G["Global multipliers"]
        BG["brightness_global"]
        SG["speed_global"]
        PG["pulse_global"]
        FG["follow_speed_global"]
        DG["dwell_influence"]
        IG["idle_trend_weight"]
    end

    R --> O1["move_speed\nfollow_smoothing"]
    E --> O2["brightness range\npulse speed"]
    A --> O3["focus stability\nmode/gesture persistence"]
    S --> O4["gesture frequency\nengagement eagerness"]
    X --> O5["wander interval\nwander spread"]
    M --> O6["anti-repetition strength"]

    BG --> O2
    SG --> O1
    PG --> O2
    FG --> O1
    DG --> O7["dwell bonus weight"]
    IG --> O8["idle trend effect strength"]
```

**Example mapping used in code**
- `move_speed *= lerp(0.6, 1.4, responsiveness) * speed_global`
- `pulse_speed *= lerp(1.3, 0.7, energy) * pulse_global`
- `brightness_max *= lerp(0.7, 1.3, energy) * brightness_global`

---

## 10) Auto-Tuning Loop (Every 5 Seconds)

```mermaid
flowchart TB
    T0["Timer\nupdate_interval = 5.0s"] --> SENSE["Read activity signals\ncounts, trends, engagement context"]
    SENSE --> TARGET["Compute adaptive target\nfor activity"]
    TARGET --> DELTA["Propose param deltas\n12 tunable params"]
    DELTA --> REVERT["Apply mean reversion\n(base 0.02 + progressive 0.06*distance)"]
    REVERT --> CURIOUS["Periodic curiosity nudge\ninterval 30s, strength 0.04"]
    CURIOUS --> LIMITS["Clamp per-step\npersonality <= 0.03\nglobal <= 0.08\nignore tiny < 0.002"]
    LIMITS --> BUDGET["Interaction budget gate\nrestore ~300s\nscale deltas if over budget"]
    BUDGET --> CLAMP["Apply floors/caps/min/max"]
    CLAMP --> APPLY["Write new meta values"]
    APPLY --> LOG["Store adjustment history\nfor reports + viewer"]
```

### Runtime auto-tuning constraints (key values)

| Parameter | Min | Max | Safe floor | Soft cap | Home value |
|---|---:|---:|---:|---:|---:|
| responsiveness | 0.0 | 1.0 | 0.30 | 0.90 | 0.50 |
| energy | 0.0 | 1.0 | 0.25 | 0.85 | 0.45 |
| attention_span | 0.0 | 1.0 | 0.10 | — | 0.50 |
| sociability | 0.0 | 1.0 | 0.20 | — | 0.45 |
| exploration | 0.0 | 1.0 | 0.15 | — | 0.40 |
| memory | 0.0 | 1.0 | — | — | 0.30 |
| brightness_global | 0.2 | 5.0 | 0.60 | 3.00 | 1.20 |
| speed_global | 0.2 | 2.0 | 0.35 | 1.60 | 0.70 |
| pulse_global | 0.3 | 3.0 | 0.35 | 2.00 | 0.80 |
| follow_speed_global | 0.5 | 3.0 | 0.60 | — | 1.00 |
| dwell_influence | 0.0 | 2.0 | — | — | 0.50 |
| idle_trend_weight | 0.0 | 2.0 | 0.10 | — | 0.40 |

---

## 11) Feedback and Meta-Tuning Layer

```mermaid
flowchart LR
    subgraph LIVE["Live runtime"]
        L1["Behavior + tuning loop"]
        L2["Writes tracking and tuning history"]
        L1 --> L2
    end

    subgraph REVIEW["Periodic analysis"]
        R1["Daily reports / analysis scripts"]
        R2["Derive better home/floor/cap/reversion settings"]
        R1 --> R2
    end

    subgraph OVERRIDE["Hot-reload config"]
        O1["autotune_overrides.json"]
        O2["Controller reloads every ~30s"]
        O1 --> O2
    end

    LIVE --> REVIEW
    REVIEW --> O1
    O2 --> LIVE
```

**What this loop achieves**
- The 5-second loop keeps behavior responsive in the moment.
- The review layer adjusts the tuner itself over longer periods.
- Together, the light adapts without drifting into unusable extremes.

---

## 12) Final Step: Point Light to 12 DMX Channels

```mermaid
flowchart TB
    P["Current virtual light position + pulse"] --> D["For each panel:\ncompute distance to panel center"]
    D --> R{"distance <= falloff_radius?"}
    R -- No --> OFF["Panel DMX = 1"]
    R -- Yes --> F["falloff factor\n(near=bright, far=dim)"]
    F --> B["Apply pulse + brightness range"]
    B --> M["Map to DMX byte 1..255"]
    M --> OUT["12-channel Art-Net frame"]
```

**Key visual lever**
- `falloff_radius` has the strongest impact on overall look:
  - smaller radius -> tighter spotlight
  - larger radius -> broader wash

---

## Terms (quick glossary)

- **Active zone**: where people are treated as engaging with the installation.
- **Passive zone**: sidewalk traffic area that can influence behavior without direct engagement.
- **Meta parameters**: personality sliders and global multipliers that reshape mode defaults.
- **Auto-tuning**: the 5-second adjustment loop that updates meta parameters.
- **Meta-tuning**: slower review process that updates auto-tuner configuration.

---

## Notes for operators

- This document assumes tracker calibration is loaded from `IO/camera_calibration.json`.
- If your calibrated file is currently in `calibration/camera_calibration.json`, copy or sync it to the runtime path used by `camera_tracker_osc.py`.
