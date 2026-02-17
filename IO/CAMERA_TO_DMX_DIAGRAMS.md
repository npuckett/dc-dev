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

## How to Read This Set

Read top-to-bottom once for system flow, then jump back to sections 8-13 for behavior depth. Labels prefixed with `WANDER:` mark movement-boundary logic that strongly affects spatial DMX output. If a section feels abstract, use section 9 as the anchor and map other diagrams back to it.

---

## 1) End-to-End System Overview

This is the high-level map of the whole runtime pipeline. It shows where camera data enters, where behavior decisions are made, and where physical light output is produced. It also highlights that the wander box sits in the movement path between behavior decisions and panel brightness output.

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
        L35["WANDER: Motion targeting\nbox + follow"]
        L4["Point light -> 12 panel intensities"]
        L5["Art-Net send\nuniverse 0"]
        L1 --> L2 --> L3 --> L35 --> L4 --> L5
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

This diagram shows how runtime processes are started and kept alive in production. It highlights that the light controller depends on network and camera tracker readiness, then runs with automatic restart policies. This helps explain why OSC and DMX are stable even during transient failures.

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

This sequence focuses on message-level handoff between tracking and behavior layers. The important idea is that the tracker sends only person positions and counts, while all meaning (zones, modes, gestures, tuning, wander-box movement boundaries) is decided downstream by the controller. The loop cadence also clarifies where latency and responsiveness are controlled.

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
        Beh-->>Ctl: behavior params dict + wander-box intent
        Ctl->>Dmx: 12 DMX values (1..255)
    end
```

---

## 4) Tracker Pipeline (Inside `camera_tracker_osc.py`)

This is the internal tracking chain from frame capture to OSC output. It shows where live slider settings affect the result: detection confidence, camera fusion distance, and temporal smoothing. The output of this stage is a stable stream of world-space person positions, not lighting decisions.

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

This diagram explains how an image-space foot point becomes a physical floor coordinate. Calibration values define camera geometry, and a ray-floor intersection gives each person location in centimeters. If this stage is off, all downstream behavior can look wrong even when detection itself is good.

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

This is the runtime heartbeat of the light controller. It combines incoming tracked people with behavior logic, wander/follow target selection, panel brightness calculation, and output publishing. It also shows where observability data is emitted to the database and WebSocket viewer.

```mermaid
sequenceDiagram
    participant OSC as OSC Handler
    participant PM as TrackedPersonManager
    participant BS as BehaviorSystem
    participant WB as WanderBehavior
    participant PS as PanelSystem
    participant AN as Art-Net
    participant DB as Tracking DB
    participant WS as WebSocket

    loop Render/update loop (~30 FPS)
        OSC->>PM: latest tracked persons (id,x,z)
        PM->>PM: classify active/passive zones
        PM->>BS: counts + dwell + positions
        BS->>BS: mode logic + parameter pipeline
        BS-->>WB: WANDER: interval + move/follow constraints + box
        WB-->>PS: next light target position (x,y,z)
        BS-->>PS: pulse_speed, falloff_radius, brightness range, smoothing
        PS->>PS: compute per-panel brightness
        PS->>AN: send 12 DMX bytes
        BS->>DB: behavior and trend records
        PM->>DB: tracking events
        BS->>WS: live state for viewer
    end
```

---

## 7) Behavior Mode State Machine (Runtime Values)

This state machine defines the light’s base personality before overlays are applied. Each mode sets a baseline for movement speed, brightness range, pulse period, falloff radius, and wander behavior profile. Stickiness and minimum-duration guards prevent jittery mode flipping when people move around quickly.

```mermaid
stateDiagram-v2
    [*] --> IDLE

    IDLE: IDLE\nmove 20 cm/s\nbright 3..15\npulse 4000 ms\nfalloff 80 cm\nWANDER: broad/loose
    ENGAGED: ENGAGED\nmove 25 cm/s\nbright 8..30\npulse 2500 ms\nfalloff 50 cm\nWANDER: tight/anchored
    CROWD: CROWD\nmove 60 cm/s\nbright 12..45\npulse 1500 ms\nfalloff 40 cm\nWANDER: minimal
    FLOW: FLOW\nmove 25 cm/s\nbright 5..20\npulse 3000 ms\nfalloff 70 cm\nWANDER: direction-biased

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

This pipeline shows how one frame’s final behavior parameters are built step by step. Mode defaults are progressively reshaped by transitions, context, engagement depth, trend signals, and personality controls. The result is a stable parameter bundle that drives both movement behavior and panel intensity rendering.

```mermaid
flowchart TB
    A["1. Start from mode base values"] --> B["2. Blend transitions if mode is switching"]
    B --> C["3. Apply engaged breathing overlay\n(phase + depth)"]
    C --> D["4. Apply proximity response\nnear panels: brighter/slower/tighter"]
    D --> E["5. Apply dwell rewards\nnotice/greet/engage/bond"]
    E --> F["6. Apply active count influence\n(single vs multi-person)"]
    F --> G["7. Apply flow / trend influence\nmostly in idle/flow"]
    G --> H["8. WANDER: update target box\nsize/position by mode + trends + engagement"]
    H --> I["9. Apply gesture overlays\nwelcome, nod, sway, orbit, bloom..."]
    I --> J["10. Apply meta personality sliders"]
    J --> K["11. Apply global multipliers"]
    K --> L["12. Clamp safe ranges + return params dict"]

    L --> OUT["Output params\nbrightness_min/max\npulse_speed\nfalloff_radius\nmove_speed\nfollow_smoothing\nWANDER: interval + box intent"]
```

---

## 9) Wander Box: Behavior Inputs to Motion Output

This diagram makes the wander box role explicit. Behavior state and meta parameters continuously reshape the wander box, which constrains where the light can pick targets when it is not tightly following a person. That target selection then changes point-light position, which directly changes panel distances and final DMX values.

```mermaid
flowchart TB
    subgraph IN["What changes the wander box"]
        M["WANDER input: Mode\n(IDLE/ENGAGED/CROWD/FLOW)"]
        T["WANDER input: Trend + flow\n(passive rate, direction)"]
        D["WANDER input: Dwell + active people"]
        P["WANDER input: Meta sliders\n(exploration, responsiveness, sociability)"]
    end

    M --> WB1
    T --> WB1
    D --> WB1
    P --> WB1

    WB1["WANDER: compute target_wander_box\n(min/max x,y,z)"] --> WB2["WANDER: animated box lerp\n(smooth transitions)"]
    WB2 --> WB3["WANDER: pick next target\ninside current box"]
    WB3 --> POS["Light position trajectory\n(move_speed + follow_smoothing apply)"]
    POS --> DIST["Panel distance field changes"]
    DIST --> DMX["DMX output pattern changes\n(spot focus, spread, panel emphasis)"]
```

### Wander box mechanics (runtime values)

- Base wander box starts near the panels: `x: -290..-30`, `y: 0..150`, `z: -32..28`.
- Box transitions are smoothed (`wander_box_lerp_speed = 3.0`) so motion does not jump when mode/context changes.
- In engagement, behavior can contract movement around people using tight paddings (`±15cm x`, `±35cm y`, `±15cm z`) and controlled approach limits (`z` clamped roughly `-32..60`).

### Worked example: IDLE vs ENGAGED

When discussing output impact, this is the simplest practical comparison:

| Stage | IDLE (no active person) | ENGAGED (1 active person) |
|---|---|---|
| Mode baseline | `move_speed=20`, `wander_interval=5.0`, `falloff=80` | `move_speed=25`, `wander_interval=4.0`, `falloff=50`, `follow_smoothing=0.03` |
| Wander box behavior | Uses broader base box, chooses exploratory targets | Contracts/anchors around person; target updates stay close to person |
| Position path | Slower, wider drift across panel span | Tighter, more deliberate tracking near person position |
| DMX result | Wider illumination spread, gentler panel transitions | More localized hotspots, stronger panel contrast, faster local changes |

In plain terms: the wander box is the movement boundary that decides *where* the light can go between updates. Behavior parameters decide *how fast and how tightly* it moves inside that boundary. Together they strongly shape the spatial pattern of DMX output, not just brightness.

### Mini scenario: passive flow shifts panel emphasis (10-20s)

This timeline shows a common street condition: people move through passive zone mostly left-to-right while nobody is actively engaged. The behavior system treats that as directional flow pressure and shifts wander preference toward the incoming side. As the point light trajectory shifts, panel distance relationships change and DMX emphasis follows.

```mermaid
flowchart LR
    T0["t=0s\nNo active person\nMode: IDLE or FLOW candidate"] --> T1["t=0..10s\nPassive detections accumulate\nflow tracker updates (~1.5s)"]
    T1 --> T2["t~10..15s\nSustained direction signal\n(passive_rate + flow_direction)"]
    T2 --> T3["WANDER: nudge box center\nin flow direction"]
    T3 --> T4["WANDER: target picks bias\ntoward shifted side"]
    T4 --> T5["Light path drifts to that side\nover multiple updates"]
    T5 --> T6["Nearest panels on that side brighten more often\nfar-side panels dim more often"]
    T6 --> T7["Observed output: directional DMX emphasis\nwithout full hard switch"]
```

Practical read: this is one reason output can look intentionally directional even when nobody is standing in the active zone. The wander box is carrying crowd-flow context into spatial light behavior.

---

## 10) Meta Parameters -> Actual Light Behavior

This view isolates personality controls from mode logic. It shows how sliders and global multipliers shape concrete output parameters like speed, pulse timing, brightness, follow tightness, and roaming behavior in the wander box. Use this to explain why two runs with the same tracked people can still feel behaviorally different.

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
    X --> O5["WANDER: interval\nroam bias in box"]
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
- `wander_interval *= lerp(1.5, 0.5, exploration)` (changes how often targets are picked inside the wander box)

---

## 11) Auto-Tuning Loop (Every 5 Seconds)

This is the short-timescale adaptation loop that keeps the light responsive to current street activity. It computes bounded parameter changes, applies safety constraints, and uses a budget mechanism to avoid abrupt over-adjustment. Over time, this loop changes the personality and global multipliers while staying within safe operating limits.

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

## 12) Feedback and Meta-Tuning Layer

This is the slower learning loop above the 5-second tuner. Historical analysis writes override settings that retune the tuner itself, then runtime hot-reloads those settings without restarting the process. In practice, this prevents drift and helps the system evolve over days instead of only reacting second-to-second.

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

## 13) Final Step: Point Light to 12 DMX Channels

This is the last transform from behavior state to physical output. The point light and falloff model produce per-panel intensity values, and each is mapped to a DMX byte for Art-Net transmission. The same behavior parameters therefore influence output both indirectly through wander-box-constrained position and directly through brightness/pulse/falloff settings.

```mermaid
flowchart TB
    WPOS["WANDER/FOLLOW position result\n(current light x,y,z)"] --> P["Current virtual light pulse state"]
    P --> D["For each panel:\ncompute distance to panel center"]
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
- **Wander box**: the current allowed movement boundary for the light target (`min/max x,y,z`), continuously updated by behavior context.
- **Meta parameters**: personality sliders and global multipliers that reshape mode defaults.
- **Auto-tuning**: the 5-second adjustment loop that updates meta parameters.
- **Meta-tuning**: slower review process that updates auto-tuner configuration.

---

## Notes for operators

- This document assumes tracker calibration is loaded from `IO/camera_calibration.json`.
- If your calibrated file is currently in `calibration/camera_calibration.json`, copy or sync it to the runtime path used by `camera_tracker_osc.py`.
