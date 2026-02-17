# Drop Ceiling — Behavior System Diagrams

A visual walkthrough of how the Drop Ceiling light thinks, moves, and learns — from camera input to physical light output.

Each diagram builds on the previous one, starting with the highest-level overview and drilling progressively deeper into the system's internals. If you're new to the project, read them in order.

> All diagrams use [Mermaid](https://mermaid.js.org/) syntax and render natively on GitHub.

---

## 1. System Overview

The Drop Ceiling installation is a single simulated point light that moves above a grid of LED panels. Two cameras watch pedestrians below, and the light responds in real-time.

At the highest level, data flows through six stages: two RTSP cameras capture video, YOLO detects people, calibration projects detections to real-world floor coordinates, cross-camera fusion and temporal tracking produce stable positions, the behavior system decides what the light should do, and Art-Net sends DMX values to the physical panels.

```mermaid
flowchart LR
    subgraph INPUT["Camera Input"]
        direction TB
        CAM1["<b>Camera 1 (Right)</b>
        ───────────────
        Model: Reolink RLC-520A
        Position: X=-30, Z=78
        IP: 10.42.0.75
        RTSP port 555
        Resolution: 2048 x 1536"]

        CAM2["<b>Camera 2 (Left)</b>
        ───────────────
        Model: Reolink RLC-520A
        Position: X=-270, Z=78
        IP: 10.42.0.172
        RTSP port 555
        Resolution: 2048 x 1536"]
    end

    subgraph DETECT["Detection and Calibration"]
        direction TB
        ROBUST["<b>RobustCamera</b>
        ───────────────
        Threads: 1 daemon per camera
        Buffer flush: grab() x3
        Reconnect: auto on failure"]

        YOLO["<b>YOLO 11n</b>
        ───────────────
        Input: 416px resize
        Class: person only
        Confidence: 0.10 - 0.80
        Output: bounding boxes"]

        CAL["<b>Calibration</b>
        ───────────────
        Method: ray-plane intersect
        Floor: Y = -66 cm
        Pre-computed: R_T, K_inv
        Output: world (X, Z) cm"]

        ROBUST --> YOLO --> CAL
    end

    subgraph TRACKING["Fusion and Tracking"]
        direction TB
        FUSE["<b>Cross-Camera Fusion</b>
        ───────────────
        Merge: different cameras only
        Threshold: 50 - 300 cm
        Method: greedy nearest-neighbor"]

        SMOOTH["<b>Temporal Tracking</b>
        ───────────────
        Velocity: prediction + correct
        EMA alpha: 0.01 - 0.20
        Prune: 60 frames lost"]

        FUSE --> SMOOTH
    end

    subgraph TRANSPORT["OSC Transport"]
        OSC["<b>OSC Output</b>
        ───────────────
        /tracker/count n
        /tracker/person/id x z
        Target: 127.0.0.1:7000
        Protocol: UDP, 25 Hz"]
    end

    subgraph BRAIN["Behavior Engine"]
        direction TB
        TRACK["<b>Person Manager</b>
        ───────────────
        Zone: active / passive classify
        Velocity: per-person tracking
        Dwell: time in zone
        Callbacks: enter / exit / move"]
        BEH["<b>Behavior System</b>
        ───────────────
        Mode: IDLE / ENGAGED / CROWD / FLOW
        Gestures: 16 types, phase-gated
        Personality: 6 meta sliders
        Learning: feedback + daily"]
        TRACK --> BEH
    end

    subgraph CONTROLLER["Light Controller"]
        direction TB
        LIGHT["<b>Point Light</b>
        ───────────────
        Position: x, y, z (cm)
        Brightness: min / max range
        Falloff: radius (cm)
        Pulse: sine wave phase"]
        PANELS["<b>Panel System</b>
        ───────────────
        Layout: 4 units x 3 panels
        Calc: distance-based falloff
        Output: 12 DMX values (1-255)"]
        LIGHT --> PANELS
    end

    subgraph OUTPUT["Physical Output"]
        ARTNET["<b>Art-Net</b>
        ───────────────
        Protocol: Art-Net UDP
        Channels: 12 (Universe 0)
        Target: 10.42.0.200
        Rate: 30 FPS"]
        LEDS["<b>LED Panels</b>
        ───────────────
        Units: 4 ceiling-mounted
        Panels per unit: 3
        Control: single DMX ch each"]
        ARTNET --> LEDS
    end

    CAM1 --> ROBUST
    CAM2 --> ROBUST
    CAL -->|"world detections"| FUSE
    SMOOTH -->|"tracked (id, x, z)"| OSC
    OSC -->|"x, z per person"| TRACK
    BEH -->|"behavior_params dict
    brightness, speed, falloff,
    pulse, smoothing, wander"| LIGHT
    PANELS -->|"12 DMX values
    (1-255 per panel)"| ARTNET

    style INPUT fill:#1a1a2e,stroke:#e94560,color:#fff
    style DETECT fill:#16213e,stroke:#0f3460,color:#e0e1dd
    style TRACKING fill:#0f3460,stroke:#e94560,color:#fff
    style TRANSPORT fill:#1a1a2e,stroke:#0f3460,color:#fff
    style BRAIN fill:#1a1a2e,stroke:#16213e,color:#fff
    style CONTROLLER fill:#1a1a2e,stroke:#533483,color:#fff
    style OUTPUT fill:#1a1a2e,stroke:#e94560,color:#fff
```

Three Python files implement the system:

| File | Responsibility |
|---|---|
| `camera_tracker_osc.py` | Camera capture, YOLO detection, calibration, fusion, temporal tracking, OSC output |
| `light_behavior.py` | State machine, gestures, trend analysis, feedback learning |
| `lightController_osc.py` | Main loop, OSC input, point light, panel math, Art-Net output, auto-tuning |

The behavior system outputs a **params dict** every frame containing target values for brightness, speed, falloff radius, pulse rate, follow smoothing, and wander interval. The controller interpolates toward these targets and computes per-panel DMX values.

---

## 2. Behavior Mode State Machine

The light is always in one of four modes. Mode determines the base personality — how fast it moves, how bright it shines, and whether it follows someone or wanders on its own.

Transitions are **not instantaneous**. Conditions must persist for a minimum duration (stickiness) before the mode switches, and parameters interpolate smoothly over a transition period. This prevents erratic flickering between states.

```mermaid
flowchart LR
    START(( )) --> IDLE

    IDLE["**IDLE**
    ───────────────
    Trigger: no one in active zone
    Behavior: gentle wandering
    ───────────────
    Speed: 20 cm/s
    Brightness: 3-15
    Pulse: 4000 ms
    Falloff: 80 cm
    Smoothing: 0"]

    ENGAGED["**ENGAGED**
    ───────────────
    Trigger: 1-2 in active zone
    Behavior: follows nearest person
    ───────────────
    Speed: 25 cm/s
    Brightness: 8-30
    Pulse: 2500 ms
    Falloff: 50 cm
    Smoothing: 0.03"]

    CROWD["**CROWD**
    ───────────────
    Trigger: 3+ in active zone
    Behavior: follows centroid
    ───────────────
    Speed: 60 cm/s
    Brightness: 12-45
    Pulse: 1500 ms
    Falloff: 40 cm
    Smoothing: 0.03"]

    FLOW["**FLOW**
    ───────────────
    Trigger: heavy passive traffic
    Behavior: drifts with crowd flow
    ───────────────
    Speed: 25 cm/s
    Brightness: 5-20
    Pulse: 3000 ms
    Falloff: 70 cm
    Smoothing: 0"]

    IDLE -->|"person enters active zone
    stickiness: 0s (immediate)
    transition: 0.8s"| ENGAGED

    IDLE -->|"15s sustained passive traffic
    transition: 2.0s"| FLOW

    ENGAGED -->|"5s after last person leaves
    transition: 3.0s (slow goodbye)"| IDLE

    ENGAGED -->|"3s with 3+ people
    transition: 0.5s (quick)"| CROWD

    CROWD -->|"5s after crowd thins
    transition: 2.0s"| ENGAGED

    CROWD -->|"5s after everyone leaves
    transition: 4.0s"| IDLE

    FLOW -->|"10s of low traffic
    transition: 3.0s"| IDLE

    FLOW -->|"person enters active zone
    stickiness: 0s (immediate)
    transition: 0.8s"| ENGAGED

    GUARD["**Mode Guard**
    ───────────────
    Min duration: 8 seconds
    prevents rapid flip-flopping"]

    GUARD -.-> IDLE
    GUARD -.-> ENGAGED
    GUARD -.-> CROWD
    GUARD -.-> FLOW

    style START fill:#e94560,stroke:#e94560,color:#fff
    style IDLE fill:#0d1b2a,stroke:#415a77,color:#e0e1dd
    style ENGAGED fill:#0f3460,stroke:#e94560,color:#fff
    style CROWD fill:#533483,stroke:#e94560,color:#fff
    style FLOW fill:#1b263b,stroke:#778da9,color:#e0e1dd
    style GUARD fill:#1a1a2e,stroke:#778da9,color:#e0e1dd
```

**Key design choice**: Engaging is fast (0.5–0.8s), disengaging is slow (2.0–4.0s). The light is eager to connect and reluctant to let go.

---

## 3. The 17-Layer Parameter Pipeline

This is the heart of the system. Every frame, `calculate_parameters()` in `light_behavior.py` builds the output params dict by passing it through 17 sequential layers. Each layer can modify brightness, speed, falloff, pulse rate, smoothing, and/or wander interval.

The layers are grouped into four stages: **Foundation** sets the starting point, **Personality & Context** applies the light's character and time awareness, **Environmental Response** reacts to what's happening around the light, and **Overlays** add momentary expressive effects.

```mermaid
flowchart TB
    subgraph FOUNDATION["1 - Foundation"]
        direction TB
        L1["<b>Mode Base Values</b><br/>───────────────<br/>Input: current BehaviorMode<br/>Lookup: MODE_PARAMS table<br/>Sets: all 7 output params"]
        L2["<b>Transition Interpolation</b><br/>───────────────<br/>Condition: mode is switching<br/>Method: lerp old params to new<br/>Duration: 0.5 - 4.0s per type"]
        L3["<b>People-Count Scaling</b><br/>───────────────<br/>Input: active person count<br/>Effect: +20% brightness per person<br/>Affects: brightness_min, brightness_max"]
        L1 --> L2 --> L3
    end

    subgraph PERSONALITY["2 - Personality and Context"]
        direction TB
        L4["<b>MetaParameter Modifiers</b><br/>───────────────<br/>Input: 6 personality sliders (0-1)<br/>Input: 6 global multipliers (default 1.0)<br/>Method: lerp + multiply<br/>Affects: speed, brightness, pulse, smoothing, wander"]
        L5["<b>Time-of-Day</b><br/>───────────────<br/>Input: current hour<br/>Brightness: x0.4 (night) to x1.1 (rush)<br/>Pulse: x0.7 to x1.5<br/>Wander Y: constrained by period"]
        L6["<b>Dwell Rewards</b><br/>───────────────<br/>Input: person dwell time<br/>Phases: Notice / Greet / Engage / Bond<br/>Effect: longer dwell = brighter, tighter<br/>Scale: dwell_influence multiplier"]
        L7["<b>Anti-Repetition</b><br/>───────────────<br/>Input: recent gesture + position history<br/>Scale: memory meta parameter<br/>Effect: suppresses repeated patterns"]
        L4 --> L5 --> L6 --> L7
    end

    subgraph ENVIRONMENT["3 - Environmental Response"]
        direction TB
        L8["<b>Idle Trends</b><br/>───────────────<br/>Windows: 1m / 5m / 30m / 1h<br/>Derived: anticipation (0-1)<br/>Derived: energy_level (0-1)<br/>Derived: flow_momentum (-1 to +1)"]
        L9["<b>Aggression</b><br/>───────────────<br/>Input: time without engagement<br/>Range: 0-1 (EMA smoothed)<br/>Cap: time-of-day dependent<br/>Effect: wander width, pulse, gesture rate"]
        L10["<b>Flow Positioning</b><br/>───────────────<br/>Input: flow direction + strength<br/>Effect: wander box X shifts toward<br/>incoming pedestrian traffic"]
        L11["<b>Almost-Engaged Attraction</b><br/>───────────────<br/>Input: people slowing near active zone<br/>Strategies: brightness / drift / pause<br/>A/B tested: conversion per strategy"]
        L12["<b>Feedback Learning</b><br/>───────────────<br/>Input: 50-context ring buffer<br/>Dims: position x aggression x time x flow<br/>Output: engagement correlation weights"]
        L13["<b>Proximity Response</b><br/>───────────────<br/>Input: person Z distance (78-283 cm)<br/>Near: speed x0.6, bright x1.4, smooth x0.7<br/>Far: speed x1.4, bright x0.8, smooth x1.3<br/>Method: linear interpolation"]
        L8 --> L9 --> L10 --> L11 --> L12 --> L13
    end

    subgraph OVERLAYS["4 - Momentary Overlays"]
        direction TB
        L14["<b>Flow Bias</b><br/>───────────────<br/>Input: flow_balance value<br/>Effect: shifts wander box X"]
        L15["<b>Entry Pulse</b><br/>───────────────<br/>Trigger: person enters active zone<br/>Effect: +25 brightness (one-shot)"]
        L16["<b>Breathing Overlay</b><br/>───────────────<br/>Type: multiplicative sine wave<br/>Brightness: +/-12% at full depth<br/>Radius: +/-6% at full depth<br/>Ramp: 0 to full over 8 seconds"]
        L17["<b>Settle / Bloom</b><br/>───────────────<br/>Settle: radius -15%, speed -20%, bright +8%<br/>Bloom: radius to 300cm, bright +50%<br/>Bloom chance: 15%/min, 45s cooldown"]
        L14 --> L15 --> L16 --> L17
    end

    L3 --> L4
    L7 --> L8
    L13 --> L14

    L17 --> OUT["<b>Final behavior_params dict</b><br/>───────────────<br/>brightness_min / brightness_max<br/>pulse_speed / falloff_radius<br/>move_speed / follow_smoothing<br/>wander_interval"]

    style FOUNDATION fill:#0d1b2a,stroke:#1b263b,color:#e0e1dd
    style PERSONALITY fill:#1b263b,stroke:#415a77,color:#e0e1dd
    style ENVIRONMENT fill:#415a77,stroke:#778da9,color:#e0e1dd
    style OVERLAYS fill:#778da9,stroke:#e0e1dd,color:#0d1b2a
    style OUT fill:#e94560,stroke:#fff,color:#fff
```

**What each output parameter controls:**

| Parameter | What it does |
|---|---|
| `brightness_min` / `brightness_max` | DMX brightness range for the pulse cycle |
| `pulse_speed` | Period of the brightness sine wave (ms) |
| `falloff_radius` | How far from the light panels still illuminate (cm) |
| `move_speed` | How fast the light moves toward its target (cm/s) |
| `follow_smoothing` | How tightly the light tracks a person (0 = no follow, 0.2 = tight) |
| `wander_interval` | Time between picking new wander targets (s) |

---

## 4. MetaParameters → Actual Light Values

The personality system consists of **6 sliders** (0.0–1.0) that define the light's character and **6 global multipliers** that scale the output. Together they transform the mode base values into the light's actual behavior.

`apply_meta_modifiers()` in `light_behavior.py` maps each personality slider to one or more output parameters using linear interpolation, then applies the global multipliers.

```mermaid
flowchart LR
    subgraph SLIDERS["Personality Sliders (0.0 - 1.0)"]
        RESP["<b>responsiveness</b><br/>───────────<br/>Low: contemplative, slow<br/>High: reactive, quick"]
        ENER["<b>energy</b><br/>───────────<br/>Low: calm, gentle<br/>High: lively, dynamic"]
        ATTN["<b>attention_span</b><br/>───────────<br/>Low: easily distracted<br/>High: focused, loyal"]
        SOCI["<b>sociability</b><br/>───────────<br/>Low: reserved, withdrawn<br/>High: eager to engage"]
        EXPL["<b>exploration</b><br/>───────────<br/>Low: stays in place<br/>High: wanders widely"]
        MEMO["<b>memory</b><br/>───────────<br/>Low: forgets quickly<br/>High: avoids repetition"]
    end

    subgraph OUTPUTS["Output Parameters"]
        SPEED["<b>move_speed</b><br/>───────────<br/>Range: x0.6 - x1.4<br/>Unit: cm/s"]
        FOLLOW["<b>follow_smoothing</b><br/>───────────<br/>Range: 0.03 - 0.20<br/>0 = no follow"]
        PULSE["<b>pulse_speed</b><br/>───────────<br/>Range: x1.3 - x0.7<br/>Unit: ms period"]
        BRIGHT["<b>brightness</b><br/>───────────<br/>Range: x0.7 - x1.3<br/>Unit: DMX (1-255)"]
        WANDER["<b>wander_interval</b><br/>───────────<br/>Range: x1.5 - x0.5<br/>Unit: seconds"]
        GESTURE["<b>gesture frequency</b><br/>───────────<br/>Range: x1.5 - x0.5<br/>Unit: interval (s)"]
        ANTIREP["<b>anti-repetition</b><br/>───────────<br/>Strength: 0.0 - 1.0<br/>Suppresses repeats"]
    end

    RESP -->|"lerp"| SPEED
    RESP -->|"lerp"| FOLLOW
    ENER -->|"lerp"| PULSE
    ENER -->|"lerp"| BRIGHT
    EXPL -->|"lerp"| WANDER
    SOCI -->|"lerp"| GESTURE
    ATTN -->|"weight"| GESTURE
    MEMO -->|"scale"| ANTIREP

    subgraph MULTIPLIERS["Global Multipliers (default 1.0)"]
        BG["<b>brightness_global</b><br/>range: 0.3 - 5.0"]
        SG["<b>speed_global</b><br/>range: 0.3 - 3.0"]
        PG["<b>pulse_global</b><br/>range: 0.3 - 3.0"]
        FG["<b>follow_speed_global</b><br/>range: 0.3 - 3.0"]
        DI["<b>dwell_influence</b><br/>range: 0.0 - 3.0"]
        TW["<b>trend_weight</b><br/>range: 0.0 - 3.0"]
    end

    BG -->|"x"| BRIGHT
    SG -->|"x"| SPEED
    PG -->|"x"| PULSE
    FG -->|"x"| FOLLOW
    DI -.->|"scales dwell<br/>bonus layer"| BRIGHT
    TW -.->|"scales trend<br/>response layer"| SPEED

    style SLIDERS fill:#16213e,stroke:#0f3460,color:#e0e1dd
    style OUTPUTS fill:#1a1a2e,stroke:#e94560,color:#fff
    style MULTIPLIERS fill:#533483,stroke:#0f3460,color:#e0e1dd
```

**Example**: With `responsiveness = 0.8` and `speed_global = 1.2`:
- Base mode speed is 20 cm/s (IDLE)
- Responsiveness maps to ×1.24 (lerp between 0.6 and 1.4 at 0.8)
- Global multiplier applies: 20 × 1.24 × 1.2 = **29.8 cm/s**

The sliders interact with each other through the gesture system: `sociability` controls how often gestures fire, `attention_span` influences which gestures are chosen (high attention → more SETTLE and focused gestures), and `memory` suppresses recently-used gestures.

---

## 5. AutoTuning Feedback Loop

The `AutoTuningManager` runs every 5 seconds and continuously adjusts the personality parameters based on observed activity. This creates a light that learns and adapts its character over hours and days.

The key insight is **asymmetric adjustment**: personality sliders (responsiveness, energy, sociability) are only pushed *up* when activity is high — they're never pushed down. Only mean reversion gently brings them back toward home values when things are quiet. This prevents the light from becoming permanently suppressed on a quiet sidewalk.

```mermaid
flowchart TB
    subgraph SENSE["Sense (every 5s)"]
        ACT["<b>Read Activity Levels</b><br/>───────────────<br/>short_activity: 5 min window<br/>medium_activity: 30 min window<br/>long_activity: 1 hour window"]
        AGG["<b>Read Aggression State</b><br/>───────────────<br/>level: 0-1 (EMA smoothed)<br/>seconds_since_engagement: int"]
    end

    subgraph COMPUTE["Compute"]
        TARGET["<b>Adaptive Target</b><br/>───────────────<br/>Method: rolling median<br/>Samples: ~500 (~42 min)<br/>Clamp: 0.03 - 0.40<br/>Purpose: relative busy/quiet"]
        EXCESS["<b>Activity Excess</b><br/>───────────────<br/>Formula: short_activity<br/>minus adaptive_target<br/>Positive = busier than normal<br/>Negative = quieter than normal"]
        ACT --> TARGET --> EXCESS
        AGG --> EXCESS
    end

    subgraph DELTAS["Calculate Deltas"]
        PERS_UP["<b>Personality (up only)</b><br/>───────────────<br/>Params: responsiveness, energy, sociability<br/>When busy: pushed UP<br/>When quiet: NOT pushed down<br/>Max step: 0.03 per cycle"]
        DISP_INV["<b>Display (inverse)</b><br/>───────────────<br/>Params: brightness, speed, pulse globals<br/>When busy: decrease (personality handles it)<br/>When quiet: increase (compensates)<br/>Max step: 0.08 per cycle"]
        EXPL_Q["<b>Exploration</b><br/>───────────────<br/>When quiet: increase (search more)<br/>When busy: decrease (stay focused)<br/>Max step: 0.03 per cycle"]
        EXCESS --> PERS_UP
        EXCESS --> DISP_INV
        EXCESS --> EXPL_Q
    end

    subgraph ADJUST["Adjust and Constrain"]
        REVERT["<b>Mean Reversion</b><br/>───────────────<br/>Target: home values (defaults)<br/>Strength: 0.02 + 0.06 x distance<br/>Type: progressive (stronger when far)<br/>Always active"]
        CURIOSITY["<b>Curiosity Perturbation</b><br/>───────────────<br/>Interval: every 30 seconds<br/>Strength: 0.015<br/>Bias: 60% toward home values<br/>Purpose: explore parameter space"]
        BUDGET["<b>Budget Gate</b><br/>───────────────<br/>Cost: sum(abs(deltas)) x 60<br/>Restore: over ~300 seconds<br/>Effect: scales down changes when depleted<br/>Purpose: prevents runaway drift"]
        CLAMP["<b>Clamp</b><br/>───────────────<br/>Safe floors: prevent zombie light<br/>Soft caps: prevent obnoxious behavior<br/>Hard range: per-parameter min/max<br/>Min step: 0.002 (below = zeroed)"]
        PERS_UP --> REVERT
        DISP_INV --> REVERT
        EXPL_Q --> REVERT
        REVERT --> CURIOSITY --> BUDGET --> CLAMP
    end

    CLAMP --> APPLY["<b>Apply</b><br/>───────────────<br/>Target: MetaParameters<br/>Also: sync slider UI positions"]

    APPLY --> META["<b>MetaParameters</b><br/>───────────────<br/>6 personality sliders<br/>6 global multipliers<br/>Updated for next frame"]

    META -.->|"personality shapes<br/>behavior output"| SENSE

    subgraph DAILY["Daily Learning (midnight)"]
        direction LR
        SNAP["<b>End-of-Day Snapshot</b><br/>───────────────<br/>60% final value<br/>40% midpoint of range"]
        DB["<b>Persist</b><br/>───────────────<br/>Stored per time-of-day<br/>period in database"]
        LOAD["<b>Next Startup</b><br/>───────────────<br/>Load learned values<br/>Blend: 30% toward learned"]
        SNAP --> DB --> LOAD
    end

    CLAMP -.->|"parameter journeys<br/>logged all day"| SNAP
    LOAD -.->|"learned home values"| TARGET

    subgraph OVERRIDES["External Meta-Tuner"]
        JSON["<b>autotune_overrides.json</b><br/>───────────────<br/>Hot-reloaded: every 30 seconds<br/>Can override: home values, safe floors,<br/>caps, curiosity, budget"]
    end

    JSON -.->|"override tuning<br/>hyperparameters"| TARGET

    style SENSE fill:#0d1b2a,stroke:#1b263b,color:#e0e1dd
    style COMPUTE fill:#1b263b,stroke:#415a77,color:#e0e1dd
    style DELTAS fill:#415a77,stroke:#778da9,color:#e0e1dd
    style ADJUST fill:#778da9,stroke:#e0e1dd,color:#0d1b2a
    style DAILY fill:#533483,stroke:#e94560,color:#fff
    style OVERRIDES fill:#1a1a2e,stroke:#e94560,color:#fff
    style META fill:#e94560,stroke:#fff,color:#fff
```

**Tuned parameters and their constraints:**

| Parameter | Min | Max | Safe Floor | Soft Cap | Home |
|---|---|---|---|---|---|
| responsiveness | 0.1 | 0.95 | 0.30 | — | 0.50 |
| energy | 0.1 | 0.95 | 0.25 | — | 0.50 |
| attention_span | 0.1 | 0.95 | — | — | 0.50 |
| sociability | 0.1 | 0.95 | 0.20 | — | 0.50 |
| exploration | 0.1 | 0.95 | — | — | 0.50 |
| memory | 0.1 | 0.95 | — | — | 0.50 |
| brightness_global | 0.3 | 5.0 | 0.50 | 3.0 | 1.00 |
| speed_global | 0.3 | 3.0 | 0.40 | 1.6 | 1.00 |
| pulse_global | 0.3 | 3.0 | — | 2.0 | 1.00 |
| follow_speed_global | 0.3 | 3.0 | — | — | 1.00 |
| dwell_influence | 0.0 | 3.0 | — | — | 1.00 |
| idle_trend_weight | 0.0 | 3.0 | — | — | 1.00 |

**Step sizes**: Personality params max ±0.03 per 5s cycle. Global multipliers max ±0.08. Changes below 0.002 are zeroed to prevent micro-drift.

---

## 6. Multi-Timescale Adaptation

The system processes information across five timescales simultaneously. Fast loops handle moment-to-moment responsiveness, while slow loops gradually shape the light's long-term character. This creates behavior that feels both reactive and intentional.

```mermaid
flowchart TB
    subgraph FRAME["Per-Frame (~33ms)"]
        F1["<b>Mode Switching</b><br/>───────────<br/>Check: active zone occupancy<br/>Apply: stickiness timers"]
        F2["<b>Gesture Triggering</b><br/>───────────<br/>Check: cooldowns, phase gates<br/>Fire: weighted random selection"]
        F3["<b>Proximity Response</b><br/>───────────<br/>Input: person Z distance<br/>Scale: speed, brightness, smoothing"]
        F4["<b>Position Interpolation</b><br/>───────────<br/>Method: exponential decay toward target<br/>Rate: move_speed cm/s"]
        F5["<b>Breathing Overlay</b><br/>───────────<br/>Advance: sine wave phase<br/>Apply: brightness +/-12%, radius +/-6%"]
        F6["<b>Panel Brightness</b><br/>───────────<br/>Calc: 12x distance falloff<br/>Output: DMX values to Art-Net"]
    end

    subgraph SECONDS["Every 1.5 - 5 seconds"]
        S1["<b>Flow Tracking</b><br/>───────────<br/>Interval: 1.5 seconds<br/>Window: 30 seconds sliding<br/>Output: direction (-1 to +1), strength (0-1)"]
        S2["<b>Auto-Tuning Cycle</b><br/>───────────<br/>Interval: 5 seconds<br/>Read: trends, aggression<br/>Write: MetaParameters (12 params)"]
        S3["<b>Attraction Strategy</b><br/>───────────<br/>Rotate: brightness / drift / pause<br/>Track: conversion per strategy<br/>Cooldown: 5 seconds"]
    end

    subgraph MINUTES["Rolling Windows (1m - 60m)"]
        M1["<b>Recent</b> (1 min)<br/>───────────<br/>Use: immediate reactivity<br/>Drives: ready-for-action posture"]
        M2["<b>Short</b> (5 min)<br/>───────────<br/>Output: short_activity weight<br/>Feeds: auto-tuning primary signal"]
        M3["<b>Medium</b> (30 min)<br/>───────────<br/>Output: medium_activity weight<br/>Use: general activity level"]
        M4["<b>Long</b> (1 hour)<br/>───────────<br/>Output: long_activity weight<br/>Use: big-picture energy level"]
        M5["<b>Aggression EMA</b><br/>───────────<br/>Rises: without engagement<br/>Capped: by time-of-day table<br/>Range: 0 - 0.8 depending on hour"]
    end

    subgraph DAILY["Daily"]
        D1["<b>Time-of-Day Modifiers</b><br/>───────────<br/>Periods: late_night / waking /<br/>active / rush / evening<br/>Scales: brightness, pulse, wander Y"]
        D2["<b>Parameter Resets</b><br/>───────────<br/>Times: midnight / 6am / noon / 6pm<br/>Method: 40% blend toward home values"]
        D3["<b>Daily Report</b><br/>───────────<br/>Time: 12:01 AM<br/>Content: engagement stats, population,<br/>parameter journey summary"]
        D4["<b>Daily Learning</b><br/>───────────<br/>Compute: optimal starting values<br/>Granularity: per time-of-day period"]
    end

    subgraph WEEKLY["Weekly"]
        W1["<b>Engagement History</b><br/>───────────<br/>Method: 7-day weighted average<br/>Granularity: by time period"]
        W2["<b>Cap Loosening</b><br/>───────────<br/>Trigger: param consistently hits ceiling<br/>Effect: nudge cap up 10% for next day"]
        W3["<b>Feedback Learning</b><br/>───────────<br/>Buffer: 50 recent engagement contexts<br/>Dims: position x aggression x time x flow<br/>Rate: +/-0.02 per engagement"]
    end

    FRAME --> SECONDS --> MINUTES --> DAILY --> WEEKLY

    style FRAME fill:#e94560,stroke:#fff,color:#fff
    style SECONDS fill:#533483,stroke:#e94560,color:#fff
    style MINUTES fill:#415a77,stroke:#778da9,color:#e0e1dd
    style DAILY fill:#1b263b,stroke:#415a77,color:#e0e1dd
    style WEEKLY fill:#0d1b2a,stroke:#1b263b,color:#e0e1dd
```

**How the timescales connect:**

- The **minute-scale** trend windows feed the **5-second** auto-tuning cycle as activity signals
- The **daily** learning cycle sets starting home values for the next day's auto-tuning
- The **weekly** engagement history informs daily learning (7-day weighted average)
- **Per-frame** rendering reads the MetaParameters that the 5-second tuner has adjusted
- **Flow tracking** (1.5s) feeds both the per-frame flow mode logic and the minute-scale trend analysis

---

## 7. Light Position → Panel DMX Output

The final stage converts the virtual point light into 12 physical DMX values. The `PanelSystem` models the exact physical layout of the installation: 4 lighting units mounted at different X positions, each containing 3 LED panels at different angles.

```mermaid
flowchart TB
    subgraph LIGHT["Virtual Point Light"]
        POS["Position (x, y, z)<br/>from wander/follow system"]
        PULSE["Pulse Phase<br/>sin(phase) oscillation<br/>period = pulse_speed"]
        BRANGE["Brightness Range<br/>brightness_min → brightness_max"]
        FALLOFF["Falloff Radius<br/>40 – 80 cm"]
    end

    subgraph LAYOUT["Physical Panel Layout (top view)"]
        direction LR
        U0["Unit 0<br/>X = −30"]
        U1["Unit 1<br/>X = −110"]
        U2["Unit 2<br/>X = −190"]
        U3["Unit 3<br/>X = −270"]
    end

    subgraph UNIT_DETAIL["Each Unit: 3 Panels"]
        P1["Panel 1 (top)<br/>Y=90, Z=0<br/>faces down"]
        P2["Panel 2 (lower-left)<br/>Y=30, Z=12<br/>angled 22.5°"]
        P3["Panel 3 (lower-right)<br/>Y=30, Z=−12<br/>angled −22.5°"]
    end

    subgraph CALC["Per-Panel Calculation (×12)"]
        DIST["distance = ‖panel_center − light.position‖"]
        CHECK{"distance ><br/>falloff_radius?"}
        OFF["Panel OFF<br/>DMX = 1"]
        FALL["falloff = 1.0 − distance / falloff_radius"]
        INTENSITY["intensity = (sin(phase) + 1) / 2<br/>oscillates 0.0 – 1.0"]
        COMBINE["final = falloff × intensity"]
        DMX["dmx = brightness_min +<br/>final × (brightness_max − brightness_min)<br/>clamped 1 – 255"]
    end

    POS --> DIST
    FALLOFF --> CHECK
    DIST --> CHECK
    CHECK -->|"Yes"| OFF
    CHECK -->|"No"| FALL
    PULSE --> INTENSITY
    FALL --> COMBINE
    INTENSITY --> COMBINE
    BRANGE --> DMX
    COMBINE --> DMX

    subgraph OUTPUT["Art-Net Output"]
        direction LR
        CHANNELS["Channel Map:<br/>CH1: U0-P1 · CH2: U0-P2 · CH3: U0-P3<br/>CH4: U1-P1 · CH5: U1-P2 · CH6: U1-P3<br/>CH7: U2-P1 · CH8: U2-P2 · CH9: U2-P3<br/>CH10: U3-P1 · CH11: U3-P2 · CH12: U3-P3"]
        SEND["Art-Net UDP → 10.42.0.200<br/>Universe 0 · 30 FPS"]
    end

    DMX --> CHANNELS --> SEND

    style LIGHT fill:#533483,stroke:#e94560,color:#fff
    style LAYOUT fill:#1b263b,stroke:#415a77,color:#e0e1dd
    style UNIT_DETAIL fill:#16213e,stroke:#0f3460,color:#e0e1dd
    style CALC fill:#415a77,stroke:#778da9,color:#e0e1dd
    style OUTPUT fill:#e94560,stroke:#fff,color:#fff
```

**The brightness equation in plain language:**

1. Measure the distance from the light to each panel's center point
2. If the panel is outside the falloff radius, it's off (DMX = 1)
3. Otherwise, compute a falloff factor: closer panels are brighter (linear decay)
4. Multiply the falloff by the current pulse intensity (a sine wave that breathes between 0 and 1)
5. Map the result into the brightness range and convert to a DMX byte (1–255)

The falloff radius is the single most impactful parameter on how the light "looks." A small radius (40cm in CROWD mode) creates a tight spotlight that isolates individual panels. A large radius (80cm in IDLE mode) creates a soft wash across multiple panels.

---

## Wander Box: Behavior Inputs to Spatial Motion

The wander box is a 3D bounding volume that constrains where the virtual point light can move. Rather than roaming freely, the light picks random targets inside this box and lerps toward them, creating the characteristic slow drift seen in IDLE mode and the tight tracking seen in ENGAGED mode. Every behavior modifier — flow direction, aggression, engagement contraction — works by reshaping the box, not by steering the light directly.

The box uses a three-layer animation system: `current_wander_box` reflects the raw mode and modifier state, `animated_wander_box` smooths that with an exponential lerp (speed 3.0, ~95% converged in one second), and `WanderBehavior` picks random points inside the animated box at timed intervals. This layered approach prevents jarring jumps when mode or engagement state changes.

```mermaid
flowchart TB
    subgraph BASE ["Base Wander Box (IDLE Default)"]
        BASEBOX["**Default Dimensions**
        ───────────────
        X: -290 to -30 cm
        Y: 0 to 150 cm
        Z: -32 to 28 cm
        Source: light_behavior.py
        Covers: full panel array width"]
    end

    subgraph MODIFIERS ["Behavior Modifiers to Target Box"]
        FLOW["**Flow Positioning**
        ───────────────
        Mode: IDLE only
        Effect: shift X +/-60 cm
        Source: flow_balance trend
        Direction: follows crowd flow"]

        AGG["**Aggression**
        ───────────────
        Mode: IDLE only
        Z expand: +40 cm
        Y expand: +30 cm
        Wander interval: faster
        Trigger: high aggression param"]

        ENGAGE["**Engagement Contraction**
        ───────────────
        Mode: ENGAGED
        Method: contract around people
        1 person: +/-15cm X, +/-35cm Y,
        +/-15cm Z centered on them
        2 people: 70/30 weighted center
        3+: centroid of all positions
        Y offset: +100 cm"]

        MOMENTUM["**Flow Momentum**
        ───────────────
        Mode: FLOW
        Effect: shift X up to +/-40 cm
        Source: flow velocity
        Applied to: current box"]

        DRIFT["**Almost-Engaged Drift**
        ───────────────
        Phase: engagement candidate
        Effect: shift X +/-50 cm
        Direction: toward candidate
        Blended with: engagement timer"]
    end

    subgraph LERP ["Three-Layer Animation"]
        CURRENT["**current_wander_box**
        ───────────────
        Role: base + mode modifiers
        Updates: per calculate_parameters
        Reflects: mode and trend state"]

        ANIMATED["**animated_wander_box**
        ───────────────
        Role: smoothed version of current
        Lerp speed: 3.0 (exponential)
        Convergence: ~95% in 1 second
        Method: per-axis exponential lerp
        dt-scaled for frame rate"]

        CURRENT -->|"exponential lerp"| ANIMATED
    end

    subgraph WANDER ["WanderBehavior Output"]
        PICK["**Random Target Selection**
        ───────────────
        Trigger: wander_interval timer
        Base interval: 2.0 - 5.0s
        Exploration scale: x0.5 to x1.5
        Target: random point inside
        animated_wander_box bounds"]

        MOVE["**Position Lerp**
        ───────────────
        Method: 3% per-frame lerp
        Smoothing: continuous motion
        Override: gesture targets
        Output: smooth (x, y, z)"]

        PICK --> MOVE
    end

    subgraph LIGHTOUT ["To Light System (see Diagram 7)"]
        POINTLIGHT["**PointLight.update()**
        ───────────────
        Input: wander position
        Speed: move_speed param
        Effect: virtual light moves
        through 3D panel space"]

        PANEL["**PanelSystem**
        ───────────────
        Distance: light to each panel
        Falloff: linear within radius
        Result: 12 DMX brightness values"]

        POINTLIGHT --> PANEL
    end

    BASE --> CURRENT
    FLOW --> CURRENT
    AGG --> CURRENT
    ENGAGE --> CURRENT
    MOMENTUM --> CURRENT
    DRIFT --> CURRENT
    ANIMATED --> PICK
    MOVE --> POINTLIGHT

    style BASE fill:#1b263b,stroke:#415a77,color:#e0e1dd
    style MODIFIERS fill:#533483,stroke:#e94560,color:#fff
    style LERP fill:#16213e,stroke:#0f3460,color:#e0e1dd
    style WANDER fill:#0f3460,stroke:#e94560,color:#fff
    style LIGHTOUT fill:#0d1b2a,stroke:#415a77,color:#e0e1dd
    style BASEBOX fill:#1b263b,stroke:#415a77,color:#e0e1dd
    style FLOW fill:#533483,stroke:#e94560,color:#fff
    style AGG fill:#533483,stroke:#e94560,color:#fff
    style ENGAGE fill:#533483,stroke:#e94560,color:#fff
    style MOMENTUM fill:#533483,stroke:#e94560,color:#fff
    style DRIFT fill:#533483,stroke:#e94560,color:#fff
    style CURRENT fill:#16213e,stroke:#0f3460,color:#e0e1dd
    style ANIMATED fill:#16213e,stroke:#0f3460,color:#e0e1dd
    style PICK fill:#0f3460,stroke:#e94560,color:#fff
    style MOVE fill:#0f3460,stroke:#e94560,color:#fff
    style POINTLIGHT fill:#0d1b2a,stroke:#415a77,color:#e0e1dd
    style PANEL fill:#0d1b2a,stroke:#415a77,color:#e0e1dd
```

---

## 8. Engagement Lifecycle

This diagram traces a complete person interaction from arrival to departure, showing which systems activate at each stage. The dwell phases progressively deepen the connection, while the gesture system and breathing overlay add expressive texture throughout.

```mermaid
sequenceDiagram
    participant P as Person
    participant Z as Zone Classification
    participant B as Behavior System
    participant G as Gesture System
    participant L as Light Output

    Note over P,L: ── Person Approaches ──

    P->>Z: Enters passive zone
    Z->>B: passive_count += 1
    B->>G: Trigger ACKNOWLEDGE
    G->>L: Brief move toward passerby

    Note over P,L: ── Person Enters Active Zone ──

    P->>Z: Crosses into active zone
    Z->>B: active_count += 1
    B->>B: Mode: IDLE → ENGAGED (immediate)
    B->>G: Trigger WELCOME
    G->>L: Entry pulse: +25 brightness flash
    B->>L: Transition interpolation (0.8s)

    rect rgb(30, 40, 70)
        Note over B,L: NOTICE PHASE (0–3s)
        B->>L: Light turns toward person
        B->>L: Brightness ramping up
        Note right of G: No positional gestures yet
    end

    rect rgb(40, 50, 90)
        Note over B,L: GREET PHASE (3–10s)
        B->>L: Brightness increase settled
        B->>L: Breathing overlay begins ramping in (8s ramp)
        loop Every 8–15s
            G->>L: NOD (1.2s, most common)
            G->>L: LEAN (1.5s, leaning in)
            G->>L: BREATHE (4.0s, shared rhythm)
        end
    end

    rect rgb(50, 60, 110)
        Note over B,L: ENGAGE PHASE (10–30s)
        B->>L: Breathing at full depth (±12% brightness, ±6% radius)
        B->>L: Tighter tracking, brighter output
        loop Every 10–20s
            G->>L: SWAY (3.0s, lateral oscillation)
            G->>L: ORBIT (4.0s, lazy circle)
            G->>L: BREATHE (5.0s, deeper)
            G->>L: NOD / LEAN (carried forward)
        end
    end

    rect rgb(60, 70, 130)
        Note over B,L: BOND PHASE (30s+)
        B->>L: Maximum intimacy
        B->>L: Very settled, infrequent gestures
        loop Every 15–30s
            G->>L: SWAY (4.0s)
            G->>L: ORBIT (5.0s)
            G->>L: SETTLE (3.0s, tighten in closer)
            G->>L: BREATHE (6.0s)
        end
    end

    Note over P,L: ── Person Leaves ──

    P->>Z: Exits active zone
    Z->>B: active_count = 0
    B->>B: Start 5s stickiness timer

    alt Dwell was > 5s and no one remains
        B->>G: Trigger FAREWELL
        G->>L: Reluctant move toward last position
    end

    Note over B: 5s passes with no one...
    B->>B: Mode: ENGAGED → IDLE
    B->>L: Transition interpolation (3.0s slow goodbye)
    B->>L: Breathing overlay ramps out
    B->>L: Return to gentle wandering
```

**What the person experiences:**

1. Walking past, the light briefly acknowledges them — a subtle flicker of awareness
2. Stepping under the panels, the light immediately locks on with a welcoming pulse
3. Standing still, they notice the light beginning to breathe — a slow shared rhythm
4. After 10 seconds, the light starts to sway and orbit gently, as if comfortable in their presence
5. After 30 seconds, the light settles in close — maximum intimacy, minimal movement
6. Walking away, the light lingers, reluctantly following their last position before slowly fading back to its wandering state

The entire interaction is shaped by the current MetaParameters — a highly social, energetic personality will greet faster, gesture more frequently, and track more tightly. A shy, calm personality will be subtle and slow, with longer pauses between gestures.

---

## Appendix: How Everything Connects

This final diagram shows the complete system with all feedback loops visible at once — the "full picture" view of how inputs, processing layers, adaptation loops, and outputs relate to each other.

```mermaid
flowchart TB
    CAM["<b>Camera + YOLO</b><br/>───────────────<br/>Detection: person bounding boxes<br/>Output: OSC x, z per person"] -->|"OSC x,z"| TRACKER["<b>Person Manager</b><br/>───────────────<br/>Zone: active / passive classify<br/>Tracking: velocity, dwell time"]

    TRACKER -->|"active/passive counts<br/>person positions"| BEHAVIOR["<b>Behavior System</b><br/>───────────────<br/>Mode: state machine<br/>Dwell: phase tracking<br/>Gestures: event + interaction"]

    TRACKER -->|"zone crossings<br/>per timescale"| TRENDS["<b>Trend Analysis</b><br/>───────────────<br/>Windows: 1m / 5m / 30m / 1h<br/>Output: activity weights"]

    TRACKER -->|"velocity vectors"| FLOWTRACK["<b>Flow Tracking</b><br/>───────────────<br/>Direction: -1 to +1<br/>Strength: 0 to 1<br/>Window: 30s, update 1.5s"]

    TRENDS -->|"activity weights<br/>anticipation, energy"| BEHAVIOR
    FLOWTRACK -->|"flow direction"| BEHAVIOR

    TRENDS -->|"short_activity<br/>medium, long"| AUTOTUNE["<b>AutoTuning Manager</b><br/>───────────────<br/>Cycle: every 5 seconds<br/>Params: 12 (6 sliders + 6 globals)<br/>Method: adaptive target + deltas"]

    BEHAVIOR -->|"aggression state"| AUTOTUNE

    AUTOTUNE -->|"adjusted values"| META["<b>MetaParameters</b><br/>───────────────<br/>Personality: 6 sliders (0-1)<br/>Globals: 6 multipliers"]

    META -->|"personality +<br/>global multipliers"| BEHAVIOR

    BEHAVIOR -->|"behavior_params<br/>7 output values"| PIPELINE["<b>17-Layer Pipeline</b><br/>───────────────<br/>See: Diagram 3"]

    PIPELINE -->|"final params"| POINTLIGHT["<b>Point Light</b><br/>───────────────<br/>Position: x, y, z<br/>Brightness: min/max<br/>Falloff: radius<br/>Pulse: phase"]

    POINTLIGHT -->|"light state"| PANELSYS["<b>Panel System</b><br/>───────────────<br/>Panels: 12 (4 units x 3)<br/>Calc: distance to DMX"]

    PANELSYS -->|"12 DMX channels"| ARTNET["<b>Art-Net Output</b><br/>───────────────<br/>Target: 10.42.0.200<br/>Universe: 0 / Rate: 30 FPS"]

    POINTLIGHT -->|"state snapshot"| WEBSOCKET["<b>WebSocket Broadcast</b><br/>───────────────<br/>Clients: Public 3D Viewer<br/>Rate: ~15 FPS"]

    BEHAVIOR -->|"engagement context<br/>snapshots"| FEEDBACK["<b>Feedback Learning</b><br/>───────────────<br/>Buffer: 50 contexts<br/>Dims: position x time x flow<br/>Rate: +/-0.02 per event"]

    FEEDBACK -->|"learned weights"| BEHAVIOR

    AUTOTUNE -->|"parameter journey<br/>end-of-day"| DAILY["<b>Daily Learning</b><br/>───────────────<br/>Compute: optimal starts<br/>Granularity: per time-of-day"]

    DAILY -->|"learned home values<br/>30% blend on startup"| AUTOTUNE

    DAILY -->|"daily report"| DB[("<b>Tracking Database</b><br/>───────────────<br/>Hourly stats, learnings,<br/>engagement history")]

    TRENDS -->|"raw events"| DB
    DB -->|"historical patterns"| TRENDS
    DB -->|"7-day weighted avg"| DAILY

    OVERRIDES["<b>autotune_overrides.json</b><br/>───────────────<br/>Hot-reload: every 30s<br/>Overrides: home values, caps, budget"] -.->|"hot-reload"| AUTOTUNE

    TOD["<b>Time of Day</b><br/>───────────────<br/>Maps: hour to period<br/>Scales: brightness, pulse,<br/>wander Y, aggression cap"] -->|"modifiers"| BEHAVIOR

    style CAM fill:#e94560,stroke:#fff,color:#fff
    style ARTNET fill:#e94560,stroke:#fff,color:#fff
    style WEBSOCKET fill:#533483,stroke:#e94560,color:#fff
    style META fill:#e94560,stroke:#fff,color:#fff
    style DB fill:#0d1b2a,stroke:#415a77,color:#e0e1dd
    style AUTOTUNE fill:#533483,stroke:#e94560,color:#fff
    style DAILY fill:#1b263b,stroke:#415a77,color:#e0e1dd
    style FEEDBACK fill:#1b263b,stroke:#415a77,color:#e0e1dd
    style OVERRIDES fill:#1a1a2e,stroke:#778da9,color:#e0e1dd
```

---

# Part 2: Camera Tracking System

The following diagrams explain how two RTSP cameras capture video, detect people
using YOLO, project detections to real-world floor coordinates via calibration,
fuse cross-camera observations, and output tracked positions over OSC. This is
the upstream data source that feeds the behavior system described in Part 1.

---

## Diagram 9 — Camera System Overview

High-level pipeline from physical cameras to the OSC messages consumed by the light controller. Two Reolink PoE cameras feed RTSP streams into threaded capture buffers, YOLO detects people, calibration projects pixel positions onto the floor plane, and cross-camera fusion produces stable world-coordinate tracks. The final OSC messages carry person count and per-person (id, x, z) positions to `lightController_osc.py` at roughly 25 updates per second.

```mermaid
flowchart LR
    CAM1["**Camera 1**
    ───────────────
    Model: Reolink RLC-520A
    IP: 10.42.0.75
    Protocol: RTSP port 555
    Resolution: 2048 x 1536
    FPS: 25"]

    CAM2["**Camera 2**
    ───────────────
    Model: Reolink RLC-520A
    IP: 10.42.0.172
    Protocol: RTSP port 555
    Resolution: 2048 x 1536
    FPS: 25"]

    RC["**RobustCamera**
    ───────────────
    Threads: 1 daemon per camera
    Buffer: single-copy frame
    Flush: grab() x3 per read
    Reconnect: auto on failure
    Max frame age: 0.5s"]

    YOLO["**YOLO 11n Detection**
    ───────────────
    Input: 416px wide resize
    Class: person only (id 0)
    Confidence: 0.10 - 0.80
    Output: bounding boxes"]

    CAL["**Calibration**
    ───────────────
    Method: ray-plane intersect
    Floor plane: Y = -66 cm
    Pre-computed: R_T, K_inv
    Output: world (X, Z) cm"]

    FUSE["**TrackingFusion**
    ───────────────
    Merge: cross-camera only
    Threshold: 50 - 300 cm
    Smoothing: EMA alpha 0.03
    Velocity: prediction + correct"]

    OSC["**OSC Output**
    ───────────────
    Target: 127.0.0.1:7000
    /tracker/count n
    /tracker/person/id x z
    Protocol: UDP"]

    LC["**lightController_osc.py**
    ───────────────
    Zone classification
    Behavior system
    Light output"]

    CAM1 --> RC
    CAM2 --> RC
    RC --> YOLO
    YOLO --> CAL
    CAL --> FUSE
    FUSE --> OSC
    OSC --> LC

    style CAM1 fill:#1a1a2e,stroke:#e94560,color:#fff
    style CAM2 fill:#1a1a2e,stroke:#e94560,color:#fff
    style RC fill:#16213e,stroke:#0f3460,color:#e0e1dd
    style YOLO fill:#0f3460,stroke:#e94560,color:#fff
    style CAL fill:#533483,stroke:#e94560,color:#fff
    style FUSE fill:#1b263b,stroke:#415a77,color:#e0e1dd
    style OSC fill:#0d1b2a,stroke:#415a77,color:#e0e1dd
    style LC fill:#1a1a2e,stroke:#778da9,color:#e0e1dd
```

---

## Diagram 10 — Physical Setup and Coordinate System

Camera positions, ArUco calibration markers, and tracking zones placed in the real-world coordinate system. The origin sits at the back-right corner of Panel Unit 0 at floor level, with X running negative toward Unit 3, Y pointing up, and Z extending outward into the sidewalk. Seven ArUco markers at known 3D positions provide the calibration reference, and two depth zones (active and passive) define the engagement boundaries used by the behavior system.

```mermaid
flowchart TB
    subgraph COORD ["Coordinate System"]
        ORIGIN["**Origin**
        ───────────────
        Location: back-right corner
        of Panel Unit 0, floor level
        X: negative toward Unit 3
        Y: positive upward
        Z: positive into tracking zone"]
    end

    subgraph LEVELS ["Reference Levels"]
        FLOOR["**Storefront Floor**
        ───────────────
        Y: 0 cm
        Role: reference plane"]

        LEDGE["**Camera Ledge**
        ───────────────
        Y: -15 cm
        Height: 51 cm above street"]

        STREET["**Street Level**
        ───────────────
        Y: -66 cm
        Role: pedestrian floor plane
        Used by: calibration intersect"]
    end

    subgraph CAMERAS ["Camera Positions"]
        C1["**Camera 1 (Right)**
        ───────────────
        Position: X=-30, Y=-15, Z=78
        IP: 10.42.0.75
        Aligned with: Unit 0 center
        FOV: 80 deg horiz, 48 deg vert
        Pitch: 22 deg down
        Yaw: ~25 deg left"]

        C2["**Camera 2 (Left)**
        ───────────────
        Position: X=-270, Y=-15, Z=78
        IP: 10.42.0.172
        Aligned with: Unit 3 center
        FOV: 80 deg horiz, 48 deg vert
        Pitch: 22 deg down
        Yaw: ~25 deg right"]
    end

    subgraph PANELS ["Panel Units (60 cm wide, 80 cm spacing)"]
        P0["**Unit 0**
        ───────────────
        Center X: -30
        Edges: 0 to -60"]

        P1["**Unit 1**
        ───────────────
        Center X: -110
        Edges: -80 to -140"]

        P2["**Unit 2**
        ───────────────
        Center X: -190
        Edges: -160 to -220"]

        P3["**Unit 3**
        ───────────────
        Center X: -270
        Edges: -240 to -300"]
    end

    subgraph ZONES ["Tracking Zones"]
        ACTIVE["**Active Zone**
        ───────────────
        X: -350 to 50
        Z: 78 to 283
        Depth: ~2 m from panels
        Role: engaged interaction"]

        PASSIVE["**Passive Zone**
        ───────────────
        X: -350 to 50
        Z: 283 to 553
        Depth: ~2.7 m beyond active
        Role: sidewalk passersby"]
    end

    COORD --> LEVELS
    COORD --> CAMERAS
    CAMERAS --> PANELS
    PANELS --> ZONES

    style COORD fill:#0d1b2a,stroke:#778da9,color:#e0e1dd
    style LEVELS fill:#1a1a2e,stroke:#778da9,color:#e0e1dd
    style CAMERAS fill:#16213e,stroke:#e94560,color:#fff
    style PANELS fill:#1b263b,stroke:#415a77,color:#e0e1dd
    style ZONES fill:#0f3460,stroke:#e94560,color:#fff
    style ORIGIN fill:#0d1b2a,stroke:#415a77,color:#e0e1dd
    style FLOOR fill:#1a1a2e,stroke:#778da9,color:#e0e1dd
    style LEDGE fill:#1a1a2e,stroke:#778da9,color:#e0e1dd
    style STREET fill:#1a1a2e,stroke:#778da9,color:#e0e1dd
    style C1 fill:#16213e,stroke:#e94560,color:#fff
    style C2 fill:#16213e,stroke:#e94560,color:#fff
    style P0 fill:#1b263b,stroke:#415a77,color:#e0e1dd
    style P1 fill:#1b263b,stroke:#415a77,color:#e0e1dd
    style P2 fill:#1b263b,stroke:#415a77,color:#e0e1dd
    style P3 fill:#1b263b,stroke:#415a77,color:#e0e1dd
    style ACTIVE fill:#0f3460,stroke:#e94560,color:#fff
    style PASSIVE fill:#0f3460,stroke:#e94560,color:#fff
```

---

## Diagram 11 — Camera Capture and YOLO Detection

Each camera runs a dedicated daemon thread that continuously reads RTSP frames and stores the most recent one under a lock. The main tracking loop retrieves these frames, resizes them to 416 pixels wide for speed, and passes them through YOLO 11n to detect people. Detected bounding boxes are scaled back to full resolution and cached per camera so that unchanged frames reuse the previous detection without re-running inference.

```mermaid
flowchart TB
    subgraph THREAD ["RobustCamera Thread (1 per camera)"]
        CONNECT["**_connect()**
        ───────────────
        Backend: cv2.CAP_FFMPEG
        Buffer size: 1 frame
        Timeout: 10s connection
        Retry: 2s between attempts"]

        LOOP["**_capture_loop()**
        ───────────────
        Method: cap.read()
        Store: frame under lock
        Buffer flush: cap.grab() x3
        Purpose: discard stale RTSP
        Failure limit: 30 consecutive"]

        RECONNECT["**Reconnect Logic**
        ───────────────
        Trigger: 30+ read failures
        Action: disconnect, sleep 2s
        Stats: reconnect counter
        Error log: every 10th failure"]

        CONNECT --> LOOP
        LOOP -->|failure > 30| RECONNECT
        RECONNECT --> CONNECT
    end

    subgraph GETFRAME ["Frame Retrieval"]
        GET["**get_frame()**
        ───────────────
        Lock: threading.Lock
        Copy: single np.ndarray copy
        Age check: < 0.5s max
        Returns: (ok, frame, is_new)
        Tracking: frame_num counter"]
    end

    subgraph DETECT ["Tracker._detect_all()"]
        RESIZE["**Resize**
        ───────────────
        Target width: 416 px
        Method: cv2.INTER_LINEAR
        Aspect: preserved
        Scale factor: saved for later"]

        PREDICT["**YOLO Predict**
        ───────────────
        Model: yolo11n.pt
        Classes: [0] person only
        Confidence: slider value
        Device: auto (CPU/CUDA)
        Verbose: False"]

        SCALE["**Scale to Full Resolution**
        ───────────────
        Input: YOLO boxes (416px space)
        Operation: multiply by scale_inv
        Output: boxes in 2048x1536 space
        Format: (x1, y1, x2, y2, conf)"]

        CACHE["**Detection Cache**
        ───────────────
        Storage: cfg[_last_boxes]
        Per camera: list of detections
        Reused when: frame is not new
        Cleared on: new frame processed"]

        RESIZE --> PREDICT
        PREDICT --> SCALE
        SCALE --> CACHE
    end

    THREAD --> GET
    GET -->|frame| DETECT

    style THREAD fill:#16213e,stroke:#0f3460,color:#e0e1dd
    style GETFRAME fill:#1a1a2e,stroke:#778da9,color:#e0e1dd
    style DETECT fill:#0f3460,stroke:#e94560,color:#fff
    style CONNECT fill:#16213e,stroke:#0f3460,color:#e0e1dd
    style LOOP fill:#16213e,stroke:#0f3460,color:#e0e1dd
    style RECONNECT fill:#1a1a2e,stroke:#e94560,color:#fff
    style GET fill:#1a1a2e,stroke:#778da9,color:#e0e1dd
    style RESIZE fill:#0f3460,stroke:#e94560,color:#fff
    style PREDICT fill:#0f3460,stroke:#e94560,color:#fff
    style SCALE fill:#0f3460,stroke:#e94560,color:#fff
    style CACHE fill:#1b263b,stroke:#415a77,color:#e0e1dd
```

---

## Diagram 12 — Calibration: Pixel to World Coordinates

The ray-plane intersection math that transforms a bounding-box foot position in image pixels to a real-world floor position in centimeters. For each detection, the bottom-center of the bounding box is treated as the person's feet; that pixel is undistorted, projected into a camera-space ray using the inverse intrinsic matrix, rotated into world space, and intersected with the known floor plane at Y = -66 cm. The result is a (world_x, world_z) pair in centimeters that feeds directly into the fusion stage.

```mermaid
flowchart TB
    subgraph INPUT ["Bounding Box Input"]
        BBOX["**bbox_to_floor()**
        ───────────────
        Input: x1, y1, x2, y2
        Foot X: (x1 + x2) / 2
        Foot Y: y2 (box bottom)
        Assumption: feet touch floor"]
    end

    subgraph PRECOMPUTED ["Pre-Computed at Load Time"]
        LOAD["**CalibrationManager.__init__**
        ───────────────
        Source: camera_calibration.json
        Per camera:
          K = camera matrix 3x3
          K_inv = np.linalg.inv(K)
          R, _ = cv2.Rodrigues(rvec)
          R_T = R.T (transpose)
          cam_pos = -R_T @ tvec
          dist = distortion coeffs"]
    end

    subgraph INTRINSICS ["Camera Intrinsics"]
        KINTR["**Camera Matrix (K)**
        ───────────────
        fx: 1220.36 px
        fy: 1220.36 px
        cx: 1024.0 (image center)
        cy: 768.0 (image center)
        Source: 80 deg HFOV estimate
        Formula: 1024 / tan(40 deg)
        Distortion: all zeros"]
    end

    subgraph RAYSTEPS ["Ray-Plane Intersection Steps"]
        STEP1["**Step 1: Undistort**
        ───────────────
        Method: cv2.undistortPoints
        Input: (foot_x, foot_y)
        Params: K, dist, P=K
        Output: (ux, uy)"]

        STEP2["**Step 2: Camera Ray**
        ───────────────
        Pixel to ray: K_inv @ [ux, uy, 1]
        Normalize: ray / norm(ray)
        Space: camera coordinates"]

        STEP3["**Step 3: World Ray**
        ───────────────
        Transform: R_T @ ray_cam
        Space: world coordinates
        Direction: from camera outward"]

        STEP4["**Step 4: Intersect Floor**
        ───────────────
        Floor plane: Y = -66 cm
        Parameter: t = (floor_y - cam_y)
                       / ray_world_y
        Guard: abs(ray_y) > 1e-6
        Guard: t > 0 (forward only)"]

        STEP5["**Step 5: Hit Point**
        ───────────────
        Compute: cam_pos + t * ray_world
        Extract: (hit_x, hit_z)
        Units: centimeters
        X: lateral position
        Z: depth from panels"]

        STEP1 --> STEP2
        STEP2 --> STEP3
        STEP3 --> STEP4
        STEP4 --> STEP5
    end

    subgraph OUTPUT ["World Position"]
        RESULT["**Return Value**
        ───────────────
        Format: (world_x, world_z)
        Units: centimeters
        Plane: Y = -66 (street)
        Failure: returns None"]
    end

    INPUT --> STEP1
    PRECOMPUTED --> STEP2
    INTRINSICS --> STEP1
    STEP5 --> OUTPUT

    style INPUT fill:#1a1a2e,stroke:#778da9,color:#e0e1dd
    style PRECOMPUTED fill:#533483,stroke:#e94560,color:#fff
    style INTRINSICS fill:#533483,stroke:#e94560,color:#fff
    style RAYSTEPS fill:#0f3460,stroke:#e94560,color:#fff
    style OUTPUT fill:#0d1b2a,stroke:#415a77,color:#e0e1dd
    style BBOX fill:#1a1a2e,stroke:#778da9,color:#e0e1dd
    style LOAD fill:#533483,stroke:#e94560,color:#fff
    style KINTR fill:#533483,stroke:#e94560,color:#fff
    style STEP1 fill:#0f3460,stroke:#e94560,color:#fff
    style STEP2 fill:#0f3460,stroke:#e94560,color:#fff
    style STEP3 fill:#0f3460,stroke:#e94560,color:#fff
    style STEP4 fill:#0f3460,stroke:#e94560,color:#fff
    style STEP5 fill:#0f3460,stroke:#e94560,color:#fff
    style RESULT fill:#0d1b2a,stroke:#415a77,color:#e0e1dd
```

---

## Diagram 13 — Cross-Camera Fusion and Temporal Tracking

Detections from the two cameras are fused spatially and then matched temporally to produce stable, smoothed tracks. Spatial fusion uses greedy nearest-neighbor matching across cameras only (same-camera detections are never merged) with a configurable distance threshold. Temporal tracking predicts each existing track's next position using its velocity, matches incoming detections to predictions, applies exponential moving average smoothing, and prunes tracks unseen for more than 60 frames.

```mermaid
flowchart TB
    subgraph INPUTDETS ["Per-Frame World Detections"]
        DETS["**Detection List**
        ───────────────
        Per detection:
          x: world X (cm)
          z: world Z (cm)
          camera: source camera name
          conf: YOLO confidence
        Typical: 0-6 detections/frame"]
    end

    subgraph FUSESTAGE ["Stage 1: Spatial Fusion"]
        FUSE_ALG["**_fuse() Algorithm**
        ───────────────
        Method: greedy nearest-neighbor
        Constraint: cross-camera only
        Same-camera: never merged
        Threshold: fusion_dist squared
        Default: 150 cm (22500 sq)"]

        FUSE_MERGE["**Merge Operation**
        ───────────────
        Match: closest unmatched pair
        across different cameras
        Position: average of cluster
        Result: single fused position
        Unmatched: kept as-is"]

        FUSE_ALG --> FUSE_MERGE
    end

    subgraph MATCHSTAGE ["Stage 2: Temporal Matching"]
        PREDICT["**Velocity Prediction**
        ───────────────
        For each existing track:
          pred_x = track_x + vel_x
          pred_z = track_z + vel_z
        Purpose: anticipate movement
        between frames"]

        MATCH["**Track Matching**
        ───────────────
        Metric: distance to predicted pos
        Threshold: fusion_dist * 0.6
        Default: 90 cm (8100 sq)
        Reason: prevent cross-person
        jumps at closer range
        Method: nearest-neighbor"]

        NEWTRACK["**New Track Creation**
        ───────────────
        Trigger: unmatched detection
        ID: incrementing _next_id
        Initial velocity: (0, 0)
        Initial position: raw detection"]

        PREDICT --> MATCH
        MATCH -->|unmatched| NEWTRACK
    end

    subgraph SMOOTHSTAGE ["Stage 3: EMA Smoothing"]
        EMA["**Exponential Moving Average**
        ───────────────
        Alpha: 0.01 - 0.20 (slider)
        Default: 0.03 (very smooth)
        Formula:
          new_x = pred_x + a*(raw - pred_x)
          vel_x += a*((new_x - old_x) - vel_x)
        Effect: low alpha = smooth path
                high alpha = responsive"]
    end

    subgraph PRUNE ["Track Lifecycle"]
        LIFECYCLE["**Pruning**
        ───────────────
        Lost counter: frames unseen
        Threshold: max_lost_frames
        Default: 60 frames (2.4s)
        Action: delete track entirely
        Effect: ID freed for reuse"]
    end

    subgraph TRACKOUT ["Tracked Output"]
        TRACKS["**Track List**
        ───────────────
        Per track:
          id: stable integer
          x: smoothed world X (cm)
          z: smoothed world Z (cm)
        Rate: 25 updates/second
        Consumers: OSC sender"]
    end

    INPUTDETS --> FUSESTAGE
    FUSESTAGE --> MATCHSTAGE
    MATCHSTAGE --> SMOOTHSTAGE
    SMOOTHSTAGE --> TRACKOUT
    MATCHSTAGE --> PRUNE
    PRUNE -->|remove stale| MATCHSTAGE

    style INPUTDETS fill:#1a1a2e,stroke:#778da9,color:#e0e1dd
    style FUSESTAGE fill:#0f3460,stroke:#e94560,color:#fff
    style MATCHSTAGE fill:#533483,stroke:#e94560,color:#fff
    style SMOOTHSTAGE fill:#16213e,stroke:#0f3460,color:#e0e1dd
    style PRUNE fill:#1b263b,stroke:#415a77,color:#e0e1dd
    style TRACKOUT fill:#0d1b2a,stroke:#415a77,color:#e0e1dd
    style DETS fill:#1a1a2e,stroke:#778da9,color:#e0e1dd
    style FUSE_ALG fill:#0f3460,stroke:#e94560,color:#fff
    style FUSE_MERGE fill:#0f3460,stroke:#e94560,color:#fff
    style PREDICT fill:#533483,stroke:#e94560,color:#fff
    style MATCH fill:#533483,stroke:#e94560,color:#fff
    style NEWTRACK fill:#533483,stroke:#e94560,color:#fff
    style EMA fill:#16213e,stroke:#0f3460,color:#e0e1dd
    style LIFECYCLE fill:#1b263b,stroke:#415a77,color:#e0e1dd
    style TRACKS fill:#0d1b2a,stroke:#415a77,color:#e0e1dd
```

---

## Diagram 14 — ArUco Calibration Process

The one-time camera calibration uses seven ArUco markers placed at known 3D positions to compute each camera's rotation and translation via `cv2.solvePnP`. Each marker provides four corner correspondences between known world coordinates and detected image pixels, giving enough constraints for a robust pose estimate. The solver tries multiple corner orderings and validates results against reprojection error, camera position sanity, and expected location drift before saving the best solution to `camera_calibration.json`.

```mermaid
flowchart TB
    subgraph MARKERS ["ArUco Marker Layout"]
        MSETUP["**7 Markers (ArUco 4x4_50)**
        ───────────────
        Size: 20 cm per marker
        Front row (Z=168):
          ID 0: X=-30 (Cam 1 only)
          ID 1: X=-150 (shared)
          ID 2: X=-270 (Cam 2 only)
        Back row (Z=219):
          ID 3: X=-30 (Cam 1 only)
          ID 6: X=-150 (shared)
          ID 4: X=-270 (Cam 2 only)
        Subway wall (Z=628):
          ID 5: X=-150, Y=-15 (shared)"]

        VISIBILITY["**Camera Visibility**
        ───────────────
        Camera 1 sees: 0, 1, 3, 5, 6
        Camera 2 sees: 1, 2, 4, 5, 6
        Shared markers: 1, 5, 6
        Min required: 3 per camera
        Points per marker: 4 corners"]
    end

    subgraph DETECTION ["Marker Detection"]
        DETECT_M["**ArUco Detection**
        ───────────────
        Convert: BGR to grayscale
        Dictionary: DICT_4X4_50
        Threshold: adaptive 3-23 step 4
        Perimeter: 0.02 - 4.0 rate
        Border error: 0.25 max"]

        REFINE["**Corner Refinement**
        ───────────────
        Method: CORNER_REFINE_SUBPIX
        Sub-pixel: cv2.cornerSubPix
        Window: 3x3
        Iterations: 50 max
        Accuracy: 0.01 epsilon"]

        DETECT_M --> REFINE
    end

    subgraph SOLVE ["Pose Estimation"]
        CORRESPOND["**3D-2D Correspondences**
        ───────────────
        Per marker: 4 corner points
        3D: world coords + offsets
        Horizontal offsets (cm):
          (-10,0,-10) (10,0,-10)
          (10,0,10) (-10,0,10)
        Vertical offsets (cm):
          (-10,10,0) (10,10,0)
          (10,-10,0) (-10,-10,0)"]

        SOLVEPNP["**cv2.solvePnP**
        ───────────────
        Primary: SOLVEPNP_SQPNP
        Fallback: SOLVEPNP_ITERATIVE
        with extrinsic guess
        Corner combos: 16 rotations
        tries (4 horiz x 4 vert)
        Refinement: solvePnPRefineLM"]

        VALIDATE["**Validation Gates**
        ───────────────
        Reprojection error: < 100 px
        Camera Z: must be positive
        Position drift: < 50 cm from
        expected camera location
        Selection: best valid solution"]

        CORRESPOND --> SOLVEPNP
        SOLVEPNP --> VALIDATE
    end

    subgraph SAVE ["Calibration Output"]
        CALOUT["**camera_calibration.json**
        ───────────────
        Per camera:
          rvec: rotation (3x1)
          tvec: translation (3x1)
          camera_matrix: K (3x3)
          dist_coeffs: (5 values)
          image_size: [2048, 1536]
        Global:
          synth_width: 800
          synth_height: 600
          cm_per_pixel: 1.0"]
    end

    MARKERS --> DETECTION
    DETECTION --> SOLVE
    SOLVE --> SAVE

    style MARKERS fill:#1a1a2e,stroke:#778da9,color:#e0e1dd
    style DETECTION fill:#0f3460,stroke:#e94560,color:#fff
    style SOLVE fill:#533483,stroke:#e94560,color:#fff
    style SAVE fill:#0d1b2a,stroke:#415a77,color:#e0e1dd
    style MSETUP fill:#1a1a2e,stroke:#778da9,color:#e0e1dd
    style VISIBILITY fill:#1a1a2e,stroke:#778da9,color:#e0e1dd
    style DETECT_M fill:#0f3460,stroke:#e94560,color:#fff
    style REFINE fill:#0f3460,stroke:#e94560,color:#fff
    style CORRESPOND fill:#533483,stroke:#e94560,color:#fff
    style SOLVEPNP fill:#533483,stroke:#e94560,color:#fff
    style VALIDATE fill:#533483,stroke:#e94560,color:#fff
    style CALOUT fill:#0d1b2a,stroke:#415a77,color:#e0e1dd
```

---

## Diagram 15 — Main Loop and Tunable Parameters

The tracker's main loop runs at a fixed 25 FPS, cycling through detection, fusion, OSC output, and optional visualization on every frame. Three live slider parameters — confidence threshold, fusion distance, and EMA smoothing alpha — let the operator tune the tradeoff between responsiveness and stability without restarting. Periodic background tasks auto-save settings every 5 seconds, log health metrics every 5 minutes, and reload the YOLO model hourly to prevent tracker drift.

```mermaid
flowchart TB
    subgraph MAINLOOP ["Tracker._run() Main Loop"]
        TIMING["**Frame Rate Control**
        ───────────────
        Target: 25 FPS (40ms/frame)
        Method: sleep(0.001) while
        elapsed < frame_interval
        Clock: time.time()"]

        STEP_DETECT["**Step 1: _detect_all()**
        ───────────────
        For each camera:
          get_frame()
          resize to 416px
          YOLO predict
          scale boxes back
          bbox_to_floor()
        Output: world detections list"]

        STEP_FUSE["**Step 2: fusion.process()**
        ───────────────
        Spatial fusion (cross-camera)
        Temporal matching (velocity)
        EMA smoothing (alpha=0.03)
        Output: tracked (id, x, z) list"]

        STEP_OSC["**Step 3: _send_osc()**
        ───────────────
        /tracker/count n
        /tracker/person/id x z
        Target: 127.0.0.1:7000
        Error log: 1st + every 100th"]

        STEP_RENDER["**Step 4: _render()**
        ───────────────
        Skip if: headless mode
        Display: camera feeds +
        bounding boxes + world coords
        Key 'q': quit
        Key 's': save settings"]

        TIMING --> STEP_DETECT
        STEP_DETECT --> STEP_FUSE
        STEP_FUSE --> STEP_OSC
        STEP_OSC --> STEP_RENDER
        STEP_RENDER --> TIMING
    end

    subgraph PERIODIC ["Periodic Tasks"]
        SAVE_S["**Settings Auto-Save**
        ───────────────
        Interval: every 5 seconds
        Condition: settings dirty flag
        File: tracker_settings.json"]

        HEALTH["**Health Logging**
        ───────────────
        Interval: every 300s (5 min)
        Reports: FPS, frame counts
        camera health, track count"]

        RESET["**YOLO Reset**
        ───────────────
        Interval: every 3600s (1 hr)
        Action: reload model state
        Purpose: prevent tracker drift"]
    end

    subgraph SLIDERS ["Live Tunable Parameters (3 Sliders)"]
        S1["**Confidence**
        ───────────────
        Range: 0.10 - 0.80
        Default: 0.40
        Affects: YOLO detection threshold
        Low: more detections, more noise
        High: fewer detections, precise"]

        S2["**Fusion Distance**
        ───────────────
        Range: 50 - 300 cm
        Default: 150 cm
        Affects: cross-camera merge radius
        and track match (60% of value)
        Low: strict matching
        High: loose matching"]

        S3["**Smoothing**
        ───────────────
        Range: 0.01 - 0.20
        Default: 0.03
        Affects: EMA alpha for position
        Low: very smooth, laggy path
        High: responsive, jittery path"]
    end

    MAINLOOP --> PERIODIC
    SLIDERS --> MAINLOOP

    style MAINLOOP fill:#16213e,stroke:#0f3460,color:#e0e1dd
    style PERIODIC fill:#1b263b,stroke:#415a77,color:#e0e1dd
    style SLIDERS fill:#533483,stroke:#e94560,color:#fff
    style TIMING fill:#16213e,stroke:#0f3460,color:#e0e1dd
    style STEP_DETECT fill:#0f3460,stroke:#e94560,color:#fff
    style STEP_FUSE fill:#0f3460,stroke:#e94560,color:#fff
    style STEP_OSC fill:#0d1b2a,stroke:#415a77,color:#e0e1dd
    style STEP_RENDER fill:#1a1a2e,stroke:#778da9,color:#e0e1dd
    style SAVE_S fill:#1b263b,stroke:#415a77,color:#e0e1dd
    style HEALTH fill:#1b263b,stroke:#415a77,color:#e0e1dd
    style RESET fill:#1b263b,stroke:#415a77,color:#e0e1dd
    style S1 fill:#533483,stroke:#e94560,color:#fff
    style S2 fill:#533483,stroke:#e94560,color:#fff
    style S3 fill:#533483,stroke:#e94560,color:#fff
```

---

## Diagram 16 — End-to-End Data Transform

A sequential walkthrough showing the exact data shape at each stage of the pipeline, from a 2048x1536 H.264 frame to the final OSC messages consumed by the behavior system. Each handoff is annotated with the concrete values and formats involved — pixel coordinates, scale factors, ray equations, fusion thresholds, and EMA parameters. This diagram serves as a debugging reference: if tracking behaves unexpectedly, you can trace the data transforms step by step to isolate where the problem occurs.

```mermaid
sequenceDiagram
    participant RTSP as RTSP Stream
    participant RC as RobustCamera
    participant YOLO as YOLO 11n
    participant CAL as Calibration
    participant FUS as Fusion
    participant TRK as Tracking
    participant OSC as OSC Output
    participant LC as lightController

    Note over RTSP: 2048 x 1536 H.264 frame

    RTSP->>RC: raw frame (BGR numpy array)
    Note over RC: Thread buffer, grab() flush x3

    RC->>YOLO: resized frame (416 x 312 px)
    Note over YOLO: model.predict(conf=0.40, classes=[0])

    YOLO->>YOLO: bbox in 416px space (x1, y1, x2, y2, conf)
    YOLO->>CAL: bbox in 2048px space (scaled by 4.93x)

    Note over CAL: foot = ((x1+x2)/2, y2)

    rect rgb(83, 52, 131)
        Note over CAL: Ray-Plane Intersection
        CAL->>CAL: undistort (ux, uy)
        CAL->>CAL: ray_cam = K_inv @ [ux, uy, 1]
        CAL->>CAL: ray_world = R_T @ ray_cam
        CAL->>CAL: t = (-66 - cam_y) / ray_y
        CAL->>CAL: hit = cam_pos + t * ray_world
    end

    CAL->>FUS: world detection (x_cm, z_cm, camera, conf)

    Note over FUS: Merge detections from<br/>different cameras within 150cm

    rect rgb(15, 52, 96)
        Note over FUS: Cross-Camera Fusion
        FUS->>FUS: greedy nearest-neighbor
        FUS->>FUS: average merged positions
    end

    FUS->>TRK: fused positions [(x, z), ...]

    rect rgb(22, 33, 62)
        Note over TRK: Temporal Tracking
        TRK->>TRK: predict: pos + velocity
        TRK->>TRK: match within 90cm (60% of 150)
        TRK->>TRK: EMA smooth (alpha=0.03)
        TRK->>TRK: update velocity
        TRK->>TRK: prune lost > 60 frames
    end

    TRK->>OSC: tracked [(id, x_cm, z_cm), ...]

    OSC->>LC: /tracker/count 3
    OSC->>LC: /tracker/person/1 -120.5 195.3
    OSC->>LC: /tracker/person/2 -45.8 310.7
    OSC->>LC: /tracker/person/3 -250.1 155.2

    Note over LC: Zone classification<br/>Behavior system<br/>Light output
```

---

## Cross-References

| Resource | Description |
|---|---|
| [BEHAVIOR_SYSTEM.md](BEHAVIOR_SYSTEM.md) | Full prose reference for all behavior systems |
| [light_behavior.py](light_behavior.py) | State machine, gestures, trend analysis, feedback learning |
| [lightController_osc.py](lightController_osc.py) | Main loop, OSC, Art-Net, WebSocket, AutoTuning |
| [public-viewer/](public-viewer/) | Three.js web viewer (connects via WebSocket) |
| [V2_5Dev/camera_tracker_osc.py](V2_5Dev/camera_tracker_osc.py) | Camera tracker V2.5 (RTSP, YOLO, calibration, fusion, OSC) |
| [camera_calibration.py](camera_calibration.py) | Camera calibration tool (ArUco, solvePnP, synth view) |
| [camera_calibration.json](../calibration/camera_calibration.json) | Calibration data (rvec, tvec, K, distortion per camera) |
| [world_coordinates.json](world_coordinates.json) | ArUco marker 3D positions and camera intrinsics |

---

*These diagrams describe the behavior and camera tracking systems as of the current codebase. The auto-tuning, feedback learning, and daily learning mechanisms mean the light's personality evolves over time — these diagrams capture the architecture, but the actual parameter values will drift as the system adapts to its environment.*
