# Drop Ceiling — Behavior System Reference

A comprehensive guide to how the light thinks, moves, and learns.

---

## Table of Contents

1. [Overview](#overview)
2. [Behavior Modes](#behavior-modes)
3. [Mode Transitions & Stickiness](#mode-transitions--stickiness)
4. [Dwell Phases](#dwell-phases)
5. [Gesture Library](#gesture-library)
6. [Engaged Interaction System](#engaged-interaction-system)
7. [Proximity Response](#proximity-response)
8. [Idle Trend Analysis](#idle-trend-analysis)
9. [Aggression System](#aggression-system)
10. [Flow Tracking](#flow-tracking)
11. [Almost-Engaged Detection](#almost-engaged-detection)
12. [Personality (MetaParameters)](#personality-metaparameters)
13. [Presets](#presets)
14. [Time-of-Day Modifiers](#time-of-day-modifiers)
15. [AutoTuning](#autotuning)
16. [Feedback Learning](#feedback-learning)
17. [Public Viewer & WebSocket](#public-viewer--websocket)

---

## Overview

The Drop Ceiling installation is a single simulated point light that moves above a grid of LED panels. A camera tracks pedestrians passing below, and the light responds in real-time — wandering when nobody is around, following and acknowledging people who stop, and adapting its personality over days and weeks.

The system is split across two main files:

| File | Role |
|---|---|
| `light_behavior.py` | State machine, gestures, trend analysis, feedback learning |
| `lightController_osc.py` | Main loop (pygame/OpenGL), OSC, Art-Net, WebSocket, AutoTuning |

At its core the light maintains a **BehaviorMode** (what it's doing), a set of **MetaParameters** (its personality), and a rolling history of **trend data** (what's been happening around it). Every frame the behavior system outputs target parameters — speed, brightness, falloff radius, pulse rate, follow smoothing — which the controller interpolates toward.

---

## Behavior Modes

The light operates in one of four modes at all times:

### IDLE

No one is in the active zone. The light wanders gently inside a configurable box close to the panels.

| Parameter | Value |
|---|---|
| Move speed | 20 cm/s |
| Wander interval | 5.0 s |
| Brightness | 3 – 15 |
| Pulse speed | 4000 ms |
| Falloff radius | 80 cm |
| Follow smoothing | 0 (not following) |

Idle is not static — the light reads multi-timescale trend data, adjusts its wander box toward expected pedestrian flow, and may trigger attention-seeking gestures if it has been alone for a while (see [Aggression](#aggression-system)).

### ENGAGED

One or two people are in the active zone. The light follows the nearest person with gentle smoothing, brightens, and tightens its falloff.

| Parameter | Value |
|---|---|
| Move speed | 25 cm/s |
| Wander interval | 4.0 s |
| Brightness | 8 – 30 |
| Pulse speed | 2500 ms |
| Falloff radius | 50 cm |
| Follow smoothing | 0.03 |

This mode activates the [Engaged Interaction System](#engaged-interaction-system), which layers breathing brightness waves and periodic positional gestures on top of the follow tracking.

### CROWD

Three or more people are in the active zone. The light becomes energetic — faster movement, higher brightness, quicker pulse.

| Parameter | Value |
|---|---|
| Move speed | 60 cm/s |
| Wander interval | 0 s |
| Brightness | 12 – 45 |
| Pulse speed | 1500 ms |
| Falloff radius | 40 cm |
| Follow smoothing | 0.03 |

Crowd mode follows the **centroid** of all tracked people and may trigger **bloom** moments (full-panel illumination).

### FLOW

Heavy pedestrian traffic in the passive zone with no one stopping in the active zone. The light drifts with the directional flow of the crowd.

| Parameter | Value |
|---|---|
| Move speed | 25 cm/s |
| Wander interval | 3.0 s |
| Brightness | 5 – 20 |
| Pulse speed | 3000 ms |
| Falloff radius | 70 cm |
| Follow smoothing | 0 |

Flow mode is entered after 15 seconds of sustained passive traffic (≥3 people/min) with no active engagement. The light shifts its wander box in the direction people are walking, aiming to position itself where new arrivals are coming from.

---

## Mode Transitions & Stickiness

Mode changes are not instantaneous. Two mechanisms prevent erratic switching:

### Stickiness (persistence required)

Before the system commits to a new mode, the conditions for that mode must persist for a minimum duration:

| From → To | Required Persistence |
|---|---|
| IDLE → ENGAGED | **0 s** (immediate — someone entered the active zone) |
| IDLE → FLOW | 15 s of passive traffic |
| ENGAGED → IDLE | 5 s after last person leaves |
| ENGAGED → CROWD | 3 s with 2+ people |
| CROWD → ENGAGED | 5 s after crowd thins |
| CROWD → IDLE | 5 s after everyone leaves |
| FLOW → IDLE | 10 s of low traffic |
| FLOW → ENGAGED | **0 s** (immediate) |

### Minimum Mode Duration

The light stays in any mode for at least **8 seconds** before it will consider switching, preventing rapid flip-flopping.

### Transition Interpolation

Once a switch is committed, the light's parameters are interpolated over a transition duration:

| Transition | Duration | Character |
|---|---|---|
| IDLE → ENGAGED | 0.8 s | Quick engage |
| ENGAGED → IDLE | 3.0 s | Slow fade — reluctant goodbye |
| ENGAGED → CROWD | 0.5 s | Quick escalation |
| CROWD → ENGAGED | 2.0 s | Gradual de-escalation |
| CROWD → IDLE | 4.0 s | Slow fade when everyone leaves |
| IDLE → FLOW | 2.0 s | Gradual flow transition |
| FLOW → IDLE | 3.0 s | Slow exit from flow |
| FLOW → ENGAGED | 0.8 s | Quick engage from flow |

The asymmetry is intentional: engaging is fast and responsive, disengaging is slow and graceful.

---

## Dwell Phases

When a person stays in the active zone, their dwell time progresses through four phases. Each phase deepens the connection:

| Phase | Time Range | Behavior |
|---|---|---|
| **Notice** | 0 – 3 s | Light turns toward the person. Entry pulse fires. |
| **Greet** | 3 – 10 s | Light settles, brightness increases. Subtle gestures begin (nod, lean). |
| **Engage** | 10 – 30 s | Deeper connection. Sway, orbit, deeper breathing overlay. |
| **Bond** | 30 s + | Maximum intimacy. Very settled, infrequent but warm gestures. |

Dwell bonuses accumulate: longer dwell → brighter light, tighter tracking, broader gesture repertoire. The `dwell_influence` meta parameter scales how much these bonuses apply (0 = none, 2 = double).

---

## Gesture Library

Gestures are brief, one-shot behavioral events overlaid on the current mode. There are 16 gesture types organized into two categories:

### Event Gestures (triggered by state changes)

| Gesture | Description | Trigger |
|---|---|---|
| **ACKNOWLEDGE** | Brief move toward a passerby | Passive zone detection |
| **CURIOUS** | Slow approach toward a person | Sustained passive presence |
| **WELCOME** | Entrance flash for a new person | Active zone entry |
| **BORED** | Attention-seeking movement | Extended idle time (aggression-driven) |
| **FAREWELL** | Reluctant goodbye | Person leaves after >5 s dwell, no one remains |
| **SURPRISED** | Quick pulse when someone appears suddenly | Sudden active zone entry |
| **THINKING** | Slow drift pause, as if contemplating | Random during idle |
| **HESITANT** | Partial approach then retreat | Near-engagement situations |
| **PLAYFUL** | Quick zig-zag movement | High energy states |
| **BLOOM** | Expand radius to illuminate all panels | Periodic during engagement (≤15% chance/min, 45 s cooldown) |

Minimum cooldown between event gestures: **5 seconds**.

### Engaged Interaction Gestures (ongoing during engagement)

These are subtle, periodic gestures that fire throughout the ENGAGED and CROWD modes, giving the light a sense of continued presence rather than just passively tracking.

| Gesture | Axes | Amplitude | Description |
|---|---|---|---|
| **NOD** | Y | 12 cm | Small vertical bob — gentle acknowledgment |
| **LEAN** | Z | 15 cm | Brief forward shift — leaning in toward person |
| **SWAY** | X | 18 cm | Gentle lateral oscillation — relaxed presence |
| **ORBIT** | X + Y | 15 / 8 cm | Slow lazy circle around person's position |
| **SETTLE** | Z + radius | −8 cm / −10 cm | Tighten in closer — getting comfortable |
| **BREATHE** | brightness + radius | ±12% / ±6% | Visible brightness wave — shared rhythm |

All engaged gestures use sine-curve animation for smooth organic motion.

---

## Engaged Interaction System

The engaged interaction system runs whenever the light is in ENGAGED or CROWD mode. It consists of two overlapping layers:

### Layer 1: Breathing Overlay

A continuous sinusoidal modulation of brightness and falloff radius that creates a "breathing" effect — as if the light and the person are sharing a rhythm.

| Setting | Value |
|---|---|
| Ramp-up time | 8 seconds (depth 0 → 1 gradually) |
| Cycle period | 6 seconds per breath |
| Brightness depth | ±12% at full depth |
| Radius depth | ±6% at full depth |

The breathing overlay ramps in gently after the greet phase begins and ramps out when the person leaves. It is applied multiplicatively on top of all other brightness calculations.

### Layer 2: Positional Gestures

Periodic small movements that break up the follow tracking with moments of personality. Which gestures are available and how often they occur depends on the current [dwell phase](#dwell-phases):

**Greet phase** (3–10 s): every 8–15 s
| Gesture | Duration | Weight |
|---|---|---|
| NOD | 1.2 s | 3 (most common) |
| LEAN | 1.5 s | 2 |
| BREATHE | 4.0 s | 2 |

**Engage phase** (10–30 s): every 10–20 s
| Gesture | Duration | Weight |
|---|---|---|
| NOD | 1.0 s | 2 |
| LEAN | 1.8 s | 2 |
| SWAY | 3.0 s | 3 |
| ORBIT | 4.0 s | 2 |
| BREATHE | 5.0 s | 3 |

**Bond phase** (30 s+): every 15–30 s
| Gesture | Duration | Weight |
|---|---|---|
| SWAY | 4.0 s | 3 |
| ORBIT | 5.0 s | 2 |
| BREATHE | 6.0 s | 3 |
| SETTLE | 3.0 s | 2 |
| NOD | 1.0 s | 1 |

Notice phase (0–3 s) has no positional gestures — the entry pulse covers acknowledgment.

### Farewell

When a person leaves the active zone after more than 5 seconds of dwell and no other people remain, the light triggers a **FAREWELL** gesture — a brief reluctant movement toward their last position before returning to idle.

---

## Proximity Response

The active zone spans Z = 78 (close to panels) to Z = 283 (far edge). As a person moves closer to the panels, the light adjusts its behavior:

| Z Position | Proximity Factor | Speed | Brightness | Smoothing |
|---|---|---|---|---|
| ≤ 100 cm (near) | 1.0 | ×0.6 (slower, deliberate) | ×1.4 (brighter) | ×0.7 (more precise) |
| ≥ 280 cm (far) | 0.0 | ×1.4 (faster) | ×0.8 (dimmer) | ×1.3 (looser) |
| Between | linear | interpolated | interpolated | interpolated |

This creates an effect where the light becomes more intimate and careful as someone steps close, and more animated and loose when they're farther away.

---

## Idle Trend Analysis

When no one is in the active zone, the light doesn't simply wander randomly. It queries the tracking database across four timescales and adjusts its behavior accordingly:

| Timescale | Window | Use |
|---|---|---|
| **Recent** | 1 minute | Immediate reactivity — is anyone nearby right now? |
| **Short** | 5 minutes | Should we be ready for action? |
| **Medium** | 30 minutes | General activity level of this half-hour |
| **Long** | 1 hour | Big-picture energy level |

Additionally the system loads **historical pattern data** — the typical activity level for this time period based on previous days.

From these timescales, three computed influence values are derived:

- **Activity anticipation** (0–1): Should the light be poised and ready, or relaxed?
- **Flow momentum** (−1 to +1): Sustained directional flow (shifts wander box X)
- **Energy level** (0–1): Overall energy to match

These influence idle wander speed, brightness baseline, wander box position, and whether the light drifts toward the side where traffic is coming from.

Trend queries run in a **background thread** to avoid blocking the main render loop.

---

## Aggression System

Aggression is a 0–1 EMA-smoothed "attention-seeking" level that rises when the light is ignored and falls when people engage.

### What increases aggression

- Time passing without any active zone engagement
- High passive traffic with low conversion (people walking by but not stopping)

### What decreases aggression

- Someone entering the active zone
- Recent engagement success

### Time-of-day caps

Aggression is capped based on the hour to match the character of the location (financial district):

| Time Block | Max Aggression | Rationale |
|---|---|---|
| 0:00 – 5:00 | 0.2 | Late night — area is dead, no point being aggressive |
| 6:00 – 7:00 | 0.3 | Early morning — commuters won't stop |
| 8:00 – 9:00 | 0.4 – 0.5 | Morning rush easing |
| 10:00 – 11:00 | 0.7 – 0.8 | Late morning — people might explore |
| 12:00 – 14:00 | 0.7 – 0.8 | Lunch — highest caps |
| 15:00 – 16:00 | 0.5 – 0.6 | Afternoon |
| 17:00 – 18:00 | 0.4 | Evening rush — low |
| 19:00 – 20:00 | 0.4 – 0.5 | Evening |
| 21:00 – 23:00 | 0.2 – 0.3 | Night |

Higher aggression manifests as:
- Wider and faster wander movements
- More frequent BORED gestures
- Brighter pulses toward passive zone traffic
- Stronger attraction attempts on almost-engaged candidates

---

## Flow Tracking

A real-time system that detects the dominant direction of pedestrian traffic.

- Updates every **1.5 seconds** (faster than trend analysis)
- Uses a **30-second sliding window**
- Tracks left-to-right vs. right-to-left movement counts
- EMA smoothed with α = 0.25 for responsiveness

The flow direction (−1 to +1) and strength (0 to 1) are used to:

1. **Shift the wander box** in IDLE/FLOW modes toward where people are coming from
2. **Trigger FLOW mode** when passive traffic is heavy and sustained
3. **Inform feedback learning** — was the light aligned with or opposed to flow when engagement happened?

---

## Almost-Engaged Detection

The system watches for people in the passive zone who are slowing down near the active zone boundary — potential converts.

### Detection criteria

| Criterion | Threshold |
|---|---|
| Speed | Below 50 cm/s (slowing down) |
| Distance to active zone | Within 100 cm |
| Duration | Slow for ≥ 1 second |

### Attraction strategies (A/B tested)

When a candidate is detected, the light tries one of these strategies and logs the outcome:

| Strategy | Description |
|---|---|
| **Brightness Pulse** | Subtle brightness increase toward the person |
| **Drift Toward** | Move the light gently toward them |
| **Pause and Look** | Stop wandering and focus on them |

Strategies are rotated for A/B testing. Conversion statistics are tracked per strategy:

```
strategy_stats = {
    'brightness_pulse': {'attempts': 0, 'conversions': 0},
    'drift_toward':     {'attempts': 0, 'conversions': 0},
    'pause_and_look':   {'attempts': 0, 'conversions': 0},
}
```

Cooldown between attraction attempts: **5 seconds**.

---

## Personality (MetaParameters)

Six personality sliders (0.0 – 1.0) shape how the light behaves. These are the expressive character of the installation:

| Parameter | Low (0.0) | High (1.0) |
|---|---|---|
| **responsiveness** | Slow, contemplative | Quick, reactive |
| **energy** | Calm, gentle | Lively, dynamic |
| **attention_span** | Easily distracted | Focused, loyal |
| **sociability** | Reserved | Eager to engage |
| **exploration** | Stays put | Wanders widely |
| **memory** | Forgets quickly | Avoids repetition |

### Global Multipliers

In addition to the personality sliders, there are global multipliers that scale specific output parameters:

| Multiplier | Default | Effect |
|---|---|---|
| `brightness_global` | 1.0 | Scales all brightness values |
| `speed_global` | 1.0 | Scales all movement speeds |
| `pulse_global` | 1.0 | Scales pulse speed |
| `follow_speed_global` | 1.0 | Scales follow tracking speed |
| `dwell_influence` | 1.0 | How much dwell time affects behavior (0 = none, 2 = double) |
| `idle_trend_weight` | 1.0 | How much passive trends affect IDLE behavior |
| `trend_weight` | 1.0 | General trend influence |
| `time_of_day_weight` | 1.0 | Time-of-day modifier strength |
| `anti_repetition_weight` | 1.0 | Anti-repetition influence |

### Feature Toggles

| Toggle | Default |
|---|---|
| `gestures_enabled` | True |
| `follow_enabled` | True |
| `flow_mode_enabled` | True |
| `dwell_rewards_enabled` | True |
| `entrance_flash_enabled` | True |
| `self_analysis_enabled` | True |
| `status_text_enabled` | True |

---

## Presets

Six built-in personality presets provide starting points:

| Preset | Resp. | Energy | Attn. | Social | Explore | Memory | Character |
|---|---|---|---|---|---|---|---|
| **default** | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | Balanced neutral |
| **shy** | 0.3 | 0.3 | 0.7 | 0.2 | 0.3 | 0.6 | Reserved, attentive, remembers |
| **eager** | 0.8 | 0.7 | 0.4 | 0.9 | 0.6 | 0.4 | Quick, social, easily distracted |
| **zen** | 0.2 | 0.2 | 0.9 | 0.4 | 0.4 | 0.8 | Calm, deeply focused, high memory |
| **playful** | 0.7 | 0.8 | 0.3 | 0.7 | 0.9 | 0.3 | Energetic, exploratory, short memory |
| **night_owl** | 0.4 | 0.3 | 0.6 | 0.5 | 0.2 | 0.7 | Slow, moderate, remembers patterns |

---

## Time-of-Day Modifiers

The time of day directly modifies output parameters on top of mode and personality:

| Period | Hours | Brightness | Pulse | Y Range | Mood |
|---|---|---|---|---|---|
| Late night | 0:00 – 6:00 | ×0.4 | ×1.5 | 0 – 60 | "sleepy" |
| Waking | 6:00 – 9:00 | ×0.7 | ×1.2 | 0 – 100 | "waking" |
| Active | 9:00 – 17:00 | ×1.0 | ×1.0 | 0 – 150 | "active" |
| Rush | 17:00 – 20:00 | ×1.1 | ×0.9 | 0 – 150 | "rush" |
| Evening | 20:00 – 0:00 | ×0.6 | ×1.3 | 0 – 80 | "evening" |

---

## AutoTuning

The `AutoTuningManager` in `lightController_osc.py` continuously adjusts the personality parameters over time, creating a light that learns and adapts.

### Tuned Parameters

The tuner manages **12 parameters** — 6 personality sliders and 6 global multipliers:

| Parameter | Min | Max | Safe Floor | Soft Cap | Home Value |
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

**Safe floors** prevent parameters from dropping low enough to create a "zombie light" that doesn't respond. **Soft caps** prevent individual display parameters from becoming obnoxious.

### Update Loop

The tuner runs every **5 seconds**. Each cycle:

1. **Compute activity signal**: ratio of active people to (active + passive + 1)
2. **Compute delta per parameter** based on the activity signal:
   - Personality params (responsiveness, sociability, energy) **increase** with activity
   - Display params (brightness, speed, pulse) **inversely** adjust — they decrease slightly when already active (the personality shift handles the increased engagement)
   - Exploration **increases** when it's quiet (more searching behavior)
3. **Apply mean reversion**: a gentle pull (strength = 0.008) toward home values, always active, preventing parameters from drifting to extremes
4. **Apply curiosity perturbation**: every 60 seconds, a small random nudge (strength = 0.015) to a random parameter — exploration of the parameter space
5. **Apply damping**: when aggression > 0.6 or time since last engagement < 10 s, deltas are reduced to prevent overcorrection during volatile moments
6. **Budget check**: changes are rate-limited by a budget system (cost_scale = 40, restores at budget_max/180 per second) to prevent rapid large swings

### Step Sizes

| Scope | Max Step |
|---|---|
| Personality params | 0.03 per update |
| Global multipliers | 0.08 per update |
| Minimum step | 0.002 (below this, change is zeroed to prevent drift) |

### Daily Learning

At the end of each day, the system:

1. **Queries the database** for a 7-day weighted average of engagement metrics by time period
2. **Computes learned adjustments** per time period (morning, afternoon, evening, late_night)
3. **Applies learnings** at a 30% blend rate — the next day starts with the previous day's learned values gently mixed in

This means the light gradually shifts its personality to match what worked at different times of day over the previous week.

---

## Feedback Learning

The `FeedbackLearning` system tracks what the light was doing when people engaged, and gradually weights successful contexts higher.

### How it works

When someone enters the active zone, the system captures a snapshot:

- **Mode** the light was in (idle, flow)
- **Aggression level** at the moment
- **Flow alignment** (was the light positioned toward incoming traffic?)
- **Light position** (left, center, right)
- **Time of day** (morning, afternoon, evening, late_night)
- **Movement and brightness** values

These snapshots are stored in a ring buffer of the last 50 engagement contexts.

### Learned Weights

Each context dimension has a weight (starting at 1.0, range 0.5 – 2.0):

| Category | Weights |
|---|---|
| **Aggression level** | low (< 0.3), mid (0.3 – 0.6), high (> 0.6) |
| **Position** | left, center, right |
| **Flow alignment** | aligned, neutral, opposed |
| **Time of day** | morning, afternoon, evening, late_night |
| **Mode** | from_idle, from_flow |

When engagement occurs, the matching weights are nudged upward. Over time, the light learns patterns like: "center position + mid aggression + afternoon = best engagement."

### Learning Rate

- **Per engagement**: ±0.02 weight change
- **Weight range**: 0.5 – 2.0
- **Ring buffer**: 50 recent contexts

The learning rate is deliberately slow to prevent overcorrection from noisy data.

---

## Public Viewer & WebSocket

The public viewer is a Three.js web application that connects via WebSocket and renders the light, panels, and tracked people in real-time 3D.

### WebSocket Payload (relevant behavior fields)

| Field | Type | Description |
|---|---|---|
| `behavior_description` | string | Human-readable "Mode · Action" text (e.g., "Engaged · Breathing Together") |
| `status` | string | Detailed status text from the behavior system |
| `mode` | string | Current mode name (idle / engaged / crowd / flow) |
| `gesture` | string | Current gesture name or "none" |
| `dwell_phase` | string | Current dwell phase (notice / greet / engage / bond / none) |
| `engaged_breathing` | object | `{ active, depth, phase, gesture_count }` |

### Behavior Description Examples

The viewer displays a crossfading subheading that summarizes the light's current activity:

| State | Description |
|---|---|
| Idle, no activity | "Idle · Wandering" |
| Idle, bored gesture | "Idle · Seeking Attention" |
| Engaged, 1 person | "Engaged · Following" |
| Engaged, breathing active | "Engaged · Breathing Together" |
| Engaged, nod gesture | "Engaged · Nodding" |
| Engaged, sway gesture | "Engaged · Swaying" |
| Engaged, orbit gesture | "Engaged · Orbiting" |
| Engaged, lean gesture | "Engaged · Leaning In" |
| Crowd mode | "Crowd · High Energy" |
| Flow mode | "Flow · Drifting with Traffic" |
| Farewell gesture | "Engaged · Saying Goodbye" |

---

*This document describes the behavior system as of the current codebase. The system is designed to evolve — the AutoTuner, feedback learning, and daily learning mechanisms mean that the light's personality will shift gradually over weeks of operation, adapting to the patterns of the site.*
