# V6.5: Passive-Flow-Driven Behavior

**Commit:** `7f3b16d` — March 3, 2026  
**Previous:** V6.1f (`1a7a12b`) — resolver caps, rate-limiting, proximity dampening

---

## Overview

V6.5 reorients the entire behavior system around **passive sidewalk traffic** as the primary driver. Data from 33 daily reports (Jan 29 – Mar 2) showed a 57–68:1 passive-to-active ratio, yet the system was barely using passive data — it only applied minor brightness/wander tweaks during IDLE mode. The system spent 46–71% of its time in IDLE and 24–43% in FLOW, with passive counts passed to modules but never actually read.

V6.5 makes three core changes:
1. **Three passive tiers** replace the binary idle/flow split
2. **Flow-reactive ambient breathing** with Y-axis oscillation
3. **Passive data wired through** all V6 subsystems

---

## Design Decisions

| Question | Choice |
|----------|--------|
| IDLE vs FLOW distinction | Three tiers: quiet / flow / busy |
| Movement model | Hybrid — position follows flow, speed/energy from density |
| Breathing behavior | Flow-reactive tempo (faster when busy) |
| Scope | ~6 existing files, no new modules |

---

## Files Changed (8 files, +472 / -105)

### 1. `IO/light_behavior.py` — Core Behavior System

**BehaviorMode enum — new AWARE tier**
- Added `AWARE = "aware"` to the `BehaviorMode` enum (line ~40)
- Three passive tiers now exist: IDLE (quiet), FLOW (normal), AWARE (busy)

**Passive tier thresholds**
- `PASSIVE_TIER_FLOW = 2` — passive_rate >= 2 people/min triggers FLOW
- `PASSIVE_TIER_AWARE = 10` — passive_rate >= 10 people/min triggers AWARE

**`determine_mode()` — three-tier passive selection**
- Rewritten to check passive_rate against tier thresholds
- IDLE: `passive_rate < 2/min` or no detections (true quiet)
- FLOW: `passive_rate 2–10/min` — primary daytime state
- AWARE: `passive_rate >= 10/min` — rush hour, lunch crowds
- Active zone still takes priority: 1 person → ENGAGED, 2+ → CROWD

**MODE_PARAMS — AWARE base parameters**
```python
BehaviorMode.AWARE: {
    'move_speed': 35,
    'brightness_max': 30,
    'pulse_speed': 2200,
    'falloff_radius': 65,
}
```

**TRANSITIONS — 12 new entries**
- All transitions to/from AWARE mode added
- AWARE↔FLOW, AWARE↔IDLE, AWARE↔ENGAGED, AWARE↔CROWD
- Durations: IDLE→AWARE = 4.0s, FLOW→AWARE = 2.5s, AWARE→FLOW = 3.0s

**MODE_STICKINESS — expanded**
- ~10 new entries for AWARE transitions
- IDLE→FLOW reduced from 15s to 5s (FLOW is now the common state, not a rare upgrade)
- FLOW→AWARE = 10s, AWARE→FLOW = 8s, AWARE→IDLE = 12s
- Prevents rapid tier oscillation at boundary rates

**`apply_idle_trends()` — completely rewritten**

Was: Only ran in IDLE mode, applied minor brightness/wander tweaks.  
Now: Runs in IDLE, FLOW, and AWARE modes with tier-scaled influence.

| Parameter | IDLE (quiet) | FLOW (normal) | AWARE (busy) |
|-----------|-------------|---------------|--------------|
| Tier multiplier | 1.0× | 1.5× | 2.0× |
| Flow shift range | ±50cm | ±120cm | ±180cm |
| Energy influence | narrow | moderate | wide |
| Brightness boost | up to +15% | up to +25% | up to +40% |
| Exploration | — | flow consistency mapping | stronger mapping |

**`_apply_ambient_falloff()` — completely rewritten**

Was: X/Z oscillation only, fixed tempo, no mode awareness.  
Now: Full 3D breathing with flow-reactive behavior.

New features:
- **Y-axis oscillation** — gradient breathes vertically (was never driven before)
- **Flow-reactive tempo** — tempo scales with passive tier:
  - Quiet: 0.7× (slow, meditative breathing)
  - Flow: 1.0× (natural pace)
  - Aware: 1.6× (energised, faster breathing)
- **Flow-reactive depth** — oscillation amplitude scales:
  - Quiet: 0.5× (subtle shimmer)
  - Flow: 1.0× (standard)
  - Aware: 1.8× (dramatic shape shifts)
- **Directional phase offsets** — when pedestrians walk L→R, the breathing "wave" ripples in that direction, creating organic responsiveness
- **EMA-smoothed tempo** — tempo changes are smoothed (alpha=0.02) so there are no jarring shifts when the tier changes

New `BehaviorState` fields:
- `ambient_falloff_phase_y: float = 0.0`
- `ambient_tempo_factor: float = 1.0`

**`AMBIENT_FALLOFF_CONFIG` expanded:**
```python
# Y-axis periods and depths (new)
'y_period': 31.0,
'y_depth_idle': 0.03, 'y_depth_flow': 0.05,
'y_depth_aware': 0.08, 'y_depth_engaged': 0.04,

# Tempo scaling per tier (new)
'tempo_quiet': 0.7, 'tempo_flow': 1.0, 'tempo_aware': 1.6,
'tempo_ema_alpha': 0.02,

# Depth multipliers per tier (new)
'depth_quiet_mult': 0.5, 'depth_flow_mult': 1.0, 'depth_aware_mult': 1.8,

# Flow directional offset (new)
'flow_phase_offset_strength': 0.4,
```

**`get_status()` — bug fix + new fields**

**Bug discovered and fixed:** `active_count`, `passive_count`, and `passive_rate` were nested inside `driving_factors` in the status dict, but every V6 module read them at the top level (`behavior_status.get('active_count', 0)`). They were **always returning 0**.

Fix: Promoted to top-level keys:
```python
return {
    'mode': ...,
    # V6.5: Promote counts to top-level so V6 modules read them correctly
    'active_count': self.state.last_active_count,
    'passive_count': self.state.last_passive_count,
    'passive_rate': self.state.last_passive_rate,
    ...
}
```

This means all V6 modules (autotuner regime heuristics, feedback learning group_size, falloff density shape, health check group buckets) now receive **real data** instead of always-zero values.

---

### 2. `IO/V6Dev/falloff_strategies.py` — Falloff Shape Engine

**MODE_FALLOFF_DEFAULTS — AWARE entry**
```python
'aware': {
    'scale_x': 1.5, 'scale_y': 1.1, 'scale_z': 1.5,
    'radius_mult': 1.15,
}
```
Wider and taller than FLOW — the gradient reaches further during busy periods.

**`FalloffStrategyManager.__init__()` — settle animation fields**
- `_settle_active: bool = False`
- `_settle_start: float = 0`
- `_settle_duration: float = 3.0`
- `_last_mode: str = 'idle'`

**`compute_shape()` — expanded to 6 layers**

Was: 4 layers (mode defaults, proximity, flow shape, gesture).  
Now: 6 layers (+ density + settle).

The method now detects ENGAGED→passive transitions to trigger the settle animation:
```python
# Detect mode transition from engaged → passive for settle animation
if self._last_mode in ('engaged', 'crowd') and mode in ('idle', 'flow', 'aware'):
    self._settle_active = True
    self._settle_start = time.time()
self._last_mode = mode
```

**`_compute_flow_shape()` — AWARE support**
- AWARE mode gets 1.3× stronger rotation and 1.6× Z-stretch vs FLOW
- Creates a more dramatic directional shape during busy periods

**`_compute_density_shape()` — NEW METHOD**
- Only active in FLOW and AWARE modes
- Uses log-scaled `passive_count` to widen the gradient
- X/Z scale up to 1.3, Y up to 1.15, radius_mult up to 1.1
- More people on the sidewalk → wider, taller gradient reach

**`_compute_settle_shape()` — NEW METHOD**
- 3-second animation triggered when exiting ENGAGED/CROWD mode
- Quick expansion phase (first 30%) → slow contraction (remaining 70%)
- Creates a visible "exhale" when someone walks away
- Scale peak: 1.15× X/Z, 1.08× Y

**`suggest_gesture()` — AWARE support**
- AWARE mode with energy > 0.5 → SWEEP gesture
- Gives the light visible sweeping motion during busy periods

---

### 3. `IO/V6Dev/v6_integration.py` — V6 Central Wiring

**Feedback context — passive data**
- `FeedbackContext` now populated with `passive_rate` and `passive_tier`
- Tier derived from current mode: aware → 'busy', flow → 'flow', else → 'quiet'

**`_log_status()` — passive tier in logs**
- Added `passive=X.X/min(tier)` to the periodic V6 status log
- Example: `V6 frame=1000, adj=5, mode=FLOW, regime=steady, passive=4.2/min(flow), budget=180`

**Status output — passive prediction**
- `v6_expected_passive` and `v6_expected_tier` added to extended status
- Available to the web viewer for display

---

### 4. `IO/V6Dev/feedback_learning_v6.py` — Feedback Learning

**FeedbackContext — new fields**
```python
# V6.5: Passive traffic context
passive_rate: float = 0.0       # people/min in passive zone
passive_tier: str = 'quiet'     # quiet / flow / busy
```

**`get_all_buckets()` — passive tier bucket**
- Added `f'passive_{ctx.passive_tier}'` to the bucket list
- Creates three new learning buckets: `passive_quiet`, `passive_flow`, `passive_busy`
- The system learns which parameter settings work best at each density level
- Over time, it can differentiate: "during busy sidewalk, higher brightness works better"

---

### 5. `IO/V6Dev/smart_autotuner.py` — Auto-Tuner

**Anti-passivity logic — FLOW mode support**

Was: Only pushed engagement params up when `mode == 'idle' and idle_duration > 60s`.  
Now: Also acts during FLOW mode with a longer threshold and gentler push.

```python
if current_mode == 'idle' and idle_duration > 60.0:
    push_strength = 1.0       # full push
elif current_mode == 'flow' and mode_duration > 180.0:
    push_strength = 0.5       # gentle push after 3 min in FLOW without engagement
```

FLOW is the new common daytime state — without this change, the system would never trigger anti-passivity during normal operation.

**`_get_home_values()` — passive personality biases**
- Applies energy and exploration biases from `get_passive_personality()`
- During expected-busy hours, home energy target shifts up (+0.12)
- During expected-quiet hours, home energy target shifts down (-0.06)
- Clamped to [0.2, 0.9] for energy and [0.2, 0.8] for exploration

---

### 6. `IO/V6Dev/predictive_context.py` — Predictive Context Engine

**HourlyPrediction — new fields**
```python
expected_passive_count: float = 0.0    # avg passive people this hour
expected_passive_tier: str = 'quiet'   # quiet / flow / busy
```

**`_build_prediction_cache()` — passive count computation**
- Collects weighted passive counts alongside existing active/people data
- Converts hourly count to rate (÷60) for tier classification
- Assigns `expected_passive_tier` using the same thresholds as `determine_mode()`

**`get_passive_personality()` — NEW METHOD**

Returns time-of-day behavioral biases based on historically learned passive traffic:

| Tier | breath_tempo_bias | energy_bias | exploration_bias |
|------|------------------|-------------|-----------------|
| Busy | +0.4 | +0.12 | +0.08 |
| Flow | 0.0 | 0.0 | 0.0 |
| Quiet | -0.2 | -0.06 | -0.03 |

All biases are scaled by prediction confidence (low data → biases approach 0).  
Fed into the autotuner's home values to shift the character's "resting mood" throughout the day.

---

### 7. `IO/autotune_overrides.json` — Runtime Configuration

- Added `v6_5_passive_tiers` section documenting tier thresholds
- Updated `last_review_changes` with V6.5 changelog
- Home values, floors, and caps unchanged (V6.1f values preserved)

---

## Data Flow Diagram

```
Camera Tracker (V4)
    │
    ├── active_count ──────────────────────────────┐
    ├── passive_count ─────────────────────────────┤
    └── passive_rate ──────────────────────────────┤
                                                    │
    BehaviorSystem.determine_mode()  ◄─────────────┘
        │
        ├── IDLE   (passive_rate < 2/min)
        ├── FLOW   (passive_rate 2–10/min)    ◄── primary daytime state
        ├── AWARE  (passive_rate >= 10/min)
        ├── ENGAGED (active_count >= 1)
        └── CROWD  (active_count >= 2)
                │
    ┌───────────┘
    │
    ▼
get_status()
    │
    ├── mode, active_count, passive_count, passive_rate  ◄── NEW: top-level
    ├── idle_trends (tier-scaled)
    └── flow info
        │
        ▼
    V6Integration.tick()
        │
        ├── ModeIntelligence ──── 'aware' already mapped to V6Mode.AWARE
        │
        ├── PredictiveContext ──── expected_passive_count/tier
        │       └── get_passive_personality() → energy/exploration biases
        │
        ├── FeedbackLearning ──── passive_tier bucket (learns per density)
        │
        ├── SmartAutoTuner ──── anti-passivity in FLOW mode
        │       └── home values shifted by passive personality
        │
        └── FalloffStrategyManager
                ├── AWARE defaults (wider reach)
                ├── density_shape (passive_count → wider gradient)
                └── settle_shape (post-engagement exhale)
```

## Ambient Breathing — Before vs After

| Aspect | V6.1f (Before) | V6.5 (After) |
|--------|---------------|--------------|
| Axes | X, Z only | X, Y, Z |
| Tempo | Fixed | Flow-reactive (0.7×–1.6×) |
| Depth | Fixed per mode | Tier-scaled (0.5×–1.8×) |
| Direction | Independent axes | Flow-aligned wave motion |
| Transitions | Instant | EMA-smoothed (alpha=0.02) |

## Expected Behavior Changes

1. **Daytime (8am–6pm):** System should spend most time in FLOW mode instead of IDLE. Breathing is natural-paced, position tracks flow direction. When sidewalk gets busy (lunch, evening commute), AWARE mode activates with faster breathing and wider reach.

2. **Evening/Night:** IDLE mode returns as traffic drops. Slow, meditative breathing. Subtle movement.

3. **Engagement transitions:** When someone enters the active zone, the normal ENGAGED mode activates. When they leave, the new settle animation creates a visible "exhale" (3s expansion then contraction) before returning to the passive tier.

4. **Learning:** Feedback learning now differentiates by passive density. Over days/weeks, the system learns which brightness/speed/pulse settings attract engagement during quiet vs busy sidewalk conditions.

5. **Bug fix impact:** With `active_count`/`passive_count` now actually reaching V6 modules, the autotuner's regime heuristics, feedback learning's group size bucketing, and falloff density scaling will all start working with real data instead of always-zero values. This alone may produce noticeable behavioral improvement.
