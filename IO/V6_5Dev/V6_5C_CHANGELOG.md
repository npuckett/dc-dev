# V6.5c Changelog — March 4, 2026

Two commits implementing performance fixes from the 24-hour analysis and a movement system overhaul.

---

## Commit 1: Top-7 24h Analysis Recommendations (`4c0faee`)

Based on [V6_5B_24HR_ANALYSIS_2026-03-04.md](../V6_5B_24HR_ANALYSIS_2026-03-04.md).

### 1. Re-enabled systemd service
- `light-controller.service` copied to `/etc/systemd/system/`, daemon-reloaded, enabled
- Auto-starts on boot, auto-restarts on crash (Restart=always, RestartSec=5)

### 2. Widened passive tracking zone
- **File:** `lightController_osc.py` — `PASSIVE_TRACKZONE`
- X width: 400 → 650 cm (restored to original)
- Depth: 270 → 350 cm (zone now ends at ~Z=633)
- **Why:** 48.5% of tracking events (562K/day) fell into "unknown" zone. 86% were outside the old narrow X range.

### 3. Capped crowd brightness
- **File:** `light_behavior.py` — end of `calculate_parameters()`
- Hard cap `brightness_max` at 600 after all modifiers (breathing, falloff, bloom, entry pulse)
- **Why:** Crowd mode avg brightness hit 551 with peaks potentially exceeding 600 due to stacked multipliers.

### 4. Tuned almost-engaged pipeline
- **File:** `light_behavior.py` — `AlmostEngagedState` + `apply_almost_engaged_attraction()`
- Mode restriction: IDLE-only → IDLE, FLOW, and AWARE
- Speed threshold: 50 → 120 cm/s (pedestrians walk ~130 cm/s, slowing ≈ 80–120)
- Distance threshold: 100 → 180 cm (wider catch radius)
- **Why:** 0 triggers in 24h. Root cause: people near active zone trigger FLOW/AWARE mode, so the IDLE-only check always skipped.

### 5. Fixed bloom count in hourly_stats
- **File:** `tracking_database.py` — `aggregate_hour()`
- Added SQL query counting `gesture_type='bloom'` from `light_behavior` table
- Added `bloom_count` to the INSERT column list
- **Why:** `bloom_count` was always 0 despite 248 bloom gestures occurring — the column existed in the schema but was never populated.

### 6. Finer bandit time periods
- **File:** `V6Dev/strategy_bandit.py` — `TimePeriod` enum + `BanditContext.time_period`
- 4 periods → 5 periods:
  - `EARLY_MORNING` (6–9), `LATE_MORNING` (9–12), `AFTERNOON` (12–17), `EVENING` (17–21), `NIGHT` (21–6)
- **Why:** Old 6-hour blocks were too coarse. Morning commute (6–9) has very different traffic than mid-morning (9–12).

### 7. Reduced idle exploration waste
- **File:** `V6Dev/strategy_bandit.py` + `V6Dev/v6_integration.py`
- Strategy cycle time in idle mode: 2x longer (30s minimum instead of 15s)
- Quality boost for idle arms: 1.4x (capped at 1.0) to speed bandit convergence
- `should_switch_strategy()` now accepts `mode` parameter
- **Why:** Idle contexts consumed 28% of bandit pulls (1,537/5,470) with only 60% win rate due to weak engagement signal.

---

## Commit 2: Data-Driven Movement (`1b7b274`)

Overhauls passive-mode movement (IDLE/FLOW/AWARE). ENGAGED/CROWD movement is unchanged.

### A. Park State
- **Files:** `light_behavior.py` (MODE_PARAMS), `lightController_osc.py` (WanderBehavior)
- Light periodically stops moving entirely; falloff oscillation (8.5–15s cycles) carries all visual animation
- Per-mode `rest_probability`:
  - IDLE: 40%, FLOW: 25%, AWARE: 10%, ENGAGED/CROWD: 0%
- Park duration: 4–8 seconds (random)
- Scaled by passive traffic rate:
  - `passive_rate > 5/min` → halves park chance (busier = more spatial movement)
  - `passive_rate < 1/min` → +25% park chance (quieter = more stillness)

### B. Pedestrian Speed Matching
- **File:** `light_behavior.py` — new `apply_speed_matching()` method
- `move_speed` scaled by average pedestrian walking speed (1-minute window from DB)
- Baseline: 130 cm/s (typical adult walking speed) → ratio 1.0
- Clamped to [0.3, 1.5] range, EMA-smoothed (α=0.08, ~12s settling)
- Per-mode `speed_match_factor`:
  - IDLE: 0.8 (slightly dampened coupling)
  - FLOW/AWARE: 1.0 (full coupling)
  - ENGAGED/CROWD: 0.0 (disabled — uses dwell-driven speed)
- New state field: `speed_match_ema` on `BehaviorState`

### C. Flow-Biased Target Selection
- **File:** `lightController_osc.py` — new `_biased_point()` method on `WanderBehavior`
- Replaces uniform random with triangular distribution for X-axis target picking
- Peak shifts up to 60% toward the incoming pedestrian side ("greeting" approaching traffic)
- Falls back to uniform random when `flow_strength < 0.15`
- Y (height) and Z (depth) remain uniform random
- Flow direction and strength exposed from `FlowState` through `calculate_parameters()` output

### New Parameters in MODE_PARAMS
| Key | IDLE | FLOW | AWARE | ENGAGED | CROWD |
|-----|------|------|-------|---------|-------|
| `rest_probability` | 0.4 | 0.25 | 0.1 | 0.0 | 0.0 |
| `speed_match_factor` | 0.8 | 1.0 | 1.0 | 0.0 | 0.0 |

### New Keys in behavior_params Output
| Key | Type | Description |
|-----|------|-------------|
| `flow_direction` | float (-1..+1) | Smoothed pedestrian flow direction |
| `flow_strength` | float (0..1) | Confidence in flow direction |
| `rest_probability` | float (0..1) | Current park chance (after traffic scaling) |

---

## Files Changed

| File | Commit 1 | Commit 2 |
|------|----------|----------|
| `lightController_osc.py` | Passive zone widened | Park state, biased targets, wander params |
| `light_behavior.py` | Brightness cap, almost-engaged tuning | MODE_PARAMS keys, speed matching, flow export |
| `tracking_database.py` | Bloom count fix | — |
| `V6Dev/strategy_bandit.py` | Time periods, idle exploration | — |
| `V6Dev/v6_integration.py` | Mode-aware strategy cycling | — |
| systemd service | Enabled | — |
