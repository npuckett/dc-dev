# Auto-Tuning Analysis — February 12, 2026

## Overview

The auto-tuning system ran for approximately 24 hours (first adjustment at `2026-02-11 15:40`, last at `2026-02-11 16:52`). This analysis examines the tracking database to diagnose why the system felt static and unresponsive, and documents the changes made to fix it.

---

## 1. Database Findings

### 1.1 Behavior Adjustments Summary

| Metric | Value |
|--------|-------|
| Total auto-tune adjustments | 462 |
| Time span | ~1.2 hours of active tuning |
| Avg energy level | 0.958 |
| Min energy level | 0.205 |
| Max energy level | 1.0 |
| Avg aggression level | 0.084 |
| Max aggression level | 0.339 |

### 1.2 Energy Level Distribution

| Energy Range | Count | % of Time |
|-------------|-------|-----------|
| 0.0–0.2 | 0 | 0% |
| 0.2–0.4 | 5 | 1.1% |
| 0.4–0.6 | 13 | 2.8% |
| 0.6–0.8 | 4 | 0.9% |
| **0.8–1.0** | **440** | **95.2%** |

Energy reached 1.0 within **2 minutes** of the first adjustment (15:40 → 15:42) and stayed there for the vast majority of the session.

### 1.3 Activity Level Distribution

**Short activity (5-min window):**

| Level | Count | % |
|-------|-------|---|
| Maxed (≥0.95) | 387 | 83.8% |
| Mid (0.5–0.95) | 46 | 10.0% |
| Low (<0.5) | 29 | 6.3% |

**Medium activity (30-min window):**

| Level | Count | % |
|-------|-------|---|
| Maxed (≥0.95) | 439 | 95.0% |
| Mid (0.5–0.95) | 2 | 0.4% |
| Low (<0.5) | 21 | 4.5% |

### 1.4 Cross-Reference: Short × Medium Activity

| Short Activity | Medium Activity | Count |
|---------------|----------------|-------|
| ≥0.95 | ≥0.95 | 382 |
| 0.5–0.95 | ≥0.95 | 45 |
| 0.2–0.5 | <0.2 | 12 |
| 0.2–0.5 | ≥0.95 | 12 |
| <0.2 | <0.2 | 5 |

**82.7% of the time, both short AND medium activity were simultaneously maxed.**

### 1.5 Light Behavior Mode Distribution

| Mode | Count | % | Avg Brightness | Avg Pulse Speed | Avg Move Speed |
|------|-------|---|---------------|----------------|----------------|
| idle | 183,864 | 54.0% | 24.12 | 4096.0 | 33.71 |
| flow | 156,564 | 46.0% | 37.17 | 3139.88 | 31.16 |
| engaged | 275 | 0.1% | 54.26 | 1589.87 | 8.47 |

The light spent 99.9% of its time in `idle` or `flow` mode, with almost no engagement events.

### 1.6 Parameter Values at Maximum Energy

When energy was maxed and short/medium activity were at 1.0, the parameters settled to:

| Parameter | Value | Notes |
|-----------|-------|-------|
| responsiveness | 0.000 | At floor (min_val=0.0) |
| energy | 0.000 | At floor |
| sociability | 0.000 | At floor |
| exploration | 0.000 | At floor |
| memory | 0.000 | At floor |
| idle_trend_weight | 0.000 | At floor |
| speed_global | 0.200 | At floor (min_val=0.2) |
| follow_speed_global | 0.500 | At floor |
| pulse_global | 0.301 | Near floor (0.3) |
| attention_span | 1.000 | At max |
| brightness_global | 3.000 | At cap |
| dwell_influence | 2.000 | At max |

**The light became a "bright zombie": maximum brightness and attention, but zero responsiveness, sociability, and energy.**

### 1.7 Tracking Events Overview

| Metric | Value |
|--------|-------|
| Total tracking events | 3,395,912 |
| Unique people tracked | 35,632 |
| Data span | Feb 3 – Feb 11 |
| Active zone events | 25 |
| Passive zone events | 3,345,370 |
| Unknown zone events | 50,517 |

### 1.8 Actual Traffic Patterns (Feb 11)

| Time Bucket (5 min) | Events | Unique People | Computed `short_activity` (old) |
|---------------------|--------|---------------|-------------------------------|
| 15:40 | 1,210 | 117 | 0.151 |
| 15:45 | 1,096 | 103 | 0.137 |
| 15:55 | 1,406 | 130 | 0.176 |
| 16:05 | 2,621 | 230 | 0.328 |
| 16:15 | 3,508 | 313 | 0.439 |
| 16:35 | 3,123 | 253 | 0.390 |
| 16:50 | 369 | 29 | 0.046 |

Individual 5-minute buckets show moderate activity (0.04–0.44), but the **cumulative** 5-minute sliding window query used by the system was hitting 8,000+ events and saturating to 1.0.

---

## 2. Root Cause Analysis

### 2.1 Activity Level Saturation

**The #1 problem.** The activity level computation used simple linear scaling:

```python
# In light_behavior.py
short_activity_level = min(1.0, total_short / 8000.0)    # 5-min window
medium_activity_level = min(1.0, total_medium / 50000.0)  # 30-min window
```

With 100–300 unique people per 5-minute window, each generating ~10 tracking events, the 5-minute event count routinely reached 8,000+. The divisor of 8,000 was calibrated too low for the actual traffic volume on this sidewalk.

**Result:** `short_activity` was pinned at 1.0 for 83.8% of the session. The auto-tuner saw maximum activity almost all the time, losing all ability to distinguish between "moderately busy" and "very busy."

### 2.2 Inverted Personality Response

The auto-tuner's delta logic treated all parameters the same way:

```python
# Old logic
combined_need = 0.5 * (target - short) + 0.3 * (target - medium) + 0.2 * (target - long)
# When short=1.0, medium=1.0, long=0.0:
# combined_need = 0.5*(-0.5) + 0.3*(-0.5) + 0.2*(0.5) = -0.30

deltas = {
    'responsiveness': combined_need * 0.08,  # → -0.024 (pushing DOWN)
    'sociability':    combined_need * 0.07,  # → -0.021 (pushing DOWN)
    'energy':         combined_need * 0.06,  # → -0.018 (pushing DOWN)
    ...
}
```

When activity was above the 0.5 target, `combined_need` went negative, and **all personality parameters were pushed down**. This is philosophically backwards:

- **Old behavior:** Busy sidewalk → decrease responsiveness → light ignores people
- **Desired behavior:** Busy sidewalk → increase responsiveness → light engages with the crowd

### 2.3 Dead Zone at Extremes

Once parameters hit their floor values (0.0 for personality, 0.2–0.5 for globals), the tuner kept trying to push them lower but couldn't. Meanwhile, `brightness_global` was driven UP by the long-term deficit (`long_activity=0.0`, below target), reaching its cap of 3.0. `dwell_influence` and `attention_span` also maxed out.

With most parameters clamped at their min/max, only `dwell_influence` had any room to move — the system was effectively frozen.

### 2.4 Budget Too Generous

```python
restore_rate = budget_max / 120.0  # = 60/120 = 0.5 per second
# At 5-second update intervals: 2.5 budget restored per cycle
# Typical adjustment cost: ~12.5
# Budget max: 60.0 (refills in 120 seconds)
```

The budget refilled in 2 minutes, faster than the tuner could meaningfully spend it. Every adjustment could use full step sizes, enabling rapid convergence to extremes.

---

## 3. Changes Made

### 3.1 Log-Scaled Activity Levels (`light_behavior.py`)

Replaced linear scaling with logarithmic scaling to prevent saturation:

```python
# OLD
short_activity_level = min(1.0, total_short / 8000.0)
medium_activity_level = min(1.0, total_medium / 50000.0)

# NEW
short_activity_level = min(1.0, log1p(total_short / 1500.0) / log1p(30.0))
medium_activity_level = min(1.0, log1p(total_medium / 8000.0) / log1p(40.0))
```

**Impact on actual traffic data:**

| Events (5-min) | Old Activity | New Activity |
|----------------|-------------|-------------|
| 369 | 0.046 | 0.064 |
| 1,210 | 0.151 | 0.172 |
| 2,621 | 0.328 | 0.294 |
| 3,508 | 0.439 | 0.351 |
| 8,000 | 1.000 | 0.538 |
| 20,000 | 1.000 | 0.775 |

The log curve compresses high values and expands low values, giving the tuner a usable signal across the full range of real-world traffic.

### 3.2 Redesigned Delta Logic (`lightController_osc.py`)

**Personality params now increase with activity** (match the crowd's energy):

```python
# Positive activity_excess (busy) → push responsiveness/sociability UP
deltas['responsiveness'] = activity_excess * 0.04 * damping
deltas['sociability'] = activity_excess * 0.04 * damping
deltas['energy'] = activity_excess * 0.03 * damping
```

**Display params inversely adjust** (don't overwhelm when busy, attract when quiet):

```python
# Positive activity_excess → push brightness/speed/pulse DOWN (moderate)
# Positive long_deficit (quiet long-term) → push them UP (attract)
deltas['brightness_global'] = -activity_excess * 0.04 + long_deficit * 0.06
deltas['speed_global'] = -activity_excess * 0.03 + long_deficit * 0.04
```

### 3.3 Mean Reversion (`lightController_osc.py`)

Added home values for all parameters that the system gently drifts back toward:

```python
home_values = {
    'responsiveness': 0.50,  'energy': 0.45,
    'attention_span': 0.50,  'sociability': 0.45,
    'exploration': 0.40,     'memory': 0.30,
    'brightness_global': 1.2, 'speed_global': 0.70,
    'pulse_global': 0.80,   'follow_speed_global': 1.0,
    'dwell_influence': 0.50, 'idle_trend_weight': 0.40,
}
# Applied every update cycle with strength 0.008
```

This prevents parameters from sticking at extremes. Even if the tuner pushes a param to its cap, mean reversion gently pulls it back. This creates natural oscillation around useful operating points.

### 3.4 Curiosity Perturbation (`lightController_osc.py`)

Every 60 seconds, all parameters receive a small random nudge (±0.015):

```python
if now - self._last_curiosity_time > self._curiosity_interval:
    for name in self.param_order:
        nudge = (random.random() - 0.5) * 2.0 * 0.015
        deltas[name] += nudge
```

This prevents static equilibrium and ensures the system is always exploring slightly different behavior combinations.

### 3.5 Raised Safe Floors (`lightController_osc.py`)

| Parameter | Old Floor | New Floor |
|-----------|----------|----------|
| responsiveness | 0.15 | 0.30 |
| energy | 0.10 | 0.25 |
| sociability | 0.05 | 0.20 |
| exploration | 0.05 | 0.15 |
| speed_global | 0.30 | 0.35 |
| pulse_global | 0.30 | 0.35 |
| follow_speed_global | 0.50 | 0.60 |
| attention_span | (none) | 0.10 |
| idle_trend_weight | (none) | 0.10 |

The light can no longer crash to zero on personality traits.

### 3.6 Slower Budget Restore (`lightController_osc.py`)

```python
# OLD: budget_max / 120.0 (refills in 2 minutes)
# NEW: budget_max / 180.0 (refills in 3 minutes)
```

Makes the tuner more deliberate with its changes.

---

## 4. Expected Behavior After Changes

### What to Watch For

1. **Activity levels should vary**: `short_activity` between ~0.1–0.5 for typical traffic, only reaching 0.7+ during genuine rush periods
2. **Personality params should track activity**: responsiveness and sociability should **increase** when the sidewalk is busy
3. **No more zombie state**: energy/responsiveness/sociability should never sit at 0.0 (floors are 0.25/0.30/0.20)
4. **Natural oscillation**: mean reversion + curiosity should create gentle parameter movement even during steady traffic
5. **Brightness should be moderate**: ~1.0–2.0 during busy times, rising toward 2.5+ only during truly quiet periods

### When to Consider Time-of-Day Resets

If after several hours the system still settles into a flat pattern, periodic resets at time-of-day transitions (6am, 12pm, 6pm, midnight) could blend parameters back toward home values by 30–50%. The existing `AGGRESSION_TIME_CAPS` system already provides some time-of-day sensitivity.

---

## 5. Files Modified

- `IO/light_behavior.py` — Log-scaled activity level computation (lines ~1473, ~1493)
- `IO/lightController_osc.py` — AutoTuningManager: safe floors, home values, curiosity, redesigned deltas, mean reversion, budget restore rate (lines ~1195–1505)
