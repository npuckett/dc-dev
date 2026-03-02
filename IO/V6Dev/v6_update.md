# V6 Advanced Decision & Auto-Tuning System

## Overview

V6 adds **eight new modules** that slot into the existing V5 light controller without modifying any V5 files. The system replaces V5's reactive heuristics with data-driven, self-improving decision-making.

### What's New

| V5 Problem | V6 Solution | Module |
|---|---|---|
| Raw activity count as fitness signal | **Composite engagement score** (conversion rate, dwell depth, mode diversity, return visits, param stability) | `engagement_score.py` |
| Hand-authored 6-anchor time-of-day profiles | **Learned profiles** from 20+ daily reports with recency weighting | `predictive_context.py` |
| Round-robin A/B test for almost-engaged | **Thompson Sampling bandit** that exploits winners while still exploring | `strategy_bandit.py` |
| Feedback weights only increase (monotonic) | **Bidirectional feedback** with hourly decay toward neutral | `feedback_learning_v6.py` |
| Y-axis scale never used, rotation barely used | **Per-mode falloff shapes**, proximity-reactive shaping, 4 new gestures (REACH, EMBRACE, BEACON, TWIRL) | `falloff_strategies.py` |
| Heuristic delta rules, no cross-param awareness | **Gradient estimation** via sliding window regression, cross-param correlation detection | `smart_autotuner.py` |
| Binary mode switching | **Predictive pre-transitions**, adaptive stickiness, CROWD_SOCIAL vs CROWD_SCATTERED | `mode_intelligence.py` |
| 13-step last-write-wins modifier chain | **Intent-based resolution** with priority levels and per-source budgets | `modifier_resolver.py` |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  V5 Main Loop                        │
│  behavior.update() → behavior_params, behavior_status│
│                          │                           │
│                    ┌─────▼─────┐                     │
│                    │ V6 Bridge │  ◄── v6_integration  │
│                    └─────┬─────┘                     │
│           ┌──────────────┼──────────────┐            │
│     ┌─────▼─────┐  ┌────▼────┐  ┌──────▼──────┐     │
│     │ Engagement│  │Predictive│  │  Strategy   │     │
│     │  Scorer   │  │ Context  │  │   Bandit    │     │
│     └─────┬─────┘  └────┬────┘  └──────┬──────┘     │
│           │              │              │            │
│     ┌─────▼─────┐  ┌────▼────┐  ┌──────▼──────┐     │
│     │  Smart    │  │Feedback │  │   Falloff   │     │
│     │AutoTuner  │  │Learn V6 │  │ Strategies  │     │
│     └─────┬─────┘  └────┬────┘  └──────┬──────┘     │
│           │              │              │            │
│     ┌─────▼─────┐       │              │            │
│     │   Mode    │       │              │            │
│     │  Intel    │       │              │            │
│     └─────┬─────┘       │              │            │
│           └──────────────┼──────────────┘            │
│                    ┌─────▼─────┐                     │
│                    │ Modifier  │                      │
│                    │ Resolver  │                      │
│                    └─────┬─────┘                     │
│                          │                           │
│              Modified behavior_params                │
│                          │                           │
│              light.target_falloff_scale ←            │
│              light.target_falloff_rotation ←         │
│              MetaParameters updated ←                │
└─────────────────────────────────────────────────────┘
```

---

## Integration Guide

### Minimal Integration (3 lines in lightController_osc.py)

```python
# At the top of main():
from V6Dev.v6_integration import V6Integration
v6 = V6Integration(meta, sliders, db, behavior, light,
                    reports_dir='reports/daily')
auto_tuner.set_enabled(False)  # V6 replaces V5 auto-tuner

# In the main loop, after behavior.update() and before light.update():
behavior_params = v6.tick(behavior_status, behavior_params,
                          tracked_people, dt, now)
```

### Full Integration (with person events + reports)

```python
# Setup
from V6Dev.v6_integration import V6Integration, V6Config
config = V6Config(
    enable_smart_autotuner=True,
    enable_predictive_context=True,
    enable_strategy_bandit=True,
    enable_feedback_v6=True,
    enable_falloff_strategies=True,
    enable_mode_intelligence=True,
    enable_modifier_resolver=True,
)
v6 = V6Integration(meta, sliders, db, behavior, light, config=config)
auto_tuner.set_enabled(False)

# Person event hooks (chain after V5 callbacks):
original_entered = tracked_manager.on_person_entered
def on_entered(person):
    original_entered(person)
    v6.on_person_entered(person, time.time())
tracked_manager.on_person_entered = on_entered

original_left = tracked_manager.on_person_left
def on_left(person):
    original_left(person)
    v6.on_person_left(person, time.time())
tracked_manager.on_person_left = on_left

# Daily report hook:
original_report = report_scheduler.on_report_ready
def on_report(report):
    original_report(report)
    v6.on_daily_report(report)
report_scheduler.on_report_ready = on_report

# WebSocket state extension:
ws_state = {...}  # V5 state dict
ws_state.update(v6.get_state_extension())
```

### Selective Module Activation

Each module can be independently toggled:

```python
config = V6Config(
    enable_smart_autotuner=True,
    enable_predictive_context=True,
    enable_strategy_bandit=False,      # keep V5 round-robin
    enable_feedback_v6=False,          # keep V5 feedback 
    enable_falloff_strategies=True,    # new spatial expression
    enable_mode_intelligence=True,
    enable_modifier_resolver=False,    # direct application instead
)
```

---

## Module Details

### 1. Engagement Score (`engagement_score.py`)

Replaces raw activity count with a composite quality metric:

| Component | Weight | Description |
|---|---|---|
| `conversion_rate` | 30% | What fraction of detected people become engaged |
| `dwell_depth` | 25% | How deep into the dwell progression people get (notice→greet→engage→bond) |
| `mode_diversity` | 20% | Shannon entropy across mode distribution (higher = more diverse) |
| `return_visits` | 15% | Are people coming back? |
| `parameter_stability` | 10% | Are parameters settled (good) or thrashing (bad)? |

Methods:
- `compute(behavior_status)` — real-time frame-by-frame scoring
- `smoothed_score(window)` — exponentially-weighted moving average
- `compute_from_report(report)` — offline scoring from daily JSON
- `compute_hourly_from_report(report)` — per-hour breakdown

### 2. Predictive Context (`predictive_context.py`)

Builds learned time-of-day profiles from accumulated daily reports:

- **Recency weighting**: Recent days count more; same-day-of-week gets 2× bonus
- **Regime classification**: `dead` (<5/hr), `trickle` (5-50), `steady` (50-500), `rush` (500+), `event` (>2σ anomaly)
- **Outputs**: `get_context()` returns predicted traffic, regime, optimal home values
- **Adaptive tuning**: `get_tune_interval()` returns 3s during rush, 30s during dead
- **Budget scaling**: `get_budget_multiplier()` returns 0.5× dead, 1.5× rush, 3× event

### 3. Strategy Bandit (`strategy_bandit.py`)

Thompson Sampling with 4 strategies for almost-engaged candidates:

| Strategy | Description | Falloff Effect |
|---|---|---|
| `BRIGHTNESS_PULSE` | Rapid brightness flash toward candidate | — |
| `DRIFT_TOWARD` | Light slowly moves toward candidate | — |
| `PAUSE_AND_LOOK` | Light stops and "stares" (focused falloff) | — |
| `FALLOFF_RESHAPE` | **V6 new** — reshapes falloff to reach toward candidate | scale_x=1.4, scale_z=2.2, dynamic rotation |

- Beta distribution priors per (strategy, context_bucket)
- Context = TimePeriod × FlowBucket (morning/afternoon/evening/night/rush × low/medium/high)
- 14-day half-life decay (0.998/cycle)
- Persists to `bandit_priors.json`

### 4. Feedback Learning V6 (`feedback_learning_v6.py`)

Bidirectional feedback with ~30 context buckets:

**Key changes from V5:**
- Weights can **decrease** (negative feedback at 0.5× learning rate)
- Hourly decay 0.997 toward neutral (1.0)
- Bounds [0.4, 2.5] (V5 was uncapped above)
- Dwell quality scaling: 30s+ engagement → 2× reward; <3s → 0.3×
- New context dimensions: speed_bucket, group_bucket, regime_bucket
- **V6 additions**: `falloff_reach_mult` and `falloff_width_mult` modifiers

### 5. Falloff Strategies (`falloff_strategies.py`)

First-class use of V5's anisotropic falloff — **the Y-axis finally gets used**.

**Per-mode default shapes:**
| Mode | X | Y | Z | Character |
|---|---|---|---|---|
| idle | 1.3 | 0.9 | 1.1 | Slightly wide, relaxed |
| engaged | 0.8 | 1.3 | 0.9 | Tall (Y!), focused |
| crowd | 1.5 | 1.1 | 1.3 | Wide coverage |
| flow | 1.2 | 0.9 | 1.4 | Deep into traffic |

**New V6 gestures:**
| Gesture | Effect | Duration |
|---|---|---|
| REACH | Z=2.5 deep extension toward a person | 3.0s attack curve |
| EMBRACE | X=2.0 + Y=1.6 — wide & tall "hug" | 4.0s sustain curve |
| BEACON | rhythmic Z pulse (1.0→2.0→1.0) | 2.0s pulse curve |
| TWIRL | rotation sweep from -0.5 to +0.5 rad | 2.5s bell curve |

**Proximity-reactive shaping:**
- Person at Z < 0.3 → Y scale grows to 1.4 (enveloping feel)
- Flow → rotation aligns into traffic direction (±0.5 rad)

### 6. Smart Auto-Tuner (`smart_autotuner.py`)

Gradient-informed replacement for V5's heuristic delta rules:

- **Fitness function**: `EngagementScorer.smoothed_score()` instead of raw activity
- **Gradient estimation**: Sliding window of 50 (param_vector, score) samples; OLS regression estimates ∂score/∂param for each parameter
- **Cross-parameter detection**: Pearson correlation between param pairs during high-score periods; correlated params get linked adjustments
- **Regime strategies**: dead=20% churn, trickle=boost attention-seeking, rush=personality up + output calm, event=max exploration
- **Budget**: Context-dependent max (0.5× dead, 1.5× rush, 3× event)
- **Curiosity**: Random perturbation every 30s biased toward home values
- **Mean reversion**: Context-aware strength, anomaly-adjusted

### 7. Mode Intelligence (`mode_intelligence.py`)

Enhanced mode transition logic:

- **Pre-transition blending**: Starts interpolating toward ENGAGED at 30% blend when a candidate scores >0.55 (approaching, lingering, close)
- **Adaptive stickiness**: Bond phase = 10s lock, greet = 3s, notice = 2s
- **CROWD sub-modes**: `CROWD_SOCIAL` (clustered group, clustering >0.6) vs `CROWD_SCATTERED` (dispersed)
- **Session familiarity**: Each ENGAGED visit increases familiarity (+0.15 up to 2.0×); decays 0.97×/min
- **Momentum**: Repeated engagements build intensity multiplier (+0.05 per event, max 1.6×, decays 0.995/s when idle)

### 8. Modifier Resolver (`modifier_resolver.py`)

Intent-based system replacing last-write-wins:

**Priority levels:**
1. `SAFETY` — floors/caps (never overridden)
2. `CONTEXT` — time-of-day regime
3. `STRATEGY` — bandit/autotuner
4. `FEEDBACK` — learned weights  
5. `AESTHETIC` — mode defaults, gestures

**Resolution algorithm:**
1. Each subsystem emits `ModifierIntent(parameter, direction, strength, source, priority, confidence)`
2. Intents grouped by parameter, sorted by priority
3. Weighted combination: weight = confidence × (6 - priority_level)
4. Per-source budget clamping prevents any one system from dominating
5. Safety floors/caps applied last

**Convenience builders** for converting each V6 module's output to intents:
- `intents_from_autotuner_deltas()` 
- `intents_from_feedback()`
- `intents_from_strategy()`
- `intents_from_mode_overlay()`
- `intents_from_context()`

---

## File Structure

```
IO/V6Dev/
├── __init__.py              Package init (exports V6Integration)
├── engagement_score.py      Composite engagement scoring
├── predictive_context.py    Learned time-of-day profiles
├── strategy_bandit.py       Thompson Sampling bandit
├── feedback_learning_v6.py  Bidirectional feedback with decay
├── falloff_strategies.py    Per-mode falloff shaping + gestures
├── smart_autotuner.py       Gradient-informed auto-tuning
├── mode_intelligence.py     Predictive mode transitions
├── modifier_resolver.py     Intent-based modifier resolution
├── v6_integration.py        Bridge layer (V6 → V5)
└── v6_update.md             This documentation
```

---

## Persistence Files

V6 creates/reads these files:

| File | Module | Purpose |
|---|---|---|
| `bandit_priors.json` | strategy_bandit | Beta distribution arms per (strategy, context) |
| `feedback_weights_v6.json` | feedback_learning_v6 | Learned feedback weights per context bucket |
| `autotune_overrides.json` | smart_autotuner | Hot-reloadable config (home values, floors, caps) |
| `reports/daily/*.json` | predictive_context | Read-only; daily reports generated by V5 |

---

## Data Flow Per Frame

```
Frame N:
  V5 behavior.update(dt) → behavior_params + behavior_status
                                    │
                              v6.tick(...)
                                    │
  ┌─────────────────────────────────┤
  │                                 │
  ├─ ModeIntelligence.update()      │ → ModeOverlay (pre-transition, stickiness)
  ├─ PredictiveContext.get_context() │ → regime, predicted_traffic, home_values
  ├─ StrategyBandit.select()        │ → strategy_effect (for almost-engaged)
  ├─ FeedbackLearningV6.get_mods()  │ → multiplicative modifier weights
  ├─ SmartAutoTuner.update()        │ → parameter deltas (gradient-informed)
  ├─ FalloffStrategyManager()       │ → FalloffShape (scale_xyz, rotation)
  │                                 │
  └──── ModifierResolver ───────────┘
              │
              ▼
      Modified behavior_params
      Updated MetaParameters
      New falloff_scale_x/y/z/rotation
              │
              ▼
      light.target_falloff_scale = [sx, sy, sz]
      light.target_falloff_rotation = r
```

---

## Falloff: What Changed

V5 had the anisotropic falloff infrastructure but barely used it:
- **Y-axis**: Never driven (always ~1.0)
- **Rotation**: Limited to ±0.4 rad, only in 2 gestures
- **No per-mode defaults**: Every mode started at [1, 1, 1]
- **No proximity response**: Person distance didn't affect shape

V6 makes falloff a **primary expressive tool**:
- Every mode has a characteristic shape (idle=wide, engaged=tall, flow=deep)
- Y-axis drives EMBRACE gesture (1.6 — first real Y use)
- Rotation sweeps ±0.5 rad for TWIRL gesture
- REACH extends Z to 2.5 toward approaching people
- BEACON pulses Z rhythmically as an attention signal
- Proximity naturally modulates Y (close → enveloping)
- Flow direction drives rotation alignment

---

## Testing

```bash
cd IO
python -c "from V6Dev import V6Integration; print('V6 modules loaded OK')"
```

Individual module tests:
```python
from V6Dev.engagement_score import EngagementScorer
from V6Dev.predictive_context import PredictiveContextEngine
from V6Dev.strategy_bandit import StrategyBandit
from V6Dev.feedback_learning_v6 import FeedbackLearningV6
from V6Dev.falloff_strategies import FalloffStrategyManager
from V6Dev.smart_autotuner import SmartAutoTuner
from V6Dev.mode_intelligence import ModeIntelligence
from V6Dev.modifier_resolver import ModifierResolver
```

---

## V6.1 Calibration Changes (2026-03-01)

### Problem Diagnosis

Analysis of daily reports (Feb 25 – Mar 1) revealed a **passivity spiral**:
- `idle_trend_weight` climbed from 0.924 → 0.995
- `energy` hit floor at 0.301
- Idle mode share: 30.2% → 71.2%
- Engaged mode share: 15.4% → 4.4%
- Active:passive ratio: 1:11 → 1:57

Root cause: V6 was designed for a high-engagement installation but deployed on a
passive-heavy sidewalk (7k–16k people/day, active:passive ratio typically 1:50 to 1:100).
The engagement score was near-zero, the autotuner had no gradient signal, and dead-regime
logic pushed everything DOWN — exactly the wrong direction for attracting attention.

### Changes by File

#### `engagement_score.py`
- **Weights rebalanced**: conversion_rate 0.30→0.20, added passive_awareness (0.15) and proactive_reach (0.10), reduced return_visits 0.15→0.10, stability 0.10→0.05
- **Conversion normalization**: 10%→3% (realistic for sidewalk)
- **Smoothing window**: 12→8 (respond faster)
- **Daytime score floor**: 0.15 minimum so gradient estimator always has signal
- **New methods**: `record_gesture_attempt()`, `record_strategy_attempt()`, `record_mode_transition()`, `reset_proactive_counters()`

#### `predictive_context.py`
- **Regime thresholds**: Changed from total people to active-zone people: dead <2, trickle 2–15, steady 15–100, rush 100–500
- **V5 cold-start seeding**: Added 6-anchor time profiles from V5 for bootstrapping when <14 days history
- **Active-zone counting**: Uses `_use_active_zone_counts` flag instead of total count
- **Dead-regime inverted**: budget multiplier 0.5×→1.5×, tune interval 30s→15s (dead = MORE tuning, not less)
- **Trickle boost**: 0.75×→1.25×, interval 15s→8s

#### `smart_autotuner.py`
- **idle_trend_weight hard cap**: (0.0, 2.0)→(0.0, 0.75)
- **Safe floor**: 0.10→0.20
- **Gradient window**: 50→30
- **Home values**: idle_trend_weight seeded at 0.50
- **Dead regime multiplier**: 0.2→0.6
- **Trickle boosts**: tripled (+0.005→+0.015) for brightness, exploration, energy
- **Smoothed score window**: 12→8
- **Anti-passivity block**: When idle >60s, pushes energy/responsiveness/sociability UP toward home+0.1, pushes idle_trend_weight DOWN when >0.65

#### `modifier_resolver.py`
- **Autotuner budget**: (0.30, 0.08)→(0.50, 0.12) — autotuner gets more room to make changes
- **Context engine budget**: (0.60, 0.15)→(0.80, 0.20)
- **Dead-regime context intents inverted**: brightness/exploration UP (+0.05), energy UP (+0.03) instead of all DOWN
- **Trickle-regime intents added**: brightness/exploration UP (+0.03)

#### `feedback_learning_v6.py`
- **Learning rate**: 0.03→0.05 (learn faster from rare engagement events)
- **Hourly decay**: 0.997→0.993 (forget stale quiet-hour weights faster)
- **Quiet mode boost**: When no engagement for >5 minutes, ramps brightness ×1.0→1.2 and pulse over 10 minutes

#### `mode_intelligence.py`
- **Pre-transition score**: 0.55→0.40 (start blending earlier on passive installation)
- **Distance weight**: 0.3→0.4 in candidate scoring (proximity is strongest engagement predictor)
- **Passive-zone awareness**: People near active/passive boundary (z 250–283) get +0.10 bonus in candidate scoring

#### `falloff_strategies.py`
- **New `idle_beacon` mode shape**: scale_x=1.6, scale_y=1.2, scale_z=1.4, radius_mult=1.15 — maximum visual presence when idle for extended periods
- **BEACON energy threshold**: 0.4→0.3 (fires more often when energy is low)

#### `v6_integration.py`
- **Active-zone count passthrough**: Context engine now receives active_zone_count instead of total
- **Proactive counter wiring**: `scorer.record_gesture_attempt()` on person enter, `scorer.record_strategy_attempt()` on bandit selection
- **`v6_health_check()`**: Runs every 5 minutes, checks idle_trend_weight, energy, autotuner budget, engagement score
- **Passivity spiral detection**: In `on_daily_report()`, detects idle_trend_weight >0.85 AND energy <0.35, auto-resets to safe values (idle_trend_weight→0.50, energy→0.50+, responsiveness→0.50+)
- **Passivity spiral warning**: Logged in periodic status line

### Design Philosophy

These changes follow three principles:

1. **Dead ≠ Do Nothing**: On a passive-heavy sidewalk, dead/trickle periods are the majority. The system must be MORE expressive during these periods, not less. Visibility attracts attention.

2. **Cap the Damage**: `idle_trend_weight` hard-capped at 0.75, with auto-reset at 0.85. This prevents the runaway loop where the system teaches itself to be invisible.

3. **Always Have Signal**: A score floor of 0.15, faster decay, and active-zone counting ensure the gradient estimator always has something to work with — even during quiet hours.
