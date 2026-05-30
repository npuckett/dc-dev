# Drop Ceiling — Software Behaviour, Trend Analysis & Self-Tuning

*Source report for the ACADIA 2026 project submission. Documents the final
production software (version **V6.5c**, March 2026). All file references point
into `dc-dev/IO/`.*

---

## 0. System at a glance

Drop Ceiling is a streetfront interactive light installation in a financial-district
storefront. A single **simulated point light** moves through a 3D field of LED panels
and is driven, in real time, by pedestrians tracked on the sidewalk outside.

The software is three cooperating processes (run as `systemd` services):

| Process | File | Role |
|---|---|---|
| Camera tracker | `camera_tracker_osc.py` | YOLO person detection on 2 RTSP cameras → world coordinates → OSC |
| Light controller | `lightController_osc.py` | Receives OSC, runs the behaviour system, outputs Art-Net DMX, serves WebSocket, logs to SQLite |
| Meta-tuner | `autotune_meta_review.py` | Scheduled deep analysis of the database that rewrites the controller's tuning config |

Data path:

```
2× Reolink cameras ──RTSP──▶ YOLO tracker ──OSC :7000──▶ light controller
        (25 fps)                                              │
                                                              ├─▶ Art-Net :6454 ─▶ LED panels
                                                              ├─▶ SQLite (tracking_history.db)
                                                              └─▶ WebSocket :8765 ─▶ public web viewer
                                                              ▲
                          meta-tuner (3×/day) ──autotune_overrides.json──┘
```

The light's character emerges from three nested control loops operating at very
different timescales:

1. **Per-frame (~30 Hz):** behaviour mode + gestures react to who is present *now*.
2. **Seconds-to-minutes:** short-term trend analysis biases idle/flow behaviour.
3. **Hours-to-days:** the meta-tuner and daily-learning loop reshape the
   personality, governed by an **interaction budget** that supplies *friction*.

---

## 1. The Meta-Parameters

The "meta-parameters" are the light's **personality**: high-level knobs (0–1 for
personality, multipliers for output) that change *how* the per-mode base values are
turned into actual light. They are defined in `light_behavior.py` in the
`MetaParameters` dataclass (`light_behavior.py:429`) and persisted in
`slider_settings.json`. Crucially, they are the variables the self-tuning system is
allowed to move — the behaviour modes give the *shape* of a response, the
meta-parameters give its *flavour*, and the tuner slowly evolves that flavour.

### 1a. Six personality parameters (0.0 – 1.0)

| Parameter | Low (0.0) | High (1.0) | What it drives on the light |
|---|---|---|---|
| `responsiveness` | slow, contemplative | quick, reactive | follow-smoothing, move speed, transition durations — how snappily the light locks onto a person |
| `energy` | calm, gentle | lively, dynamic | pulse speed, max brightness, gesture frequency — overall liveliness |
| `attention_span` | easily distracted | focused, loyal | follow-smoothing, dwell rewards, mode stickiness — how long it stays committed to one person |
| `sociability` | reserved | eager to engage | gesture chance, entrance flash, engaged brightness — willingness to reach out |
| `exploration` | stays put | wanders widely | wander-box size, wander interval, positional variety — appetite for new space |
| `memory` | forgets quickly | avoids repetition | anti-repetition strength, trend-influence weight — how much past behaviour suppresses repeats |

### 1b. Output multipliers (scaling)

| Multiplier | Default | Effect on the light |
|---|---|---|
| `brightness_global` | 1.0 | scales every brightness value (master dimmer / amplifier) |
| `speed_global` | 1.0 | scales all movement speeds |
| `pulse_global` | 1.0 | scales pulse (breathing) rate |
| `follow_speed_global` | 1.0 | scales how fast the light chases a tracked person |
| `dwell_influence` | 1.0 | how strongly dwell-time bonuses (brightness, tightness) apply (0 = off, 2 = double) |
| `idle_trend_weight` | 1.0 | how strongly *passive-zone trends* bend idle behaviour |

(Plus `trend_weight`, `time_of_day_weight`, `anti_repetition_weight` and a set of
boolean feature toggles such as `gestures_enabled`, `flow_mode_enabled`,
`self_analysis_enabled`.)

### 1c. How they reach the light

Every frame the **behaviour mode** (IDLE / FLOW / AWARE / ENGAGED / CROWD) supplies
*base* values from `MODE_PARAMS` (`light_behavior.py:600`) — move speed, brightness
min/max, pulse speed, falloff radius, follow smoothing. The meta-parameters then
modulate those bases (e.g. `final_pulse = base_pulse × lerp(1.3, 0.7, energy)`),
time-of-day modifiers scale them again (`TIME_CONFIGS`, `light_behavior.py:589`), and
proximity (how close the person is to the panels) applies a final per-axis multiply.
The result is interpolated toward smoothly so nothing snaps.

The **light itself** (the `PointLight`) only understands position, target,
brightness min/max, pulse speed, falloff radius/shape, and move speed — the
meta-parameters' entire job is to colour those few physical quantities.

> **Where the values actually sat in production** (`slider_settings.json`): the
> tuner had pushed `brightness_global ≈ 2.43`, `speed_global ≈ 1.46`,
> `responsiveness ≈ 0.83`, `energy ≈ 0.77`, `exploration ≈ 0.71` — i.e. a bright,
> brisk, exploratory personality had emerged from weeks of operation, well away from
> the neutral 0.5 / 1.0 starting points.

---

## 2. Behaviour modes (the per-frame loop)

The mode is chosen by `determine_mode()` (`light_behavior.py:996`) from the live
active/passive counts, with **stickiness** (conditions must persist) and
**minimum mode duration (8 s)** preventing flicker:

| Mode | Trigger | Character |
|---|---|---|
| **IDLE** | <2 people/min passing | gentle wander near panels; may "park" and let the falloff breathe |
| **FLOW** | ≥2 people/min on sidewalk | drifts laterally with the direction of foot traffic |
| **AWARE** *(new in V6.5)* | ≥10 people/min on sidewalk | energetic, wide reach, frequent repositioning |
| **ENGAGED** | 1 person in the active zone | follows the nearest person, breathing + subtle gestures |
| **CROWD** | 2+ in the active zone | follows the centroid, brightest, fastest, may "bloom" all panels |

Layered on top: a 16-entry **gesture library** (nod, lean, sway, orbit, settle,
breathe, bloom, sweep, focus…), **dwell phases** (notice→greet→engage→bond) that
deepen engagement over time, and a **proximity response** that makes the light
slower/brighter/tighter as someone steps closer.

These are the immediate, "real-time" layer. The next two sections are the parts the
submission highlights: how the light reads **trends** and **analyses itself** to keep
that immediate behaviour from going stale.

---

## 3. Short-term trends (seconds → minutes)

Two fast systems run continuously inside the controller.

### 3a. Multi-timescale idle trend analysis

When nobody is engaging, the light does **not** wander randomly. The `IdleTrends`
structure (`light_behavior.py:63`) queries the database across **four nested
windows** and folds them into three influence signals. Queries run on a **background
thread** so they never block the 30 Hz render loop.

| Window | Length | Question it answers |
|---|---|---|
| Recent | **1 minute** | Is anyone right next to us *now*? (immediate reactivity) |
| Short | **5 minutes** | Should we be poised for action? |
| Medium | **30 minutes** | What's the activity level of this half-hour? |
| Long | **1 hour** | What's the big-picture energy? |

The data comes from `TrackingDatabase.get_current_stats()` /
`get_trends(minutes)` (`tracking_database.py:991` & `:1032`), which return unique
people, average walking speed, active/passive event counts and L→R / R→L flow counts
for any window.

From these the system derives three normalised influences (`IdleTrends`,
`light_behavior.py:95`):

- **activity_anticipation (0–1)** — how "ready" to be; raises idle brightness baseline
  and shortens the wander interval when the recent windows are busy.
- **flow_momentum (−1…+1)** — sustained directional bias; shifts the wander box toward
  the side traffic is arriving from.
- **energy_level (0–1)** — overall energy to match in idle pulse/speed.

How strongly these bend behaviour is itself a meta-parameter: `idle_trend_weight`.

### 3b. Real-time flow tracking

A separate, faster `FlowState` (`light_behavior.py:170`) tracks the **dominant walking
direction** over a **30-second sliding window**, updated every **1.5 s**, EMA-smoothed
(α = 0.25). It outputs a direction (−1…+1) and a confidence strength (0–1) that:

1. shift the wander box toward incoming traffic,
2. trigger FLOW mode when passive traffic is heavy and sustained, and
3. (V6.5c) bias wander **target selection** — `_biased_point()` uses a triangular
   distribution peaking up to 60 % toward the side people are coming from, so the
   light "greets" approaching traffic instead of moving randomly.

### 3c. Aggression (an attention-seeking short-term loop)

`AggressionState` (`light_behavior.py:112`) is an EMA-smoothed 0–1 level that **rises**
the longer the light is ignored (passive traffic with no one stopping) and **falls**
when someone engages. It is **capped by hour of day** (`AGGRESSION_TIME_CAPS`,
`light_behavior.py:157`) to match the site — near-zero late at night, highest around
lunch. Higher aggression = wider/faster wander, more "bored" gestures, brighter pulses
toward passers-by.

**Net effect of the short-term layer:** the idle light is always quietly forecasting
the next minute-to-hour and pre-positioning/energising itself, instead of reacting only
after a person has already arrived.

---

## 4. Long-term trends & self-analysis (hours → days)

This is the heavier, scheduled analysis of the interaction database. There are **three
distinct scheduled jobs**, plus continuous in-loop self-analysis.

> **On the "≈4× a day" figure:** the deep meta-analysis that rewrites the light's
> tuning runs **3× per day**, not 4. The schedule is fixed in
> `systemd/autotune-meta-review.timer` at **06:00, 14:00, 22:00** (chosen to straddle
> the quiet→commute, midday, and evening→quiet transitions). Counting the separate
> midnight daily-report/daily-learning pass, the system does **four scheduled
> database analyses a day total** — which is likely the origin of the "4" — but the
> trend-driven *retuning* specifically is 3×.

### 4a. The meta-tuning review — 3×/day deep analysis

`autotune_meta_review.py` is launched by the timer as a one-shot process with an
**8-hour analysis window** (`--window 8`). Each run:

1. **Reads the database** — `behavior_adjustments`, `light_behavior`,
   `tracking_events` for the last 8 h (`get_adjustment_stats`, `get_param_values`,
   `get_mode_distribution`, `get_budget_stats`).
2. **Diagnoses pathologies** (`diagnose()`, `autotune_meta_review.py:334`):
   - parameters stuck at their floor/ceiling >80 % of the time,
   - activity so low it suggests night or a sensor fault,
   - mode starvation (e.g. <1 % engagement, >95 % idle ⇒ "raise personality floors"),
   - over-responsiveness (>30 % engaged ⇒ "lower personality home"),
   - **budget health** (depleted / throttled / always-full — see §5),
   - static parameters (no variance ⇒ "increase curiosity").
3. **Computes conservative config changes** (`compute_adjustments()`), capped at
   **25 changes per review** so one noisy window can't rewrite everything. It nudges
   `home_values`, `safe_floors`, `caps`, `curiosity`, `reversion`, and the
   **interaction budget**.
4. **Writes `autotune_overrides.json` atomically.** The running controller
   **hot-reloads** this file on its next update cycle, so changes take effect within
   seconds without a restart.
5. **Logs the entire analysis** into the `meta_tuning_reviews` table
   (`save_meta_tuning_review`, `tracking_database.py:1415`) — diagnosis,
   recommendations, old vs new config, per-parameter stats, mode distribution — making
   the system's own decisions auditable.

This is the loop that lets the personality (§1) evolve to match the site over weeks.

### 4b. Daily reporting & daily learning — once/day (with a 6 am catch-up)

- `generate_reports.py` runs from `daily-reports.timer` at **00:15** (after the
  midnight data cutoff) and again at **06:00** to catch late data. It rolls
  `hourly_stats` into a per-day `reports/daily/YYYY-MM-DD.json` (totals, peak/quietest
  hour, flow balance, hourly breakdown, mode distribution) plus an `_index.json`, then
  `deploy_reports.sh` pushes them to GitHub Pages for the public web viewer.
- Inside the controller, `DailyReportScheduler` (`lightController_osc.py:465`) also
  fires at **12:01 AM**: it briefly pauses tracking, generates yesterday's report, and
  feeds it to the V6 daily-learning hook (`on_daily_report`,
  `v6_integration.py:407`). Daily learning computes a 7-day weighted average of
  engagement by time-of-day and blends ~30 % of it into the next day's starting
  personality, so the light gradually shifts toward what worked at each hour.

### 4c. Continuous in-loop self-analysis

The light constantly studies *its own* output (the `light_behavior` table) to fight
repetition — the explicit design goal of "evolution, not just reaction":

- **Position entropy** (`get_position_entropy`, `tracking_database.py:846`) — grid
  occupancy over the last hour; low entropy ⇒ it's stuck in one area ⇒ bias toward
  unexplored space.
- **Mode distribution** (`get_mode_distribution`) — % time in each mode over 24 h.
- **Response similarity** (`get_response_similarity`, `:936`) — variance of its own
  responses to similar people-counts; high similarity ⇒ force more variety.
- **Position cooldown** (`is_position_recently_visited`) — don't revisit a spot within
  30 s.

These metrics are bundled by `get_behavior_analysis()` (`tracking_database.py:982`).

---

## 5. The interaction budget (the "friction")

The budget is the governor that stops the self-tuning system from chasing noise — the
*friction* the submission refers to. It lives in `autotune_overrides.json` under
`budget` and is enforced by the V6 `SmartAutoTuner` (referenced throughout
`v6_integration.py`, e.g. `self.autotuner.budget`).

Mechanism (production values from `autotune_overrides.json`):

```json
"budget": { "max": 200.0, "restore_seconds": 600.0, "cost_scale": 30.0 }
```

- The tuner runs every few seconds and wants to nudge parameters. **Every change
  spends budget** proportional to its size (`cost_scale`).
- Budget **restores gradually** (toward `max` over `restore_seconds`).
- When the budget is low, proposed changes are **throttled** — the light can make a
  few meaningful adjustments, then must "earn back" the right to keep moving. This
  prevents rapid, large personality swings even when the live signal is jittery.

The budget is also **self-regulated** by the 3×/day meta-review, which is the elegant
part: `get_budget_stats()` + `diagnose()` measure whether the budget is actually
providing useful friction and `compute_adjustments()` retunes it (`autotune_meta_review.py:539`):

- budget **almost always full** (>90 %) ⇒ it isn't constraining ⇒ **tighten**
  (lower `max`, lengthen `restore_seconds`);
- budget **depleted >50 %** or throttling >30 % of changes ⇒ too tight ⇒ **loosen**
  (raise `max`, shorten restore), scaled by how badly it was starved;
- budget changes are **limited to once per day** (`was_budget_adjusted_today`) so the
  friction setting itself stays stable.

Two further friction mechanisms ride alongside the budget in the tuner config:
**mean-reversion** (`reversion`: a constant gentle pull of every parameter back toward
its `home_value`, preventing drift to extremes) and **curiosity** (`curiosity`: a
small periodic random nudge to one parameter so the space keeps being explored). The
meta-review tunes both — raising reversion when too many parameters are clamped,
raising curiosity when parameters have gone static.

**Summary:** short-term trends and live presence constantly *push* the parameters;
the budget, mean-reversion and caps *resist*; and the 3×/day meta-review periodically
adjusts how hard each side pushes. That balance is what keeps the light feeling alive
but never frantic over a 24/7 deployment.

---

## 6. The database — what is stored and why

A single **SQLite** database, `tracking_history.db` (WAL mode, batched commits every
50 writes or 1 s), defined in `tracking_database.py:212`. It serves three masters:
real-time behaviour, trend analysis, and the public reports. Data is tiered — raw
events live ~48 h, aggregates live forever — so the file stays performant for years.

| Table | Stores | Why it exists | Retention |
|---|---|---|---|
| `tracking_events` | every person position: x, z, velocity, speed, **zone**, **flow_direction** | the raw interaction record; source for all short-term trend queries | 48 h |
| `light_behavior` | the light's own state each tick: mode, position, target, brightness, pulse/move speed, people counts, gesture | **self-analysis** — lets the light study and avoid repeating itself | 48 h |
| `behavior_adjustments` | every auto-tuning event: short/medium/long activity, energy, aggression, **budget before/after/cost**, old→new values | the audit trail the meta-review reads to diagnose the tuner | 48 h |
| `person_sessions` | individual visit start/end/duration, zone entered, flow | dwell-time and conversion analysis | 48 h |
| `hourly_stats` | per-hour rollup: people, active/passive, avg speed, flow, blooms, dominant mode, avg brightness | permanent trend history after raw events are pruned | **forever** |
| `daily_stats_v2` | per-day rollup: peak/quietest hour, flow balance, totals, blooms | long-term day-over-day trends | **forever** |
| `autotune_daily_learnings` | what the system learned each day: optimal values, parameter "journeys", strategy summary, learned caps | feeds next-day personality blending | **forever** |
| `meta_tuning_reviews` | full record of each 3×/day review: diagnosis, recommendations, old/new config, mode dist, param stats | makes the self-tuning transparent and reversible | **forever** |

Notable derived fields recorded at write time: **zone** (active vs passive vs unknown,
from the calibrated x/z) and **flow_direction** (L→R / R→L / stationary, from velocity)
— computing these on ingest is what makes the minute-scale trend queries cheap.

Aggregation is automatic: `aggregate_hour()` rolls completed hours into `hourly_stats`;
`prune_with_aggregation()` guarantees an hour is summarised *before* its raw rows are
deleted, so no trend data is ever lost to pruning.

---

## 7. Real-time data ingestion

**Capture.** `camera_tracker_osc.py` runs YOLO (`yolo11n.pt`, CUDA when available)
on **two Reolink RTSP cameras** at a target **25 fps**, batching both cameras into a
single inference call. Each detection's foot point is projected onto the floor plane
using ArUco-marker camera calibration (`camera_calibration.json`), giving a real-world
(x, z) in centimetres. Detections from the two cameras are **fused** (merged within a
distance threshold) and **tracked/smoothed** across frames (EMA + velocity coasting for
briefly-lost tracks).

**Transport.** Tracked people are sent over **OSC/UDP to port 7000**:

- `/tracker/count <n>`
- `/tracker/person/<id> <x> <z>`

**Consumption.** The controller's OSC handler feeds `TrackedPersonManager`
(`lightController_osc.py:865`), which applies calibration offsets, classifies each
person into **active / passive / unknown** zones, computes per-person velocity, and
raises enter/leave/move callbacks into the behaviour system. The same updates are
written to the database via `record_position()`. So a single pedestrian step becomes,
within ~40 ms: a light reaction, a database row, and a WebSocket update.

**Output.** The resulting light state is rendered to the LED panels over
**Art-Net/UDP port 6454** (per-panel DMX from a distance-falloff model), and the light
also drives a local 3D OpenGL preview.

---

## 8. The web interface

The controller serves a **WebSocket on port 8765** (`WebSocketBroadcaster`,
`lightController_osc.py:1028`), and a static **Three.js public viewer**
(`public-viewer/`) renders the installation live in the browser. There are two clearly
separated data planes.

### 8a. Immediate / real-time plane (WebSocket)

The broadcaster pushes state at **~15 fps** (`WEBSOCKET_BROADCAST_INTERVAL = 0.066`),
hardened for crowds (200-client cap, batched sends, change-detection so identical
frames aren't re-serialised). Each payload carries the light position, the tracked
people, the current **mode**, **gesture**, **dwell phase**, a human-readable
**behaviour_description** ("Engaged · Breathing Together", "Flow · Drifting with
Traffic"), live population counts, and a V6 state extension (predicted traffic regime,
tuner budget, top gradients, current strategy). The viewer renders the light + panels +
people in 3D and shows the plain-language status — the immediate data is simply the
broadcast state, computed by the controller and displayed as received (no client-side
trend maths).

### 8b. Long-term / trends plane (daily JSON)

Long-term trends are **pre-computed server-side**, not calculated in the browser. The
`generate_reports.py` job writes `reports/daily/YYYY-MM-DD.json` (+ `_index.json`) and
`deploy_reports.sh` publishes them to GitHub Pages. The reports page
(`public-viewer/reports.js` / `reports.html`) fetches the index and the per-day JSON
and visualises: hourly activity curves, peak vs quietest hour, dominant flow direction
/ flow balance, totals (unique people, active visits, blooms), and the light's mode
distribution and auto-tuning strategy summary for the day. Because the heavy
aggregation already happened in SQLite, the browser only fetches a few KB of JSON and
draws charts — it never touches the database directly.

**Division of labour:** WebSocket = "what is the light doing this instant"; daily JSON
= "what happened over hours/days and how did the light adapt." Together they let a
visitor see both the live organism and its longer memory.

---

## 9. One-paragraph version (for the submission body)

> Drop Ceiling's light is governed by a small set of *meta-parameters* — a six-axis
> personality (responsiveness, energy, attention span, sociability, exploration,
> memory) plus output multipliers — that colour how each behaviour mode is turned into
> movement, brightness and pulse. These parameters are not fixed: the system reads
> pedestrian *trends* at nested timescales (1-minute to 1-hour activity windows plus a
> 30-second flow tracker) to bias its idle and flow behaviour in real time, and three
> times a day it performs a deep 8-hour analysis of its interaction database —
> diagnosing stuck parameters, mode starvation and over-reaction — and rewrites its own
> tuning config, which the running light hot-reloads. An *interaction budget* meters how
> much the tuner may change per unit time, supplying the friction that keeps adaptation
> deliberate rather than twitchy; the daily analysis even retunes that budget. All of
> it is recorded in a tiered SQLite database (raw events for 48 hours, hourly and daily
> aggregates forever, plus a full audit of every tuning decision), which simultaneously
> drives the light, feeds the trend analysis, and publishes the daily web reports.

---

*Compiled from the V6.5c source: `light_behavior.py`, `lightController_osc.py`,
`tracking_database.py`, `autotune_meta_review.py`, `generate_reports.py`,
`V6Dev/v6_integration.py`, `systemd/*.timer`, `slider_settings.json`,
`autotune_overrides.json`, `public-viewer/`, and `BEHAVIOR_SYSTEM.md` /
`LIGHT_BEHAVIOR_DESIGN.md`.*
