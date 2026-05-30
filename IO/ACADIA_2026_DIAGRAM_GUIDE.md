# Drop Ceiling — System Logic for Diagrams & Drawings

*Companion to [ACADIA_2026_SOFTWARE_REPORT.md](ACADIA_2026_SOFTWARE_REPORT.md). This
document pairs short prose with structured "diagram-ready" blocks. Each block marked
**◧ DRAW** is a suggestion for a figure you can lift directly. Production software =
**V6.5c** (March 2026), `dc-dev/IO/`.*

---

## 1. The whole system in one picture

Drop Ceiling turns a sidewalk into an instrument. Two cameras watch pedestrians; a
single simulated point light, hovering in a 3D field of LED panels, responds to them.
What makes it more than a motion sensor is that it runs **three nested control loops at
three timescales** — it reacts to the instant, anticipates the next hour, and reshapes
its own personality over days — with an *interaction budget* providing the friction that
keeps that self-adaptation deliberate rather than twitchy.

**◧ DRAW — Top-level pipeline (left-to-right signal flow):**

```
[2× Reolink cameras]                                    [LED panels]
        │ RTSP 25fps                                          ▲ Art-Net/UDP :6454
        ▼                                                     │
┌──────────────────┐   OSC/UDP :7000   ┌────────────────────────────────────┐
│ camera_tracker   │ ────────────────▶ │        lightController_osc          │
│   _osc.py        │  /tracker/count   │  (behaviour + tuning + DB + WS)     │
│  YOLO → floor xy │  /tracker/person  │                                     │
└──────────────────┘                   └───────┬─────────────────┬───────────┘
                                                │ WebSocket :8765 │ writes
                                                ▼                 ▼
                                       [Three.js web viewer]  [SQLite DB]
                                                                  ▲
                              autotune_overrides.json (hot-reload)│ reads
                                                ┌─────────────────┴──────────┐
                                                │  autotune_meta_review.py    │
                                                │  (scheduled 3×/day)         │
                                                └─────────────────────────────┘
```

Three OS processes (each a `systemd` service): **tracker**, **controller**,
**meta-tuner**. Everything they share passes through two channels — **OSC** (live
positions) and the **SQLite database** (memory).

---

## 2. The three nested loops (the conceptual core)

This is the single most important diagram for the submission — it shows the project is
about *time and adaptation*, not just reactivity.

**◧ DRAW — Three concentric loops, fastest innermost:**

```
        ┌────────────────────────────────────────────────────────┐
        │  LOOP 3  — SELF-TUNING        timescale: hours → days    │
        │  meta-review (3×/day) + daily learning reshape           │
        │  the PERSONALITY. Friction = interaction budget.         │
        │   ┌──────────────────────────────────────────────────┐  │
        │   │ LOOP 2 — ANTICIPATION     timescale: sec → hours  │  │
        │   │ trend analysis (1/5/30/60 min) + 30s flow tracker │  │
        │   │ bias idle/flow behaviour & pre-position the light │  │
        │   │   ┌────────────────────────────────────────────┐ │  │
        │   │   │ LOOP 1 — REACTION    timescale: ~33 ms      │ │  │
        │   │   │ behaviour mode + gestures follow who is     │ │  │
        │   │   │ present NOW (30 Hz render / Art-Net)        │ │  │
        │   │   └────────────────────────────────────────────┘ │  │
        │   └──────────────────────────────────────────────────┘  │
        └────────────────────────────────────────────────────────┘
```

| Loop | Cadence | Reads | Writes / Affects |
|---|---|---|---|
| **1 — Reaction** | ~30 Hz | live OSC people + zones | light position, brightness, pulse, gesture |
| **2 — Anticipation** | 1.5 s (flow), background thread (trends) | DB trend queries (1/5/30/60 min) | wander box bias, idle energy, mode selection |
| **3 — Self-tuning** | 3×/day (+ midnight daily learning) | 8 h of DB history | rewrites meta-parameters via `autotune_overrides.json` |

The outer loops do not move the light directly — **they change the rules the inner loop
plays by.** Loop 2 biases *where/how energetically* it idles; Loop 3 changes *the
personality* that colours every reaction.

---

## 3. Meta-parameters — the light's personality

Prose: the behaviour *mode* decides the **shape** of a response (e.g. "follow the
nearest person, brighten, tighten the beam"). The **meta-parameters** decide its
**flavour** (eager vs reserved, brisk vs contemplative, bright vs dim). They are the
only variables the self-tuning loop is allowed to move — which is exactly why the light
can "grow a character" over weeks. Defined in `MetaParameters`
([light_behavior.py:429](light_behavior.py)), persisted in `slider_settings.json`.

### 3a. Six personality axes (0.0 – 1.0)

**◧ DRAW — a 6-spoke radar/wheel; show "neutral 0.5" ring vs "as-deployed" values.**

| Parameter | 0.0 ⟶ 1.0 | Light outputs it bends | As-deployed* |
|---|---|---|---|
| `responsiveness` | contemplative ⟶ reactive | follow-smoothing, move speed, transition time | **0.83** |
| `energy` | calm ⟶ lively | pulse rate, max brightness, gesture frequency | **0.77** |
| `attention_span` | distractible ⟶ loyal | follow-smoothing, dwell rewards, mode stickiness | 0.51 |
| `sociability` | reserved ⟶ eager | gesture chance, entrance flash, engaged brightness | 0.44 |
| `exploration` | stays put ⟶ wanders | wander-box size, wander interval, position variety | **0.71** |
| `memory` | forgets ⟶ avoids repeats | anti-repetition strength, trend-influence weight | 0.49 |

\* from `slider_settings.json` — the tuner had evolved a **bright, brisk, exploratory**
character, far from the 0.5 neutral start. *(Good caption for a "before/after personality" figure.)*

### 3b. Six output multipliers (scaling knobs)

**◧ DRAW — a small "mixing console" of faders.**

| Multiplier | Default | Effect | As-deployed |
|---|---|---|---|
| `brightness_global` | 1.0 | master brightness gain | **2.43** |
| `speed_global` | 1.0 | all movement speeds | **1.46** |
| `pulse_global` | 1.0 | breathing/pulse rate | 0.66 |
| `follow_speed_global` | 1.0 | chase speed onto a person | 0.60 |
| `dwell_influence` | 1.0 | how strongly dwell-time bonuses apply | 0.49 |
| `idle_trend_weight` | 1.0 | how strongly passive-zone trends bend idle | 0.51 |

### 3c. How a meta-parameter becomes light (the modulation chain)

**◧ DRAW — a horizontal "signal chain" / pipeline of multiply stages:**

```
 MODE base value        ×  META-PARAM         ×  TIME-OF-DAY      ×  PROXIMITY     →  interp →  LIGHT
 (MODE_PARAMS)             (personality)         (TIME_CONFIGS)      (Z distance)
 e.g. pulse=2500ms      × energy 0.77 → 0.86  × evening ×1.3      × near ×—       →  ~2800ms
 e.g. brightness 8–30   × brightness_global   × ×0.6 late-night   × near ×1.4     →  final DMX
```

Key idea for the figure: **the light fixture only understands a handful of physical
quantities** — position, brightness min/max, pulse speed, falloff radius/shape, move
speed. The whole meta-parameter system exists to colour those few numbers.

---

## 4. Reaction loop — modes & gestures (timescale: instant)

Mode is chosen by `determine_mode()` ([light_behavior.py:996](light_behavior.py)) from
live active/passive counts, with **stickiness** (conditions must persist) and an
**8 s minimum dwell** preventing flicker.

**◧ DRAW — a state machine. Nodes = modes; edges = transitions labelled with the
trigger and the transition duration. Make engage-edges fast, disengage-edges slow.**

```
            ≥10 ppl/min passive
   IDLE ───────────────────────▶ AWARE
    │  ▲                           │  ▲
≥2/min│  │<2/min(10s)        person │  │
    ▼  │                    enters │  │
   FLOW ◀───────────────────────── │  │
    │      active-zone entry (0s)  │  │
    │            ┌─────────────────┘  │
    ▼            ▼                     │
  ENGAGED ◀──▶ CROWD   (2+ active)     │
    │  fast in (0.4s) / slow out (3s)  │
    └──────────────────────────────────
```

| Mode | Trigger | Character (move / bright / pulse / falloff) |
|---|---|---|
| **IDLE** | <2 ppl/min | gentle wander or *park*; 20 cm/s, dim, slow 4 s pulse, wide 90 cm |
| **FLOW** | ≥2 ppl/min sidewalk | drift with traffic; 25 cm/s, medium, 75 cm |
| **AWARE** | ≥10 ppl/min sidewalk | energetic, wide reach; 35 cm/s, brighter, fast 2.2 s pulse |
| **ENGAGED** | 1 in active zone | follow nearest; breathing + subtle gestures; tight 45 cm |
| **CROWD** | 2+ in active zone | follow centroid; brightest/fastest; may *bloom* all panels |

**Dwell phases** (deepen over time, for an "engagement timeline" figure):
`notice 0–3 s → greet 3–10 s → engage 10–30 s → bond 30 s+`, each unlocking warmer,
less frequent gestures. **Gesture library** = 16 one-shot/ongoing motions (nod, lean,
sway, orbit, settle, breathe, bloom, sweep, focus…).

---

## 5. Anticipation loop — short-term trends (timescale: seconds → hour)

Prose: when nobody is engaging, the light does not wander randomly. It continually
*forecasts* the next minute-to-hour from the database and pre-positions/energises
itself, so it is already "leaning toward" arriving traffic.

**◧ DRAW — nested time windows feeding three influence signals:**

```
 DB queries (background thread)            derived influences           affects
 ┌───────────────────────────┐
 │ Recent   1 min  ──┐        │           activity_anticipation ──▶ idle brightness, wander interval
 │ Short    5 min  ──┤        │  fold →   flow_momentum (−1..+1) ──▶ wander-box X shift
 │ Medium  30 min  ──┤        │           energy_level (0..1)    ──▶ idle pulse/speed
 │ Long    60 min  ──┘        │
 └───────────────────────────┘   weighted by meta-param `idle_trend_weight`
```

Plus two faster sub-systems:

- **Flow tracker** (`FlowState`, [light_behavior.py:170](light_behavior.py)) —
  dominant walk direction over a **30 s window**, updated every **1.5 s**, EMA α=0.25.
  Drives wander-box bias, triggers FLOW mode, and (V6.5c) biases wander **targets**
  toward incoming traffic (triangular distribution peaking ≤60 % toward arrivals).
- **Aggression** (`AggressionState`, [:112](light_behavior.py)) — a 0–1 "attention
  seeking" level that **rises when ignored, falls when engaged**, and is **capped by
  hour of day** (`AGGRESSION_TIME_CAPS`) to suit a financial-district site (near-zero
  at night, peak at lunch).

**◧ DRAW — aggression as a tank**: inflow = "ignored time + passers-by who don't stop",
outflow = "engagement", with a ceiling valve labelled "time-of-day cap (hourly curve)."

---

## 6. Self-tuning loop + the interaction budget (timescale: hours → days)

This is the conceptual heart and the best candidate for a feedback-loop diagram.

### 6a. Schedule (be precise in captions)

- **Meta-review retuning: 3×/day** — `autotune-meta-review.timer` at **06:00 / 14:00 /
  22:00**, 8-hour analysis window.
- **Daily report + daily learning: midnight (00:15) + 06:00 catch-up.**
- So **four scheduled DB analyses per day total**, but the *trend-driven personality
  retuning* is **3×**. (If your text says "four times a day," it's the count of all
  scheduled passes; the retuning specifically is three.)

### 6b. The friction: interaction budget

Prose: every adjustment the tuner makes **spends budget**; budget **refills slowly**;
when it runs low, changes are **throttled**. This is the friction that prevents the
light from chasing noisy, second-to-second signals — it can make a few meaningful moves,
then must "earn back" the right to keep changing.
(`SmartAutoTuner`, [V6Dev/smart_autotuner.py:607](V6Dev/smart_autotuner.py).)

```
budget (max 200)  ── spend = Σ|Δparam| × cost_scale(30) ──▶ depletes
       ▲                                                      │
       └──────── refill = max / restore_seconds(600) ─────────┘   (+bonus when engaged)
   if total_cost > budget → scale all Δ down (throttle)
```

### 6c. The full self-tuning feedback diagram

**◧ DRAW — closed loop. The clever bit: the budget is itself re-tuned by the review.**

```
                 live presence + short-term trends
                              │ PUSH
                              ▼
      ┌─────────────────────────────────────────────┐
      │   per-frame tuner (SmartAutoTuner, ~8s)      │
      │   gradient ascent on an "engagement score"   │
      └───────┬───────────────────────────┬─────────┘
   RESIST ↑   │ writes Δ                   │ logs every change
 budget /     ▼                            ▼
 reversion /  meta-parameters ───────────▶ [DB: behavior_adjustments]
 caps         (light personality)                   │
      ▲                                             │ reads 8h history
      │ rewrites home/floors/caps/budget            ▼
      │ (autotune_overrides.json, hot-reload)  ┌──────────────────────┐
      └────────────────────────────────────────┤ meta-review 3×/day   │
                                                │ diagnose: stuck /    │
                                                │ static / starved /   │
                                                │ budget too tight/    │
                                                │ loose → retune       │
                                                └──────────────────────┘
```

**Three forces to label in the figure:** **PUSH** (live signal wants change),
**RESIST** (budget + mean-reversion toward "home" + hard caps), and **META** (the
3×/day review adjusts how hard each side pushes). Balance of the three = "alive but not
frantic, 24/7."

Self-diagnoses the review can reach (`diagnose()`,
[autotune_meta_review.py:334](autotune_meta_review.py)): parameters floor/ceiling-stuck
>80 %, activity implausibly low (night/sensor fault), mode starvation (<1 % engaged →
raise floors), over-reaction (>30 % engaged → lower home), static parameters → raise
curiosity, and budget always-full/depleted/throttling → tighten or loosen.

### 6d. Daily learning (separate, gentler)

At midnight the controller computes a **7-day weighted average of engagement by
time-of-day** and **blends ~30 %** into the next day's starting personality
(`on_daily_report`, [V6Dev/v6_integration.py:407](V6Dev/v6_integration.py)) — so the
light gradually learns "what worked at 9am vs 9pm."

---

## 7. Continuous self-analysis (anti-repetition)

The light also studies *its own* output to avoid getting stale — the explicit design
goal of "evolution, not just reaction."

**◧ DRAW — a small "mirror" loop: light → records own state → reads it back → varies.**

| Metric | Source | Triggers |
|---|---|---|
| position entropy (1 h) | `get_position_entropy` ([tracking_database.py:846](tracking_database.py)) | low ⟶ bias to unexplored space |
| response similarity (24 h) | `get_response_similarity` ([:936](tracking_database.py)) | high ⟶ force more variety |
| mode distribution (24 h) | `get_mode_distribution` | balance check |
| position cooldown (30 s) | `is_position_recently_visited` | don't revisit a spot |

---

## 8. The database — memory of the installation

One **SQLite** file (`tracking_history.db`, WAL mode, batched commits), serving three
masters at once: **live behaviour**, **trend analysis**, and **public reports**. Data is
**tiered** — raw events live ~48 h, aggregates live forever — so the file stays fast for
years while keeping permanent history.

**◧ DRAW — a tiered/funnel diagram: raw (48h) → hourly (∞) → daily (∞), with side
tables for the light's own behaviour and its tuning audit.**

```
 INGEST (per OSC msg)         AGGREGATE (hourly/midnight)        KEPT FOREVER
 ┌───────────────────┐        ┌──────────────────────┐          ┌──────────────────┐
 │ tracking_events   │ ─48h─▶ │ hourly_stats         │ ───────▶ │ daily_stats_v2   │
 │ x,z,vel,zone,flow │        │ people,active/passive│          │ peak/quiet hour, │
 └───────────────────┘        │ speed,flow,blooms    │          │ flow balance     │
 ┌───────────────────┐        └──────────────────────┘          └──────────────────┘
 │ light_behavior    │ ─48h─▶ (feeds self-analysis §7)
 │ mode,pos,bright…  │
 └───────────────────┘        ┌──────────────────────┐          ┌──────────────────┐
 ┌───────────────────┐        │ autotune_daily_      │          │ meta_tuning_     │
 │ behavior_         │ ─48h─▶ │ learnings (∞)        │          │ reviews (∞)      │
 │ adjustments       │        │ optimal values, etc. │          │ full audit of    │
 │ +budget before/   │        └──────────────────────┘          │ every 3×/day run │
 │  after/cost       │                                           └──────────────────┘
 └───────────────────┘
```

| Table | Stores | Why | Keep |
|---|---|---|---|
| `tracking_events` | person x/z, velocity, **zone**, **flow_direction** | raw interaction record; source of all trend queries | 48 h |
| `light_behavior` | light's own mode/position/brightness/gesture | self-analysis (anti-repetition) | 48 h |
| `behavior_adjustments` | every tuning Δ + activity + **budget before/after/cost** | audit the tuner reads | 48 h |
| `person_sessions` | visit start/end/duration, zone, flow | dwell/conversion | 48 h |
| `hourly_stats` | per-hour rollup | permanent trend history | ∞ |
| `daily_stats_v2` | per-day rollup (peak/quiet, flow balance) | day-over-day trends | ∞ |
| `autotune_daily_learnings` | optimal values, param journeys, strategy | next-day blending | ∞ |
| `meta_tuning_reviews` | diagnosis, old→new config, recommendations | transparent/reversible tuning | ∞ |

Two fields are computed *on ingest* so trend queries stay cheap: **zone**
(active/passive/unknown) and **flow_direction** (L→R / R→L / stationary).

---

## 9. Real-time ingestion (the sensing front-end)

**◧ DRAW — camera → YOLO → floor projection → fusion → OSC, with the coordinate frame.**

```
2× Reolink RTSP ─▶ batched YOLO (CUDA, 25fps) ─▶ ArUco-calibrated
                                                 foot→floor projection (cm)
   ─▶ cross-camera fusion + EMA smoothing + velocity coasting
   ─▶ OSC/UDP :7000   /tracker/count <n>   /tracker/person/<id> <x> <z>
   ─▶ controller: zone-classify (active/passive), per-person velocity,
      enter/leave/move callbacks → behaviour + DB write
```

A single pedestrian step becomes, within ~40 ms: a light reaction, a DB row, and a
WebSocket frame. Light output leaves over **Art-Net/UDP :6454** as per-panel DMX from a
distance-falloff model.

**◧ DRAW — the spatial plan (this is a strong architectural figure):** panels along the
storefront; **active zone** (engaging) close in; **passive zone** (sidewalk traffic)
beyond it; two cameras angled inward; ArUco markers. Coordinates in cm, X along the
panels, Z out toward the street.

---

## 10. Web interface — two data planes

Prose: the public viewer separates "what the light is doing *right now*" from "what
happened *over time*." Live state streams over WebSocket; long-term trends are
pre-computed server-side and published as static JSON — the browser never touches the
database or does heavy maths.

**◧ DRAW — two parallel pipes from controller to browser:**

```
                       ┌────────────────────────── controller ──────────────────────────┐
 IMMEDIATE plane  ◀────│ WebSocketBroadcaster :8765, ~15 fps                              │
  (Three.js live)      │  → light pos, people, mode, gesture, dwell phase,               │
                       │    plain-language "behaviour_description", population, V6 state  │
                       └─────────────────────────────────────────────────────────────────┘
                       ┌─────────────────────────────────────────────────────────────────┐
 TRENDS plane     ◀────│ generate_reports.py (nightly) → reports/daily/*.json + _index    │
  (charts page)        │  → deploy to GitHub Pages → reports.js fetches + charts          │
                       └─────────────────────────────────────────────────────────────────┘
  (transport: Tailscale Funnel exposes :8765 as wss:// for HTTPS GitHub Pages)
```

| Plane | Transport | Cadence | Content | Computed where |
|---|---|---|---|---|
| Immediate | WebSocket :8765 | ~15 fps | live light/people/mode/status | controller, sent as-is |
| Trends | static JSON over HTTPS | nightly | hourly curves, peak/quiet, flow balance, mode mix, tuning summary | SQLite (server-side) |

---

## 11. Suggested figure set (minimal, high-impact)

For a 10-image Projects submission, this sequence tells the whole story:

1. **Spatial plan** — cameras, zones, panels, light (§9).
2. **Top-level pipeline** — the three processes + two channels (§1).
3. **Three nested loops** — the conceptual core (§2).
4. **Mode state machine** — reaction layer (§4).
5. **Personality radar** — neutral vs as-deployed meta-parameters (§3a).
6. **Modulation chain** — mode × meta × time × proximity → light (§3c).
7. **Trend windows → influences** — anticipation layer (§5).
8. **Self-tuning feedback w/ budget** — PUSH vs RESIST vs META (§6c).
9. **Tiered database funnel** — raw→hourly→daily + audit (§8).
10. **Two web planes** — live vs trends (§10).

---

## 12. How this relates to the ACADIA 2026 brief — *Humanism Recoded* (Projects)

> **Verified call details** (ACADIA 2026, [call for submissions](https://2026.acadia.org/call-for-papers)):
> theme **"Humanism Recoded: Reframing Computation and Making through Embodiment and
> Culture"**; **Detroit / Lawrence Technological University, Oct 22–24, 2026**.
> **Projects** = *"600-word text (excluding citations and captions) plus a maximum of 10
> images,"* blind peer-reviewed, published in the proceedings (CumInCAD, DOI) and
> *"exhibited as posters in the Exhibition."* Projects should present *"built work,
> speculative prototypes, installations, or experimental workflows demonstrating the
> integration of computational tools with material, cultural, or social contexts."*
> Projects deadline: **June 1, 2026** (final extension). Of the 12 subthemes, the two
> most relevant here are **"Embodied Codes"** (*"computation situated in bodies,
> practices, and local contexts"*) and **"Machines that Care"** (*"robotics and
> autonomous systems reimagined as partners in repair, care, and cultural making"*).

ACADIA 2026, **"Humanism Recoded: Reframing Computation and Making through Embodiment
and Culture,"** reframes computation around human values, bodies and cultural practice
rather than abstracting it away from them — and its **Projects** category invites
*"built work, speculative prototypes, installations, or experimental workflows
demonstrating the integration of computational tools with material, cultural, or social
contexts."* Drop Ceiling answers that directly: it is a *built, public, 24/7* architectural light
installation whose computation is literally *embodied* in a storefront and *encultured*
by its site. The system does not run a fixed program; it senses the embodied rhythms of
a specific Toronto financial-district sidewalk — commute surges, lunch crowds, the dead
of night — and **recodes its own behaviour around them**, learning over days which
character "works" at which hour. The interaction budget is the conceptual hinge for the
brief: rather than maximising responsiveness, the project deliberately introduces
*friction*, treating restraint and slowness as humane design values and resisting the
extractive logic of an always-optimising machine. In this sense the light behaves less
like a sensor-actuator and more like an inhabitant that adapts to its neighbours.

Within the Projects category specifically, the contribution is an **experimental
authoring workflow for adaptive environments**: a transparent, three-loop architecture
(instant reaction, hourly anticipation, multi-day self-tuning) in which a small set of
human-legible "personality" meta-parameters — responsiveness, energy, sociability,
exploration, memory — are continuously, accountably retuned from the installation's own
interaction history, with every decision logged and even re-published to the public as a
daily report. This speaks to the **"Machines that Care"** and **"Embodied Codes"**
subthemes: an autonomous system reframed not as a controller to be optimised but as a
cultural participant that watches, remembers, hesitates, and grows a character in
dialogue with the people who pass beneath it. The project's value to the ACADIA audience
is methodological as much as aesthetic — it demonstrates how trend analysis, budgeted
self-modification, and self-analysis can be composed into an adaptive architecture that
remains interpretable and authorable by a designer rather than opaque.

---

*Sources for the conference framing:
[ACADIA 2026 Call for Submissions](https://2026.acadia.org/call-for-papers) and the
[EasyChair CFP](https://easychair.org/cfp/acadia2026). All system facts trace to the
V6.5c source under `dc-dev/IO/`.*

---

## 13. Draft 600-word Projects submission text

*Ready-to-submit draft body — **543 words** (the call allows 600, excluding citations
and captions, leaving ~57 words of headroom for a site-specific detail or a closing
line). Everything above is reference/diagram material; this is the actual proposed text.
Slot project title, authors and image captions around it.*

> Drop Ceiling is a permanent, public light installation in a downtown Toronto
> storefront: a single luminous point drifts through a field of LED panels behind the
> glass while, on the sidewalk outside, two cameras quietly watch the people passing
> by. It is not a motion sensor that flashes on contact. It is a system designed to
> behave like an inhabitant of its corner — one that reacts in the moment, anticipates
> the rhythms of the street, and slowly grows a character of its own over weeks of
> operation.
>
> Computationally, the work is organised as three nested control loops running at three
> timescales. The innermost loop, at thirty frames per second, drives reaction: a
> behaviour state machine (idle, flow, aware, engaged, crowd) follows whoever is
> present, layering sixteen small gestures — a nod, a lean, an orbit, a settle, a slow
> shared "breathing" — that deepen the longer a person stays. The middle loop, at
> seconds-to-minutes, drives anticipation: the system reads pedestrian trends across
> nested one-, five-, thirty- and sixty-minute windows, plus a thirty-second flow
> tracker, and biases where and how energetically the light idles so that it leans
> toward arriving foot traffic rather than waiting to be triggered. The outermost loop,
> running across hours and days, drives self-tuning.
>
> That self-tuning is the conceptual core. The light's expression is governed by a small
> set of human-legible meta-parameters — responsiveness, energy, sociability,
> exploration, memory, and a handful of output multipliers — that colour how each
> behaviour mode becomes movement, brightness, and pulse. Three times a day the system
> performs a deep analysis of its own interaction database, diagnoses pathologies in its
> recent behaviour (parameters stuck at limits, modes starved, responses gone static or
> repetitive), and rewrites its own tuning configuration, which the running light
> hot-reloads within seconds. Every decision is logged; a nightly report republishes the
> day's patterns to a public web view.
>
> Crucially, the system does not simply maximise responsiveness. An interaction budget
> meters how much the tuner may change per unit of time: every adjustment spends budget,
> budget refills slowly, and when it runs low further change is throttled. This
> deliberate friction — reinforced by mean-reversion toward "home" values and the daily
> review's ability to retune the budget itself — keeps adaptation slow and intentional
> rather than twitchy. The result reads less like an optimising machine and more like a
> temperament: the light can be eager or reserved, brisk or contemplative, and it earns
> the right to change.
>
> Drop Ceiling addresses Humanism Recoded by situating computation in a body and a
> place. Its intelligence is not abstracted onto a screen but embodied in a storefront
> and encultured by a specific site, whose commute surges, lunch crowds, and
> dead-of-night quiet it learns and answers. The project speaks directly to the Embodied
> Codes and Machines that Care subthemes: an autonomous system reframed not as a
> controller to be optimised but as a cultural participant that watches, remembers,
> hesitates, and grows a character in dialogue with the people beneath it. Its
> contribution to the Projects track is as much methodological as aesthetic — an
> authoring approach for adaptive environments in which a designer shapes a few
> interpretable personality parameters and lets the installation retune them,
> accountably and reversibly, from its own lived experience of the street.
