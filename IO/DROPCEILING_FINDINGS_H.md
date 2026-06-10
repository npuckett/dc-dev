# DropCeiling — A Second Reading of the Run Data

*A fresh analysis (Jun 2026) that builds on
[`DROPCEILING_FINDINGS.md`](DROPCEILING_FINDINGS.md) but reads the data the
first pass did not: the per-cycle tuner behaviour (217K rows), the one
surviving meta-review (verbatim), the per-day brightness curve, the
mode-by-day entropy timeline, the spatial footprint of each mode, the gesture
spatial map, the parameter correlation structure, and the diary's sister
metric, the empirical regime-conditional tuner deltas.*

*Each finding here either **extends** a G-series claim with a new angle, or
**replaces** a guess with a measurement. The figures are H1–H10
(`IO/diagrams/H*.svg`); the source data is `IO/analysis/h_data.json`; the
extraction script is `IO/analysis/h_data_prep.py`; the renderer is
`IO/diagrams/src/H_findings_figures.py`.*

*House style: greyscale + one warm accent reserved for the light, same as the
E- and G-series.*

---

## A. What the tuner actually does

The G-series established that the personality was punctuated (G1), that the
friction was the empirical motivation for V6 (G9), and that the first
self-diagnosis survived (Finding 2). What the G-series did *not* do is
**look inside the tuner cycles themselves** — at the 217,285 rows in
`behavior_adjustments`, each carrying a snapshot of all 12 parameters, the
proposed delta, the aggression level, the energy level, and the short/medium/long
activity. That is what the H-series does.

### Finding H1 — The tuner's behaviour depends on regime, and the signs are the story ⚠️ new

Every tuner cycle (`n = 217,285`, Feb 11 – Mar 2) is now split into four
**activity regimes** by the cycle's `short_activity` and `medium_activity`:

| Regime | Criterion | n cycles |
|---|---|---:|
| **dead** | `short < 0.02` and `medium < 0.05` | 45,055 |
| **trickle** | `short < 0.10` | 94,543 |
| **steady** | `short < 0.30` | 66,429 |
| **rush** | `short ≥ 0.30` | 11,257 |

For each regime, we computed the mean per-cycle Δparam (the actual signed
change the tuner applied to each of the 12 parameters in that cycle). The
heatmap (H1) shows the signs:

- **DEAD** is essentially zero everywhere. With no one around, the tuner has
  no signal to follow and just drifts around the home values (mean Δ < 0.0001
  for every parameter).
- **TRICKLE** reverses most directions toward home. Output parameters (brightness,
  speed, pulse) drift down; personality parameters (responsiveness, energy,
  sociability) drift down too — the mean reversion does its work.
- **STEADY** is mixed: `dwell_influence` and `follow_speed` go *up*; the
  outputs and idle_trend_weight go *down*.
- **RUSH** is the loudest signal. `responsiveness` +0.0001/cycle, `energy` +0.0001,
  `sociability` +0.0001, `follow_speed` +0.0002; `brightness` −0.0002,
  `speed` −0.0002, `pulse` −0.0002, `idle_trend_weight` −0.0003. **The
  personality becomes more social and attentive, and stops wandering idle,
  precisely when the street is busiest.** The deltas are small per cycle
  (≈0.0003), but at ~14,000 cycles/day they accumulate to ~3 units of
  net_change per day.

- **Show it:** **H1** — the 4×12 heatmap with the regime as row, parameter as
  column, the colour the signed mean Δ. The bottom row (RUSH) is the only
  one with strong colour.
- **Say it:** *the tuner is regime-conditional in its empirical behaviour,
  not just in its design. The same code, against the same friction, pushes
  different directions depending on what's happening on the sidewalk.*

### Finding H2 — The meta-review's diagnosis, every parameter against a wall ⚠️ new

The single surviving `meta_tuning_reviews` row (Feb 13 10:23) contains the
raw `pct_at_floor_json` and `pct_at_ceiling_json` for all 12 parameters
over its 8-hour window. H2 plots these as diverging bars:

| Parameter | % at floor | % at ceiling |
|---|---:|---:|
| memory | **100.0%** | — |
| responsiveness | 96.6% | — |
| energy | 96.6% | — |
| sociability | 96.6% | — |
| follow_speed_global | 96.6% | — |
| dwell_influence | 68.2% | — |
| brightness_global | — | 96.6% |
| speed_global | — | 96.6% |
| pulse_global | — | 96.6% |

**Eight of the twelve parameters were at a limit for over 96.6% of the
8-hour window. Memory was floor-clamped 100% of the time.** And the mode
distribution for that window: **97.6% idle, 0.7% engaged, 1.3% flow** —
the system was almost never with anyone. The diagnosis the meta-review
issued ("Floor-clamped params: memory(100%), responsiveness(96.6%)…
Ceiling-clamped params: brightness_global(96.6%)… Activity rarely exceeds
0.1 (p90=0.091) — target may be too high") is exactly right, and the
17-line `changes_summary` it produced (raise 4 homes, drop 3 ceilings, raise
5 floors, tighten the budget) is what un-pinned everything in the days that
followed (visible in G1 from Feb 14 onward).

- **Show it:** **H2** — the 9-row diverging bar chart. Warm = floor, grey =
  ceiling. The single row that stands out (dwell_influence, 68%) is the only
  parameter with any room to move.
- **Say it:** *on its second morning, the system ran a self-diagnosis and
  found every part of its personality pinned against a wall. The fixes it
  prescribed are still in `autotune_overrides.json`.*

### Finding H3 — Per-cycle aggression as a 217K × 24-hour heatmap ⚠️ new

G4 (the G-series) showed the *mean* aggression by hour, smoothed to 24 data
points. H3 shows the *actual* per-cycle aggression as a day × hour heatmap
(20 days × 24 hours = 480 cells, all from real data). The 4 AM peak is
visible as a dark band on the left; the 12-16h valley as a light band in the
middle; the V5 step (Feb 25) is visible as a subtle cooling across the
whole row. The 02-19 03:00 cell is bright white because that hour had a
single 03:00 cycle with 0 aggression — the EMA is reset to 0 when there's no
input.

This figure is the same data as G4, but it preserves the day-by-day
resolution. It is the *time-of-day aggression* signature, un-averaged.

- **Show it:** **H3** — the orange heatmap with V5 (Feb 25) marked as a
  dashed line and the 4 AM band visible.
- **Say it:** *aggression is a clock-driven function, not a property of the
  street. The dark band at 04:00 and the light band at 14:00 are the same on
  every day of the run.*

### Finding H4 — The 12-dial personality is a 3-factor system ⚠️ new

H9 in the H-series extracts the 33-day `param_journeys` from
`autotune_daily_learnings` (the daily net_change per parameter) and computes
the **inter-parameter correlation matrix**. Three clusters emerge cleanly:

| Cluster | Parameters | r (within) |
|---|---|---:|
| **Outputs** | brightness_global, speed_global, pulse_global | 0.99+ |
| **Personality** | responsiveness, energy, sociability, follow_speed_global, attention_span | 0.93 – 0.99 |
| **Trade-offs** | dwell_influence ↔ exploration, exploration ↔ memory, dwell_influence ↔ memory | −0.91 to −0.94 |

The H4 figure is a heatmap of the daily net_change, with the three clusters
boxed. The boxed structure is the empirical fact: the 12 named parameters
are not 12 independent dials. They collapse to **two positive-correlated
factors (output energy + social energy) and one trade-off axis
(dwell-attention vs exploration)**. This has consequences for design — if
you want to slow the light down, you can't just lower `speed_global`; you
have to also expect `brightness_global` and `pulse_global` to come down
with it.

- **Show it:** **H4** — the per-day net_change heatmap with the three
  clusters boxed.
- **Say it:** *the system has 12 names for 3 factors. Naming 12 dials is a
  design choice; using 3 is a finding.*

### Finding H5 — Behavioural richness tripled when the vocabulary did ⚠️ new

The 47 daily-report JSONs in `IO/reports/daily/` each contain
`light_behavior.mode_distribution` — the fraction of the day the light spent
in each of {idle, flow, aware, engaged, crowd}. The H-series computes the
**Shannon entropy** of this distribution (normalised by `log(5) = 1.61`, so
a day with all-idle = 0 and a day with 5 equal modes = 1.0).

The timeline (H5) shows three regimes:

| Period | Hn (mean) | Modes present |
|---|---:|---|
| Feb 13–16 | ~0.0 | 1 mode (idle) |
| Feb 17–22 | ~0.0–0.2 | 1–3 modes (mostly idle, occasionally flow) |
| Feb 23 – Mar 2 | ~0.5 | 2–4 modes (idle, flow, engaged) |
| **Mar 3 onward** | **~0.8** | **4–5 modes (all of them, including aware)** |

The transition is **discontinuous** and the date is **Mar 3** — the day the
V6 software deployed with the `aware` mode in the code. From that day, the
entropy jumps from 0.5 to 0.8 and stays there. The 12-day dip on Mar 15–16
(low Hn) is when the system logged very few ticks (the V6.5 build was being
smoke-tested). The 5-mode vocabulary expanded and the system's
behavioural diversity expanded with it.

- **Show it:** **H5** — the orange-tinted area chart, with V6 (Mar 3) and
  "AWARE in data" (Mar 15) marked as dashed verticals.
- **Say it:** *the system went from one way of waiting to five. The jump
  isn't a learning event; it's a vocabulary event.*

---

## B. What the light actually did

### Finding H6 — The 758 engagement episodes, by hour and date ⚠️ new

G3 in the G-series showed the overall episode-duration distribution. H6
shows the same 758 episodes, but plotted as a 3-day × 24-hour grid
(Mar 15, 16, 17 — the only dates with light_behavior data, since the V6
tuner was active). The result:

- Most episodes start in the **afternoon band (12:00–17:00)**.
- The single hottest cell is **Mar 16 17:00 (56 episodes)**.
- The 03-15 and 03-17 days are quieter (16 and 36 episodes respectively);
  Mar 16 was the busiest by far.
- Most nights (00:00–05:00) have 0–5 episodes.
- The 12 bond episodes (durations > 30 s) are scattered, but the longer ones
  concentrate in the afternoon band on Mar 16 (Tue, the busiest day).

This figure is the **time-of-day and time-of-week signature of the system's
most interactive moments**, distinct from the G3 distribution (which is just
durations). The placement of the bonds matters: they didn't happen at
random; they happened on the busiest afternoon.

- **Show it:** **H6** — the orange heatmap of episode counts by hour.
- **Say it:** *twelve people over three days, and most of them showed up on
  Tuesday afternoon.*

### Finding H7 — The spatial footprint of each mode ⚠️ new

H8 in the H-series extracts the (x, z) position of the light for every tick
of `light_behavior` in the V6.5 window (82,529 rows), grouped by mode, and
plots the resulting density per mode. The centroids:

| Mode | μx (cm) | μz (cm) | n | What it means |
|---|---:|---:|---:|---|
| **idle** | −160 | −2 | 40,375 | panel centre, at the panel face |
| **flow** | −158 | −1 | 28,769 | same as idle (close) |
| **aware** | −153 | 0 | 5,504 | slight forward, slight off-centre |
| **engaged** | −133 | **+9** | 6,964 | forward 9 cm (toward the street) |
| **crowd** | **−188** | **+14** | 917 | far left, 14 cm forward |

The light's spatial behaviour has a clear **mode-conditioned geometry**:
engaged pulls forward and outward, crowd pulls further forward and *off* to
one side. This is the *spatial logic of the system* — the panels represent
attention by the light physically moving toward the people. The panels are
fixed; the light isn't. The 9 cm of forward travel during engagement is
small in absolute terms but is the most consistent spatial shift in the
data.

- **Show it:** **H7** — five density contours with the centroid marked for
  each mode. The crowd centroid is the most off-axis.
- **Say it:** *engagement is a posture. The light steps toward the people
  when it's with them.*

### Finding H8 — Gestures are not all alike: some are postures, some are motions ⚠️ new

H3 in the H-series extracted, for each gesture, the mean (x, z) of the
*target* position the light moved to during that gesture. The result, in
the V6.5 window:

| Gesture | μz (cm) | Type |
|---|---:|---|
| sway | **+20** | motion (outward) |
| orbit | **+22** | motion (outward) |
| sweep | **+20** | motion (outward) |
| bloom | −1 | posture (panel face) |
| welcome | −1 | posture (panel face) |
| playful | −1 | posture (panel face) |
| thinking | −2 | posture (panel face) |
| curious | −1 | posture (panel face) |
| bored | −2 | posture (panel face) |

**Three of the nine gestures (sway, orbit, sweep) move the light's body
outward by ~20 cm. The other six are postures performed at the panel
face.** The V5-era gestures and the V6-era gestures are not the same kind
of thing: `sway`, `orbit`, `sweep` are *kinetic* (they animate a motion),
`welcome`, `thinking`, `curious` are *static body attitudes*. The V6
gestures (REACH, EMBRACE, BEACON, TWIRL) are *falloff-shape* modulations,
not body motions at all.

- **Show it:** **H8** — the spatial map of 8 gestures, with the 3 motion
  gestures (sway, orbit, sweep) at z ≈ +20, the 5 posture gestures at
  z ≈ 0. Discs show the std of target_x as a rough amplitude.
- **Say it:** *the gesture vocabulary is a vocabulary of two kinds of
  motion. Some gestures are body movements; others are body attitudes.*

---

## C. What the street did, what the light was

### Finding H9 — The light took a week to come on ⚠️ new

H7 in the H-series is the **48-day daily-mean brightness curve**, taken
from the `hourly_stats_filled.avg_brightness` column (the only brightness
column that covers the full run, since the merged DB's `light_behavior` is
sparse).

The first 5 days (Jan 29 – Feb 2): **0 DMX**. The panels were literally off.
The next 7 days (Feb 3 – 9): **no data in the DB** (the prune/recovery gap
covered by H-series-era work; the daily reports recovered it). The next 4
days (Feb 10 – 13): **0–60 DMX** — the V2 → V4 transition, with the system
tracing its first behaviour. The next 12 days (Feb 14 – 25): **15–35 DMX**
— the V4/V5 equilibrium, dim and stable. **Mar 2**: a single bright spike
to ~390 DMX (the day the V6 was being smoke-tested). Mar 3 onward: **80–100
DMX** with a small dip on Mar 16-17.

This is the *the light was off* finding, distinct from G9 (which is the
budget timeline). The H9 figure has the same shape as the "software came
on" story: **the system didn't really start until the V4 deployment on
Feb 12**, and it didn't reach stable brightness until V5 (Feb 25) or V6
(Mar 3).

- **Show it:** **H9** — the orange-filled daily-mean curve, with the major
  deployment dates marked.
- **Say it:** *the first 5 days of the run were dark. The light came on
  around Feb 12, settled into a dim 20 DMX in the V4 era, and didn't
  reach its V6.5 brightness until the V6 deployment on Mar 3.*

### Finding H10 — The 12 step-size distributions, by regime ⚠️ new

H10 in the H-series is the **per-parameter step-size distribution,
conditioned on regime**. The 8 subplots show, for each of 8 selected
parameters, the *standard deviation* of the per-cycle Δparam (a bar, the
spread) and the *mean* (a marker, the direction). Reading across:

- All 8 parameters show their **largest spread in the RUSH regime** (the
  warm bars are the tallest). The tuner has the most room to move when
  activity is high.
- **follow_speed_global** has the most extreme DEAD-regime spread — even
  when no one is around, the tuner wiggles it around 0.014. This is
  consistent with the curiosity-driven random nudge.
- **dwell_influence** has the cleanest regime separation: std grows from
  0.005 in dead to 0.007 in rush.
- The means (markers) are tiny in absolute terms (~0.0001/cycle) but their
  *signs* are the same as the H1 heatmap.

- **Show it:** **H10** — 8 small multiples, each a 4-bar chart of regime
  std with a mean marker.
- **Say it:** *the tuner has more elbow room in busy regimes. The means
  are small but consistent.*

---

## D. New publication angles

| Angle | H-figures | Venue fit |
|---|---|---|
| **Tuner behaviour is regime-conditional** — the empirical case that the friction isn't just static, it's response-conditional | H1, H2, H10 | systems / design methods; pairs with G9 (which made the V4/V5 budget argument) |
| **The 12-dial personality is 3 factors** — naming is design, structure is data | H4 | a clean quantitative finding, strong visual; design methods / HCI |
| **The 5-mode vocabulary expanded and the entropy jumped** — a single timeline figure for the "vocabulary → diversity" claim | H5 | a single high-impact figure; works in TEI pictorial or any systems talk |
| **The spatial logic of the modes** — engagement is a posture, the light steps toward the people | H7 | an embodied-finding visual; perfect for the "spatial signature" angle in TEI |
| **The light took a week to come on** — the deployment arc as a 48-day curve | H9 | the operational story; pairs with F1 (the software/data timeline) |
| **Gestures are of two kinds** — motion vs posture, distinguished by their target_z | H8 | a precise linguistic/embodied distinction; HCI / TEI |

## E. New figures worth building (in house style)

| ID | Figure | From | Strength |
|---|---|---|---|
| **H1** | Regime-conditional param deltas (4 regimes × 12 params heatmap) | Finding H1 | the empirical reading of the tuner's regime logic |
| **H2** | The first self-diagnosis — every parameter at a wall | Finding H2 | the meta-review's verdict in numbers |
| **H3** | Per-cycle aggression × day × hour | Finding H3 | the same data as G4, un-averaged |
| **H4** | 12-param daily net-change heatmap, 3 factor clusters boxed | Finding H4 | the 3-factor structure of the personality |
| **H5** | 48-day mode entropy, with the V6 step visible | Finding H5 | the vocabulary → diversity timeline |
| **H6** | Engagement episodes by hour × date (3-day grid) | Finding H6 | the temporal signature of the 758 episodes |
| **H7** | Spatial footprint of each mode (V6.5) | Finding H7 | the geometric logic of the modes |
| **H8** | Where each gesture goes (V6.5) | Finding H8 | motion vs posture, distinguished |
| **H9** | 48-day brightness curve, with deployment dates | Finding H9 | the light was dark for a week |
| **H10** | Per-param step-size distribution, by regime | Finding H10 | the spread, not just the mean |

*All source data is in `IO/analysis/h_data.json` (regenerated by
`IO/analysis/h_data_prep.py` from the merged DB and the two source DBs). All
figures render from `IO/diagrams/src/H_findings_figures.py` — re-run with
`../../.venv/bin/python src/H_findings_figures.py` from the `diagrams/`
directory.*

## F. What was *not* redone

The H-series deliberately *avoids* the figures the G-series already made
well. G1, G2, G3, G4, G5, G6, G7, G8, G9, G10 remain the canonical versions
of their findings; the H-series offers 10 different angles. If a venue
needs only one or two figures, prefer the G-series for the public-facing
emotional claims (loneliest at 4 AM, the breathing street, the twelve
bonds) and the H-series for the technical/architectural claims (3-factor
personality, regime-conditional tuning, 5-mode entropy, the spatial logic
of modes).

## G. Method notes / caveats

1. **H1/H10 use the per-cycle `old_values`/`new_values` from
   `behavior_adjustments.adjustments_json`.** That JSON column was populated
   by the V5 auto-tuner (`lightController_osc.py:2096-2115`) and is
   present in all 217,285 rows. It is not in the `daily_stats_v2` table.
2. **H2 uses the one surviving `meta_tuning_reviews` row** (Feb 13 10:23
   from `IO/tracking_history.db`). The other 161 expected rows are empty in
   both source DBs (Finding D-2 caveat carried over from the G-series).
3. **H5 uses the daily-report `light_behavior.mode_distribution`** which is
   available for 33 of the 47 report days; the other 14 (mostly the
   pre-Feb-13 days and the calibration-bug days) have `unknown: 1.0` and
   are filtered out.
4. **H6 reconstructs 758 engagement episodes** from the V6.5
   `light_behavior` ticks (Mar 15–17) using the same method as the G3
   figure. The match in counts is a sanity check.
5. **H7/H8 use the V6.5 window only** (Mar 15–17, ~82K ticks). The V5
   window (Feb 23–25, ~89K ticks) is comparable in size and would let us
   compare the spatial signature before/after the V6.5 redesign. We did
   not do this here; the G-series already covered the V5/V6.5
   gesture-comparison angle.
6. **H9 is the only H-figure that uses the `hourly_stats_filled` table
   directly** (the merged DB's continuous 48-day table). Every other
   H-figure uses the source DBs (autotune_daily_learnings,
   behavior_adjustments, light_behavior, meta_tuning_reviews) or the daily
   reports.
