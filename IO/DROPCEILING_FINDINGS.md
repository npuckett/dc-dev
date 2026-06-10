# DropCeiling — Fresh Findings from the Run Data

*An independent second analysis of the full merged database (Jun 2026), going past the
first audit's coverage/quality work into the tables it never opened: the tuner's 217K-row
decision log, the light's 172K-row life-log, the 33-day daily-learning record, and the raw
position stream. Each finding states what the data shows, why it matters for publication,
how to show it, and how to say it. Companion documents:
[`DROPCEILING_STORY.md`](DROPCEILING_STORY.md) (the narrative playbook) and
[`analysis/DATA_TIMELINE_AND_MERGE.md`](analysis/DATA_TIMELINE_AND_MERGE.md) (data
provenance). Some findings below **revise** the earlier story — noted where they do.*

---

## A. How the system actually changed

### Finding 1 — Personality change was punctuated, not gradual ⚠️ revises the drift story

The never-before-opened `autotune_daily_learnings` table holds **33 daily snapshots of all
twelve meta-parameters** (Feb 12 – Mar 2, then zeros). The real trajectory:

| Phase | Dates | What the numbers do |
|---|---|---|
| **Day-one chaos** | Feb 12 | parameters pinned at their limits: exploration 1.00, attention 1.00, memory 0.00, brightness 3.05×, speed 1.63× |
| **Taming** | Feb 13–14 | everything converges within 48 h of the first self-review |
| **The V4/V5 equilibrium** | Feb 14–24 | a stable character held for 11 days: responsiveness ~0.53, energy ~0.48, brightness ~0.94× |
| **Version step** | Feb 25 (V5 update) | a discrete shift to a calmer character: energy 0.48 → 0.38, responsiveness 0.53 → 0.45 |
| **Record ends** | Mar 3 (V6) | the new tuner stopped writing this table — zeros from here |

The earlier radar (E3) compared config endpoints (home 0.53 → deployed 0.83) and implied a
smooth drift. The daily record shows the truth is more interesting: **equilibria
punctuated by software versions**, with the final V6.5 character (0.83/0.77) reached in an
era this table no longer recorded. Personality change and version change are entangled —
which is the honest claim, and a better one.

- **Show it:** a 33-day multi-line chart of the six personality parameters, with version
  dates as vertical rules — the steps will be visible at Feb 13, Feb 25, Mar 2. (Proposed
  figure **G1**; the data is extracted and ready.)
- **Say it:** *the personality did not drift; it settled, was re-settled by each version,
  and settled again.*

### Finding 2 — The system's first self-diagnosis survives, verbatim

The single surviving `meta_tuning_reviews` row is the **first time the system read its own
history** (Feb 13, 10:23). Its diagnosis: *"memory floor-clamped 100%, responsiveness
96.6%, energy 96.6%… brightness_global ceiling-clamped 96.6%… Activity rarely exceeds
0.1 — target may be too high."* Its response: raise the homes, drop brightness home from
1.2 to 0.93. The next day's data shows the equilibrium forming. Cause and effect, on the
record.

- **Show it:** quote the diagnosis as a typeset artifact (a "diary entry") beside the
  Feb 12→14 trajectory from Finding 1.
- **Say it:** *on its second morning the system noticed every part of its personality was
  pinned against a wall, and moved the walls.*

### Finding 3 — The friction had two opposite regimes ⚠️ refines the budget story

Parsing the budget telemetry inside `behavior_adjustments` (sampled across 217K rows):

| Era | Avg budget (of 200) | % of cycles depleted |
|---|---|---|
| Feb 13–23 (V4) | **~1.3** | **84–88%** |
| Feb 25 – Mar 2 (V5) | **~200** | **0%** |

For eleven days the tuner lived in budget poverty — throttled almost every cycle. Then the
V5 update reduced the tuner's appetite below the refill rate and the budget never bound
again. A fixed friction constant was either suffocating or absent, depending on the
controller around it. This is the *empirical motivation* for V6's design choice of letting
the meta-review retune the budget itself — previously asserted, now demonstrated.

- **Show it:** budget level over time (two flat regimes with a cliff at Feb 24–25), or a
  before/after pair of "% cycles throttled."
- **Say it:** *the same friction setting starved one version of the software and never
  touched the next; friction has to be tuned by something that can watch it work.*

---

## B. How the light actually lived

### Finding 4 — A life of waiting, restructured by V6.5

From the light's own state log (172K ticks across two windows):

| Mode | V5 era (Feb 23–25) | V6.5 era (Mar 15–17) |
|---|---|---|
| idle | **69.7%** | **48.6%** |
| flow | 21.3% | 35.1% |
| aware | — | 6.7% |
| engaged | 8.9% | 8.5% |
| crowd | 0.2% | 1.1% |

Engagement stayed almost exactly constant (~8.5–9%). What V6.5 changed is **how the light
waits**: twenty points of "nothing" became flow/aware — attending to passing traffic
instead of wandering alone. The social fraction of a public installation's life may be
nearly fixed by the site; the design question is the texture of the other 91%.

- **Show it:** two stacked time-budget bars, V5 vs V6.5 (the idle→flow/aware shift is the
  whole figure).
- **Say it:** *the redesign didn't make more people stop; it changed what the light did
  while it waited.*

### Finding 5 — The gesture economy inverted: from solitary to social

Top gestures by share of all gesture ticks:

| | V5 era | V6.5 era |
|---|---|---|
| #1 | thinking (31%) | **welcome (24%)** |
| #2 | curious (27%) | thinking (16%) |
| #3 | bored (21%) | curious (12%) |
| welcome's share | 6.3% | **24.3%** |

V5's light mostly performed *solitary* gestures — thinking, curious, bored. By V6.5 its
most frequent gesture was **welcome**, with playful and sweep newly in the repertoire.
(Total gesture volume also dropped ~3×, by design — the V6.5c park-state made stillness a
choice.) Caveat to carry: the available vocabulary also grew between eras, so this is
behaviour *and* repertoire change together.

- **Show it:** two ranked gesture bars (V5 / V6.5) with welcome highlighted in the warm
  accent.
- **Say it:** *it stopped talking to itself and started greeting people.*

### Finding 6 — The brightness ladder: engagement is legible in light

Average brightness by mode (V6.5 era) climbs monotonically: **idle 30 → flow 92 → aware
158 → engaged 221 → crowd 388** (cap 600). Anyone on the sidewalk could, in principle,
read the social state of the alcove from the window's glow. The output really is the
12-value display the C-series describes, and it encodes one variable above all: how much
attention is present.

- **Show it:** a five-step ladder/staircase chart, warm accent rising.
- **Say it:** *the panels are a single instrument that displays attention.*

### Finding 7 — Engagement is fleeting; the bond was rare and real

Reconstructing engagement episodes from `active_count` runs (V6.5 window): **758
episodes**; median ≈ a walk-through; only **10.7% lasted past the 3-second "notice"
phase**; **12 episodes (1.6%) reached the 30-second "bond" phase**. The longest: **193
seconds**. The four-act dwell structure (notice → greet → engage → bond) was built for an
audience of twelve — and that is exactly the right framing, not a failure: the
architecture of deepening existed so that the rare person who stayed got somewhere to go.

- **Show it:** an episode-duration distribution with the dwell-phase boundaries marked,
  the 12 bond episodes individually visible as dots.
- **Say it:** *twelve people, over three days, stayed long enough to reach the final act —
  the system kept a structure ready for each of them.*

### Finding 8 — The light is loneliest at 4 AM

The aggression (attention-seeking) signal, averaged by hour across 217K tuning cycles, is
a clean **mirror image of the traffic curve**: highest at 03:00–05:00 (~0.16) when the
street is empty, lowest through the working afternoon (~0.03) when it is full. The
mechanism behaves exactly as designed — and reads as an emotional portrait: the urge to
attract peaks precisely when no one is there to see it, held down at night only by the
time-of-day caps.

- **Show it:** aggression-by-hour and people-by-hour on one axis pair — two curves in
  opposite phase.
- **Say it:** *the light wants attention most at four in the morning, when there is no one
  to give it.*

---

## C. What the street did

### Finding 9 — The street breathes: a commute tide the light surfed

Flow directionality by hour (10.4M directional events, measured days): mornings run
strongly **right-to-left into the district** (balance −0.33 at 08:00), afternoons reverse
**left-to-right homeward** (+0.48 at 16:00 — the day's extreme). The evening tide is also
~2× the volume of the morning one. The light's flow-following idle behaviour — drifting
toward where people come from — was riding a genuine, measurable tide, not noise.

- **Show it:** a diverging horizontal bar chart by hour (left = into the district, right =
  homeward). One of the strongest single figures in the dataset. (Proposed **G2**.)
- **Say it:** *the street inhales at eight and exhales at five, and the light leaned into
  each.*

### Finding 10 — Tuesday is the big day; Friday is half-gone

Day-of-week signature (measured full days): **Tue 21.2K** > Thu 18.8K > Wed 16.0K > Mon
15.4K > **Sat 14.0K ≈ Fri 14.0K** > Sun 11.2K people/day. The mid-week office peak is
expected; **Friday matching Saturday** is the contemporary detail — a hybrid-work
financial district, visible from a shop window.

- **Show it:** a seven-bar week with Fri/Sat visually paired.
- **Say it:** *by the count of passing bodies, Friday downtown is now a weekend day.*

### Finding 11 — The desire line misses the alcove by design

Raw position occupancy by depth (1.27M positions, Mar 15–17): pedestrians overwhelmingly
travel the band **z 300–500 cm** (peak 350–400), the sidewalk's natural desire line. Only
~6% of positions ever fall inside the active zone (z < 283). This is the geometric ground
truth under everything: the 3.6% active share, the rarity of engagement, the alcove as a
deliberate *threshold* people must choose to cross. The site plan (B3) and this histogram
are two views of the same fact — one drawn, one measured.

- **Show it:** the Z-occupancy histogram laid directly over the B3 site plan, zones
  shaded; or a position heat-strip on the plan itself.
- **Say it:** *the sidewalk has a desire line, and the installation lives just off it —
  engagement meant stepping out of the river.*

---

## D. Instrument caveats discovered (carry these into any analysis)

1. **The `speed` column is a smoothing artifact** — 86.8% of values sit under 40 cm/s with
   a flat tail (real walking ≈ 130 cm/s). The EMA position filter (α≈0.03) destroys true
   speed; early reports' `avg_speed` values (1,700+) are likewise meaningless. Don't use
   speed for claims; the *flow direction* counts remain valid.
2. **`autotune_daily_learnings` goes to zeros at V6** (Mar 3) — the V6 tuner didn't feed
   it. The 33-day trajectory is V4/V5-era only; the V6.5 endpoint lives in config files.
3. Already documented but bears repeating with the above: the Jan 29 – Feb 12 active/passive
   labels are estimates; Feb 3–9 day values come from report files; `aware` exists in data
   only from Mar 15.

---

## E. Publication angles this opens

| Angle | Findings | Venue fit |
|---|---|---|
| **The reality of public engagement** — constant ~8.5% social fraction, 758 fleeting episodes, 12 bonds, the redesign changing the waiting not the stopping | 4, 7, 9, 11 | TEI / DIS / HCI — empirical, honest counterweight to engagement-success narratives |
| **Adaptive systems in the wild** — punctuated equilibrium, the two friction regimes, the first self-review, instrument co-evolution | 1, 2, 3, caveats | systems/design-methods papers; pairs with §18's data story |
| **The self-narrating machine** — 33 days of `strategy_summary` texts: the system describing its own becoming in plain English ("Energy decreased 0.94 → …") | 1, 2 | design-research / "Algorithmic Fictions"-type venues; also a beautiful pictorial text element: the diary entries set against the trajectory chart |
| **A shop window as urban sensor** — the breathing street, Tuesday-peak, Friday≈Saturday, the desire line | 9, 10, 11 | urban data / media architecture; standalone from the AI story |
| **An emotional portrait in data** — loneliest at 4 AM, the bond dozen, welcome overtaking thinking | 5, 7, 8 | the affective register for design audiences; strong captions for the TEI pictorial |

## F. New figures worth building (in house style)

| ID | Figure | From | Strength |
|---|---|---|---|
| **G1** | Personality trajectories, 33 days, version rules overlaid | Finding 1 | replaces/extends E3 with the true story |
| **G2** | The breathing street — diverging flow-by-hour | Finding 9 | strongest new single image |
| **G3** | Engagement episode durations + the 12 bonds | Finding 7 | the honest-engagement centrepiece |
| **G4** | Aggression vs traffic, two curves in opposite phase | Finding 8 | the 4 AM portrait |
| **G5** | Gesture economy, V5 vs V6.5 ranked bars | Finding 5 | "solitary → social" |
| **G6** | Mode time-budget bars, V5 vs V6.5 | Finding 4 | the restructured waiting |
| **G7** | Brightness ladder by mode | Finding 6 | simple, quantified, warm |
| **G8** | Z-occupancy over the site plan | Finding 11 | drawn vs measured geometry |
| **G9** | Budget regimes (starved → never-binding) | Finding 3 | the friction evidence |
| **G10** | The diary — strategy-summary quotes on the trajectory timeline | Findings 1–2 | text-as-figure; pictorial-ready |

*All source data is extracted or one query away (`analysis/merged_run.db` + the two source
DBs for raw positions and daily learnings). G1's data is already pickled from this
analysis session.*
