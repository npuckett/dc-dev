# DropCeiling — The Story, and How to Tell It

*The distilled version. The full working reference is
[`ACADIA_2026_DIAGRAM_GUIDE.md`](ACADIA_2026_DIAGRAM_GUIDE.md) (18 sections of analysis,
drafts, and notes); this document keeps only what proved to work — the narrative moves
that explain the project best, the figures that carry them, and the numbers that hold up.
Use this as the starting point for any new publication, talk, or page.*

---

## 1. The project in one paragraph

DropCeiling is an open-source system that turns standard LED office light panels into an
adaptive Augmented Reality character. Twelve panels in a Toronto storefront window are lit
by a single virtual point light whose position, intensity, and falloff respond to
pedestrians tracked on the sidewalk outside. Interaction is entirely implicit — no screen,
no instruction, only presence. The light's behaviour is governed by three nested control
loops at three timescales: it reacts to the people present, anticipates the rhythm of the
street, and slowly retunes its own personality across days. An interaction budget meters
how fast that personality can change, keeping adaptation deliberate across the whole run.
It ran twenty-four hours a day for fifty-four days, taking in roughly one million
pedestrian inputs and over twenty-three million tracking events, and it measurably changed
along the way. It was developed with AI coding agents working against a shared 3D model of
the site — the model didn't describe the installation, it became its brain.

---

## 2. The seven narrative moves

These are the explanatory methods that survived a full season of drafting. Each names the
move, why it works, and the figure that carries it.

### Move 1 — Open on the body and the site, not the system

Start with the alcove: two columns on a busy sidewalk create a natural pocket in front of
the window, and that architecture *is* the interaction design — it divides the world into
an active zone (people who step in) and a passive zone (people who pass). Presence alone
is the input. Leading with technology buries the point; leading with the street makes
every system decision legible afterward.

> Carried by: **P1** (installation photo, night) + **B3** (spatial plan). Pair them.

### Move 2 — The three nested loops, with the indirection principle

The single most load-bearing explanation in the project. Three loops at three timescales:
**reaction** (~33 ms — modes and gestures follow who is present now), **anticipation**
(seconds–hours — trend windows and a flow tracker pre-position the light), **self-tuning**
(hours–days — the meta-review reshapes the twelve meta-parameters).

The key sentence: *only the innermost loop ever writes the light; each outer loop sets the
terms for the one inside it.* Loop 1 sets the light's **values**, Loop 2 sets its
**context**, Loop 3 sets the **coefficients** of its reaction. Three consequences follow:
it is the mechanism of "character" (the same stimulus draws a different response weeks
later); timescale separation buys stability (slow self-modification physically cannot
cause fast flicker); and it stays authorable (a designer tunes three legible things, not
one tangled controller).

> Carried by: **A0** (master overview — pipeline + nested loops + the point-light output,
> greyscale, the current canonical system diagram).

![A0 master overview](diagrams/png/A0_master_overview.png)

### Move 3 — The collapse to twelve

The strongest single idea to leave a reader with: a city corner's worth of context —
cameras, trends, memory, a self-reviewing tuner, twelve personality parameters, five modes
and a gesture library — collapses, thirty times a second, to the state of one virtual
point light (~9 values: position, intensity, falloff shape), from which each of twelve
panels brightens by distance. One DMX byte per panel. Put the scale number right beside
it: *twenty-three million tracking events, distilled live into twelve values of light.*
The restraint is the point — the complexity is spent on how to behave, not on resolution.

> Carried by: **C1‑bw** (the funnel, greyscale). **C3** (per-panel falloff math) as the
> technical inset when depth is wanted.

![C1 funnel](diagrams/png/C1_funnel_to_12_bw.png)

### Move 4 — Show friction working, don't name it

"The system has an interaction budget" is a claim; tracing one proposed change through the
friction stack is a demonstration. The worked example: the tuner proposes +0.050 to
`energy`; mean-reversion trims it, the step clamp cuts it to +0.012, the budget pays for
it (or throttles it to +0.006 when depleted), the value clamp bounds it. A 0.050 grab
becomes a ~0.012 step. Three forces in balance: live signal **pushes**, friction
**resists**, and the 3×/day meta-review **re-governs** how hard each side pushes — it even
retunes the budget itself. This is what kept the algorithm as responsive on the final day
as the first: the budget, a pull toward learned resting values, and a periodic curiosity,
together preventing it from going twitchy *or* going stale.

> Carried by: **A7‑bw** (the self-tuning loop, PUSH/RESIST/META, greyscale) +
> **A7.3b** (the worked example — needs the greyscale recolour pass).

![A7 self-tuning loop](diagrams/png/A7_self_tuning_feedback_bw.png)

### Move 5 — Character as data

The personality is not a metaphor; it is twelve numbers, and they moved. Over the run the
light drifted from its neutral start into a markedly more responsive (0.53 → 0.83),
energetic (0.48 → 0.77), exploratory (0.40 → 0.71) character — real values from the
deployed configuration, shown as two rings on a radar. The gesture vocabulary gives the
character a body: eight ongoing motions (nod, lean, sway, orbit, settle, breathe, sweep,
focus) drawn as glyphs with their real centimetre amplitudes. And the daily rhythm shows
the character in time: idle through the night, following the morning commute, rising
through `aware` and `engaged` across the afternoon rush — its behaviour breathing with the
street.

> Carried by: **E3** (personality radar), **E1** (gesture plate), **E4** (one day in the
> life, real data, Tue Mar 10 — 3,171 people at the 16:00 peak).

![E3 personality radar](diagrams/png/E3_personality_radar.png)

![E4 daily rhythm](diagrams/png/E4_daily_rhythm.png)

### Move 6 — Tell the data story straight

The run's data history is messy, and the mess is evidence. The software was developed
*while it ran* (V2 → V4 → V5 → V6 → V6.5c), and what the database captured tracked what
the software could do — modes and gestures appear in the data on the dates the versions
added them. An early calibration offset made 97% of pedestrians read as "active" for five
days (physically impossible; corrected by estimation and flagged). A week of data the
database pruned was recovered intact from the system's own daily report files — the tiered
retention design saving its own history. Foregrounding this reads as rigour: a real
deployment, not a demo.

> Carried by: **F1** (software updates vs database capture — the co-evolution timeline),
> **F2** (the calibration artifact, 97% → 3.6%), **F3** (tiered retention recovered the
> lost week).

![F1 software/data timeline](diagrams/png/F1_software_data_timeline.png)

![F2 calibration artifact](diagrams/png/F2_calibration_artifact.png)

### Move 7 — The making-of as method

The development story is a contribution of its own, kept distinct from the runtime story:
the system was authored with AI coding agents, and because the problem is spatial, the
solution was a shared 3D model — panel positions, cameras, zones — exported as
natural-language context for the agents and as JSON for the runtime. The control software
itself is a simple 3D game (pygame + OpenGL) where physical space, virtual light, and
control logic share one coordinate frame: you watch the light think in the same model the
agents reasoned about. Keep build-time AI (agents authored it) and runtime adaptation (it
tuned itself) clearly separate — they are different layers and conflating them costs
credibility.

> Carried by: **P4** (the pygame 3D twin) beside **P1** (the real install) — same
> coordinate frame; optionally the spatial-editor screenshots and **P6** (ArUco marker).

---

## 3. The canonical figure set

The curated finals, in house style (greyscale, black strokes, one warm accent reserved for
the light). Everything else in `diagrams/` is working material.

| Figure | File (`IO/diagrams/`) | Carries | Status |
|---|---|---|---|
| **A0** | `A0_master_overview` | the whole system: pipeline + 3 loops + point-light output + 12 panels | ✅ final |
| **C1‑bw** | `C1_funnel_to_12_bw` | the collapse to twelve | ✅ final |
| **A7‑bw** | `A7_self_tuning_feedback_bw` | PUSH / RESIST / META | ✅ final |
| **A7.3b** | `A7_3b_resist_worked_example` | friction, worked (+0.050 → +0.012) | ⚠️ recolour to house style |
| **E1** | `E1_gesture_plate` | gesture vocabulary, 8 motion glyphs, real cm | ✅ final |
| **E3** | `E3_personality_radar` | personality drift, real values | ✅ final |
| **E4** | `E4_daily_rhythm` | one day in the life + mode lane, real data | ✅ final |
| **F1** | `F1_software_data_timeline` | software/data co-evolution | ✅ final |
| **F2** | `F2_calibration_artifact` | the 97% artifact, shown honestly | ✅ final |
| **F3** | `F3_report_recovery` | tiered retention recovered the lost week | ✅ final |
| **D1 / D2** | `D1_run_totals`, `D2_eval_cadence` | run scale; fast-vs-slow cadence | ⚠️ refresh with real totals (~23.3M) |
| **B3** | `B3_spatial_plan` | site plan, zones, cameras (to scale, cm) | ⚠️ scale reference; redraw for print |
| **A3** | `A3_mode_state_machine` | the five modes + transitions | ⚠️ recolour to house style |
| **P1–P6** | `assets/` | photos: install, units, system, pygame twin, authoring, marker | P5 needs crop |

Still to build (data and photos are now in hand): **E2** dwell photo/response strip,
**E5** week-1 vs week-7 comparison (early block Feb 13–24 vs late block Mar 5–16), **E6**
the recolour pass across the remaining figures.

---

## 4. Numbers you can use

From the merged, verified run database (`analysis/merged_run.db`) and deployed configs.
Full provenance in [`analysis/DATA_TIMELINE_AND_MERGE.md`](analysis/DATA_TIMELINE_AND_MERGE.md).

**Solid:**
- Ran **24/7 for ~54 days** (Jan 29 – Mar 17, 2026); **48 days of data coverage**.
- **~23.3 million tracking events**; peak day 1.18M events / 22,385 people (Mar 10).
- **12 panels**, **2 cameras**, **~9-value light state**, **12 meta-parameters**, 30 Hz.
- Personality drift: responsiveness **0.53 → 0.83**, energy **0.48 → 0.77**, exploration
  **0.40 → 0.71** (deployed config vs home values).
- Post-fix active-zone share: **~3.6%** of detections (1–12% daily range).

**Estimated — always label it:**
- "~1 million inputs/encounters" (state the unit; the DB's per-hour people sum is 537K and
  over-counts, so "inputs" is the safe word).
- Jan 29 – Feb 12 active/passive split (re-estimated by matched weekday+hour).
- Feb 3–9 day values (recovered from report files, not the live DB).

**Don't claim:**
- "162 meta-reviews" (the audit table is empty in both surviving DBs).
- Per-visit dwell durations (`person_sessions` is empty).
- "16 gestures" / "five modes" *for the whole run* — the vocabulary grew; `aware` exists
  in the data only from Mar 15. Say "the final system."

---

## 5. Craft rules

**Visual:**
- Greyscale structure; **one warm accent reserved exclusively for the light** — the eye
  should find the luminous body on every page.
- Pair photograph and diagram: the lived moment beside its mechanism.
- Time runs along the page (gestures, days, the whole run).
- Standalone figures carry no cross-references to other figures; estimated data is flagged
  *on the figure*, not just in a caption.
- "Personality" is not a variable — it's the informal name for the 12 meta-parameters as a
  set. Define it once, early.

**Voice (learned from the ACADIA edit pass):**
- Plain and concrete; end paragraphs on the fact, not a flourish or aphorism.
- Reach for the literal mechanism over elevated abstraction ("without becoming lazy and
  simply turning all the sliders to maximum," not "without growing rigid").
- Don't compress away natural rhythm; spend words where the sentence needs them.
- Skip academic comparisons unless they're load-bearing; describe what *this* system does.

---

## 6. Where this goes next

| Outlet | Status | Spine |
|---|---|---|
| **ACADIA 2026** (Projects, 600 words) | submitted | the collapse to twelve + friction; `ACADIA_2026_DropCeiling_submission.docx` |
| **TEI 2027** (Pictorial, ≤12 pp, due Aug 6 2026) | planned | embodiment-led hybrid — body/duet → collapse foldout → 54-day drift; `TEI_2027_PICTORIAL_PLAN.md` |
| **Interactive diagram engine** (web) | planned | static JSON aggregates on CDN, honesty toggles for estimated data; `analysis/WEB_DEPLOYMENT_PLAN.md` |

The same seven moves serve all three — the venue changes which move leads, not what the
story is.
