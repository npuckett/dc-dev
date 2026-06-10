# Drop Ceiling — Rendered Diagrams

SVG (vector, for layout/print) + PNG (raster preview) renders of the figures described
in [../ACADIA_2026_DIAGRAMS_RENDERABLE.md](../ACADIA_2026_DIAGRAMS_RENDERABLE.md).
Editable sources are in [`src/`](src). Re-render everything with `./build.sh`.

| ID | File | Figure | Engine |
|----|------|--------|--------|
| A0 | `A0_master_overview` | **Master overview** — combines A1 + A2: pipeline + the three nested loops + the actual output parameters; indexes into A3/A4, A5, A7. Output box has an icon placeholder (TBD). | Mermaid flowchart |
| A1 | `A1_pipeline` | Top-level pipeline (3 processes, 2 channels) | Mermaid flowchart |
| A2 | `A2_nested_loops` | Three nested loops (subgraph form) | Mermaid flowchart |
| A3 | `A3_mode_state_machine` | Behaviour mode state machine | Mermaid state diagram |
| A4 | `A4_modulation_chain` | Meta-param → light modulation chain | Mermaid flowchart |
| A5 | `A5_trend_windows` | Trend windows → influence signals | Mermaid flowchart |
| A6 | `A6_aggression_tank` | Aggression as a regulated tank | Mermaid flowchart |
| A7 | `A7_self_tuning_feedback` | Self-tuning feedback (PUSH/RESIST/META) | Mermaid flowchart |
| A8 | `A8_budget_mechanism` | Interaction-budget mechanism | Mermaid flowchart |
| A9 | `A9_self_analysis_mirror` | Self-analysis mirror loop | Mermaid flowchart |
| A10 | `A10_ingestion` | Real-time ingestion pipeline | Mermaid flowchart |
| A11 | `A11_web_two_planes` | Web interface — two data planes | Mermaid flowchart |
| B1 | `B1_db_funnel` | Tiered database funnel | Graphviz `dot` |
| B2 | `B2_nested_loops` | Concentric nested loops (filled rings) | Graphviz `dot` |
| B3 | `B3_spatial_plan` | Installation plan, to scale (cm) | Graphviz `neato` |
| D1 | `D1_run_totals` | Run totals over 54 days (log-scale bars), greyscale | matplotlib |
| D2 | `D2_eval_cadence` | Fast vs slow evaluations per day (the friction argument), greyscale | matplotlib |

### Submission-ready (black/white/grey, matches A0)

Restyled versions of the two most narrative-critical figures, for the ACADIA submission. The
detailed originals (`C1_funnel_to_12`, `A7_self_tuning_feedback`) remain for the guide.

| ID | File | Figure | Engine |
|----|------|--------|--------|
| C1‑bw | `C1_funnel_to_12_bw` | **The funnel** — whole adaptive system collapses to 12 light values (the strongest single idea); greyscale | Mermaid flowchart |
| A7‑bw | `A7_self_tuning_feedback_bw` | **Self-tuning loop** — PUSH / RESIST / META; "adapts but never runs away"; greyscale | Mermaid flowchart |

### E-series (TEI 2027 pictorial — greyscale + warm light accent)

New figures for the TEI pictorial (see `../TEI_2027_PICTORIAL_PLAN.docx`). Rendered by
`src/E_personality_radar.py` and `src/E_gesture_plate.py` via the project venv:
`../.venv/bin/python src/E_personality_radar.py`.

| ID | File | Figure | Engine |
|----|------|--------|--------|
| E1 | `E1_gesture_plate` | **Gesture vocabulary** — 8 motion glyphs (nod, lean, sway, orbit, settle, breathe, sweep, focus) with real cm amplitudes + dwell phase; the embodiment centrepiece | matplotlib |
| E3 | `E3_personality_radar` | **Personality drift** — 6-axis radar, neutral start vs as-deployed after 54 days (responsiveness/energy/exploration the big movers); real values | matplotlib |
| E4 | `E4_daily_rhythm` | **One day in the life** — 24h activity (people/hr) with rush-hour peak + a mode lane (idle/flow/aware/engaged); real data, Tue Mar 10 | matplotlib |

*Still to build for the pictorial (need data/photos): E2 dwell photo-response strip, E5 week-1-vs-7 comparison, E6 greyscale recolour pass — see the plan doc.*

### F-series (the run-data story — §18 of the guide)

Real-data figures documenting the software/data co-evolution, the calibration artifact, and
the gap recovery. Rendered from `analysis/merged_run.db` (`hourly_stats_filled`).

| ID | File | Figure | Engine |
|----|------|--------|--------|
| F1 | `F1_software_data_timeline` | **Software vs capture** — events/day (measured / partial / estimated / report-recovered) over the version milestones (V2→V6.5c, DB fix, AWARE-in-data); shared date axis | matplotlib |
| F2 | `F2_calibration_artifact` | **The artifact** — active-zone share: the impossible ~97% pre-fix plateau collapsing to a realistic ~1–12%, with corrected/measured points | matplotlib |
| F3 | `F3_report_recovery` | **Tiered-retention recovery** — raw (48h) → hourly → daily report; how Feb 3–9 survived in the reports after being pruned from the DB | Mermaid flowchart |

> F1/F2 re-render via `../.venv/bin/python src/F_software_data_timeline.py` and
> `src/F2_calibration_artifact.py` (F2 reads `/tmp/f2data.pkl`; regenerate it from the merged
> DB if missing). F3 via `mmdc`.

### H-series (the second-reading figures — DROPCEILING_FINDINGS_H.md)

A second, independent reading of the run data — 10 figures that the G-series
didn't build, from per-cycle tuner behaviour, the surviving meta-review, the
48-day brightness curve, the mode entropy timeline, the spatial footprint of
each mode, the gesture spatial map, the parameter correlation structure, and
the empirical regime-conditional tuner deltas. House style: greyscale + warm
accent (same as E and G). Data prep: `analysis/h_data_prep.py` writes
`analysis/h_data.json`.

| ID | File | Figure | Engine |
|----|------|--------|--------|
| H1 | `H1_regime_deltas` | **Regime-conditional param deltas** — 4×12 heatmap: mean per-cycle Δparam split by activity regime (dead/trickle/steady/rush); RUSH row is the only one with strong colour | matplotlib |
| H2 | `H2_meta_review` | **The first self-diagnosis** — every parameter at a wall, Feb 13 10:23 (8 of 12 parameters at 96.6% clamp, memory at 100%) | matplotlib |
| H3 | `H3_aggression_heatmap` | **Per-cycle aggression × day × hour** — the same data as G4 un-averaged; the 4 AM dark band and 14:00 light band visible on every day | matplotlib |
| H4 | `H4_param_clusters` | **The 12-dial personality is a 3-factor system** — daily net-change heatmap with the 3 clusters (outputs r=0.99, personality r=0.99, trade-offs r=−0.94) boxed | matplotlib |
| H5 | `H5_mode_entropy` | **Behavioural richness tripled when the vocabulary did** — Shannon entropy of per-day mode distribution, normalised; the Mar 3 V6 step is a discontinuity | matplotlib |
| H6 | `H6_episode_grid` | **Where the 758 episodes happened** — 3 days × 24 hours, the 03-16 17:00 cell is hottest (56 episodes) | matplotlib |
| H7 | `H7_mode_footprints` | **The spatial footprint of each mode** — density of light (x, z) per mode in V6.5; ENGAGED pulls forward 9 cm, CROWD pulls forward 14 cm and off-centre | matplotlib |
| H8 | `H8_gesture_spatial` | **Motion vs posture** — mean target (x, z) per gesture; sway/orbit/sweep at z=+20, the others at z=0 | matplotlib |
| H9 | `H9_brightness_timeline` | **The light took a week to come on** — 48-day daily-mean brightness from hourly_stats_filled; first 5 days = 0 DMX, V6 deployment = the cliff | matplotlib |
| H10 | `H10_step_distribution` | **Per-param step-size distribution, by regime** — 8 small multiples of std and mean per cycle, the spread grows in RUSH | matplotlib |

> Re-render from `diagrams/`: `../../.venv/bin/python src/H_findings_figures.py`
> (reads `../analysis/h_data.json`; regenerate with
> `../analysis/h_data_prep.py` if missing).

### G-series (the fresh-findings figures — DROPCEILING_FINDINGS.md)

The ten figures that carry Findings 1–11 of
[`../DROPCEILING_FINDINGS.md`](../DROPCEILING_FINDINGS.md). Real data, greyscale + warm
accent (same house style as the E-series). Data prep: `analysis/g_data_prep.py` writes
`analysis/g_data.json` from `analysis/merged_run.db` + the two source DBs.

| ID | File | Figure | Engine |
|----|------|--------|--------|
| G1 | `G1_personality_trajectories` | **Punctuated personality** — 19 daily snapshots of 5 meta-parameters, version steps (Feb 13, Feb 25) as warm rules; replaces/extends E3 | matplotlib |
| G2 | `G2_breathing_street` | **The breathing street** — diverging flow-by-hour, 08:00 inhales (warm) / 16:00 exhales (warm); 10.4M directional events | matplotlib |
| G3 | `G3_engagement_episodes` | **The fleeting 758** — log-scale episode-duration histogram with the four dwell-phase boundaries and the twelve 30 s+ bonds as warm dots | matplotlib |
| G4 | `G4_loneliest_hour` | **Loneliest at 4 AM** — aggression (warm) vs people (grey) over 24 h, opposite phase | matplotlib |
| G5 | `G5_gesture_economy` | **Solitary → social** — two ranked gesture bars (V5 / V6.5), `welcome` highlighted in the warm accent | matplotlib |
| G6 | `G6_mode_budget` | **Restructured waiting** — two stacked mode-time bars (V5 / V6.5), idle redistributed to flow/aware | matplotlib |
| G7 | `G7_brightness_ladder` | **Brightness ladder** — average DMX brightness by mode (idle 30 → crowd 388, cap 600) | matplotlib |
| G8 | `G8_desire_line` | **The desire line** — Z-occupancy histogram with active/passive zone bands, peak at z 350–400 cm | matplotlib |
| G9 | `G9_friction_regimes` | **Two friction regimes** — budget level + %-depleted over the run, cliff at the V5 update | matplotlib |
| G10 | `G10_diary` | **The diary** — verbatim `strategy_summary` excerpts (Day 1, first self-review, V5, V6) anchored on the energy trajectory | matplotlib |

> Re-render from `diagrams/`: `../../.venv/bin/python src/G_findings_figures.py` (reads
> `../analysis/g_data.json`; regenerate that with `../analysis/g_data_prep.py` if missing).

> **D-series (scale charts)** are rendered by `src/D_scale_charts.py` with the project
> venv: `../.venv/bin/python src/D_scale_charts.py`. Values are the §15-0 *estimates*
> (extrapolated from 34 surviving daily reports); update the arrays in that script if the
> full database later yields hard totals.

### A7 exploded — one sub-diagram per node of the self-tuning loop

Each node of `A7_self_tuning_feedback` is itself a process; these zoom in.

| ID | File | The A7 node it explains | Engine |
|----|------|--------------------------|--------|
| A7.1 | `A7_1_push_signal` | **PUSH** — how live presence + trends become `behavior_status` + an engagement score | Mermaid flowchart |
| A7.2 | `A7_2_tuner_pipeline` | **TUNER** — the full `SmartAutoTuner.update()` cycle (gate → gradients → deltas → apply) | Mermaid flowchart |
| A7.3 | `A7_3_resist_forces` | **RESIST** — the 4-stage friction stack (reversion → step clamp → budget → value clamp) | Mermaid flowchart |
| A7.3b | `A7_3b_resist_worked_example` | **RESIST, worked** — a +0.050 nudge to `energy` traced through all 4 stages, with the ample-budget vs throttled branches | Mermaid flowchart |
| A7.4 | `A7_4_metaparams_state` | **meta-parameters** — the 12-value state with ranges/floors + slider/behaviour sync | Graphviz table |
| A7.5 | `A7_5_adjustments_record` | **behavior_adjustments** — the row logged each cycle + its producer/consumers | Graphviz table |
| A7.6 | `A7_6_light_output` | **light output** — meta-params → modulation chain → Art-Net DMX | Mermaid flowchart |
| A7.7 | `A7_7_metareview_pipeline` | **meta-review** — read 8 h → diagnose → compute → write overrides → log | Mermaid flowchart |

### C series — complexity → 12 light values (the output funnel)

The A/B diagrams show how the system *decides*; the C series shows how all of that
collapses to the only thing the fixtures actually read: **12 DMX bytes**, one per panel,
recomputed every frame from a tiny `PointLight` state (position x/y/z, intensity, falloff).

| ID | File | Shows | Engine |
|----|------|-------|--------|
| C1 | `C1_funnel_to_12` | **The funnel** — the whole adaptive system narrowing to the light state and fanning back out to 12 channels (hero figure) | Mermaid flowchart |
| C2 | `C2_light_state_pinch` | **The pinch point** — the exact `PointLight` fields read at render time, and which upstream system writes each | Graphviz table |
| C3 | `C3_per_panel_math` | **Per-panel math** — `get_panel_brightness()` step-by-step (displacement → rotate → anisotropic scale → falloff → ×intensity → clamp), run ×12 | Mermaid flowchart |
| C4 | `C4_panel_dmx_map` | **The 12 panels** — 4 units × 3 panels, fixed positions → DMX channels 0–11 → Art-Net frame → fixtures | Graphviz table |
| C5 | `C5_per_frame_sequence` | **One frame (~30 Hz)** — the actors and call order proving the whole funnel re-runs every frame | Mermaid sequence |

The narrative: **C1** is the headline (many → few → 12). **C2** names the "few" — about nine
scalars: position[x,y,z], `current_brightness`, `falloff_radius`, `falloff_scale[sx,sy,sz]`,
`falloff_rotation`. **C3** is the actual arithmetic each panel runs against that state.
**C4** grounds "12 values" in the physical wiring (channel map, 512-byte frame, universe 0).
**C5** makes the key point explicit: this isn't computed once — the entire collapse happens
~30 times a second. All values trace to `PointLight` / `PanelGrid` in `lightController_osc.py`.

Read them in node order (1→7) as a guided tour of one trip around the A7 loop:
PUSH (1) feeds the TUNER (2), which is resisted (3), mutates the parameter state (4),
logs a row (5), shapes the light (6), and is periodically re-governed by the
meta-review (7) — whose override file loops back to (2). The parent `A7_self_tuning_feedback`
diagram now carries the **A7.1–A7.7** tags on each node so the set reads as a click-through.

`A7.3b` is the recommended figure when you want to *show the friction working* rather
than just name it: it follows one proposed `energy` nudge (+0.050) as mean-reversion,
the step clamp, the budget, and the value clamp whittle it down to a ~0.012 step (or
~0.006 when the budget is depleted). All constants are the V6.5c defaults; the arithmetic
is illustrative (gradient/regime supply the initial +0.050).

### P-series — captured project images (not generated here)

Real photos, renders, and screens pulled from elsewhere in the repo into `assets/` so all
submission imagery lives in one place. These are **source material to crop/clean**, not
finished figures.

| ID | File | What it is | Original location |
|----|------|-----------|-------------------|
| P1 | `assets/P1_installation_photo.jpg` | The installation in the Haworth window at night (hero photo) | `screenshots/dcView.jpg` |
| P2 | `assets/P2_unit_geometry.png` | 3D render of the four units / numbered panel geometry | `DEVversion/artnetTest/4panels_3panelLight.png` |
| P3 | `assets/P3_original_system_diagram.png` | The project's own early system diagram (A-series is the cleaned reconstruction) | `screenshots/systemDiagram.png` |
| P4 | `assets/P4_pygame_3d_twin.png` | The pygame+OpenGL control runtime (the "3D game" twin) | `UIs/10_pointLightController3D_pygame.png` |
| P5 | `assets/P5_agent_authoring_session.png` | A dev session: 3D scene beside an AI coding-agent chat | `UIs/09_pointLightController3D.png` |
| P6 | `assets/P6_aruco_marker.png` | An ArUco fiducial marker (calibration set) | `calibration/marker_1.png` |

> More candidates exist in the repo if needed: the full `UIs/` controller-evolution series
> (01–11), early tracker screens in `DEVversion/`, the `calibration/` + `NewMarkers/` marker
> sets, and `IO/BEHAVIOR_DIAGRAMS.pdf`. The original behaviour brief to the agents is
> `DEVversion/artnetTest/lightPanelBehavior.md` (primary source for §15d).

## Layout
```
diagrams/
├── *.svg            ← vector renders (use these for the submission)
├── png/*.png        ← raster previews (A* at 2×, B* at 150 dpi)
├── assets/          ← captured project images P1–P6 (photos/renders/screens)
├── src/             ← editable .mmd / .dot / .py sources
├── puppeteer.json   ← headless-chrome args for mmdc
└── build.sh         ← re-render all generated diagrams
```

## Notes
- **SVG is authoritative** for the proceedings/poster — infinitely scalable, editable in
  Illustrator/Inkscape. PNGs are convenience previews.
- **B3 is a to-scale layout reference**, not a finished figure: `neato` pins panels,
  cameras and ArUco markers to their real centimetre coordinates from
  `lightController_osc.py`. For the poster, open it and fill the active/passive zone
  rectangles (bounds are printed in the figure).
- All numeric values trace to the V6.5c source (see the parent guide for line refs).
- The Mermaid figures use plain ASCII in labels (no `×`, `→`, `≥`, `±`) so they parse
  cleanly across renderers; the parent guide keeps the typographic versions.
