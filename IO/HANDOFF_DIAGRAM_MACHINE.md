# Requirements brief: a three.js "diagram machine" for the Drop Ceiling control logic

> **Purpose of this document.** This is not an implementation plan. It's a requirements
> brief, worked out through Q&A, meant to be handed to a fresh thread that will turn it
> into a formal agent implementation plan. It captures decisions made, the reasoning
> behind them, and the questions that are still deliberately open.

## Context

`IO/diagrams/` currently holds ~58 static SVG/PNG diagrams (Mermaid, Graphviz, and
matplotlib, built via `build.sh`) that explain how the Drop Ceiling installation's
control software works — everything from the high-level architecture (A-series: nested
loops, mode state machine, self-tuning feedback) down to the exact per-panel arithmetic
that turns a single abstract "point light" into 12 DMX byte values sent to physical LED
panels (C-series: `C1_funnel_to_12` → `C5_per_frame_sequence`). These diagrams were
originally built to support ACADIA 2026 / TEI 2027 submissions, but that process
surfaced a more general, ongoing need: **the system has no way to actually show these
relationships happening** — only static flowcharts and tables describing them in the
abstract. There's no visualization that shows real (or representative) positions,
brightness values, and falloff shapes changing over time in the actual 3D space of the
installation.

The goal is a new tool — a three.js-based "diagram machine" — that can render this logic
as data-driven, styled diagrams: some spatial/3D (panels, light, people, room), some
graph-style (state machines, funnels, nested loops), all in one consistent black/white/
greyscale visual system. It should work both as an interactive web tool for exploring/
authoring, and as a headless batch renderer for producing stills and animation frame
sequences for papers, talks, and documentation.

This brief exists to hand off enough decided scope, style rules, and open questions that
the next thread can go straight to architecture and implementation planning without
re-deriving the "what" from scratch — and without needing to read the whole repo first.
See **Roadmap** and **Start here** below before going any further afield.

## Roadmap

Build in this order. Each phase has a narrow, checkable target — don't pull in material
for a later phase while working on an earlier one.

1. **Phase 1 — Core engine + the C-series pilot, synthetic data only.**
   The narrowest possible end-to-end slice: the style system (black/white/greyscale
   rules), the fixed-camera-preset system, two layer types (the 12 panels + the abstract
   point-light/falloff shape), the shared annotation/label style, a minimal declarative
   config format, and a Puppeteer-driven PNG batch export — all proven on **one**
   hand-authored synthetic scenario (a point light moving through space, brightness and
   per-panel DMX values computed live using the exact formulas from
   `lightController_osc.py`). This is the whole "C-series funnel" idea
   (`C1_funnel_to_12` → `C5_per_frame_sequence`), rebuilt as one interactive + batch-
   renderable diagram. Nothing else — no real data, no graph diagrams, no multi-viewport
   composition — until this works end to end.
2. **Phase 2 — Real data.** Only after Phase 1 works. Build the new frame-level export
   script against `merged_run.db` for a small curated set of real moments (see
   **Data**), and wire it in as a second data-source option in the same config format
   used in Phase 1.
3. **Phase 3 — Graph/node layer type.** Add node/edge/containment-ring primitives to the
   same engine and reproduce one A-series diagram (e.g. the mode state machine) to prove
   the "everything, same machine" scope decision. Pull in only the one relevant `.mmd`/
   `.dot` source for that diagram, not the whole `src/` folder.
4. **Phase 4 — Multi-viewport composition + batch manifests.** Compose multiple fixed
   views into one sheet; define a batch-manifest format for running many scenarios ×
   cameras × outputs together (echoing `build.sh`'s "rebuild everything" loop).
5. **Phase 5 (later, out of scope for now) — public-facing polish**, if that's ever
   revisited.

## Start here (minimal reading — don't read the rest of either repo up front)

This tool is developed in `dcDiagramMachine`; the files below live in the **sibling
`dc-dev` repo** (see Repo layout / Data access sections below for the path convention).
Everything needed to start Phase 1:

1. **This document.**
2. **`dc-dev/IO/lightController_osc.py`** — read the `PointLight` and `PanelGrid`
   classes only (position/falloff state, `get_panel_brightness()`, DMX clamp). This is
   the source of truth every formula in the tool must match. **Use this exact file** —
   `dc-dev` has several older versions (`V4Dev/lightController_osc.py`,
   `IO/V3Dev/lightController_osc.py`, `IO/v1_backup_*/`); `IO/lightController_osc.py`
   at the repo root of `IO/` is the current one (V6.5c).
3. **`dc-dev/IO/world_coordinates.json`** — the 12 panel positions/angles in cm.
4. **`dc-dev/IO/diagrams/README.md`**, **just the "C series — complexity → 12 light
   values" section** — the exact fields, formulas, and narrative the Phase 1 pilot needs
   to reproduce. Skip the rest of the README for now.
5. *(Optional, quick skim)* **`dc-dev/IO/public-viewer/viewer.js`** — just the
   `CONFIG.PANEL_LOCAL_POSITIONS`/`CONFIG.PANEL_ANGLES` constants and its light/falloff
   visual model, as a cross-check on (2)/(3).

Everything else referenced later in this doc (panel-designer geometry, the P-series
images, the analysis/data-prep scripts, `build.sh`) is **reference material for later
phases** — pull each one in only when the phase that needs it comes up, not now.

## What this is (and isn't)

- It is **a machine for making many different diagrams**, not one fixed scene. A single
  diagram = a scenario/config that composes a subset of the machine's capabilities
  (which objects are visible, what drives them, which camera view(s)).
- Scope is the **full system** long-term — not just the C-series funnel, but eventually
  the A-series concepts too (mode state machine, nested loops, self-tuning loop,
  DB funnel) — but built incrementally, one diagram/capability at a time, not all at once.
- It is **additive, not a replacement**. The existing Mermaid/Graphviz/matplotlib
  pipeline in `IO/diagrams/` stays exactly as it is for all current diagrams. This new
  tool is for new and revised diagrams going forward — nothing needs to be migrated.
- It is a **standalone new tool** — not a fork/extension of `IO/public-viewer` (see
  Start here / Reference material below), though it should borrow that project's real
  position/geometry constants where relevant for accuracy.
- It is **personal/authoring-focused for now**. No public deployment in this phase;
  that's a plausible later phase, not a current requirement.
- There is **no hard deadline**. The original conference submissions prompted this need,
  but the tool itself isn't tied to a specific date.

## Visual style (settled)

Follows classical architectural/technical drawing conventions:

- **Black and white only**, with greyscale reserved for exactly one thing: **the
  brightness/darkness of the 12 physical light panels**. White = max brightness (255 /
  DMX max), black = minimum (0). This mapping applies **only** to the panel fills.
- Every other object — room bounds, cameras, ArUco markers, tracked people, panel
  frames/outlines, and the abstract "point light" + falloff shape — is **black outline
  only, white background, no fill**. This keeps "real physical output" (greyscale panels)
  visually distinct from "internal math abstraction" (wireframe point light) and from
  "context geometry" (outline-only room/people/cameras).
- The abstract point-light object (position + falloff radius/scale/rotation +
  `current_brightness`, per `C2_light_state_pinch`) is drawn as a labeled black-outline
  wireframe shape (sphere + falloff ellipsoid), **not** shaded by its own brightness — it
  stays an abstraction, distinct from the panels it drives.
- Data/value overlays (positions, brightness, falloff params, per-panel DMX bytes, etc.)
  must be **styled as part of the diagram itself** — typeset, print-ready annotation
  language (think dimension lines / callouts on an architectural drawing, or the existing
  matplotlib G/H-series "house style" labels) — not app-like UI chrome. Whether a given
  value surfaces in a side HUD panel or as an in-scene floating label is a per-diagram
  choice, but both must share one consistent annotation typography/leader-line system.
  This shared annotation style is a real design deliverable, not an afterthought.

## Architecture direction (settled)

**One flexible 3D scene/toolkit with swappable, composable layers** — not a library of
disjoint scene templates per diagram type. A "layer" is the unit of composition: room
geometry, cameras/ArUco markers, tracked people, the point-light abstraction, the 12
panels, DMX/data readouts, and — importantly — **graph/node-edge elements** (state
machine nodes and transitions, funnel stages, nested-loop rings, DB-funnel tiers) are all
first-class layer types living in the same coordinate system and sharing the same style
system. This is the mechanism by which "full system, built incrementally" works: new
diagrams are new combinations of existing layer types, not new bespoke scenes.

This means the "everything, for total visual consistency" scope decision (see below) is
a real architectural commitment: the layer/object model has to support both literal
spatial objects and abstract graph-diagram primitives (nodes, edges, containment/rings)
under one renderer and one style system.

### Graph-style content lives in the same machine

State machines, the DB funnel, nested-loop rings, etc. (currently Mermaid/Graphviz) get
rendered inside the same three.js system for total visual consistency across the whole
diagram set — not left to the old tools. Default treatment: **flat/orthographic** (no
perspective distortion, reads like a blueprint) for print clarity, but the same engine
can add real 3D depth when it clarifies the concept (e.g. nested loops as literal
concentric rings viewed at an angle, matching `B2_nested_loops`'s "concentric filled
rings" idea taken into true 3D).

### Multi-viewport composition

Diagrams should be able to **combine multiple fixed viewports in one composed layout**
— e.g., a flat orthographic overview alongside an angled 3D detail inset on the same
sheet, the way a technical drawing combines plan + elevation + isometric detail. This
applies to both spatial (C/B-series) and graph-style (A-series) content. This is a step
beyond simple camera-preset-switching: the composition system needs to support laying
out **multiple simultaneous fixed views** on one canvas/export, not just one view at a
time.

### Camera model

- Final rendered output (both interactive "captured" views and all batch exports) uses
  **fixed camera presets** — a handful of named, deliberate shots per diagram (e.g. full-
  room isometric, front elevation of the panel wall, single-panel close-up) — not free
  user-driven orbit. This matches the plan/elevation/detail convention of architectural
  drawing and keeps output reproducible.
- Open question (not settled): whether the *interactive* authoring mode should still
  allow free-orbit for exploration purposes (snapping back to/capturing from defined
  presets), even though final output is always preset-based. Recommend allowing it for
  authoring ergonomics, but this is for the next thread to decide.

## Data (settled, with one real prerequisite gap)

- Diagrams are driven by **both** real recorded data and hand-authored synthetic
  scenarios:
  - **Synthetic scenarios** isolate/teach one concept cleanly (e.g. one gesture in
    isolation, one mode, one falloff-rotation sweep) — authored parameter sequences, not
    tied to any real run.
  - **Real data** grounds diagrams in what actually happened during deployment.
- **Known gap:** the frame-level math this system is built to explain (`C2`/`C3`/`C5`)
  runs every ~33ms, but the data already exported to JSON
  (`dc-dev/IO/analysis/web_data/*.json`, `dc-dev/IO/analysis/h_data.json`) is aggregated
  at hourly/episode/cycle granularity — it does not contain raw per-frame `PointLight`
  state. True frame-accurate replay of a real moment is **not possible with current
  exports**.
  - **Decision:** plan a **new export script** (in the spirit of the existing
    `dc-dev/IO/analysis/g_data_prep.py` / `h_data_prep.py` pattern) that pulls short raw
    per-frame windows (a few seconds each) from `dc-dev/IO/analysis/merged_run.db` for a
    **curated set of specific real moments** worth illustrating — not a full
    frame-resolution export of the whole dataset. This script reads `dc-dev` data (see
    Data access from dc-dev) but writes its output into `dcDiagramMachine`'s own data
    folder — it's a one-off/occasional sync step, not a live runtime dependency. This is
    a genuine, scoped prerequisite piece of work the next thread needs to plan alongside
    the three.js tool itself, not an afterthought.
  - Coarser real data (`hourly.json`, `h_data.json`) is fine as-is for A-series-level
    narrative diagrams (mode distributions, aggression over time, etc.) that don't need
    frame accuracy.
- All computed values (falloff math, brightness formulas, DMX mapping) must trace back
  to `dc-dev/IO/lightController_osc.py` exactly, the same way the existing C-series
  diagrams do — this tool is a visualization of that source of truth, not an
  approximation of it.

## Authoring format (settled)

Each diagram/scenario is defined by a **declarative config file** (JSON/YAML) — not
code — specifying: which layers/objects are visible, the data source (a synthetic
scenario definition, or a pointer to a curated real-data window), camera preset(s) /
viewport composition, and time range / playback settings. No code changes should be
needed to add a new diagram. This matters because the point of "a machine for making
diagrams" is that the user (or a future coding agent) can produce new diagrams by
writing config, not by writing new renderer code each time. The exact schema is left for
the next thread to design, but this constraint (config-driven, not code-per-diagram)
should shape the architecture from the start.

## Output modes (settled)

- **Interactive web tool** — for personal exploration/authoring. No build step assumed
  (see Tech stack). Not public-facing in this phase.
- **Headless batch renderer** — produces **PNG frame sequences** (numbered, matching the
  existing `build.sh` convention), driven by the same declarative configs, so a
  "scenario" can become either an interactive session or a batch of exported frames
  without duplicating setup. Assembling frame sequences into video (e.g. via `ffmpeg`) is
  a separate/later concern, not a hard requirement of the renderer itself.
- Batch rendering should follow the existing headless-Chrome pattern already used for
  the current diagram pipeline (`dc-dev/IO/diagrams/puppeteer.json` +
  `build.sh`'s approach to `mmdc`) — i.e., Puppeteer driving a real three.js scene
  headlessly, rather than a from-scratch headless-GL setup.

## Tech stack (settled)

**Plain vanilla three.js, no framework** — no React, no build step. This matches
`IO/public-viewer`'s existing approach, keeps the interactive tool trivially deployable
as static files, and keeps headless batch rendering straightforward to drive with
Puppeteer (mirroring the existing `build.sh`/`puppeteer.json` pattern). Explicitly
**not** the `panel-designer` stack (React + `@react-three/fiber` + zustand) — that was
considered and set aside in favor of deployability and headless-rendering simplicity.

## Repo layout: this tool lives in `dcDiagramMachine`, not in `dc-dev`

Development happens in a **separate, standalone repo**: `dcDiagramMachine`
(`github.com/npuckett/dcDiagramMachine`), already created and git-initialized. On this
machine it sits as a **sibling directory to `dc-dev`**:

```
Documents/GitHub/
├── dc-dev/              ← this repo: the control software + all diagram source data
└── dcDiagramMachine/    ← the new tool: developed here, currently empty
```

`dc-dev` is the **source of truth**, read from but not vendored/copied into
`dcDiagramMachine` — no duplicating `lightController_osc.py`'s math, no copying data
files into the new repo's git history. `dcDiagramMachine` is where all new code
(renderer, layers, config loader, batch pipeline) is written and committed.

## Data access from dc-dev

Because the two repos are separate, the next thread needs to decide *how*
`dcDiagramMachine` reads `dc-dev` data — but the following should hold regardless of
that mechanism:

- **Assume sibling-checkout-on-disk access**, the same way this brief's file paths work
  right now (`../dc-dev/...` relative to `dcDiagramMachine`'s root). Don't hardcode that
  exact relative path, though — put it behind one config value (e.g. a `DC_DEV_PATH` env
  var or a one-line config file, default `../dc-dev`) so it survives repos being cloned
  into different sibling layouts or onto another machine.
- **Read-only.** `dcDiagramMachine` never writes back into `dc-dev`.
- **One path is local-machine-only, not just cross-repo:** `IO/analysis/merged_run.db`
  is real (240MB-class SQLite data) on this machine but is `*.db`-gitignored in
  `dc-dev` — it won't exist in a fresh clone of either repo. Phase 2 (real-data export)
  only works on a machine that has this file locally; that's expected, not a bug to
  route around.
- The **Phase 1 pilot needs zero live cross-repo querying** — it only needs to *read* a
  handful of static files once at setup (see Start here), not maintain an ongoing
  connection to `dc-dev`. Don't over-build the data-access layer before Phase 1 proves
  it's needed.

**Key `dc-dev` paths, by what they're needed for** (all paths below are relative to the
`dc-dev` checkout root):

| Path | What it is | Needed for |
|---|---|---|
| `IO/lightController_osc.py` | Canonical math source of truth — `PointLight`/`PanelGrid` classes, `get_panel_brightness()`, DMX clamp. **This is the current version** (V6.5c, May 2026) — ignore `V4Dev/lightController_osc.py`, `IO/V3Dev/...`, and the `IO/v1_backup_*` folders, which are older dev snapshots. | Start here / Phase 1 |
| `IO/world_coordinates.json` | 12 panel positions/angles, cm | Start here / Phase 1 |
| `IO/diagrams/README.md` | Index + narrative for all 58 diagrams (read only the C-series section for Phase 1) | Start here / Phase 1, then Phase 3 |
| `IO/public-viewer/viewer.js` | Existing plain three.js viewer — panel constants + light/falloff visual model, as a cross-check | Start here (optional) |
| `IO/diagrams/build.sh`, `IO/diagrams/puppeteer.json` | Existing headless-Chrome batch-render pattern to mirror | Phase 1 (batch export) |
| `IO/analysis/merged_run.db` | Raw run database (gitignored, local-only, confirmed present on this machine) | Phase 2 (new frame-level export script) |
| `IO/analysis/h_data_prep.py`, `IO/analysis/g_data_prep.py` | Existing DB → JSON export-script pattern to follow for the new export | Phase 2 |
| `IO/analysis/web_data/*.json`, `IO/analysis/h_data.json` | Existing aggregate (hourly/episode) real data — fine for narrative-level, non-frame-accurate diagrams | Phase 2 |
| `IO/analysis/WEB_DEPLOYMENT_PLAN.md` | Context on the existing tiered data-export plan | Phase 2 |
| `IO/diagrams/src/*.mmd`, `*.dot` | Source for the one specific graph diagram being reproduced (e.g. `A3_mode_state_machine.mmd`) — don't read the whole folder | Phase 3 |
| `panel-designer/src/config.js`, `panelGeometry.js` | Panel mesh geometry constants (bevel/lip profile) for visual fidelity only | Optional polish |
| `IO/diagrams/assets/P2_unit_geometry.png`, `P4_pygame_3d_twin.png` | Reference imagery / prior-art screenshot | Optional polish |
| `IO/V2Dev/world_coordinates.json`, `IO/V2Dev/WORLD_COORDINATES_FORMAT.md` | Older coordinate variant + format doc — only if the current `IO/world_coordinates.json` is ambiguous | Optional |

## Open questions, tagged by phase

Deliberately left unresolved — either genuine implementation details best decided with
the architecture in hand, or things the user wants the next thread to propose. Resolve
each at the start of the phase it's tagged with, not before.

1. **[Phase 1] Declarative config schema** — the exact shape of the per-diagram JSON/
   YAML (layer list, data-source reference, camera-preset format, time-range/playback).
2. **[Phase 1] Shared annotation/label style system** — concrete typography, leader-line,
   and layout rules for data overlays, designed once and reused everywhere.
3. **[Phase 1] Free-orbit vs. preset-only in interactive mode** — whether the
   interactive tool allows free camera exploration that snaps to/captures from defined
   presets, or is preset-only throughout. Recommend allowing free-orbit for authoring
   ergonomics; not load-bearing for the pilot either way.
4. **[Phase 1, optional] Pilot confirmation** — this brief recommends the C-series
   funnel as the Phase 1 target (see Roadmap) since it's the most concretely specified
   math and matches the original "position of people → brightness of 12 lights" framing.
   Treat this as decided unless the next thread finds a good reason to pick differently.
5. **[Phase 2] Frame-level real-data export script** — scope/design of the new
   `merged_run.db` export tool for curated per-frame real moments (output format, how
   moments are selected, how it plugs into the Phase 1 config format's data-source field).
6. **[Phase 3] Layer/object model for graph content** — how graph-diagram primitives
   (nodes, edges, containment rings) fit the layer abstraction built in Phase 1.
7. **[Phase 4] Viewport composition mechanics** — multiple renders composited into one
   canvas? Multiple `<canvas>` elements in a CSS layout? Split viewport in one WebGL
   context? Decide once Phase 1's single-viewport version is working.
8. **[Phase 4] Batch manifest format** — how a batch (multiple scenarios × camera
   presets × output settings run together) is specified and invoked.
