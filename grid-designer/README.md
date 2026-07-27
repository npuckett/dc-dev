# grid-designer

A three.js tool for **planning the Drop Ceiling V2 installation** as a **tiled 3D surface** — a snow
drift built from rigid ceiling-light panels. You author a drift form, a tiling algorithm decides
which cells get a square and which get a plate, the panels are placed on it, and the tool measures
**what the connectors have to absorb**.

That last part is the point. Read **[V3_SPEC.md](V3_SPEC.md)** for the model and
**[HANDOFF.md](HANDOFF.md)** for the decision log and what is still open.

| project | what it is | status |
|---|---|---|
| `grid-designer/` | **this** — tiled 3D drift surface planning (schema v3) | active |
| `panel-designer/` | first attempt: single-rooted kinematic tree | abandoned; some modules were copied out |
| `spatial-editor/` | generic r3f scene editor | unrelated prior art |
| `IO/public-viewer/` | live viewer for the **existing** installation | separate concern |

---

## Running it

```bash
npm install --prefix grid-designer
npm run dev --prefix grid-designer
```

Serves on **port 5175** (`strictPort` — `panel-designer` and `spatial-editor` both want 5173).
There is a launch config at `.claude/launch.json` named `grid-designer`.

```bash
npm run build --prefix grid-designer
```

The working config **is persisted** to `localStorage` (autosaved, debounced), plus named slots — so
a reload, including the hot reload from editing a source file, no longer destroys your design. This
cost real work twice under v2.

---

## The physical system

Two panel types, standard drop-in ceiling-light sizes:

- **square** — 60 × 60 cm
- **plate** — 60 × 121 cm

At a **1 cm** joint, `60 + 1 + 60 = 121` exactly, so a plate is a precise drop-in for two squares
plus their joint. That exactness is load-bearing for the kit staying modular — and v3 has to trade
it away to get height. See "the trade" below.

Panels are joined by 3D-printed connectors spanning the gap. Panels never share a vertex, which is
why the geometry is tractable at all.

### World conventions

- Units **centimetres**, **Y up**, X–Z are the floor axes.
- **The window / shore is at z = 0.** The wall is the plane **x = 0**.
- The wall is at low x, so it is the **left** edge of the plan view and the **right** side of the 3D
  view. This mirror trips everyone up.
- Panel local frame: width → local X, length → local Z, lit face +Y, housing back at y = −3.7.

---

## The model (schema v3)

v2 modelled the installation as 6 independent 2D column fold-chains. **v3 throws that out.** It is
now one 3D surface tiled by rigid panels; panels pitch, roll and yaw. "Row" and "column" are retired
words — what survives are **three coordinate systems**, and never conflating them is the single most
important thing in the code:

| space | what it is |
|---|---|
| **material** `(u,v)` / `(i,j)` | position on the *unrolled flat sheet*. The tiling lives here and is an exact lattice; gaps are exactly `gap` by construction. |
| **plan** `(x,z)` | the floor. The drift form `H(x,z)` is authored here. |
| **world** `(x,y,z)` | where the panels actually are. |

**The sheet is longer than its shadow.** A curved surface's plan projection is smaller than its
developed area, so material → plan is an **arc-length unroll**. Never assume a cell's world position
from its material index.

### The pipeline

```
form.js      → an authored smooth drift H(x,z): asymmetric profiles, a sheared ridge,
               and both graded edges (wall x=0, window z=0) meeting the ground with
               non-zero slope straight out of the formula
target.js    → what the panels are ASKED to be: the drift quantized into PLANAR FACETS
               on the panel lattice, plus the arc-length unroll
tiling.js    → a deterministic domino tiling: which cells get a 60×60 square and which
               get a 60×121 plate, decided by whether a rigid plate physically FITS
               the TARGET (the same surface placement seats panels on), plus any
               cells you have pinned by hand
placement.js → rigid panel placements on the target
report.js    → per-joint gaps, skew, dihedral, holonomy, surface fit, collisions
collide.js   → exact 15-axis OBB SAT
presets.js   → six drifts, each pinning one answer to the trade
```

### Why the target is allowed to be angular

Rigid flat panels **cannot be** a smooth drift: a 60 cm panel deviates from a curved target by
roughly `(30²/2)·curvature` wherever you put it. Panelizing a smooth shape therefore wedges the
joints open, drives the housings into each other, lifts the graded edges off the floor, and leaves
nowhere flat enough to lay a rigid plate.

So don't panelize a shape chosen without reference to the panels. **`form.angularity` quantizes the
drift into planar facets aligned to the panel lattice**, with facet boundaries always on cell
boundaries — so a crease only ever lands where there is already a physical joint to absorb it. Tiles
sharing a facet are exactly coplanar, and their joints stay closed.

`form.facetCells` sets how many cells share a plane: **1** gives a fold at every joint and hugs the
toe (the graded edges land); **4** gives broad planes with crisp creases and closes the joints, but
cannot hug the toe (the edges rise).

**The tiler and the placer must measure the same surface.** `solveTiling` takes the target as an
injected argument, and `placement.js` passes in the one it is about to seat panels on. This is not a
performance detail — when the two disagreed (the tiler reading the smooth form while placement used
the faceted target) the tiler was blind to faceting entirely: it placed plates where they physically
could not fit *and* refused plates that would have fitted almost perfectly.

**Consequence worth knowing.** A faceted target is locally planar by construction, so plate sagitta
collapses toward zero almost everywhere, the fit gate stops binding, and greedy placement takes
nearly every candidate — plate counts run 20–22 of 26 tiles, and the strategies stop having anything
to decide. Two levers give the choice back: **`tiling.maxPlates`** (below) and manual pinning.

### Placement modes

- **`surface-fit`** (default) — every tile is fitted to the target independently, so the
  incompatibility is shared across **all** joints. That is what the physical connectors do.
- **`chain`** — tiles hinge off one another along a spanning tree. Every tree joint is then *exact*
  (gap to 1e-8, zero skew) and **all** the error lands on the cycle-closing edges. Kept because it
  is what v2 effectively did, and the contrast is instructive: at amplitude 80, tree edges measure
  2.9e-8 cm against cycle edges at 17.5 cm.

A single hinge has one degree of freedom, so a chained tile can match the target's pitch but never
its roll. Over eight rows that error compounds and the sheet lifts off the floor. **Exact joints and
a doubly-curved surface are not compatible** — which is why `surface-fit` is the default.

---

## The trade

The model forces a question it cannot answer: **how much joint deviation is this installation
willing to build?** Three facts collide.

1. **Height costs joint deviation.** Push the crest up and the joints wedge open. Geometry, not an
   implementation limit.
2. **The nominal gap buys height and costs modularity.** On convex curvature the lit faces open
   while the **housings converge**, so a 1 cm joint runs panels into each other at only ~40 cm of
   amplitude. Widening to 2 cm removes the collisions and reaches ~95 cm — but breaks
   `2·size + gap = plateLength`, so every plate carries a real mismatch.
3. **Faceting closes the joints and lifts the edges.**

### The plate budget

**`tiling.maxPlates`** caps how many 60×121 plates the build may spend — `null` (the default) is
unlimited. It exists because the fit gate no longer constrains plate count once the target is
faceted, so without a cap the tiler simply takes nearly every plate that fits and the strategy
selector does nothing. A budget makes the strategies choose **which** plates to spend, which is the
question they are good at: measured on a faceted 6×8 sheet, all three produce *different* tilings at
every budget, each spending it exactly.

It is also the honest constraint — plates are half the panel kit and there are only so many of them.

**Pinned plates count against it**, because a plate placed by hand is still a plate you have to buy.
If the pins alone exceed the budget every one is still placed — the override contract reports rather
than refuses — and `W_PLATE_BUDGET_EXCEEDED` says so.

The presets are six chosen points, all collision-free except `crest`:

| preset | peak | worst joint | flagged | plates | worst edge clearance |
|---|---|---|---|---|---|
| `shelf` | 44 cm | 4.20 cm | 14/32 | 17/25 | **2.5 cm** |
| `closed` | 46 cm | **1.19 cm** | **0/20** | 20/22 | 8.7 cm |
| `drift` *(default)* | 61 cm | 3.54 cm | 6/36 | 21/27 | 20.6 cm |
| `dune` | 93 cm | 16.44 cm | 8/37 | 20/28 | 19.0 cm |
| `modular` | 42 cm | 1.83 cm | 1/52 | **22/26** | 9.1 cm |
| `crest` | **105 cm** | 11.54 cm | 10/27 | 19/23 | 34.5 cm |

`crest` is the only one with panel interpenetration (1 pair) — that is what it
exists to show. Every other preset is collision-free.

`shelf` is the brief-compliant one — the only preset where the graded edges really land (4 of 5
tiles down on the wall edge, 4 of 4 on the window edge). `closed` has not one joint out of
tolerance. `modular` is the only one keeping `60+1+60 = 121` exactly. `crest` reaches v2's swell
height deliberately and is deliberately **not** buildable, because the report should be able to say
no.

---

## The rules

Encoded in `core/v3/schema.js` (validation) and `core/v3/placement.js` (layout violations). The
distinction matters: **validation errors block the commit; layout violations do not.** Blocking a
violation would reject every intermediate state of a slider drag.

| rule | code | kind |
|---|---|---|
| `version` must be exactly 3 — v1 and v2 configs are **rejected, never migrated** | `E_SHAPE` | error |
| structure / ranges (sheet 4–8 × 5–10, amplitude 0–250, angularity 0–1, facetCells 1–4, gap 0–10, …) | `E_SHAPE`, `E_RANGE` | error |
| `2·size + gap` should equal `plateLength` | `W_PLATE_LENGTH` | warning |
| at least 4 plates, placed by a nameable rule | `W_FEW_PLATES` | warning |
| the assembly's centre of mass must project inside its ground-contact hull | `E_UNSUPPORTED` | violation |
| a wall-edge or window-edge tile is not grounded | `W_EDGE_FLOATING` | warning |
| a grounded edge tile is within 6° of horizontal — "grounded but **not flat**" | `W_TOE_FLAT` | warning |
| a tile's solid dips below the floor | `W_BELOW_FLOOR` | warning |
| fewer than 3 ground contacts, so there is no support polygon to test | `W_NO_SUPPORT` | warning |
| a manual override is malformed, off-grid, or two claim the same cell | `E_OVERRIDE_SHAPE`, `E_OVERRIDE_BOUNDS`, `E_OVERRIDE_CONFLICT` | error |
| a **hand-placed** plate bows further from the target than the fit tolerance | `W_PLATE_OVERRIDE_MISFIT` | warning |
| `tiling.maxPlates` is not `null` or a non-negative integer | `E_SHAPE` | error |
| more plates are placed than the budget allows (only reachable by pinning) | `W_PLATE_BUDGET_EXCEEDED` | warning |
| the plate budget is set below `MIN_PLATES` | `W_BUDGET_BELOW_MIN` | warning |

**Grounding is reported, never forced.** A rigid 60 cm panel cannot hug a curved target, so the
graded edges come within a few cm of the floor and no closer; the residual is reported per edge as a
clearance profile. And a planar facet **cannot be grounded along two intersecting lines and still be
tilted** — any plane containing two intersecting floor lines *is* the floor — so the wall/window
corner is necessarily where "both edges down" and "not flat" trade against each other.

---

## The UI

- **3D viewport** — orbit, ground grid, the **WINDOW / SHORE** line at z = 0, the translucent
  **WALL** at x = 0, a dismissible measuring box, and four **colour modes**: `type`, `gap`
  (deviation per tile), `clearance` (grounding at a glance), `facet` (makes the angular target
  legible). Plus a translucent **ghost of the target surface** and **collision highlighting**.
- **Preset bar** — the six drifts, each showing what it trades away.
- **Drift form** — every knob, with `angularity`, `facetCells` and `placement.mode` annotated
  inline, because their meaning is not guessable from a label.
- **Report** — joint deviation against tolerance, the holonomy split in `chain` mode, shape
  residual, collisions, and per-edge grounding clearance.
- **Plan view** — the material lattice showing squares and plates; row 0 at the bottom, wall
  left. **Click a square to arm it**, mergeable neighbours highlight (green `+` when the plate
  would fit, amber `~` when it would not, with the cost in the tooltip), click one to **combine
  into a plate**; click a plate to **split** it. Escape disarms. Pinned tiles are marked, and the
  report shows how many are hand-pinned versus algorithm-chosen.
- **Plate budget** — a "limit plates" toggle and a count, with the report reading `plates / budget`
  and turning red if pinning has pushed you over it.
- **CONFIG JSON** + named slots + **Export** (OBJ, one named object per panel with baked world
  transforms; and JSON).

---

## Architecture

```
src/
├── config.js                  panel dimensions + profile          (kept from v2)
├── geometry/panelGeometry.js  the panel solid                     (kept from v2 — do not touch)
├── persistence.js             localStorage autosave + named slots
├── core/v3/                   ← HEADLESS ZONE
│   ├── form.js                the smooth drift H(x,z) + analytic gradient
│   ├── target.js              faceting + the arc-length unroll
│   ├── schema.js              config, normalize, validate
│   ├── tiling.js              domino tiling; square vs plate by fit
│   ├── placement.js           surface-fit and chain placement, grounding
│   ├── report.js              joints, holonomy, fit, collisions
│   ├── collide.js             15-axis OBB SAT
│   └── presets.js             the six drifts
├── v3/                        store + components
└── utils/exporters.js         OBJ / JSON                          (kept from v2)
```

### The headless-core contract

Everything in `src/core/` **must**: use **explicit `.js` extensions** on relative imports (plain
node ESM does not resolve extensionless paths — `panel-designer` omitted them, which is exactly why
its node tests are dead); import **only `three`'s math classes**, never a scene graph, the DOM, or
the store; and be **pure and deterministic**, same input → byte-identical output.

This is what makes the whole model testable in node without a browser, and it is the single most
valuable convention in the project.

### The store contract

Every mutation runs through `commit()`: build a candidate, run `validateConfig`, commit **only if
valid**, otherwise keep the previous state and stash the errors. Derived data (`layout`, `report`)
is memoized in a `WeakMap` keyed on config identity, so it is computed once per change rather than
per frame — `solveLayout` and `buildReport` are not cheap.

---

## Testing

Plain node scripts, no framework. Each prints a pass/fail summary and exits non-zero on failure.

```bash
cd grid-designer
node tests/test-form.mjs          #   89  drift heightfield, analytic gradient
node tests/test-v3-schema.mjs     #  290  schema, normalization, every code
node tests/test-v3-target.mjs     #   40  unroll, faceting, coplanarity
node tests/test-v3-tiling.mjs     # 2054  partition, strategies, plate fit, overrides, budget
node tests/test-v3-collide.mjs    #   76  SAT, incl. a 6-axis mutation check
node tests/test-v3-placement.mjs  #  536  placement, grounding, both modes
node tests/test-v3-report.mjs     #   49  joints, holonomy, collisions
node tests/test-v3-presets.mjs    #   64  each preset delivers its claim
node tests/test-v3-obj.mjs        #   25  OBJ round-trip
node tests/test-geometry.mjs      #   50  the panel solid
node tests/test-persistence.mjs   #   46  storage, version discard
npm run build
```

**3301 checks.** Three conventions worth keeping:

- **Closed-form expectations**, derived in the test from the constants, never golden numbers. Sign
  and frame conventions are the classic bug source here and only a derivation catches them. The flat
  form (`amplitude: 0`) is the case whose answer is known exactly, and it is the control for
  everything else.
- **Non-vacuous negatives.** A check that collisions are absent is worthless without a case where
  they are present. `collide.js` carries a mutation check proving its 9 cross-product axes are
  load-bearing.
- **Presets re-measure their own claims**, so if the core math shifts, the assertions catch the
  prose going stale.
