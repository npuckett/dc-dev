# grid-designer

A three.js tool for **planning the Drop Ceiling V2 installation** as an **open folded network** of
rigid ceiling-light panels. You set an angle and a gap; the tool lays flat panels on a checkerboard
lattice — alternating ground and high — derives the angled panel on every edge between them, and
tells you **how much more you can fold before the connectors run out**.

Click the plan grid to shape it: take a cell out for a ragged edge, add one at a free face to grow,
drop a single ramp to open the network. Nothing you do moves anything else — the lattice is
generated, so every cycle closes exactly.

Read **[V4_SPEC.md](V4_SPEC.md)** for the model and **[HANDOFF.md](HANDOFF.md)** for the decision
log — including §0, which records the approach this one replaced and why.

> ### The inversion, in one line
>
> **v3 chose a form and then asked whether the panels could be it. v4 chooses folds the connectors
> can already build and lets the form be whatever those compose into.**
>
> That is not a refactor, it is the opposite direction. v3 was built out to connectors, fasteners,
> collision detection and a relaxation solver, and the honest verdict was that **it became very good
> at saying no and never acquired a way to say yes** — every lever that was not the form itself
> measured as useless or actively harmful ([HANDOFF §0](HANDOFF.md)).
>
> In v4 the gap is an **input the geometry honours exactly** rather than an outcome that gets
> measured, every joint is inside the connector envelope **by construction**, and the headline
> number is a permission: *at a 2 cm gap you may fold to 33.6°.* There is no reconciliation step, no
> relaxation, and nothing to fit.

| project | what it is | status |
|---|---|---|
| `grid-designer/` | **this** — the folded ribbon (schema v4) | active |
| `grid-designer/src/{core/v3,v3}/` | the retired surface-fit tool | kept, unmounted; see [HANDOFF §0](HANDOFF.md) |
| `panel-designer/` | first attempt: single-rooted kinematic tree | abandoned; some modules were copied out |
| `spatial-editor/` | generic r3f scene editor | unrelated prior art |
| `IO/public-viewer/` | live viewer for the **existing** installation | separate concern |

---

## The v4 model in one page

**A strip is an open chain of rigid 60 × 60 panels folded in a vertical plane**, running from the
window (`z = 0`) away toward `+z`. The first strip stands against the wall (`x = 0`).

```
unit k      1     2     3     4     5     6     7     8     9
glyph       _     /     -     \     _     /     -     \     _
role      base  rise  high  fall  base  rise  high  fall  base
tilt α      0    +θ     0    −θ     0    +θ     0    −θ     0
```

One angle **θ**, shared by every angled unit, symmetric. Because the flats sit between the angled
units, **every fold is θ rather than 2θ** — which is why this pattern buys height cheaply.

**`high` units are the branch anchors.** The sideways build runs out of units 3 and 7, angled back
down to the floor to meet the next strip and make the 3D network. v4 marks them and does not yet
build them.

Three things you can do to any unit: **remove** it (the chain does not re-solve — every other panel
stays exactly where it was), **flip** it (lit face up/down on a flat, out/in on an angled one), or
override its role.

### Why the gap is exact

The chain is a polyline on the panels' lit-face plane, and each joint steps along the **bisector** of
the two panel directions:

```
S_{k+1} = E_k + gap · normalize(u_k + u_{k+1})
```

so `|S_{k+1} − E_k| = gap` at every joint, at every angle, by construction. That single property is
what removes the entire v3 reconciliation problem.

### The number the old tool could never produce

`report.envelope` reports **`maxAngleDeg`** (the largest θ this gap admits with every joint clean)
and **`minGapCm`** (the smallest gap this angle admits), both found by bisecting on the connector's
own flag set rather than by reading a formula — because the part fouls 3–6° before the panels do,
and the honest limit is where the flags start.

At the default 2 cm gap: **33.6°**, limited by `W_FASTENER_PINCHED` — the bolt pinched at the depth
its insert sits at, not the panels and not the fold.

### The front bar is a separate limit, and this pattern needs it fixed

A flat front bar sized for the gap **bites the bezels on a concave joint past 12.37°** — the two lit
faces tilt up toward a bar that stays flat across the gap. Measured at 12.37° for gaps of 1, 1.5, 2,
3 and 4 cm: **constant**, because the overhang that collides is `frontLipCm` and does not depend on
the gap. **Widening the gap does not buy a degree.**

Half of a trapezoid wave's joints are valleys, so at any useful θ this needs an answer. It is
reported separately from the envelope (`envelope.frontBar`, `W_FRONT_BAR_FOULS_BEZEL` per joint)
rather than folded into `maxAngleDeg`, for two reasons: it would collapse the headline number from
33.6° to 12.37° and hide the limit that governs the connector, and **it is fixable in the part** — a
relief or a chamfer on the bar's underside, or a narrower bar on concave stations. A limit you can
design away does not belong in the same number as one you cannot.

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

### The panel section

Measured from `updatedPanelGeo/` and held in `config.js` as **nine caliper-measurable parameters**;
everything else — the solid, the collision boxes, the connector's grip — derives from them through
`panelSectionRings()`. Refining the panel means editing those numbers and nothing else.

```
inboard  ←──────────────────────────  0 = the outer edge

bezel peak ──┐                                    depth 0
             │╲___                                bezelDrop 0.10
   diffuser  │    ╲______________ front corner    diffuserDepth 0.363
             │                   │  outer wall, 1.10 high
             │     ______________│  back corner   outerWallDepth 1.20
             │    /   FLANGE 3.0cm — nearly flat  flangeDrop 0.10
             │   /  taper 1.62cm at ~60°
             │__/                                 overallThickness 4.10
                back plate
```

**The 3 cm back flange is the point** — it is what a connector grips, and it is the feature the
inherited 3.7 cm section did not have at all. The panels are 60 × 60 and 60 × 121 exactly, so the
`60 + 1 + 60 = 121` modularity argument survives.

One edge of every panel carries a **50 cm power supply** on the back, sitting on that flange. See
"The power supply" below — it is a real placement constraint, not a detail.

### World conventions

- Units **centimetres**, **Y up**, X–Z are the floor axes.
- **The window / shore is at z = 0.** The wall is the plane **x = 0**.
- The wall is at low x, so it is the **left** edge of the plan view and the **right** side of the 3D
  view. This mirror trips everyone up.
- Panel local frame: width → local X, length → local Z, lit face +Y, housing back at y = −3.7.

---

## The retired model (schema v3) — kept for the record

> Everything from here to "The connectors" describes the **surface-fit tool, which is no longer
> mounted**. It is documented because its measurements are still the evidence base — the connector
> envelope v4 builds inside was established here — and because [HANDOFF §2](HANDOFF.md) refers to it
> throughout. `src/core/v3/connectors.js` and `collide.js` are still live and imported by v4;
> `form.js`, `target.js`, `tiling.js`, `placement.js`, `relax.js`, `presets.js`, `report.js` and
> `schema.js` are not.

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
relax.js     → OPTIONAL: nudge those placements until every joint is inside the
               connectors' envelope, and report the ones that will not go
connectors.js→ where the 3D-printed parts go and what each has to be shaped like
report.js    → per-joint gaps, skew, dihedral, holonomy, surface fit, collisions,
               and the connector kit
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

## The connectors

**There is no substructure.** The 3D-printed parts *are* the structure, so their count and
placement is a stiffness decision rather than a detailing one.

v1 (`3dprintFiles/`) used **one part everywhere**: a Y-junction edge clamp holding a fixed **62°**,
sitting on the assembly's *outside* edges. It could be one part because every v1 joint was the same
angle. v3's are all different, and worse, **a single joint is different along its own length** —
measured across the presets, the rim-to-rim span swings up to **12.77 cm** from one end of one joint
to the other, while inside a **10 cm window** it swings 0.02–0.15 cm. That is why the parts are
short and sit in the gap: a joint no rigid part can hold becomes a handful of near-constant local
problems.

### Two pieces, bolted

Each station takes **two printed pieces**, pulled together by **three M3 countersunk bolts** into
heat-set inserts:

| | |
|---|---|
| **front bar** | a plain rectangle lying across the gap, bearing on both panels' bezels, three countersunk holes |
| **back half** | fills the gap behind it and reaches under both flanges, three heat-set inserts |

Nothing snaps or rotates on: the back half goes on from behind, the bar drops on from the front,
three bolts pull the rims between them. **Every clamp face stands `shimCm` (0.75 mm) off the panel**
— the parts never touch, and a rubber shim goes in that space on the real build.

Three M3s need **1.00 cm of gap** (the countersink head plus 2 mm walls), against 1.45 cm for one
countersunk M5. And the check is made **at depth, not at the face**: on a convex joint the gap closes
as `gap − 2·d·sin(fold/2)` and the insert sits at the bottom of the back half. That is the first
constraint in the tool that couples gap and fold.

### It grips the flange, because that is what the panel offers

Reading the measured section out of `config.js`: the front is a 1.5 cm chamfered bezel over a
recessed diffuser — a decorative surface on the lit face, and a poor thing to clamp. The back
carries a **3 cm nearly-flat flange** with open air behind it. So the flange is the load-bearing
half and the bezel lip only locates the bar. Every panel-facing dimension is traced off
`PANEL_PROFILE`, so refining the panel moves the clamp with it.

### The range is the panel's, not the part's

This is the most useful thing the tool learned about the connector. Sweeping gap against fold:

| gap | connector fouls a panel at | **the panels collide at** |
|---|---|---|
| 0.4 cm | 15° | **19°** |
| 0.6 cm | 25° | **28°** |
| 1.0 cm | 45° | **49°** |
| 1.5 cm | 71° | **77°** |

**The part is already within 3–6° of a hard geometric limit that belongs to the panels**, and no
connector dimension moves it — split depth, lip length and floor depth were all swept and none
changes the boundary. Only the shim does, by a degree. So there is nothing to expand in the
narrow-gap/large-fold corner by redesigning the part; the levers are a wider gap, less fold, or
**relaxation** (below).

The limit has a closed form: the two back corners meet at
`gap − 2·shim·cos(φ) − 2·(outerWallDepth + shim)·sin(φ) = 0`, with `φ = fold/2`.

### Collision detection is done in section, not by bounding box

Both the panel and the connector are **swept solids along the joint**, so a 2D section test is exact
for them. A bounding box is not close: a connector's box always encloses the rim it wraps, which is
why the OBB check had to exclude the two panels it grips — and therefore could never see a part
biting into its own panel. `polygonsOverlap` can, and runs at both ends of the loft.

### The kit

Every station gets a back half and a front bar cut to its own gap, then both families are **binned**
so the kit reports how few distinct parts a design needs. The universal front bar — one width
serving a whole band of gaps — is **retired for now**: it could not be made to work at the extremes.
`frontBarBand` and `solveFrontBars` survive for the day it becomes possible again.

| preset | parts | back halves | front bars | total types |
|---|---|---|---|---|
| `shelf` | 69 | 37 | 10 | 47 |
| `closed` | 51 | 8 | 4 | 12 |
| `drift` | 82 | 14 | 8 | 22 |
| `dune` | 85 | 22 | 19 | 41 |
| `modular` | 112 | 8 | 5 | 13 |
| `crest` | 65 | 20 | 16 | 36 |

**`minPerJoint` defaults to 2 for a structural reason**: a joint held by one part is free to rotate
about it, and with no substructure that is a real degree of freedom.

### The power supply

A **50 cm box on the back of one 60 cm edge** of every panel, sitting on the flange the connector
grips. Its top face is at the flange's outer depth while the flange falls away, so it is **flush at
the panel edge and at most 1 mm proud** at the flange's inner edge — set into the housing, not a box
on top of it.

So a **0.8 mm relief** in the back half's lip clears it and a powered edge carries a connector after
all. What it costs instead is reported rather than hidden: the lip then bears on the **supply
housing** rather than the panel frame (`W_BEARS_ON_POWER_SUPPLY`), and whether a driver housing is
something to clamp against is a hardware question this model cannot answer. `supplyMode: 'block'`
keeps the stricter reading measurable — under it, 6–25 joints per preset carry no connector at all.

Which edge is powered is a **global convention** (`connectors.powerEdge`). Panel orientation is a
real design freedom the tool does not model yet.

### Planned: a threaded boss

A separate locking screw, on top of the three bolts. Not built. The sizing rule, measured:

> **`hookCm`-style standoff is the fold budget; the lips are free.**

The jaw can reach further inboard — 13.5 mm of the panel's 25 mm flange is spare, enough for M8 —
without costing any fold capacity. **The open question is what the screw bites into**, and it is not
a geometry question: threading into the flange means modifying existing LED fixtures. See HANDOFF
§4.7.

---

## Relaxing into the envelope

Because the connector cannot be redesigned out of its hard cases, a joint outside the envelope is a
**placement** problem. `placement.relax` is a spring relaxation that moves the panels until every
joint has room for its fastener and stays under the fold the panels allow.

**It moves the placements, never the form.** `form` is authored intent; shifting it would quietly
redesign the drift. `surface-fit` already fits each tile independently — this does the same thing
globally, every joint pulling on its two tiles and every tile pulling back toward where the surface
put it. The tiling is untouched too.

**Deterministic**: Gauss–Seidel, a fixed iteration count, joints in adjacency order. No convergence
test and no early exit, for the same reason `placement.js` runs exactly three fixed-point iterations.

**It can fail, and says so.** `unresolved` names every joint still outside and what it is short by;
`moved` reports how far each tile was shoved, because a design that only works after a lot of
shoving is not the design that was authored. Neither is a veto.

Measured: `modular` goes from **25 joints outside the envelope to 0**, for 0.3 cm of movement at
worst and 0.01 → 0.02 cm of shape residual. `closed` is already inside and is not moved at all.

Off by default — it changes the shape the panels make, and that should be a deliberate choice.

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
| a connector's section closes on itself | `W_CONNECTOR_INFEASIBLE` | violation |
| a piece bites into a panel, measured **in section** | `W_BACK_HALF_FOULS_PANEL`, `W_FRONT_BAR_FOULS_PANEL` | violation |
| the two panels themselves collide at a joint | `W_PANELS_COLLIDE_AT_JOINT` | violation |
| the gap is too narrow for the fastener, **measured at depth** | `W_FASTENER_PINCHED` | violation |
| a connector fouls a panel it does **not** grip, or another connector | `W_CONNECTOR_CLASH` | violation |
| the flange lip bears on the power supply housing, not the panel frame | `W_BEARS_ON_POWER_SUPPLY` | warning |
| the gap is too narrow to fit a part into / too wide for the spine to stay stiff | `W_CONNECTOR_PINCH`, `W_CONNECTOR_SPAN` | warning |
| a part wedges too much along its own length — the number `lengthCm` controls | `W_CONNECTOR_TWIST` | warning |
| a joint carries one part, so it is a hinge rather than a fixture | `W_JOINT_SINGLE_CONNECTOR` | warning |
| a joint is too short to seat its parts at full length (shortened, not refused) | `W_CONNECTOR_CROWDED` | warning |
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

Same shell as before — top bar, fixed left control column, 3D viewport — rebuilt against the v4
model in `src/v4/`. The v3 panels are still on disk and no longer mounted.

- **3D viewport** — orbit, ground grid, the **WINDOW / SHORE** line at z = 0, the translucent
  **WALL** at x = 0, a dismissible measuring box, and four **colour modes**: `role` (the pattern
  made visible), `fold` (worst joint fold touching each panel), `flip` (so a flipped panel is
  unmistakable), `flags`. The default camera is a three-quarter from in front of the window, which
  is what keeps the wall on the right and the window at the bottom; **orbit round to +X for the
  profile**, which is the view the fold pattern reads in.
- **The ribbon** — panel count and **angle** are the two knobs you actually drive, and both carry
  their limits inline: the angle slider says how much headroom the connector leaves *and*, in its
  own line, when you have gone past what a flat front bar can lie across. Plus gap, phase, strips,
  the wall/window offsets and "sit on the floor".
- **The glyph readout** — `_/-\_/-\_`, the clearest possible statement of what the strip is.
- **Units table** — one row per panel, unit 1 at the window: **remove**, **flip** and a role
  override. The flip toggle is labelled with the word that fits the unit — **up/down** on a flat,
  **out/in** on an angled one — and the branch anchors are marked `↔`. Hovering a row highlights
  that panel in 3D. Removed units dim rather than vanish.
- **Metrics** — the measuring box **plus the per-axis detail it hides**: per strip (plan run,
  height, developed length, and the compression ratio the folding bought), per row across the
  strips, and per unit along the chain (z from/to, run, y from/to, rise).
- **Report** — the **envelope first**: `maxAngleDeg`, `minGapCm`, headroom in both, and what
  actually stops you spelled out in words. Then the front bar's own separate limit, the joint table
  (span, fold, convex/concave, stations, flags), collisions and warnings.
- **CONFIG JSON** + named slots + **Export**:
  - **OBJ** — the assembly, world transforms baked, **grouped for assigning materials**:
    - `diffuser_NNN_<id>` — **one object per panel.** The lit faces are kept individual precisely so
      each can be given its own brightness. Zero-padded so they sort in build order.
    - `frame` — every panel's housing, merged. One material.
    - `connectors` — every printed part, back halves and front bars, merged.
    - `power_supplies` — every driver box, merged. Absent entirely under `powerEdge: none`.

    The diffuser/frame cut is not a new decision: `panelGeometry.js` already emits the solid as two
    material groups and the viewport has always drawn it as two materials. Every object carries a
    named `usemtl`, so the grouping survives import.
  - **Connector STL** — one of each unique type, both families, **in millimetres**, laid flat for
    printing (largest face down, no layer boundary across a gripping face).
  - **Connector manifest** — how many of each to run, where each sits on the plate, and what each is
    forced to absorb by not getting its own exact geometry. The STL cannot say that `P00` is needed
    46 times; this is the document that goes with it.
  - **JSON** — the config, the single source of truth.

---

## Architecture

```
src/
├── config.js                     panel dimensions + profile       (kept from v2)
├── geometry/panelGeometry.js     the panel solid                  (kept from v2 — do not touch)
├── geometry/connectorGeometry.js the connector solid — lofts the profile
├── persistence.js                localStorage autosave + named slots
├── core/v4/                      ← HEADLESS ZONE — THE ACTIVE MODEL
│   ├── schema.js              config v4, normalize, validate
│   ├── chain.js               roles, fold kinematics, world placement, OBBs, bounds
│   ├── connectors.js          v4 stations → v3's part machinery
│   └── report.js              joints, the ENVELOPE, collisions, metrics
├── v4/                        store + components (mounted by main.jsx)
├── core/v3/                   ← HEADLESS ZONE — retired except connectors + collide
│   ├── connectors.js          ★ the part, and the joint feasibility envelope — LIVE, v4 imports it
│   ├── collide.js             ★ 15-axis OBB SAT — LIVE, v4 imports it
│   ├── form.js                the smooth drift H(x,z) + analytic gradient
│   ├── target.js              faceting + the arc-length unroll
│   ├── schema.js              config, normalize, validate
│   ├── tiling.js              domino tiling; square vs plate by fit
│   ├── placement.js           surface-fit and chain placement, grounding
│   ├── relax.js               spring relaxation into the connector envelope
│   ├── report.js              joints, holonomy, fit, collisions, the connector kit
│   ├── schema.js              config v3, normalize, validate
│   └── presets.js             the six drifts
├── v3/                        store + components (kept, unmounted)
├── utils/exporters.js         OBJ / JSON                          (kept from v2)
└── utils/connectorExport.js   printable STL plate + manifest (MILLIMETRES)
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

# --- v4, the active model -------------------------------------------------
node tests/test-v4-schema.mjs     #  228  config v4, normalization, every range code
node tests/test-v4-lattice.mjs    #  939  the checkerboard, pitch, closure, handedness, editing
node tests/test-v4-connectors.mjs #  189  stations, the fold sign, the flip mismatch
node tests/test-v4-report.mjs     #  272  the envelope boundary, the front bar, corners, metrics
node tests/test-v4-obstacles.mjs  #   37  the column: extents, hits, clearance, report-not-veto
node tests/test-v4-obj.mjs        #   27  the OBJ grouping, and that the split loses no triangle

# --- v3, retired but still green ------------------------------------------
node tests/test-form.mjs          #   89  drift heightfield, analytic gradient
node tests/test-v3-schema.mjs     #  290  schema, normalization, every code
node tests/test-v3-target.mjs     #   40  unroll, faceting, coplanarity
node tests/test-v3-tiling.mjs     # 2054  partition, strategies, plate fit, overrides, budget
node tests/test-v3-collide.mjs    #   76  SAT, incl. a 6-axis mutation check
node tests/test-v3-placement.mjs  #  518  placement, grounding, both modes
node tests/test-v3-report.mjs     #   49  joints, holonomy, collisions
node tests/test-v3-presets.mjs    #   64  each preset delivers its claim
node tests/test-v3-obj.mjs        #   25  OBJ round-trip
node tests/test-geometry.mjs      #   83  the panel solid
node tests/test-persistence.mjs   #   47  storage, version discard
node tests/test-v3-connectors.mjs         #  168  stations, frames, the fold sign, the power supply
node tests/test-v3-connector-geometry.mjs #   79  both sections, both solids, the grip, the render sweep
node tests/test-v3-connector-report.mjs   #  187  flags, the kit partition, clashes
node tests/test-v3-connector-export.mjs   #   62  plate, STL bytes, manifest, assembly OBJ
node tests/test-v3-relax.mjs              #   71  determinism, placements-only, and that it can fail
npm run build
```

**5594 checks across 22 suites.** Three conventions worth keeping:

- **Closed-form expectations**, derived in the test from the constants, never golden numbers. Sign
  and frame conventions are the classic bug source here and only a derivation catches them. The flat
  form (`amplitude: 0`) is the case whose answer is known exactly, and it is the control for
  everything else.
- **Non-vacuous negatives.** A check that collisions are absent is worthless without a case where
  they are present. `collide.js` carries a mutation check proving its 9 cross-product axes are
  load-bearing.
- **Presets re-measure their own claims**, so if the core math shifts, the assertions catch the
  prose going stale.
