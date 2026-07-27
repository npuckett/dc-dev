# grid-designer — handoff & decision log

Written 2026-07-26, covering the **v3 pivot** session. Read **[README.md](README.md)** first
for how the tool works and **[V3_SPEC.md](V3_SPEC.md)** for the model; this document records **why it
is the way it is**, what was tried and rejected, and what is still open.

The v2 handoff's "durable findings" section is superseded by §2 here, but its central insight
survives unchanged and is still the reason any of this is tractable:

> Don't solve rigid origami. Place panels deterministically and **measure** what the connectors have
> to absorb.

Branch `v3-drift-tiling`. All suites green (3301 checks across 11 suites), build clean, app verified
in the browser.

---

## 1. What the pivot was

v2 modelled the installation as **6 independent 2D column fold-chains** — each column a strip at
fixed x, folding window → back. The brief changed: it is now **one 3D surface tiled by rigid
panels**, a snow drift, where panels pitch, roll and yaw, and where the choice between a 60×60 square
and a 60×121 plate is made by **a tiling algorithm** rather than by hand.

Built in order, one commit each (the messages carry the detail):

| commit | what |
|---|---|
| `45c7e05` | V3 spec |
| `4fda7c4` | localStorage persistence + named slots (P0) |
| `9b811b1` | the drift form — parametric heightfield (P1) |
| `ac9281a` | schema v3 + the tiling that decides square vs plate (P2) |
| `b948321` | exact OBB collision detection |
| `c86fd2a` | place the tiles on the drift surface (P3) |
| `a0e543b` | let the target be angular, so the panels can BE the surface |
| `e225c1c` | the joint report (P4) |
| `928b00f` | the v3 UI (P5) |
| `4c895bc` | drift presets, each pinning one answer to the trade (P6) |
| `8889ba4` | remove the v2 model |
| `27181d9` | README + HANDOFF rewritten for v3 (P7) |
| `56f6d46` | make the tiler measure the surface the panels actually sit on |
| `748e537` | manual square/plate control — combine and split in the plan view |

---

## 2. Durable findings

Things learned that hold regardless of what the tool becomes.

### 2.1 Rigid panels cannot be a smooth surface, and the target should stop pretending

A 60 cm rigid panel deviates from a curved target by roughly `(30²/2)·curvature` **wherever you put
it**. Panelizing a shape chosen without reference to the panels therefore produces, all at once:
joints wedged open, housings interpenetrating, the graded edges hovering off the floor, and no region
flat enough to lay a rigid plate.

The fix is not a better solver. It is to **let the target be angular** — quantize it into planar
facets aligned to the panel lattice, so the panels *are* the surface instead of approximating it,
with creases only where a physical joint already exists to absorb them. This was the user's
insight mid-session and it is the most important idea in v3.

### 2.2 Exact joints and a doubly-curved surface are incompatible

A rigid tile hinged off a placed neighbour across a shared edge has **exactly one degree of
freedom**. So it can match the target's pitch but **never its roll**. Over eight rows the roll error
compounds and the sheet lifts clean off the floor — measured 28 cm on the graded edges.

Hence `surface-fit` (share the misfit across all joints, which is what connectors do) rather than
`chain` (make tree joints exact and dump everything on the cycle-closing edges). Measured at
amplitude 120: shape residual 0.00 cm vs the chain's 9.83 cm; worst graded-edge clearance 9.0 cm vs
28.7 cm.

`chain` is kept because the contrast is the honest way to show what the pivot bought, and because a
v2 column chain **is** this algorithm on a 1-dimensional graph.

### 2.3 The three-way trade

**How much joint deviation is the installation willing to build?** Nothing in the model answers this.

1. **Height costs joint deviation.**
2. **The nominal gap buys height and costs modularity.** On convex curvature the lit faces open while
   the **housings converge**, so a 1 cm joint collides at only ~40 cm of amplitude; 2 cm removes the
   collisions and reaches ~95 cm. But `60+1+60 = 121` exactly is what makes a plate a true drop-in,
   and any wider gap breaks it permanently. The hardware plate is a standard 121 cm size, so the
   mismatch is real, not a config error.
3. **Faceting closes the joints and lifts the edges.** Broad facets cannot hug the toe.

The six presets are six chosen points. See README's table.

### 2.4 The sheet is longer than its shadow

Material → plan is an **arc-length unroll** (`dx/du = 1/√(1+(∂H/∂x)²)`). Sampling the target at a
tile's *plan* position instead of its *material* position makes the sheet fall short of the wall and
ride up the slope — a natural-looking mistake that cost real debugging time. Both graded edges are
fixed points of the map wherever the surface is zero along them, which is why they land where the
brief says.

### 2.5 A planar facet cannot be grounded along two intersecting lines and still be tilted

Any plane containing two intersecting floor lines **is** the floor. So the wall/window corner is
necessarily where "both edges touch the ground" and "grounded but not flat" trade against each other.
Every other part of both edges can be grounded and pitched; the corner has to give. This is geometry
and constrains the brief itself.

### 2.6 Facet planes must be fitted to corners, not interiors

Adjacent facets share two corners. Fit to the corners and neighbours meet in a **crease**; fit to the
interior and they meet in a **step** that the straddling panels swallow as an open joint. Interior
fitting measured **worse than no faceting at all** — which is what sent us looking. Exact continuity
would need the four corners coplanar, which in general they are not; that residual is what remains.

### 2.7 A drift shorter than its sheet is a trap

`H = 0` outside the footprint, so a footprint shorter than the sheet leaves a slope discontinuity at
the boundary that the straddling tiles cannot follow: 9.8 cm worst joint deviation against 2.67 cm
once matched. An omitted `form.footprint` now derives from the sheet.

### 2.8 The tiler and the placer must measure the same surface

`tiling.js` decided square-vs-plate by measuring sagitta against the SMOOTH form through
`materialToPlanApprox`, while `placement.js` seated the panels on the FACETED target via the real
arc-length unroll. The tiler was therefore blind to `angularity` and `facetCells` — it reported an
identical sagitta for every faceting setting — and was wrong in **both** directions at once.
Measured at amplitude 100, tiler claiming 1.70cm throughout:

- true sagitta **5.26cm** at facetCells 2 — 2.6x over tolerance, placing plates that cannot fit
- true sagitta **0.17cm** at facetCells 4 — refusing plates that would have fitted almost perfectly

`solveTiling(config, target)` now takes the target as an argument and placement injects its own.
Generalises to: **any two stages that reason about "the surface" must be handed the same object**,
not each construct their own idea of it.

Sagitta is measured in 3D against the chord, not in a flattened (distance, height) plane — the
unroll makes plan spacing between samples non-uniform, so the 2D version understated the bow.

### 2.9 A faceted target makes the fit gate go slack

Direct consequence of §2.1 and §2.8, and it changes what the tiling strategies are for. A faceted
target is locally planar by construction, so sagitta collapses toward zero almost everywhere, nearly
every candidate domino passes the fit gate, and greedy placement takes them all: plate counts run
20-22 of 26 tiles and **all three strategies produce identical tilings**, because none of them has to
choose. That is a real property, not a bug — but it means the strategy selector does almost nothing
at default settings.

Two levers give the choice back, and both are now built: manual pinning (`tiling.overrides`) and a
plate budget (`tiling.maxPlates`, `null` = unlimited). The budget binds AFTER the strategy has
ranked the survivors, which is the whole point — the strategy decides *which* plates to spend.
Measured on a faceted 6×8 sheet, all three strategies produce different tilings at every budget and
each spends it exactly. Pinned plates count against the budget, since a plate placed by hand is
still a plate you have to buy; pins over budget are all placed and raise
`W_PLATE_BUDGET_EXCEEDED`.

### 2.10 A forced plate must be placed, not refused

v2 learned this (its §3.6) and v3 re-learned it: refusing every physically awkward merge makes the
feature unusable, because in a designed profile almost every merge is awkward. So a manual override
that does not fit is **placed anyway** and reported — `W_PLATE_OVERRIDE_MISFIT` with the measured
sagitta. An override is the user overruling the algorithm on purpose; the tool's job is to state the
consequence, not to veto.

Note the v2 asymmetry does NOT port literally. "Split does not restore what the merge changed" had
meaning in v2 because merging coerced hinge geometry. v3's surface-fit placement has no per-tile
coercion to give back, so split instead **pins both cells as squares** — otherwise the algorithm
simply re-creates the plate on the next solve. Same spirit, different mechanism.

### 2.11 Site facts (unchanged from v2, still unresolved)

- The existing installation is 12 panels in a Toronto storefront window; `IO/DROPCEILING_STORY.md` is
  the best overview.
- **Unresolved inconsistency:** `IO/world_coordinates.json` and `IO/lightController_osc.py` disagree
  about subpanel positions and angles (±30° with one set of offsets vs ±22.5° with another). The
  public viewer mirrors the controller. Nothing in grid-designer depends on either, but **if V2
  planning ever has to reconcile against V1 as-built, resolve this first.**

---

## 3. Rejected approaches (with measurements)

- **Spanning-tree placement as the default** — §2.2. Elegant, exact on tree edges, and produces the
  wrong shape. Demoted to a comparison mode.
- **Sampling the target in plan coordinates** — §2.4.
- **Least-squares facet planes over facet interiors** — §2.6.
- **Additive / translational height fields for exact planar quads.** On a rectangular plan lattice,
  all-quads-planar ⟺ `h(i,j) = f(i) + g(j)`. That family **cannot** be zero along two intersecting
  edges and still be a mound, so it is incompatible with the brief's grounded edges. Recorded because
  it is the obvious next idea and it does not work.
- **`gapTolerance` as a buildability gate.** v2 shipped presets with 49 cm worst deviation and
  flagged 40/74 joints; the report is information, not a veto. Only collisions and support are hard.

---

## 4. Open questions

1. **How much joint deviation is acceptable?** The tool now measures it precisely and cannot decide
   it. Every preset is a guess at the answer. **This is the top question for the user.**
1b. **What IS the plate inventory?** `tiling.maxPlates` now exists and restores the strategies to
   usefulness (§2.9), but nothing in the repo records how many 60×121 plates the build actually has.
   That number would turn the budget from an exploration knob into a constraint.
2. **Is a wider joint acceptable?** Going 1 cm → 2 cm is what unlocks height, at the cost of the
   plate's exact modularity. Needs a connector-design answer.
3. **A foldable (planar-quad) target is the real next step.** Faceting with independent planes still
   leaves residual gaps. A true PQ mesh — planar faces meeting exactly along shared edges — would let
   the surface be **as tall as you like** with joints staying near nominal, because the joints become
   the folds. On a rectangular lattice that forces the additive family (§3, ruled out), so it needs
   the fold lines' **plan positions** to move — a real optimization, and the highest-value remaining
   work.
4. **How deep can the installation be?** Still unrecorded anywhere in the repo. Presets run 433–501
   cm deep.
5. **Is the wall structurally usable for support?** v2 asked this and it is still unanswered; v3 does
   not currently use the wall for support at all.
6. **Reconcile V1 as-built geometry** if V2 planning needs it — §2.8.

---

## 5. Next steps / known gaps

### 5.1 Not built

- **Connector design and the per-panel connection network.** Deferred deliberately — the user was
  explicit that this comes after. v3 measures what the connectors must absorb; it does not design
  them. `panel-designer/src/utils/exporters.js` has an `exportConnectorSpec` worth cribbing, and the
  joint report already computes everything it needs.
- **The planar-quad target** — §4.3.
- **Prompt-driven generation.** `core/v3/schema.js`'s doc comment is written for an LLM audience for
  exactly this. Never wired up; no API calls anywhere in the tool.
- **Light behaviour / animation.** The V2 concept is an *agentic* body of water driven by sidewalk
  data; this tool only plans static physical configurations.
- **Deploy.** `.github/workflows/static.yml` only covers `IO/public-viewer/`.

### 5.2 Minor known quirks

- For strongly skewed joints `gapMid` can be smaller than `gapMin` — correct (midpoints can be closer
  than endpoints) but don't present the three as an ordered triple.
- The 3D canvas can appear blank for a beat on first paint. It is screenshot-vs-first-frame timing,
  not a bug; it renders within ~2 s.
- Changing `sheet.cols`/`rows` in the UI does **not** rescale `form.footprint`, because the store's
  config always carries a concrete footprint after the first normalize. Growing the sheet therefore
  extends flat tiled material past the drift rather than stretching the drift. Defensible, but it
  surprises people — consider a "refit footprint to sheet" action.
- `dist/` is committed from a v2 build and is stale.
- The 3D viewport's default camera starts low and close; orbit out to read the drift. Not tuned.
- Changing `sheet.cols`/`rows` does not rescale `form.footprint` (see above), so a manual pin set
  made at one sheet size will not mean the same thing at another — overrides are keyed on `(i, j)`.

---

## 6. Working practice

Ran as **Opus 5 orchestrating, Sonnet 5 coding**: each work package delegated with a precise spec,
then independently verified by the orchestrator — re-running suites, probing core functions directly
rather than reading summaries, and driving the real UI in a browser.

This caught things worth catching, in both directions. Subagents found three real spec errors of
mine (an inverted SAT mutation-check expectation; a maximal-domino-packing artifact; missing range
constants and `E_RANGE` checks on the two faceting knobs). Independent probing caught two of my own
model errors that the tests as written would have passed: sampling the target in plan coordinates,
and a **left-handed basis** for plates whose long axis runs along `u`, which corrupted every
collision box and made a perfectly flat grid report 21 interpenetrating pairs.

**Verify the numbers, not the narrative** — and prefer structural proof (no diff in `core/`) over
output comparison where available.

---

## 7. Picking this up in a new thread

Read in this order: **README.md** → **V3_SPEC.md** → this file → the doc comment at the top of
`src/core/v3/target.js` (why the target is faceted, which is the crux) → `src/core/v3/placement.js`'s
header (frames and handedness).

Then run the suites to confirm the tree is green, and `npm run dev --prefix grid-designer` to look at
it. Load the `shelf` preset to see the brief satisfied, then `crest` to see the report say no. Click
two adjacent squares in the plan view to combine them and watch the report react.

§4.1, §4.1b and §4.3 are the queue. The user has flagged interface tuning as the next thing they want
to work through themselves.
