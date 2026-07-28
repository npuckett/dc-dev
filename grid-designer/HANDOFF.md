# grid-designer — handoff & decision log

Written 2026-07-26/27, covering the **v3 pivot**, the connector work that followed, and the **v4
pivot away from both**. Read **[README.md](README.md)** for how the tool works and
**[V4_SPEC.md](V4_SPEC.md)** for the current model; this document records **why it is the way it
is**, what was tried and rejected, and what is open.

Branch `v3-drift-tiling`. All suites green (5594 checks across 22 suites), build clean, app verified
in the browser.

> ## STATUS, 2026-07-27 — v4 is the model; §0 below is now history
>
> The surface-fit direction described in §0 was retired and **replaced**, in the same session, by
> the folded ribbon: [V4_SPEC.md](V4_SPEC.md), `src/core/v4/`, `src/v4/`. §0 is preserved unedited
> because it is the argument for the replacement, and §2's findings still hold. **§8 is the v4
> record** and **§9 is the network** — the ribbon generalised to a 2-D lattice, which is what the
> tool now builds. Read §9 first.
>
> What is live from v3: `core/v3/connectors.js` (the part and its feasibility envelope) and
> `core/v3/collide.js`. Everything else in `core/v3/` and all of `src/v3/` is kept, tested and
> unmounted.

---

## 0. HISTORY — why the surface-fit direction was retired

**Everything in this repository works and is tested. The approach it embodies is being left
behind.** Read this section before anything else; the rest of the document is the evidence.

### The verdict

v3's workflow is: **author a drift surface → tile it with rigid panels → measure the damage.** After
building it out to connectors, fasteners, collision detection and a relaxation solver, the honest
summary is:

> **The tool became very good at saying no, and never acquired a way to say yes.**

On a typical authored drift (99 cm over a 374 cm footprint, a 2.9 cm joint) it reports: worst joint
deviation 19 cm against a 1.5 cm tolerance, 58 of 60 joints flagged, 5 panel collisions, graded edges
27 cm off the floor, 32 of 60 joints wider than any connector can span, and 0 plates placeable. Every
one of those numbers is correct. None of them comes with a route to a design that works.

### Every lever that is not the form measured as useless or harmful

This is the finding that justifies the pivot, and it was arrived at by measurement, not taste:

| lever | result |
|---|---|
| redesign the connector | it fouls a panel only **3–6° before the panels collide with each other**; no connector dimension moves that boundary (§2.20) |
| relax placements to widen tight joints | **works** — 25 joints → 0 on `modular` for 0.3 cm of movement. This is the one thing that worked. |
| relax placements toward the nominal gap | collisions 5 → 15, shape residual 0.30 → 5.07 cm (§2.24) |
| relax placements to close over-wide joints | diverges: 204 cm of movement, deviation 19 → 154 cm, residual 0.30 → 28.53 cm (§2.24) |
| relax placements to ground the floating edges | no improvement; pulls panels off the target (§2.24) |
| push colliding panels apart | 40 → 35 collisions for **4× the shape error**; nothing at all at worst (§2.25) |
| change the form | deviation 19.07 → 3.61 cm, collisions 5 → 0, floating 27.3 → 14.5 cm |

The last row is the only lever with real authority, and the tool has no way to use it. It can grade a
form; it cannot propose one.

### Why that is structural, not a missing feature

A rigid 60 cm panel deviates from a curved target by roughly `(30²/2)·curvature` **wherever you put
it** (§2.1). Panelizing a surface chosen without reference to the panels therefore produces joints
wedged open, housings interpenetrating and edges off the floor — all at once, and none of it fixable
downstream. Faceting (§2.1) was the right response and bought a great deal, but it treats the
symptom: the surface is still authored first and reconciled afterwards.

**The next direction inverts that** — build from what the panels and their connectors can actually
do, and let the form be whatever that composes into. §5.3 is an inventory of what carries over.

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
| `9b53b06` | plate budget — give the tiling strategies something to decide |
| `461ca25` | decide where the connectors go (P9) |
| `d4956f1` | the connector solid, gripped off the real rim (P10) |
| `c4e32a4` | the connector kit, and what each part is forced to absorb (P11) |
| `a2ec09a` | show the connectors and let them be tuned (P12) |
| `e4d8195` | printable STL + manifest, and the docs (P13) |
| `a516bf9` | size the connector section for the planned locking screw |
| `96fccff` | the measured panel, parametrically, and the power supply |
| `53c5878` | the two-piece bolted clamp (P14) |
| `f5715e3` | shim clearance, and render both pieces |
| `59cb110` | section-level collision, per-station bars, footprint controls |
| `2315d7f` | relax the placements into the connector envelope (P15) |
| *this* | docs brought up to date with the two-piece clamp and the relaxation |

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

### 2.12 A short connector sees almost none of its joint's variation

The finding the whole connector package rests on, and it is a measurement rather than an argument.
Along one joint the rim-to-rim span swings by up to **12.77 cm** (`dune`). Inside a **10 cm window**
it swings **0.02–0.15 cm on average**, worst **2.12 cm**. Measured as a three-way comparison so the
claim cannot pass vacuously: on `dune`, a 10 cm part sees 1.81 cm, a 30 cm part 3.95 cm, the whole
joint 12.77 cm.

So a joint no rigid part can hold becomes a handful of near-constant local problems. This is §2.1
one level down — stop asking a rigid thing to be a curved thing — applied to the hardware instead of
the surface. It generalises: **when a rigid part cannot match a varying condition, shorten the part
before improving the part.**

### 2.13 The panel rim is a wedge, not a plate — SUPERSEDED, and the lesson is the point

v1's connector used a parallel-sided channel (9.5 mm wide, 8.5 mm deep; its 28.5 mm overall is
exactly jaw + slot + jaw). That assumes the rim is a parallel-sided plate. Reading the profile out of
`config.js`: below the 1.0 cm outer wall the housing **tapers inward at 0.675 per cm**, so the
undercut is a wedge that opens with depth — 1.7 mm at 2.5 mm in, 5.7 mm at 8.5 mm. A parallel jaw of
any useful reach cuts into the taper.

**This finding was itself derived from a wrong section, and that is why it is kept.** The taper it
describes exists, but the section it was read from had NO BACK FLANGE and was 3.7cm thick; the real
panels (`updatedPanelGeo/`) are 4.1cm with a 3cm flange, and the connector now grips that flange
rather than hooking the taper at all.

The durable lesson survives twice over: **v1's numbers are a reference, not a spec** — and so were
the four inherited profile numbers in `config.js`, which had no recorded provenance and turned out
to describe a different panel. Anything geometric with no measurement behind it should be treated as
a guess until a caliper says otherwise. §2.19 is what replaced this.

### 2.14 Twist needs no term in the connector geometry

Both tiles are rigid planes, so along a joint the fold is **constant** and the span varies
**linearly**. A part is therefore a loft between two cross-sections differing only in span, and a
twisted joint is simply one whose two ends want different spans. Recorded because the obvious
implementation — a twist parameter rotating one end relative to the other — is both more code and
wrong.

### 2.15 The two ways to make a connector impossible do not co-occur on a drift

The part's cross-section self-intersects when the two hooks, swinging under the joint as it folds,
meet: a 0.4 cm gap holds ±14°, 1.0 cm holds ±36°, 2.5 cm holds anything. Across all six presets —
464 stations — **not one is infeasible**, and not by luck: *where the surface folds hard the gap has
already wedged open, and where the gap is tight the surface is nearly flat.*

That correlation is a property of these forms, **not a law**, so the rule is kept and tested against
a synthetic station instead of being deleted as unreachable. The general practice: a rule with no
reachable test is a rule that quietly stops working, so give it a testable seam
(`connectorStationFlags`) rather than burying it in the consumer.

### 2.16 A connector's bounding box always overlaps the panels it grips

The channel closes around the rim and the slot is a void *inside* the box, so an OBB test reports a
collision for every part against both of its own panels — 224 pairs on `modular`. Those two are
excluded by construction, and what the clash rule then detects is a part fouling a **third** panel or
another part. The test proves the exclusion is load-bearing by removing it and watching every grip
pair light up.

Generally: **a bounding volume is the wrong primitive for a part designed to interlock.** It is
still the right one for "does this foul something it should be nowhere near".

### 2.17 The connector's fold capacity is set by hook DEPTH alone — SUPERSEDED by §2.20

Found while checking whether the section could carry the planned threaded boss, and it is the useful
kind of answer: the two hooks meet at their **mouth** corners as a joint folds, and the mouth sits at
`-(outerThickness + hookCm)` no matter how far the jaw reaches inboard or how thick it is. Measured
across 0.6/1/1.5/2 cm gaps:

| change | fold boundary |
|---|---|
| grip 8.5 → 17 mm | **unchanged** (21.5 / 36 / 55.5 / 77°) |
| jaw 9.5 → 15 mm | **unchanged** |
| back wall 3 → 6 mm | **unchanged** |
| hook 6 → 3 mm | rises to 26.5 / 45 / 70 / 90° |
| hook 6 → 10 mm | falls to 17 / 28.5 / 44 / 60° |

So **`gripCm`, `jawCm` and `wallCm` are free; `hookCm` is the fold budget.** The headroom figure —
13.5 mm of spare flange, enough for an M8 boss — still holds and is still the answer for the planned
locking screw.

**The budget claim does not.** It was measured on the one-piece hook clamp, which no longer exists.
On the two-piece design the fold limit contains no connector dimension at all — see §2.20.

The method is what to keep: **when a section has several dimensions and one hard limit, find which
dimensions the limit is actually a function of before designing against all of them.** Three of four
were free then; all of them are free now.

Generally: when a section has several dimensions and one hard limit, find which dimensions the limit
is actually a function of before designing against all of them. Three of these four turned out to be
free, which is a much better position than the intuition that everything trades.

### 2.19 An inherited constant with no provenance is a guess

`config.js` carried four panel-profile numbers copied out of `panel-designer` with nothing recorded
about where they came from. They described a panel **0.4cm too thin, with a 2.5cm flat lip that is
really a 1.5cm chamfered bezel, and — the one that mattered — with NO BACK FLANGE at all**: its
taper began at the outer wall, where the real panel has 3cm of flat material.

Everything built on top inherited the error. The first connector was a wedge hook engaging a taper
that does not exist where it was modelled, mounted on the front because the (wrongly modelled) back
looked tighter. It was not a design mistake; it was a measurement mistake wearing a design's clothes.

Two things follow, and both are now standing practice here:

1. **Numbers that face the physical world get a provenance line or a caliper.** `PANEL_PROFILE` now
   says where every value came from.
2. **Make them parameters, not shapes.** The section is nine measurable numbers and
   `panelSectionRings()`; the solid, the collision boxes and the connector grip all derive from it,
   and `test-geometry.mjs` derives its expectations from the same function — including a closed-form
   volume summed frustum by frustum — so neither the mesh nor the test can quietly agree with a
   stale constant.

### 2.20 The connector's working range belongs to the PANEL

Swept exhaustively over gap × fold: the connector fouls a panel only **3–6° before the two panels
collide with each other anyway** (15° vs 19° at a 0.4cm gap, 45° vs 49° at 1cm). Split depth, lip
length and floor depth were all varied and **none of them moves the boundary**; only the shim does,
by one degree.

So "the connector struggles with extreme joints" had a false premise. There is nothing to expand by
redesigning the part — at the limit it is the panels' own back corners meeting, and the closed form
is `gap − 2·shim·cos(φ) − 2·(outerWallDepth + shim)·sin(φ) = 0`.

Generalises to something worth doing before any optimisation: **measure whether the thing you are
about to improve is actually the binding constraint.** Here it was within a few degrees of a limit
belonging to a part nobody was proposing to change.

### 2.21 A bounding volume is the wrong primitive for a part designed to interlock

Restating §2.16 now that it has been acted on. A connector's OBB always encloses the rim it wraps, so
the OBB test had to exclude the two panels each part grips — and therefore **could not, even in
principle, see a part biting into its own panel**, which is the failure that actually matters.

Both the panel and the connector are swept solids along the joint, so a 2D **section** test is exact
for them. That is what the tool does now, at both ends of the loft. The OBB test is kept for what it
is genuinely good at: part against a panel it does not touch, and part against part.

### 2.22 A relaxation must be able to fail, and its tolerances are physical

Two mistakes in the first working version of `relax.js`, both found by measuring rather than reading:

- **The "unresolved" threshold was 1e-6 cm**, which reported 25 already-resolved joints as failures.
  A threshold that fine is measuring floating point, not buildability. It is now 0.05mm — fifteen
  times finer than the shim, and coarser than anything that could matter.
- **A spring relaxation settles short of what it aims at.** Measured 0.14mm short on `modular` when
  aiming exactly at the envelope. The correction now aims 0.5mm INSIDE it so the balance point lands
  on it — biasing the target, never the acceptance test.

And the property the whole thing rests on: **a relaxation that always succeeds has stopped being a
measurement.** Its test keeps the failure path reachable by starving it of iterations rather than
relying on a preset that happens to be hard, so it cannot start passing vacuously if the presets
improve.

### 2.24 Placement can widen a joint. It cannot narrow one, flatten one, or ground one.

The relaxation's scope, established by trying everything and measuring:

| asked to | result |
|---|---|
| widen joints below the fastener minimum | **works.** `modular` 25 outside → 0, for 0.3 cm of movement |
| bring folds under the panel limit | works, when it binds |
| pull gaps toward the NOMINAL gap | collisions 5 → 15, residual 0.30 → 5.07 cm |
| close joints wider than a connector spans | diverges — 204 cm of movement, deviation 19 → 154 cm, residual 0.30 → 28.53 cm |
| ground the floating graded edges | no improvement; pulls panels off the target |

The pattern: **a joint that is too NARROW is a placement error, and a joint that is too WIDE is
holonomy.** A tight joint means two panels happen to sit close; nudging them apart costs nothing
elsewhere. A wide joint is wide because the surface curves away underneath it, and closing it means
taking the panels off the surface — the deficit reappears as deviation, collisions, or both.

Same for grounding: at `angularity 0.22, facetCells 3` the *target itself* lifts the toe, so
grounding the panels means leaving the target. It is a form problem, not a placement one.

So the relaxation corrects the floor and **reports** the ceiling, and every unresolved joint carries
`fixable` saying which kind it is. Generalises to: **before adding a corrective force, check whether
the quantity it targets is a free variable or a consequence.**

### 2.25 An obvious-looking correction is worth measuring before shipping

Pushing interpenetrating panels apart is the most natural thing to add to a relaxation and it does
not work here, because the collisions on a steep drift are **housings converging under a fold**, not
panels in the wrong place. Translating them apart moves them off the surface without touching the
cause:

    amp 140 / gap 1cm   collisions 40 → 35, shape residual 0.30 → 1.20cm
    amp 200 / gap 2cm   collisions 43 → 43, shape residual 0.47 → 1.46cm

Shipped, but **off by default, with those numbers in the schema next to the flag**. Three separate
proposals in this project measured worse than doing nothing (this, the nominal-gap spring, the
grounding spring). The cost of measuring first is minutes; the cost of shipping one is a tool that
quietly makes designs worse.

### 2.26 A one-sided envelope, and a one-point measurement

Two bugs in the same check, both of which made the relaxation report success while doing nothing:

1. **The envelope had a floor and no ceiling.** It asked `gap ≥ 1.0 cm` and nothing else, so a joint
   21.6 cm open passed while the connector's own `maxSpanCm` said 8 cm. The limit existed as a flag;
   the envelope never consulted it.
2. **The gap was measured at the joint MIDPOINT.** Midpoint gaps ran 2.64–5.44 cm on a study whose
   joints opened to **21.97 cm at their ends** — so even after the ceiling existed it saw nothing.

Together: 0 joints reported outside where 32 of 60 were. The connector code had already learned this
— `spanMinCm`/`spanMaxCm` exist on every station for exactly this reason — and the relaxation, written
later, did not reuse it.

**A joint is a wedge, not a number.** Any check on it has to be sampled along its length, and any
range needs both ends.

### 2.27 Site facts (unchanged from v2, still unresolved)

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
  it is the obvious next idea and it does not work. **REHABILITATED for v4 by §9.14** — as a *drift
  surface* it is still wrong, but as a *folded network* it is the only family whose cycles close at a
  varying angle, and it is what the wave is built on. The fact did not change; the question did.
- **`gapTolerance` as a buildability gate.** v2 shipped presets with 49 cm worst deviation and
  flagged 40/74 joints; the report is information, not a veto. Only collisions and support are hard.

---

## 4. Open questions

**Re-read §0 first.** Questions 1, 3 and 4 below were framed inside the surface-fit approach and
several are dissolved rather than answered by the pivot — a form built from feasible joints does not
have a "how much joint deviation will we accept" question, because the answer is "none, by
construction". The ones that survive are about the PHYSICAL system and are marked **[carries over]**.


1. ~~**How much joint deviation is acceptable?**~~ **Dissolved by the pivot.** It was the top
   question only because the approach produced deviation and then asked you to tolerate it. Building
   from feasible joints removes the question. What survives is the connector's own envelope, which is
   measured, not chosen: gap 1.0–8 cm, fold below where the panels' back corners meet.
1b. **[carries over] What IS the plate inventory?** `tiling.maxPlates` now exists and restores the strategies to
   usefulness (§2.9), but nothing in the repo records how many 60×121 plates the build actually has.
   That number would turn the budget from an exploration knob into a constraint.
2. **[carries over] Is a wider joint acceptable?** Going 1 cm → 2 cm is what unlocks height, at the
   cost of the plate's exact modularity. The connector work has three things to say now, all pushing the same
   way: the fastener needs **1.00 cm of gap** and the check bites at depth, not at the face; the
   fold a joint can take is `2·asin(gap / 2·outerWallDepth)`-ish, so gap buys fold directly; and
   `modular`'s narrowest station is 0.60 cm, which the relaxation has to shove open. **A wider joint
   is easier to connect, easier to fold, and needs less relaxation.** Still the user's call, and it
   is now the single highest-leverage decision left.
2b. **How many distinct printed parts is acceptable?** The kit answers to `binSpanCm` /
   `binAngleDeg`, and unlike the plate budget the cost is print queue rather than hardware — the
   user has said many unique parts is fine, so the default bins (0.5 cm / 5°) are set fine rather
   than coarse. The number to watch is the forced fit the manifest reports, currently 0.25 cm and
   2.5° worst.
3. **A foldable (planar-quad) target is the real next step.** Faceting with independent planes still
   leaves residual gaps. A true PQ mesh — planar faces meeting exactly along shared edges — would let
   the surface be **as tall as you like** with joints staying near nominal, because the joints become
   the folds. On a rectangular lattice that forces the additive family (§3, ruled out), so it needs
   the fold lines' **plan positions** to move — a real optimization, and the highest-value remaining
   work.
4. **[carries over] How deep can the installation be?** Still unrecorded anywhere in the repo. Presets run 433–501
   cm deep.
5. **[carries over] Is the wall structurally usable for support?** v2 asked this and it is still unanswered; v3 does
   not currently use the wall for support at all.
6. **Reconcile V1 as-built geometry** if V2 planning needs it — §2.27 (this pointed at §2.8, which
   is about the tiler and the placer sharing a surface; the site facts are §2.27).
6b. **[carries over] Is bearing on the power supply housing acceptable?** The supply sits on the flange the back
   half grips, flush to within 1mm. A relief clears the interference, but the lip then bears on a
   driver housing rather than on the panel frame. The tool reports it per station
   (`W_BEARS_ON_POWER_SUPPLY`, 12–50 parts per preset) and cannot decide it. If the answer is no,
   `supplyMode: 'block'` is the honest model and 6–25 joints per preset lose their connector.

6c. **[carries over] Which way does each panel face?** The powered edge is currently a global convention. Panel
   orientation is a real design freedom — it decides which joints are affected and where cables run
   — and nothing in the tool models it.

7. **[carries over] What does the locking screw bite into?** The plan is a threaded area on the connector so it can
   be locked to the panel with a screw, printed or metal (§5.1). The *geometry* is settled — §2.17
   says the jaw can carry up to an M8 boss for free — but the fastening target is not, and it is not
   a question the tool can answer:
   - **into the panel flange.** The strongest lock, and it means drilling or tapping the existing LED
     fixtures. That is a decision about hardware you already own, not about the printed part.
   - **a grub screw clamping the rim.** No panel modification at all; locks by friction, which is
     what the snap fit already relies on, so it adds security rather than a positive lock.
   - **right through the assembly**, jaw → panel → hook. Dimensionally possible (the stack is 33.3 mm
     at the back of the grip) and it is a true clamp, but it needs a clear hole through the flange.

   The first and third both modify the panels. **Whether that is acceptable is the question to
   answer before the boss is designed**, because it decides the screw's axis, length and head.

---

## 5. Next steps / known gaps

### 5.1 Not built

- **Connector design** is built and has been through two whole redesigns (P9–P15). Fastening is
  solved — three M3 countersunk bolts into heat-set inserts, which is what the two-piece split was
  for. What it does *not* yet do:
  - **no separate locking screw.** The three bolts hold the halves to each other and clamp the rims
    between them; nothing threads into the PANEL. That is still wanted, the section is sized for it
    (§2.17's headroom figure survives), and the open question is not geometry but **what the screw
    bites into** (§4.7).
  - **the universal front bar is retired.** One bar width serving a band of gaps worked on the
    well-behaved presets (1 bar for `closed` and `modular`, 2 for `drift`) and could not be made to
    work at the extremes. Each station now gets its own. `frontBarBand` / `solveFrontBars` survive
    for the day it becomes possible again — the kit still bins the widths, so the report still says
    how few distinct bars a design would need.
  - **no structural analysis.** `CONNECTOR_LIMITS.maxSpanCm = 8` is a judgement about a
    9.5 mm strap, not a calculation. The spine does not thicken or rib as the span grows, so a
    15.8 cm joint gets a part that is flagged rather than redesigned.
  - **no cable routing.** The v1 photographs show power leads running through the joints; the
    current part ignores them entirely.
  - **the parts are not labelled.** The manifest maps a plate position to a part id; the printed
    object carries no marking, so ~80 parts across ~20 types have to be kept in order by hand. An
    embossed id is the obvious next thing, and now that there are TWO families it matters more.
  - **the viewport does not show which joints the relaxation could not fix.** The report names them;
    the 3D view does not highlight them, which is the thing you most want to see when it reports a
    failure.
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
- ~~Changing `sheet.cols`/`rows` does not rescale `form.footprint`.~~ **Fixed.** The footprint now
  follows the sheet until you set it by hand, which locks it; "refit to sheet" hands control back.
  The lock is UI-only state — it governs how a LATER edit behaves, not the design.
- `dist/` is committed from a v2 build and is stale.
- **`tests/screenshot.mjs` is stale v2.** Its docstring still describes `endSupport`, `E_UNGROUNDABLE`
  and the `wallcrash` preset, none of which exist in v3, and it starts its own dev server on 5175
  (`strictPort`), so it cannot run alongside another session's. It is not in the README's suite list
  and was not run for P9–P13. Either port it to v3 or delete it.
- A second launch entry, **`grid-designer-alt` on 5176**, exists for exactly that port collision.
- The viewport's `facet` colour mode still colours via `hashHue`, which maps adjacent keys to hues
  1/360 apart. It happens to look fine because facet keys are 2D, but the connector kit hit this
  properly (P12) and switched to a golden-ratio step on the index; `facet` would benefit from the
  same treatment.
- The 3D viewport's default camera starts low and close; orbit out to read the drift. Not tuned.
- A manual pin set made at one sheet size does not mean the same thing at another — overrides are
  keyed on `(i, j)`.

---

### 5.3 WHAT THE NEXT DIRECTION INHERITS

The pivot is away from *author a surface, then reconcile it*. Most of this repository is not that,
and carries over unchanged. Sorted by how much is worth keeping.

#### Keep — measured facts about the physical system

These cost real effort to establish and do not depend on how a form is arrived at.

- **`src/config.js` — the panel.** Nine caliper-measurable parameters from `updatedPanelGeo/`, with
  `panelSectionRings()` as the single source of truth for the section, and `POWER_SUPPLY`. Every
  downstream consumer derives rather than restates (§2.19). **This is the most valuable file in the
  repo for a new direction** and should be the thing the new model is built on top of.
- **`src/geometry/panelGeometry.js`** — the solid and the supply box, swept from that section.
- **`src/core/v3/connectors.js`** — the two-piece bolted clamp, its section, and the **joint
  feasibility envelope**: gap ≥ 1.0 cm for the fastener (checked at depth), gap ≤ 8 cm for the part,
  fold ≤ where the panels' own back corners meet. A form built from joints inside that envelope is
  buildable **by construction** — which is precisely what the new direction should exploit.
- **`polygonsOverlap` / `panelSectionAt`** — exact section-level collision for swept solids (§2.21).
- **`collide.js`** — 15-axis OBB SAT, mutation-tested. Model-agnostic.
- **The connector kit, export and manifest** — binning, STL plate, part manifest. Independent of how
  the joints came to be.

#### Keep — machinery

- The **headless-core contract** (§6), the determinism rules, and the whole testing convention:
  plain node scripts, closed-form expectations, known-answer controls, non-vacuous negatives.
- **`persistence.js`**, the store contract, the viewport/plan-view shell.

#### Retire with the approach

- **`form.js` as the driver.** The parametric drift is a fine authoring tool; it is the wrong
  *input* to a panel layout. Keep it as a target to compare against, not as the thing panels are
  fitted to.
- **`target.js`'s faceting.** It exists to make an authored surface reachable by rigid panels. If the
  form is built from reachable joints, there is nothing to quantize.
- **`placement.js`'s `surface-fit` / `chain`.** Both answer "where do panels go on this surface".
- **`relax.js`.** It exists to drag a fitted layout back toward feasibility. Building inside the
  envelope from the start makes it unnecessary. Its lessons (§2.24–§2.26) outlive it.
- **The six presets.** They are six points in the surface-fit trade.

#### The idea the new direction is probably reaching for

Already recorded as §4.3 and still the highest-value unbuilt thing: **build the surface out of joints
that are known-good instead of measuring how bad an authored surface is.** Every joint has a feasible
band, the connector work has now quantified it exactly, and a form composed of feasible folds is
buildable without any reconciliation step.

The trap that makes this non-trivial is recorded in §3: on a rectangular lattice, all-quads-planar
forces `h(i,j) = f(i) + g(j)`, and that family **cannot** be zero along two intersecting edges and
still be a mound — so it is incompatible with the brief's two grounded edges. Getting past it needs
the fold lines' **plan positions** to move, which is a real optimisation and the thing worth building.

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

**Start with §0.** It says what was built, what it proved, and why the direction is being left. Then
**§5.3**, which is the inventory of what carries over — most of the repository does.

**That new direction was built — it is v4, and §8 is its record.** Read §8 first; the list below is
the background it was built on, and is still the right background.

If you are continuing the NEW direction, the reading order is:

0. **§8** — what v4 is, what it measured, and the one hardware question it raised.
1. **§0** — the verdict and the evidence for it.
2. **`src/config.js`'s header** — the measured panel, and why it is parameters rather than a shape.
   This is the foundation the new model should sit on.
3. **`src/core/v3/connectors.js`'s header** — the two-piece clamp and, more importantly, the joint
   feasibility envelope. A form composed of joints inside it is buildable by construction.
4. **§2.24–§2.26** — what a placement solver can and cannot do, so the next one is not asked to do
   the impossible again.
5. **§3 and §5.3's last paragraph** — the planar-quad trap, which is the thing standing between here
   and a form built out of known-good folds.

If you are maintaining what exists: `npm run dev --prefix grid-designer` (port 5175, or the
`grid-designer-alt` / `grid-designer-fresh` entries on 5176 / 5177), run the 20 suites. The app
mounts **v4**: push the angle past 33.6° to watch the envelope readout turn from a permission into a
refusal with the reason named, drop it to 12° to see the front-bar warnings clear, and orbit round
to +X for the profile the fold pattern reads in.

The retired v3 UI is `src/v3/AppV3.jsx`, unmounted — point `src/main.jsx` at it to load `shelf` and
see the brief satisfied, or `crest` to see the report say no.

**Verify the numbers, not the narrative** — and prefer structural proof over output comparison. That
practice caught every one of the errors recorded here, including three of my own proposals that
measured worse than doing nothing.

---

## 8. v4 — the folded ribbon

Built 2026-07-27, immediately after §0's verdict. **[V4_SPEC.md](V4_SPEC.md)** is the specification;
this section is what building it established.

### 8.1 What it is

An open chain of rigid 60 × 60 panels folded in a vertical plane, running from the window away
toward +z, in the trapezoid pattern `_ / - \ _ / - \ _`. One angle θ, shared by every angled unit.
The flats between the angled units split each direction change in half, so **every fold is θ rather
than 2θ** — the reason this pattern buys height cheaply.

Built as two work packages, each delegated and then independently verified by re-running the suites
and probing the core directly rather than reading the summary (§6):

| package | what |
|---|---|
| WP1 | `src/core/v4/` — schema, chain, connectors, report + 746 checks across four suites |
| WP2 | `src/v4/` — store, shell, controls, units table, metrics, report, viewport |

### 8.2 The bisector construction, and why the gap stops being a measurement

The chain is a polyline on the panels' **lit-face plane**, and each joint steps along the bisector
of the two panel directions:

```
S_{k+1} = E_k + gap · normalize(u_k + u_{k+1})
```

so `|S_{k+1} − E_k| = gap` exactly, at every joint, at every angle, and the joint is symmetric about
that step. Verified over 49 (gap, angle) combinations: **max |span − gap| = 0, exactly**, along with
zero span spread and zero twist.

**This is the whole pivot in one line of arithmetic.** In v3 the gap was an outcome: you authored a
surface, panelized it, and measured how far each joint had been wedged from nominal — 58 of 60
joints out of tolerance on a typical drift. In v4 it is an input the geometry honours by
construction. There is nothing to deviate, so there is no tolerance on it and no relaxation solver.

### 8.3 The envelope has to be bisected, not read off

`foldLimitDeg(gap)` is the panels' own back-corner limit and it is far too generous to use directly:
at a 2 cm gap it says 90°. The binding constraint at that gap is **`W_FASTENER_PINCHED` at 33.6°** —
the M3 bolt pinched at the depth its insert sits at, on a convex joint where the gap narrows with
depth. §2.20 already said the connector fouls 3–6° before the panels do; the consequence is that the
honest limit is **where the flags start**, found by bisecting `connectorStationFlags` over 60 fixed
steps, and never by evaluating a formula.

Both directions are reported, because both are permissions:

- `maxAngleDeg` — the largest θ this gap admits. **33.60° at gap 2 cm** (headroom 3.60° on the
  default 30° design).
- `minGapCm` — the smallest gap this angle admits. **1.90 cm at θ = 30°** (headroom 0.10 cm).

Both were verified to be real boundaries rather than plausible numbers: clean at `maxAngleDeg −
0.05°` and dirty at `+0.05°`, clean at `minGapCm + 0.005` and dirty at `−0.005`.

**This is the thing §0 says v3 never had.** The old tool could grade a design; this one states what
you are allowed to do before you do it.

### 8.4 The front bar cannot lie across a valley — a new finding, and a real problem

Assign a bar width and ask `sectionFouling` directly (v3's ordering assigns widths *after* flagging,
so `W_FRONT_BAR_FOULS_PANEL` has never fired in either version) and it says:

> **a flat front bar bites the bezels on a concave joint past 12.37° of fold.**

In a valley the two lit faces tilt up toward a bar that stays flat across the gap, and its overhang
meets the rising bezel peaks. Bisected at gaps of 1, 1.5, 2, 3 and 4 cm: **12.37° at every one of
them.** That constancy is the tell — the overhang that collides is `frontLipCm`, which does not
depend on the gap, so **widening the gap does not buy a single degree.**

This matters here in a way it did not for a drift: **exactly half of a trapezoid wave's joints are
valleys**, so at any useful θ the front bar needs an answer. The default 30° design is 17.6° past it
on four of its eight joints.

It is deliberately kept **out** of `ENVELOPE_HARD_FLAGS` and given its own reading
(`envelope.frontBar`, and `W_FRONT_BAR_FOULS_BEZEL` per joint):

- folding it in would collapse `maxAngleDeg` from 33.60° to 12.37° and hide the limit that governs
  the **connector**, which is the one a fold actually has to respect;
- and it is **fixable in the part** — a relief or chamfer on the bar's underside, or a narrower bar
  on concave stations. A limit you can design away does not belong in the same number as one you
  cannot.

**This is the open hardware question v4 hands back.** §5.1 already recorded that the universal front
bar was retired and each station gets its own; this says the concave stations need a different
*section*, not just a different width.

### 8.5 Smaller things worth keeping

- **A flip is a statement about the joint, not a display option.** The connector grips the back
  flange, so a joint whose two panels face opposite ways has its flanges on opposite sides and **no
  part in this family can span it**. v4 emits no station for such a joint and reports
  `W_JOINT_FLIP_MISMATCH`. Flipping one unit costs two joints their connectors.
- **Removal is kinematically inert, by design.** Removing a unit leaves every other unit's position
  *bit-identical*; only the two joints touching it disappear. A layout that re-solved itself when
  you deleted a panel would make the pattern unusable to reason about.
- **Adjacent panels' OBBs genuinely interpenetrate** — 0.12 cm at 30°, 2.4 cm at 75°, always at a
  convex fold where the housings converge. Excluding adjacent pairs from the collision pass is
  therefore load-bearing, not a formality: their overlap is the joint's business and
  `sectionFouling` judges it exactly, where an OBB pair cannot.
- **1e-9 rounding is too coarse for a feasibility verdict.** v3's `r()` at 1e-9 gave an 8.0 cm gap a
  measured 8.000000001 and tripped `W_CONNECTOR_SPAN` — a picometre deciding buildability. Unit
  records keep 1e-9; joint rim points, normals and run vectors use 1e-12.
- **`dihedralDeg` is `|foldDeg|`, not an independent `acos`.** `acos` is badly conditioned near a
  flat joint and returned ~1e-4° of noise at θ → 0 where `atan2` is exact.

### 8.6 Open — the next package

**The sideways branches.** The `high` units (3, 7, …) are marked as branch anchors and nothing runs
out of them yet. The plan is angled panels from a high unit down to the floor, sideways along x, to
meet the next strip and close the network in 3D. `strip.count` and the `(strip, unit)` override key
already exist for it; `strip.count > 1` currently renders independent parallel ribbons with no
cross-strip joints.

Note that this is where §3's planar-quad trap will reappear, and where it will be decided rather
than argued: a branch that lands on the floor *and* meets its neighbour's branch is a closed cycle,
and a cycle of rigid panels does not generally admit exact joints (§2.2). The v4 answer available
and not available to v3 is that the branch's own angle is a free parameter chosen from inside the
envelope, rather than dictated by a surface.

Also open, in rough order of how much they matter:

- **the front bar's concave section** (§8.4) — the one real hardware question this pass raised
- per-unit angle overrides; plates (`2x4`) in the chain
- the connector manifest still self-identifies as `grid-designer v3` and has a `design.sheet` slot
  a v4 config cannot fill; the parts and quantities are correct
- part labelling, cable routing, structural analysis — all still §5.1's list, all still unbuilt

---

## 9. v4, part two — the network

The ribbon generalised to a 2-D lattice, per **[V4_SPEC.md §9](V4_SPEC.md)**. The unit of design
stops being a panel in a chain and becomes a **flat cell on a lattice**; the angled panels are
derived, one per edge between two present cells.

### 9.1 The rules forced the structure

The brief was a growth grammar — ground flats add ramps UP at any face, high flats add ramps DOWN,
repeat in x and z, ragged edges allowed. That admits **exactly two levels**, which makes every flat
cell's four neighbours the opposite level, which is a **checkerboard**. There was no design freedom
left to exercise; the rules had already chosen.

### 9.2 The half-angle identity, and why the lattice is uniform

The bisector step of §8.2 collapses when one panel is horizontal:

```
normalize(u_flat + u_tilt) = (cos(θ/2), sin(θ/2))
```

because `1 + cos θ = 2cos²(θ/2)`, `sin θ = 2 sin(θ/2) cos(θ/2)`, and the norm is `2cos(θ/2)`.

So **every** level change costs the same plan distance and the same rise, in x and in z alike:

```
CELL PITCH   P = 60 + 2·gap·cos(θ/2) + 60·cos θ      identical in both axes
LEVEL RISE   R = 2·gap·sin(θ/2)      + 60·sin θ
```

`P = 115.825228`, `R = 31.035276` at θ = 30°, gap = 2 — agreeing with the shipped ribbon to nine
decimals. **The flat cells therefore sit on a uniform square lattice and every cycle closes with
zero residual.** Verified by walking ground→ramp→high→ramp→ground→ramp→high→ramp around a corner and
landing back on the start point.

**This is the answer to §3's planar-quad trap, and it is worth being precise about why it is not a
counterexample.** §3 says an all-quads-planar form on a rectangular lattice must be `h(i,j) = f(i) +
g(j)`, and that such a family cannot be zero along two intersecting edges *and* be a mound. The
checkerboard IS separable — `level = A(i) XOR B(j)` — and it escapes the trap by not being a mound.
It is periodic. The trap was never a statement about rigid panels; it was a statement about mounds.

### 9.3 A ramp needs one cell, not two

The first cut required a ramp's **both** cells to be present. That made the plan editor a lie:
clicking one flat removed up to five panels, so you could not edit the design panel by panel, which
is the only thing the editor is for.

The fix was one operator — `&&` to `||` — and the justification was already in the model: a **wall
anchor** is a ramp with nothing at its far end, and it is a perfectly good panel. A ramp held at one
end cantilevers off its single joint. Only a ramp with **neither** cell is absent, because that one
would float.

One click is now exactly one panel: 37 → 36 → 35 → 36 → 37, verified in the browser as well as the
suite. Removing an interior flat costs four joints (one orphaned end per ramp) rather than eight.

### 9.4 Editing and growing are one operation

Because the lattice is **generated** rather than chained, switching a cell off leaves every other
panel bit-identical — verified live in the browser, not just in the suite. So:

- switching a cell **off** makes a ragged edge;
- switching one **on** at a free face is "growing panel by panel", and it lands on the lattice by
  construction, so it closes exactly;
- switching an **edge** off opens the network without cutting material out of its boundary.

There is no separate growth mechanism, no hinge to drag, and nothing to reconcile. That equivalence
is the reason the lattice model was chosen over a kinematic tree, which is what `panel-designer/`
was and why it was abandoned: a tree cannot close a cycle, and a network is nothing but cycles.

### 9.5 A flip is much more expensive on a network

A ramp cannot be flipped — its orientation is fixed by which cells it joins. So flipping a *cell*
mismatches **every joint that cell has**, up to four, and each of them loses its connector. On the
ribbon a flip cost two joints; here it can cost four. The store's notice says so at the moment of
the click rather than leaving the report to break the news.

If flipping is to stay useful on a network, `overrides.edges` needs a `flipped` too. Not built.

### 9.6 The corner clearance — the one number with no 1-D analogue, and it is unresolved

Four ramps meet at each lattice corner and, in plan, leave a diamond hole between them: a ground
cell's +x ramp occupies `x > 60, z ∈ [0,60]` and its +z ramp `z > 60, x ∈ [0,60]`, so the region
beyond both is occupied by neither. That much is exact and intended.

Whether their **housings** clear is not settled, and the tool cannot settle it:

| measured with | clearance at θ = 30°, gap 2 |
|---|---|
| the full OBB | **−0.12 cm** (crosses zero at 28.2°) |
| a back-plate-only box | **+9.43 cm** (still +7.19 cm at 50°) |

The boxes meet at their **corners**, where the real section is 1.2 cm of outer wall and the OBB
claims 4.1 cm. The honest test is section-level — the `sectionFouling` analogue for a pair that
shares no joint — and no such function exists in `core/v3/`.

So corner pairs are reported (`metrics.cornerClearance`, `W_CORNER_RAMPS_MEET`, and their own
`report.cornerContacts` list) and deliberately kept **out of `report.collisions`**. Calling a 3 cm
box corner a panel collision would assert something the primitive cannot support, and would put 16
red pairs on a design whose panels are very probably 9 cm apart. **A positive clearance is a
guarantee; a negative one is a question.**

Unlike the front bar (§8.4), the gap buys this back directly: the crossing is 14.0° at gap 1, 28.2°
at gap 2, 58.4° at gap 4, ≥75° at gap 8.

### 9.7 Smaller findings

- **`collisions` is structurally empty on a lattice.** The only panels that can reach each other are
  the ones sharing a joint (excluded — that overlap *is* the joint) and the two ramps off a shared
  cell (the corner pairs). Everything else is a full pitch away. An empty list is therefore the
  expected result, which is exactly why the suite checks the split is a **partition** of the raw
  overlaps rather than trusting the count.
- **`maxAngleDeg` must round DOWN and `minGapCm` UP.** Rounding either to 1e-9 could move it onto
  the *dirty* side of a boundary located to 1e-16 — `isClean(reportedMinGap)` was actually false at
  15° and 60°, so the tool was issuing a permission it would itself refuse. Latent since the ribbon;
  only surfaced with the network's numbers.
- **The envelope is a property of (gap, θ) and nothing else** — every joint on a lattice has the
  same span and the same |fold|, so a 10 × 10 network reports the same 33.60° as a 1 × 2 one. That
  is what makes it quotable as a permission *before* a design exists. It is still measured over the
  real design, because a module that asserts its own inputs cannot detect its own bug.

### 9.8 Growing the network is the same click as shrinking it

The plan editor started out only able to *remove*: you could switch a cell back on inside the
bounding rectangle, but reaching past it meant the cols/rows steppers, which add a whole row at a
time and cannot reach the wall or window sides at all. Since the point of the editor is tailoring
where the surface meets the ground, that was most of the job missing.

A **ring of empty slots** one cell wide now surrounds the grid; clicking one grows the rectangle.
Three things had to be true for it to mean "add one panel":

1. **Every other new cell slot starts absent.** Otherwise widening by a column adds five.
2. **Every new edge slot not touching the clicked cell starts absent.** This one was found by
   measuring rather than by thinking: because a ramp needs only ONE cell (§9.3), widening the
   rectangle hung a ramp off every cell of the column beside it, and the first version of the click
   added **seven** panels. The tooltip promised one.
3. **Switching a cell on re-connects it to its present neighbours** — it clears exactly the
   suppressions from (2) whose far cell is there. Without it, filling a slot next to an existing
   column left the new panel floating beside its neighbour with no ramp between them. Edges to an
   *absent* neighbour stay off, or a click would sprout cantilevers into empty space.

**The re-origin is the part worth remembering.** Growing at `i = −1` shifts every index by one, so
both the overrides and `pattern.phase` have to move with it — `level = (i + j + phase) mod 2` would
otherwise invert the entire checkerboard, turning every ground cell high. That is a failure that
looks deliberate, so the suite checks it both ways: all 15 original cells keep their level after a
wall-edge growth, and omitting the flip inverts all 15.

`trim` shrinks the rectangle back to the cells in use, by the same rule in reverse. No panel moves.

### 9.9 The room's column

`config.obstacles` (V4_SPEC §9.11) — axis-aligned boxes the design has to be planned around. The
measured column (380, 285, 50 cm square, corner-anchored) ships as a default, because a design made
without it on screen is a design made against the wrong room.

Two decisions worth keeping:

- **`anchor` is explicit.** `(380, 285)` is ambiguous between a near corner and a centre, and the two
  differ by 25 cm on a 50 cm column. Rather than guess, the record says which and the UI prints the
  resulting extents beside the inputs. Guessing here produces a collision report that looks entirely
  plausible and is wrong by half a column.
- **Report, never enforce.** A fouled panel is named, outlined red in the plan and turns the column
  red in 3D, but it is still placed and still counted — verified by a test asserting a fouled design
  has exactly the same panel count and the same collision list as one with no column at all. The
  fix is a click in the plan editor, not a veto.

The overlap test deliberately uses the full panel OBB. It overstates the section near the rim (§9.6),
but here that bias is the safe direction: a false "this fouls the column" costs one click, a false
"it clears" costs a site visit. Clearance is a **lower bound** for the same reason.

At the shipped 3 × 5 the column is 88 cm clear. It first bites at **4 columns** (2 panels), and 5
columns puts 3 through it.

### 9.10 The OBJ is grouped for rendering, not for inspection

The old exporter emitted one object per panel and one per connector — the right shape for checking
geometry, the wrong shape for lighting a scene. `src/v4/objExport.js` emits instead:

| object | what |
|---|---|
| `diffuser_NNN_<id>` | **one per panel**, so each lit face can take its own brightness |
| `frame` | every panel's housing, merged |
| `connectors` | every printed part, both pieces, merged |
| `spacers` | every ground spacer post, merged (added in §9.12) |
| `power_supplies` | every driver box, merged |

The diffuser/frame cut costs nothing to maintain because it was already drawn: `panelGeometry.js`
emits the solid as `DIFFUSER_MATERIAL_INDEX` / `HOUSING_MATERIAL_INDEX` groups and the viewport has
always rendered them as two materials. This module cuts along that line, so a change to the measured
section flows through with nothing to update here.

Two things worth keeping:

- **The split is checked by triangle count, not by eye.** 2 + 42 = 44, the whole panel solid,
  asserted per panel. Losing or duplicating a triangle when cutting a geometry by material group is
  invisible in a render and is exactly how this goes wrong.
- **Sub-geometries are re-indexed, not dereferenced.** Dereferencing would have been three lines
  shorter and roughly tripled the file.

`src/utils/exporters.js` is untouched — it is shared with the retired v3 UI and its suite pins its
output, so this is a parallel builder rather than a flag on that one.

### 9.10a Correct grouping is not the same as importable grouping

The export above was right and still imported **as a single surface**. Worth understanding, because
the failure looked like a geometry bug and was not one.

The OBJ had 41 `o` blocks, 14128 vertices, global cumulative face indices and zero forward
references — structurally perfect. It also emitted `usemtl <name>` for all 41 objects and shipped
**no `mtllib` line and no `.mtl` file**. Every material name resolved to nothing. Importers that
split a mesh by material — which is most of them — found no materials to split on and merged the
lot. *The splitting information was in the file; the thing that makes a reader act on it was not.*

Two fixes, both consuming the same `buildSceneGroup`:

- **The OBJ now ships its library.** `mtlPayloadV4` walks the scene group and emits one `newmtl` per
  material actually found, so the two files cannot disagree — hardcoding a parallel palette was the
  obvious alternative and is precisely the drift that caused this. The button downloads **both**
  files, named from one stamp so the `mtllib` points at a file the user really has. Two downloads
  rather than a zip: a zip is a dependency to solve a problem two clicks already solve.
- **A GLB export**, `src/v4/glbExport.js`. One self-contained binary, a named node per object, one
  material per mesh, and a real emissive strength. For "37 diffusers each with their own brightness"
  it is the better import; OBJ stays for readers that only speak OBJ.

**Why not FBX**, which was asked for first: three.js ships an FBXLoader and **no FBXExporter**, and
never has. Writing one means hand-emitting the binary FBX record tree — a format documented by
reverse engineering — or taking a dependency. Every target that reads FBX also reads GLB.

Two details that were verified rather than assumed:

- **Every mesh gets its own material instance**, including the merged groups. `GLTFExporter` does no
  material deduplication across instances (checked), so 41 instances become 41 glTF materials and 37
  diffusers get 37 addressable brightnesses. A hoisted shared material — which reads as an
  optimisation — would collapse them to one, and "give this panel its own brightness" would mean
  "change all 37". `test-v4-export.mjs` §5 pins it at the source.
- **`emissiveIntensity` stays at exactly 1.** `GLTFExporter`'s `KHR_materials_emissive_strength`
  writer returns early at 1.0, so the default file carries a plain `emissiveFactor` and no
  extension — correct, since there is no strength to declare yet. Drive a panel to anything else and
  the extension appears with that value. The test asserts the absence at 1 and the presence at 4.25,
  because asserting the extension were always present would pin a bug.

Verified by parsing the artefacts back, not by checking the exporter ran: `tests/test-v4-export.mjs`
(69 checks) decodes the GLB container by hand — header magic, chunk lengths, BIN against the declared
buffer, node/mesh/material counts, diffuser material distinctness — resolves every `usemtl` against
every `newmtl`, re-checks index integrity, and asserts the OBJ `o` list and the glTF node list both
equal `buildSceneGroup`'s so the two writers cannot drift. `tests/screenshot-v4-export.mjs` clicks
the real buttons and parses the files that actually land on disk.

### 9.12 The clearance and the spacer are two different things, and keeping them apart is what makes either checkable

The brief was one sentence — *everything laying flat on the ground needs a 15cm spacer or gap,
matching the pattern of the other connectors* — and it named two things that wanted building in two
places.

**The gap is grounding.** `placement.groundClearanceCm` (default 15, range 0..50) moves what
`groundToFloor` aims at: the network's lowest material lands at the clearance instead of at 0. It is
a rigid translation of the whole network, like the wall and window offsets. The tempting alternative
— lift the cells that rest on the floor, leave the rest — is wrong for a reason that has nothing to
do with taste: the network is one rigid assembly and every joint between a lifted cell and an
unlifted ramp would have to open by 15cm. There is no per-cell version of this.

**The spacer is a part**, in a new `src/core/v4/spacers.js`, and it is the thing that holds the gap
open. 64 posts on the shipped default.

#### The measurement is the whole design of the module

`spacers.js` could have written `heightCm: config.placement.groundClearanceCm` and been correct on
every design. It measures the underside instead, and the report cross-checks the two:

- `lattice.js` **applies** the clearance as a translation;
- `spacers.js` **measures** the underside it landed at;
- `report.spacers.heightsCm` is the set of distinct results, and there should be exactly one.

A module that asserts its own inputs cannot detect its own bug — the same rule `connectors.js`
follows when it measures a span it knows is `gap`, and the reason the suite's central claim is worth
anything. `W_SPACER_MISMATCH` is what fires when they disagree, and it is non-vacuous: a design with
no ground cells present has its lowest flat cells one level up, something else (a ramp toe) is the
lowest material, and the posts under those cells are genuinely the wrong length.

#### "Matching the pattern of the other connectors" is a reused rule, not a look

The stations come from `stationCount` in `core/v3/connectors.js` — the same function, the same
`config.connectors.spacingCm` / `minPerJoint`, and the same `(m + 0.5)/n` symmetric spacing
`solveConnectorsV4` uses — applied to each of the cell's four edges. A joint's rim and a cell's edge
are both 60cm, so if the rule is really shared the two counts have to track each other exactly, and
`tests/test-v4-spacers.mjs` §6 sweeps the spacing knob and asserts they do. A hardcoded four per
cell would have looked identical on the default design and drifted apart the first time the knob
moved; that check is the difference between a shared rule and a coincidence.

#### Only the lowest cells, and it is measured rather than looked up

A high cell is held up by the four ramps that reach it — that is what the checkerboard is for — so
propping one would be a redundant load path and a 66cm leg standing in the room. The qualifying set
is found by measuring each present cell's underside and keeping those within 1e-6 of the lowest,
not by testing `level === 0`. They agree on every design this model can build. The measurement is
what keeps meaning the right thing if a third level ever arrives.

#### Smaller decisions worth the record

- **The post is set in by half its section**, so its outer face is flush with the rim. Centred on
  the rim line, half of it would hang outside the panel.
- **Its section is `PANEL_PROFILE.overallThickness`**, derived rather than picked — as wide as the
  housing it stands under (HANDOFF §2.19 on constants with no provenance).
- **Grounding off means no spacers at all**, not zero-height ones. A post from the floor to a cell
  that is under the floor is an artefact of asking a question that does not apply.
- **What the post is made of and how it fixes to the panel is not modelled.** The rim is 1.2cm of
  outer wall where the foot lands and does not reach full thickness until `bodyInset` inboard, so
  the load path there is a real question. Same contract as the wall anchor: do not invent a bracket.

### 9.14 The wave — a varying angle, and why the checkerboard cannot have one

**V4_SPEC §9.14** is the spec. `pattern.kind` gained a second value; **`trapezoid` is untouched and
is still the default**, and `tests/test-v4-wave.mjs` §8 proves it byte for byte (FNV-1a over
`JSON.stringify(solveLattice(cfg))` for seven configs, taken from `71f8b15`).

#### THE FINDING: a two-level checkerboard cannot carry a varying angle. Measured.

This is the durable result of the package, and it is written here because **the next person will
otherwise try to make the checkerboard scrunch.**

Alignment ("nothing gets out of basic alignment") forces the plan grid to be a product grid, so the
x-ramp angle depends only on `i` and the z-ramp angle only on `j`. On a checkerboard the levels
alternate around every 4-cycle, so the four steps are `+R(θx)`, `−R(θz)`, `+R(θx)`, `−R(θz)` and
closure demands `2·R(θx) − 2·R(θz) = 0` — **every angle equal**. With
`R(θ) = 60·sin θ + 2·gap·sin(θ/2)`, measured at gap 2:

| x-ramp | z-ramp | cycle left open by |
|---|---|---|
| 30° | 35° | **9.164 cm** |
| 30° | 40° | **17.800 cm** |

Varying the **gap** to hold `R` constant does not rescue it. Holding `R = 31.035`:

| θ | gap needed |
|---|---|
| 20° | **30.27 cm** |
| 25° | **13.12 cm** |
| 35° | **−5.62 cm** |
| 40° | **−11.01 cm** |

All outside the connector envelope's 1–8 cm; two negative. There is no such network.

#### The family that works is the one §3 rejected, read the other way round

`h(i,j) = f(i) + g(j)` makes the 4-cycle residual an identity, so the angles are free. Measured with
5 different x-angles and 6 different z-angles: **worst residual 7.1e-15 cm.**

§3's rejection of "additive / translational height fields" **still stands for the problem it was
about** — v3 needed a field zero along two intersecting edges *and* a mound in between, and this
family cannot be that. v4 is not asking for a mound. It is asking for a foldable network, and
separability is the exact condition for one. §5.3's last paragraph said this would be the move.

**The price, and it is not a bug:** `f` and `g` each zig-zag by ±R, so `h` takes three values, not
two. Even at zero scrunch the wave is a uniform **egg-crate with three storeys**. Do not "fix" it
back to two levels — that is the constraint the proof above says kills it.

#### What is linear is the plan advance, not the angle

`E(θ) = 60·cos θ + 2·gap·cos(θ/2)` is compressed on a straight ramp to `scrunch` at `attractor`, and
the angle that delivers it is bisected (60 fixed steps, on `[base, ANGLE_MAX]`). That is what
"compress in a simple linear fashion" reads as on the floor; equal steps in *degrees* would not.
`θ(0) === angleDeg` exactly. At 30°/2cm the steepest reachable scrunch is ~66.5%, so
`W_SCRUNCH_UNREACHABLE` is a real outcome inside the 0–0.9 band and names the edge it pins.

#### Three things the wave forced elsewhere, all worth the record

1. **`level` had to stop being binary**, and the honest generalisation was to rank cells by MEASURED
   height rather than to invent a third enum. `level === 0` still means "on the floor", which is the
   only thing anything downstream ever asked of it.
2. **`spacers.js` was already right, and this is the payoff.** §9.12 chose to *measure* undersides
   rather than test `level === 0`, "in case a third level ever arrives". It arrived. On a scrunched
   wave the module correctly props the **one** cell actually resting on the floor — which is itself
   a finding: a scrunched wave stands on one corner and is grounded as a rigid body, not laid on the
   floor.
3. **The envelope stopped being a property of `(gap, θ)`** and the report had to say so rather than
   keep quoting one boundary. `maxAngleDeg` under the wave means *the largest **base** angle*;
   `envelope.perJoint` carries the distinct-fold count, the dirty count and the worst joint. The
   worst joint is ranked by **signed** fold, not magnitude: ridges pinch and valleys diverge, so the
   first cut of this reported a 55° valley carrying no flags at all as the worst thing in a network
   with 18 genuinely dirty ridges.

#### Refused rather than approximated

The **wall anchor** is not built on a wave. §9.4's anchor descends exactly one level rise to the
floor; the wave has no such level, so the toe would land in mid-air. `braced` emits nothing and
`W_WAVE_NO_ANCHOR` says so. A per-cell anchor angle solved to reach the floor is a real design and
belongs in its own pass — the standing rule here is that the geometry does not invent a bracket.

### 9.13 Open

0. **A wall anchor for the wave** (§9.14) — see "refused rather than approximated" above.
1. **The corner section test** (§9.5) — the one real gap.
2. **The front bar's concave section** (§8.4) — still the open hardware question, and the network
   makes it worse: *every* ramp meets its ground cell in a valley, so exactly half of every
   network's joints are concave at any θ.
3. **What the wall anchor fixes to.** The geometry says where the toe lands; the attachment is not
   modelled.
4. **Whether the spacer's foot can bear on the rim** (§9.12) — the geometry places it; nothing
   says the section is strong enough there.
5. Flippable ramps (§9.4); plateaus of same-level flats; plates. *(Per-cell angle and >2 levels are
   delivered by §9.14 for the separable family; on the checkerboard they are impossible, proven.)*
