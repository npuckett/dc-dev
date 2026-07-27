# grid-designer — handoff & decision log

Written 2026-07-26, covering the **v3 pivot** session. Read **[README.md](README.md)** first
for how the tool works and **[V3_SPEC.md](V3_SPEC.md)** for the model; this document records **why it
is the way it is**, what was tried and rejected, and what is still open.

The v2 handoff's "durable findings" section is superseded by §2 here, but its central insight
survives unchanged and is still the reason any of this is tractable:

> Don't solve rigid origami. Place panels deterministically and **measure** what the connectors have
> to absorb.

Branch `v3-drift-tiling`. All suites green (3886 checks across 16 suites), build clean, app verified
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

### 2.23 Site facts (unchanged from v2, still unresolved)

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
   plate's exact modularity. The connector work has three things to say now, all pushing the same
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
4. **How deep can the installation be?** Still unrecorded anywhere in the repo. Presets run 433–501
   cm deep.
5. **Is the wall structurally usable for support?** v2 asked this and it is still unanswered; v3 does
   not currently use the wall for support at all.
6. **Reconcile V1 as-built geometry** if V2 planning needs it — §2.23 (this pointed at §2.8, which
   is about the tiler and the placer sharing a surface; the site facts are §2.23).
6b. **Is bearing on the power supply housing acceptable?** The supply sits on the flange the back
   half grips, flush to within 1mm. A relief clears the interference, but the lip then bears on a
   driver housing rather than on the panel frame. The tool reports it per station
   (`W_BEARS_ON_POWER_SUPPLY`, 12–50 parts per preset) and cannot decide it. If the answer is no,
   `supplyMode: 'block'` is the honest model and 6–25 joints per preset lose their connector.

6c. **Which way does each panel face?** The powered edge is currently a global convention. Panel
   orientation is a real design freedom — it decides which joints are affected and where cables run
   — and nothing in the tool models it.

7. **What does the locking screw bite into?** The plan is a threaded area on the connector so it can
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
header (frames and handedness) → `src/config.js`'s header (the measured panel section, and why it is
parameters rather than a shape) → `src/core/v3/connectors.js`'s header (the two-piece clamp) →
`src/core/v3/relax.js`'s header (why relaxation is the only lever left).

Then run the suites to confirm the tree is green, and `npm run dev --prefix grid-designer` to look at
it. Load the `shelf` preset to see the brief satisfied, then `crest` to see the report say no. Click
two adjacent squares in the plan view to combine them and watch the report react. Turn on
**connectors** in the viewport toolbar and drag **part length** from 10 cm to 25 cm — the worst
wedge-per-part goes 0.45 cm → 1.13 cm, which is §2.12 happening in front of you. Then load
`modular` and tick **relax**: 25 joints outside the fastener envelope go to 0 for 0.3 cm of
movement.

§4.2 is now the highest-leverage question left — the joint width decides fastener fit, fold capacity
and how hard the relaxation has to work, all at once. Then §4.6b (bearing on the supply housing) and
§4.6c (panel orientation), both of which the tool measures and cannot decide. §4.1b, §4.3 and the
connector gaps in §5.1 follow.
