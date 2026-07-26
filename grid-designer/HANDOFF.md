# grid-designer — handoff & decision log

Written 2026-07-25 at the close of the first build session. Read
**[README.md](README.md)** first for how the tool works; this document records **why it is the way
it is**, what was tried and rejected, and what is still open.

The project is about to take **a large pivot in direction**, so this is deliberately organised to
separate durable findings (physical facts, geometric conclusions, dead ends) from the current
implementation. If the pivot discards the tool, the "Durable findings" and "Rejected approaches"
sections are still worth keeping.

Latest commit at handoff: `7b9082c`. Working tree clean, all tests green (1137 headless + 225
browser checks).

---

## 1. Where things stand

The tool is **functional and tuned through several rounds of real use.** You can load a preset,
reshape it column by column, merge and split plates from the plan view, watch the physical rules
being checked live, measure the result, and export OBJ.

What was built, in order (one commit each — the messages carry the detail):

| commit | what |
|---|---|
| `0f1e3a8` | initial tool: schema, placement solver, joint report, 3D viewer, plan view, JSON I/O, OBJ export |
| `4292cfb` | **transpose the fold model to column strips (schema v2)** — the first model was wrong |
| `caceb5a` | end-panel grounding rule + auto-ground solver |
| `f95b530` | default joint gap 2 cm → 1 cm |
| `a941f9a` | ≥4 plates in a named pattern, every preset |
| `e25a5b5` | storm preset family: pitched fronts, wall engagement, adjustable rows |
| `e94b53d` | rebuild the panel geometry (framed diffuser front, enclosed tapered back) |
| `06d3377` | swell: steep window rows, crest peaking at the wall |
| `81c6604` | measuring box with cm dimensions; swell retuned as a swell/drift hybrid |
| `565af13` | combine two squares into a plate from the plan view + undo/redo |
| `7b9082c` | swell: crest one row deeper, 8 plates, height held at ~117 cm |

---

## 2. Durable findings

Things learned that hold regardless of what the tool becomes.

### 2.1 The gap is what makes the geometry tractable

Rigid rectangles sharing vertices cannot fold in two directions at a vertex. But the real panels
**never share a vertex** — every joint has a ~1 cm gap spanned by a 3D-printed connector, and that
slack absorbs the incompatibility. This is the single most important insight in the project:

> Don't solve rigid origami. Place panels deterministically and **measure** what the connectors
> have to absorb.

Everything downstream follows from it — the pure placement function, the joint report, and the fact
that "impossible" configurations are reported as tolerances rather than refused.

### 2.2 1 cm gap makes the kit modular exactly

60 + 1 + 60 = 121. With a 1 cm joint, a 60×121 plate is a **precise** drop-in for two squares and
their joint: no slack propagates, and a plate placed mid-chain leaves everything behind it
bit-identical. At the original 2 cm gap there was a 1 cm mismatch per plate that had to be
surfaced as a warning. If the connector design ever changes the joint width, this exactness is what
you lose — the `W_RECT_LENGTH` warning still exists to catch it.

### 2.3 Folding runs along columns, not rows

Window → back, each column an independent chain. Row-wise joints are plain gap connections. This was
**corrected after the first real test** and is the defining structural fact of the model. See
"Rejected approaches" for what the wrong version looked like.

### 2.4 Height and depth trade against each other

A column strip has a fixed chain length (its panel count × pitch). Height it doesn't spend climbing,
it spends receding. Concretely, across the swell iterations:

| swell version | peak | depth |
|---|---|---|
| symmetric arc | 106 cm | 384 cm |
| peak at wall | 151 cm | 384 cm |
| lower + drift | 116 cm | 417 cm |
| crest one row back (current) | 117 cm | **478 cm** |

**Lowering the peak makes the footprint deeper.** Any brief that constrains both needs the row count
as the third lever, and row count costs depth too. This is the live tension — see open questions.

### 2.5 Only the panels behind the crest can land the strip

From a crest height *H*, the panels between the crest and the back end must bring the surface down.
Two 60 cm panels can drop at most 120 cm, and only if near-vertical — which looks like a cliff, not
a wave. This arithmetic determined the row count of every storm preset, and it is the first thing to
check when someone asks for a taller crest.

A 121 cm plate at the landing is worth roughly twice a square's reach, which is why `surge` can hold
a 170 cm crest and still stand on the floor.

### 2.6 A plate pins its joint flat

A vertical plate spanning rows *r*, *r+1* makes them one rigid piece, so hinge *r* must be 0°. A
horizontal plate spanning two columns requires both to agree in pitch *and* position at that row.

The consequence for authoring: **design the plateau into the pitch profile first, then place the
plate.** Searching a finished profile for legal plate positions almost always finds none. Every
preset is built this way.

The consequence for editing: merging two squares has to *change geometry*, which is why the merge
action performs the adjustment and reports it rather than refusing (see §3.6).

### 2.7 Site facts (from the wider repo, for reference)

- The existing installation is 12 panels — 4 units × 3 subpanels — in a Toronto storefront window;
  ran 24/7 for 54 days, ~1M pedestrian inputs. `IO/DROPCEILING_STORY.md` is the best overview.
- Two columns on the sidewalk create an alcove, dividing the world into an active zone (people who
  step in) and a passive zone (people passing). `IO/diagrams/src/B3_spatial_plan.dot` is the
  to-scale plan.
- **Unresolved inconsistency in the repo:** `IO/world_coordinates.json` and the running controller
  `IO/lightController_osc.py` disagree about subpanel positions and angles (the JSON says
  `angle: ±30°` with one set of offsets; the controller uses `±22.5°` with another). The public
  viewer mirrors the controller. Nothing in grid-designer depends on either, but **if V2 planning
  ever has to reconcile against V1 as-built, resolve this first.**

---

## 3. Decision log

### 3.1 Fold model: column strips (schema v2)

The first implementation folded **along rows** — each row an accordion zig-zagging in plan, with
row-to-row dihedrals. After the first test the direction was corrected: folding happens **only along
columns**, window to back; row-wise joints are simple connections; columns stay edge-aligned in X.

`version: 2` is mandatory and v1 configs are **rejected on import** rather than migrated — the two
models don't map onto each other and a silent reinterpretation would be worse than an error.

### 3.2 Grounding is a violation, not a validation error

**Rule:** every column's last panel must come back down and touch the floor — one housing edge down
or lying flat, never floating. "The wave returns to the water."

It is enforced as a **layout violation** (`E_END_FLOATING`), not a validation error, because a hard
error would reject every intermediate state of a slider drag and make exploration impossible.
Instead it is loud: red badges per column, a violation box, plan-view markers, and a **Ground end /
Ground all** solver.

Measured over the end panel's **8 solid corners** (lit face + housing), so both "edge down" and
"lying flat" satisfy it. `groundTolerance` (default 0.5 cm) is a config field.

**Exception:** column 0 may be `endSupport: 'wall'` — side-bracketed to the wall, exempt from the
rule, so a strip can end high against it. This is `wallcrash`'s defining trait and is deliberately
*not* used by the other presets.

### 3.3 The wall is at x = 0, beside column 0

First implemented on the opposite edge (x = 365) and corrected. Column 0 is the **rightmost in the
3D view** and the **left edge of the plan view**. Because the grid is laid out *from* the wall,
`WALL_X = 0` is a constant that doesn't depend on cols/size/gap.

### 3.4 ≥4 plates in a named pattern

The 121 cm plate is half the panel kit; a design using one or two reads as a grid of squares with
mistakes in it. So every generated design places **at least 4 plates by one nameable rule**,
recorded in `meta.rectPattern` and captioned in the plan view. Hand-built configs below 4 get a
non-blocking `W_FEW_RECTS` nudge. The rule is documented in `schema.js` for a future LLM generator.

### 3.5 The flat-shore rule was removed

Row 0 lying flat on the floor was originally mandatory (`E_SHORE_NOT_FLAT`). When the storm family
was introduced the rule was dropped in favour of per-column `startPitchDeg`, with the chain origin
solved so a pitched front panel still rests on the floor on its contact edge. Flat is now just
`startPitchDeg: 0`, and the calm family still uses it.

### 3.6 Merging coerces geometry and says so

Adding a plate by clicking was effectively impossible: every click was refused because the spanned
hinge wasn't 0 (vertical) or the two columns didn't agree (horizontal) — never true in a designed
profile. The fix was **not** to relax the rules (they're physically real) but to have the merge
perform the implied adjustment: flatten the hinge, or match the second column to the first through
that row. Both land with the plate in a single validated commit, and the notice states exactly what
changed.

Because that is a genuine design change, this is also why **undo/redo** exists. Note the deliberate
asymmetry: **splitting a plate does not restore the flattened hinge** — splitting only says "two
panels here", not "put the bend back". Undo does that. This is documented in `merge.js`.

Invariant worth preserving: `mergeCandidates` classifies by *actually running* `mergeCells`, so the
plan view can never offer a merge the model would refuse. A sweep test over every cell asserts it.

### 3.7 Panel geometry was rebuilt, not patched

The panel solid inherited from `panel-designer` had its **front and back caps wound inside-out**
(measured: normal −Y at y = 0, +Y at y = −3.7), so both were backface-culled and read as missing
faces; its recessed lip ring was also buried under a full-footprint front face, so there was no
frame/diffuser distinction at all.

Rebuilt as a closed 2-manifold: frame flange around all four edges → reveal → recessed diffuser →
inward-tapering housing → inset back plate. Two material groups so the diffuser is the light source
and the frame/housing is matte.

Dropped in the process: `powerSupplyEdge` special-casing (it produced degenerate and coincident
quads on one edge) and a half-implemented double-sided mode. Real fixtures have a frame on all four
sides.

`tests/test-geometry.mjs` guards it — edge pairing with opposite traversal, Euler characteristic on
the welded mesh, signed volume against a closed-form derivation, cap normals, group partitioning —
and it was **mutation-tested**: reproducing the original inverted-cap bug trips 8 orientation errors
and breaks the volume match.

### 3.8 Materials need a visibility floor

Both scene lights are above the surface, so a folded-up panel's back gets no direct illumination.
With a truly dark, non-emissive housing the tray backs crushed to pure black and the frame read as
an empty gap between panels. The housing therefore uses mid-grey aluminium with a small emissive
floor (0.16). Documented in `SurfaceMeshes.jsx` — don't "fix" it back to black.

### 3.9 The swell tuning history

Four rounds, each driven by looking at the render. Recorded because the rejected states are
informative:

1. **Symmetric arc** — crest 70 → 105 → 70 cm, fronts 10–25°. *Rejected:* the two window rows read
   as flat, and it peaked mid-grid rather than at the wall.
2. **Peak at the wall** — fronts 45 → 28°, crest 151 → 82 cm. *Rejected:* too tall.
3. **Lower + drift hybrid** — fronts 42 → 20°, crest 116 → 42 cm, shore counter-bend added, lee made
   1.34–1.46× steeper than the windward slope. *Rejected:* crest too close to the window.
4. **Crest one row deeper (current)** — spine plate moved to rows 3–4, 8 rows, 8 plates. Lit faces
   visible from a window-side eye point went from 23/36 to 34/40.

The current concept is **"somewhere between a water swell and a snow drift"**: smooth continuous
curvature with a deliberately articulated shore, a long gradual windward rise, and a distinctly
steeper lee — plus accumulation against the wall, which is what drifts do.

Variants built and rejected with measurements:
- swell at **6 rows**: lands, saves 64 cm of depth, but forces a clamped −90° vertical lee panel —
  a cliff, not a slipface.
- swell at **7 rows with the crest at row 4**: column 0 **doesn't land at all** (`E_END_FLOATING`)
  and worst gap deviation triples to 49 cm.

### 3.10 Working practice that worked well

The build ran as **Fable orchestrating, Opus 5 coding**: each work package delegated with a precise
spec, then independently verified by the orchestrator — re-running the suites, probing the core
functions directly, diffing preset output against the previous commit, and reading the screenshots.

This caught things worth catching: a subagent's summary misreported its own Euler counts and a
volume figure (the code was right, the prose wasn't); another's claim of bit-identical presets was
verified by direct diff rather than trust. **Verify the numbers, not the narrative** — and prefer
structural proof (no diff in `core/`) over output comparison where possible.

---

## 4. Open questions

1. **How deep can the installation actually be?** This is the big one. `swell` is currently 478 cm
   deep — about 4.8 m into the store from the window. Height and depth trade directly (§2.4), so a
   real site constraint would let the presets be solved to it instead of guessed. **Nothing in the
   repo records the available depth.** Currently the choices are: accept 478 cm; lower the peak to
   ~90 cm and fit ~420 cm at 7 rows; or move the crest back to row 3 (417 cm, fewer faces visible).
2. **Is the wall structurally usable for support?** `wallcrash` assumes column 0 can be
   side-bracketed. Whether the storefront wall can take that load is unverified.
3. **How many plates, and where?** The ≥4 rule and the named patterns are the tool's answer, but the
   real constraint is presumably budget/inventory. A known plate count would sharpen the presets.
4. **Reconcile V1 as-built geometry** if V2 planning needs it — see §2.7.
5. **Lost state:** two manually-added plates on a swell config were destroyed by a hot reload before
   they could be read. Their positions are unknown. (See §5.1 — this must not happen again.)

---

## 5. Next steps / known gaps

### 5.1 Persistence — highest value, smallest change

**There is no persistence.** The config lives in memory, so any reload — including the hot reload
from editing a source file — resets to the flat preset. This destroyed in-progress user work
**twice** in one session. Add `localStorage` for the working config plus a few named slots for
parking variants. Until then: **Copy / Download JSON before touching any source file.**

### 5.2 Not built

- **Panel–panel collision detection.** The report measures joints only. Extreme cumulative folds can
  push panels through each other and nothing notices. `panel-designer` has a `three-mesh-bvh`
  implementation worth cribbing.
- **Connector spec export.** `panel-designer/src/utils/exporters.js` has `exportConnectorSpec`
  (groups joints by dihedral into unique connector types against a V1 baseline). The joint report
  already computes everything it needs.
- **Prompt-driven generation.** The whole schema doc-comment in `core/schema.js` is written for an
  LLM audience for exactly this — an LLM emits config JSON, the validator and the ground solver
  keep it honest. Never wired up; no API calls anywhere in the tool.
- **Light behaviour / animation.** The V2 concept is an *agentic* body of water driven by sidewalk
  data; this tool only plans static physical configurations.
- **Deploy.** `.github/workflows/static.yml` only covers `IO/public-viewer/`. grid-designer has no
  deploy path.

### 5.3 Minor known quirks

- For strongly skewed joints, `gapMid` can be smaller than `gapMin` — correct (midpoints can be
  closer than endpoints) but don't present the three numbers as an ordered min/mid/max.
- `layoutBounds` can return `-0` components for `min`. Cosmetic; `size` is what's displayed.
- The Playwright harness spawns its own dev server on 5175 — don't pre-start one or it will warn and
  reuse it.

---

## 6. Picking this up in a new thread

Read in this order: **README.md** → this file → the doc-comment at the top of
`src/core/schema.js` (the canonical schema and angle semantics, written to be read cold) → the
header of `src/core/presets.js` (how designs are authored, and why folds are designed around
plates).

Then run the suites to confirm the tree is green, and `npm run dev --prefix grid-designer` to look
at it.

If the pivot keeps the tool, §4 and §5 are the queue, with persistence first. If the pivot replaces
it, §2 ("Durable findings") is the part that transfers — especially the gap insight, the
height/depth trade, and the landing arithmetic.
