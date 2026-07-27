# grid-designer v3 — the drift shell

**Status: HISTORICAL.** This is the spec as written at the START of the v3 pivot, kept because its
reasoning is still the clearest statement of the model's intent. **It is not current in three
places, and README.md / HANDOFF.md are authoritative where they differ:**

1. **§4 makes the spanning-tree walk the placement model. It is not the default.** One hinge is one
   degree of freedom, so a chained tile matches the target's pitch but never its roll; over eight
   rows the sheet lifts off the floor. The default is `surface-fit`, which shares the misfit across
   all joints — what the connectors physically do. `chain` survives as a comparison mode. See
   HANDOFF §2.2.
2. **The target is no longer the smooth drift.** `target.js` quantizes it into planar facets on the
   panel lattice (`form.angularity`, `form.facetCells`), so the panels can *be* the surface instead
   of approximating it. This is the single biggest change from this document. See HANDOFF §2.1.
3. **§3's `materialToPlanApprox` is no longer used to decide plate fit.** The tiler reads the same
   target `placement.js` seats panels on; when the two disagreed the tiler was blind to faceting
   entirely. See HANDOFF §2.8.

Read **README.md** for how the tool works now and **HANDOFF.md** for the decision log.

---

## 0. The pivot in one paragraph

v2 modelled the installation as **6 independent 2D column chains** folding window → back. Every
column was a strip at a fixed x; the surface was effectively a corrugation, and cross-column joints
just drifted apart. v3 throws that out. The installation is now **one 3D surface tiled by rigid
panels** — a snow drift. Panels pitch *and* roll *and* yaw. Which cells get a 60×60 square and which
get a 60×121 plate is decided by **a tiling algorithm**, not by hand. Nothing is a "row" or a
"column" any more; those words survive only as *material* indices into the sheet.

The durable v2 insight is unchanged and is the whole reason this is tractable:

> Don't solve rigid origami. Place panels deterministically and **measure** what the connectors
> have to absorb.

In v2 the thing being measured was the drift between neighbouring column strips. In v3 it is the
**holonomy of a tiled doubly-curved surface** — the same idea, one dimension up. That measurement is
the tool's primary output. The user's words: *"it will be about gaps and other issues."*

---

## 1. The three coordinate systems

Keeping these distinct is the single most important thing in the v3 code. v2 conflated them and got
away with it because a column strip is 1-dimensional.

| space | symbol | what it is |
|---|---|---|
| **material** | `(u, v)` cm, or `(i, j)` in cells | position on the *unrolled flat sheet*. The tiling lives here and is an exact lattice: cell `(i,j)` owns `u ∈ [i·pitch, i·pitch + size]`, `v ∈ [j·pitch, j·pitch + size]`, `pitch = size + gap = 61`. Gaps are exactly `gap` here **by construction**. |
| **plan** | `(x, z)` cm | the floor. The **target form** `H(x, z)` is authored here, because that is what a person sculpting a drift thinks in. |
| **world** | `(x, y, z)` cm | where the panels actually are, Y up. Produced by the placement walk. |

`i` runs across the window (+X, wall side first), `j` runs window → back (+Z). These replace
"column" and "row"; **the code must call them `i`/`j` or `u`/`v`, never row/col**, so that a reader
can never confuse a v3 material index with a v2 grid position.

The sheet is **longer than its shadow**: a curved surface's plan projection is smaller than its
developed area. Never assume a cell's world position from its material index.

---

## 2. The form — `src/core/form.js`

A parametric **drift heightfield** `H(x, z) ≥ 0` over the plan, plus its analytic gradient.
Authored, not solved. `H = 0` outside the footprint (material that runs past the drift lies flat on
the floor, which is correct and looks right — the drift feathers out).

### 2.1 Requirements from the brief

- **The wall edge (x = 0) and the window edge (z = 0) both meet the ground, and are NOT flat there.**
  So `H(0, z) = H(x, 0) = 0` with **non-zero slope** at both. This is the user's "left edge and
  window edge start from the ground but not flat" and "the row on the left must touch the ground".
- It must read as **a snow drift**, not a bump: a long gradual windward rise, a distinctly steeper
  lee, a crest that is *offset*, and accumulation that varies along the ridge.

### 2.2 The profile primitive

A one-dimensional asymmetric bump on `[0, 1]` with a controllable non-flat toe:

```
B(τ; p, a) = normalize( τ^a · (1 − τ)^b ),   b = a·(1 − p)/p
```

- peaks at `τ = p` (crest position), normalized so `max B = 1`
- **toe slope at τ = 0 is non-zero iff `a ≤ 1`** — `a = 1` gives a finite linear toe, `a < 1` a
  steeper one, `a > 1` a *flat* toe which the brief forbids. **Clamp `a ∈ [0.45, 1.0]` and say why
  in a comment.** This clamp is the brief's "not flat" made mechanical.
- `b` follows from `p`, so each axis costs exactly two knobs: crest position and toe sharpness.

### 2.3 The drift

```
s = x / footprint.width      (0 at the wall)
t = z / footprint.depth      (0 at the window)

tc(s) = clamp01( crestZ + ridgeShear · s )          ← the ridge runs diagonally
H(x, z) = amplitude · B(s; crestX, toeA_x) · B(t; tc(s), toeA_z)
```

Both factors vanish at `s = 0` and at `t = 0`, so **both grounded edges come out of the form for
free** rather than being special-cased. `ridgeShear` is what stops it reading as a symmetric mound:
the crest line walks back as it goes away from the wall.

Outside `s, t ∈ [0, 1]`, `H = 0`.

Config knobs (all in `config.form`): `amplitude`, `crestX`, `crestZ`, `ridgeShear`, `toeSharpX`,
`toeSharpZ`, `footprint: { width, depth }`.

### 2.4 Exports

```js
driftHeight(form, x, z)        → number            // H
driftGradient(form, x, z)      → [dH/dx, dH/dz]    // ANALYTIC, not finite-difference
driftNormal(form, x, z)        → THREE.Vector3     // normalize(-dH/dx, 1, -dH/dz)
driftFrame(form, x, z)         → { point, normal } // convenience
sampleDriftMesh(form, nx, nz)  → { positions, indices }  // for the ghost surface in the viewport
```

`driftGradient` must be the analytic derivative of §2.3. A test asserts it against a central
difference to 1e-5 at a spread of interior points — this is the kind of sign bug that has bitten
this project before.

---

## 3. The tiling — `src/core/tiling.js`

Cover the material cell grid `cols × rows` with **1×1 squares (60×60)** and **1×2 plates
(60×121)**. In material space a plate spans two cells *plus the joint between them*:
`60 + 1 + 60 = 121` exactly — v2's §2.2 exactness carries over unchanged and is the reason the kit
stays modular.

This is a **domino tiling of the cell grid**, and it is where the user's "all choices about square
or rectangular should be made by the tiling algorithm" lives. It must be:

- **deterministic** — same config in, identical tiling out
- **form-driven** — the choice comes from the surface, not from a random number
- **nameable** — a strategy string goes in `meta.tilePattern`, because "the installation is also
  making a point": four plates scattered read as mistakes, four plates following a rule read as a
  system. This is v2's `MIN_RECTS`/`W_FEW_RECTS` rule generalized and it survives intact.

### 3.1 Strategies

Each scores every candidate domino (a cell and one orthogonal neighbour), then places
highest-score-first, skipping any that conflict with an already-placed piece, then fills the
remainder with squares. Ties break on `(i, j, orientation)` so the result is stable.

| id | rule, said out loud | score |
|---|---|---|
| `flat-lie` (default) | *a rigid plate needs a flat place to lie* | negative magnitude of the target surface's second derivative along the domino's long axis, sampled at its two cell centres |
| `ridge-aligned` | *plates run along the crest, squares take the bends* | alignment of the domino's long axis with the ridge tangent, weighted by proximity to the crest line |
| `toe-bands` | *the two grounded edges are banded by plates* | proximity to `i = 0` or `j = 0`, long axis parallel to that edge |

`flat-lie` is the default because it is the one with a physical reason behind it, and a physical
reason is the most legible kind of point to make.

Enforce `MIN_PLATES = 4` as in v2 (warning `W_FEW_PLATES`, non-blocking).

### 3.2 Output

```js
solveTiling(config) → {
  tiles: [{ id, cells: [[i,j], …], type: '2x2'|'2x4', axis: 'u'|'v',
            uv: { u0, v0, uLen, vLen } }],   // material-space footprint
  adjacency: [{ a, b, edge: {…}, axis, materialLength }],   // every shared material edge
  pattern: 'flat-lie',
  warnings: [...],
}
```

`axis` is which material direction the plate's 121 cm side runs along. Adjacency is over
**material** edges — two tiles are neighbours iff their material footprints share a boundary
segment of non-zero length. Compute it in material space where it is exact and trivial; never
infer neighbours from world positions.

---

## 4. Placement — `src/core/placement.js`

**This is the heart of v3.** It replaces `columnChain`.

### 4.1 Why a spanning tree

A rigid tile attached to an already-placed neighbour across a shared edge, at a fixed gap, has
**exactly one degree of freedom: the dihedral about that edge.** So if we place tiles in the order
of a **spanning tree** of the adjacency graph, every tree edge can be made *exact* — exactly `gap`
cm, zero skew, no lateral slip — and each tile's one remaining freedom is spent matching the target
surface.

The error then has nowhere to hide: it lands entirely on the **non-tree edges**, the ones that close
the cycles. That residual is the **holonomy** of the surface — the honest, unavoidable consequence
of tiling a doubly-curved form with rigid flat plates. Measuring it *is* the tool's job.

A v2 column chain is exactly this algorithm on a 1-dimensional graph. v3 is the same idea, one
dimension up.

### 4.2 The walk

1. **Root** at the tile containing material cell `(0, 0)` — the wall + window corner, the corner the
   brief pins to the ground.
2. Place the root: orientation from the target surface over its footprint, then **dropped to rest on
   the floor** — translate in Y so the minimum of its 8 solid corners sits at `y = 0`. It is pitched
   and rolled, not flat, which is the brief.
3. **BFS** over `adjacency` (deterministic order: by tile id) to build the tree.
4. For each child across shared edge `E`:
   - `E` in the parent's world frame is a segment; let `â` be its unit direction and `n̂ₚ` the
     parent's outward in-plane normal at that edge.
   - The child's mating edge is `E` translated by `gap · n̂ₚ`, then the child is rotated about the
     axis `â` through that translated edge by the **dihedral θ**.
   - **Choose θ to best match the target normal** at the child's centroid: project the target normal
     into the plane perpendicular to `â` and take the signed angle to the parent's normal. Closed
     form, no search.
   - The centroid depends on θ, so **fixed-point iterate exactly 3 times** from θ = 0. Three, not
     "until converged" — a fixed iteration count is what keeps the function deterministic and
     byte-reproducible. Assert convergence magnitude in a test rather than looping.
5. Emit the tile's world position + quaternion.

Tree edges are exact **by construction**; a test must assert that directly (every tree edge's gap
equals `config.gap` to 1e-6 and its skew is 0).

### 4.3 Tree strategy is a design parameter

Which spanning tree you pick decides **where the error goes**. Expose it:

- `bfs-corner` (default) — BFS from the grounded corner; error spreads roughly evenly outward.
- `comb-v` — a spine along the window edge, then independent chains running back. **This reproduces
  the v2 model exactly**, dumping all error on the cross-sheet joints. Keep it: it is the honest
  way to show what the pivot bought, and it is a regression check on the whole idea.
- `comb-u` — the transpose.

This is a genuinely useful control, not a toy: it lets the user choose which joints get the slack.

### 4.4 Grounding

Grounding is no longer per-column. The surface is **a shell that rests on its grounded boundary**.
Report, don't force:

- per tile: `minY`, `grounded` (minY ≤ `groundTolerance`)
- the **support footprint**: the convex hull, in plan, of the ground-contact points
- **`E_UNSUPPORTED`** when the whole assembly's centre of mass does not project inside that hull —
  the "does it stand up" check, which per-column grounding could never express
- **`W_EDGE_FLOATING`** when a tile on the wall edge (`i = 0`) or window edge (`j = 0`) is not
  grounded, since the brief pins those two edges to the floor
- **`W_TOE_FLAT`** when a grounded boundary tile is within a few degrees of horizontal — the brief
  says grounded *but not flat*

Violations, not validation errors — same reasoning as v2 §3.2 (a slider drag passes through bad
states and the store only commits valid configs).

---

## 5. Report — `src/core/report.js`

Per adjacent tile pair (from `tiling.adjacency`), measured on the placed solids:

- `gapMin` / `gapMid` / `gapMax` — edge-to-edge separation. (v2 quirk still applies: for skewed
  joints `gapMid` can be below `gapMin`; do not present them as an ordered triple.)
- `skewDeg` — angle between the two edge lines
- `dihedralDeg` — the fold across the joint
- `treeEdge` — boolean. **Non-tree edges are where the story is.**
- flags against `gapTolerance`

Plus three new whole-surface measures, all new to v3:

1. **Holonomy summary** — total and worst-case closure error over non-tree edges, and where it
   concentrates. This is the headline number of the tool.
2. **Surface deviation** — per tile, the max distance of its corners from the target `H`. Says
   whether the tiling still reads as the drift that was authored.
3. **Collisions** — pairwise panel interpenetration by **OBB–OBB SAT** (15-axis, exact for boxes;
   AABB broad-phase first). v2 listed this as "not built" and got away with it because column strips
   could not intersect each other. **In v3 they absolutely can, so this is required, not optional.**

---

## 6. Config (schema v3)

`version: 3`, and **v2 configs are rejected on import**, exactly as v1 was — the models do not map
onto each other and a silent reinterpretation is worse than an error (HANDOFF §3.1).

```jsonc
{
  "version": 3,
  "units": "cm",
  "name": "drift study 1",
  "sheet": { "cols": 6, "rows": 8 },      // MATERIAL cell counts (i, j)
  "cell": { "size": 60, "plateLength": 121 },
  "gap": 1.0,
  "form": {
    "amplitude": 120, "crestX": 0.62, "crestZ": 0.55, "ridgeShear": 0.18,
    "toeSharpX": 0.8, "toeSharpZ": 0.9,
    "footprint": { "width": 365, "depth": 430 }
  },
  "tiling": { "strategy": "flat-lie" },
  "placement": { "tree": "bfs-corner" },
  "gapTolerance": 1.5,
  "groundTolerance": 0.5,
  "meta": { "preset": "drift", "tilePattern": "flat-lie", "notes": "" }
}
```

Ranges: `cols` 4–8, `rows` 5–10, `amplitude` 0–250, `crestX`/`crestZ` 0.15–0.85, `ridgeShear`
−0.4–0.4, `toeSharp*` 0.45–1.0, `gap` 0–10.

---

## 7. What is kept, what goes

**Kept unchanged:** `src/config.js` (panel dimensions + profile), `geometry/panelGeometry.js` (the
panel solid — it was rebuilt and mutation-tested in v2, do not touch it), `utils/exporters.js`
(OBJ/JSON; it bakes world transforms, which is model-agnostic), the Viewport/scene shell, the
`WeakMap`-memoized store contract, and **every testing convention**: plain node scripts,
closed-form expectations rather than golden numbers, determinism assertions, Playwright screenshots.

**Kept and enforced:** the headless-core contract — explicit `.js` extensions, `three` math classes
only, pure and deterministic. This is called out in the v2 README as the most valuable convention in
the project and it is not up for renegotiation.

**Deleted:** `core/ground.js` (per-column grounding is gone), `core/merge.js` (the tiler decides now;
manual merge/split may return later as an override), `components/ColumnControls.jsx`, and the v2
`schema.js` / `placement.js` / `report.js` / `presets.js` internals.

**Open, deliberately deferred:** per-panel connector design and the connection network. The user was
explicit — *"This entire network will be worked out with connections per panel, but we will figure
that out after."* v3 measures what the connectors must absorb; it does not design them.

---

## 8. Work packages

| # | package | boundary |
|---|---|---|
| P0 | persistence (localStorage + named slots) | on the v2 tree, before anything else — hot reload has destroyed user work twice |
| P1 | `form.js` + tests | pure math, no schema dependency |
| P2 | `schema.js` v3 + `tiling.js` + tests | no placement yet |
| P3 | `placement.js` + tests | no report, no UI |
| P4 | `report.js` (incl. SAT collisions) + tests | no UI |
| P5 | UI: viewport, form controls, tiling plan view | no presets |
| P6 | drift presets + shape tuning against renders | — |
| P7 | README/HANDOFF rewrite, screenshot suite, final verification | — |

Each package is committed separately with the reasoning in the message, and independently verified
by the orchestrator before the next is dispatched — **verify the numbers, not the narrative**
(HANDOFF §3.10).
