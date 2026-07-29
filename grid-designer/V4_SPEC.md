# V4_SPEC — the folded network

**Status: the active model.** v3 (author a drift surface, tile it, measure the damage) is retired —
see [HANDOFF.md §0](HANDOFF.md). This document specifies what replaces it.

> **Read §9 first if you are working on the current model.** §1–§8 specify the one-dimensional
> **ribbon**, which was built first and is still exactly right — it is the `cols: 1` case. §9
> generalises it to the two-dimensional **network** on a lattice, which is what the tool now builds,
> and it changes what you *edit* (flat cells, with the angled panels derived) without changing any
> of the geometry below.

> **The inversion in one line.** v3 chose a form and then asked whether the panels could be it. v4
> chooses *folds the connectors can already build* and lets the form be whatever those compose into.
> Every joint in a v4 design is inside the connector envelope **by construction**, so there is no
> reconciliation step, no relaxation solver, and nothing to "fit".

---

## 0. What carries over unchanged

Per HANDOFF §5.3, and these are imported, never restated:

| module | what for |
|---|---|
| `src/config.js` | the measured panel — nine caliper parameters, `panelSectionRings()`, `POWER_SUPPLY` |
| `src/geometry/panelGeometry.js` | the panel solid and the supply box |
| `src/core/v3/connectors.js` | the two-piece bolted clamp, its profile, **and the joint feasibility envelope** — `foldLimitDeg`, `fastenerGapNeededCm`, `gapAtDepthCm`, `connectorStationFlags`, `connectorEndProfiles`, `connectorOBB`, `sectionFouling`, `CONNECTOR_LIMITS`, `stationCount` |
| `src/geometry/connectorGeometry.js` | the lofted part solid + `connectorTransform` |
| `src/core/v3/collide.js` | 15-axis OBB SAT, `findCollisions` |
| `src/persistence.js` | localStorage working config + named slots |
| the world conventions | cm, Y up, **window/shore at z = 0**, **wall plane at x = 0** |

What is *not* carried over: `form.js`, `target.js`, `tiling.js`, `placement.js`, `relax.js`,
`presets.js`, `report.js`, `schema.js` — all of `src/core/v3/` except `connectors.js` and
`collide.js`. They stay on disk as the record of the retired approach; nothing in v4 imports them.

---

## 1. The object

**A strip is an open chain of rigid panels, folded in a vertical plane, running from the window
(z = 0) away toward +z.** Panels are 60 × 60 (`2x2`) for now. Strips sit side by side along x,
the first one against the wall (x = 0).

The pattern, starting at the window, is a **trapezoid wave** of period 4:

```
unit k      1     2     3     4     5     6     7     8     9
glyph       _     /     -     \     _     /     -     \     _
role      base  rise  high  fall  base  rise  high  fall  base
tilt α      0    +θ     0    −θ     0    +θ     0    −θ     0
```

- `base` and `high` are both **flat** (α = 0). They are named apart because
  - `base` units are the ones that should land on the floor, and
  - **`high` units are the branch anchors** — the future sideways build runs out of units 3 and 7,
    angled back down to the floor, to meet the next strip and make the 3D network. v4 marks them;
    it does not yet build them.
- `rise` and `fall` are the angled units. **They share one angle θ** and are symmetric. θ is the
  single shape parameter.

Every fold in this pattern therefore has magnitude **θ**, not 2θ — the flats between the angled
units split each direction change in half. That is the whole reason the pattern buys height cheaply.

---

## 2. Chain kinematics

Per strip, work in the profile plane `(z, y)`. Unit `k` has tilt `α_k` from horizontal.

```
u_k = ( cos α_k , sin α_k )        the chain direction
n_k = ( −sin α_k , cos α_k )       the lit normal (u rotated +90°)
```

The chain is a polyline **on the reference plane** — the panels' lit-face plane, local `y = 0`.

```
S_1     = ( windowOffsetCm , 0 )
E_k     = S_k + L · u_k                       L = 60 (panel length along the chain)
S_{k+1} = E_k + gap · ŵ_k ,  ŵ_k = normalize( u_k + u_{k+1} )
```

**Why the bisector.** `|S_{k+1} − E_k| = gap` exactly, and the joint is symmetric about `ŵ_k`, so
**the rim-to-rim span on the lit face is exactly `gap` at every joint, always.** That is the property
the whole model is built to have: `gap` stops being an outcome to be measured and becomes an input
the design is guaranteed to honour.

`u_k + u_{k+1}` cannot vanish (|α| ≤ 75°), so `ŵ_k` is always defined.

### Grounding

After the chain is built, translate the whole strip in y so the lowest point of any **present**
panel solid sits at `y = 0`. Governed by `placement.groundToFloor` (default `true`); when off, the
reference plane of unit 1 sits at `y = 0` and panels may go below the floor (which the report flags).

### World placement

Strip `s` spans `x ∈ [x0(s), x0(s) + W]` with `x0(s) = wallOffsetCm + s · (W + gap)`, `W = 60`.

For unit `k` of strip `s`:

- **position** = the centre of its reference-plane segment:
  `( x0(s) + W/2 , C_k.y , C_k.z )` where `C_k = (S_k + E_k) / 2`.
  This is where `buildPanelGeometry` wants its origin — the centre of the lit face.
- **quaternion** = a rotation about **world +X**:
  - not flipped: `R_x(−α_k)`
  - flipped:     `R_x(180° − α_k)`

  Check the unflipped case: local +Z → `(0, sin α, cos α)` = `u_k` ✓, local +Y → `(0, cos α, −sin α)`
  = `n_k` ✓, local +X → `(1,0,0)` ✓, and `X × Y = Z`, so the basis is right-handed. A left-handed
  basis here would corrupt every collision box exactly as it did for v3 plates (HANDOFF §6) — the
  test suite must check handedness explicitly.

### Flip

`flipped` rotates the panel 180° about its own width axis, which is world +X. The panel stays in the
same plane and keeps the same footprint (it is centred, and square); what changes is which way the
lit face points and which side the housing extrudes to:

| role | not flipped | flipped |
|---|---|---|
| `base` / `high` (flat) | lit face **up** | lit face **down** |
| `rise` / `fall` (angled) | lit face **out** (away from the assembly) | lit face **in** |

The UI must label the toggle with the contextual word, not "flipped".

**A flip is a physical statement about the joint, not a display option.** The connector grips the
back flange, so a joint whose two panels face opposite ways has its two flanges on opposite sides
and **no connector of this family can span it**. v4 emits no station for such a joint and reports
`W_JOINT_FLIP_MISMATCH` against it. Do not fake a part for it.

### Removal

`present: false` removes a unit. **The chain kinematics are unchanged** — every other unit keeps the
position it had. Removing a panel opens a hole; it does not re-solve the strip. The two joints that
touched the removed unit cease to exist.

---

## 3. Joints and the envelope

A joint exists between consecutive units `k`, `k+1` when **both are present**.

| quantity | value |
|---|---|
| `spanCm` | `gap`, exactly. Constant along the joint. |
| `spanMinCm` / `spanMaxCm` / `spanStartCm` / `spanEndCm` | all `gap` |
| `spanSpreadCm` | `0` — the rims are parallel, so a v4 part never wedges |
| `twistDeg` | `0` |
| `foldDeg` | signed, **convex (ridge, housings pinching) positive**. Closed form: `α_k − α_{k+1}`. |
| `dihedralDeg` | `|foldDeg|` |

`foldDeg` must be computed with **v3's formula**, not the closed form — build `p̂ / q̂ / r̂` the way
`solveConnectors` does (`r̂` along the joint, negated if `q̂` disagrees with `n_A + n_B`;
`foldDeg = −atan2((n_A × n_B) · r̂, n_A · n_B)`), because that is the sign convention
`backHalfProfile` reads. The closed form is what the **test** asserts against.

Sanity: on `_ / - \`, joint 1|2 (base→rise) is **concave** (`−θ`) and joint 2|3 (rise→high) is
**convex** (`+θ`). The tops of units 3 and 7 are ridges; the bottoms are valleys.

### The envelope — and the number v3 could never produce

Feasibility is **not re-derived**. Build real stations and call `connectorStationFlags` from
`core/v3/connectors.js`. It already checks profile self-intersection, `minSpanCm` / `maxSpanCm`,
span spread, the fastener gap **at depth**, the power supply, and section-level fouling
(`W_PANELS_COLLIDE_AT_JOINT`).

On top of that, v4 owes the user the thing v3 never had — **a way to say yes**:

- `maxAngleDeg(gap)` — the largest θ this gap admits with every joint clean, found by bisection on
  the flag set (not by reading `foldLimitDeg` directly: the connector fouls 3–6° before the panels
  do, and the honest limit is where the *flags* start).
- `minGapCm(θ)` — the smallest gap that admits this angle, same way.

Both go in the report and both belong on screen. They are the whole point of the pivot.

Note that concave folds are unconstrained by panel-on-panel contact — the housings *diverge* — so
only the convex joints bind. Expect `maxAngleDeg` to be large at gap ≥ 2 cm (`foldLimitDeg(2) = 90`)
and tight at gap 1 cm (`foldLimitDeg(1) = 39.4°`). This is the trade, now stated as a permission
rather than as damage.

---

## 4. Collisions

- **Non-adjacent panel pairs**: `findCollisions` over per-unit OBBs (`collide.js`). Adjacent pairs
  are excluded — they are the joint's business, and `sectionFouling` judges them exactly.
- **Floor**: any present unit whose lowest OBB corner is below `y = 0` is flagged
  (`W_BELOW_FLOOR`). Only reachable with `groundToFloor` off.
- **Wall**: any present unit reaching `x < 0` is flagged (`W_THROUGH_WALL`).

---

## 5. Metrics — the bounding box, with detail

Keep v3's measuring box (W × H × D) and add the per-axis breakdown the box hides:

- **overall**: `min`/`max`/`size` on all three axes.
- **per strip (column)**, for each `s`:
  - `planRunCm` — z extent of the strip
  - `heightCm` — y extent
  - `developedLengthCm` — `n·L + (n−1)·gap`, the flat material the strip is made of
  - `unitCount`, `presentCount`
  - the **compression ratio** `planRunCm / developedLengthCm` — how much the folding buys
- **per row (unit index k)**, across strips: `xMinCm`, `xMaxCm`, `widthCm`.
- **per unit**: `k`, glyph, role, `tiltDeg`, `present`, `flipped`, `zStartCm`, `zEndCm`, `planRunCm`,
  `yStartCm`, `yEndCm`, `riseCm`, and `branchAnchor` (true for `high`).

---

## 6. Config schema (version 4)

```js
{
  version: 4,
  name: 'fold study 1',
  strip: {
    count: 1,          // strips along x. 1..6. Only strip 0 is built for now.
    units: 9,          // panels per strip. 1..24.
    panelType: '2x2',  // '2x2' | '2x4' — only '2x2' for now
  },
  gap: 2.0,            // cm. 0.4..8
  angleDeg: 30,        // θ, degrees. 0..75
  pattern: {
    kind: 'trapezoid', // the only kind: _ / - \
    phase: 0,          // 0..3 — which glyph unit 1 starts on
  },
  placement: {
    wallOffsetCm: 0,      // strip 0's near edge from the wall plane x = 0. 0..200
    windowOffsetCm: 0,    // unit 1's near rim from the window line z = 0. 0..200
    groundToFloor: true,
  },
  overrides: [           // sparse, per unit; keyed on (strip, unit)
    { strip: 0, unit: 3, present: true, flipped: false, role: 'auto' },
  ],
  connectors: {          // same knobs and defaults as v3
    lengthCm: 10, spacingCm: 50, minPerJoint: 2,
    binSpanCm: 0.5, binAngleDeg: 5,
    powerEdge: 'low', supplyMode: 'relief',
  },
  meta: { notes: '' },
}
```

`role` override values: `'auto' | 'base' | 'rise' | 'high' | 'fall'`. `'auto'` takes the pattern's
answer. An override that is entirely default may be dropped by `normalizeConfig`.

Same **two kinds of defaulting** contract as v3: `normalizeConfig` fills missing fields and is
idempotent; `validateConfig` returns `{ valid, errors, warnings }` and range-checks **every** knob
that has a declared range — including the ones v3 forgot (HANDOFF §5.2).

`persistence.js`'s `EXPECTED_CONFIG_VERSION` moves to `4`, which discards stale v3 working configs
and slots. That is the documented mechanism, not a regression.

---

## 7. Module layout

```
src/core/v4/                 headless zone — explicit .js extensions, three math only,
  schema.js                  same layout in → byte-identical out
  chain.js                   roles, kinematics, world placement, OBBs, bounds
  connectors.js              v4 stations → v3's part machinery
  report.js                  joints, flags, envelope headroom, collisions, metrics
  obstacles.js               the room's boxes, tested against the design (§9.11)
src/v4/
  store.js                   zustand, same contract as v3's store
  AppV4.jsx                  shell
  StripPanel.jsx             units / angle / gap / offsets / grounding
  UnitsPanel.jsx             the per-unit table — role, flip, remove
  MetricsPanel.jsx           bounding box + per-strip / per-row / per-unit detail
  ReportPanel.jsx            joints, flags, the envelope headroom readout
  RibbonViewport.jsx         the 3D scene
  JsonPanel.jsx / SlotsPanel.jsx / ExportButtons.jsx   ported from v3
  objExport.js               the export scene, the OBJ, and its .mtl library
  glbExport.js               the same scene as one self-contained binary glTF
```

### The exported document

Both writers consume one `buildSceneGroup`, so they describe the same scene by construction:

| object | what |
|---|---|
| `diffuser_NNN_<id>` | **one per present panel** — its own object AND its own material, because per-panel brightness is the point |
| `frame` | every panel's housing, merged |
| `connectors` | every printed part, both pieces, merged |
| `power_supplies` | every driver box, merged (omitted when `powerEdge: 'none'`) |

**OBJ + MTL** downloads as two files from one button. The `.mtl` is not optional decoration: an OBJ
whose `usemtl` names resolve to nothing imports as a single merged surface, which is exactly the bug
this replaced. Keep the `.mtl` beside the `.obj`.

**GLB** is one self-contained binary — named nodes, one material per mesh, emissive diffusers, and
`KHR_materials_emissive_strength` once a panel is driven off 1.0. The better import when the job is
per-panel brightness. **Not FBX**: three.js ships no FBX *exporter*, and every FBX target reads GLB.
See HANDOFF §9.10a.

`src/main.jsx` mounts `AppV4`. The v3 UI stays on disk, unmounted.

---

## 8. What is deliberately NOT in this pass

- **The sideways branches off units 3 and 7.** Marked, not built. This is the next package and the
  reason `strip.count` and the `(strip, unit)` override key already exist.
- **Strip-to-strip joints.** With `count: 1` there are none.
- Plates (`2x4`), per-unit angle overrides, cable routing, part labelling.

---

# §9 — THE NETWORK

The ribbon generalises to two dimensions. **§1–§8 above are not superseded** — a strip is the
`cols: 1` case of what follows, panel for panel — but the *unit of design* stops being a panel in a
chain and becomes a **flat cell on a lattice**, with the angled panels derived rather than authored.

The config stays `version: 4`. This is not a pivot; it is §8.6's next package.

## 9.1 The rules, and what they force

The brief:

- flat panels **on the ground** can add angled panels **UP** at any face; an angled-up panel then
  adds a flat.
- flat panels **above the ground** can add angled panels **DOWN** at any face; an angled-down panel
  then adds a flat.
- repeat in x and z, but let the edges be ragged.

Those rules admit **exactly two levels**, and make every flat cell's four neighbours the opposite
level. So the level field is a **checkerboard**:

```
level(i, j) = (i + j + phase) mod 2        0 = ground, 1 = high
```

and **every edge between two flat cells carries exactly one angled panel.**

### Why it closes exactly

The bisector step of §2 collapses to a half-angle. For a joint between a horizontal panel and one
tilted by θ, `normalize(u_flat + u_tilt) = (cos(θ/2), sin(θ/2))` in the (plan, y) plane — because
`1 + cos θ = 2cos²(θ/2)`, `sin θ = 2 sin(θ/2) cos(θ/2)`, and the norm is `2cos(θ/2)`.

So **every** level change costs the same plan distance and the same rise, in x and in z alike:

```
gap step         plan  gap·cos(θ/2)          rise  gap·sin(θ/2)
edge to edge     2·gap·cos(θ/2) + 60·cos θ
level rise   R = 2·gap·sin(θ/2) + 60·sin θ
CELL PITCH   P = 60 + 2·gap·cos(θ/2) + 60·cos θ      identical in x and z
```

At θ = 30°, gap = 2: `P = 115.825228`, `R = 31.035276`. Both agree with the shipped 1-D strip to
nine decimals (`refStart` of unit 3 is `115.825227532`; its height above unit 1 is `31.035276180`).

**The flat cells therefore sit on a uniform square lattice, and every cycle closes with zero
residual.** This is the same guarantee as §2's exact gap, and it is why the network needs no solver.
It is also why the checkerboard is not a stylistic choice: allowing two same-level flats to abut
would put a `gap`-wide edge and a 55.8 cm edge on the same lattice, and the pitch would stop being
uniform unless every level change ran in an unbroken line across the whole grid — HANDOFF §3's
planar-quad trap, arriving by a different door.

### The corner holes are real and intended

A ground cell's +x ramp occupies `x > 60, z ∈ [0,60]`; its +z ramp occupies `z > 60, x ∈ [0,60]`.
The corner region beyond both is occupied by **neither**, so four ramps meet at a lattice corner
without touching and leave a diamond opening. The surface stays open, as asked. Whether their
*housings* clear at large θ is a question for the collision pass, not an assumption — see §9.6.

## 9.2 Cells

Flat cells are indexed `(i, j)`: **i along x from the wall, j along z from the window.**

- plan centre `x = 30 + i·P`, `z = 30 + j·P` (before the offsets of §9.5)
- reference y: `0` for a ground cell, `R` for a high cell
- quaternion: identity, or `R_x(180°)` when flipped
- `branchAnchor` is retired as a concept — every cell is now a branch point. The field goes.

## 9.3 Ramps — derived, never authored

**A ramp needs ONE cell, not two.** It exists when its edge is enabled and **at least one** of the
two cells it joins is present; with neither it would float, and only then is it absent.

The first cut required both, and it was wrong for a reason worth recording: removing a flat silently
took up to four angled panels with it, so the design could not be edited panel by panel — which is
the entire point of the plan editor. It also contradicted the model's own precedent, since a **wall
anchor** (§9.4) is exactly a ramp with nothing at its far end. A ramp held at one end cantilevers
off its single joint, and an open-ended folded surface is made of precisely that.
Edges are named by their low cell and axis: `(i, j, 'x')` joins `(i,j)`–`(i+1,j)`, `(i, j, 'z')`
joins `(i,j)`–`(i,j+1)`.

Let `lo` be the edge's **ground** cell, `hi` its **high** cell, and `ê` the unit plan direction from
`lo` toward `hi` (one of ±X, ±Z). Then:

```
u = cos θ · ê + sin θ · Ŷ            the chain direction, rising
n = −sin θ · ê + cos θ · Ŷ           the lit normal
w = Ŷ × ê                            the panel's WIDTH direction
```

`w` is fixed by right-handedness (`w × n = u`), not chosen. Check both cases the ribbon already
knows: `ê = +Z` gives `w = +X`, which is exactly §2's strip; `ê = +X` gives `w = −Z`. Taking `+Z`
there instead yields `−u` and a left-handed basis — the failure mode HANDOFF §6 records for plates,
so **the suite must assert `X × Y = Z` on every ramp in all four directions.**

- reference segment start = `lo` centre `+ 30·ê` `+ gap·(cos(θ/2)·ê + sin(θ/2)·Ŷ)`, and runs 60
  along `u`. Its far end must land on `hi`'s edge less one gap step — **assert this closes**, do not
  assume it.
- position = the segment midpoint; flip = `R_x(180°)` about local X, i.e. about `w`.

Every ramp is defined **rising from its ground cell**, so `ê`'s sign carries the direction and there
is no separate "fall" case. `role` is reported as `rise`/`fall` relative to increasing i or j purely
for the readout.

## 9.4 The wall anchor

`placement.wallAnchor: 'free' | 'braced'`.

**Braced** adds one extra ramp per present **high** cell in column `i = 0`, descending in −x, with
no cell after it: `ê = −X` from a virtual ground cell at `i = −1`. It lands at ground level, so the
network is propped against the wall line rather than cantilevered off its own edge. Ground cells at
`i = 0` need nothing — they are already on the floor.

These anchor ramps are ordinary panels and take ordinary connectors on their one real joint. **What
they attach to at the wall is not modelled and is an open hardware question** — the geometry says
where the toe lands, nothing more. Do not invent a bracket.

## 9.5 Offsets, and what they now measure

`placement.wallOffsetCm` and `windowOffsetCm` are applied as a **translation of the finished
network**, so that its material's minimum x (respectively z) sits at that offset. They therefore
mean what a tape measure would read — including with the anchor ramps, whose toes become the
minimum x. On a `cols: 1`, `wallAnchor: 'free'` design this is identical to today's behaviour.

`groundToFloor` is applied the same way, in y — but to `placement.yOffsetCm` rather than to 0. See
§9.12.

## 9.6 Report additions

- **per-row / per-column metrics** become genuinely two-dimensional: for each `i`, the column's z
  extent, height and cell count; for each `j`, the row's x extent. This is the "length of each row
  and column" ask, now with something to say.
- **counts by role**: ground flats, high flats, ramps, anchor ramps, and the total panel count.
- **the corner clearance.** Two ramps rising off the same cell are *not* joined to each other, so
  the ordinary non-adjacent collision pass already tests them. Report the **worst corner clearance**
  explicitly as a number rather than waiting for it to become a collision — it is the quantity that
  decides how far θ can go in 2-D, and it has no 1-D analogue.
- the envelope, the front-bar limit and the flip-mismatch rule are unchanged.

## 9.7 Editing

Sparse overrides, all defaulting to present:

```js
overrides: {
  cells: [{ i, j, present, flipped }],
  edges: [{ i, j, axis, present }],       // axis: 'x' | 'z'
}
```

**Every tile is one panel, and one click adds or removes exactly that panel.** Switching a cell off
leaves its ramps cantilevered off their far ends (§9.3); switching an edge off removes just that
ramp. Neither moves anything else: **every other panel keeps a bit-identical position**, exactly as
§2's removal rule requires, because the lattice is generated rather than chained.

**`lattice.cols` / `rows` is a bounding rectangle, not the design** — the design is which cells
inside it are present. The plan editor draws a **ring of empty slots one cell wide all the way
round**, and clicking one grows the rectangle to hold it. Two rules make that add exactly one panel:

- every other new cell slot starts **absent**, and
- every new *edge* slot that does not touch the clicked cell starts absent too. Without the second,
  widening by a column would hang a ramp off every cell of the column beside it (a ramp needs only
  one cell), so one click added seven panels instead of one.

Switching a cell **on** clears any such suppression on edges to its **present** neighbours, so "add
a panel" means "add it and connect it". Edges to an absent neighbour stay off — otherwise a click
would sprout cantilevers into empty space.

### Re-origining

Cell (0,0) is defined as the corner nearest the wall and window, so growing at `i = −1` or `j = −1`
shifts every index by one. Two things move with it, and both are load-bearing:

- **every cell and edge override**, or the design would appear to slide one pitch across the lattice
  while the panels stayed put;
- **`pattern.phase`**, because `level = (i + j + phase) mod 2` — shifting an index by one inverts the
  whole checkerboard unless the phase absorbs it. Phase is 0 or 1, so `−phase ≡ +phase (mod 2)` and a
  plain flip is exactly right. The suite asserts all 15 original cells keep their level after a wall
  -edge growth, *and* that omitting the flip inverts all 15.

The network is anchored by its own minimum x/z to `wallOffsetCm` / `windowOffsetCm`, so growing at
those edges holds the near edge where you put it and moves the rest outward. **`trim`** shrinks the
rectangle back to the cells in use, by the same re-origin rule in reverse.

Growing panel by panel is therefore not a separate mechanism — it is switching a cell on at a free
face, and it lands on the lattice by construction.

## 9.9 Corrections made during implementation

Recorded here rather than edited silently into the text above, because the reasoning matters:

1. **§9.4's `ê = −X` was wrong.** §9.3 defines `ê` as running *from the ground cell toward the high
   one*, and the anchor's ground cell is the virtual one at `i = −1`, so it is `ê = +X`. Taken
   literally the spec built a ramp two pitches out from the wall, touching nothing. The prose
   ("descending in −x") describes what the geometry does; the implementation uses `+X`.
2. **`pattern.phase` is 0..1, not 0..3.** The level field has period 2, so `phase: 3` would have
   silently meant `phase: 1`. `validateConfig` now rejects 2 and 3.
3. **`PANEL_TYPES` is `['2x2']` only.** §9.1's pitch is one number in both axes; a 60 × 121 plate
   has no single plan size, so listing `'2x4'` would be a promise the model cannot keep.
   `solveLattice` throws on a non-square panel as a tripwire.
4. **`foldDeg`'s closed form is no longer `α_A − α_B`.** It is `−θ` at a ramp's ground end and `+θ`
   at its high end, whichever way the ramp runs.
5. **`twistDeg` is `acos(|runA · runB|)`.** Because `ê` carries direction, the two panels at a joint
   can number their width axes antiparallel; the signed form read 180° on half of every network's
   joints.
6. **Corner-pair overlaps are reported, but not as collisions** — see §9.6 and the note below.

## 9.10 The one thing left open

**The corner clearance is a bounding-box verdict, and the box is the wrong primitive there.**

Two ramps off the same cell meet box-corner to box-corner. At that corner the real panel is 1.2 cm
of outer wall; the OBB claims the full 4.1 cm. So the boxes report −0.12 cm of overlap at θ = 30°
(crossing zero at **28.2°** at a 2 cm gap) while a back-plate-only box puts the same pair **9.4 cm
apart**, still 7.2 cm apart at 50°.

The honest test is a section-level one — the `sectionFouling` analogue for a pair that shares no
joint — and no such function exists. Until it does:

- `metrics.cornerClearance` reports the number and `W_CORNER_RAMPS_MEET` says what it means;
- corner pairs are kept **out** of `report.collisions`, because calling a 3 cm-thick box corner a
  panel collision asserts something the primitive cannot support, and would put 16 red pairs on a
  design whose panels are nowhere near each other;
- `report.cornerContacts` lists them separately so nothing is hidden.

**A positive clearance is a guarantee; a negative one is a question.** Unlike the front bar, the gap
buys this back directly: the crossing is 14.0° at gap 1, 28.2° at gap 2, 58.4° at gap 4.

## 9.11 Obstacles — the room, not the design

`config.obstacles` is a list of axis-aligned boxes standing on the floor: structural columns, ducts,
plinths. They are **facts about the room**, in the same category as the wall plane at `x = 0` and the
window line at `z = 0`, and nothing about them is derived from the lattice.

The measured one ships as a **default**, because a design made without it on screen is a design made
against the wrong room:

```js
{ id: 'column', xCm: 380, zCm: 285, widthCm: 50, depthCm: 50, heightCm: 300, anchor: 'corner' }
```

**`anchor` has to be stated, not guessed.** `'corner'` reads `(x, z)` as the box's minimum corner —
what a tape from the window/wall datum to the near face gives — and `'centre'` as its middle. On a
50 cm column the two differ by 25 cm, which is exactly the size of error that survives a review, so
the UI prints the resulting extents (`x 380–430, z 285–335`) next to the inputs.

**Report, never enforce.** A panel running through a column is named (`W_PANEL_HITS_OBSTACLE`),
outlined red in the plan grid, and the column turns red in the 3D view — but the panel is still
placed and still counted. Switching it off is the user's call, the same contract the connector flags
and the plate overrides already follow. `report.obstacles` also carries the **clearance to the
nearest panel** when nothing is hit, so "clear" is a distance rather than a silence.

The overlap test uses the panels' full OBBs, which overstate the real section near the rim. Here that
bias is the right way round and is left alone: a false "this fouls the column" costs one click, a
false "it clears" costs a site visit. Clearance is reported as a **lower bound** for the same reason.

## 9.12 The spacers were dropped, and what replaced them is a plain y offset

> Let's lose the feet under the flat panels, that was a failed idea, but add a simple slider to
> adjust the entire system in y

**What was here.** A `src/core/v4/spacers.js` that stood a post under every floor-resting flat cell,
two per edge by the connectors' own spacing rule — 64 of them on the default design — plus a
`report.spacers` cross-check, a viewport toggle, a metrics count, and their own group in the OBJ and
the GLB. It answered an earlier brief: *everything laying 'flat' on the ground needs a 15cm spacer or
gap, matching the pattern of the other connectors.*

**Why it is gone.** The user looked at the feet and judged the idea failed. That is the whole reason,
and it is recorded rather than dressed up as a technical finding — nothing in the model was wrong
with them. They are deleted rather than hidden behind a flag: a part nobody wants is not a setting,
and `git` (commit `71f8b15` built them) is the record. The one thing worth carrying forward is that
the module was *measuring* undersides rather than testing `level === 0`, and that habit outlived it.

### What is left: `placement.yOffsetCm`

The gap half of the old §9.12 was never a part — it was grounding, and it survives under an honest
name. `placement.yOffsetCm`, **default 15**, range **−100 .. 400**. When `groundToFloor` is on the
network is translated in y so its **lowest present material** sits at the offset. It is a **rigid
translation of the whole network**, exactly like `wallOffsetCm` and `windowOffsetCm` — moving only
some cells would tear every joint between them, because the network is one rigid assembly. Nothing
inside the design changes shape when it moves, and `tests/test-v4-lattice.mjs` §10 asserts precisely
that: every panel corner, minus the offset, is identical at −50, 0, 15 and 200.

The band is the ROOM, not a part. 0..50 was the height a spacer could sensibly be. What bounds it now
is the wall the design hangs against: the mullions run to 375cm, so 400 clears the top of the tallest
one with the deepest network still under it, and the negative end lets a design be sunk below y = 0.

With `groundToFloor` **off** the offset is inert. There is nothing anchoring the design in y for it
to be measured from, so it changes nothing.

### The legacy key

`normalizeConfig` still reads **`placement.groundClearanceCm`** and folds it into `yOffsetCm`, and
`EXPECTED_CONFIG_VERSION` was deliberately **not** bumped. The shape did not change — only a name and
a band — so a bump would discard every saved slot and the working localStorage config rather than
migrate them. The new key wins when both are present. `tests/test-v4-schema.mjs` §L pins it.

## 9.14 The wave — a varying angle, and the proof the checkerboard cannot have one

`pattern.kind` gains a second value. **`trapezoid` is unchanged and stays the default**; everything
in §9.1–§9.13 above describes it, and its solve is asserted byte-identical to the commit before this
section existed (`tests/test-v4-wave.mjs` §8, seven configs, FNV-1a over `JSON.stringify`).

### The requirement

> a wave based angle, but in the simplest way possible. base angle at the front which will be the
> lowest, but the mesh needs to compress in a simple linear fashion. nothing gets out of basic
> alignment, but there should be % scrunching in the rows and columns towards a basic attractor.
> everything stays linear, with the new gap

### THE IMPOSSIBILITY — read this before trying to make the checkerboard scrunch

**A two-level checkerboard cannot carry a varying angle.** This is proven and measured, not
suspected.

"Nothing gets out of basic alignment" means the plan grid stays a **product grid**: the x of column
`i` must not depend on `j`. That forces the x-ramp angle to depend only on `i` and the z-ramp angle
only on `j`. Now walk any 4-cycle `(i,j) → (i+1,j) → (i+1,j+1) → (i,j+1) → (i,j)`. On a checkerboard
the levels alternate around it, so the four height steps are `+R(θx(i))`, `−R(θz(j))`, `+R(θx(i))`,
`−R(θz(j))`, and closure demands

```
2·R(θx(i)) − 2·R(θz(j)) = 0        ⟹        every angle equal
R(θ) = 60·sin θ + 2·gap·sin(θ/2)
```

Measured residuals at gap 2, in cm:

| x-ramp | z-ramp | loop left open by |
|---|---|---|
| 30° | 35° | **9.164** |
| 30° | 40° | **17.800** |

The escape hatch of varying the **gap** to hold `R` constant is dead too. Holding `R = 31.035` (the
30°/2cm value) needs:

| θ | gap needed |
|---|---|
| 20° | **30.27 cm** |
| 25° | **13.12 cm** |
| 35° | **−5.62 cm** |
| 40° | **−11.01 cm** |

Every one outside the connector envelope's 1–8cm, and two of them negative. There is no such
network — not "it is hard to find".

### What works: the separable field

```
h(i, j) = f(i) + g(j)
```

makes the 4-cycle residual an algebraic identity rather than a constraint, so the angles are free.
Measured over a lattice with 5 different x-angles and 6 different z-angles: worst 4-cycle residual
**7.1e-15 cm**.

This is exactly the family **HANDOFF §3 rejected** — and the rejection still stands *for the problem
it was about*. v3 needed a field that was zero along two intersecting edges and a mound in between,
and this family cannot be that. v4 is not asking for a mound; it is asking for a foldable network,
and separability is the exact condition for one. HANDOFF §5.3's closing paragraph predicted this
would be the move.

**Consequence, and it must not be "fixed":** with `f` and `g` each zig-zagging by ±R, `h` takes
three values (0, R, 2R), not two. Cell (1,1) is two rises up where the checkerboard would put it
back on the floor. **Even at zero scrunch the wave is a uniform egg-crate with three storeys, not a
checkerboard.** Forcing it back to two levels is precisely the constraint the proof above says kills
it.

### The model

Edges are indexed `k = 0 … cols−2` (x) and `0 … rows−2` (z).

```
E(θ) = 60·cos θ + 2·gap·cos(θ/2)      plan advance across one ramp    (cell pitch = 60 + E)
R(θ) = 60·sin θ + 2·gap·sin(θ/2)      rise across one ramp

f(0) = 0 ;  f(i+1) = f(i) + σ(i)·R(θx(i)) ,  σ(i) = (−1)^i        (and g, on z)
x(i+1) = x(i) + 60 + E(θx(i))                                     (and z, on j)
```

Plan lines and height runs are cumulative and depend on **one index only** — that is the alignment.

### The scrunch

`config.angleDeg` is the **base angle at the front**, the lowest in the design; scrunching only
steepens. `pattern.wave` carries four knobs:

```js
wave: { scrunchX: 0, scrunchZ: 0, attractorX: 1, attractorZ: 1 }
```

For edge `k` of an axis with `N` edges, `t = N > 1 ? k/(N−1) : 0`:

```
factor(t)   = scrunch · min(1, t / attractor)      ( = scrunch when attractor = 0 )
E_target(k) = E(angleDeg) · (1 − factor(t))
θ(k)        = the θ solving E(θ) = E_target(k), by 60 fixed bisection steps on [angleDeg, ANGLE_MAX]
```

**The thing that is linear is the plan advance, not the angle** — which is what "compress in a
simple linear fashion" reads as on the floor. `E` is strictly decreasing in θ, so the bisection is
well posed; 60 halvings, never "until converged", per this core's standing rule.

`θ(0) === angleDeg` exactly (early return, no bisection) at any `attractor > 0`. At `attractor = 0`
the whole axis compresses uniformly and edge 0 is **not** at the base angle — the one documented
exception, kept because "a tighter grid at one angle" is a real thing to ask for.

`W_SCRUNCH_UNREACHABLE` names any edge whose target advance is past what `ANGLE_MAX` delivers; the
edge is pinned there. At 30°/2cm the steepest reachable scrunch is ~66.5%, so the band's top (0.9)
is a place the solver reports from, not a promise.

### Downstream

- **`level` is no longer binary.** On the wave it is the cell's **rank among the distinct heights**,
  so 0 still means "on the floor" and the count is whatever the field has. `panel.heightCm`,
  `lattice.wave.heights[i][j]` and `lattice.wave.storeyCount` carry the truth. Both fields are
  emitted **only under `kind: 'wave'`** — the trapezoid record is frozen.
- **`lattice.pitchCm` / `riseCm` stay the base angle's numbers** under the wave, i.e. what edge 0
  does. `lattice.wave.x` / `.z` carry the per-edge tables (angle, scrunch factor, advance, pitch,
  signed rise, cumulative line, plan run against the unscrunched run).
- **The envelope is no longer a property of `(gap, θ)`.** `maxAngleDeg` is still meaningful and is
  still bisected, but it now means *the largest **base** angle this design admits*. `envelope.
  angleIsPerJoint` and `envelope.perJoint` (distinct folds, dirty count, worst joint by **signed**
  fold — convex is what binds) are emitted alongside it, wave only.
- **Grounding measures, it does not assume.** The y offset lands the network's **lowest present
  material** on the number, found by measuring rather than by testing `level === 0` (§9.12). On the
  wave's uneven floor that is usually **one corner** — a scrunched wave is not sitting on the floor,
  it is standing on one cell and grounded as a rigid body. `tests/test-v4-wave.mjs` §12 checks it.
- **The wall anchor is not built under the wave**, and is refused rather than approximated: §9.4's
  anchor descends exactly one level rise to the floor and the wave has no such level, so the toe
  would hang in mid-air. `W_WAVE_NO_ANCHOR` says so.

### The surface drifts down as it compresses

`f` is `0, R₀, R₀−R₁, R₀−R₁+R₂ …`, and scrunching only steepens, so the negative terms outweigh the
positive ones and each axis sheds height as it tightens. A run of folds whose far end is steeper
than its near end does that; it is a property of the shape, not an artefact. Grounding then tilts
the whole rigid assembly. Reported (the tables carry every `f(i)`), not corrected — correcting it
would mean breaking the zig-zag, and the zig-zag is what makes the panels alternate.

## 9.8 Not in this pass

Plateaus of same-level flats, plates (`2x4`), cross-network bracing other than the wall anchor, a
wall anchor for the wave (§9.14), and any physical design for the wall attachment.

*(Per-cell angle and more than two levels were on this list until §9.14, which delivers both — for
the separable family only. On the checkerboard they remain impossible, and §9.14 has the proof.)*

---

## 9.15 The room

Everything the installation has to be placed *against*. None of it is design: it is the building,
measured on site, and nothing in `src/core/v4/lattice.js` reads any of it. `obstacles.js` only ever
answers questions — it never moves a panel or refuses a config, the same "report the cost, do not
veto" contract the connector flags follow.

### The datum

| axis | zero at | positive |
|---|---|---|
| **x** | the wall's **room-side face** | away from the wall, into the room |
| **y** | **the floor the panels stand on** | up |
| **z** | the **window side** | away from the window, into the room |

Two of those are easy to get wrong and both have already been got wrong once:

- **`z = 0` is a reference plane, not the glass.** The real window sits at negative z. Obstacles
  therefore take signed coordinates, and anything assuming the room lives in the positive quadrant
  is wrong.
- **`y = 0` is the floor the panels stand on** — the surface visible in the site photo. From the
  street it reads as a deep sill; it is not. Reading it as a sill would put a solid exactly where the
  installation sits, and every clearance against it would be wrong in the reassuring direction.

`RibbonViewport` draws the origin triad so the convention can be checked rather than trusted. Note
+X points to screen-**left**: the camera looks in from the window, so the wall renders on the right.

### Elements

`obstacles[]`, each an axis-aligned box with `anchor` (`corner` = min x/z, or `centre`), `baseYCm`
(where it starts in y — routinely negative, and *not* anchored, because "how far up does it start"
has no corner/centre ambiguity), and `kind`:

| kind | what | drawn |
|---|---|---|
| `solid` | material — column, mullion, sill, cap | dense |
| `zone` | reserved **empty space** the design must keep out of — the heating runs | outlined, faint |
| `glass` | material, but see-through | nearly clear |

All three are tested identically; `kind` never reaches `solveObstacles`.

### The window turns a corner

This is the corner of the building. The glazing runs along x, reaches **(−81.3, −59.7)**, and turns
90° to run up the returning elevation with the same section, glass detail, height off the sidewalk,
and spacing pattern measured from the corner.

The two elevations are **one description with an axis swapped** — `facade('x')` and `facade('z')`.
`MULLION_SECTION` is stated as `acrossCm` (6.35, the face width) and `depthCm` (19.05, how far it
reaches back) rather than width/depth, because *which world axis each maps to is exactly what the
turn changes*. Naming them x and z lays the return's mullions on their side — plausible in plan,
wrong in section.

Mullion centres step **129.5, then 152.4 × 3** from the corner on both elevations. Heights are
identical on both — a corner changes plan, not section:

```
caps        y −65 → 375    1cm, one per mullion, outside the glazing plane
glass       y −25 → 375    0.5 thick, flush to the STREET face
mullions    y −25 → 375    6.35 across × 19.05 deep
sill        y −30 → −25    top flush with the mullion bottom
sill (low)  y −70 → −65    at sidewalk level, 40 below
```

The **corner post is shared** and appears in both elevations' lists: the two first mullions overlap
in a 6.35 × 6.35 column, and that overlap *is* the post described twice. The element count is not a
part count.

### The heating runs are trenches

```
heating          x −62.25 → 512.7   y −25 → 0   z −59.7 → 0
heating-return   x −62.25 → 0       y −25 → 0   z 0 → 533.35
```

**They stop at the floor, and that is load-bearing.** The wall slab is drawn from y = 0 up, so a
trench below the floor and a wall above it never meet — which is the only reason the return run can
legitimately reach x = 0, the wall's room-side face. It passes *under* the wall. If a run ever
climbs past y = 0 again it starts intersecting the wall silently.

They **tile** rather than overlap: gap 1 covers the corner square across gap 2's whole x range, so
gap 2 starting at z = 0 leaves neither an overlap nor a missed strip (asserted with the SAT).

### The saved-design contract

Obstacles are site measurements, so a design saved before an element was measured must not be able
to withhold it. `mergeObstacles` folds a saved list onto the room's known elements **by id**:

- **absent** → the room as measured
- **`[]`** → deliberately no room, for studying the design alone
- **non-empty** → the room as measured, those entries overriding by id, unknown ids appended

### Still assumed, not measured

- **the sill's depth** — set to the mullion footprint, the minimal claim. If it oversails toward the
  room it can reach the network.
- **the ceiling** — nothing knows where the top of the room is. The mullions already reach 375.
