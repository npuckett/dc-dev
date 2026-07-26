# grid-designer

A three.js tool for **planning the configuration of Drop Ceiling V2** — the next build of the
storefront light-panel installation. You propose whole-grid fold patterns, adjust them by hand,
check them against the physical rules, and export the result as OBJ.

This is an **ideation tool for the whole grid**, not a panel-by-panel modeller. If you want to
know why that distinction matters, see the sibling projects:

| project | what it is | status |
|---|---|---|
| `grid-designer/` | **this** — whole-grid folded-surface planning for V2 | active |
| `panel-designer/` | first attempt: builds a surface panel-by-panel as a single-rooted kinematic tree | abandoned; the data model can't express a grid. Some modules were copied out of it (see below) |
| `spatial-editor/` | generic react-three-fiber scene editor, has a `world_coordinates.json` converter | unrelated, but useful prior art |
| `IO/public-viewer/` | the live viewer for the **existing** installation (thedropceiling.com) | separate concern |

For the state of play, the decision log, and open questions, read **[HANDOFF.md](HANDOFF.md)**.

---

## Running it

```bash
npm install --prefix grid-designer
npm run dev --prefix grid-designer
```

Serves on **port 5175** with `strictPort` — deliberately not 5173, because `panel-designer` and
`spatial-editor` both default to that and would collide. There is a launch config at
`.claude/launch.json` (name `grid-designer`) for the Claude Code preview pane.

```bash
npm run build --prefix grid-designer     # production build
```

**Note: there is no persistence.** The working config lives in memory only, so a page reload —
including the hot reload triggered by editing any source file — resets to the flat preset. If you
have a design you care about, hit **Copy** in the CONFIG JSON panel or **Download JSON** first.
This has cost real work twice; see HANDOFF.md's "next steps".

---

## The physical system

Two panel types, the standard drop-in ceiling-light sizes:

- **square** — 60 × 60 cm (the base grid unit)
- **plate** — 60 × 121 cm (a "rect"), which is *exactly* two squares plus one joint: 60 + 1 + 60 = 121

That exactness is why the default joint gap is **1 cm**: a plate is a perfect drop-in replacement
for two squares and the joint between them, with no slack to absorb. Cell pitch is therefore 61 cm.

Panels are joined by 3D-printed connectors that span the gap, which is the key to the whole
geometric approach — see "why there is no origami solver" below.

### World conventions

- Units **centimetres**, **Y up**, X–Z are the floor axes.
- **6 columns** along +X (`c = 0..5`), **rows recede along +Z** (`r = 0..rows-1`), rows adjustable 5–8.
- **The window / shore is at z = 0** — row 0 is the row nearest the glass, and the primary viewing
  position is from the window side looking in.
- **The wall is at x = 0, beside column 0.** Column 0 is therefore the **rightmost in the 3D view**
  and the **left edge of the plan view**. This mirror trips everyone up; the plan view says so in
  its own hint text.
- Panel local frame: width → local X, depth → local Z, **lit face +Y** with the front surface at
  local y = 0 and the housing extending back to y = −3.7.

### Panel profile (`src/config.js`)

Overall thickness 3.7, frame flange 2.5 wide, diffuser recessed 1.0 behind the frame, back plate
inset 4.0 on all four sides. `src/geometry/panelGeometry.js` builds this as a closed 2-manifold
shell: a visible frame all the way around a recessed glowing diffuser on the front, and an
enclosed inward-tapering tray on the back that holds the LEDs.

---

## The fold model (schema v2)

**Folding runs along the columns only — from the window straight back.** Each of the 6 columns is
an independent 2D chain in the Y–Z plane at a fixed x. Nothing folds along a row.

Each column carries:

- `startPitchDeg` — the pitch of its front (window) panel. The chain origin is solved so that this
  panel always **rests on the floor** on its contact edge whatever the pitch (`frontRestY` in
  `schema.js`), so a pitched front costs nothing in support.
- `foldsDeg[]` — `rows − 1` **signed hinge dihedrals**, one per joint going back. Cumulative:
  panel *r*'s pitch is `startPitchDeg + Σ folds[0..r-1]`.
- `endSupport` — `'floor'` (default) or `'wall'`, the latter valid only on column 0.

**Joints along a row are plain gap connections.** Because neighbouring columns fold independently,
their shared row edges drift apart in 3D. That divergence is **measured and reported**, not
prevented — `src/core/report.js` flags every joint whose gap or skew exceeds connector tolerance.
That is the honest output: it tells you where the connectors have to flex.

### Why there is no origami solver

Rigid rectangles that share vertices cannot fold in two directions at once — at a vertex with four
90° sectors, only one crease pair can move. The naive conclusion is that you need a rigid-origami
solver. You don't: **the physical system has a ~1 cm gap at every joint, spanned by a connector, so
panels never share a vertex.** The slack absorbs the incompatibility.

So placement is a **deterministic pure function** (config → per-panel world transforms) and the
joint report tells you what the connectors must accommodate. This is far simpler and far more
useful than constraint-solving a fiction.

---

## The rules the tool enforces

Encoded in `src/core/schema.js` (validation) and `src/core/placement.js` (layout violations). The
distinction matters:

- **Validation errors block the change.** The store never commits an invalid config.
- **Layout violations do not block.** They are loud, visible, and fixable — because blocking them
  would make slider exploration impossible (every intermediate drag state would be rejected).

| rule | code | kind |
|---|---|---|
| **Every column's last panel must touch the floor** — one housing edge down or lying flat, never floating. "The wave returns to the water." Column 0 may be exempted by `endSupport: 'wall'` (side-bracketed). | `E_END_FLOATING` | violation |
| A rigid plate cannot bend: a **vertical** plate requires the hinge it spans to be exactly 0° | `E_FOLD_ON_REMOVED_JOINT` | error |
| A **horizontal** plate requires its two columns to agree in cumulative pitch *and* chain position at that row | `E_CROSSCOL_ANGLE_MISMATCH`, `W_CROSSCOL_POSITION` | error / warning |
| **At least 4 plates, placed by one nameable pattern rule** — the 121 cm plate is half the kit, and a design using one or two reads as a grid of squares with mistakes in it | `W_FEW_RECTS` | warning |
| No two plates may share a cell | `E_RECT_OVERLAP` | error |
| Panels dug into the floor | `W_BELOW_FLOOR` | warning |
| A column that stops advancing (accordion backtrack) | `W_CHAIN_BACKTRACK` | warning |
| Structure / ranges: `version` must be 2, units cm, rows 5–8, folds ±120°, gap 0–10 cm | `E_SHAPE`, `E_RANGE` | error |

`version: 2` is required — **v1 (row-accordion) configs are rejected outright**, because that model
was wrong (see HANDOFF.md).

There is **no** flat-shore rule any more. Row 0 lying flat was once mandatory; the storm presets
deliberately pitch every front panel, so it is now just `startPitchDeg: 0`.

### Auto-solvers

- **Ground end / Ground all** (`src/core/ground.js`) — solves a column's last surviving hinge so the
  end panel lands (scan + bisection, picking the root nearest the current angle, preferring to rest
  *on* the floor over sinking into it). It returns `null` when the chain ends too high for the last
  panel to reach, which the store surfaces as a synthetic `E_UNGROUNDABLE`: that means the *profile*
  has to arc back down, and the solver deliberately won't mangle earlier folds to hide it. The
  policy of *which* hinge to solve lives in the callers, not the solver.
- **Merge** (`src/core/merge.js`) — combining two squares into a plate performs the physically
  implied adjustment: a vertical merge flattens the hinge it spans, a horizontal merge matches the
  second column to the clicked-first one through that row. Both land with the plate in a single
  validated commit and report what they changed in plain language.

---

## Presets

Two families. Measured with seed 1:

| preset | rows | plates | pattern rule | bbox W×H×D cm | peak | flagged/joints |
|---|---|---|---|---|---|---|
| `flat` | 5 | 4 | mirrored quad | 365 × 4 × 304 | 4 | 0/45 |
| `calm` | 5 | 4 | mirrored pairs | 365 × 20 × 302 | 20 | 8/45 |
| `wave` | 5 | 6 | crest plates | 365 × 82 × 285 | 82 | 18/43 |
| `crash` | 5 | 6 | landing plates | 365 × 77 × 300 | 77 | 20/43 |
| `random` | 5 | 6 | one of four seeded templates | 365 × 68 × 302 | 68 | 19/43 |
| `swell` | 8 | 8 | doubled spine plates | 365 × 117 × **478** | 117 | 40/74 |
| `surge` | 7 | 12 | double plates | 365 × 170 × 241 | 170 | 29/59 |
| `wallcrash` | 6 | 6 | wall splash | 365 × 200 × 339 | 200 | 30/54 |

**Calm family** (`flat`/`calm`/`wave`/`crash`/`random`) — flat front row, 5 rows.
**Storm family** (`swell`/`surge`/`wallcrash`) — every column's front panel pitched, higher row
resolution, plates crossing rows. `wallcrash` additionally engages the wall: column 0 is
`endSupport: 'wall'` and ends deliberately high, water splashing up the −X wall.

`random` draws one of four named templates from the seed (`mirrored-pairs`, `alternating-bands`,
`diagonal-cascade`, `shore-rafts`) and is deterministic per seed; verified valid, grounded and
warning-free across 200 seeds.

**Preset profiles are authored around their plates, not the reverse.** A plate pins its joint flat,
so each preset designs the plateau first and differences the hinge angles out of the pitch profile.
Hunting for legal plate positions in a profile authored without them mostly finds none.

---

## The UI

- **3D viewport** — orbit controls, ground grid, the blue **WINDOW / SHORE** line at z = 0, the
  translucent **WALL** plane at x = 0, red/amber markers on flagged joints, and a dismissible
  **measuring box** with width / peak / depth labels in cm (`bounds` toggle).
- **Plan view** — 6 × rows map, row 0 at the bottom, wall on the left. **Click a square** to arm it;
  every mergeable neighbour highlights (**green `+`** = free, **amber dashed `~`** = also changes
  geometry, with the consequence in the tooltip). Click one to **combine into a plate**. Click a
  plate to **split** it. Escape lets go. Cell tint shows cumulative pitch.
- **Column controls** — per column: a `front` pitch slider plus one slider per hinge, a profile
  sparkline, cumulative pitch readout, grounded/floating badge with **Ground end**, and (column 0
  only) a **bracket to wall / stand on floor** toggle.
- **Toolbar** — rows stepper (5–8), Ground all, Copy col 0 → all, Shift →, Flatten, bounds, Undo,
  Redo. Cmd/Ctrl+Z and Shift+Cmd/Ctrl+Z work too.
- **CONFIG JSON** — paste a config and Apply (normalized, validated, errors listed), or Copy the
  current one. This is the save/load mechanism until persistence exists.
- **Export** — OBJ (one named object per panel with baked world transforms) and JSON.

---

## Architecture

```
src/
├── config.js                 panel dimensions + profile + edge helpers  (copied from panel-designer)
├── geometry/panelGeometry.js  the panel solid: framed diffuser front, tapered enclosed back
├── core/                      ← HEADLESS ZONE
│   ├── schema.js              DEFAULT_CONFIG, normalizeConfig, validateConfig, columnChain, frontRestY
│   ├── placement.js           solveLayout (config → world transforms), panelSolidCorners, layoutBounds
│   ├── report.js              jointReport — per-joint gap/skew/dihedral + flags
│   ├── ground.js              solveGroundingFold, groundAllFolds
│   ├── merge.js               mergeCandidates, mergeCells
│   └── presets.js             the two preset families + seeded random templates
├── store.js                   zustand: commit-through-validation, memoized derived data, undo/redo
├── components/                Viewport, SurfaceMeshes, JointFlags, GridMap, ColumnControls,
│                              ControlPanel, PresetBar, JsonPanel, ExportButtons
└── utils/exporters.js         buildExportGroup (headless) + OBJ/JSON download
```

### The headless-core contract

Everything in `src/core/` **must**:

1. use **explicit `.js` extensions** on all relative imports — plain node ESM does not resolve
   extensionless paths. (`panel-designer` omitted them, which is precisely why its node tests are
   dead and its geometry never got verified. Don't repeat it.)
2. import **only `three`'s math classes** — never a scene graph, never the DOM, never the store.
3. be **pure and deterministic** — same config in, byte-identical output. Several tests assert this
   by `JSON.stringify` equality across repeat calls.

This is what makes the whole model testable in node without a browser, and it is the single most
valuable convention in the project. The store is a thin commit-and-cache layer over it; the
components are thin renderers of its output.

### The store contract

Every mutation runs through `commit()`: produce a candidate config, run `validateConfig`, and
commit **only if valid** — otherwise keep the previous state and stash the errors in `lastErrors`
for the UI. Derived data (`layout`, `report`, `violations`, `bounds`) is memoized in a `WeakMap`
keyed on config identity, so it is computed once per change rather than per frame. UI-only state
(`showBounds`, `armedCell`, undo history) lives outside `config` and never reaches the exporters.

---

## Testing

Plain node scripts — no test framework. Each prints a pass/fail summary and exits non-zero on
failure.

```bash
cd grid-designer
node tests/test-merge.mjs        # 103  merge candidates/coercion, sweep invariant
node tests/test-geometry.mjs     #  50  panel solid: manifold, orientation, volume, groups
node tests/test-validation.mjs   # 213  schema, normalization, every error/warning code
node tests/test-presets.mjs      # 341  all presets: valid, grounded, patterns, determinism
node tests/test-placement.mjs    # 268  chains, closed forms, bounds, rects
node tests/test-report.mjs       # 125  joint metrics
node tests/test-obj.mjs          #  37  OBJ export round-trip
npm run build
node tests/screenshot.mjs        # 225  Playwright; spawns its own dev server
```

**1137 headless + 225 browser checks.** `tests/screenshot.mjs` starts and stops its own dev server
(don't pre-start one on 5175), drives the real UI, and writes reference PNGs to
`tests/screenshots/`. Those PNGs are committed and are the visual record of each feature.

Two testing conventions worth keeping:

- **Closed-form expectations.** Chain geometry, bounds, and panel volume are checked against
  formulas derived in the test from the constants, not against golden numbers. Sign conventions are
  the classic bug source here and this is what catches them.
- **Regression guards on preset output.** Preset configs are compared bit-identically against the
  previous commit whenever one preset is being tuned, so a shared-helper change can't silently
  reshape the others.
