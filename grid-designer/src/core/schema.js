/**
 * grid-designer — configuration schema, normalization and validation.
 *
 * HEADLESS ZONE (src/core/): pure functions, importable from plain node.
 *   - explicit `.js` extensions on ALL relative imports
 *   - may import `three` math classes only; never components / store / DOM
 *   - same input → same output, no hidden state
 *
 * =============================================================================
 * WHAT THIS DESCRIBES  (schema v2 — "per-column fold strips")
 * =============================================================================
 * A `cols × rows` grid of ceiling light panels (Drop Ceiling installation V2,
 * the "agentic body of water") standing on the floor of a storefront. The window
 * / shore is the line z = 0; rows recede into the store along +Z; columns run
 * across the window along +X. A WALL closes the −X side of the grid, beside COLUMN
 * 0, running window → back (see THE WALL below).
 *
 * FOLDING HAPPENS ONLY ALONG COLUMNS. Each column c is an independent STRIP: a
 * chain of `rows` panels hinged at the `rows - 1` joints between consecutive
 * rows, folding up and back from the window. There is NO in-row accordion — the
 * v1 fold family (zigzagDeg / jointOverridesDeg / rowFoldsDeg) is gone.
 *
 * Consequences worth internalizing before generating a config:
 *   - Columns NEVER move in X. Column c occupies x ∈ [c·(size+gap),
 *     c·(size+gap)+size] in every configuration, so column centers sit on the
 *     lattice x = size/2 + c·(size+gap) (30, 91, 152, … at the defaults, where
 *     the cell pitch is size + gap = 60 + 1 = 61cm).
 *   - Side-by-side joints along a row are therefore SIMPLE CONNECTIONS: a
 *     constant `gap` in X by construction. Their 3D edge separation only grows
 *     when two adjacent columns' fold profiles differ in pitch or in the height
 *     / depth they have reached — that is exactly what report.js measures.
 *   - ROW 0 IS THE FRONT PANEL AT THE WINDOW, AND ITS PITCH IS DESIGNABLE.
 *     `columns[c].startPitchDeg` (default 0) tilts it; the chain's front edge
 *     stays on the window line z = 0 and the front panel stays ON THE FLOOR BY
 *     CONSTRUCTION — see PITCHED FRONTS below. There is no flat-shore rule any
 *     more; a flat front is simply `startPitchDeg: 0`.
 *   - WAVES ARE MADE BY PHASE-SHIFTING FOLD SEQUENCES ACROSS COLUMNS. Give
 *     column c a fold sequence whose crest sits one step further back than
 *     column c-1's, and the surface reads as a swell travelling across the
 *     window. Identical sequences in every column give a cylindrical fold with
 *     perfect in-row joints.
 *   - GROUND SUPPORT MATTERS. Panels dipping below y = 0 have nothing to stand
 *     on; placement.js emits W_BELOW_FLOOR for them. A chain that rises before
 *     it descends (all cumulative pitches ≥ 0, or a descent no deeper than the
 *     height already gained) stays supported.
 *   - THE WAVE RETURNS TO THE WATER — see the next section. Every FLOOR column's
 *     LAST panel must come back down and touch the floor. A WALL-SUPPORTED
 *     column is exempt: it may end high, splashing up the wall.
 *
 * =============================================================================
 * PITCHED FRONTS — `columns[c].startPitchDeg`
 * =============================================================================
 * `startPitchDeg` is the cumulative pitch of ROW 0 — the front / window panel —
 * clamped to the same ±120° as a hinge. Every later row's pitch is that value
 * plus the running sum of the folds in front of it:
 *
 *   ψ(0, c) = startPitchDeg(c),   ψ(r, c) = startPitchDeg(c) + Σ_{k < r} fold(c, k)
 *
 * THE FRONT EDGE STAYS ON THE WINDOW LINE (z = 0) AND THE FRONT PANEL STAYS ON
 * THE FLOOR. The chain's start height y0 is not a constant any more: it is
 * derived per column (`frontRestY`) as the value that puts the LOWEST point of
 * the front panel's SOLID (its 4 lit-face corners plus the 4 at the back of the
 * housing, `PANEL_PROFILE.overallThickness` = 3.7cm along −n̂) at exactly y = 0:
 *
 *   y0(θ, L) = max(0, 3.7·cos θ) − min(0, L·sin θ)
 *
 * with L the front panel's footprint ALONG the chain (60 for a square or a
 * horizontal plate, 121 for a vertical plate at row 0). Read it as two cases:
 *   θ ≥ 0  (the front pitches UP, the common case) → y0 = 3.7·cos θ: the panel
 *          rests on the housing edge under its FRONT edge, at the window.
 *   θ < 0  (the front DIVES toward the floor) → the front edge has to lift by
 *          L·sin|θ| so the panel's REAR edge is what rests on the ground; the
 *          whole strip starts that much higher. Nothing is clamped and nothing
 *          is special-cased: a profile that keeps diving behind row 0 drives
 *          later panels under the floor and simply raises W_BELOW_FLOOR.
 * At θ = 0 the formula gives exactly 3.7 — the historical `SHORE_Y` — so a
 * config without `startPitchDeg` lays out bit-identically to before.
 *
 * =============================================================================
 * THE WALL — `columns[c].endSupport`
 * =============================================================================
 * The real room has a wall at the −X edge of the grid: the plane x = `WALL_X` = 0,
 * which is the OUTER edge of column 0, running from the window straight back. The
 * column beside it — c = `WALL_COLUMN` = 0 — can therefore be SIDE-BRACKETED to
 * that wall instead of standing on the floor:
 *
 *   `columns[0].endSupport: 'wall'`      → the strip may end HIGH, water
 *                                          splashing up the wall
 *   `endSupport: 'floor'` (the default)  → the grounded-end rule applies
 *
 * 'wall' on any other column is an E_SHAPE error: only the wall-adjacent column
 * has a wall to bracket to. A wall-supported column is skipped by the
 * E_END_FLOATING check, by `groundAllFolds`, and by the UI's "Ground all".
 *
 * NOTE ON SIDES. The wall is at LOW x, so it is the LEFT edge of the plan view
 * (core/../components/GridMap.jsx draws column 0 at the left) and the RIGHT edge of
 * the 3D view (the camera looks in from the window, so +X runs leftward there).
 *
 * =============================================================================
 * THE GROUNDED-END RULE (a MUST for every generated design)
 * =============================================================================
 * THE LAST PANEL OF EVERY FLOOR-SUPPORTED COLUMN MUST TOUCH THE FLOOR. Either it
 * lies flat on the ground, or it is tilted with one of its housing edges down on
 * the ground — it may NEVER float in mid-air. ("The wave returns to the water.")
 *
 * Formally: over the last panel's SOLID (its 4 lit-face corners plus those 4
 * corners pushed `PANEL_PROFILE.overallThickness` = 3.7cm along −n̂, the back of
 * the housing), the minimum world y must be ≤ `groundTolerance` (default 0.5cm).
 * A flat panel sits at exactly 0; an edge-down panel lands within a few mm of it.
 *
 * This is NOT a validateConfig error — dragging a fold slider necessarily passes
 * through states where the end is in the air, and the store rejects invalid
 * commits, which would make the tool unusable. It is a LAYOUT VIOLATION instead:
 * `solveLayout()` returns `violations: [{ code: 'E_END_FLOATING', col,
 * clearanceCm, message }]` and marks each `columnChains[c]` with `grounded` /
 * `endClearanceCm`. A design with a non-empty `violations` array is not finished.
 *
 * HOW TO DESIGN A FOLD SEQUENCE THAT LANDS:
 *   The chain climbs by 60·sin ψ_r per row (plus the 1cm gap), so the
 *   height it has to give back by the end is essentially 60·Σ_r sin ψ_r. Author
 *   pitch profiles whose sines CANCEL — arc up and then back down — e.g.
 *   [0, 35, −35, 0, 0], [0, 35, 0, −35, 0], [0, 10, 10, −10, −10]. Profiles that
 *   only ever rise ([0, 20, 40, 60, 80]) end tens of cm in the air and cannot be
 *   rescued: the last panel is 60cm long, so once the chain's last vertex is much
 *   above ~57cm NO value of the last hinge can reach the ground.
 *   `core/ground.js` exports `solveGroundingFold(config, c)` / `groundAllFolds`,
 *   which solve the last surviving hinge of a column for floor contact — that is
 *   the mechanical fix; designing the arc is still the generator's job.
 *   A WALL-SUPPORTED column needs none of this: bracketed to the −X wall, it is
 *   allowed — and in the 'wallcrash' preset, meant — to end high.
 *
 * =============================================================================
 * JSON CONFIG SCHEMA (v2) — this is the shape an LLM generator should emit
 * =============================================================================
 *
 *   {
 *     "version": 2,                            // must be exactly 2
 *     "units": "cm",                           // must be "cm"
 *     "name": "wave study 3",                  // optional, free text
 *     "grid": { "cols": 6, "rows": 5 },        // 6 columns × rows; rows must be 5..8
 *                                              // (MIN_ROWS..MAX_ROWS — below 5 the
 *                                              // shore-plus-wave reading collapses,
 *                                              // above 8 the strip runs out of store)
 *     "cell": { "size": 60, "rectLength": 121 },  // square cell 60cm; 2-cell rect 121cm
 *     "gap": 1.0,                              // physical joint gap spanned by connectors.
 *                                              // KEEP 2·size + gap = rectLength: 60+1+60 = 121
 *                                              // is exactly the plate's long side, so one rect
 *                                              // drops in for two squares + their joint with
 *                                              // nothing left over. Other values are legal and
 *                                              // simply raise W_RECT_LENGTH with the mismatch.
 *     "gapTolerance": 1.5,                     // report flags joints deviating > this from gap
 *     "groundTolerance": 0.5,                  // a column's last panel counts as
 *                                              // touching the floor when its solid
 *                                              // reaches within this many cm of y = 0
 *     "columns": [                             // exactly grid.cols entries; 0 = leftmost (x = 0 side)
 *       { "foldsDeg": [30, -60, 60, -30],      // exactly grid.rows-1 SIGNED hinge angles
 *         "startPitchDeg": 0,                  // [k] = hinge between rows k and k+1
 *         "endSupport": "floor" },             // pitch of ROW 0 (the front / window panel),
 *                                              // ±120; the front panel stays on the floor
 *                                              // whatever it is (see PITCHED FRONTS).
 *                                              // endSupport 'wall' is legal ONLY on the
 *                                              // last column and exempts it from the
 *                                              // grounded-end rule (see THE WALL).
 *       { "foldsDeg": [20, -40, 40, -20] },    // both new fields default when omitted
 *       { "foldsDeg": [0, 0, 0, 0] },
 *       { "foldsDeg": [0, 0, 0, 0] },
 *       { "foldsDeg": [0, 0, 0, 0] },
 *       { "foldsDeg": [0, 0, 0, 0] }
 *     ],
 *     "rects": [                               // 60×121 two-cell panels — AT LEAST 4, see below
 *       { "row": 1, "col": 2, "orientation": "horizontal" },  // (1,2)+(1,3): one plate across two columns
 *       { "row": 2, "col": 5, "orientation": "vertical" }     // (2,5)+(3,5): removes joint k=2 of column 5
 *     ],
 *     "meta": {                                // provenance, free-form
 *       "preset": "wave", "seed": 42,
 *       "rectPattern": "crest plates",         // NAME the rule the rects follow
 *       "notes": ""
 *     }
 *   }
 *
 * =============================================================================
 * THE PLATE-PATTERN RULE (a MUST for every generated design)
 * =============================================================================
 * EVERY DESIGN MUST USE AT LEAST `MIN_RECTS` = 4 RECT PLATES, PLACED BY AN
 * EXPLICIT PATTERN RULE. The 60×121 plate is half the panel kit; a design that
 * uses one or two of them by accident reads as a grid of squares with a couple of
 * mistakes in it. Four or more, placed by a rule you can say out loud — "mirrored
 * pairs", "a plate at every column's crest", "a diagonal cascade", "the last two
 * rows of alternating columns" — read as a system, which is the point.
 *
 * Generators (LLM or procedural) MUST therefore:
 *   1. place ≥ 4 plates,
 *   2. place them by ONE nameable rule, not by scattering them, and
 *   3. write that rule's name into `meta.rectPattern` (free-form string) and
 *      mention it in `meta.notes`.
 * Fewer than 4 raises the non-blocking warning `W_FEW_RECTS`.
 *
 * DESIGN THE FOLDS AND THE PLATES TOGETHER. A plate is rigid, so it PINS the
 * surface it covers: a vertical plate at (r, c) removes joint r of column c (that
 * fold must be 0, i.e. rows r and r+1 sit at one pitch — a PLATEAU in the pitch
 * profile), and a horizontal plate at (r, c) forces columns c and c+1 to the same
 * pitch at row r. Authoring the folds first and hunting for legal plate positions
 * afterwards mostly finds none; author the plateaus the pattern needs INTO the
 * pitch profiles from the start (see core/presets.js for six worked examples).
 *
 * Two consequences for the grounded-end rule above:
 *   - a plate spanning the LAST two rows (a vertical rect at (rows-2, c)) removes
 *     the column's last hinge, so the landing is solved at the hinge in FRONT of
 *     it and the panel being landed is the 121cm plate itself — which, being twice
 *     as long, can give back nearly twice as much height;
 *   - grounding must therefore be solved AFTER the plates are placed, never before.
 *
 * =============================================================================
 * ANGLE SEMANTICS (verbatim rules — follow these exactly)
 * =============================================================================
 * Joint k of column c is the hinge between rows k and k+1 (k = 0 .. rows-2).
 * `columns[c].foldsDeg[k]` is that hinge's SIGNED dihedral in degrees, clamped
 * to ±120. A POSITIVE fold pitches the next panel UP (toward +Y).
 *
 * Cumulative pitch of cell (r, c) is `startPitchDeg` plus the running sum of the
 * folds in front of it:
 *
 *   ψ(r, c) = startPitchDeg(c) + Σ_{k < r} foldsDeg(c, k),   ψ(0, c) = startPitchDeg(c)
 *
 * (see `cellPitches()`), and the chain's depth direction there is
 * d = (0, sinψ, cosψ) — so ψ = 90° stands the panel straight up and ψ > 90°
 * points it back toward the window.
 *
 * A VERTICAL rect at (r, c) is one rigid 121cm plate spanning rows r and r+1 of
 * column c: it REMOVES joint k = r of that column, so `foldsDeg[r]` must be 0.
 * A HORIZONTAL rect at (r, c) is one rigid plate spanning columns c and c+1 at
 * row r: both chains must have reached the SAME pitch at row r, or the plate
 * cannot lie in both.
 *
 * =============================================================================
 * VALIDATION CODES
 * =============================================================================
 * Errors (ok === false):
 *   E_SHAPE                     structural problems: bad version/units, grid.cols
 *                               < 1, grid.rows outside 5..8, wrong columns /
 *                               foldsDeg lengths, malformed column or rect entries,
 *                               a non-numeric startPitchDeg, an endSupport that is
 *                               neither 'floor' nor 'wall', endSupport 'wall' on a
 *                               column that is not the wall-adjacent one,
 *                               out-of-bounds rects, non-positive gapTolerance /
 *                               groundTolerance (the message names the problem)
 *   E_RANGE                     a fold or startPitchDeg outside ±120, gap not in
 *                               (0, 10]
 *   E_RECT_OVERLAP              two rects share a cell
 *   E_FOLD_ON_REMOVED_JOINT     columns[c].foldsDeg[r] ≠ 0 at a joint removed by
 *                               a vertical rect (the plate is rigid there)
 *   E_CROSSCOL_ANGLE_MISMATCH   a horizontal rect's two cells sit at different
 *                               cumulative pitches (> 0.1°)
 * Warnings (do not affect ok):
 *   W_CROSSCOL_POSITION         a horizontal rect's two chains agree in pitch but
 *                               their cell origins are more than `gapTolerance`
 *                               apart in (y, z) — they diverged earlier and only
 *                               re-converged in ANGLE, so one plate cannot span
 *                               them cleanly; the joint report shows the cost
 *   W_RECT_LENGTH               cell.rectLength ≠ 2·cell.size + gap. AT THE
 *                               DEFAULTS THIS NEVER FIRES: 2·60 + 1 = 121 =
 *                               rectLength, so a rect plate spans exactly the
 *                               two squares plus the joint it replaces. It only
 *                               fires on a hand-set combination that does not
 *                               add up (e.g. gap 2 → ideal 122 against a 121cm
 *                               plate = 1cm of slack per rect), which is
 *                               accepted and surfaced, never silently corrected
 *   W_FEW_RECTS                 fewer than MIN_RECTS = 4 rect plates. See THE
 *                               PLATE-PATTERN RULE above: the modularity of the
 *                               panel system has to READ in the design, which
 *                               takes four plates placed by a rule you can name
 *                               (and record in meta.rectPattern). A warning rather
 *                               than an error because the UI has to let you remove
 *                               plates one at a time
 *
 * The grounded-end rule is deliberately NOT here: it is a solved-layout property,
 * reported as `solveLayout(...).violations` (E_END_FLOATING), not a config error —
 * and it is skipped entirely for a wall-supported column.
 */

import { PANEL_PROFILE } from '../config.js'

// -----------------------------------------------------------------------------
// Limits
// -----------------------------------------------------------------------------
/**
 * Per-joint signed dihedral limit, degrees. `columns[c].startPitchDeg` — the
 * pitch of the front / window panel — shares the same limit.
 */
export const MAX_FOLD_DEG = 120
export const MAX_GAP_CM = 10
/**
 * Row-resolution band. Fewer than 5 rows and the design stops reading as a shore
 * plus a wave breaking behind it (the front panel, the climb, the crest, the
 * descent and the landing need five cells); more than 8 and a 61cm-pitch strip
 * runs past 5m into the store. `grid.rows` outside this raises E_SHAPE.
 */
export const MIN_ROWS = 5
export const MAX_ROWS = 8
/** The two ways a column's deep end can be held up. See THE WALL above. */
export const END_SUPPORTS = ['floor', 'wall']
/**
 * The x of the WALL plane closing the −X side of the grid, cm.
 *
 * It is 0 — the OUTER (low-x) edge of column 0 — and, unlike the shore line's
 * extent or the column lattice, it does NOT depend on cols / size / gap: the grid
 * is laid out FROM the wall, with column 0 against it. Hence a constant rather
 * than a function of the config.
 */
export const WALL_X = 0
/**
 * The only column that may set `endSupport: 'wall'` — the one beside the wall.
 *
 * Column 0 occupies x ∈ [0, size], so its outer edge IS the wall plane `WALL_X`.
 * Every other column has grid between it and the wall and therefore nothing to
 * side-bracket to, which `validateConfig` reports as E_SHAPE.
 */
export const WALL_COLUMN = 0
/** Two cells of a horizontal rect must agree in pitch to within this many degrees. */
export const PITCH_MATCH_EPSILON_DEG = 0.1
/**
 * Fewest rect plates a finished design should use — see THE PLATE-PATTERN RULE
 * above. Below this, `validateConfig` raises the non-blocking `W_FEW_RECTS`.
 */
export const MIN_RECTS = 4

// -----------------------------------------------------------------------------
// Defaults
// -----------------------------------------------------------------------------
export const DEFAULT_GRID = { cols: 6, rows: 5 }
export const DEFAULT_CELL = { size: 60, rectLength: 121 }
/**
 * Physical joint gap in cm — the span a printed connector bridges between two
 * panels. 1.0 is not arbitrary: it is the value that makes a 60×121 rect plate a
 * PERFECT DROP-IN for two 60cm squares plus the joint between them, because
 *
 *     2·size + gap = 2·60 + 1 = 121 = cell.rectLength
 *
 * exactly. Swapping two squares for one plate therefore moves nothing else in
 * the surface: the chain advances the same distance, the cells behind it land on
 * the same lattice positions, and W_RECT_LENGTH does not fire. (At the old 2.0cm
 * default the ideal was 122 against 121cm of real hardware, and that 1cm of slack
 * propagated down every column that used a plate.)
 */
export const DEFAULT_GAP = 1.0
export const DEFAULT_GAP_TOLERANCE = 1.5
/**
 * How close a column's last panel has to get to y = 0 to count as TOUCHING the
 * floor, cm. 0.5cm ≈ the slack a printed foot / the floor's own flatness eats.
 */
export const DEFAULT_GROUND_TOLERANCE = 0.5
/** A flat front panel — the historical, and still the default, row-0 pitch. */
export const DEFAULT_START_PITCH_DEG = 0
/** Columns stand on the floor unless explicitly bracketed to the wall. */
export const DEFAULT_END_SUPPORT = 'floor'
/**
 * The chain start height of a FLAT front panel: its lit face sits here and its
 * housing rests on the floor. Kept as a named constant because it is the
 * historical anchor of the whole lattice — `frontRestY(0, L)` reproduces it
 * exactly for any L, which is what makes a config without `startPitchDeg` lay out
 * bit-identically to the pre-pitched-fronts model.
 */
export const SHORE_Y = PANEL_PROFILE.overallThickness

/**
 * Flat 6×5 surface: every joint unfolded, every front flat, no rects.
 *
 * Deliberately plate-free — it is the bare reference geometry other code measures
 * against, not a design — so it is the one config that raises `W_FEW_RECTS` by
 * construction. `presetFlat()` is the designed flat surface, and it does carry its
 * four plates.
 *
 * The per-column `startPitchDeg` / `endSupport` are spelled out rather than left
 * to `normalizeConfig` so the reference config also documents the column shape.
 */
export const DEFAULT_CONFIG = Object.freeze({
  version: 2,
  units: 'cm',
  name: 'flat 6×5',
  grid: { cols: 6, rows: 5 },
  cell: { size: 60, rectLength: 121 },
  gap: DEFAULT_GAP,
  gapTolerance: DEFAULT_GAP_TOLERANCE,
  groundTolerance: DEFAULT_GROUND_TOLERANCE,
  columns: [
    { foldsDeg: [0, 0, 0, 0], startPitchDeg: 0, endSupport: 'floor' },
    { foldsDeg: [0, 0, 0, 0], startPitchDeg: 0, endSupport: 'floor' },
    { foldsDeg: [0, 0, 0, 0], startPitchDeg: 0, endSupport: 'floor' },
    { foldsDeg: [0, 0, 0, 0], startPitchDeg: 0, endSupport: 'floor' },
    { foldsDeg: [0, 0, 0, 0], startPitchDeg: 0, endSupport: 'floor' },
    { foldsDeg: [0, 0, 0, 0], startPitchDeg: 0, endSupport: 'floor' },
  ],
  rects: [],
  meta: { preset: 'flat', notes: '' },
})

// -----------------------------------------------------------------------------
// Small helpers
// -----------------------------------------------------------------------------
const isPlainObject = (v) => typeof v === 'object' && v !== null && !Array.isArray(v)
const isFiniteNumber = (v) => typeof v === 'number' && Number.isFinite(v)
const isInt = (v) => isFiniteNumber(v) && Number.isInteger(v)

const DEG = Math.PI / 180

export function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v
}

// -----------------------------------------------------------------------------
// normalizeConfig
// -----------------------------------------------------------------------------
/**
 * Fill in defaults so a minimal hand- or LLM-written config works.
 *
 * Never mutates `partial` — everything is deep-copied. Values that are present
 * are passed through untouched (even when invalid) so `validateConfig` can
 * report them; only *missing* pieces get defaults. `columns` is deliberately NOT
 * synthesized when absent: its length is load-bearing and a missing array is a
 * real error, not something to guess at.
 *
 * @param {object} partial
 * @returns {object} a fresh, fully-defaulted config
 */
export function normalizeConfig(partial) {
  const src = isPlainObject(partial) ? partial : {}

  const grid = {
    cols: isPlainObject(src.grid) && src.grid.cols !== undefined ? src.grid.cols : DEFAULT_GRID.cols,
    rows: isPlainObject(src.grid) && src.grid.rows !== undefined ? src.grid.rows : DEFAULT_GRID.rows,
  }

  const cell = {
    size: isPlainObject(src.cell) && src.cell.size !== undefined ? src.cell.size : DEFAULT_CELL.size,
    rectLength:
      isPlainObject(src.cell) && src.cell.rectLength !== undefined
        ? src.cell.rectLength
        : DEFAULT_CELL.rectLength,
  }

  const out = {
    version: src.version !== undefined ? src.version : 2,
    units: src.units !== undefined ? src.units : 'cm',
    grid,
    cell,
    gap: src.gap !== undefined ? src.gap : DEFAULT_GAP,
    gapTolerance: src.gapTolerance !== undefined ? src.gapTolerance : DEFAULT_GAP_TOLERANCE,
    groundTolerance:
      src.groundTolerance !== undefined ? src.groundTolerance : DEFAULT_GROUND_TOLERANCE,
    columns: normalizeColumns(src.columns),
    rects: Array.isArray(src.rects) ? src.rects.map(normalizeRect) : [],
    meta: { notes: '', ...(isPlainObject(src.meta) ? src.meta : {}) },
  }

  // `name` is optional — only carry it through when supplied.
  if (src.name !== undefined) out.name = src.name

  return out
}

/**
 * Deep-copy the column entries, defaulting the two OPTIONAL per-column fields so
 * older configs (and hand-written ones) keep working: a column without
 * `startPitchDeg` gets a flat front, one without `endSupport` stands on the floor.
 * `foldsDeg` is deliberately NOT defaulted — its length is load-bearing.
 */
function normalizeColumns(columns) {
  if (!Array.isArray(columns)) return columns
  return columns.map((column) => {
    if (!isPlainObject(column)) return column
    return {
      foldsDeg: Array.isArray(column.foldsDeg) ? column.foldsDeg.slice() : column.foldsDeg,
      startPitchDeg:
        column.startPitchDeg !== undefined ? column.startPitchDeg : DEFAULT_START_PITCH_DEG,
      endSupport: column.endSupport !== undefined ? column.endSupport : DEFAULT_END_SUPPORT,
    }
  })
}

function normalizeRect(rect) {
  if (!isPlainObject(rect)) return rect
  return {
    row: rect.row,
    col: rect.col,
    orientation: rect.orientation,
  }
}

// -----------------------------------------------------------------------------
// Angle semantics
// -----------------------------------------------------------------------------
/**
 * The set of fold joints of column `c` removed by vertical rects.
 * A vertical rect at (r, c) spans rows r and r+1, removing joint k = r.
 *
 * @param {object} config normalized config
 * @param {number} c column index
 * @returns {Set<number>}
 */
export function removedJoints(config, c) {
  const removed = new Set()
  const rects = Array.isArray(config.rects) ? config.rects : []
  for (const rect of rects) {
    if (!isPlainObject(rect)) continue
    if (rect.orientation === 'vertical' && rect.col === c) removed.add(rect.row)
  }
  return removed
}

/**
 * Signed dihedral at joint k of column c, clamped to ±MAX_FOLD_DEG.
 * Missing / non-numeric entries read as 0 so the walkers never produce NaN.
 *
 * @param {object} config normalized config
 * @param {number} c column index
 * @param {number} k joint index (0 .. rows-2)
 * @returns {number} degrees
 */
export function columnFoldDeg(config, c, k) {
  const column = Array.isArray(config.columns) ? config.columns[c] : undefined
  if (!isPlainObject(column) || !Array.isArray(column.foldsDeg)) return 0
  const v = Number(column.foldsDeg[k])
  if (!Number.isFinite(v)) return 0
  return clamp(v, -MAX_FOLD_DEG, MAX_FOLD_DEG)
}

/**
 * The pitch of column `c`'s FRONT (window) panel — row 0 — clamped to
 * ±MAX_FOLD_DEG. Missing / non-numeric reads as 0 (a flat front), so the walkers
 * never produce NaN and a pre-pitched-fronts config behaves exactly as before.
 *
 * @param {object} config normalized config
 * @param {number} c column index
 * @returns {number} degrees
 */
export function columnStartPitchDeg(config, c) {
  const column = Array.isArray(config.columns) ? config.columns[c] : undefined
  if (!isPlainObject(column)) return 0
  const v = Number(column.startPitchDeg)
  if (!Number.isFinite(v)) return 0
  return clamp(v, -MAX_FOLD_DEG, MAX_FOLD_DEG)
}

/**
 * How column `c`'s deep end is held up: 'wall' only when it says so (and only the
 * wall-adjacent column may — see `validateConfig`), 'floor' for anything else.
 *
 * @param {object} config normalized config
 * @param {number} c column index
 * @returns {'floor'|'wall'}
 */
export function columnEndSupport(config, c) {
  const column = Array.isArray(config.columns) ? config.columns[c] : undefined
  return isPlainObject(column) && column.endSupport === 'wall' ? 'wall' : 'floor'
}

/**
 * The chain start height y0 for a front panel pitched `pitchDeg` and occupying
 * `along` cm of the chain — the value that rests that panel's SOLID on the floor
 * with its front edge still on the window line z = 0:
 *
 *   y0 = max(0, 3.7·cos θ) − min(0, L·sin θ)
 *
 * The first term is the housing's own drop below the lit face at the front edge
 * (zero once the panel has tipped past vertical, where the front edge itself is
 * the lowest point); the second lifts the whole chain when a NEGATIVE pitch dives
 * the panel's rear edge below the floor, so it is the rear edge that rests.
 *
 * θ = 0 gives exactly `SHORE_Y` (3.7·1, no float noise) for any L — the backward
 * compatibility guarantee.
 *
 * @param {number} pitchDeg row-0 pitch, degrees
 * @param {number} along the front panel's footprint along the chain, cm
 * @returns {number} cm
 */
export function frontRestY(pitchDeg, along) {
  const psi = Number(pitchDeg)
  if (!Number.isFinite(psi)) return SHORE_Y
  const L = Number(along)
  const rise = Number.isFinite(L) ? L * Math.sin(psi * DEG) : 0
  const housing = PANEL_PROFILE.overallThickness * Math.cos(psi * DEG)
  return Math.max(0, housing) - Math.min(0, rise)
}

/**
 * `frontRestY` for a whole column — the y its chain walk starts from.
 *
 * @param {object} config config (raw or normalized — normalized internally)
 * @param {number} c column index
 * @returns {number} cm
 */
export function chainStartY(config, c) {
  const cfg = normalizeConfig(config)
  const segments = columnSegments(cfg, c)
  const along = segments.length > 0 ? Number(segments[0].length) : Number(cfg.cell.size)
  return frontRestY(columnStartPitchDeg(cfg, c), along)
}

/**
 * The chain segments of column `c`, front (window) to back.
 *
 * Each segment consumes one or two rows and carries the length it advances the
 * chain by:
 *   'square'  → one row, `cell.size`
 *   'vrect'   → TWO rows, `cell.rectLength` (a vertical plate; joint k = r is
 *               interior to it and never visited, hence removed)
 *   'hrect'   → one row, `cell.size`; the plate this column owns, whose 121cm
 *               length runs across columns c and c+1
 *   'phantom' → one row, `cell.size`, emits NO panel: the cell belongs to the
 *               horizontal plate owned by column c-1. The chain advances exactly
 *               as a square would, so the rest of the column is unaffected.
 *
 * Shared by placement.js (which emits panels) and the chain walker below (which
 * validation uses), so the two can never disagree about the surface.
 *
 * @param {object} config normalized config
 * @param {number} c column index
 * @returns {Array<{rows:number[], length:number, kind:string}>}
 */
export function columnSegments(config, c) {
  const rowCount = Number(config.grid.rows)
  const size = Number(config.cell.size)
  const rectLength = Number(config.cell.rectLength)
  if (!Number.isFinite(rowCount) || rowCount < 1) return []

  const vAnchor = new Set()
  const hAnchor = new Set()
  const hPhantom = new Set()
  for (const rect of Array.isArray(config.rects) ? config.rects : []) {
    if (!isPlainObject(rect)) continue
    if (rect.orientation === 'vertical' && rect.col === c) vAnchor.add(rect.row)
    else if (rect.orientation === 'horizontal' && rect.col === c) hAnchor.add(rect.row)
    else if (rect.orientation === 'horizontal' && rect.col + 1 === c) hPhantom.add(rect.row)
  }

  const segments = []
  for (let r = 0; r < rowCount; ) {
    if (vAnchor.has(r) && r + 1 < rowCount) {
      segments.push({ rows: [r, r + 1], length: rectLength, kind: 'vrect' })
      r += 2
      continue
    }
    if (hAnchor.has(r)) segments.push({ rows: [r], length: size, kind: 'hrect' })
    else if (hPhantom.has(r)) segments.push({ rows: [r], length: size, kind: 'phantom' })
    else segments.push({ rows: [r], length: size, kind: 'square' })
    r += 1
  }
  return segments
}

/**
 * Walk column `c`'s chain in the 2D (Y, Z) plane — the cheap model both
 * validation and placement are checked against.
 *
 * Start at (y = `chainStartY(cfg, c)`, z = 0) with pitch ψ = `startPitchDeg`, then
 * per segment:
 *   d = (sinψ, cosψ)                             (in (y, z))
 *   advance p += length · d
 *   at the joint k after the segment (k = its last row, when k ≤ rows-2):
 *     p += gap · bisector(d(ψ), d(ψ + f))        ← the gap along the BISECTOR
 *     ψ += f                                      with f = columnFoldDeg(c, k)
 * Advancing along the bisector is what makes every in-column joint exact: the
 * near panel's trailing edge and the far panel's leading edge end up exactly
 * `gap` apart, edge-parallel, with zero skew.
 *
 * The START is where PITCHED FRONTS enter the geometry: z is always 0 (the front
 * edge is on the window line) and y is whatever rests the front panel's solid on
 * the floor at its pitch. At `startPitchDeg` 0 that is exactly 3.7 = SHORE_Y, so
 * the flat lattice is untouched.
 *
 * `origins[r]` is where cell (r, c)'s own 60cm sub-rectangle STARTS along the
 * chain. For a vertical plate's second cell that is `rectLength - size` past the
 * plate's near edge (the plate's cells are credited from its outer ends — see
 * placement.js `panelCellFaceRectLocal`), which at the defaults is exactly
 * `size + gap` = 61: the plate's second cell starts precisely where a separate
 * square would have, so a plate changes nothing behind it. Any mismatch between
 * `rectLength` and `2·size + gap` (W_RECT_LENGTH) shows up here as slack.
 *
 * @param {object} config config (raw or normalized — normalized internally)
 * @param {number} c column index
 * @returns {{ col:number, pitchesDeg:number[], origins:Array<[number,number]>,
 *             points:Array<[number,number]>, segments:Array }}
 */
export function columnChain(config, c) {
  const cfg = normalizeConfig(config)
  const rowCount = Number(cfg.grid.rows)
  const size = Number(cfg.cell.size)
  const gapRaw = Number(cfg.gap)
  const gap = Number.isFinite(gapRaw) ? gapRaw : 0
  if (!Number.isFinite(rowCount) || rowCount < 1) {
    return { col: c, pitchesDeg: [], origins: [], points: [], segments: [] }
  }

  const segments = columnSegments(cfg, c)
  const startPitch = columnStartPitchDeg(cfg, c)
  const pitchesDeg = new Array(rowCount).fill(startPitch)
  const origins = new Array(rowCount)
  const points = []

  let y = frontRestY(startPitch, segments.length > 0 ? segments[0].length : size)
  let z = 0
  let psi = startPitch
  points.push([y, z])

  for (const seg of segments) {
    const dy = Math.sin(psi * DEG)
    const dz = Math.cos(psi * DEG)
    seg.pitchDeg = psi
    seg.origin = [y, z]

    for (const r of seg.rows) pitchesDeg[r] = psi
    origins[seg.rows[0]] = [y, z]
    if (seg.rows.length === 2) {
      const back = Number(seg.length) - size
      origins[seg.rows[1]] = [y + back * dy, z + back * dz]
    }

    const L = Number.isFinite(Number(seg.length)) ? Number(seg.length) : 0
    y += L * dy
    z += L * dz
    points.push([y, z])

    const k = seg.rows[seg.rows.length - 1]
    if (k <= rowCount - 2) {
      const f = columnFoldDeg(cfg, c, k)
      const ny = Math.sin((psi + f) * DEG)
      const nz = Math.cos((psi + f) * DEG)
      let bx = dy + ny
      let bz = dz + nz
      const len = Math.hypot(bx, bz)
      // |f| ≤ 120° means d(ψ) and d(ψ+f) are never antiparallel; guard anyway.
      if (len < 1e-12) {
        bx = dy
        bz = dz
      } else {
        bx /= len
        bz /= len
      }
      y += gap * bx
      z += gap * bz
      points.push([y, z])
      psi += f
    }
  }

  return { col: c, pitchesDeg, origins, points, segments }
}

/**
 * Per-row cumulative pitch (degrees) for column `c` — the transpose of v1's
 * `cellTilts`. Both validation (E_CROSSCOL_ANGLE_MISMATCH) and the placement
 * solver use it, so the surface being checked is the surface being placed.
 *
 * `[0]` is the column's `startPitchDeg`, NOT necessarily 0: the front panel's
 * pitch is designable now, and every later row's pitch is that plus the folds in
 * front of it.
 *
 * @param {object} config config (raw or normalized — normalized internally)
 * @param {number} c column index
 * @returns {number[]} length grid.rows, cumulative pitch in degrees per row
 */
export function cellPitches(config, c) {
  return columnChain(config, c).pitchesDeg
}

// -----------------------------------------------------------------------------
// validateConfig
// -----------------------------------------------------------------------------
/**
 * Validate a config. Safe to call on raw (un-normalized) input.
 *
 * @param {object} config
 * @returns {{ ok: boolean,
 *             errors: Array<{code:string,message:string,path:string}>,
 *             warnings: Array<{code:string,message:string,path:string}> }}
 */
export function validateConfig(config) {
  const errors = []
  const warnings = []
  const err = (code, message, path) => errors.push({ code, message, path })
  const warn = (code, message, path) => warnings.push({ code, message, path })

  if (!isPlainObject(config)) {
    err('E_SHAPE', 'config must be an object', '')
    return { ok: false, errors, warnings }
  }

  const cfg = normalizeConfig(config)

  // --- 1a. Top-level shape -------------------------------------------------
  if (cfg.version !== 2) {
    err(
      'E_SHAPE',
      `version must be 2 — v1 (row-accordion) configs are not supported (got ${JSON.stringify(cfg.version)})`,
      'version',
    )
  }
  if (cfg.units !== 'cm') {
    err('E_SHAPE', `units must be "cm" (got ${JSON.stringify(cfg.units)})`, 'units')
  }

  const cols = cfg.grid.cols
  const rowCount = cfg.grid.rows
  let gridOk = true
  if (!isInt(cols) || cols < 1) {
    err('E_SHAPE', `grid.cols must be an integer ≥ 1 (got ${JSON.stringify(cols)})`, 'grid.cols')
    gridOk = false
  }
  if (!isInt(rowCount) || rowCount < MIN_ROWS || rowCount > MAX_ROWS) {
    err(
      'E_SHAPE',
      `grid.rows must be an integer in ${MIN_ROWS}..${MAX_ROWS} (got ${JSON.stringify(rowCount)}) — ` +
        `below ${MIN_ROWS} the design stops reading as a shore with a wave breaking behind it, ` +
        `above ${MAX_ROWS} the strip runs out of store`,
      'grid.rows',
    )
    gridOk = false
  }

  if (!isFiniteNumber(cfg.cell.size) || cfg.cell.size <= 0) {
    err('E_SHAPE', `cell.size must be a positive number (got ${JSON.stringify(cfg.cell.size)})`, 'cell.size')
  }
  if (!isFiniteNumber(cfg.cell.rectLength) || cfg.cell.rectLength <= 0) {
    err(
      'E_SHAPE',
      `cell.rectLength must be a positive number (got ${JSON.stringify(cfg.cell.rectLength)})`,
      'cell.rectLength',
    )
  }
  if (!isFiniteNumber(cfg.gapTolerance) || cfg.gapTolerance <= 0) {
    err(
      'E_SHAPE',
      `gapTolerance must be a positive number (got ${JSON.stringify(cfg.gapTolerance)})`,
      'gapTolerance',
    )
  }
  if (!isFiniteNumber(cfg.groundTolerance) || cfg.groundTolerance <= 0) {
    err(
      'E_SHAPE',
      `groundTolerance must be a positive number — how close a column's last panel must get to y = 0 to count as touching the floor (got ${JSON.stringify(cfg.groundTolerance)})`,
      'groundTolerance',
    )
  }

  // --- 1b. gap range -------------------------------------------------------
  if (!isFiniteNumber(cfg.gap)) {
    err('E_SHAPE', `gap must be a number (got ${JSON.stringify(cfg.gap)})`, 'gap')
  } else if (cfg.gap <= 0 || cfg.gap > MAX_GAP_CM) {
    err('E_RANGE', `gap must be in (0, ${MAX_GAP_CM}] cm (got ${cfg.gap})`, 'gap')
  }

  // --- 1c. columns ---------------------------------------------------------
  let columnsOk = true
  if (!Array.isArray(cfg.columns)) {
    err(
      'E_SHAPE',
      'columns must be an array of { foldsDeg: [...] } — one entry per grid column',
      'columns',
    )
    columnsOk = false
  } else if (gridOk && cfg.columns.length !== cols) {
    err(
      'E_SHAPE',
      `columns must have exactly grid.cols (${cols}) entries (got ${cfg.columns.length})`,
      'columns',
    )
    columnsOk = false
  }

  if (Array.isArray(cfg.columns)) {
    cfg.columns.forEach((column, c) => {
      const path = `columns[${c}]`
      if (!isPlainObject(column)) {
        err('E_SHAPE', 'column entry must be an object', path)
        columnsOk = false
        return
      }

      // --- the front panel's pitch (row 0) ---------------------------------
      if (!isFiniteNumber(column.startPitchDeg)) {
        err(
          'E_SHAPE',
          `startPitchDeg must be a number — the pitch of this column's front / window panel, ` +
            `0 for a flat front (got ${JSON.stringify(column.startPitchDeg)})`,
          `${path}.startPitchDeg`,
        )
        columnsOk = false
      } else if (Math.abs(column.startPitchDeg) > MAX_FOLD_DEG) {
        err(
          'E_RANGE',
          `startPitchDeg must be within ±${MAX_FOLD_DEG}° (got ${column.startPitchDeg})`,
          `${path}.startPitchDeg`,
        )
      }

      // --- how the deep end is held up ------------------------------------
      if (!END_SUPPORTS.includes(column.endSupport)) {
        err(
          'E_SHAPE',
          `endSupport must be ${END_SUPPORTS.map((s) => `"${s}"`).join(' or ')} ` +
            `(got ${JSON.stringify(column.endSupport)})`,
          `${path}.endSupport`,
        )
        columnsOk = false
      } else if (column.endSupport === 'wall' && c !== WALL_COLUMN) {
        err(
          'E_SHAPE',
          `endSupport "wall" is only valid for the WALL-ADJACENT column (columns[${WALL_COLUMN}], ` +
            `the first one) — the wall closes the −X edge of the grid at x = ${WALL_X}, which is ` +
            `column ${WALL_COLUMN}'s outer edge, so column ${c} has grid between it and the wall ` +
            `and nothing to bracket to. Use "floor" and land its last panel instead.`,
          `${path}.endSupport`,
        )
        columnsOk = false
      }

      if (!Array.isArray(column.foldsDeg)) {
        err(
          'E_SHAPE',
          `foldsDeg must be an array of signed hinge angles (got ${JSON.stringify(column.foldsDeg)})`,
          `${path}.foldsDeg`,
        )
        columnsOk = false
        return
      }
      if (gridOk && column.foldsDeg.length !== rowCount - 1) {
        err(
          'E_SHAPE',
          `foldsDeg must have exactly grid.rows-1 (${rowCount - 1}) entries — one per joint between consecutive rows (got ${column.foldsDeg.length})`,
          `${path}.foldsDeg`,
        )
        columnsOk = false
      }
      column.foldsDeg.forEach((v, k) => {
        if (!isFiniteNumber(v)) {
          err(
            'E_SHAPE',
            `foldsDeg entry must be a number (got ${JSON.stringify(v)})`,
            `${path}.foldsDeg[${k}]`,
          )
          columnsOk = false
        } else if (Math.abs(v) > MAX_FOLD_DEG) {
          err(
            'E_RANGE',
            `fold must be within ±${MAX_FOLD_DEG}° (got ${v})`,
            `${path}.foldsDeg[${k}]`,
          )
        }
      })
    })
  }

  // --- 2. Rects ------------------------------------------------------------
  if (!Array.isArray(cfg.rects)) {
    err('E_SHAPE', 'rects must be an array', 'rects')
    return finish(errors, warnings)
  }

  const validRects = []
  cfg.rects.forEach((rect, i) => {
    const path = `rects[${i}]`
    if (!isPlainObject(rect)) {
      err('E_SHAPE', 'rect entry must be an object', path)
      return
    }
    let shapeOk = true
    if (!isInt(rect.row)) {
      err('E_SHAPE', `rect.row must be an integer (got ${JSON.stringify(rect.row)})`, `${path}.row`)
      shapeOk = false
    }
    if (!isInt(rect.col)) {
      err('E_SHAPE', `rect.col must be an integer (got ${JSON.stringify(rect.col)})`, `${path}.col`)
      shapeOk = false
    }
    if (rect.orientation !== 'horizontal' && rect.orientation !== 'vertical') {
      err(
        'E_SHAPE',
        `rect.orientation must be "horizontal" or "vertical" (got ${JSON.stringify(rect.orientation)})`,
        `${path}.orientation`,
      )
      shapeOk = false
    }
    if (!shapeOk || !gridOk) return

    // Bounds — dedicated messages, E_SHAPE code.
    if (rect.row < 0 || rect.row > rowCount - 1) {
      err('E_SHAPE', `rect out of bounds: row ${rect.row} not in 0..${rowCount - 1}`, `${path}.row`)
      return
    }
    if (rect.col < 0 || rect.col > cols - 1) {
      err('E_SHAPE', `rect out of bounds: col ${rect.col} not in 0..${cols - 1}`, `${path}.col`)
      return
    }
    if (rect.orientation === 'horizontal' && rect.col > cols - 2) {
      err(
        'E_SHAPE',
        `horizontal rect out of bounds: needs col ≤ ${cols - 2} to span columns ${rect.col} and ${rect.col + 1} (got ${rect.col})`,
        `${path}.col`,
      )
      return
    }
    if (rect.orientation === 'vertical' && rect.row > rowCount - 2) {
      err(
        'E_SHAPE',
        `vertical rect out of bounds: needs row ≤ ${rowCount - 2} to span rows ${rect.row} and ${rect.row + 1} (got ${rect.row})`,
        `${path}.row`,
      )
      return
    }
    validRects.push({ rect, index: i })
  })

  // Overlap — two rects sharing a cell.
  const owner = new Map() // "r,c" -> rect index
  for (const { rect, index } of validRects) {
    const cells =
      rect.orientation === 'horizontal'
        ? [[rect.row, rect.col], [rect.row, rect.col + 1]]
        : [[rect.row, rect.col], [rect.row + 1, rect.col]]
    for (const [r, c] of cells) {
      const key = `${r},${c}`
      if (owner.has(key)) {
        err(
          'E_RECT_OVERLAP',
          `rects[${index}] overlaps rects[${owner.get(key)}] at cell (${r}, ${c}) — each cell may belong to at most one rect`,
          `rects[${index}]`,
        )
      } else {
        owner.set(key, index)
      }
    }
  }

  // --- 3. Vertical rects: the joint they remove must not be folded ---------
  if (columnsOk && gridOk) {
    for (const { rect, index } of validRects) {
      if (rect.orientation !== 'vertical') continue
      const fold = Number(cfg.columns[rect.col]?.foldsDeg?.[rect.row])
      if (isFiniteNumber(fold) && fold !== 0) {
        err(
          'E_FOLD_ON_REMOVED_JOINT',
          `columns[${rect.col}].foldsDeg[${rect.row}] is ${fold}°, but the vertical rect rects[${index}] at (${rect.row}, ${rect.col}) is one rigid plate spanning rows ${rect.row} and ${rect.row + 1} — that joint is removed, so set it to 0 or remove the rect`,
          `columns[${rect.col}].foldsDeg[${rect.row}]`,
        )
      }
    }
  }

  // --- 4. Horizontal rects: the two chains must agree at that row ---------
  if (columnsOk && gridOk) {
    const chainCache = new Map()
    const chainFor = (c) => {
      if (!chainCache.has(c)) chainCache.set(c, columnChain(cfg, c))
      return chainCache.get(c)
    }
    for (const { rect, index } of validRects) {
      if (rect.orientation !== 'horizontal') continue
      const r = rect.row
      const c = rect.col
      const A = chainFor(c)
      const B = chainFor(c + 1)
      const pA = A.pitchesDeg[r]
      const pB = B.pitchesDeg[r]
      if (
        isFiniteNumber(pA) &&
        isFiniteNumber(pB) &&
        Math.abs(pA - pB) > PITCH_MATCH_EPSILON_DEG
      ) {
        err(
          'E_CROSSCOL_ANGLE_MISMATCH',
          `horizontal rect rects[${index}] at (${r}, ${c}) is one rigid plate but cell (${r}, ${c}) sits at pitch ${pA.toFixed(2)}° and cell (${r}, ${c + 1}) at ${pB.toFixed(2)}° — make the two columns' cumulative pitches equal at row ${r} (adjust columns[${c}].foldsDeg[0..${r - 1}] / columns[${c + 1}].foldsDeg[0..${r - 1}])`,
          `rects[${index}]`,
        )
        continue
      }
      const oA = A.origins[r]
      const oB = B.origins[r]
      if (
        Array.isArray(oA) &&
        Array.isArray(oB) &&
        isFiniteNumber(cfg.gapTolerance) &&
        Math.hypot(oA[0] - oB[0], oA[1] - oB[1]) > cfg.gapTolerance
      ) {
        warn(
          'W_CROSSCOL_POSITION',
          `horizontal rect rects[${index}] at (${r}, ${c}) spans two chains that agree in pitch but not in position: cell (${r}, ${c}) starts at (y ${oA[0].toFixed(2)}, z ${oA[1].toFixed(2)}) and cell (${r}, ${c + 1}) at (y ${oB[0].toFixed(2)}, z ${oB[1].toFixed(2)}) — the columns diverged in front of this row, so the rigid plate cannot lie in both; the joint report shows the deviation`,
          `rects[${index}]`,
        )
      }
    }
  }

  // --- 5. Rect length slack — surfaced, never corrected --------------------
  // Silent at the defaults (2·60 + 1 = 121 = rectLength): a plate is an exact
  // stand-in for two squares plus their joint. Only a mismatched combination
  // (say gap 2 against a 121cm plate) has slack to report.
  if (cfg.rects.length > 0 && isFiniteNumber(cfg.cell.size) && isFiniteNumber(cfg.gap) && isFiniteNumber(cfg.cell.rectLength)) {
    const ideal = 2 * cfg.cell.size + cfg.gap
    if (cfg.cell.rectLength !== ideal) {
      warn(
        'W_RECT_LENGTH',
        `cell.rectLength ${cfg.cell.rectLength}cm ≠ 2·cell.size + gap (${ideal}cm) — ${(ideal - cfg.cell.rectLength).toFixed(2)}cm of slack per rect is accepted and surfaced, not corrected`,
        'cell.rectLength',
      )
    }
  }

  // --- 6. The plate-pattern rule — at least MIN_RECTS plates, by a named rule --
  // Last, so the warning order matches the doc order above. A warning, not an
  // error: the grid map removes plates one click at a time, and a design in the
  // middle of being re-patterned has to stay committable.
  if (cfg.rects.length < MIN_RECTS) {
    warn(
      'W_FEW_RECTS',
      `designs should use ≥ ${MIN_RECTS} rect plates (got ${cfg.rects.length}) — the modularity of ` +
        `the panel system should read in the pattern. Place at least ${MIN_RECTS} 60×121 plates by one ` +
        `nameable rule (mirrored pairs, a plate at every column's crest, a diagonal cascade, the last ` +
        `two rows of alternating columns, …), design the pitch profiles to hold the plateaus that rule ` +
        `needs, and record the rule in meta.rectPattern`,
      'rects',
    )
  }

  return finish(errors, warnings)
}

function finish(errors, warnings) {
  return { ok: errors.length === 0, errors, warnings }
}
