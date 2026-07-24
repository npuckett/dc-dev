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
 * across the window along +X.
 *
 * FOLDING HAPPENS ONLY ALONG COLUMNS. Each column c is an independent STRIP: a
 * chain of `rows` panels hinged at the `rows - 1` joints between consecutive
 * rows, folding up and back from the window. There is NO in-row accordion — the
 * v1 fold family (zigzagDeg / jointOverridesDeg / rowFoldsDeg) is gone.
 *
 * Consequences worth internalizing before generating a config:
 *   - Columns NEVER move in X. Column c occupies x ∈ [c·(size+gap),
 *     c·(size+gap)+size] in every configuration, so column centers sit on the
 *     lattice x = size/2 + c·(size+gap) (30, 92, 154, … at the defaults).
 *   - Side-by-side joints along a row are therefore SIMPLE CONNECTIONS: a
 *     constant `gap` in X by construction. Their 3D edge separation only grows
 *     when two adjacent columns' fold profiles differ in pitch or in the height
 *     / depth they have reached — that is exactly what report.js measures.
 *   - Row 0 is the SHORE and is flat on the ground BY CONSTRUCTION: every chain
 *     starts at pitch 0, z = 0, y = PANEL_PROFILE.overallThickness. There is no
 *     shore validation rule; you cannot tilt row 0.
 *   - WAVES ARE MADE BY PHASE-SHIFTING FOLD SEQUENCES ACROSS COLUMNS. Give
 *     column c a fold sequence whose crest sits one step further back than
 *     column c-1's, and the surface reads as a swell travelling across the
 *     window. Identical sequences in every column give a cylindrical fold with
 *     perfect in-row joints.
 *   - GROUND SUPPORT MATTERS. Panels dipping below y = 0 have nothing to stand
 *     on; placement.js emits W_BELOW_FLOOR for them. A chain that rises before
 *     it descends (all cumulative pitches ≥ 0, or a descent no deeper than the
 *     height already gained) stays supported.
 *   - THE WAVE RETURNS TO THE WATER — see the next section. Every column's LAST
 *     panel must come back down and touch the floor.
 *
 * =============================================================================
 * THE GROUNDED-END RULE (a MUST for every generated design)
 * =============================================================================
 * THE LAST PANEL OF EVERY COLUMN MUST TOUCH THE FLOOR. Either it lies flat on
 * the ground, or it is tilted with one of its housing edges down on the ground —
 * it may NEVER float in mid-air. ("The wave returns to the water.")
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
 *   The chain climbs by 60·sin ψ_r per row (plus a couple of cm of gap), so the
 *   height it has to give back by the end is essentially 60·Σ_r sin ψ_r. Author
 *   pitch profiles whose sines CANCEL — arc up and then back down — e.g.
 *   [0, 35, −35, 0, 0], [0, 35, 0, −35, 0], [0, 10, 10, −10, −10]. Profiles that
 *   only ever rise ([0, 20, 40, 60, 80]) end tens of cm in the air and cannot be
 *   rescued: the last panel is 60cm long, so once the chain's last vertex is much
 *   above ~57cm NO value of the last hinge can reach the ground.
 *   `core/ground.js` exports `solveGroundingFold(config, c)` / `groundAllFolds`,
 *   which solve the last surviving hinge of a column for floor contact — that is
 *   the mechanical fix; designing the arc is still the generator's job.
 *
 * =============================================================================
 * JSON CONFIG SCHEMA (v2) — this is the shape an LLM generator should emit
 * =============================================================================
 *
 *   {
 *     "version": 2,                            // must be exactly 2
 *     "units": "cm",                           // must be "cm"
 *     "name": "wave study 3",                  // optional, free text
 *     "grid": { "cols": 6, "rows": 5 },        // 6 columns × 5 rows
 *     "cell": { "size": 60, "rectLength": 121 },  // square cell 60cm; 2-cell rect 121cm
 *     "gap": 2.0,                              // physical joint gap spanned by connectors
 *     "gapTolerance": 1.5,                     // report flags joints deviating > this from gap
 *     "groundTolerance": 0.5,                  // a column's last panel counts as
 *                                              // touching the floor when its solid
 *                                              // reaches within this many cm of y = 0
 *     "columns": [                             // exactly grid.cols entries; 0 = leftmost (x = 0 side)
 *       { "foldsDeg": [30, -60, 60, -30] },    // exactly grid.rows-1 SIGNED hinge angles
 *       { "foldsDeg": [20, -40, 40, -20] },    // [k] = hinge between rows k and k+1
 *       { "foldsDeg": [0, 0, 0, 0] },
 *       { "foldsDeg": [0, 0, 0, 0] },
 *       { "foldsDeg": [0, 0, 0, 0] },
 *       { "foldsDeg": [0, 0, 0, 0] }
 *     ],
 *     "rects": [                               // optional 60×121 two-cell panels
 *       { "row": 1, "col": 2, "orientation": "horizontal" },  // (1,2)+(1,3): one plate across two columns
 *       { "row": 2, "col": 5, "orientation": "vertical" }     // (2,5)+(3,5): removes joint k=2 of column 5
 *     ],
 *     "meta": { "preset": "wave", "seed": 42, "notes": "" }   // provenance, free-form
 *   }
 *
 * =============================================================================
 * ANGLE SEMANTICS (verbatim rules — follow these exactly)
 * =============================================================================
 * Joint k of column c is the hinge between rows k and k+1 (k = 0 .. rows-2).
 * `columns[c].foldsDeg[k]` is that hinge's SIGNED dihedral in degrees, clamped
 * to ±120. A POSITIVE fold pitches the next panel UP (toward +Y).
 *
 * Cumulative pitch of cell (r, c) is the running sum of the folds in front of it:
 *
 *   ψ(r, c) = Σ_{k < r} foldsDeg(c, k),   ψ(0, c) = 0
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
 *   E_SHAPE                     structural problems: bad version/units, grid
 *                               dims < 1, wrong columns / foldsDeg lengths,
 *                               malformed column or rect entries, out-of-bounds
 *                               rects, non-positive gapTolerance / groundTolerance
 *                               (message names the specific problem)
 *   E_RANGE                     a fold outside ±120, gap not in (0, 10]
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
 *   W_RECT_LENGTH               cell.rectLength ≠ 2·cell.size + gap (the real
 *                               hardware is 121 vs 122 — ~1cm of slack is
 *                               accepted and surfaced, never silently corrected)
 *
 * The grounded-end rule is deliberately NOT here: it is a solved-layout property,
 * reported as `solveLayout(...).violations` (E_END_FLOATING), not a config error.
 */

import { PANEL_PROFILE } from '../config.js'

// -----------------------------------------------------------------------------
// Limits
// -----------------------------------------------------------------------------
/** Per-joint signed dihedral limit, degrees. */
export const MAX_FOLD_DEG = 120
export const MAX_GAP_CM = 10
/** Two cells of a horizontal rect must agree in pitch to within this many degrees. */
export const PITCH_MATCH_EPSILON_DEG = 0.1

// -----------------------------------------------------------------------------
// Defaults
// -----------------------------------------------------------------------------
export const DEFAULT_GRID = { cols: 6, rows: 5 }
export const DEFAULT_CELL = { size: 60, rectLength: 121 }
export const DEFAULT_GAP = 2.0
export const DEFAULT_GAP_TOLERANCE = 1.5
/**
 * How close a column's last panel has to get to y = 0 to count as TOUCHING the
 * floor, cm. 0.5cm ≈ the slack a printed foot / the floor's own flatness eats.
 */
export const DEFAULT_GROUND_TOLERANCE = 0.5
/** Every chain starts with its lit face at this height (housings on the floor). */
export const SHORE_Y = PANEL_PROFILE.overallThickness

/** Flat 6×5 surface: every joint unfolded, no rects. */
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
    { foldsDeg: [0, 0, 0, 0] },
    { foldsDeg: [0, 0, 0, 0] },
    { foldsDeg: [0, 0, 0, 0] },
    { foldsDeg: [0, 0, 0, 0] },
    { foldsDeg: [0, 0, 0, 0] },
    { foldsDeg: [0, 0, 0, 0] },
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

function normalizeColumns(columns) {
  if (!Array.isArray(columns)) return columns
  return columns.map((column) => {
    if (!isPlainObject(column)) return column
    return {
      foldsDeg: Array.isArray(column.foldsDeg) ? column.foldsDeg.slice() : column.foldsDeg,
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
 * Start at (y = SHORE_Y, z = 0) with pitch ψ = 0, then per segment:
 *   d = (sinψ, cosψ)                             (in (y, z))
 *   advance p += length · d
 *   at the joint k after the segment (k = its last row, when k ≤ rows-2):
 *     p += gap · bisector(d(ψ), d(ψ + f))        ← the gap along the BISECTOR
 *     ψ += f                                      with f = columnFoldDeg(c, k)
 * Advancing along the bisector is what makes every in-column joint exact: the
 * near panel's trailing edge and the far panel's leading edge end up exactly
 * `gap` apart, edge-parallel, with zero skew.
 *
 * `origins[r]` is where cell (r, c)'s own 60cm sub-rectangle STARTS along the
 * chain. For a vertical plate's second cell that is `rectLength - size` past the
 * plate's near edge (the plate's cells are credited from its outer ends — see
 * placement.js `panelCellFaceRectLocal`), which is where the ~1cm of hardware
 * slack shows up.
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
  const pitchesDeg = new Array(rowCount).fill(0)
  const origins = new Array(rowCount)
  const points = []

  let y = SHORE_Y
  let z = 0
  let psi = 0
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
  if (!isInt(rowCount) || rowCount < 1) {
    err('E_SHAPE', `grid.rows must be an integer ≥ 1 (got ${JSON.stringify(rowCount)})`, 'grid.rows')
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

  return finish(errors, warnings)
}

function finish(errors, warnings) {
  return { ok: errors.length === 0, errors, warnings }
}
