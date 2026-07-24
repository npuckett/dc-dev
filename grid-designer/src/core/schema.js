/**
 * grid-designer — configuration schema, normalization and validation.
 *
 * HEADLESS ZONE (src/core/): pure functions, importable from plain node.
 *   - explicit `.js` extensions on ALL relative imports
 *   - may import `three` math classes only; never components / store / DOM
 *   - same input → same output, no hidden state
 *
 * =============================================================================
 * WHAT THIS DESCRIBES
 * =============================================================================
 * A single continuous folded surface made of a `cols × rows` grid of ceiling
 * light panels (Drop Ceiling installation V2, the "agentic body of water").
 * Row 0 is the SHORE: the row at the storefront window, fixed straight and flat
 * on the floor. Rows behind it (increasing row index, receding along +Z) fold
 * up and back.
 *
 * Two independent families of folds shape the surface:
 *   1. WITHIN-ROW ZIG-ZAG (accordion): each row has one base angle applied with
 *      alternating sign at the joints between its columns, plus optional signed
 *      per-joint overrides.
 *   2. ROW-TO-ROW DIHEDRAL: one hinge angle per row pair, pitching each
 *      successive row plane up (or down).
 *
 * World conventions (for downstream placement, not enforced here): units cm,
 * Y up, columns along +X (c = 0..cols-1), rows recede along +Z (r = 0..rows-1),
 * cell pitch = cell.size + gap.
 *
 * =============================================================================
 * JSON CONFIG SCHEMA (v1) — this is the shape an LLM generator should emit
 * =============================================================================
 *
 *   {
 *     "version": 1,                            // must be exactly 1
 *     "units": "cm",                           // must be "cm"
 *     "name": "wave study 3",                  // optional, free text
 *     "grid": { "cols": 6, "rows": 5 },        // 6 columns × 5 rows
 *     "cell": { "size": 60, "rectLength": 121 },  // square cell 60cm; 2-cell rect 121cm
 *     "gap": 2.0,                              // physical joint gap spanned by connectors
 *     "gapTolerance": 1.5,                     // report flags joints deviating > this from gap
 *     "rowAnchor": "center",                   // "center" (default) | "start"
 *     "rows": [                                // exactly grid.rows entries; index 0 = shore
 *       { "zigzagDeg": 0,  "jointOverridesDeg": {} },            // row 0 MUST be 0, no overrides
 *       { "zigzagDeg": 20, "jointOverridesDeg": { "2": -35 } },  // key = joint j, value = SIGNED absolute angle
 *       { "zigzagDeg": 35, "jointOverridesDeg": {} },
 *       { "zigzagDeg": 35, "jointOverridesDeg": {} },
 *       { "zigzagDeg": 10, "jointOverridesDeg": {} }
 *     ],
 *     "rowFoldsDeg": [15, 40, 55, -10],        // grid.rows-1 entries; [r] = dihedral between rows r and r+1
 *     "rects": [                               // optional 60×121 two-cell panels
 *       { "row": 1, "col": 2, "orientation": "horizontal" },  // occupies (1,2)+(1,3); removes joint j=2 of row 1
 *       { "row": 2, "col": 5, "orientation": "vertical" }     // occupies (2,5)+(3,5); one rigid plate
 *     ],
 *     "meta": { "preset": "wave", "seed": 42, "notes": "" }    // provenance, free-form
 *   }
 *
 * =============================================================================
 * ANGLE SEMANTICS (verbatim rules — follow these exactly)
 * =============================================================================
 * Joint j sits between columns j and j+1 (j = 0 .. cols-2).
 *
 *   Fold angle  φ(r,j) = (j % 2 === 0 ? +1 : −1) · rows[r].zigzagDeg
 *
 * unless `rows[r].jointOverridesDeg[j]` exists — that SIGNED value is used
 * as-is (it is an absolute angle, not a delta, and it carries its own sign;
 * the alternation rule does not apply to it). Fold angles are clamped to ±80°.
 *
 * Row pitch is cumulative:
 *
 *   ψ_r = Σ_{k < r} rowFoldsDeg[k],   ψ_0 = 0
 *
 * with each `rowFoldsDeg` entry clamped to ±120°. A positive `rowFoldsDeg[r]`
 * pitches row r+1 UP relative to row r.
 *
 * Per-cell in-row tilt is the running sum of the surviving folds to its left:
 * see `cellTilts()` below. A horizontal rect spanning columns c and c+1 REMOVES
 * joint j = c of that row, so no fold is applied there and both of its cells
 * share one tilt (it is a single rigid plate).
 *
 * =============================================================================
 * VALIDATION CODES
 * =============================================================================
 * Errors (ok === false):
 *   E_SHAPE                       structural problems: bad version/units, grid
 *                                 dims < 1, wrong rows / rowFoldsDeg lengths,
 *                                 malformed row or rect entries, out-of-bounds
 *                                 rects (message names the specific problem)
 *   E_RANGE                       zigzag or override outside ±80, rowFold
 *                                 outside ±120, gap not in (0, 10]
 *   E_SHORE_NOT_FLAT              row 0 has a non-zero zigzagDeg or any overrides
 *   E_RECT_OVERLAP                two rects share a cell
 *   E_OVERRIDE_ON_REMOVED_JOINT   a jointOverridesDeg entry targets a joint that
 *                                 a horizontal rect removed
 *   E_CROSSROW_ANGLE_MISMATCH     a vertical rect's two cells have different
 *                                 computed in-row tilts (> 0.1°) — a rigid plate
 *                                 cannot span them
 * Warnings (do not affect ok):
 *   W_CROSSROW_FOLD               a vertical rect spans a row boundary whose
 *                                 rowFoldsDeg ≠ 0 — the rigid plate cannot bend,
 *                                 so the joint report will show the deviation
 *   W_RECT_LENGTH                 cell.rectLength ≠ 2·cell.size + gap (the real
 *                                 hardware is 121 vs 122 — ~1cm of slack is
 *                                 accepted and surfaced, never silently corrected)
 */

// -----------------------------------------------------------------------------
// Limits
// -----------------------------------------------------------------------------
export const MAX_ZIGZAG_DEG = 80
export const MAX_ROW_FOLD_DEG = 120
export const MAX_GAP_CM = 10
/** Two cells of a vertical rect must agree in tilt to within this many degrees. */
export const TILT_MATCH_EPSILON_DEG = 0.1

// -----------------------------------------------------------------------------
// Defaults
// -----------------------------------------------------------------------------
export const DEFAULT_GRID = { cols: 6, rows: 5 }
export const DEFAULT_CELL = { size: 60, rectLength: 121 }
export const DEFAULT_GAP = 2.0
export const DEFAULT_GAP_TOLERANCE = 1.5
export const DEFAULT_ROW_ANCHOR = 'center'

/** Flat 6×5 surface: no zig-zag, no row folds, no rects. */
export const DEFAULT_CONFIG = Object.freeze({
  version: 1,
  units: 'cm',
  name: 'flat 6×5',
  grid: { cols: 6, rows: 5 },
  cell: { size: 60, rectLength: 121 },
  gap: DEFAULT_GAP,
  gapTolerance: DEFAULT_GAP_TOLERANCE,
  rowAnchor: DEFAULT_ROW_ANCHOR,
  rows: [
    { zigzagDeg: 0, jointOverridesDeg: {} },
    { zigzagDeg: 0, jointOverridesDeg: {} },
    { zigzagDeg: 0, jointOverridesDeg: {} },
    { zigzagDeg: 0, jointOverridesDeg: {} },
    { zigzagDeg: 0, jointOverridesDeg: {} },
  ],
  rowFoldsDeg: [0, 0, 0, 0],
  rects: [],
  meta: { preset: 'flat', notes: '' },
})

// -----------------------------------------------------------------------------
// Small helpers
// -----------------------------------------------------------------------------
const isPlainObject = (v) => typeof v === 'object' && v !== null && !Array.isArray(v)
const isFiniteNumber = (v) => typeof v === 'number' && Number.isFinite(v)
const isInt = (v) => isFiniteNumber(v) && Number.isInteger(v)

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
 * report them; only *missing* pieces get defaults. `rows` and `rowFoldsDeg` are
 * deliberately NOT synthesized when absent: their lengths are load-bearing and
 * a missing array is a real error, not something to guess at.
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
    version: src.version !== undefined ? src.version : 1,
    units: src.units !== undefined ? src.units : 'cm',
    grid,
    cell,
    gap: src.gap !== undefined ? src.gap : DEFAULT_GAP,
    gapTolerance: src.gapTolerance !== undefined ? src.gapTolerance : DEFAULT_GAP_TOLERANCE,
    rowAnchor: src.rowAnchor !== undefined ? src.rowAnchor : DEFAULT_ROW_ANCHOR,
    rows: normalizeRows(src.rows),
    rowFoldsDeg: Array.isArray(src.rowFoldsDeg) ? src.rowFoldsDeg.slice() : src.rowFoldsDeg,
    rects: Array.isArray(src.rects) ? src.rects.map(normalizeRect) : [],
    meta: { notes: '', ...(isPlainObject(src.meta) ? src.meta : {}) },
  }

  // `name` is optional — only carry it through when supplied.
  if (src.name !== undefined) out.name = src.name

  return out
}

function normalizeRows(rows) {
  if (!Array.isArray(rows)) return rows
  return rows.map((row) => {
    if (!isPlainObject(row)) return row
    const overrides = {}
    if (isPlainObject(row.jointOverridesDeg)) {
      for (const key of Object.keys(row.jointOverridesDeg)) {
        overrides[String(key)] = row.jointOverridesDeg[key]
      }
    }
    return {
      zigzagDeg: row.zigzagDeg !== undefined ? row.zigzagDeg : 0,
      jointOverridesDeg: isPlainObject(row.jointOverridesDeg) ? overrides : {},
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
 * The set of in-row joints removed by horizontal rects in row `r`.
 * A horizontal rect at (r, c) spans columns c and c+1, removing joint j = c.
 *
 * @param {object} config normalized config
 * @param {number} r row index
 * @returns {Set<number>}
 */
export function removedJoints(config, r) {
  const removed = new Set()
  const rects = Array.isArray(config.rects) ? config.rects : []
  for (const rect of rects) {
    if (!isPlainObject(rect)) continue
    if (rect.orientation === 'horizontal' && rect.row === r) removed.add(rect.col)
  }
  return removed
}

/**
 * Signed fold angle at joint j of row r, per the ANGLE SEMANTICS above:
 * alternating ±zigzagDeg, or the signed override used as-is. Clamped to ±80.
 *
 * @param {object} config normalized config
 * @param {number} r row index
 * @param {number} j joint index (0 .. cols-2)
 * @returns {number} degrees
 */
export function foldAngleDeg(config, r, j) {
  const row = Array.isArray(config.rows) ? config.rows[r] : undefined
  if (!isPlainObject(row)) return 0
  const overrides = isPlainObject(row.jointOverridesDeg) ? row.jointOverridesDeg : {}
  const key = String(j)
  if (Object.prototype.hasOwnProperty.call(overrides, key)) {
    const v = Number(overrides[key])
    return Number.isFinite(v) ? clamp(v, -MAX_ZIGZAG_DEG, MAX_ZIGZAG_DEG) : 0
  }
  const base = Number(row.zigzagDeg)
  if (!Number.isFinite(base)) return 0
  const signed = (j % 2 === 0 ? 1 : -1) * base
  return clamp(signed, -MAX_ZIGZAG_DEG, MAX_ZIGZAG_DEG)
}

/**
 * Per-column in-row tilt (degrees) for row `r` — the cheap 2D accordion chain.
 *
 * Walk the row left to right: α starts at 0, so the tilt of the cell at column
 * 0 is 0. The tilt of the cell at column c is α after applying the folds of
 * joints 0..c-1. Joints removed by horizontal rects apply no fold, which is
 * what makes both cells of a horizontal rect share one tilt.
 *
 * Both the placement solver and E_CROSSROW_ANGLE_MISMATCH validation use this.
 *
 * @param {object} config config (raw or normalized — normalized internally)
 * @param {number} r row index
 * @returns {number[]} length cols, tilt in degrees per column
 */
export function cellTilts(config, r) {
  const cfg = normalizeConfig(config)
  const cols = Number(cfg.grid.cols)
  if (!Number.isFinite(cols) || cols < 1) return []
  const removed = removedJoints(cfg, r)
  const tilts = new Array(cols)
  let alpha = 0
  for (let c = 0; c < cols; c++) {
    tilts[c] = alpha
    const j = c // joint between columns c and c+1
    if (j <= cols - 2 && !removed.has(j)) {
      alpha += foldAngleDeg(cfg, r, j)
    }
  }
  return tilts
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
  if (cfg.version !== 1) {
    err('E_SHAPE', `version must be 1 (got ${JSON.stringify(cfg.version)})`, 'version')
  }
  if (cfg.units !== 'cm') {
    err('E_SHAPE', `units must be "cm" (got ${JSON.stringify(cfg.units)})`, 'units')
  }
  if (cfg.rowAnchor !== 'center' && cfg.rowAnchor !== 'start') {
    err(
      'E_SHAPE',
      `rowAnchor must be "center" or "start" (got ${JSON.stringify(cfg.rowAnchor)})`,
      'rowAnchor',
    )
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

  // --- 1b. gap range -------------------------------------------------------
  if (!isFiniteNumber(cfg.gap)) {
    err('E_SHAPE', `gap must be a number (got ${JSON.stringify(cfg.gap)})`, 'gap')
  } else if (cfg.gap <= 0 || cfg.gap > MAX_GAP_CM) {
    err('E_RANGE', `gap must be in (0, ${MAX_GAP_CM}] cm (got ${cfg.gap})`, 'gap')
  }

  // --- 1c. rows array ------------------------------------------------------
  let rowsOk = true
  if (!Array.isArray(cfg.rows)) {
    err('E_SHAPE', 'rows must be an array', 'rows')
    rowsOk = false
  } else if (gridOk && cfg.rows.length !== rowCount) {
    err(
      'E_SHAPE',
      `rows must have exactly grid.rows (${rowCount}) entries (got ${cfg.rows.length})`,
      'rows',
    )
    rowsOk = false
  }

  const maxJoint = gridOk ? cols - 2 : Infinity

  if (Array.isArray(cfg.rows)) {
    cfg.rows.forEach((row, r) => {
      const path = `rows[${r}]`
      if (!isPlainObject(row)) {
        err('E_SHAPE', 'row entry must be an object', path)
        rowsOk = false
        return
      }
      if (!isFiniteNumber(row.zigzagDeg)) {
        err('E_SHAPE', `zigzagDeg must be a number (got ${JSON.stringify(row.zigzagDeg)})`, `${path}.zigzagDeg`)
        rowsOk = false
      } else if (Math.abs(row.zigzagDeg) > MAX_ZIGZAG_DEG) {
        err(
          'E_RANGE',
          `zigzagDeg must be within ±${MAX_ZIGZAG_DEG}° (got ${row.zigzagDeg})`,
          `${path}.zigzagDeg`,
        )
      }

      const overrides = row.jointOverridesDeg
      if (!isPlainObject(overrides)) {
        err('E_SHAPE', 'jointOverridesDeg must be an object', `${path}.jointOverridesDeg`)
        rowsOk = false
      } else {
        for (const key of Object.keys(overrides)) {
          const jPath = `${path}.jointOverridesDeg["${key}"]`
          const j = Number(key)
          if (!Number.isInteger(j) || key.trim() === '') {
            err('E_SHAPE', `override key must be an integer joint index (got "${key}")`, jPath)
            continue
          }
          if (gridOk && (j < 0 || j > maxJoint)) {
            err(
              'E_SHAPE',
              `joint index ${j} out of range — joints are 0..${maxJoint} for ${cols} columns`,
              jPath,
            )
            continue
          }
          const v = overrides[key]
          if (!isFiniteNumber(v)) {
            err('E_SHAPE', `override value must be a number (got ${JSON.stringify(v)})`, jPath)
          } else if (Math.abs(v) > MAX_ZIGZAG_DEG) {
            err('E_RANGE', `override must be within ±${MAX_ZIGZAG_DEG}° (got ${v})`, jPath)
          }
        }
      }
    })

    // --- 2. Shore row must be flat ----------------------------------------
    const shore = cfg.rows[0]
    if (isPlainObject(shore)) {
      if (isFiniteNumber(shore.zigzagDeg) && shore.zigzagDeg !== 0) {
        err(
          'E_SHORE_NOT_FLAT',
          `row 0 is the shore (at the window) and must stay flat: zigzagDeg must be 0 (got ${shore.zigzagDeg})`,
          'rows[0].zigzagDeg',
        )
      }
      if (isPlainObject(shore.jointOverridesDeg) && Object.keys(shore.jointOverridesDeg).length > 0) {
        err(
          'E_SHORE_NOT_FLAT',
          'row 0 is the shore (at the window) and must stay flat: no jointOverridesDeg allowed',
          'rows[0].jointOverridesDeg',
        )
      }
    }
  }

  // --- 1d. rowFoldsDeg -----------------------------------------------------
  if (!Array.isArray(cfg.rowFoldsDeg)) {
    err('E_SHAPE', 'rowFoldsDeg must be an array', 'rowFoldsDeg')
  } else {
    if (gridOk && cfg.rowFoldsDeg.length !== rowCount - 1) {
      err(
        'E_SHAPE',
        `rowFoldsDeg must have exactly grid.rows-1 (${rowCount - 1}) entries (got ${cfg.rowFoldsDeg.length})`,
        'rowFoldsDeg',
      )
    }
    cfg.rowFoldsDeg.forEach((v, i) => {
      if (!isFiniteNumber(v)) {
        err('E_SHAPE', `rowFoldsDeg entry must be a number (got ${JSON.stringify(v)})`, `rowFoldsDeg[${i}]`)
      } else if (Math.abs(v) > MAX_ROW_FOLD_DEG) {
        err(
          'E_RANGE',
          `rowFoldsDeg must be within ±${MAX_ROW_FOLD_DEG}° (got ${v})`,
          `rowFoldsDeg[${i}]`,
        )
      }
    })
  }

  // --- 3. Rects ------------------------------------------------------------
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

  // Override on a joint removed by a horizontal rect.
  if (rowsOk) {
    for (const { rect, index } of validRects) {
      if (rect.orientation !== 'horizontal') continue
      const row = cfg.rows[rect.row]
      if (!isPlainObject(row) || !isPlainObject(row.jointOverridesDeg)) continue
      const key = String(rect.col)
      if (Object.prototype.hasOwnProperty.call(row.jointOverridesDeg, key)) {
        err(
          'E_OVERRIDE_ON_REMOVED_JOINT',
          `rows[${rect.row}].jointOverridesDeg["${key}"] targets joint ${key}, which the horizontal rect rects[${index}] at (${rect.row}, ${rect.col}) removes — delete the override or move the rect`,
          `rows[${rect.row}].jointOverridesDeg["${key}"]`,
        )
      }
    }
  }

  // Cross-row rigid-plate checks for vertical rects.
  if (rowsOk && gridOk) {
    const tiltCache = new Map()
    const tiltsFor = (r) => {
      if (!tiltCache.has(r)) tiltCache.set(r, cellTilts(cfg, r))
      return tiltCache.get(r)
    }
    for (const { rect, index } of validRects) {
      if (rect.orientation !== 'vertical') continue
      const r = rect.row
      const c = rect.col
      const tA = tiltsFor(r)[c]
      const tB = tiltsFor(r + 1)[c]
      if (
        isFiniteNumber(tA) &&
        isFiniteNumber(tB) &&
        Math.abs(tA - tB) > TILT_MATCH_EPSILON_DEG
      ) {
        err(
          'E_CROSSROW_ANGLE_MISMATCH',
          `vertical rect rects[${index}] at (${r}, ${c}) is one rigid plate but cell (${r}, ${c}) tilts ${tA.toFixed(2)}° and cell (${r + 1}, ${c}) tilts ${tB.toFixed(2)}° — make them equal (adjust rows[${r}]/rows[${r + 1}] zigzagDeg, or add jointOverridesDeg for joints 0..${c - 1} of those rows)`,
          `rects[${index}]`,
        )
      }
      const fold = Array.isArray(cfg.rowFoldsDeg) ? cfg.rowFoldsDeg[r] : undefined
      if (isFiniteNumber(fold) && fold !== 0) {
        warn(
          'W_CROSSROW_FOLD',
          `vertical rect rects[${index}] at (${r}, ${c}) spans the row ${r}/${r + 1} boundary, which folds ${fold}° — the rigid plate cannot bend, so the joint report will show the deviation`,
          `rects[${index}]`,
        )
      }
    }
  }

  // Rect length slack — surfaced, never corrected.
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
