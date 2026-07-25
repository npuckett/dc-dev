/**
 * grid-designer — combining two 60cm squares into one 60×121 plate.
 *
 * HEADLESS ZONE (src/core/): pure functions, importable from plain node.
 *   - explicit `.js` extensions on ALL relative imports
 *   - may import `three` math classes only; never components / store / DOM
 *   - same input → same output, no hidden state
 *
 * =============================================================================
 * WHY THIS MODULE EXISTS
 * =============================================================================
 * Removing a plate is easy: two squares plus their joint are a legal stand-in for
 * a 121cm plate at the defaults (2·60 + 1cm gap = 121), so a split always
 * validates. ADDING one is not, because a plate is RIGID and the schema enforces
 * that (core/schema.js):
 *
 *   vertical plate over rows (r, r+1) of column c
 *     → E_FOLD_ON_REMOVED_JOINT unless `columns[c].foldsDeg[r]` is exactly 0.
 *       The plate spans that hinge; a rigid plate cannot bend.
 *   horizontal plate over columns (c, c+1) at row r
 *     → E_CROSSCOL_ANGLE_MISMATCH unless the two chains are at the SAME
 *       cumulative pitch at row r (and W_CROSSCOL_POSITION unless they are also at
 *       the same point). One plate cannot lie in two different planes.
 *
 * In any designed profile every joint is non-zero and neighbouring columns are
 * phase-shifted, so a bare "place a plate here" click is rejected essentially
 * everywhere — which is why the plan view could take plates away but never put
 * them back. This module supplies the missing half: a merge that carries the
 * GEOMETRIC CONSEQUENCE of making the pair rigid, and hands back ONE config with
 * the consequence and the plate applied together, so validation only ever sees
 * the finished state.
 *
 * =============================================================================
 * WHAT A MERGE IS ALLOWED TO CHANGE
 * =============================================================================
 * VERTICAL (rows r, r+1 of column c) → `columns[c].foldsDeg[r] = 0`.
 *   Not a liberty: that hinge no longer exists once one plate spans both rows.
 *   Everything behind it keeps its own fold, so the strip's shape in front of the
 *   plate is untouched and the rows behind it ride along with the new pitch.
 *
 * HORIZONTAL (columns c, c+1 at row r) → the CLICKED-FIRST column wins: its
 *   `startPitchDeg` and its folds `0..r-1` are copied into the other column, which
 *   makes the two chains identical from the window up to row r and therefore equal
 *   in both pitch and position there. Folds from r onward are left alone, so the
 *   other column keeps its own shape BEHIND the plate.
 *
 * Nothing else is touched, ever. `mergeCells` returns the list of changes it made
 * so the UI can say so before and after the click.
 *
 * =============================================================================
 * A MERGE IS NOT THE INVERSE OF A SPLIT
 * =============================================================================
 * Deliberately. Merging rows (2, 3) of a column flattens joint 2; splitting that
 * plate again gives back two squares and a joint that is STILL 0°, not the angle
 * it had before the merge. The same holds for a horizontal merge: the coerced
 * column does not remember the profile it gave up. That is the honest consequence
 * of making a pair of cells rigid — the geometry really did change — so it is
 * surfaced rather than hidden, and the store's UNDO stack (not a re-split) is the
 * way back to the previous shape.
 */

import {
  PITCH_MATCH_EPSILON_DEG,
  columnChain,
  normalizeConfig,
  validateConfig,
} from './schema.js'

/** The two cells a rect entry covers, as `[row, col]` pairs. */
function rectCells(rect) {
  if (!rect || typeof rect !== 'object') return []
  if (rect.orientation === 'horizontal') {
    return [
      [rect.row, rect.col],
      [rect.row, rect.col + 1],
    ]
  }
  if (rect.orientation === 'vertical') {
    return [
      [rect.row, rect.col],
      [rect.row + 1, rect.col],
    ]
  }
  return []
}

/** "r,c" → rect index, for every cell any plate already covers. */
function platedCells(cfg) {
  const map = new Map()
  const rects = Array.isArray(cfg.rects) ? cfg.rects : []
  rects.forEach((rect, i) => {
    for (const [r, c] of rectCells(rect)) map.set(`${r},${c}`, i)
  })
  return map
}

const isCell = (cell) =>
  cell !== null &&
  typeof cell === 'object' &&
  Number.isInteger(cell.row) &&
  Number.isInteger(cell.col)

const fmt = (cell) => `(${cell.row}, ${cell.col})`

/** Trim float noise off an angle for the human-readable descriptions. */
const deg = (v) => `${Math.round(Number(v) * 10) / 10}°`

function reject(code, message) {
  return { ok: false, code, message }
}

/**
 * Merge two adjacent squares into one 60×121 plate, applying the geometric
 * consequence of making them rigid.
 *
 * `cellA` is the CLICKED-FIRST cell: for a horizontal merge its column's profile
 * is the one that survives, and the other column is matched to it.
 *
 * Never mutates its input. The returned config is a fresh object that has passed
 * `validateConfig`; if it somehow would not, the merge is reported as a failure
 * rather than handed back broken.
 *
 * @param {object} config a config (raw or normalized — normalized internally)
 * @param {{row:number, col:number}} cellA the cell clicked first
 * @param {{row:number, col:number}} cellB its orthogonal neighbour
 * @returns {{ok:true, config:object, rect:object, orientation:string,
 *            kind:'free'|'coerce', changes:Array<object>, description:string}
 *          | {ok:false, code:string, message:string, errors?:Array}}
 */
export function mergeCells(config, cellA, cellB) {
  const cfg = normalizeConfig(config)
  const cols = Number(cfg.grid?.cols)
  const rows = Number(cfg.grid?.rows)
  if (!Number.isInteger(cols) || !Number.isInteger(rows) || cols < 1 || rows < 1) {
    return reject('E_MERGE_SHAPE', `config grid is not a usable ${cols}×${rows} lattice`)
  }

  if (!isCell(cellA) || !isCell(cellB)) {
    return reject(
      'E_MERGE_SHAPE',
      'a merge needs two cells, each { row, col } with integer coordinates',
    )
  }

  const inBounds = (cell) =>
    cell.row >= 0 && cell.row <= rows - 1 && cell.col >= 0 && cell.col <= cols - 1
  for (const cell of [cellA, cellB]) {
    if (!inBounds(cell)) {
      return reject(
        'E_MERGE_BOUNDS',
        `cell ${fmt(cell)} is outside the ${cols}×${rows} grid (rows 0..${rows - 1}, cols 0..${cols - 1})`,
      )
    }
  }

  if (cellA.row === cellB.row && cellA.col === cellB.col) {
    return reject(
      'E_MERGE_SAME_CELL',
      `a plate is made of TWO cells — ${fmt(cellA)} was given twice`,
    )
  }

  const dr = Math.abs(cellA.row - cellB.row)
  const dc = Math.abs(cellA.col - cellB.col)
  if (dr + dc !== 1) {
    return reject(
      'E_MERGE_NOT_ADJACENT',
      `${fmt(cellA)} and ${fmt(cellB)} are ${dr > 0 && dc > 0 ? 'diagonal neighbours' : 'not neighbours'} — ` +
        'a 60×121 plate covers two cells that share an edge, side by side in a row or ' +
        'one behind the other in a column',
    )
  }

  const plated = platedCells(cfg)
  for (const cell of [cellA, cellB]) {
    const i = plated.get(`${cell.row},${cell.col}`)
    if (i !== undefined) {
      return reject(
        'E_MERGE_OCCUPIED',
        `cell ${fmt(cell)} already belongs to rects[${i}] — a plate is exactly two cells, so ` +
          'split that plate first (click it) rather than growing it to three',
      )
    }
  }

  const orientation = dc === 1 ? 'horizontal' : 'vertical'
  const row = Math.min(cellA.row, cellB.row)
  const col = Math.min(cellA.col, cellB.col)
  const rect = { row, col, orientation }

  const next = structuredClone(cfg)
  const changes = []
  let consequence = ''

  if (orientation === 'vertical') {
    const folds = next.columns?.[col]?.foldsDeg
    if (!Array.isArray(folds) || row >= folds.length) {
      return reject(
        'E_MERGE_SHAPE',
        `column ${col} has no joint ${row} to remove — its foldsDeg is not the expected ${rows - 1} entries`,
      )
    }
    const before = Number(folds[row])
    if (before !== 0) {
      folds[row] = 0
      changes.push({ kind: 'flatten-joint', col, joint: row, fromDeg: before, toDeg: 0 })
      consequence = `flattened joint ${row} of column ${col} (${deg(before)} → 0°)`
    } else {
      consequence = `joint ${row} of column ${col} was already flat`
    }
  } else {
    // Horizontal: the clicked-first column's profile in FRONT of this row wins.
    const from = cellA.col
    const to = cellB.col
    const src = next.columns?.[from]
    const dst = next.columns?.[to]
    if (!src || !dst || !Array.isArray(src.foldsDeg) || !Array.isArray(dst.foldsDeg)) {
      return reject(
        'E_MERGE_SHAPE',
        `columns ${from} and ${to} are not both fold strips with a foldsDeg array`,
      )
    }
    const A = columnChain(cfg, col)
    const B = columnChain(cfg, col + 1)
    const pitchMatches =
      Math.abs(Number(A.pitchesDeg[row]) - Number(B.pitchesDeg[row])) <= PITCH_MATCH_EPSILON_DEG
    const oA = A.origins[row]
    const oB = B.origins[row]
    const tol = Number(cfg.gapTolerance)
    const positionMatches =
      Array.isArray(oA) &&
      Array.isArray(oB) &&
      Number.isFinite(tol) &&
      Math.hypot(oA[0] - oB[0], oA[1] - oB[1]) <= tol

    if (pitchMatches && positionMatches) {
      consequence = `columns ${col} and ${col + 1} already agree at row ${row}`
    } else {
      const startPitchDeg = Number(src.startPitchDeg)
      const foldsDeg = src.foldsDeg.slice(0, row).map(Number)
      dst.startPitchDeg = startPitchDeg
      for (let k = 0; k < row; k++) dst.foldsDeg[k] = foldsDeg[k]
      changes.push({
        kind: 'match-columns',
        from,
        to,
        throughRow: row,
        startPitchDeg,
        foldsDeg,
      })
      consequence =
        `matched column ${to} to column ${from} through row ${row} ` +
        `(its front pitch${row > 0 ? ` and joints 0..${row - 1}` : ''})`
    }
  }

  if (!Array.isArray(next.rects)) next.rects = []
  next.rects.push(rect)

  const result = validateConfig(next)
  if (!result.ok) {
    return {
      ok: false,
      code: 'E_MERGE_REJECTED',
      message:
        `merging ${fmt(cellA)} and ${fmt(cellB)} into a ${orientation} plate would leave an ` +
        `invalid design: ${result.errors.map((e) => `${e.code} ${e.message}`).join(' · ')}`,
      errors: result.errors,
    }
  }

  const kind = changes.length === 0 ? 'free' : 'coerce'
  const description =
    `merged ${fmt(cellA)}+${fmt(cellB)} into a ${orientation} ` +
    `${cfg.cell.rectLength}cm plate — ${consequence}`

  return { ok: true, config: next, rect, orientation, kind, changes, description }
}

/**
 * The four orthogonal neighbours `cell` could be merged with, classified.
 *
 * A candidate is reported ONLY if `mergeCells` actually succeeds for it, so the
 * UI can never highlight a merge the model would then refuse. Neighbours that are
 * out of bounds, already part of a plate, or whose coercion would break another
 * plate's constraint are simply absent.
 *
 * `kind` is 'free' when the merge changes nothing but the plate list, and 'coerce'
 * when it must also adjust geometry — `changes` says exactly what, and
 * `description` says it in words for a tooltip.
 *
 * Order is fixed (row−1, row+1, col−1, col+1) so the result is deterministic.
 *
 * @param {object} config a config (raw or normalized)
 * @param {{row:number, col:number}} cell the armed cell
 * @returns {Array<{row:number, col:number, orientation:'horizontal'|'vertical',
 *                  rect:{row:number,col:number,orientation:string},
 *                  kind:'free'|'coerce', changes:Array<object>, description:string}>}
 */
export function mergeCandidates(config, cell) {
  if (!isCell(cell)) return []
  const cfg = normalizeConfig(config)
  const neighbours = [
    { row: cell.row - 1, col: cell.col },
    { row: cell.row + 1, col: cell.col },
    { row: cell.row, col: cell.col - 1 },
    { row: cell.row, col: cell.col + 1 },
  ]
  const out = []
  for (const nb of neighbours) {
    const result = mergeCells(cfg, cell, nb)
    if (!result.ok) continue
    out.push({
      row: nb.row,
      col: nb.col,
      orientation: result.orientation,
      rect: result.rect,
      kind: result.kind,
      changes: result.changes,
      description: result.description,
    })
  }
  return out
}
