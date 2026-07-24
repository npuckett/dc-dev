/**
 * grid-designer — the auto-ground solver: bring a column's last panel back down
 * to the floor by solving its last hinge.
 *
 * HEADLESS ZONE (src/core/): pure functions, importable from plain node.
 *   - explicit `.js` extensions on ALL relative imports
 *   - may import `three` math classes only; never components / store / DOM
 *   - same input → same output, no hidden state; NOTHING here mutates its input
 *
 * =============================================================================
 * THE RULE BEING ENFORCED
 * =============================================================================
 * "The wave returns to the water": the LAST panel of every column must touch the
 * floor — lying flat, or tilted with one of its housing edges down. It may never
 * float. The measurement lives in placement.js (`columnEndGrounding`, over the
 * panel SOLID's 8 corners) and `solveLayout` reports failures as E_END_FLOATING
 * violations. This module is the mechanical fix for one column.
 *
 * =============================================================================
 * WHICH HINGE GETS SOLVED
 * =============================================================================
 * The LAST SURVIVING hinge of the column — the deepest joint that still exists.
 * A vertical plate spanning the last two rows REMOVES the joint between them
 * (that plate is one rigid 121cm panel), so for such a column the solved hinge is
 * the one in FRONT of the plate and the panel being grounded is the plate itself.
 * A column with no surviving hinge at all is a single rigid run: there is nothing
 * to solve and the solver returns null.
 *
 * =============================================================================
 * METHOD
 * =============================================================================
 * min_y(f) — the lowest point of the end panel's solid as a function of that
 * hinge's angle — is continuous and cheap (one column chain walk, 8 points), so:
 *
 *   1. SCAN f over [−120, 120] in 1° steps, evaluating min_y.
 *   2. BISECT every interval that brackets the target clearance (the middle of
 *      the acceptance band) to ~1e-9°, collecting every root.
 *   3. Keep the roots (and any scan sample) whose clearance lands in the band,
 *      and pick the one CLOSEST TO THE CURRENT ANGLE — grounding a column should
 *      move it as little as possible.
 *   4. SNAP the winner to the coarsest of 1° / 0.5° / 0.25° / 0.1° / 0.01° that
 *      still lands in the band, so a solved design stays readable as JSON.
 *
 * Preference inside the band goes to solutions at or above the floor (clearance
 * ≥ 0): the panel resting ON the ground rather than 2mm into it also keeps the
 * lit face clear of y = 0, i.e. never trades E_END_FLOATING for W_BELOW_FLOOR.
 *
 * UNREACHABLE ENDS. A 60cm panel can drop its far edge at most ~60cm, so once a
 * column's chain ends much above that the hinge cannot reach the floor at either
 * extreme and the solver returns null — the fold profile itself has to arc back
 * down (see the grounded-end rule in schema.js). That is a real design failure,
 * not a solver limitation.
 */

import {
  MAX_FOLD_DEG,
  columnFoldDeg,
  columnSegments,
  normalizeConfig,
  validateConfig,
} from './schema.js'
import { columnEndGrounding, groundToleranceOf, round9 } from './placement.js'

/**
 * How far INTO the floor a solved end may sit, cm. Small and negative so the
 * solver can always find a crossing, while `preferAtOrAboveFloor` still favours
 * clearance ≥ 0.
 */
export const GROUND_UNDERSHOOT_CM = 0.25

/** Scan step for the initial bracketing sweep, degrees. */
const SCAN_STEP_DEG = 1
/** Angles the snapping pass tries, coarsest first. */
const SNAP_STEPS_DEG = [1, 0.5, 0.25, 0.1, 0.01]

// -----------------------------------------------------------------------------
// Joints
// -----------------------------------------------------------------------------
/**
 * The joint indices of column `c` that actually exist, front to back.
 *
 * Derived from `columnSegments` (not from `foldsDeg.length`), so the joints a
 * vertical plate removes are absent by construction: a segment ending at row k
 * carries the joint k, and a 121cm plate covering rows k, k+1 ends at k+1 —
 * joint k is interior to it and never appears.
 *
 * @param {object} config config (raw or normalized)
 * @param {number} c column index
 * @returns {number[]}
 */
export function survivingJoints(config, c) {
  const cfg = normalizeConfig(config)
  const rowCount = Number(cfg.grid.rows)
  const joints = []
  for (const seg of columnSegments(cfg, c)) {
    const k = seg.rows[seg.rows.length - 1]
    if (k <= rowCount - 2) joints.push(k)
  }
  return joints
}

/**
 * The deepest joint of column `c` that still exists, or −1 for a rigid run.
 *
 * @param {object} config config (raw or normalized)
 * @param {number} c column index
 * @returns {number}
 */
export function lastSurvivingJoint(config, c) {
  const joints = survivingJoints(config, c)
  return joints.length > 0 ? joints[joints.length - 1] : -1
}

// -----------------------------------------------------------------------------
// solveGroundingFold
// -----------------------------------------------------------------------------
/** A copy of `config` with one hinge changed. Never mutates the input. */
function withFold(cfg, c, k, deg) {
  return {
    ...cfg,
    columns: cfg.columns.map((column, i) =>
      i === c && Array.isArray(column?.foldsDeg)
        ? { ...column, foldsDeg: column.foldsDeg.map((f, j) => (j === k ? deg : f)) }
        : column,
    ),
  }
}

/**
 * Solve column `c`'s last surviving hinge for floor contact.
 *
 * @param {object} config config (raw or normalized). NOT mutated.
 * @param {number} c column index
 * @returns {{ jointIndex:number, foldDeg:number, clearanceCm:number } | null}
 *          null when the column has no hinge to solve, or when no angle in
 *          [−120, 120] brings the end panel down to the floor.
 */
export function solveGroundingFold(config, c) {
  const cfg = normalizeConfig(config)
  if (!Array.isArray(cfg.columns) || !cfg.columns[c] || !Array.isArray(cfg.columns[c].foldsDeg)) {
    return null
  }
  const k = lastSurvivingJoint(cfg, c)
  if (k < 0 || k >= cfg.columns[c].foldsDeg.length) return null

  const tol = groundToleranceOf(cfg)
  const lo = -GROUND_UNDERSHOOT_CM
  const hi = tol
  const target = (lo + hi) / 2
  const inBand = (v) => v >= lo && v <= hi
  const current = columnFoldDeg(cfg, c, k)
  const clearanceAt = (deg) => columnEndGrounding(withFold(cfg, c, k, deg), c).clearanceCm

  // --- 1. scan ------------------------------------------------------------
  const samples = []
  for (let deg = -MAX_FOLD_DEG; deg <= MAX_FOLD_DEG; deg += SCAN_STEP_DEG) {
    samples.push({ deg, y: clearanceAt(deg) })
  }

  // --- 2. bisect every bracketing interval --------------------------------
  const candidates = []
  for (const s of samples) if (inBand(s.y)) candidates.push(s.deg)
  for (let i = 0; i + 1 < samples.length; i++) {
    const a = samples[i]
    const b = samples[i + 1]
    if ((a.y - target) * (b.y - target) > 0) continue
    let x0 = a.deg
    let x1 = b.deg
    let y0 = a.y
    for (let it = 0; it < 60 && x1 - x0 > 1e-9; it++) {
      const xm = (x0 + x1) / 2
      const ym = clearanceAt(xm)
      if ((y0 - target) * (ym - target) <= 0) {
        x1 = xm
      } else {
        x0 = xm
        y0 = ym
      }
    }
    const root = (x0 + x1) / 2
    if (inBand(clearanceAt(root))) candidates.push(root)
  }
  if (candidates.length === 0) return null

  // --- 3. closest to where the hinge already sits -------------------------
  candidates.sort((a, b) => Math.abs(a - current) - Math.abs(b - current) || a - b)
  const best = candidates[0]

  // --- 4. snap to a readable angle ----------------------------------------
  const snapped = snapFold(best, clearanceAt, inBand)
  const foldDeg = round9(snapped)
  if (Math.abs(foldDeg) > MAX_FOLD_DEG) return null
  return { jointIndex: k, foldDeg, clearanceCm: round9(clearanceAt(foldDeg)) }
}

/**
 * Round a solved angle to the coarsest step that still grounds the column,
 * preferring a result at or above the floor (clearance ≥ 0) over one that digs
 * a fraction of a millimetre into it.
 */
function snapFold(deg, clearanceAt, inBand) {
  let fallback = null
  for (const step of SNAP_STEPS_DEG) {
    const candidate = Math.round(deg / step) * step
    if (Math.abs(candidate) > MAX_FOLD_DEG) continue
    const y = clearanceAt(candidate)
    if (!inBand(y)) continue
    if (y >= 0) return candidate
    if (fallback === null) fallback = candidate
  }
  return fallback === null ? deg : fallback
}

// -----------------------------------------------------------------------------
// groundAllFolds
// -----------------------------------------------------------------------------
/**
 * Ground every FLOATING column of a config, leaving grounded ones untouched.
 *
 * Applied column by column, and a solved hinge is kept only if the config still
 * validates with it — the last hinge of a column also sets the cumulative pitch
 * of its last row, which a horizontal plate landing there may have pinned to a
 * neighbouring column (E_CROSSCOL_ANGLE_MISMATCH). Such a column is left alone
 * (and stays in `violations`) rather than breaking the whole config.
 *
 * `includeSunken` also re-solves columns whose end panel is driven well INTO the
 * floor. Those already satisfy the grounded-end rule (they certainly touch), so
 * the UI's "Ground all" leaves them alone — but a generator wants the panel
 * RESTING on the floor rather than buried in it, so core/presets.js opts in.
 *
 * @param {object} config config (raw or normalized). NOT mutated.
 * @param {{includeSunken?: boolean}} [options]
 * @returns {object|null} a new config, or null when nothing changed (every column
 *          already settled, or no unsettled column could be solved)
 */
export function groundAllFolds(config, options = {}) {
  const includeSunken = options.includeSunken === true
  let cfg = normalizeConfig(config)
  const cols = Number(cfg.grid?.cols)
  if (!Number.isInteger(cols) || cols < 1) return null

  let changed = false
  for (let c = 0; c < cols; c++) {
    const { grounded, clearanceCm } = columnEndGrounding(cfg, c)
    const sunken = clearanceCm < -GROUND_UNDERSHOOT_CM
    if (grounded && !(includeSunken && sunken)) continue
    const solved = solveGroundingFold(cfg, c)
    if (!solved) continue
    const candidate = withFold(cfg, c, solved.jointIndex, solved.foldDeg)
    if (!validateConfig(candidate).ok) continue
    cfg = candidate
    changed = true
  }

  return changed ? cfg : null
}
