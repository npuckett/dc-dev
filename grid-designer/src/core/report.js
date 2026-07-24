/**
 * grid-designer — joint-consistency report.
 *
 * HEADLESS ZONE (src/core/): pure functions, importable from plain node.
 *   - explicit `.js` extensions on ALL relative imports
 *   - may import `three` math classes only; never components / store / DOM
 *
 * Measurement technique follows panel-designer/src/geometry/analyze.js (world
 * edge endpoints; signed dihedral from the cross product of the two face
 * normals dotted with the hinge axis) — reimplemented against `solveLayout`'s
 * output rather than imported.
 *
 * =============================================================================
 * WHY THIS EXISTS
 * =============================================================================
 * The surface is deliberately NOT solved as rigid origami: rigid plates sharing
 * vertices cannot fold in both families at once, but the real hardware has a
 * ~2cm gap at every joint bridged by a printed connector. `placement.js` places
 * rows as rigid pitched planes with an exact in-row accordion, which leaves all
 * of the slack on the row-to-row joints. This module measures that slack and
 * flags joints a connector could not physically span.
 *
 * =============================================================================
 * WHAT COUNTS AS A JOINT
 * =============================================================================
 * A LOGICAL joint is the shared boundary between two adjacent OCCUPIED cells:
 *   - 'in-row'   : (r, c) and (r, c+1)
 *   - 'row-pair' : (r, c) and (r+1, c)
 * Boundaries interior to a single panel are NOT joints — a horizontal rect's
 * removed middle joint and a vertical rect's spanned row boundary both fall out
 * automatically, because both cells map to the same owning panel.
 *
 * =============================================================================
 * HOW THE TWO FACING EDGES ARE FOUND
 * =============================================================================
 * Not by nearest-edge search (which mis-picks under strong folds) and not by
 * whole-panel edges (a rect's full edge is twice as long as its neighbour's, so
 * midpoints would be offset by half a cell). Instead each (panel, cell) pair
 * owns an axis-aligned sub-rectangle of the panel's lit face
 * (`panelCellFaceRectLocal`), and each panel publishes how its local axes map to
 * the grid directions (`panelLocalAxes`, which encodes the horizontal rect's
 * +90° yaw). The joint's two edges are then exactly:
 *   in-row   : the +col side of the left cell's sub-rect, and the −col side of
 *              the right cell's sub-rect
 *   row-pair : the +row side of the near cell's sub-rect, and the −row side of
 *              the far cell's sub-rect
 * Endpoints of both edges are ordered along a shared JOINT AXIS, chosen so a
 * positive configured fold reads as a positive dihedral:
 *   in-row   → the +row (depth) direction
 *   row-pair → the −col direction
 * so an ideal joint's `dihedralDeg` reproduces `foldAngleDeg(cfg, r, j)` for
 * in-row joints and `rowFoldsDeg[r]` for row-pair joints.
 *
 * =============================================================================
 * METRICS AND FLAGS
 * =============================================================================
 *   gapMid      distance between the two edge midpoints
 *   gapMin/Max  min/max of the two proximity-paired endpoint-to-endpoint
 *               distances (pairing chosen to minimize total distance)
 *   skewDeg     angle between the two edge directions, folded into [0, 90]
 *   dihedralDeg signed angle between the two lit-face normals about the joint
 *               axis (sign from (nA × nB) · jointAxis, as in analyze.js)
 *
 *   GAP_OUT_OF_TOL  gapMid or either endpoint distance outside
 *                   [gap − gapTolerance, gap + gapTolerance]
 *   SKEW            skewDeg > SKEW_TOLERANCE_DEG (5°)
 * `flags` may carry both; an empty `flags` array means `ok: true`.
 *
 * =============================================================================
 * EXPECTED PHYSICS (what a healthy report looks like)
 * =============================================================================
 *   - Every in-row joint between two panels belonging to the SAME row's chain is
 *     EXACT by construction: gapMid = gapMin = gapMax = gap, skew 0. The
 *     bisector advance in `placement.js` guarantees it. (The one exception is a
 *     vertical rect's lower cell, whose in-row neighbours live in the next row's
 *     plane — see below.)
 *   - Row-pair joints are exact when the two adjacent rows have identical
 *     zig-zag profiles (including a flat pair), and deviate as the profiles
 *     diverge — that is where all of the surface's slack is deliberately parked.
 *   - A vertical rect's far-side joints (its lower cell against row r+1's
 *     in-row neighbours, and against row r+2 across the next row boundary)
 *     deviate as soon as the spanned rowFold ≠ 0: the plate is rigid and stays
 *     in row r's plane while its neighbours pitch away. Validation already warns
 *     W_CROSSROW_FOLD; this report quantifies it.
 *   - The 121 vs 2·60 + 2 = 122 hardware slack shows up as a ≈1cm deviation on
 *     the far side of every rect. Accepted, surfaced, never corrected.
 */

import * as THREE from 'three'
import { foldAngleDeg, normalizeConfig } from './schema.js'
import {
  cellEdgeWorld,
  negAxis,
  panelLocalAxes,
  panelWorldNormal,
  round9,
} from './placement.js'

const RAD = 180 / Math.PI

/** Joints whose facing edges are more than this far from parallel are flagged. */
export const SKEW_TOLERANCE_DEG = 5

// -----------------------------------------------------------------------------
// jointReport
// -----------------------------------------------------------------------------
/**
 * Measure every logical joint of a solved layout.
 *
 * @param {object} layout output of `solveLayout`
 * @param {object} config the config that produced it (raw or normalized)
 * @returns {{ joints: Array, summary: object }}
 */
export function jointReport(layout, config) {
  const cfg = normalizeConfig(config)
  const cols = Number(cfg.grid.cols)
  const rowCount = Number(cfg.grid.rows)
  const gap = Number(cfg.gap)
  const tol = Number(cfg.gapTolerance)

  // cell → owning panel + which of that panel's cells it is
  const owner = new Map()
  for (const panel of layout.panels) {
    panel.cells.forEach((cell, cellIndex) => {
      owner.set(`${cell[0]},${cell[1]}`, { panel, cellIndex })
    })
  }
  const at = (r, c) => owner.get(`${r},${c}`)

  const joints = []

  // --- in-row joints ------------------------------------------------------
  for (let r = 0; r < rowCount; r++) {
    for (let j = 0; j + 1 < cols; j++) {
      const A = at(r, j)
      const B = at(r, j + 1)
      if (!A || !B) continue
      if (A.panel === B.panel) continue // interior to a horizontal rect

      const axA = panelLocalAxes(A.panel)
      const axB = panelLocalAxes(B.panel)
      const edgeA = cellEdgeWorld(A.panel, A.cellIndex, axA.col, axA.row, cfg)
      const edgeB = cellEdgeWorld(B.panel, B.cellIndex, negAxis(axB.col), axB.row, cfg)

      joints.push(
        measureJoint({
          id: `in-row:r${r}:j${j}`,
          klass: 'in-row',
          row: r,
          col: j,
          cellA: [r, j],
          cellB: [r, j + 1],
          A,
          B,
          edgeA,
          edgeB,
          expectedFoldDeg: foldAngleDeg(cfg, r, j),
          gap,
          tol,
        }),
      )
    }
  }

  // --- row-pair joints ----------------------------------------------------
  for (let r = 0; r + 1 < rowCount; r++) {
    for (let c = 0; c < cols; c++) {
      const A = at(r, c)
      const B = at(r + 1, c)
      if (!A || !B) continue
      if (A.panel === B.panel) continue // interior to a vertical rect

      const axA = panelLocalAxes(A.panel)
      const axB = panelLocalAxes(B.panel)
      const edgeA = cellEdgeWorld(A.panel, A.cellIndex, axA.row, negAxis(axA.col), cfg)
      const edgeB = cellEdgeWorld(B.panel, B.cellIndex, negAxis(axB.row), negAxis(axB.col), cfg)

      const fold = Array.isArray(cfg.rowFoldsDeg) ? Number(cfg.rowFoldsDeg[r]) : 0

      joints.push(
        measureJoint({
          id: `row-pair:r${r}:c${c}`,
          klass: 'row-pair',
          row: r,
          col: c,
          cellA: [r, c],
          cellB: [r + 1, c],
          A,
          B,
          edgeA,
          edgeB,
          expectedFoldDeg: Number.isFinite(fold) ? fold : 0,
          gap,
          tol,
        }),
      )
    }
  }

  // --- summary ------------------------------------------------------------
  let okCount = 0
  let worstGapDeviation = 0
  let worstSkew = 0
  for (const j of joints) {
    if (j.ok) okCount++
    const dev = Math.max(
      Math.abs(j.gapMid - gap),
      Math.abs(j.gapMin - gap),
      Math.abs(j.gapMax - gap),
    )
    if (dev > worstGapDeviation) worstGapDeviation = dev
    if (j.skewDeg > worstSkew) worstSkew = j.skewDeg
  }

  return {
    joints,
    summary: {
      total: joints.length,
      ok: okCount,
      flagged: joints.length - okCount,
      worstGapDeviation: round9(worstGapDeviation),
      worstSkew: round9(worstSkew),
      gap: round9(gap),
      gapTolerance: round9(tol),
      skewToleranceDeg: SKEW_TOLERANCE_DEG,
    },
  }
}

/**
 * Measure one joint from its two world-space facing edges.
 * @returns {object} joint record
 */
function measureJoint(spec) {
  const { edgeA, edgeB, gap, tol } = spec
  const [a0, a1] = edgeA
  const [b0, b1] = edgeB

  const midA = a0.clone().add(a1).multiplyScalar(0.5)
  const midB = b0.clone().add(b1).multiplyScalar(0.5)
  const gapMid = midA.distanceTo(midB)

  // Proximity pairing of the endpoints. Both edges are already oriented along
  // the same joint axis, so the straight pairing wins in every non-degenerate
  // case; picking the cheaper pairing keeps the metric honest if it doesn't.
  const straight = a0.distanceTo(b0) + a1.distanceTo(b1)
  const swapped = a0.distanceTo(b1) + a1.distanceTo(b0)
  const d0 = straight <= swapped ? a0.distanceTo(b0) : a0.distanceTo(b1)
  const d1 = straight <= swapped ? a1.distanceTo(b1) : a1.distanceTo(b0)
  const gapMin = Math.min(d0, d1)
  const gapMax = Math.max(d0, d1)

  const dirA = a1.clone().sub(a0).normalize()
  const dirB = b1.clone().sub(b0).normalize()
  const skewDeg = Math.acos(clamp01(Math.abs(dirA.dot(dirB)))) * RAD

  // Signed dihedral about the joint axis (= dirA), analyze.js's approach.
  const nA = panelWorldNormal(spec.A.panel)
  const nB = panelWorldNormal(spec.B.panel)
  const cross = new THREE.Vector3().crossVectors(nA, nB)
  const sinSign = cross.dot(dirA)
  const unsigned = Math.acos(clampPM1(nA.dot(nB))) * RAD
  const dihedralDeg = sinSign >= 0 ? unsigned : -unsigned

  const lo = gap - tol
  const hi = gap + tol
  const outOfTol = (v) => v < lo || v > hi

  const flags = []
  if (outOfTol(gapMid) || outOfTol(d0) || outOfTol(d1)) flags.push('GAP_OUT_OF_TOL')
  if (skewDeg > SKEW_TOLERANCE_DEG) flags.push('SKEW')

  return {
    id: spec.id,
    class: spec.klass,
    row: spec.row,
    col: spec.col,
    cellA: spec.cellA,
    cellB: spec.cellB,
    panelA: spec.A.panel.id,
    panelB: spec.B.panel.id,
    expectedGap: round9(gap),
    expectedFoldDeg: round9(spec.expectedFoldDeg),
    gapMid: round9(gapMid),
    gapMin: round9(gapMin),
    gapMax: round9(gapMax),
    skewDeg: round9(skewDeg),
    dihedralDeg: round9(dihedralDeg),
    flags,
    ok: flags.length === 0,
  }
}

const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v)
const clampPM1 = (v) => (v < -1 ? -1 : v > 1 ? 1 : v)

// -----------------------------------------------------------------------------
// formatReport
// -----------------------------------------------------------------------------
/**
 * Render a `jointReport` result as a readable fixed-width table.
 *
 * @param {{joints: Array, summary: object}} reportResult
 * @returns {string}
 */
export function formatReport(reportResult) {
  const { joints, summary } = reportResult
  const lines = []
  lines.push('=== JOINT CONSISTENCY REPORT ===')
  lines.push(
    `gap ${fx(summary.gap, 2)}cm  tolerance ±${fx(summary.gapTolerance, 2)}cm  ` +
      `skew limit ${summary.skewToleranceDeg}°`,
  )
  lines.push('')
  lines.push(
    pad('CLASS', 9) +
      pad('JOINT', 18) +
      pad('PANELS', 16) +
      rpad('FOLD', 8) +
      rpad('gapMid', 9) +
      rpad('gapMin', 9) +
      rpad('gapMax', 9) +
      rpad('skew', 8) +
      rpad('dihed', 9) +
      '  FLAGS',
  )
  lines.push('-'.repeat(104))

  for (const j of joints) {
    lines.push(
      pad(j.class, 9) +
        pad(`(${j.cellA}) (${j.cellB})`, 18) +
        pad(`${j.panelA}→${j.panelB}`, 16) +
        rpad(`${fx(j.expectedFoldDeg, 1)}°`, 8) +
        rpad(fx(j.gapMid, 3), 9) +
        rpad(fx(j.gapMin, 3), 9) +
        rpad(fx(j.gapMax, 3), 9) +
        rpad(`${fx(j.skewDeg, 1)}°`, 8) +
        rpad(`${fx(j.dihedralDeg, 1)}°`, 9) +
        '  ' +
        (j.ok ? 'OK' : j.flags.join('+')),
    )
  }

  lines.push('-'.repeat(104))
  lines.push(
    `total ${summary.total}   ok ${summary.ok}   flagged ${summary.flagged}   ` +
      `worst gap deviation ${fx(summary.worstGapDeviation, 3)}cm   ` +
      `worst skew ${fx(summary.worstSkew, 2)}°`,
  )
  lines.push(
    summary.flagged === 0
      ? '=== ALL JOINTS WITHIN TOLERANCE ==='
      : `=== ${summary.flagged} JOINT(S) FLAGGED ===`,
  )
  return lines.join('\n')
}

const pad = (s, n) => String(s).padEnd(n)
const rpad = (s, n) => String(s).padStart(n - 1) + ' '
const fx = (v, d) => (Number.isFinite(v) ? v.toFixed(d) : String(v))
