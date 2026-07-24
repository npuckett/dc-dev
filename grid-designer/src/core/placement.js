/**
 * grid-designer — deterministic placement solver.
 *
 * HEADLESS ZONE (src/core/): pure functions, importable from plain node.
 *   - explicit `.js` extensions on ALL relative imports
 *   - may import `three` math classes only; never components / store / DOM
 *   - same config in → identical output out (deep-equal across calls)
 *
 * =============================================================================
 * MODEL — "pitched row planes × in-row accordion"
 * =============================================================================
 * The surface is NOT solved as rigid origami. Every joint is spanned by a ~2cm
 * printed connector, so panels never share vertices; we place each row as a
 * rigid pitched plane, run a 2D accordion chain inside that plane, and let
 * `report.js` measure whatever slack lands on the row-to-row joints.
 *
 * WORLD CONVENTIONS
 *   Units cm. Y up. Columns along +X (c = 0..cols-1). Rows recede along +Z
 *   (r = 0..rows-1); the shore/window is at −Z and row 0's near edge sits on
 *   the world X axis at z = 0. Everything is lifted by
 *   PANEL_PROFILE.overallThickness (3.7) in +Y so the housings rest on the
 *   floor — a flat panel's lit face is at y = 3.7.
 *
 * PANEL LOCAL FRAME (matches geometry/panelGeometry.js)
 *   Lit face lies in the local XZ plane at local y = 0, centered on the origin.
 *   Panel *width* runs along local X, panel *depth/length* along local Z, and
 *   the lit face looks along +Y.
 *     '2x2' → width = cell.size,  length = cell.size
 *     '2x4' → width = cell.size,  length = cell.rectLength
 *
 * ROW-LOCAL FRAME
 *   X_row runs along the row, Y_row points out of the row plane (the lit side),
 *   Z_row is depth (toward the next row). (X_row, Y_row, Z_row) is
 *   right-handed and maps to (world X, n_r, d_r) under the row-plane transform.
 *
 * -----------------------------------------------------------------------------
 * STAGE A — row chain (2D accordion, per row, in the X_row–Y_row plane)
 * -----------------------------------------------------------------------------
 *   1. p = (0, 0), heading α = 0°. Positive α lifts toward +Y_row.
 *   2. Walk columns left to right. A chain segment is either
 *        - a square, L = cell.size, consuming 1 column, or
 *        - a horizontal rect, L = cell.rectLength, consuming 2 columns and
 *          skipping the joint between them (the plate is continuous there).
 *   3. Segment at heading α: row-local center = p + (L/2)·(cosα, sinα) in
 *      (X_row, Y_row); Z_row center = cell.size/2 for squares and horizontal
 *      rects, cell.rectLength/2 for vertical rects (see below). The panel's
 *      row-local rotation is a right-handed rotation of +α about +Z_row, which
 *      lifts the panel's +X_local end toward +Y_row. Then p += L·(cosα, sinα).
 *   4. At each surviving joint j (the last column the segment consumed, when
 *      j ≤ cols-2 and no horizontal rect removed it):
 *        p += gap·(cos(α + φ/2), sin(α + φ/2))   ← gap along the BISECTOR
 *        α += φ                                   with φ = foldAngleDeg(cfg,r,j)
 *      Advancing along the bisector is what makes every in-row joint exact:
 *      the left panel's trailing edge and the right panel's leading edge end up
 *      exactly `gap` apart, edge-parallel, with zero skew.
 *   5. Per-cell tilt is asserted identical to schema.js `cellTilts(cfg, r)`.
 *   6. rowAnchor 'center' shifts the whole chain in X_row so its X extent (min
 *      to max over all chain vertices — which are exactly the panels' footprint
 *      endpoints) is centered on the nominal grid center
 *      X = (cols·(size+gap) − gap)/2. 'start' leaves the chain starting at 0.
 *      A flat row spans cols·size + (cols−1)·gap = the nominal full width, so a
 *      flat centered row lands exactly on the pitch lattice (30, 92, 154, …).
 *
 * -----------------------------------------------------------------------------
 * STAGE B — row planes (2D chain in the Y–Z plane, once per config)
 * -----------------------------------------------------------------------------
 *   ψ_0 = 0, ψ_{r+1} = ψ_r + rowFoldsDeg[r]  (each entry clamped to ±120°)
 *   depth  d_r = (0,  sinψ_r, cosψ_r)
 *   normal n_r = (0,  cosψ_r, −sinψ_r)
 *   O_0 = (0, 3.7, 0);  O_{r+1} = O_r + size·d_r + gap·bisector(d_r, d_{r+1})
 *   M_r maps (X_row, Y_row, Z_row) → X_row·(1,0,0) + Y_row·n_r + Z_row·d_r + O_r,
 *   which is exactly a rotation of −ψ_r about world +X. ψ = 90° ⇒ d = (0,1,0):
 *   the row stands straight up, so a positive rowFold pitches the next row UP.
 *
 * -----------------------------------------------------------------------------
 * RECTS
 * -----------------------------------------------------------------------------
 * HORIZONTAL rect at (r,c) covers (r,c)+(r,c+1) and removes in-row joint j = c.
 * The '2x4' geometry is 60 wide (local X) × 121 long (local Z), so it is yawed
 * +90° about local Y to run its length along the chain: local +Z → +X_row and
 * local +X → −Z_row. Composition is M_r ∘ chainRotation(α) ∘ yaw90.
 *
 * VERTICAL rect at (r,c) covers (r,c)+(r+1,c) as one rigid plate. It is placed
 * entirely in ROW r's plane at that cell's chain position and tilt, unyawed, so
 * its 121 length runs along d_r with its near edge at Z_row = 0 (hence a
 * Z_row center of cell.rectLength/2). It does NOT bend into row r+1's plane.
 * Row r+1's chain treats column c as occupied — it advances exactly as if a
 * square were there (so its other panels are unaffected) but emits no panel.
 * The mismatch between the plate and row r+1's neighbours is left for
 * `report.js` to measure; that is the whole point of the design.
 *
 * -----------------------------------------------------------------------------
 * OUTPUT
 * -----------------------------------------------------------------------------
 *   solveLayout(config) → {
 *     panels: [{ id, row, col, cells, type, rectOrientation,
 *                position: [x,y,z], quaternion: [x,y,z,w],
 *                tiltDeg, rowPitchDeg }],
 *     rowPlanes: [{ origin: [x,y,z], pitchDeg }],
 *     warnings: [{ code, message }],
 *   }
 * All numbers are plain JSON-able doubles rounded to 1e-9 (which also
 * normalizes −0 → 0) so repeated solves are byte-identical.
 */

import * as THREE from 'three'
import { PANEL_PROFILE } from '../config.js'
import {
  MAX_ROW_FOLD_DEG,
  clamp,
  cellTilts,
  foldAngleDeg,
  normalizeConfig,
  removedJoints,
} from './schema.js'

const DEG = Math.PI / 180

/** Row-local chain rotation axis (= Z_row / depth). */
const CHAIN_AXIS = new THREE.Vector3(0, 0, 1)
/** Yaw axis for horizontal rects (= panel local +Y, the lit direction). */
const YAW_AXIS = new THREE.Vector3(0, 1, 0)
/** Row-plane pitch axis (= world +X, along the rows). */
const PITCH_AXIS = new THREE.Vector3(1, 0, 0)

// -----------------------------------------------------------------------------
// Determinism helpers
// -----------------------------------------------------------------------------
/**
 * Round to 1e-9 and normalize −0 → 0.
 *
 * Trig gives values like 3.7000000000000004 and −0 depending on which branch
 * produced them; both break `deepStrictEqual` / `JSON.stringify` comparisons
 * without changing the geometry. Rounding here is the determinism guarantee.
 *
 * @param {number} v
 * @returns {number}
 */
export function round9(v) {
  if (!Number.isFinite(v)) return v
  const r = Math.round(v * 1e9) / 1e9
  return r === 0 ? 0 : r
}

const vec3out = (v) => [round9(v.x), round9(v.y), round9(v.z)]
const quatOut = (q) => [round9(q.x), round9(q.y), round9(q.z), round9(q.w)]

// -----------------------------------------------------------------------------
// Panel dimensions / local face geometry
// -----------------------------------------------------------------------------
/**
 * Lit-face footprint of a panel in its local frame: width along local X,
 * length along local Z. Driven by the config (not config.js's PANEL_DIMENSIONS)
 * so a config with non-default cell sizes stays self-consistent; at the default
 * 60 / 121 the two agree exactly.
 *
 * @param {object} panel a layout panel
 * @param {object} config normalized config
 * @returns {{ width: number, length: number }}
 */
export function panelFaceDims(panel, config) {
  const size = Number(config.cell.size)
  if (panel.type === '2x4') return { width: size, length: Number(config.cell.rectLength) }
  return { width: size, length: size }
}

/**
 * How a panel's local axes map onto the grid directions, given the placement
 * conventions above. `col` is the local signed axis pointing toward increasing
 * column; `row` points toward increasing row (deeper, away from the shore).
 *
 *   square          → +col = +X_local,  +row = +Z_local
 *   vertical rect   → +col = +X_local,  +row = +Z_local   (unyawed)
 *   horizontal rect → +col = +Z_local,  +row = −X_local   (yawed +90° about Y)
 *
 * @param {object} panel a layout panel
 * @returns {{ col: {axis:'x'|'z', sign:1|-1}, row: {axis:'x'|'z', sign:1|-1} }}
 */
export function panelLocalAxes(panel) {
  if (panel.rectOrientation === 'horizontal') {
    return { col: { axis: 'z', sign: 1 }, row: { axis: 'x', sign: -1 } }
  }
  return { col: { axis: 'x', sign: 1 }, row: { axis: 'z', sign: 1 } }
}

/** Flip a signed local axis. */
export function negAxis(a) {
  return { axis: a.axis, sign: a.sign > 0 ? -1 : 1 }
}

/**
 * The local-frame lit-face rectangle attributable to ONE cell of a panel.
 *
 * A square owns its whole face. A rect (either orientation) owns two cells laid
 * out along its local +Z length, and each cell is credited exactly `cell.size`
 * of that length, anchored at its own outer end:
 *
 *     cell 0 → local Z ∈ [−L/2, −L/2 + size]
 *     cell 1 → local Z ∈ [ L/2 − size,  L/2 ]
 *
 * With the real hardware (rectLength 121 vs 2·60 + 2 = 122) those two windows
 * overlap by 1cm in the middle — which is fine, because the middle boundary is
 * internal to the plate and never a joint. Anchoring at the outer ends (rather
 * than splitting 60.5/60.5) makes the plate's outer edges line up exactly with
 * the neighbouring cells' lattice positions, so the ~1cm of accepted slack
 * shows up once, in the report, instead of being smeared over every joint.
 *
 * @param {object} panel a layout panel
 * @param {number} cellIndex index into panel.cells
 * @param {object} config normalized config
 * @returns {{ x0:number, x1:number, z0:number, z1:number }} local, at face y = 0
 */
export function panelCellFaceRectLocal(panel, cellIndex, config) {
  const { width, length } = panelFaceDims(panel, config)
  const hw = width / 2
  const hl = length / 2
  if (panel.cells.length < 2) return { x0: -hw, x1: hw, z0: -hl, z1: hl }
  const size = Number(config.cell.size)
  const z = cellIndex === 0 ? [-hl, -hl + size] : [hl - size, hl]
  return { x0: -hw, x1: hw, z0: z[0], z1: z[1] }
}

/**
 * Read a panel's stored quaternion back as a THREE.Quaternion.
 *
 * The stored components are rounded to 1e-9 for determinism, which leaves the
 * quaternion very slightly non-unit; `applyQuaternion` assumes unit length and
 * would otherwise scale transformed points by |q|² (≈ 5e-10 relative, i.e. tens
 * of nanometres over a 121cm panel). Normalizing on read removes that.
 *
 * @param {object} panel a layout panel
 * @returns {THREE.Quaternion}
 */
function readQuat(panel) {
  return new THREE.Quaternion(
    panel.quaternion[0],
    panel.quaternion[1],
    panel.quaternion[2],
    panel.quaternion[3],
  ).normalize()
}

/**
 * Transform a point on the panel's local lit face (local y = 0) into world space.
 *
 * @param {object} panel a layout panel
 * @param {number} x local X
 * @param {number} z local Z
 * @returns {THREE.Vector3}
 */
export function localFaceToWorld(panel, x, z) {
  const v = new THREE.Vector3(x, 0, z).applyQuaternion(readQuat(panel))
  v.x += panel.position[0]
  v.y += panel.position[1]
  v.z += panel.position[2]
  return v
}

/** World-space lit-face normal (local +Y). */
export function panelWorldNormal(panel) {
  return new THREE.Vector3(0, 1, 0).applyQuaternion(readQuat(panel)).normalize()
}

/**
 * The 4 world-space corners of a panel's lit face.
 *
 * Corner order is the local-space order
 *   [0] (−X, −Z)   [1] (+X, −Z)   [2] (+X, +Z)   [3] (−X, +Z)
 * i.e. counter-clockwise when viewed from +Y (looking down onto the lit face).
 *
 * @param {object} panel a layout panel
 * @param {object} config config (raw or normalized)
 * @returns {number[][]} four [x, y, z] triples
 */
export function panelWorldCorners(panel, config) {
  const cfg = normalizeConfig(config)
  const { width, length } = panelFaceDims(panel, cfg)
  const hw = width / 2
  const hl = length / 2
  return [
    [-hw, -hl],
    [hw, -hl],
    [hw, hl],
    [-hw, hl],
  ].map(([x, z]) => vec3out(localFaceToWorld(panel, x, z)))
}

/**
 * The 4 world-space corners of the sub-rectangle one cell of a panel owns.
 * Same corner order as `panelWorldCorners`.
 *
 * @param {object} panel a layout panel
 * @param {number} cellIndex index into panel.cells
 * @param {object} config config (raw or normalized)
 * @returns {number[][]} four [x, y, z] triples
 */
export function panelCellWorldCorners(panel, cellIndex, config) {
  const cfg = normalizeConfig(config)
  const r = panelCellFaceRectLocal(panel, cellIndex, cfg)
  return [
    [r.x0, r.z0],
    [r.x1, r.z0],
    [r.x1, r.z1],
    [r.x0, r.z1],
  ].map(([x, z]) => vec3out(localFaceToWorld(panel, x, z)))
}

/**
 * One edge of a cell's lit-face sub-rectangle, in world space.
 *
 * `faceAxis` selects which side: the edge at the extreme of the sub-rectangle
 * along that signed local axis. `orientAxis` (perpendicular to `faceAxis`)
 * orders the two endpoints so the edge direction points along it — which lets
 * the report pair endpoints and sign dihedrals consistently.
 *
 * @param {object} panel a layout panel
 * @param {number} cellIndex index into panel.cells
 * @param {{axis:'x'|'z', sign:1|-1}} faceAxis side of the sub-rect to take
 * @param {{axis:'x'|'z', sign:1|-1}} orientAxis direction the edge should run
 * @param {object} config normalized config
 * @returns {[THREE.Vector3, THREE.Vector3]}
 */
export function cellEdgeWorld(panel, cellIndex, faceAxis, orientAxis, config) {
  const r = panelCellFaceRectLocal(panel, cellIndex, config)
  let a
  let b
  if (faceAxis.axis === 'x') {
    const x = faceAxis.sign > 0 ? r.x1 : r.x0
    a = [x, r.z0]
    b = [x, r.z1]
    if (orientAxis.axis !== 'z') throw new Error('orientAxis must be perpendicular to faceAxis')
    if (orientAxis.sign < 0) [a, b] = [b, a]
  } else {
    const z = faceAxis.sign > 0 ? r.z1 : r.z0
    a = [r.x0, z]
    b = [r.x1, z]
    if (orientAxis.axis !== 'x') throw new Error('orientAxis must be perpendicular to faceAxis')
    if (orientAxis.sign < 0) [a, b] = [b, a]
  }
  return [localFaceToWorld(panel, a[0], a[1]), localFaceToWorld(panel, b[0], b[1])]
}

// -----------------------------------------------------------------------------
// Stage A — the per-row accordion chain
// -----------------------------------------------------------------------------
/**
 * Build one row's chain segments and walk them in row-local 2D.
 *
 * @param {object} cfg normalized config
 * @param {number} r row index
 * @param {Map<string,object>} hrects "r,c" → horizontal rect
 * @param {Map<string,object>} vrects "r,c" → vertical rect (keyed by anchor cell)
 * @param {Set<number>} coveredFromAbove columns of row r owned by a vertical
 *        rect anchored at row r-1 (chain advances, no panel emitted)
 * @returns {{ segments: Array, shift: number }}
 */
function solveRowChain(cfg, r, hrects, vrects, coveredFromAbove) {
  const cols = cfg.grid.cols
  const size = Number(cfg.cell.size)
  const rectLength = Number(cfg.cell.rectLength)
  const gap = Number(cfg.gap)
  const removed = removedJoints(cfg, r)

  // --- segment list -------------------------------------------------------
  const segments = []
  for (let c = 0; c < cols; ) {
    if (hrects.has(`${r},${c}`) && c + 1 < cols) {
      segments.push({
        cols: [c, c + 1],
        L: rectLength,
        emit: true,
        type: '2x4',
        rectOrientation: 'horizontal',
      })
      c += 2
      continue
    }
    if (vrects.has(`${r},${c}`)) {
      segments.push({ cols: [c], L: size, emit: true, type: '2x4', rectOrientation: 'vertical' })
    } else if (coveredFromAbove.has(c)) {
      segments.push({ cols: [c], L: size, emit: false, type: null, rectOrientation: null })
    } else {
      segments.push({ cols: [c], L: size, emit: true, type: '2x2', rectOrientation: null })
    }
    c += 1
  }

  // --- walk ---------------------------------------------------------------
  let px = 0
  let py = 0
  let alpha = 0
  const vertexX = [0]
  for (const seg of segments) {
    seg.alphaDeg = alpha
    const a = alpha * DEG
    seg.cx = px + (seg.L / 2) * Math.cos(a)
    seg.cy = py + (seg.L / 2) * Math.sin(a)
    px += seg.L * Math.cos(a)
    py += seg.L * Math.sin(a)
    vertexX.push(px)

    const j = seg.cols[seg.cols.length - 1]
    if (j <= cols - 2 && !removed.has(j)) {
      const phi = foldAngleDeg(cfg, r, j)
      const bis = (alpha + phi / 2) * DEG
      px += gap * Math.cos(bis)
      py += gap * Math.sin(bis)
      vertexX.push(px)
      alpha += phi
    }
  }

  // --- anchor -------------------------------------------------------------
  let shift = 0
  if (cfg.rowAnchor === 'center') {
    let lo = vertexX[0]
    let hi = vertexX[0]
    for (const x of vertexX) {
      if (x < lo) lo = x
      if (x > hi) hi = x
    }
    const nominalCenter = (cols * (size + gap) - gap) / 2
    shift = nominalCenter - (lo + hi) / 2
  }

  return { segments, shift }
}

// -----------------------------------------------------------------------------
// solveLayout
// -----------------------------------------------------------------------------
/**
 * Place every panel of a config in world space. Pure and deterministic.
 *
 * Assumes the config validates (`validateConfig(...).ok`). Structurally broken
 * input (missing arrays, non-numeric dims) may throw; a validated config never
 * will.
 *
 * @param {object} config config (raw or normalized — normalized internally)
 * @returns {{ panels: Array, rowPlanes: Array, warnings: Array }}
 */
export function solveLayout(config) {
  const cfg = normalizeConfig(config)
  const cols = Number(cfg.grid.cols)
  const rowCount = Number(cfg.grid.rows)
  const size = Number(cfg.cell.size)
  const gap = Number(cfg.gap)
  const rectLength = Number(cfg.cell.rectLength)
  const warnings = []

  if (!Number.isInteger(cols) || cols < 1 || !Number.isInteger(rowCount) || rowCount < 1) {
    throw new Error(`solveLayout: grid must be integers ≥ 1 (got ${cols}×${rowCount})`)
  }

  // --- rect index ---------------------------------------------------------
  const hrects = new Map()
  const vrects = new Map()
  const covered = new Map() // row → Set of columns owned by a rect from the row above
  for (const rect of Array.isArray(cfg.rects) ? cfg.rects : []) {
    if (!rect || typeof rect !== 'object') continue
    if (rect.orientation === 'horizontal') {
      hrects.set(`${rect.row},${rect.col}`, rect)
    } else if (rect.orientation === 'vertical') {
      vrects.set(`${rect.row},${rect.col}`, rect)
      const below = rect.row + 1
      if (!covered.has(below)) covered.set(below, new Set())
      covered.get(below).add(rect.col)
    }
  }

  // --- Stage B: row planes ------------------------------------------------
  const pitchDeg = new Array(rowCount)
  pitchDeg[0] = 0
  const folds = Array.isArray(cfg.rowFoldsDeg) ? cfg.rowFoldsDeg : []
  for (let r = 0; r + 1 < rowCount; r++) {
    const raw = Number(folds[r])
    const f = Number.isFinite(raw) ? clamp(raw, -MAX_ROW_FOLD_DEG, MAX_ROW_FOLD_DEG) : 0
    pitchDeg[r + 1] = pitchDeg[r] + f
  }

  const depthDir = (r) =>
    new THREE.Vector3(0, Math.sin(pitchDeg[r] * DEG), Math.cos(pitchDeg[r] * DEG))

  const origins = new Array(rowCount)
  origins[0] = new THREE.Vector3(0, PANEL_PROFILE.overallThickness, 0)
  for (let r = 0; r + 1 < rowCount; r++) {
    const dr = depthDir(r)
    const dn = depthDir(r + 1)
    const bis = dr.clone().add(dn)
    // |rowFold| ≤ 120° means d_r and d_{r+1} are never antiparallel; guard anyway.
    if (bis.lengthSq() < 1e-12) bis.copy(dr)
    else bis.normalize()
    origins[r + 1] = origins[r].clone().addScaledVector(dr, size).addScaledVector(bis, gap)
  }

  const rowQuats = new Array(rowCount)
  for (let r = 0; r < rowCount; r++) {
    rowQuats[r] = new THREE.Quaternion().setFromAxisAngle(PITCH_AXIS, -pitchDeg[r] * DEG)
  }

  const yawQuat = new THREE.Quaternion().setFromAxisAngle(YAW_AXIS, Math.PI / 2)

  // --- Stage A + compose --------------------------------------------------
  const panels = []
  for (let r = 0; r < rowCount; r++) {
    const coveredCols = covered.get(r) ?? new Set()
    const { segments, shift } = solveRowChain(cfg, r, hrects, vrects, coveredCols)

    // Assert-level consistency with schema.js's cheap 2D chain. If these ever
    // disagree, validation (E_CROSSROW_ANGLE_MISMATCH) is checking a different
    // surface than the one being placed — a real bug, not a rounding artefact.
    const tilts = cellTilts(cfg, r)
    for (const seg of segments) {
      for (const c of seg.cols) {
        if (Math.abs(seg.alphaDeg - tilts[c]) > 1e-9) {
          throw new Error(
            `solveLayout: chain tilt ${seg.alphaDeg}° at cell (${r}, ${c}) disagrees with ` +
              `cellTilts ${tilts[c]}° — placement and schema are out of sync`,
          )
        }
      }
    }

    for (const seg of segments) {
      if (!seg.emit) continue
      const anchorCol = seg.cols[0]
      const zCenter = seg.rectOrientation === 'vertical' ? rectLength / 2 : size / 2

      const local = new THREE.Vector3(seg.cx + shift, seg.cy, zCenter)
      const position = local.applyQuaternion(rowQuats[r]).add(origins[r])

      const chainQuat = new THREE.Quaternion().setFromAxisAngle(CHAIN_AXIS, seg.alphaDeg * DEG)
      const quat = rowQuats[r].clone().multiply(chainQuat)
      if (seg.rectOrientation === 'horizontal') quat.multiply(yawQuat)

      const cells =
        seg.rectOrientation === 'vertical'
          ? [
              [r, anchorCol],
              [r + 1, anchorCol],
            ]
          : seg.cols.map((c) => [r, c])

      panels.push({
        id: `p${r}_${anchorCol}`,
        row: r,
        col: anchorCol,
        cells,
        type: seg.type,
        rectOrientation: seg.rectOrientation,
        position: vec3out(position),
        quaternion: quatOut(quat),
        tiltDeg: round9(seg.alphaDeg),
        rowPitchDeg: round9(pitchDeg[r]),
      })

      if (Math.cos(seg.alphaDeg * DEG) <= 0) {
        warnings.push({
          code: 'W_CHAIN_BACKTRACK',
          message:
            `panel p${r}_${anchorCol} sits at an in-row tilt of ${round9(seg.alphaDeg)}° — the ` +
            `accordion chain no longer advances along the row, so panels in row ${r} overlap ` +
            `or double back`,
        })
      }
    }
  }

  const rowPlanes = []
  for (let r = 0; r < rowCount; r++) {
    rowPlanes.push({ origin: vec3out(origins[r]), pitchDeg: round9(pitchDeg[r]) })
  }

  return { panels, rowPlanes, warnings }
}
