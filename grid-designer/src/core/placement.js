/**
 * grid-designer — deterministic placement solver.
 *
 * HEADLESS ZONE (src/core/): pure functions, importable from plain node.
 *   - explicit `.js` extensions on ALL relative imports
 *   - may import `three` math classes only; never components / store / DOM
 *   - same config in → identical output out (deep-equal across calls)
 *
 * =============================================================================
 * MODEL — "per-column fold strips" (schema v2)
 * =============================================================================
 * The surface is NOT solved as rigid origami. Every joint is spanned by a ~2cm
 * printed connector, so panels never share vertices. Each COLUMN is solved as an
 * exact 2D chain in its own Y–Z plane (the fold strip), and `report.js` measures
 * whatever slack lands on the side-by-side (in-row) connections between columns.
 *
 * WORLD CONVENTIONS
 *   Units cm. Y up. Columns along +X (c = 0..cols-1) and NEVER moving: column c
 *   owns x ∈ [c·(size+gap), c·(size+gap)+size] forever. Rows recede along +Z
 *   (r = 0..rows-1) with the shore/window at z = 0. Everything is lifted by
 *   PANEL_PROFILE.overallThickness (3.7, = schema.js SHORE_Y) in +Y so the
 *   housings rest on the floor — a flat panel's lit face is at y = 3.7.
 *
 * PANEL LOCAL FRAME (matches geometry/panelGeometry.js)
 *   Lit face lies in the local XZ plane at local y = 0, centered on the origin.
 *   Panel *width* runs along local X, panel *depth/length* along local Z, and
 *   the lit face looks along +Y.
 *     '2x2' → width = cell.size,  length = cell.size
 *     '2x4' → width = cell.size,  length = cell.rectLength
 *
 * -----------------------------------------------------------------------------
 * THE COLUMN CHAIN
 * -----------------------------------------------------------------------------
 * The chain walk itself lives in schema.js (`columnChain`) so validation and
 * placement can never disagree about the surface: it returns each segment's
 * cumulative pitch ψ and its start point (y, z), starting from (3.7, 0) at ψ = 0
 * and advancing `length·d` per segment plus `gap·bisector` at every surviving
 * joint, with d = (sinψ, cosψ) in (y, z). This module turns that into world
 * placements:
 *
 *   depth dir      d = (0, sinψ, cosψ)
 *   panel center   = ( xCenter, p + (L_chain/2)·d )
 *   quaternion     = rotX(−ψ)              (pure X-rotation; ψ = 90° stands the
 *                                           panel up, so +fold pitches it UP)
 *
 * where L_chain is what the panel occupies ALONG the chain (cell.size for
 * squares and horizontal plates, cell.rectLength for a vertical plate) and
 * xCenter is size/2 + c·(size+gap) for anything owned by one column.
 *
 * -----------------------------------------------------------------------------
 * RECTS
 * -----------------------------------------------------------------------------
 * VERTICAL rect at (r,c) covers (r,c)+(r+1,c): one 121cm plate lying ALONG the
 * chain, so its local +Z (its length) already points down the chain and it needs
 * NO yaw. It removes joint k = r (validation forces that fold to 0), so the
 * plate is genuinely rigid and the chain simply advances 121 instead of 60.
 *
 * HORIZONTAL rect at (r,c) covers (r,c)+(r,c+1): one plate ACROSS two columns.
 * The '2x4' geometry is 60 wide (local X) × 121 long (local Z), so it is yawed
 * +90° about local Y to run its length along +X (composition rotX(−ψ)·yaw90).
 * It is owned by column c's chain — its (y, z) come from there — and is CENTERED
 * on the two-column slot in X: the slot spans [c·(size+gap), c·(size+gap) +
 * 2·size+gap] = 122cm at the defaults, and the 121cm plate sits centered in it,
 * 0.5cm shy of each outer edge (the W_RECT_LENGTH slack, placed honestly).
 * Column c+1 treats row r as a PHANTOM square: its chain advances 60 plus the
 * normal gap/fold so the rest of that column is unaffected, but it emits no
 * panel. Whatever mismatch that leaves is left for `report.js` to measure.
 *
 * -----------------------------------------------------------------------------
 * OUTPUT
 * -----------------------------------------------------------------------------
 *   solveLayout(config) → {
 *     panels: [{ id, row, col, cells, type, rectOrientation,
 *                position: [x,y,z], quaternion: [x,y,z,w], rowPitchDeg }],
 *     columnChains: [{ col, points: [[y,z], …], pitchesDeg: [ψ per row] }],
 *     warnings: [{ code, message }],
 *   }
 * Panels are emitted column-major (all of column 0 front-to-back, then column 1,
 * …). `rowPitchDeg` is the panel's cumulative pitch. `columnChains[c].points`
 * are the chain vertices of column c in (y, z): the start, then after every
 * segment and every gap advance — handy for drawing the fold profile. v1's
 * `rowPlanes` and per-panel `tiltDeg` are GONE (there is no in-row accordion any
 * more, so every panel's tilt is zero by construction).
 *
 * Layout warnings:
 *   W_CHAIN_BACKTRACK  a panel's cumulative pitch has cosψ ≤ 0 — the chain no
 *                      longer advances away from the window (legal, but it folds
 *                      back over itself)
 *   W_BELOW_FLOOR      lit-face corners dip below y = 0: those panels have
 *                      nothing to stand on
 *
 * All numbers are plain JSON-able doubles rounded to 1e-9 (which also normalizes
 * −0 → 0) so repeated solves are byte-identical.
 */

import * as THREE from 'three'
import { columnChain, normalizeConfig } from './schema.js'

const DEG = Math.PI / 180

/** Yaw axis for horizontal rects (= panel local +Y, the lit direction). */
const YAW_AXIS = new THREE.Vector3(0, 1, 0)
/** Chain pitch axis (= world +X, along the rows). */
const PITCH_AXIS = new THREE.Vector3(1, 0, 0)

/** Lit-face corners below −FLOOR_EPSILON count as unsupported. */
const FLOOR_EPSILON = 1e-6

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
 * @returns {{ panels: Array, columnChains: Array, warnings: Array }}
 */
export function solveLayout(config) {
  const cfg = normalizeConfig(config)
  const cols = Number(cfg.grid.cols)
  const rowCount = Number(cfg.grid.rows)
  const size = Number(cfg.cell.size)
  const gap = Number(cfg.gap)
  const warnings = []

  if (!Number.isInteger(cols) || cols < 1 || !Number.isInteger(rowCount) || rowCount < 1) {
    throw new Error(`solveLayout: grid must be integers ≥ 1 (got ${cols}×${rowCount})`)
  }

  const pitch = size + gap
  const yawQuat = new THREE.Quaternion().setFromAxisAngle(YAW_AXIS, Math.PI / 2)

  const panels = []
  const columnChains = []
  const backtracking = []

  for (let c = 0; c < cols; c++) {
    const chain = columnChain(cfg, c)
    columnChains.push({
      col: c,
      points: chain.points.map(([y, z]) => [round9(y), round9(z)]),
      pitchesDeg: chain.pitchesDeg.map(round9),
    })

    for (const seg of chain.segments) {
      if (seg.kind === 'phantom') continue

      const r = seg.rows[0]
      const psi = seg.pitchDeg
      const [oy, oz] = seg.origin
      const along = seg.kind === 'vrect' ? Number(seg.length) : size
      const half = along / 2

      const xCenter =
        seg.kind === 'hrect'
          ? c * pitch + (2 * size + gap) / 2 // centered across the two-column slot
          : c * pitch + size / 2

      const position = new THREE.Vector3(
        xCenter,
        oy + half * Math.sin(psi * DEG),
        oz + half * Math.cos(psi * DEG),
      )

      const quat = new THREE.Quaternion().setFromAxisAngle(PITCH_AXIS, -psi * DEG)
      if (seg.kind === 'hrect') quat.multiply(yawQuat)

      const cells =
        seg.kind === 'vrect'
          ? [
              [r, c],
              [r + 1, c],
            ]
          : seg.kind === 'hrect'
            ? [
                [r, c],
                [r, c + 1],
              ]
            : [[r, c]]

      const id = `p${r}_${c}`
      panels.push({
        id,
        row: r,
        col: c,
        cells,
        type: seg.kind === 'square' ? '2x2' : '2x4',
        rectOrientation: seg.kind === 'vrect' ? 'vertical' : seg.kind === 'hrect' ? 'horizontal' : null,
        position: vec3out(position),
        quaternion: quatOut(quat),
        rowPitchDeg: round9(psi),
      })

      if (Math.cos(psi * DEG) <= 0) backtracking.push({ id, psi: round9(psi) })
    }
  }

  for (const { id, psi } of backtracking) {
    warnings.push({
      code: 'W_CHAIN_BACKTRACK',
      message:
        `panel ${id} sits at a cumulative pitch of ${psi}° — its column's chain no longer ` +
        `advances away from the window, so the strip folds back over itself`,
    })
  }

  // Ground support: lit-face corners below the floor have nothing to stand on.
  const below = []
  for (const panel of panels) {
    const lowest = panelWorldCorners(panel, cfg).reduce((m, corner) => Math.min(m, corner[1]), Infinity)
    if (lowest < -FLOOR_EPSILON) below.push(`${panel.id} (${round9(lowest)}cm)`)
  }
  if (below.length > 0) {
    warnings.push({
      code: 'W_BELOW_FLOOR',
      message:
        `${below.length} panel(s) dip below the floor (y = 0) and have nothing to stand on: ` +
        below.join(', '),
    })
  }

  return { panels, columnChains, warnings }
}
