/**
 * Copied from panel-designer/src/geometry/panelGeometry.js — do not edit the original.
 * Verbatim except the config import below, which now carries an explicit `.js`
 * extension (required for node ESM; the original omits it, which is why
 * panel-designer's node tests cannot load it).
 *
 * Panel Geometry — Parametric solid mesh builder
 *
 * Constructs a THREE.BufferGeometry representing a single LED panel as an
 * extruded solid with the physical cross-section:
 *
 *   - A thin front slab (the display face) at full footprint × outerThickness.
 *   - A raised back body (the housing) inset from 3 edges by BODY_INSET,
 *     rising to overallThickness, with beveled (slanted) transitions.
 *   - The powerSupplyEdge is flush: the body extends to that edge with no
 *     inset and no bevel (full depth right to the rim).
 *
 * Single vs double-sided:
 *   - single: one lit face (+Y). Back is the opaque housing.
 *   - double: symmetric — a lit face on both +Y and -Y, housing frame between.
 *
 * The geometry is centered at panel-local origin with +Y = front (display) and
 * the face spanning the local XZ-plane.
 */

import * as THREE from 'three'
import {
  PANEL_DIMENSIONS,
  PANEL_PROFILE,
  BODY_INSET,
  edgeEndpoints,
  edgeOutwardNormal,
} from '../config.js'

const { overallThickness, outerThickness, lipWidth, slantWidth } = PANEL_PROFILE

/**
 * Build the panel solid geometry.
 *
 * @param {object} opts
 * @param {string} opts.type        - '2x2' | '2x4'
 * @param {string} opts.sidedness   - 'single' | 'double'
 * @param {number} opts.powerSupplyEdge - 0..3 (which edge is flush, no lip)
 * @returns {THREE.BufferGeometry}
 */
export function buildPanelGeometry({ type, sidedness = 'single', powerSupplyEdge = 0 }) {
  const dim = PANEL_DIMENSIONS[type]
  const w = dim.width
  const h = dim.height
  const hw = w / 2
  const hh = h / 2

  const positions = []
  const indices = []

  // Depth levels (Y). +Y = front display face (y=0); -Y = back (y = -overall).
  const FRONT = 0
  const RIM = -outerThickness       // -1.0
  const BACK = -overallThickness    // -3.7

  // For double-sided panels the profile mirrors about the mid-depth plane.
  // We compute a "mid" offset so the geometry is symmetric about Y = -overall/2.
  const midY = sidedness === 'double' ? -overallThickness / 2 : 0
  // In double-sided, front face is at +midY, back housing at -midY-mirror... but
  // keep it simple: double = two front slabs back-to-back. Front slab +Y, a
  // mirrored front slab -Y, housing between. We implement by building the
  // single-sided solid then mirroring the front-face portion to the back.
  // For V1 we model double-sided as a full solid block (both faces lit visually
  // via material) to keep the collision volume correct and simple.
  const yFront = sidedness === 'double' ? midY + overallThickness / 2 : FRONT
  const yBack = sidedness === 'double' ? midY - overallThickness / 2 : BACK
  // Note: for double, thickness is symmetric so |yFront| = |yBack|.

  // ---------------------------------------------------------------------------
  // Helper: push a quad (two triangles) given 4 corner positions.
  // Corners CCW when viewed from outside (normal toward viewer).
  // ---------------------------------------------------------------------------
  let vIdx = 0
  function vert(x, y, z) {
    positions.push(x, y, z)
    return vIdx++
  }
  function quad(a, b, c, d) {
    // a,b,c,d in CCW order (front-facing)
    indices.push(a, b, c, a, c, d)
  }

  // ---------------------------------------------------------------------------
  // Define the footprint corners and the inset (body) corners.
  //
  // The body is inset by BODY_INSET on the 3 lip sides, but flush (no inset)
  // on the power-supply edge. So the body footprint is a rectangle inset on
  // 3 sides but touching the panel edge on the powerSupplyEdge side.
  //
  // Footprint corners (CCW from south-west):
  //   SW (-hw,-hh)  SE (hw,-hh)  NE (hw,hh)  NW (-hw,hh)
  // ---------------------------------------------------------------------------
  const inset = BODY_INSET

  // Inset per edge: 0 if this is the power-supply edge, else `inset`.
  const insetFor = (edge) => (edge === powerSupplyEdge ? 0 : inset)

  // Body (raised housing) corners — inset inward from each edge by insetFor(edge).
  // We compute body footprint by shrinking the outer rect per-edge.
  const bodyMinX = -hw + insetFor(3) // west inset
  const bodyMaxX = hw - insetFor(1)  // east inset
  const bodyMinZ = -hh + insetFor(0) // south inset
  const bodyMaxZ = hh - insetFor(2)  // north inset

  // Bevel transition corners: at lipWidth (the flat lip ends, bevel begins).
  // On lip sides, the bevel starts at insetFor(edge) and goes to insetFor+slant.
  // The "bevel start" footprint = outer rect inset by lipWidth (on lip sides only).
  const lipMinX = -hw + (powerSupplyEdge === 3 ? 0 : lipWidth)
  const lipMaxX = hw - (powerSupplyEdge === 1 ? 0 : lipWidth)
  const lipMinZ = -hh + (powerSupplyEdge === 0 ? 0 : lipWidth)
  const lipMaxZ = hh - (powerSupplyEdge === 2 ? 0 : lipWidth)

  // Y level where the bevel starts (= rim depth, -1.0)
  const yRim = sidedness === 'double' ? yFront - outerThickness : RIM
  // For double-sided, the back side also has a rim at yBack + outerThickness.
  const yRimBack = sidedness === 'double' ? yBack + outerThickness : null

  // ---------------------------------------------------------------------------
  // Build the geometry as a set of "rings" (perimeter loops at each Y level)
  // connected into quads. This is robust for both single & double sided.
  //
  // Rings (front → back):
  //   R0: front face plane     (yFront)   — full footprint, the lit surface
  //   R1: rim plane            (yRim)     — full footprint (outer rim edge)
  //   R2: lip end / bevel top (yRim)      — footprint inset by lipWidth
  //   R3: body top             (yBack)    — footprint inset by BODY_INSET (body)
  //
  // R1 and R2 share the same Y but different footprint → the flat lip annulus
  // sits between them. The bevel connects R2 (at yRim) to R3 (at yBack).
  // ---------------------------------------------------------------------------

  // Footprint corner generator: returns 4 corners [SW,SE,NE,NW] for a given
  // [minX,maxX,minZ,maxZ] box.
  const boxCorners = (minX, maxX, minZ, maxZ) => [
    [minX, minZ], // SW
    [maxX, minZ], // SE
    [maxX, maxZ], // NE
    [minX, maxZ], // NW
  ]

  const full = boxCorners(-hw, hw, -hh, hh)
  const lip = boxCorners(lipMinX, lipMaxX, lipMinZ, lipMaxZ)
  const body = boxCorners(bodyMinX, bodyMaxX, bodyMinZ, bodyMaxZ)

  // --- Front face (lit surface) ---
  // Full footprint at yFront, facing +Y. Corners CCW when viewed from +Y.
  // In XZ-plane viewed from +Y: CCW = SW, SE, NE, NW? Viewed from above (+Y down),
  // CCW is SW→SE→NE→NW. For a +Y-facing quad we want CCW in XZ as seen from +Y.
  {
    const c = full.map(([x, z]) => vert(x, yFront, z))
    quad(c[0], c[1], c[2], c[3]) // +Y facing
  }

  // --- Outer rim band: connect full footprint at yFront down to yRim ---
  // The vertical outer wall (outerThickness tall) on all 4 sides.
  // For each edge, connect the full-footprint corners to the same at yRim.
  {
    const top = full.map(([x, z]) => [x, yFront, z])
    const bot = full.map(([x, z]) => [x, yRim, z])
    for (let i = 0; i < 4; i++) {
      const a = vert(...top[i])
      const b = vert(...top[(i + 1) % 4])
      const c2 = vert(...bot[(i + 1) % 4])
      const d = vert(...bot[i])
      // outward facing
      quad(a, b, c2, d)
    }
  }

  // --- Flat lip annulus: from full footprint edge inward to lip footprint ---
  // This is the flat ring at yRim between `full` and `lip` corners.
  {
    const outer = full.map(([x, z]) => [x, yRim, z])
    const inner = lip.map(([x, z]) => [x, yRim, z])
    for (let i = 0; i < 4; i++) {
      const ni = (i + 1) % 4
      const a = vert(...outer[i])
      const b = vert(...outer[ni])
      const c2 = vert(...inner[ni])
      const d = vert(...inner[i])
      quad(a, b, c2, d) // faces +Y (up, the lip surface)... but lip is the
      // recessed flat area — it faces +Y (toward viewer). Correct.
    }
  }

  // --- Bevel ramp: from lip footprint (yRim) to body footprint (yBack) ---
  // The slanted transition. Connects `lip` corners at yRim to `body` corners
  // at yBack. On the power-supply edge, lip == body == full (no bevel there,
  // since inset is 0) — the quad degenerates (handled naturally: zero-width).
  {
    const top = lip.map(([x, z]) => [x, yRim, z])
    const bot = body.map(([x, z]) => [x, yBack, z])
    for (let i = 0; i < 4; i++) {
      const ni = (i + 1) % 4
      const a = vert(...top[i])
      const b = vert(...top[ni])
      const c2 = vert(...bot[ni])
      const d = vert(...bot[i])
      quad(a, b, c2, d)
    }
  }

  // --- Back body walls: vertical sides of the raised housing ---
  // From body footprint at yRim (top of bevel meets body) — but the body's
  // side wall goes from yRim down to yBack? No: body sits at yBack as the top.
  // Actually the housing body is a block from yBack up to the bevel. Its
  // vertical walls run from the bevel-attach line. Since the bevel already
  // connects lip(yRim)→body(yBack), the body's vertical wall is zero-height
  // at the body's top edge — the body IS just the top face. Let me reconsider.
  //
  // Re-derivation of the solid layers (front=+Y top, back=-Y bottom):
  //   yFront (0)    : front lit face (full footprint)
  //   yRim (-1.0)   : bottom of the outer rim. Lip annulus lives here.
  //                   Bevel starts here at the lip footprint.
  //   yBack (-3.7)  : the housing body's BACK face (full body footprint).
  //
  // So the solid is:
  //   front face (full, yFront) → outer rim walls (full, yFront→yRim) →
  //   lip annulus (full→lip, at yRim) → bevel ramp (lip→body, yRim→yBack) →
  //   back face (body footprint, at yBack).
  // The housing body has vertical walls too — from the body footprint down?
  // No. The body footprint at yBack is the bottom. The bevel rises from body
  // footprint (yBack) up to lip footprint (yRim). There's no separate vertical
  // body wall — the bevel IS the transition. On the power-supply edge (inset 0),
  // the bevel is vertical (lip==body==full corner), forming the full-depth wall.
  //
  // So we just need the back face to close the solid.

  // --- Back face (housing back) ---
  // Body footprint at yBack, facing -Y.
  {
    const c = body.map(([x, z]) => vert(x, yBack, z))
    // CCW for -Y facing = reverse of +Y
    quad(c[0], c[3], c[2], c[1])
  }

  // ---------------------------------------------------------------------------
  // DOUBLE-SIDED: mirror the front-face assembly to the back.
  // For double-sided we already shifted yFront/yBack symmetric about midY.
  // The structure above produces a symmetric solid IF yRim is computed as
  // yFront - outerThickness. For double: yFront = +t/2, yRim = +t/2 - 1.0,
  // yBack = -t/2. The back face at yBack faces -Y and the front at yFront
  // faces +Y — both are "lit" faces in material terms. The geometry is already
  // a closed solid. Good — double-sided just shifts the center plane.
  // ---------------------------------------------------------------------------

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
  geometry.setIndex(indices)
  geometry.computeVertexNormals()
  geometry.computeBoundingBox()
  geometry.computeBoundingSphere()

  return geometry
}

/**
 * Build a lightweight wireframe edges geometry for a panel (for overlay).
 * Returns a THREE.EdgesGeometry.
 */
export function buildPanelEdges(panelGeo) {
  return new THREE.EdgesGeometry(panelGeo, 1) // 1° threshold
}

/**
 * Get the 4 front-face corners of a panel in local coords (for intersection
 * sampling, joint attachment, etc.). Returns array of [x,y,z].
 * Front face is at y=yFront centered; corners in CCW order.
 */
export function panelFrontCorners(type) {
  const dim = PANEL_DIMENSIONS[type]
  const hw = dim.width / 2
  const hh = dim.height / 2
  return [
    [-hw, 0, -hh],
    [hw, 0, -hh],
    [hw, 0, hh],
    [-hw, 0, hh],
  ]
}
