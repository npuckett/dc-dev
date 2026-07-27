/**
 * grid-designer — parametric solid mesh for one LED flat panel.
 *
 * WHAT IS BUILT
 * =============================================================================
 * A closed, outward-oriented 2-manifold shell, swept from the measured section
 * in `config.js`. This file owns NO dimensions: it reads `panelSectionRings()`
 * and sweeps it. Refining the panel means editing `PANEL_PROFILE`, never here.
 *
 * The section is documented in config.js. Its shape in one line: a chamfered
 * bezel over a recessed diffuser on the front, a vertical outer wall, and — the
 * feature that matters for connectors — a 3cm nearly-flat FLANGE on the back
 * with a steep taper falling away behind it to the back plate.
 *
 * PROVENANCE / WHY THIS DIVERGED FROM panel-designer
 * =============================================================================
 * Began as a copy of panel-designer/src/geometry/panelGeometry.js (do not edit
 * that original). It has since been corrected twice.
 *
 * First, three bugs that only showed up once whole grids were rendered:
 *   1. front and back CAPS were wound inside-out, so under FrontSide materials
 *      the lit face and the housing back were simply absent;
 *   2. the recessed lip ring was buried beneath a full-footprint front face, so
 *      there was no frame-versus-diffuser distinction at all;
 *   3. `powerSupplyEdge` left one edge flush, collapsing that edge's quads onto
 *      each other.
 *
 * Then the section itself was replaced. The inherited one had a 2.5cm flat lip,
 * a 1.0cm rim and a taper running straight from the outer wall to a body inset
 * 4.0cm — 3.7cm thick overall, and with NO BACK FLANGE. The measured panels
 * (`updatedPanelGeo/`) are 4.1cm thick and carry a 3cm flange all the way round.
 * The absent flange is why the first connector design made no sense: it hooked a
 * taper that does not exist where it was modelled.
 *
 * Note on (3): the flush `powerSupplyEdge` was dropped as a modelling artifact,
 * and that remains right — the FRAME is uniform on all four edges, confirmed by
 * sectioning both source models. What is NOT uniform is the power supply, a box
 * on the back of one edge. It is a separate solid, not a change to the section;
 * see `POWER_SUPPLY` in config.js and `buildPowerSupplyGeometry` below.
 *
 * PANEL LOCAL FRAME (unchanged — core/v3/placement.js depends on it)
 * =============================================================================
 *   - Centred on the local origin in X and Z; width along X, height along Z.
 *   - +Y is the LIT direction. THE FRONT-MOST SURFACE IS THE PLANE y = 0 (the
 *     bezel's peak), and the housing runs back to y = −overallThickness. So
 *     placement.js's solid-corner and grounding math is unaffected by the new
 *     section beyond the thickness changing from 3.7 to 4.1.
 *
 * The shell is a tube traversed front → around → back: one quad ring per
 * consecutive pair of section rings, plus a cap at each end. Vertices are
 * DUPLICATED per quad so `computeVertexNormals()` yields crisp per-face normals.
 *
 * tests/test-geometry.mjs is the regression guard: manifold property, edge
 * orientation consistency, signed volume against the section computed
 * independently, and cap normals.
 */

import * as THREE from 'three'
import { PANEL_DIMENSIONS, PANEL_PROFILE, POWER_SUPPLY, panelSectionRings } from '../config.js'

/** Material slot of the diffuser cap — the lit surface. */
export const DIFFUSER_MATERIAL_INDEX = 0
/** Material slot of the bezel, outer wall, flange, taper and back plate. */
export const HOUSING_MATERIAL_INDEX = 1

/**
 * The four corners of a rectangle inset `inset` from a `halfW × halfH`
 * footprint, in the STANDARD RING ORDER [SW, SE, NE, NW].
 */
function ring(halfW, halfH, inset) {
  const w = halfW - inset
  const h = halfH - inset
  return [[-w, -h], [w, -h], [w, h], [-w, h]]
}

/**
 * Build one panel's solid geometry.
 *
 * @param {object} opts
 * @param {string} opts.type '2x2' | '2x4'
 * @param {object} [opts.profile] overrides for PANEL_PROFILE — the refinement hook
 * @returns {THREE.BufferGeometry} indexed, with two material groups
 */
export function buildPanelGeometry({ type, profile = PANEL_PROFILE }) {
  const dim = PANEL_DIMENSIONS[type]
  if (!dim) throw new Error(`buildPanelGeometry: unknown panel type ${JSON.stringify(type)}`)

  const hw = dim.width / 2
  const hh = dim.height / 2
  const sections = panelSectionRings(profile)
  const rings = sections.map((s) => ({ ...s, corners: ring(hw, hh, s.inset), y: -s.depth }))

  const positions = []
  const indices = []
  let next = 0

  const vert = (x, y, z) => {
    positions.push(x, y, z)
    return next++
  }
  /** Two triangles (a,b,c) and (a,c,d) — CCW as seen from outside. */
  const quad = (a, b, c, d) => {
    indices.push(a, b, c, a, c, d)
  }
  /** `+1` faces +Y (order SW, NW, NE, SE); `-1` faces −Y (SW, SE, NE, NW). */
  const cap = (corners, y, facing) => {
    const order = facing > 0 ? [0, 3, 2, 1] : [0, 1, 2, 3]
    const v = order.map((i) => vert(corners[i][0], y, corners[i][1]))
    quad(v[0], v[1], v[2], v[3])
  }
  /**
   * The four side quads joining ring A to ring B. Traversed in section order
   * (front → outward → back) this single winding rule gives the correct OUTWARD
   * normal for every connection — no per-connection special-casing.
   */
  const connect = (A, B) => {
    for (let i = 0; i < 4; i++) {
      const ni = (i + 1) % 4
      quad(
        vert(A.corners[i][0], A.y, A.corners[i][1]),
        vert(A.corners[ni][0], A.y, A.corners[ni][1]),
        vert(B.corners[ni][0], B.y, B.corners[ni][1]),
        vert(B.corners[i][0], B.y, B.corners[i][1]),
      )
    }
  }

  // --- group 0: the diffuser, emitted first so it owns the leading range ----
  cap(rings[0].corners, rings[0].y, +1)
  const diffuserIndexCount = indices.length

  // --- group 1: bezel, outer wall, flange, taper, back plate ----------------
  for (let i = 0; i < rings.length - 1; i++) connect(rings[i], rings[i + 1])
  cap(rings[rings.length - 1].corners, rings[rings.length - 1].y, -1)

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
  geometry.setIndex(indices)
  geometry.addGroup(0, diffuserIndexCount, DIFFUSER_MATERIAL_INDEX)
  geometry.addGroup(diffuserIndexCount, indices.length - diffuserIndexCount, HOUSING_MATERIAL_INDEX)
  geometry.computeVertexNormals()
  geometry.computeBoundingBox()
  geometry.computeBoundingSphere()

  return geometry
}

/**
 * The power supply box for one panel, in the SAME local frame as the panel.
 *
 * A separate solid rather than part of the section, because it is on ONE edge
 * and the section is uniform on all four. `edge` selects which, using the 0–3
 * convention in config.js (0 south −Z, 1 east +X, 2 north +Z, 3 west −X).
 *
 * It hangs from the flange plane back toward the back plate and is centred on
 * its edge, matching both source models.
 *
 * @param {object} opts
 * @param {string} opts.type '2x2' | '2x4'
 * @param {number} opts.edge 0–3
 * @param {object} [opts.supply] overrides for POWER_SUPPLY
 * @param {object} [opts.profile] overrides for PANEL_PROFILE
 * @returns {THREE.BufferGeometry} a closed box
 */
export function buildPowerSupplyGeometry({ type, edge = 0, supply = POWER_SUPPLY, profile = PANEL_PROFILE }) {
  const dim = PANEL_DIMENSIONS[type]
  if (!dim) throw new Error(`buildPowerSupplyGeometry: unknown panel type ${JSON.stringify(type)}`)
  const s = { ...POWER_SUPPLY, ...supply }
  const p = { ...PANEL_PROFILE, ...profile }

  // It sits under the flange, so its top is the flange's outer depth.
  const yTop = -p.outerWallDepth
  const yBot = yTop - s.height

  // Along-edge half length, and the two inboard offsets, in edge-local terms.
  const half = s.length / 2
  const near = s.edgeInset
  const far = s.edgeInset + s.depth

  const hw = dim.width / 2
  const hh = dim.height / 2
  // Map edge-local (along, inboard) to panel-local (x, z).
  const box = {
    0: { x: [-half, half], z: [-hh + near, -hh + far] },   // south, −Z
    2: { x: [-half, half], z: [hh - far, hh - near] },     // north, +Z
    3: { x: [-hw + near, -hw + far], z: [-half, half] },   // west, −X
    1: { x: [hw - far, hw - near], z: [-half, half] },     // east, +X
  }[edge]
  if (!box) throw new Error(`buildPowerSupplyGeometry: edge must be 0–3 (got ${JSON.stringify(edge)})`)

  const geometry = new THREE.BoxGeometry(
    box.x[1] - box.x[0],
    yTop - yBot,
    box.z[1] - box.z[0],
  )
  geometry.translate((box.x[0] + box.x[1]) / 2, (yTop + yBot) / 2, (box.z[0] + box.z[1]) / 2)
  geometry.computeBoundingBox()
  geometry.computeBoundingSphere()
  return geometry
}
