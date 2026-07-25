/**
 * grid-designer — parametric solid mesh for one LED flat panel.
 *
 * PROVENANCE / WHY THIS DIVERGED FROM panel-designer
 * =============================================================================
 * This file BEGAN as a copy of panel-designer/src/geometry/panelGeometry.js (do
 * not edit that original), but it is no longer verbatim — the copy was wrong in
 * three ways that only showed up once whole grids of panels were rendered:
 *
 *   1. The front and back CAPS were wound inside-out. Measured on the built
 *      output: the triangles at y = 0 had geometric normal −Y (pointing into the
 *      solid) and those at y = −overallThickness had +Y. Under FrontSide
 *      materials both were culled, so the lit face and the housing back were
 *      simply absent — panels appeared to be "missing several faces on the front
 *      and the back".
 *   2. The recessed "lip annulus" ring at y = −outerThickness was buried
 *      directly beneath a FULL-footprint front face at y = 0, so it contributed
 *      nothing: there was no frame-versus-diffuser distinction at all.
 *   3. `powerSupplyEdge` left one edge flush (no inset, no bevel), which
 *      collapsed that edge's quads onto each other — a panel-designer
 *      connector-modelling detail, and the source of the degenerate/coincident
 *      geometry. It is gone: a real drop-in fixture has a frame on ALL FOUR
 *      edges, so the profile here is uniform around the perimeter.
 *
 * Also dropped: the half-implemented `sidedness: 'double'` mode (never used),
 * and the unreferenced `buildPanelEdges` / `panelFrontCorners` helpers.
 *
 * `PANEL_PROFILE.slantWidth` no longer appears directly; it participates only
 * through `BODY_INSET` (= lipWidth + slantWidth), the horizontal run of the
 * tapered housing wall.
 *
 * WHAT IS BUILT
 * =============================================================================
 * A closed, outward-oriented 2-manifold shell of a drop-in LED flat panel: a
 * frame flange all the way around a slightly recessed diffuser, with a closed,
 * inward-tapering tray behind it holding the LEDs.
 *
 * Cross-section (looking along an edge; same on all four edges):
 *
 *        frame flange (visible border, material 1)
 *        ┌────┐                              ┌────┐   y =  0     ← FULL footprint
 *        │    │  ← frame inner reveal        │    │
 *        │    └──────────────────────────────┘    │   y = −t     ← LIP footprint
 *        │      diffuser cap (material 0, lit)    │
 *        │                                        │
 *         ╲                                      ╱              ← tapered housing:
 *          ╲                                    ╱                 BODY_INSET across,
 *           ╲__________________________________╱     y = −D       (D − t) down
 *                     back plate (material 1)         ← BODY footprint
 *
 *   t = PANEL_PROFILE.outerThickness   (1.0 cm — frame flange depth / diffuser recess)
 *   D = PANEL_PROFILE.overallThickness (3.7 cm — full depth, front face → tray back)
 *   frame ring width = PANEL_PROFILE.lipWidth (2.5 cm) on every side
 *   BODY_INSET       = 4.0 cm (the tapered wall's horizontal run)
 *
 * PANEL LOCAL FRAME (unchanged — core/placement.js depends on it)
 * =============================================================================
 *   - The panel is centred on the local origin in X and Z; width runs along X,
 *     height along Z.
 *   - +Y is the LIT direction. The OUTERMOST FRONT SURFACE IS STILL THE PLANE
 *     y = 0 (the frame flange top, on the full footprint), and the housing runs
 *     back to y = −D. So `core/placement.js`'s solid-corner and grounding math —
 *     which reasons about the face plane at y = 0 and a thickness of D — is
 *     completely unaffected by this rewrite. No panel moves.
 *   - The diffuser itself sits one `t` behind that plane, at y = −t. That recess
 *     is what makes the frame read as a border rather than a painted outline.
 *
 * The shell is emitted as a tube traversed front → around → back, built from
 * five rectangular rings plus two horizontal caps:
 *
 *   cap A: diffuser   LIP  @ y = −t   facing +Y   (material 0)
 *   R1 = LIP  @ y = −t   diffuser edge
 *   R2 = LIP  @ y =  0   frame inner top edge
 *   R3 = FULL @ y =  0   frame outer top edge
 *   R4 = FULL @ y = −t   frame outer bottom edge
 *   R5 = BODY @ y = −D   back plate edge
 *   cap B: back plate BODY @ y = −D   facing −Y   (material 1)
 *
 *   R1→R2  frame inner reveal   (outward normal points toward the panel centre —
 *                                it is the wall of the open recess)
 *   R2→R3  frame flange top ring (+Y — THE VISIBLE FRAME)
 *   R3→R4  frame outer side wall
 *   R4→R5  tapered housing wall
 *
 * Every ring has exactly two faces meeting it, so the result is closed and
 * watertight. 18 quads = 72 vertices = 36 triangles; vertices are DUPLICATED per
 * quad so `computeVertexNormals()` yields crisp per-face normals and hard edges
 * (the frame has to read sharply against the glow).
 *
 * tests/test-geometry.mjs is the regression guard for all of the above: it
 * re-checks the manifold property, the edge orientation consistency, the signed
 * volume against the closed-form profile volume, and the cap normals.
 */

import * as THREE from 'three'
import { PANEL_DIMENSIONS, PANEL_PROFILE, BODY_INSET } from '../config.js'

const { overallThickness, outerThickness, lipWidth } = PANEL_PROFILE

/** Material slot of the diffuser cap — the lit surface. */
export const DIFFUSER_MATERIAL_INDEX = 0
/** Material slot of the frame flange, side walls, taper and back plate. */
export const HOUSING_MATERIAL_INDEX = 1

/**
 * The four corners of a rectangle in the STANDARD RING ORDER used throughout
 * this file: [SW, SE, NE, NW] = [(minX,minZ), (maxX,minZ), (maxX,maxZ), (minX,maxZ)].
 *
 * @param {number} halfW half-width along X
 * @param {number} halfH half-height along Z
 * @returns {Array<[number, number]>} 4 [x, z] pairs
 */
function ring(halfW, halfH) {
  return [
    [-halfW, -halfH], // SW
    [halfW, -halfH],  // SE
    [halfW, halfH],   // NE
    [-halfW, halfH],  // NW
  ]
}

/**
 * Build one panel's solid geometry.
 *
 * @param {object} opts
 * @param {string} opts.type '2x2' | '2x4'
 * @returns {THREE.BufferGeometry} indexed, with two material groups
 *          (`DIFFUSER_MATERIAL_INDEX` then `HOUSING_MATERIAL_INDEX`)
 */
export function buildPanelGeometry({ type }) {
  const dim = PANEL_DIMENSIONS[type]
  if (!dim) throw new Error(`buildPanelGeometry: unknown panel type ${JSON.stringify(type)}`)

  const hw = dim.width / 2
  const hh = dim.height / 2

  const t = outerThickness      //  1.0 — flange depth / diffuser recess
  const D = overallThickness    //  3.7 — full depth
  const lw = lipWidth           //  2.5 — frame ring width
  const inset = BODY_INSET      //  4.0 — tapered wall horizontal run

  const FULL = ring(hw, hh)
  const LIP = ring(hw - lw, hh - lw)
  const BODY = ring(hw - inset, hh - inset)

  const positions = []
  const indices = []
  let next = 0

  /** Push a vertex, return its index. */
  const vert = (x, y, z) => {
    positions.push(x, y, z)
    return next++
  }

  /** Two triangles (a,b,c) and (a,c,d) — CCW as seen from outside. */
  const quad = (a, b, c, d) => {
    indices.push(a, b, c, a, c, d)
  }

  /**
   * A horizontal cap. `+1` faces +Y (corner order SW, NW, NE, SE); `-1` faces
   * −Y (corner order SW, SE, NE, NW).
   */
  const cap = (corners, y, facing) => {
    const order = facing > 0 ? [0, 3, 2, 1] : [0, 1, 2, 3]
    const v = order.map((i) => vert(corners[i][0], y, corners[i][1]))
    quad(v[0], v[1], v[2], v[3])
  }

  /**
   * The four side quads joining ring A (at `yA`) to ring B (at `yB`). Traversed
   * in the R1→R2→R3→R4→R5 order documented in the header, this single winding
   * rule gives the correct OUTWARD normal for every connection — no
   * per-connection special-casing.
   */
  const connect = (A, yA, B, yB) => {
    for (let i = 0; i < 4; i++) {
      const ni = (i + 1) % 4
      quad(
        vert(A[i][0], yA, A[i][1]),
        vert(A[ni][0], yA, A[ni][1]),
        vert(B[ni][0], yB, B[ni][1]),
        vert(B[i][0], yB, B[i][1]),
      )
    }
  }

  // --- group 0: the diffuser, emitted first so it owns index range [0, 6) -----
  cap(LIP, -t, +1)
  const diffuserIndexCount = indices.length

  // --- group 1: frame + housing ----------------------------------------------
  connect(LIP, -t, LIP, 0)     // R1→R2  frame inner reveal
  connect(LIP, 0, FULL, 0)     // R2→R3  frame flange top ring (the visible frame)
  connect(FULL, 0, FULL, -t)   // R3→R4  frame outer side wall
  connect(FULL, -t, BODY, -D)  // R4→R5  tapered housing wall
  cap(BODY, -D, -1)            // cap B  back plate

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
  geometry.setIndex(indices)
  geometry.addGroup(0, diffuserIndexCount, DIFFUSER_MATERIAL_INDEX)
  geometry.addGroup(
    diffuserIndexCount,
    indices.length - diffuserIndexCount,
    HOUSING_MATERIAL_INDEX,
  )
  geometry.computeVertexNormals()
  geometry.computeBoundingBox()
  geometry.computeBoundingSphere()

  return geometry
}
