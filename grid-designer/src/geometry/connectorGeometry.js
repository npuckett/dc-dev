/**
 * grid-designer — the solid for one 3D-printed connector.
 *
 * WHAT IS BUILT
 * =============================================================================
 * A closed, outward-oriented 2-manifold shell: two C-channels that clip over two
 * panels' frame rims, joined by a spine crossing the gap between them. The
 * cross-section — and the reasoning behind its wedge-shaped hook, which is where
 * this departs from v1's parallel-sided slot — lives in
 * `core/v3/connectors.js`'s `connectorProfile`. This file only sweeps it.
 *
 * THE SWEEP IS A LOFT, NOT AN EXTRUSION, AND THAT IS THE WHOLE TWIST STORY
 * =============================================================================
 * A joint's two panels are rigid planes, so along the joint:
 *
 *   - the FOLD is constant      (each panel's normal is fixed)
 *   - the SPAN varies LINEARLY  (two straight rim lines, generally not parallel)
 *
 * So a part's two ends differ only in span, and lofting between a `spanStartCm`
 * cross-section and a `spanEndCm` one reproduces the joint's twist exactly, with
 * no separate twist term anywhere in the geometry. A twisted joint just IS one
 * whose two ends want different spans.
 *
 * LOCAL FRAME
 * =============================================================================
 * Local X = p̂ (rim A toward rim B), Y = q̂ (up, the lit side), Z = r̂ (along the
 * joint), origin at the station's `mid`. That is the same right-handed basis
 * `connectorOBB` uses, so a mesh placed at `station.mid` with the quaternion of
 * (p̂, q̂, r̂) coincides with its own collision box. Do not let the two drift.
 *
 * The shell is emitted as a tube: one quad per profile edge joining the two end
 * rings, plus a triangulated cap at each end. Vertices are duplicated per face
 * so `computeVertexNormals()` gives crisp per-face normals, matching
 * panelGeometry.js — a printed part should read as facets, not as a blob.
 */

import * as THREE from 'three'
import { connectorEndProfiles, CONNECTOR_PROFILE } from '../core/v3/connectors.js'

/**
 * Build one connector's solid.
 *
 * @param {object} station a station from `solveConnectors`
 * @param {object} [profile] overrides for CONNECTOR_PROFILE
 * @returns {THREE.BufferGeometry} closed, outward-oriented, in the station frame
 */
export function buildConnectorGeometry(station, profile = CONNECTOR_PROFILE) {
  const { start, end } = connectorEndProfiles(station, profile)
  const half = station.lengthCm / 2

  if (start.points.length !== end.points.length) {
    // Cannot happen — connectorProfile emits a fixed 16-point outline — but a
    // loft between mismatched rings would silently produce a torn shell rather
    // than fail, which is exactly the kind of bug that survives to the printer.
    throw new Error('buildConnectorGeometry: end profiles have different point counts')
  }

  const positions = []
  const indices = []
  let next = 0

  const vert = (x, y, z) => {
    positions.push(x, y, z)
    return next++
  }
  const quad = (a, b, c, d) => {
    indices.push(a, b, c, a, c, d)
  }

  const N = start.points.length

  // --- the tube: one quad per profile edge ----------------------------------
  // The profile is counter-clockwise in (p, q), so for edge k → k+1 the quad
  // (start_k, start_k+1, end_k+1, end_k) faces OUTWARD. Verified in the test
  // against the signed volume rather than asserted here.
  for (let k = 0; k < N; k++) {
    const k1 = (k + 1) % N
    const s0 = start.points[k]
    const s1 = start.points[k1]
    const e0 = end.points[k]
    const e1 = end.points[k1]
    quad(
      vert(s0[0], s0[1], -half),
      vert(s1[0], s1[1], -half),
      vert(e1[0], e1[1], half),
      vert(e0[0], e0[1], half),
    )
  }

  // --- the two caps ---------------------------------------------------------
  // Triangulated once on the START outline and reused for the END: the two
  // rings share a point count and a winding, so one triangulation indexes both.
  // ShapeUtils is three's earcut — pure 2D math, no scene graph.
  const contour = start.points.map(([p, q]) => new THREE.Vector2(p, q))
  const faces = THREE.ShapeUtils.triangulateShape(contour, [])

  for (const [a, b, c] of faces) {
    // +Z cap: the CCW outline seen from +Z is already outward-facing.
    const e = end.points
    indices.push(
      vert(e[a][0], e[a][1], half),
      vert(e[b][0], e[b][1], half),
      vert(e[c][0], e[c][1], half),
    )
    // -Z cap: same triangles, reversed, so they face -Z.
    const s = start.points
    indices.push(
      vert(s[c][0], s[c][1], -half),
      vert(s[b][0], s[b][1], -half),
      vert(s[a][0], s[a][1], -half),
    )
  }

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
  geometry.setIndex(indices)
  geometry.computeVertexNormals()
  geometry.computeBoundingBox()
  geometry.computeBoundingSphere()
  return geometry
}

/**
 * The world transform that places a connector mesh built in the station frame.
 * Matches `connectorOBB` exactly — same basis, same origin.
 */
export function connectorTransform(station) {
  const p = new THREE.Vector3(...station.frame.p)
  const q = new THREE.Vector3(...station.frame.q)
  const r = new THREE.Vector3(...station.frame.r)
  return {
    position: new THREE.Vector3(...station.mid),
    quaternion: new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(p, q, r)),
  }
}
