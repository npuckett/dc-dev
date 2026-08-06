/**
 * grid-designer — the solid for one tracking camera.
 *
 * A small SPHERE (the camera body) with an ARC fanning out in the aim direction
 * — a vertical field-of-view wedge that shows which way the camera looks. Two
 * edge tubes from the body to the ends of the arc, plus the arc itself, drawn as
 * TubeGeometry so it is real mesh that renders AND exports (a Line would do
 * neither in a GLB).
 *
 * Returns an ARRAY of geometries in WORLD space, so the viewport can render them
 * as separate meshes and the exporter can merge them into one node — neither
 * needs a merge utility here. Placement (`position` / `direction`) comes from
 * `core/v4/schema.js`'s `trackingCameras()`; this file only builds the solid.
 */

import * as THREE from 'three'

const SPHERE_R_CM = 6
const FAN_LENGTH_CM = 45 // how far the sight lines reach
const FAN_HALF_DEG = 20 // half the vertical field of view the arc spans
const TUBE_R_CM = 1.2

function tube(points) {
  const curve = new THREE.CatmullRomCurve3(points)
  return new THREE.TubeGeometry(curve, Math.max(2, points.length * 2), TUBE_R_CM, 6, false)
}

/**
 * @param {{ position:[x,y,z], direction:[x,y,z] }} camera
 * @returns {THREE.BufferGeometry[]}
 */
export function buildCameraGeometries(camera) {
  const pos = new THREE.Vector3(...camera.position)
  const d = new THREE.Vector3(...camera.direction).normalize()

  // The fan opens in the plane containing the aim and world up, so its spread
  // axis is d × up (falls back to +X if the aim is vertical).
  const up = new THREE.Vector3(0, 1, 0)
  let axis = new THREE.Vector3().crossVectors(d, up)
  if (axis.lengthSq() < 1e-6) axis = new THREE.Vector3(1, 0, 0)
  axis.normalize()

  const half = (FAN_HALF_DEG * Math.PI) / 180
  const geos = []

  const sphere = new THREE.SphereGeometry(SPHERE_R_CM, 16, 12)
  sphere.translate(pos.x, pos.y, pos.z)
  geos.push(sphere)

  // Two edge sight lines, from the body to the ends of the arc.
  for (const sign of [-1, 1]) {
    const e = d.clone().applyAxisAngle(axis, sign * half)
    geos.push(tube([pos.clone(), pos.clone().addScaledVector(e, FAN_LENGTH_CM)]))
  }

  // The arc across the fan's mouth, at the sight-line radius.
  const arcPts = []
  const N = 14
  for (let i = 0; i <= N; i++) {
    const a = -half + 2 * half * (i / N)
    const e = d.clone().applyAxisAngle(axis, a)
    arcPts.push(pos.clone().addScaledVector(e, FAN_LENGTH_CM))
  }
  geos.push(tube(arcPts))

  return geos
}
