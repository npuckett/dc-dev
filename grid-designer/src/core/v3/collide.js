/**
 * grid-designer v3 — oriented-bounding-box collision detection.
 *
 * HEADLESS ZONE (src/core/): pure functions, importable from plain node.
 *   - explicit `.js` extensions on ALL relative imports (none needed here — this
 *     file has no relative imports at all, only `three`)
 *   - imports `three`'s math classes only; never a scene graph, the DOM, or the store
 *   - pure and deterministic: same boxes in, byte-identical output
 *
 * =============================================================================
 * WHAT THIS IS — V3_SPEC.md §5, item 3 ("Collisions")
 * =============================================================================
 * v2 was a model of independent 2D column strips: panels could not possibly
 * intersect each other, so collision detection was never built. v3 tiles one
 * doubly-curved 3D surface with rigid panels that pitch, roll and yaw freely —
 * they absolutely CAN interpenetrate, so detecting it is required, not optional
 * (V3_SPEC.md §5: "v2 listed this as 'not built' ... In v3 they absolutely can").
 *
 * An OBB (oriented bounding box) is:
 *   { center: [x,y,z], halfExtents: [hx,hy,hz], quaternion: [x,y,z,w] }
 * `halfExtents` are measured along the box's OWN local axes: applying
 * `quaternion` to the world unit vectors (1,0,0), (0,1,0), (0,0,1) gives the
 * box's local X/Y/Z axes in world space, and the box extends `halfExtents[i]`
 * either side of `center` along local axis `i`. `quaternion` need not be
 * pre-normalized by the caller — every entry point here normalizes it.
 *
 * =============================================================================
 * THE ALGORITHM — exact 15-axis SAT (separating axis theorem)
 * =============================================================================
 * Two convex polytopes are disjoint if and only if some axis in a specific
 * finite candidate set separates them (their projections onto that axis don't
 * overlap). For a pair of boxes the complete candidate set has 15 axes:
 *   - the 3 face normals of A (its own local X/Y/Z axes)
 *   - the 3 face normals of B
 *   - the 9 pairwise cross products Ai × Bj of A's and B's local axes (these
 *     are the box EDGE directions; Ai × Bj is normal to one edge of A and one
 *     edge of B at once — the axis an edge-edge contact needs)
 *
 * A 6-axis test (face normals only) is INCOMPLETE: it can only ever be too
 * permissive, never too strict — any axis that truly separates the boxes is
 * valid evidence of separation no matter which of the 15 you found it on, so
 * checking fewer axes can never manufacture a false "separated" verdict, only
 * miss a true one. Concretely: two long thin boxes ("rods") crossing near each
 * other, offset just enough that neither rod's face normals catch the miss,
 * are reported as OVERLAPPING by a 6-axis test even when they are genuinely
 * disjoint — the missing witness is one of the 9 edge cross-products. This is
 * exactly the case `tests/test-v3-collide.mjs` builds by hand and checks
 * against a stripped 6-axis-only implementation to prove the 9 extra axes are
 * load-bearing, not decorative.
 *
 * For each axis L (unit length), with T = B.center − A.center:
 *   ra = Σ halfExtentsA[i] · |axisA[i] · L|      (A's projected half-width)
 *   rb = Σ halfExtentsB[i] · |axisB[i] · L|      (B's projected half-width)
 *   overlap(L) = (ra + rb) − |T · L|
 * If overlap(L) ≤ 0 for ANY axis, that axis separates A and B — the boxes are
 * disjoint (or exactly touching, a zero-volume boundary contact, which we also
 * treat as "not overlapping": see `obbPenetration`'s doc comment). Otherwise
 * every axis reports positive overlap and the boxes genuinely interpenetrate;
 * the true penetration depth is the MINIMUM overlap over all 15 axes (the
 * standard OBB minimum-translation-vector construction), and its axis is the
 * direction along which separating the boxes takes the least motion.
 *
 * DEGENERATE CROSS-PRODUCTS — the classic SAT bug, and why it matters here:
 * axisA[i] and axisB[j] are unit vectors, so |axisA[i] × axisB[j]| = sin(θ)
 * where θ is the angle between them. When the two edges are (near) parallel,
 * θ → 0 and the cross product's length → 0. Normalizing a near-zero vector
 * divides by a near-zero length, which either raises 0/0 → NaN (exactly
 * parallel) or amplifies ordinary floating-point noise in axisA/axisB into a
 * "unit" vector pointing in a numerically meaningless direction (near
 * parallel) — a direction with no relation to the true local geometry. Either
 * way, using it as a separating-axis candidate is worse than useless: it can
 * fabricate a bogus separation (false "disjoint") or corrupt the
 * minimum-overlap search with garbage. We therefore SKIP any cross-product
 * axis whose squared length falls below `CROSS_AXIS_EPS_SQ` before ever
 * normalizing it. This loses no coverage: when two edges are (nearly)
 * parallel, whatever separating information they would have carried is
 * already carried by the ordinary face-normal axes of A and B (a parallel or
 * near-parallel pair of edges cannot be the UNIQUE witness of separation —
 * geometrically, the perpendicular-to-both-edges direction degenerates back
 * toward a face normal as θ → 0). `CROSS_AXIS_EPS_SQ = 1e-12` means we skip
 * once |cross| < 1e-6, i.e. θ < ~1e-6 rad (~0.00006°) — comfortably below any
 * real dihedral angle this tool will ever construct, but well above the
 * float64 noise floor, so exact-parallel (cross ≈ 0 to float64 rounding) and
 * "1e-9 rad apart" (cross ≈ 1e-9) are both safely skipped without ever
 * normalizing a near-zero vector, while a genuine few-degree joint skew
 * (cross ≈ sin(few°) ≈ 0.03–0.1) is comfortably clear of the threshold and
 * still tested.
 *
 * =============================================================================
 * EXPORTS
 * =============================================================================
 *   obbOverlap(a, b)                          → boolean
 *   obbPenetration(a, b)                      → null | { depthCm, axis }
 *   aabbOverlap(a, b)                         → boolean
 *   findCollisions(boxes, { minDepthCm = 0 }) → array of { i, j, depthCm, axis }
 */

import * as THREE from 'three'

/** Cross-product axes shorter than this (squared length) are skipped rather
 *  than normalized — see "DEGENERATE CROSS-PRODUCTS" above. */
const CROSS_AXIS_EPS_SQ = 1e-12

/**
 * The box's local X/Y/Z axes in world space: `quaternion` applied to the
 * world unit vectors. Orthonormal whenever `quaternion` is a valid rotation
 * (normalized here so a caller-supplied non-unit quaternion can't corrupt the
 * projections below).
 */
function localAxes(quaternion) {
  const q = new THREE.Quaternion(quaternion[0], quaternion[1], quaternion[2], quaternion[3]).normalize()
  return [
    new THREE.Vector3(1, 0, 0).applyQuaternion(q),
    new THREE.Vector3(0, 1, 0).applyQuaternion(q),
    new THREE.Vector3(0, 0, 1).applyQuaternion(q),
  ]
}

/** Σ halfExtents[i] · |axes[i] · L| — the box's projected half-width onto unit axis L. */
function projectedRadius(halfExtents, axes, axis) {
  return (
    halfExtents[0] * Math.abs(axes[0].dot(axis)) +
    halfExtents[1] * Math.abs(axes[1].dot(axis)) +
    halfExtents[2] * Math.abs(axes[2].dot(axis))
  )
}

/**
 * The 15 SAT candidate axes for boxes with local frames `axesA`/`axesB`: the
 * 6 face normals verbatim, then the 9 edge cross-products with the near-zero
 * ones skipped (see module doc comment). Every returned axis is unit length.
 */
function candidateAxes(axesA, axesB) {
  const axes = [axesA[0], axesA[1], axesA[2], axesB[0], axesB[1], axesB[2]]
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      const cross = new THREE.Vector3().crossVectors(axesA[i], axesB[j])
      const lengthSq = cross.lengthSq()
      if (lengthSq < CROSS_AXIS_EPS_SQ) continue // near-parallel edge pair — skip, don't normalize
      axes.push(cross.divideScalar(Math.sqrt(lengthSq)))
    }
  }
  return axes
}

/**
 * Exact oriented-bounding-box overlap test, 15-axis SAT (see module doc
 * comment). Boxes that exactly touch (zero-depth boundary contact — overlap
 * on every axis, but exactly zero on the tightest one) are NOT reported as
 * overlapping; see `obbPenetration` for the precise boundary rule.
 */
export function obbOverlap(a, b) {
  return obbPenetration(a, b) !== null
}

/**
 * Exact 15-axis SAT with penetration detail, for reporting.
 *
 * Returns `null` when the boxes are disjoint OR exactly touching (some axis
 * has overlap ≤ 0 — see below), else `{ depthCm, axis }`:
 *   - `depthCm` — the minimum-translation-vector depth: the smallest amount B
 *     would need to move along `axis` to no longer overlap A. It is the
 *     MINIMUM of the 15 axes' overlap amounts, each itself an exact distance
 *     in cm (every candidate axis is unit length), not an approximation.
 *   - `axis` — the unit vector realizing that minimum, oriented to point from
 *     A toward B (i.e. `axis · (B.center − A.center) ≥ 0`).
 *
 * BOUNDARY RULE: an axis with overlap exactly 0 (the boxes' extents meet with
 * no gap and no interpenetration, e.g. two panels sharing a joint plane with
 * zero gap) is treated the SAME as a separating axis — it makes this function
 * return `null`. Physically, a zero-volume shared boundary is not
 * interpenetration; this also keeps `obbOverlap`/`obbPenetration` consistent
 * with each other (an axis that makes one return false/null makes the other
 * too). Note this is independent of `findCollisions`'s `minDepthCm`, which
 * filters on top of this at the *pair* level once `depthCm` is already known
 * to be strictly positive — see its doc comment for why panels need that too.
 */
export function obbPenetration(a, b) {
  const centerA = new THREE.Vector3(a.center[0], a.center[1], a.center[2])
  const centerB = new THREE.Vector3(b.center[0], b.center[1], b.center[2])
  const axesA = localAxes(a.quaternion)
  const axesB = localAxes(b.quaternion)
  const T = centerB.clone().sub(centerA)

  let minOverlap = Infinity
  let mtvAxis = null

  for (const axis of candidateAxes(axesA, axesB)) {
    const ra = projectedRadius(a.halfExtents, axesA, axis)
    const rb = projectedRadius(b.halfExtents, axesB, axis)
    const distance = T.dot(axis)
    const overlap = ra + rb - Math.abs(distance)

    if (overlap <= 0) return null // separating (or exactly touching) axis found — disjoint

    if (overlap < minOverlap) {
      minOverlap = overlap
      mtvAxis = axis.clone()
      if (distance < 0) mtvAxis.negate() // orient from A toward B
    }
  }

  // Unreachable in practice: the 6 face-normal axes are always present and
  // always valid (never skipped — they come straight from a normalized
  // quaternion, never from a degenerate cross product), so at least one
  // candidate axis is always tested. Guarded anyway rather than trusting that.
  if (mtvAxis === null) return null

  return { depthCm: minOverlap, axis: [mtvAxis.x, mtvAxis.y, mtvAxis.z] }
}

/**
 * World-space half-extent of an OBB's rotated AABB along world axis `k`:
 * Σ halfExtents[i] · |axes[i][k]| — the same projection formula as the SAT
 * face-normal test, just evaluated against the 3 WORLD axes instead of the
 * box's own local axes. This is the standard rotated-extent formula; using
 * `halfExtents` directly (unrotated) would under-cover any tilted box and
 * make the broad phase unsound (it could reject a pair the narrow phase would
 * have found overlapping).
 */
function worldAabbHalfExtents(box) {
  const axes = localAxes(box.quaternion)
  return [
    Math.abs(axes[0].x) * box.halfExtents[0] + Math.abs(axes[1].x) * box.halfExtents[1] + Math.abs(axes[2].x) * box.halfExtents[2],
    Math.abs(axes[0].y) * box.halfExtents[0] + Math.abs(axes[1].y) * box.halfExtents[1] + Math.abs(axes[2].y) * box.halfExtents[2],
    Math.abs(axes[0].z) * box.halfExtents[0] + Math.abs(axes[1].z) * box.halfExtents[1] + Math.abs(axes[2].z) * box.halfExtents[2],
  ]
}

/**
 * Cheap AABB rejection, for broad-phase: true iff the boxes' WORLD-AXIS-
 * ALIGNED bounding boxes overlap (rotated extent, per `worldAabbHalfExtents`
 * — not `halfExtents` verbatim, which is only correct when a box is
 * unrotated). Inclusive (`<=`) at the boundary, so it never rejects a pair
 * that the exact narrow phase would call touching or overlapping — it is a
 * conservative filter, not a collision verdict on its own.
 */
export function aabbOverlap(a, b) {
  const halfA = worldAabbHalfExtents(a)
  const halfB = worldAabbHalfExtents(b)
  for (let k = 0; k < 3; k++) {
    const centerA = a.center[k]
    const centerB = b.center[k]
    if (Math.abs(centerB - centerA) > halfA[k] + halfB[k]) return false
  }
  return true
}

/**
 * Broad-phase (AABB) + narrow-phase (15-axis SAT) over a list of OBBs.
 * Returns every colliding pair once, as `{ i, j, depthCm, axis }` with
 * `i < j`, in deterministic order (ascending `i`, then ascending `j` — the
 * natural order of the nested `i < j` scan below; nothing here depends on
 * iteration order of an object or a Set).
 *
 * `minDepthCm` (default 0): a pair is only reported when `depthCm >
 * minDepthCm` (strict). `obbPenetration` already excludes depth ≤ 0 (disjoint
 * or exactly touching — see its doc comment), so `minDepthCm = 0` reports
 * every genuine, positive-depth interpenetration. Raise it to also ignore
 * GRAZING contact: panels in this design are meant to sit close across a
 * ~1cm joint, and floating-point noise or a hair-thin, intentional overlap at
 * a joint can otherwise register as a "collision" that would flag the whole
 * design as broken. `minDepthCm` is the tolerance a caller dials in for "this
 * much interpenetration is contact, not a defect" — it does NOT change the
 * disjoint/touching boundary rule in `obbPenetration`, only which already-
 * positive depths are worth surfacing.
 */
export function findCollisions(boxes, { minDepthCm = 0 } = {}) {
  const results = []
  for (let i = 0; i < boxes.length; i++) {
    for (let j = i + 1; j < boxes.length; j++) {
      if (!aabbOverlap(boxes[i], boxes[j])) continue
      const pen = obbPenetration(boxes[i], boxes[j])
      if (pen && pen.depthCm > minDepthCm) {
        results.push({ i, j, depthCm: pen.depthCm, axis: pen.axis })
      }
    }
  }
  return results
}
