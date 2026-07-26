/**
 * tests/test-v3-collide.mjs — headless checks for core/v3/collide.js.
 *
 * Plain node script, no test framework, NO browser. Exits non-zero on any
 * failure.
 *   node tests/test-v3-collide.mjs
 *
 * WHY THIS SUITE EXISTS (V3_SPEC.md §5, item 3 — "Collisions")
 * v2 was a model of independent 2D column strips: panels could not intersect
 * each other, so collision detection was never built. v3 tiles one
 * doubly-curved surface with rigid panels that pitch, roll and yaw freely —
 * they absolutely CAN interpenetrate — so an exact OBB–OBB SAT test is
 * required. The two failure modes this suite exists to catch, in order of how
 * badly they'd hurt to regress:
 *
 *   1. A 6-axis (face-normals-only) SAT is INCOMPLETE — it can report two
 *      genuinely disjoint boxes as colliding, because the missing witness is
 *      one of the 9 edge-edge cross-product axes. §2 below builds that case
 *      by hand and diffs the real implementation against a deliberately
 *      naive 6-axis mutant on the SAME input, to prove the 9 extra axes are
 *      load-bearing rather than merely present.
 *   2. Normalizing a near-zero cross product (near-parallel edges) is the
 *      classic SAT NaN bug. §3 checks identical, 90°-rotated, and
 *      1e-9-rad-apart orientations produce no NaN and the correct verdict.
 *
 * Style matches tests/test-geometry.mjs and tests/test-form.mjs: checks
 * against CLOSED-FORM derivations worked out from the box geometry in
 * comments beside each assertion, not golden numbers, and a seeded PRNG
 * (never Math.random) so a failing scattered check always reproduces.
 *
 * A NOTE ON THE WORK ORDER'S EDGE-EDGE EXAMPLE: the brief that produced this
 * suite asked for a construction where "no face normal separates [two boxes]
 * but an edge cross-product does" and said to "assert obbOverlap returns
 * true." That combination is mathematically impossible: the separating axis
 * theorem is exact, so if ANY axis (face normal or edge cross-product) truly
 * separates two boxes, they ARE disjoint — obbOverlap must be false, full
 * stop, regardless of which axis found it or how many axes a lesser test
 * happened to check. A subset of the 15 axes can therefore only ever be too
 * PERMISSIVE (find no separator when one of the omitted axes would have),
 * never too strict — checking fewer axes cannot manufacture a separation
 * that isn't real. So "no face normal separates them, but an edge
 * cross-product does" necessarily means the TRUE (15-axis) verdict is
 * DISJOINT, and it is the 6-axis-only test that gets it wrong, reporting a
 * false OVERLAP. §2 implements it that way — obbOverlap asserted false on
 * the correct implementation, true (wrong) on the 6-axis mutant — since
 * that's the only version of this test that can actually pass against a
 * correct implementation. See the final report for this flagged explicitly.
 */

import * as THREE from 'three'
import { aabbOverlap, findCollisions, obbOverlap, obbPenetration } from '../src/core/v3/collide.js'

let passed = 0
const failures = []

function check(name, condition, detail = '') {
  if (condition) {
    passed++
  } else {
    failures.push(`${name}${detail ? ` — ${detail}` : ''}`)
    console.error(`FAIL  ${name}${detail ? ` — ${detail}` : ''}`)
  }
}

const near = (a, b, eps = 1e-6) => Math.abs(a - b) <= eps

/** Deterministic PRNG (mulberry32) — same idiom as tests/test-form.mjs, so a
 *  failing scattered check always reproduces with the same inputs. */
function mulberry32(seed) {
  let a = seed >>> 0
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** Axis-aligned OBB helper — quaternion [0,0,0,1] is the identity rotation. */
const axisAligned = (center, halfExtents) => ({ center, halfExtents, quaternion: [0, 0, 0, 1] })

// =============================================================================
// 1. Axis-aligned cases with hand-computable answers
// =============================================================================
{
  const A = axisAligned([0, 0, 0], [5, 5, 5])

  // --- clearly disjoint ------------------------------------------------------
  const farAway = axisAligned([100, 100, 100], [5, 5, 5])
  check('clearly disjoint boxes: obbOverlap false', obbOverlap(A, farAway) === false)
  check('clearly disjoint boxes: obbPenetration null', obbPenetration(A, farAway) === null)

  // --- clearly overlapping, closed-form depth on a single dominant axis ------
  // centers 8 apart on X, half-extents 5+5=10 on every axis ⇒ overlap only on
  // X (Y, Z centers coincide, so those axes overlap by the full 10).
  // overlap(X) = (ra+rb) − |distance| = (5+5) − 8 = 2.
  const B8 = axisAligned([8, 0, 0], [5, 5, 5])
  const pen8 = obbPenetration(A, B8)
  check(
    'clearly overlapping (8 apart, half 5+5): depth = 10 − 8 = 2, axis = +X',
    pen8 !== null && near(pen8.depthCm, 2) && near(pen8.axis[0], 1) && near(pen8.axis[1], 0) && near(pen8.axis[2], 0),
    JSON.stringify(pen8),
  )
  check('obbOverlap agrees with obbPenetration on the overlapping case', obbOverlap(A, B8) === true)

  // --- exactly touching: boundary case, NOT reported as overlap --------------
  // centers 10 apart on X, half 5+5=10 ⇒ overlap(X) = 10 − 10 = 0 exactly
  // (both operands are exact integers in float64, so this is exact, not
  // approximate). See collide.js's "BOUNDARY RULE": a zero-depth boundary
  // contact is treated the same as a separating axis.
  const touching = axisAligned([10, 0, 0], [5, 5, 5])
  check('exactly touching (gap = 0 to the cm): obbOverlap false', obbOverlap(A, touching) === false)
  check('exactly touching: obbPenetration null (zero depth is not interpenetration)', obbPenetration(A, touching) === null)

  // --- overlapping on two axes, separated on the third ------------------------
  // X: distance 8 < 10 (overlap). Y: distance 8 < 10 (overlap).
  // Z: distance 20, half 5+5=10, 20 > 10 ⇒ separated on Z alone.
  const sep3rd = axisAligned([8, 8, 20], [5, 5, 5])
  check(
    'overlapping on X and Y but cleanly separated on Z: obbOverlap false',
    obbOverlap(A, sep3rd) === false,
  )
  check('same case: obbPenetration null', obbPenetration(A, sep3rd) === null)

  // --- penetration depth closed form, unequal extents + off-axis offset ------
  // A: half (3,2,4) at origin. C: half (4,1,1) at (5, 1.5, 2).
  //   overlap(X) = (3+4) − |5|   = 2
  //   overlap(Y) = (2+1) − |1.5| = 1.5   ← unique minimum, so the tie-break
  //   overlap(Z) = (4+1) − |2|   = 3       order of same-value axes never
  //                                         enters into this assertion
  const A2 = axisAligned([0, 0, 0], [3, 2, 4])
  const C = axisAligned([5, 1.5, 2], [4, 1, 1])
  const penC = obbPenetration(A2, C)
  check(
    'unequal-extent, off-axis-offset case: depth = min(2, 1.5, 3) = 1.5 on +Y',
    penC !== null && near(penC.depthCm, 1.5) && near(penC.axis[0], 0) && near(penC.axis[1], 1) && near(penC.axis[2], 0),
    JSON.stringify(penC),
  )
}

// =============================================================================
// 2. The edge-edge case 6-axis SAT gets wrong (see header comment)
// =============================================================================
{
  // Two long thin "rods", half-extents (20, 0.5, 0.5) — long along each box's
  // own local X. Rod A is axis-aligned (local X = world X). Rod B's local
  // frame is the orthonormal triple below (all thirds — hand-verifiable):
  //   bx = ( 1, 2, 2)/3     ("long" axis, half 20)
  //   by = ( 2, 1,-2)/3     (half 0.5)
  //   bz = (-2, 2,-1)/3     (half 0.5) — bx × by = bz, confirmed below, so
  //                          this is a proper right-handed rotation.
  // bx is a genuine 3D tilt (nonzero on all three world axes), unlike a
  // single-axis rotation, so cross(A.localX, B.localX) is NOT coincident
  // with any face normal of either box — it's a real 9th-axis-family witness.
  const bx = new THREE.Vector3(1 / 3, 2 / 3, 2 / 3)
  const by = new THREE.Vector3(2 / 3, 1 / 3, -2 / 3)
  const bz = new THREE.Vector3(-2 / 3, 2 / 3, -1 / 3)
  check(
    "B's hand-picked local frame is exactly orthonormal (bx·bx=by·by=bz·bz=1, all cross-dots=0)",
    near(bx.lengthSq(), 1) && near(by.lengthSq(), 1) && near(bz.lengthSq(), 1) &&
      near(bx.dot(by), 0) && near(bx.dot(bz), 0) && near(by.dot(bz), 0),
  )
  check(
    'and right-handed: bx × by = bz',
    new THREE.Vector3().crossVectors(bx, by).distanceTo(bz) < 1e-12,
  )

  const rodA = { center: [0, 0, 0], halfExtents: [20, 0.5, 0.5], quaternion: [0, 0, 0, 1] }
  const basis = new THREE.Matrix4().makeBasis(bx, by, bz)
  const quatB = new THREE.Quaternion().setFromRotationMatrix(basis)

  // L = cross(A.localX, B.localX) = cross((1,0,0), bx) = (0, -2/3, 2/3),
  // unit form (0, -1, 1)/√2 (length = √(8/9) = 2√2/3).
  // Radius of A on L:  20·|L·(1,0,0)| + 0.5·|L·(0,1,0)| + 0.5·|L·(0,0,1)|
  //                  = 0 + 0.5/√2 + 0.5/√2 = 1/√2
  // Radius of B on L:  20·|L·bx| + 0.5·|L·by| + 0.5·|L·bz|
  //   L·bx = (0·1 −1·2 +1·2)/(3√2) = 0
  //   L·by = (0·2 −1·1 −1·2)/(3√2) = -1/√2   →  |·| = 1/√2
  //   L·bz = (0·-2 −1·2 −1·1)/(3√2) = -1/√2  →  |·| = 1/√2
  //                  = 0 + 0.5/√2 + 0.5/√2 = 1/√2
  // ra + rb = 2/√2 = √2 ≈ 1.41421356
  const invSqrt2 = 1 / Math.sqrt(2)
  const raOnL = invSqrt2
  const rbOnL = invSqrt2
  const sumRadiiOnL = raOnL + rbOnL
  check('hand-derived ra + rb on the critical edge axis L = √2', near(sumRadiiOnL, Math.sqrt(2)))

  // Place B's center at k·L with k = 2 > √2, so T·L = k (L is unit) = 2,
  // strictly greater than ra+rb ≈ 1.41421356 ⇒ L separates A and B — they
  // are DISJOINT. (Center offset is pure-L, so distance on every OTHER axis
  // reduces to a plain dot product against a fixed vector — verified against
  // the 6-axis-only mutant below rather than re-derived by hand for all 6.)
  const k = 2
  const centerB = [0, -k * invSqrt2, k * invSqrt2]
  const rodB = { center: centerB, halfExtents: [20, 0.5, 0.5], quaternion: quatB.toArray() }

  const distanceOnL = k // T·L with T = k·L, |L| = 1
  check('T·L = k = 2 exceeds ra+rb = √2 ⇒ L is a genuine separating axis', distanceOnL > sumRadiiOnL)

  check(
    'CORRECT 15-axis obbOverlap: rods crossing near each other are DISJOINT (false)',
    obbOverlap(rodA, rodB) === false,
  )
  check('same pair: obbPenetration is null', obbPenetration(rodA, rodB) === null)

  // --- mutation check: a 6-axis-only (face normals only) SAT on the SAME
  // input gets this wrong. Written from scratch here, deliberately mirroring
  // collide.js's projection formula but WITHOUT the 9 edge cross-products, to
  // prove those 9 axes are load-bearing and not just present-but-unused.
  function localAxesOf(quaternion) {
    const q = new THREE.Quaternion(...quaternion)
    return [
      new THREE.Vector3(1, 0, 0).applyQuaternion(q),
      new THREE.Vector3(0, 1, 0).applyQuaternion(q),
      new THREE.Vector3(0, 0, 1).applyQuaternion(q),
    ]
  }
  function radiusOn(halfExtents, axes, axis) {
    return halfExtents[0] * Math.abs(axes[0].dot(axis)) + halfExtents[1] * Math.abs(axes[1].dot(axis)) + halfExtents[2] * Math.abs(axes[2].dot(axis))
  }
  function sixAxisOnlyOverlap(a, b) {
    const T = new THREE.Vector3(...b.center).sub(new THREE.Vector3(...a.center))
    const axesA = localAxesOf(a.quaternion)
    const axesB = localAxesOf(b.quaternion)
    for (const axis of [...axesA, ...axesB]) {
      const ra = radiusOn(a.halfExtents, axesA, axis)
      const rb = radiusOn(b.halfExtents, axesB, axis)
      if (Math.abs(T.dot(axis)) > ra + rb) return false // a face normal found a real separation
    }
    return true // no FACE-NORMAL axis found one — the naive verdict, right or wrong
  }
  const sixAxisVerdict = sixAxisOnlyOverlap(rodA, rodB)
  check(
    'MUTATION CHECK: 6-axis-only (face normals only) SAT wrongly reports these DISJOINT rods as OVERLAPPING',
    sixAxisVerdict === true,
    `6-axis verdict = ${sixAxisVerdict}, correct 15-axis verdict = ${obbOverlap(rodA, rodB)}`,
  )
  check(
    'the 6-axis mutant and the real 15-axis implementation DISAGREE on this pair — proof the 9 edge axes are load-bearing',
    sixAxisVerdict !== obbOverlap(rodA, rodB),
  )
}

// =============================================================================
// 3. Degenerate axes: identical, 90°-rotated, and near-parallel orientations
// =============================================================================
{
  const A = axisAligned([0, 0, 0], [5, 5, 5])

  // --- identical orientation, offset so it genuinely overlaps ---------------
  // Every one of A's and B's local axes coincide (parallel), so all 9
  // cross-products are exactly zero and must ALL be skipped — this is the
  // maximally-degenerate case (9 of 15 candidate axes gone) and the
  // remaining 6 (really 3 distinct directions, tested twice) must still give
  // the right, finite answer: overlap(X) = 10 − 1 = 9.
  const identical = axisAligned([1, 0, 0], [5, 5, 5])
  const penIdentical = obbPenetration(A, identical)
  check(
    'identical orientation: all 9 cross-product axes degenerate to zero and are skipped, yet depth is still exact (9)',
    penIdentical !== null && near(penIdentical.depthCm, 9) && penIdentical.axis.every(Number.isFinite),
    JSON.stringify(penIdentical),
  )

  // --- 90°-rotated: cross-products collapse onto face normals (redundant,
  // not missing) — also must be finite and correct.
  const q90 = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI / 2)
  const rotated90 = { center: [1, 0, 0], halfExtents: [5, 5, 5], quaternion: q90.toArray() }
  const pen90 = obbPenetration(A, rotated90)
  check(
    '90°-rotated pair: no NaN, depth still exact (9) — cross-products here coincide with face normals, not missing info',
    pen90 !== null && near(pen90.depthCm, 9) && pen90.axis.every(Number.isFinite),
    JSON.stringify(pen90),
  )
  check('90°-rotated pair: obbOverlap agrees (true)', obbOverlap(A, rotated90) === true)

  // --- near-parallel: 1e-9 rad apart — the exact scenario the epsilon guard
  // exists for. Cross products between corresponding axes are ~1e-9 long;
  // normalizing that without a guard is the classic NaN/garbage-direction
  // bug. Must still produce a finite, correct-to-the-perturbation answer.
  const qTiny = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), 1e-9)
  const nearParallel = { center: [1, 0, 0], halfExtents: [5, 5, 5], quaternion: qTiny.toArray() }
  const penTiny = obbPenetration(A, nearParallel)
  const noNaN = penTiny !== null && Number.isFinite(penTiny.depthCm) && penTiny.axis.every(Number.isFinite)
  check(
    'near-parallel (1e-9 rad apart): no NaN escapes',
    noNaN,
    JSON.stringify(penTiny),
  )
  check(
    'near-parallel (1e-9 rad apart): depth is within a hair of the identical-orientation answer (9)',
    noNaN && Math.abs(penTiny.depthCm - 9) < 1e-6,
    JSON.stringify(penTiny),
  )
  check('near-parallel pair: obbOverlap is true, matching the identical-orientation case', obbOverlap(A, nearParallel) === true)

  // --- also confirm a DISJOINT near-parallel pair stays disjoint (no
  // spurious flip caused by a bad degenerate axis) ---------------------------
  const nearParallelFar = { center: [100, 0, 0], halfExtents: [5, 5, 5], quaternion: qTiny.toArray() }
  check(
    'near-parallel pair placed far apart: still correctly disjoint, no NaN-induced false positive',
    obbOverlap(A, nearParallelFar) === false && obbPenetration(A, nearParallelFar) === null,
  )
}

// =============================================================================
// 4. Rotation invariance — same verdict under a common rigid transform
// =============================================================================
{
  const collidingA = axisAligned([0, 0, 0], [5, 5, 5])
  const collidingB = axisAligned([8, 0, 0], [5, 5, 5]) // overlap depth 2, see §1
  const disjointA = axisAligned([0, 0, 0], [5, 5, 5])
  const disjointB = axisAligned([100, 100, 100], [5, 5, 5])
  const baseCollidingVerdict = obbOverlap(collidingA, collidingB)
  const baseDisjointVerdict = obbOverlap(disjointA, disjointB)
  check('sanity: the "colliding" fixture pair actually collides', baseCollidingVerdict === true)
  check('sanity: the "disjoint" fixture pair is actually disjoint', baseDisjointVerdict === false)

  function applyRigidTransform(box, R, t) {
    const center = new THREE.Vector3(...box.center).applyQuaternion(R).add(t)
    const quaternion = R.clone().multiply(new THREE.Quaternion(...box.quaternion))
    return { center: center.toArray(), halfExtents: box.halfExtents, quaternion: quaternion.toArray() }
  }

  const rng = mulberry32(20260101)
  const TRIALS = 200
  let collidingMismatches = 0
  let disjointMismatches = 0
  for (let i = 0; i < TRIALS; i++) {
    const rotAxis = new THREE.Vector3(rng() - 0.5, rng() - 0.5, rng() - 0.5).normalize()
    const rotAngle = rng() * Math.PI * 2
    const R = new THREE.Quaternion().setFromAxisAngle(rotAxis, rotAngle)
    const t = new THREE.Vector3((rng() - 0.5) * 200, (rng() - 0.5) * 200, (rng() - 0.5) * 200)

    const ta = applyRigidTransform(collidingA, R, t)
    const tb = applyRigidTransform(collidingB, R, t)
    if (obbOverlap(ta, tb) !== baseCollidingVerdict) collidingMismatches++

    const da = applyRigidTransform(disjointA, R, t)
    const db = applyRigidTransform(disjointB, R, t)
    if (obbOverlap(da, db) !== baseDisjointVerdict) disjointMismatches++
  }
  check(
    `rotation invariance holds for the colliding pair across ${TRIALS} random seeded rigid transforms`,
    collidingMismatches === 0,
    `${collidingMismatches}/${TRIALS} mismatches`,
  )
  check(
    `rotation invariance holds for the disjoint pair across ${TRIALS} random seeded rigid transforms`,
    disjointMismatches === 0,
    `${disjointMismatches}/${TRIALS} mismatches`,
  )
}

// =============================================================================
// 5. Symmetry — obbOverlap(a,b) === obbOverlap(b,a), depths agree both ways
// =============================================================================
{
  function randomBox(rng, scale = 20) {
    const center = [(rng() - 0.5) * scale, (rng() - 0.5) * scale, (rng() - 0.5) * scale]
    const halfExtents = [1 + rng() * 5, 1 + rng() * 5, 1 + rng() * 5]
    const axis = new THREE.Vector3(rng() - 0.5, rng() - 0.5, rng() - 0.5).normalize()
    const angle = rng() * Math.PI * 2
    const quaternion = new THREE.Quaternion().setFromAxisAngle(axis, angle).toArray()
    return { center, halfExtents, quaternion }
  }

  const rng = mulberry32(424242)
  const TRIALS = 2000
  let overlapMismatches = 0
  let depthMismatches = 0
  for (let i = 0; i < TRIALS; i++) {
    const a = randomBox(rng)
    const b = randomBox(rng)
    if (obbOverlap(a, b) !== obbOverlap(b, a)) overlapMismatches++

    const penAB = obbPenetration(a, b)
    const penBA = obbPenetration(b, a)
    if ((penAB === null) !== (penBA === null)) {
      depthMismatches++
    } else if (penAB !== null && Math.abs(penAB.depthCm - penBA.depthCm) > 1e-6) {
      depthMismatches++
    }
  }
  check(
    `obbOverlap(a,b) === obbOverlap(b,a) across ${TRIALS} random seeded pairs`,
    overlapMismatches === 0,
    `${overlapMismatches}/${TRIALS} mismatches`,
  )
  check(
    `obbPenetration depth agrees both ways (within 1e-6cm) across ${TRIALS} random seeded pairs`,
    depthMismatches === 0,
    `${depthMismatches}/${TRIALS} mismatches`,
  )
}

// =============================================================================
// 6. Panel-shaped boxes — the real solids, at realistic dihedral angles
// =============================================================================
{
  // PANEL_PROFILE.overallThickness = 3.7cm (src/config.js) ⇒ half-thickness
  // 1.85cm. The two real panel footprints from the work order: 60×60 and
  // 60×121. halfExtents are [width/2, thickness/2, depth/2] with the box's
  // local X the hinge-perpendicular in-plane direction, Y the thickness.
  //
  // Two panels sharing a hinge line parallel to Z, gap cm apart along X when
  // both lie flat, panel B tilted by thetaDeg about that hinge (world Z
  // through the hinge point) — a plate seesawing up off the floor at a small
  // dihedral angle, exactly like two drift tiles meeting at a fold.
  //
  // CLOSED FORM (world-X face-normal projection, A's own local X):
  //   hinge at x = 30 + gap.  B's flat center would sit at hinge + 30 (its
  //   own half-width), so its ACTUAL center after rotating by theta about
  //   the hinge line is (hinge + 30·cosθ, 30·sinθ, 0), and its local X/Y axes
  //   become (cosθ, sinθ, 0) / (−sinθ, cosθ, 0).
  //   ra (A onto world X)  = 30            (A axis-aligned, half-width 30)
  //   rb (B onto world X)  = 30·|cosθ| + 1.85·|sinθ|     (depth term is 0 —
  //                           B's local Z stays world Z, ⊥ to world X)
  //   distance             = |hinge + 30·cosθ| = 30 + gap + 30·cosθ
  //   overlap(worldX)      = (ra+rb) − distance
  //                         = (30 + 30·cosθ + 1.85·sinθ) − (30 + gap + 30·cosθ)
  //                         = 1.85·sinθ − gap
  //   This is negative (separating) whenever gap > 1.85·sinθ — i.e. for ANY
  //   θ < arcsin(gap/1.85). With gap = 1cm, that's θ < arcsin(1/1.85) ≈ 32.7°,
  //   comfortably clear of any real dihedral angle this tool produces.
  function panelPair(widthCm, depthCm, thetaDeg, gapCm) {
    const halfThickness = 1.85
    const half = [widthCm / 2, halfThickness, depthCm / 2]
    const A = { center: [0, 0, 0], halfExtents: half, quaternion: [0, 0, 0, 1] }
    const theta = (thetaDeg * Math.PI) / 180
    const hingeX = widthCm / 2 + gapCm
    const armLength = widthCm / 2 // B's own half-width, its pivot arm
    const center = [hingeX + armLength * Math.cos(theta), armLength * Math.sin(theta), 0]
    const quaternion = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), theta).toArray()
    const B = { center, halfExtents: half, quaternion }
    return { A, B }
  }

  const margin = (gapCm, thetaDeg) => gapCm - 1.85 * Math.sin((thetaDeg * Math.PI) / 180)

  for (const [widthCm, depthCm, label] of [[60, 60, '60×3.7×60'], [60, 121, '60×3.7×121']]) {
    for (const thetaDeg of [0, 2, 5, 8]) {
      const gapCm = 1.0
      const { A, B } = panelPair(widthCm, depthCm, thetaDeg, gapCm)
      const m = margin(gapCm, thetaDeg)
      check(
        `${label} panels, θ=${thetaDeg}°, ${gapCm}cm gap: closed-form world-X margin ${m.toFixed(4)}cm is positive (should not collide)`,
        m > 0,
      )
      check(
        `${label} panels, θ=${thetaDeg}°, ${gapCm}cm gap: obbOverlap correctly false (a false positive here would make the tool useless)`,
        obbOverlap(A, B) === false,
      )
      check(`${label} panels, θ=${thetaDeg}°: obbPenetration null`, obbPenetration(A, B) === null)
      // The gap genuinely separates these along a direction close to world
      // X, so the AABB broad phase is entitled to reject the pair outright
      // (that's it doing its job, not a bug — broad phase only promises
      // never to reject a pair that DOES overlap, checked generically in §7).
      // What matters here is that the full pipeline agrees end-to-end:
      // findCollisions must report nothing for this pair either.
      check(
        `${label} panels, θ=${thetaDeg}°: findCollisions (broad+narrow phase together) reports no collision`,
        findCollisions([A, B]).length === 0,
      )
    }
  }

  // --- negative control: remove the gap entirely (flush, θ=0) → exact touch,
  // still not a collision (boundary rule, §1) ---------------------------------
  {
    const { A, B } = panelPair(60, 60, 0, 0)
    check('panels flush with zero gap (exact touch): obbOverlap false', obbOverlap(A, B) === false)
  }

  // --- negative control: push them into each other (negative gap) → this
  // MUST register as a genuine collision, or the "no false positive" checks
  // above would be vacuous (an implementation that always returns false
  // would also pass them) ------------------------------------------------------
  {
    const { A, B } = panelPair(60, 60, 0, -1) // 1cm of forced interpenetration
    const pen = obbPenetration(A, B)
    check(
      'panels pushed 1cm past flush: DOES register as a collision, depth = 1cm exactly (closed form: overlap(worldX) = 1.85·sin0 − (−1) = 1)',
      pen !== null && near(pen.depthCm, 1),
      JSON.stringify(pen),
    )
  }
}

// =============================================================================
// 7. aabbOverlap — rotated extent, not raw halfExtents
// =============================================================================
{
  // A 45°-about-Z rotation of a half-extent-1 cube has WORLD-X and WORLD-Y
  // half-extents of |cos45°|·1 + |sin45°|·1 = √2 (its local axes are
  // (cos45,sin45,0) and (−sin45,cos45,0)), NOT 1. This is exactly the "use
  // the rotated extent, not halfExtents verbatim" requirement.
  const A = axisAligned([0, 0, 0], [1, 1, 1])
  const q45 = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), Math.PI / 4)
  const B = { center: [2.2, 0, 0], halfExtents: [1, 1, 1], quaternion: q45.toArray() }

  // Correct broad phase: combined world-X half-extent = 1 + √2 ≈ 2.41421356,
  // which exceeds the 2.2 center separation ⇒ must overlap.
  const correctSum = 1 + Math.sqrt(2)
  check('closed form: correct rotated-extent sum (1 + √2 ≈ 2.4142) exceeds the 2.2cm center separation', correctSum > 2.2)
  check(
    'aabbOverlap uses the ROTATED extent: reports overlap for a 45°-tilted box even though raw halfExtents (1+1=2 < 2.2) would have wrongly rejected it',
    aabbOverlap(A, B) === true,
  )

  // A naive "use halfExtents verbatim, ignore rotation" broad phase would
  // have concluded 2 < 2.2 ⇒ no overlap — demonstrably the wrong call, since
  // this pair really does overlap (its true, unrotated AABBs are what would
  // have wrongly rejected it; confirm that arithmetic explicitly):
  const naiveSum = A.halfExtents[0] + B.halfExtents[0]
  check('the NAIVE (unrotated) sum (2) is less than the 2.2cm separation — proof the rotated computation is the one doing real work here', naiveSum < 2.2)

  // --- broad-phase soundness: it must never reject a pair the narrow phase
  // (SAT) calls overlapping, over a wide seeded sweep -------------------------
  function randomBox(rng, scale = 20) {
    const center = [(rng() - 0.5) * scale, (rng() - 0.5) * scale, (rng() - 0.5) * scale]
    const halfExtents = [1 + rng() * 5, 1 + rng() * 5, 1 + rng() * 5]
    const axis = new THREE.Vector3(rng() - 0.5, rng() - 0.5, rng() - 0.5).normalize()
    const angle = rng() * Math.PI * 2
    const quaternion = new THREE.Quaternion().setFromAxisAngle(axis, angle).toArray()
    return { center, halfExtents, quaternion }
  }
  const rng = mulberry32(7)
  let unsoundRejections = 0
  for (let i = 0; i < 2000; i++) {
    const a = randomBox(rng)
    const b = randomBox(rng)
    if (obbOverlap(a, b) && !aabbOverlap(a, b)) unsoundRejections++
  }
  check(
    'broad-phase soundness: aabbOverlap never rejects a pair obbOverlap calls true, across 2000 seeded pairs',
    unsoundRejections === 0,
    `${unsoundRejections}/2000 unsound rejections`,
  )
}

// =============================================================================
// 8. findCollisions — minDepthCm filtering, determinism, determinstic order
// =============================================================================
{
  // Box 0 and 1: depth 2 (§1). Box 0 and 3: depth 1.8. Box 1 and 3: depth 9.8.
  // Box 2 is far away from everything (no collisions involving it).
  const boxes = [
    axisAligned([0, 0, 0], [5, 5, 5]), // 0
    axisAligned([8, 0, 0], [5, 5, 5]), // 1
    axisAligned([100, 100, 100], [5, 5, 5]), // 2 — isolated
    axisAligned([8.2, 0, 0], [5, 5, 5]), // 3
  ]

  const all = findCollisions(boxes)
  check('findCollisions finds exactly the 3 true colliding pairs (0-1, 0-3, 1-3), none touching box 2', all.length === 3 && all.every((c) => c.i !== 2 && c.j !== 2), JSON.stringify(all))
  check('every reported pair has i < j', all.every((c) => c.i < c.j), JSON.stringify(all))
  check(
    'pairs are in deterministic ascending (i, j) order',
    all.every((c, idx) => idx === 0 || all[idx - 1].i < c.i || (all[idx - 1].i === c.i && all[idx - 1].j < c.j)),
    JSON.stringify(all),
  )

  // --- minDepthCm filtering ----------------------------------------------------
  // Depths present: 2 (0-1), 1.8 (0-3), 9.8 (1-3). A threshold of 1.9 should
  // drop the 1.8 pair (0-3) and keep the other two (strict '>').
  const filtered = findCollisions(boxes, { minDepthCm: 1.9 })
  check(
    'minDepthCm = 1.9 keeps the two pairs deeper than 1.9cm (2, 9.8) and drops the 1.8cm pair',
    filtered.length === 2 && filtered.every((c) => c.depthCm > 1.9) && !filtered.some((c) => c.i === 0 && c.j === 3),
    JSON.stringify(filtered),
  )
  const filteredHigh = findCollisions(boxes, { minDepthCm: 100 })
  check('minDepthCm above every real depth returns nothing', filteredHigh.length === 0, JSON.stringify(filteredHigh))
  const filteredZero = findCollisions(boxes, { minDepthCm: 0 })
  check('minDepthCm = 0 (default) is a no-op beyond obbPenetration\'s own depth>0 requirement — same 3 pairs as unfiltered', filteredZero.length === 3)

  // --- determinism: repeat calls are byte-identical (README's "pure and
  // deterministic ... several tests assert this by JSON.stringify equality") -
  const again = findCollisions(boxes)
  check('findCollisions is deterministic: repeat call on the same input is byte-identical JSON', JSON.stringify(all) === JSON.stringify(again))

  // --- purity of the narrow-phase primitives themselves -----------------------
  const A = axisAligned([0, 0, 0], [5, 5, 5])
  const Bx = axisAligned([8, 0, 0], [5, 5, 5])
  check(
    'obbPenetration is pure: repeat call on the same input is byte-identical JSON',
    JSON.stringify(obbPenetration(A, Bx)) === JSON.stringify(obbPenetration(A, Bx)),
  )
}

// =============================================================================
// Summary
// =============================================================================
console.log('\n=== test-v3-collide ===')
if (failures.length === 0) {
  console.log(`PASS — ${passed}/${passed} checks`)
  process.exit(0)
} else {
  console.log(`FAIL — ${failures.length}/${passed + failures.length} check(s) failed:`)
  for (const f of failures) console.log(`  - ${f}`)
  process.exit(1)
}
