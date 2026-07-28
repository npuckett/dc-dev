/**
 * tests/test-v4-lattice.mjs — headless checks for core/v4/lattice.js.
 *
 * Plain node script, no test framework. Exits non-zero on any failure.
 *   node tests/test-v4-lattice.mjs
 *
 * Ported from test-v4-chain.mjs, which this replaces. Everything that file
 * asserted about the ribbon still has to be true, because the ribbon is the
 * `cols: 1` case — so §1 is a DIRECT COMPARISON against the shipped ribbon's own
 * hardcoded numbers, and it is the most important test in the suite. If the
 * generalisation changed the physics, that is where it shows.
 *
 * The rest are closed-form expectations rather than snapshots. A snapshot test
 * of a geometry module can only ever tell you it changed; it can never tell you
 * it was right.
 *
 * The three claims worth naming, all of which would fail silently:
 *
 *   THE SPAN. Every joint's rim-to-rim distance is exactly `gap`. §4 asserts it
 *        over a lattice big enough to carry all four ramp directions. If it ever
 *        fails, v4's premise is false and nothing downstream means anything.
 *   THE CLOSURE. §3 computes each ramp's far end from ITS OWN geometry and
 *        checks it lands on its high cell's edge less one gap step. §6 walks a
 *        whole cycle round a lattice corner and comes back to the same point.
 *        V4_SPEC §9.1 asserts these; the point of a test is not to assume them.
 *   HANDEDNESS. §5. A left-handed basis is invisible in the corner geometry of a
 *        square panel and silently corrupts every collision box — it made a
 *        perfectly flat v3 grid report 21 interpenetrating pairs (HANDOFF §6).
 */

import * as THREE from 'three'
import {
  solveLattice,
  latticeStep,
  latticeBounds,
  levelAt,
  ROLE_GLYPH,
} from '../src/core/v4/lattice.js'
import { DEFAULT_CONFIG, normalizeConfig } from '../src/core/v4/schema.js'
import { PANEL_DIMENSIONS, PANEL_PROFILE } from '../src/config.js'

let passed = 0
let failed = 0
const ok = (c, m) => { if (c) passed++; else { failed++; console.log(`  FAIL: ${m}`) } }
const near = (a, b, tol, m) => ok(Math.abs(a - b) <= tol, `${m} (got ${a}, want ${b} ±${tol})`)
const nearV = (a, b, tol, m) => {
  const d = Math.max(Math.abs(a[0] - b[0]), Math.abs(a[1] - b[1]), Math.abs(a[2] - b[2]))
  ok(d <= tol, `${m} (got [${a}], want [${b}], worst ${d})`)
}

const RAD = Math.PI / 180
const L = PANEL_DIMENSIONS['2x2'].height
const W = PANEL_DIMENSIONS['2x2'].width
const T = PANEL_PROFILE.overallThickness

const V = (a) => new THREE.Vector3(a[0], a[1], a[2])
/** A panel's local axes through its emitted quaternion. */
const axesOf = (p) => {
  const q = new THREE.Quaternion(...p.quaternion)
  return {
    X: new THREE.Vector3(1, 0, 0).applyQuaternion(q),
    Y: new THREE.Vector3(0, 1, 0).applyQuaternion(q),
    Z: new THREE.Vector3(0, 0, 1).applyQuaternion(q),
  }
}
const byId = (C) => new Map(C.panels.map((p) => [p.id, p]))
/** The plan direction a ramp rises along, from its own reported role and axis. */
const rampDir = (p) => {
  const s = p.role === 'fall' ? -1 : 1
  return p.axis === 'x' ? new THREE.Vector3(s, 0, 0) : new THREE.Vector3(0, 0, s)
}

/**
 * A tolerance that is arithmetic rather than slack.
 *
 * Panel records round at 1e-9 (lattice.js's `r`, WP1's rule kept). A quaternion
 * with four rounded components, applied to a unit vector, therefore carries up
 * to ~2e-9 of direction error — measured worst 2.1e-9 over the sweep below. The
 * checks these guard are about SIGNS: a left-handed basis is wrong by 2, eight
 * orders of magnitude clear of this.
 */
const QUAT_TOL = 1e-8

console.log('=== test-v4-lattice ===')

// -----------------------------------------------------------------------------
// 1. THE RIBBON IS REPRODUCED EXACTLY — the proof the physics survived.
//
// A `cols: 1, rows: 5` lattice is the shipped 9-unit ribbon, panel for panel:
// cells j = 0..4 are its units 1, 3, 5, 7, 9 (`_ - _ - _`) and the four z-edge
// ramps are its units 2, 4, 6, 8 (`/ \ / \`).
//
// The reference values are the ribbon's own, from V4_SPEC §9.1 and
// test-v4-chain.mjs: the cell pitch is 115.825227532 (`refStart` of the third
// panel) and the high level sits 31.035276180 above the base. Everything else
// follows from those two by closed form.
// -----------------------------------------------------------------------------
console.log('1. the cols:1 lattice IS the ribbon')
{
  const C = solveLattice({
    lattice: { cols: 1, rows: 5, panelType: '2x2' },
    angleDeg: 30,
    gap: 2,
    placement: { wallOffsetCm: 0, windowOffsetCm: 0, groundToFloor: false, wallAnchor: 'free' },
  })
  ok(C.panels.length === 9, `nine panels, exactly as the 9-unit ribbon (got ${C.panels.length})`)
  ok(C.joints.length === 8, `and eight joints (got ${C.joints.length})`)

  // THE TWO HARDCODED NUMBERS.
  near(C.lattice.pitchCm, 115.825227532, 1e-9, 'the cell pitch is the ribbon\'s third-panel refStart')
  near(C.lattice.riseCm, 31.03527618, 1e-9, 'and the high level sits 31.035276180 above the base')

  const c15 = Math.cos(15 * RAD)
  const s15 = Math.sin(15 * RAD)
  const c30 = Math.cos(30 * RAD)
  const s30 = Math.sin(30 * RAD)
  const P = 60 + 4 * c15 + 60 * c30
  const R = 4 * s15 + 30
  near(P, 115.825227532, 1e-9, 'P written out in closed form agrees')
  near(R, 31.03527618, 1e-9, 'and so does R')

  // The ribbon's own kinematics, rebuilt here from S_1 = (0,0) and the bisector
  // step — the SAME arithmetic chain.js ran, so this is a comparison against the
  // ribbon rather than against the lattice restated.
  const alpha = [0, 30, 0, -30, 0, 30, 0, -30, 0].map((a) => a * RAD)
  const u = alpha.map((a) => ({ z: Math.cos(a), y: Math.sin(a) }))
  const S = [{ z: 0, y: 0 }]
  const E = []
  for (let k = 0; k < 9; k++) {
    E[k] = { z: S[k].z + L * u[k].z, y: S[k].y + L * u[k].y }
    if (k + 1 < 9) {
      const wz = u[k].z + u[k + 1].z
      const wy = u[k].y + u[k + 1].y
      const wl = Math.hypot(wz, wy)
      S[k + 1] = { z: E[k].z + (2 * wz) / wl, y: E[k].y + (2 * wy) / wl }
    }
  }
  near(S[2].z, 115.825227532, 1e-9, 'the rebuilt ribbon puts unit 3 at 115.825227532')
  near(S[2].y, 31.03527618, 1e-9, 'at a height of 31.035276180')

  const M = byId(C)
  // unit 2j+1 is cell j; unit 2j+2 is the ramp on edge (0, j, 'z').
  const unitPanel = (k) => (k % 2 === 1 ? M.get(`Ci0j${(k - 1) / 2}`) : M.get(`Ei0j${(k - 2) / 2}z`))

  let worstPos = 0
  let worstSeg = 0
  for (let k = 1; k <= 9; k++) {
    const p = unitPanel(k)
    const ribbonMid = [30, (S[k - 1].y + E[k - 1].y) / 2, (S[k - 1].z + E[k - 1].z) / 2]
    worstPos = Math.max(worstPos, ...p.position.map((v, n) => Math.abs(v - ribbonMid[n])))
    // The reference SEGMENT is the same segment, though a descending ramp's runs
    // the other way — `ê` carries the direction, so there is no "fall" case.
    const ends = [p.refStart, p.refEnd]
    const want = [[30, S[k - 1].y, S[k - 1].z], [30, E[k - 1].y, E[k - 1].z]]
    const fwd = Math.max(...ends.flatMap((e, n) => e.map((v, m) => Math.abs(v - want[n][m]))))
    const rev = Math.max(...ends.flatMap((e, n) => e.map((v, m) => Math.abs(v - want[1 - n][m]))))
    worstSeg = Math.max(worstSeg, Math.min(fwd, rev))
  }
  ok(worstPos <= 1e-9, `every panel sits exactly where the ribbon put it (worst ${worstPos})`)
  ok(worstSeg <= 1e-9, `and spans exactly the ribbon's reference segment (worst ${worstSeg})`)

  // The decimals, written out, so a reader can check the arithmetic by eye.
  nearV(M.get('Ci0j0').refStart, [30, 0, 0], 1e-9, 'unit 1 starts at the window line')
  nearV(M.get('Ei0j0z').refStart, [30, 0.5176380902, 61.9318516526], 1e-9,
    'unit 2 (the first ramp) starts at the ribbon\'s S₂')
  nearV(M.get('Ci0j1').refStart, [30, 31.0352761804, 115.8252275322], 1e-9,
    'unit 3 starts at the ribbon\'s S₃ — the number V4_SPEC §9.1 quotes')
  nearV(M.get('Ci0j4').refEnd, [30, 0, 4 * P + 60], 1e-9, 'and unit 9 ends four pitches out')

  // Tilts: the flats are flat, and rise / fall are ±θ about the increasing axis.
  ok(C.panels.map((p) => p.tiltDeg).join() ===
     [0, 0, 0, 0, 0, 30, -30, 30, -30].join(),
    `tilts are 0 for cells and ±θ for ramps (got ${C.panels.map((p) => p.tiltDeg)})`)
  // Read in ribbon order, the glyphs are the trapezoid wave.
  const glyphs = Array.from({ length: 9 }, (_, k) => unitPanel(k + 1).glyph).join('')
  ok(glyphs === '_/-\\_/-\\_', `and the profile reads _ / - \\ _ / - \\ _ (got ${glyphs})`)

  // THE ONE HONEST DIFFERENCE. `ê` carries the direction, so a DESCENDING ramp's
  // width axis is the negative of the ribbon's: the same physical panel, turned
  // 180° in its own plane. Position, lit normal, corners and OBB are identical;
  // the quaternion differs by that half turn about the normal. Asserted here
  // rather than glossed, because "the quaternions differ" is exactly the kind of
  // thing that should never be discovered by a renderer.
  let same = 0
  let spun = 0
  for (let k = 1; k <= 9; k++) {
    const p = unitPanel(k)
    const a = axesOf(p)
    const ribbonQ = new THREE.Quaternion().setFromAxisAngle(
      new THREE.Vector3(1, 0, 0), -alpha[k - 1])
    const rX = new THREE.Vector3(1, 0, 0).applyQuaternion(ribbonQ)
    const rY = new THREE.Vector3(0, 1, 0).applyQuaternion(ribbonQ)
    const rZ = new THREE.Vector3(0, 0, 1).applyQuaternion(ribbonQ)
    if (a.X.distanceTo(rX) < QUAT_TOL && a.Y.distanceTo(rY) < QUAT_TOL &&
        a.Z.distanceTo(rZ) < QUAT_TOL) same++
    else if (a.X.distanceTo(rX.clone().negate()) < QUAT_TOL &&
             a.Y.distanceTo(rY) < QUAT_TOL &&
             a.Z.distanceTo(rZ.clone().negate()) < QUAT_TOL) spun++
  }
  ok(same === 7, `seven panels carry the ribbon's exact rotation (got ${same})`)
  ok(spun === 2, `and the two descending ramps carry it spun 180° about their own normal (got ${spun})`)
  ok(same + spun === 9, 'no panel is oriented any other way')

  // ...and the spin is invisible where it matters: same normal, same solid.
  for (const id of ['Ei0j1z', 'Ei0j3z']) {
    const p = M.get(id)
    const a = axesOf(p)
    nearV(p.normal, [a.Y.x, a.Y.y, a.Y.z], QUAT_TOL, `${id}: the emitted normal is local +Y`)
    ok(p.obb.halfExtents[0] === p.obb.halfExtents[2],
      `${id}: the box is square, so the spin cannot change it`)
  }

  // The grounding shift is the ribbon's too — at the clearance the ribbon had,
  // which is 0. The shipped default is now 15 (V4_SPEC §9.12), so the comparison
  // against the ribbon's hardcoded number has to ask for the ribbon's clearance.
  const grounded = solveLattice({
    lattice: { cols: 1, rows: 5 }, angleDeg: 30, gap: 2,
    placement: { groundClearanceCm: 0 },
  })
  near(grounded.lattice.shiftCm[1], T, 1e-9,
    'grounded at clearance 0, the shift is one housing thickness — the ribbon\'s answer')
  const lifted = solveLattice({
    lattice: { cols: 1, rows: 5 }, angleDeg: 30, gap: 2,
    placement: { groundClearanceCm: 15 },
  })
  near(lifted.lattice.shiftCm[1], T + 15, 1e-9,
    'and the clearance adds to it exactly, which is all it does')
  near(grounded.bounds.size[1], 35.13527618, 1e-8, 'and the ribbon stands 35.135cm tall')
  near(grounded.bounds.size[2], 523.300910129, 1e-8, 'and runs 523.3cm from the window')
}

// -----------------------------------------------------------------------------
// 2. PITCH AND RISE — closed form, and the cells really sit on them.
// -----------------------------------------------------------------------------
console.log('2. the pitch is uniform, and the same in x and z')
{
  for (const angleDeg of [0, 5, 17.5, 30, 45, 60, 75]) {
    for (const gap of [0.4, 1, 2, 3.7, 8]) {
      const theta = angleDeg * RAD
      const P = L + 2 * gap * Math.cos(theta / 2) + L * Math.cos(theta)
      const R = 2 * gap * Math.sin(theta / 2) + L * Math.sin(theta)
      const step = latticeStep({ gapCm: gap, angleDeg, lengthCm: L })
      near(step.pitchCm, P, 1e-9, `${angleDeg}°/${gap}cm: P = 60 + 2·gap·cos(θ/2) + 60·cos θ`)
      near(step.riseCm, R, 1e-9, `${angleDeg}°/${gap}cm: R = 2·gap·sin(θ/2) + 60·sin θ`)

      // ...and the cells actually sit on that pitch, in BOTH axes.
      const C = solveLattice({
        lattice: { cols: 4, rows: 4 }, angleDeg, gap,
        placement: { wallOffsetCm: 0, windowOffsetCm: 0, groundToFloor: false, wallAnchor: 'free' },
      })
      const M = byId(C)
      let worstX = 0
      let worstZ = 0
      let worstY = 0
      for (let i = 0; i < 4; i++) {
        for (let j = 0; j < 4; j++) {
          const p = M.get(`Ci${i}j${j}`)
          worstX = Math.max(worstX, Math.abs(p.position[0] - (L / 2 + i * P)))
          worstZ = Math.max(worstZ, Math.abs(p.position[2] - (L / 2 + j * P)))
          worstY = Math.max(worstY, Math.abs(p.position[1] - (levelAt(i, j, 0) === 1 ? R : 0)))
        }
      }
      ok(worstX <= 1e-9 && worstZ <= 1e-9,
        `${angleDeg}°/${gap}cm: cell centres are on the pitch in x AND z (${worstX}, ${worstZ})`)
      ok(worstY <= 1e-9, `${angleDeg}°/${gap}cm: and at 0 or R in y (${worstY})`)
    }
  }
  // The bisector, which is what PROVES the half-angle step, agrees with it.
  for (const angleDeg of [7, 30, 61]) {
    const t = angleDeg * RAD
    const wz = 1 + Math.cos(t)
    const wy = Math.sin(t)
    const len = Math.hypot(wz, wy)
    near(wz / len, Math.cos(t / 2), 1e-12, `${angleDeg}°: normalize(u_flat + u_tilt) is (cos(θ/2), …)`)
    near(wy / len, Math.sin(t / 2), 1e-12, `${angleDeg}°: …(…, sin(θ/2))`)
  }
}

// -----------------------------------------------------------------------------
// 3. CLOSURE — every ramp's far end lands on its high cell's edge, less one gap
//    step. Computed from the ramp's OWN geometry, in all four directions.
// -----------------------------------------------------------------------------
console.log('3. every ramp closes onto its high cell')
{
  // Two claims, and they carry different arithmetic:
  //
  //   THE CLOSURE ITSELF is between two points on the joint (`rimA`, the ramp's
  //        far end, and `rimB`, the high cell's near edge), both recorded at
  //        1e-12, so it is asserted at 1e-9 with three orders to spare.
  //   THAT `rimB` REALLY IS THE CELL'S EDGE is a comparison against the cell's
  //        own `position`, which is a display record rounded at 1e-9 — so two
  //        independently-rounded metre-scale coordinates are being differenced
  //        and the honest bound is 2e-9, not 1e-9. Nine orders of margin on a
  //        60cm edge; the claim is that the ramp lands on the RIGHT cell face,
  //        and a wrong one is out by centimetres.
  const Yw = new THREE.Vector3(0, 1, 0)
  for (const angleDeg of [0, 12, 30, 55, 75]) {
    for (const gap of [0.4, 2, 8]) {
      const C = solveLattice({ lattice: { cols: 4, rows: 4 }, angleDeg, gap })
      const M = byId(C)
      const half = (angleDeg * RAD) / 2
      const dirs = new Set()
      let worstClose = 0
      let worstEdge = 0
      let steps = 0
      for (const j of C.joints) {
        if (j.end !== 'hi') continue
        const ramp = M.get(j.a)
        const cell = M.get(j.b)
        const e = rampDir(ramp)
        dirs.add(`${e.x},${e.z}`)
        // The far end of the ramp, plus one gap step along the half-angle
        // direction, IS the high cell's near edge.
        const landed = V(j.rimA)
          .addScaledVector(e, gap * Math.cos(half))
          .addScaledVector(Yw, gap * Math.sin(half))
        worstClose = Math.max(worstClose, landed.distanceTo(V(j.rimB)))
        // ...and that edge is the cell's own, half a panel back from its centre.
        worstEdge = Math.max(worstEdge,
          V(j.rimB).distanceTo(V(cell.position).addScaledVector(e, -L / 2)))
        // The ramp's own segment is exactly one panel long, along `u`. 2e-9 for
        // the same reason as `worstEdge`: two 1e-9 records differenced.
        near(V(ramp.refEnd).distanceTo(V(ramp.refStart)), L, 2e-9,
          `${ramp.id}: the reference segment is exactly L`)
        steps++
      }
      ok(dirs.size === 4, `${angleDeg}°/${gap}cm: all four ramp directions are exercised (${dirs.size})`)
      ok(worstClose <= 1e-9,
        `${angleDeg}°/${gap}cm: ${steps} ramps land one gap step short of the high edge (worst ${worstClose})`)
      ok(worstEdge <= 2e-9,
        `${angleDeg}°/${gap}cm: and that edge is the high cell's own face (worst ${worstEdge})`)
    }
  }

  // ...and the LOW end starts one gap step out of the ground cell's edge, same
  // two claims and the same two tolerances.
  for (const angleDeg of [0, 30, 75]) {
    for (const gap of [0.4, 2, 8]) {
      const C = solveLattice({ lattice: { cols: 4, rows: 4 }, angleDeg, gap })
      const M = byId(C)
      const half = (angleDeg * RAD) / 2
      let worstClose = 0
      let worstEdge = 0
      for (const j of C.joints) {
        if (j.end !== 'lo') continue
        const cell = M.get(j.a)
        const ramp = M.get(j.b)
        const e = rampDir(ramp)
        const stepped = V(j.rimA)
          .addScaledVector(e, gap * Math.cos(half))
          .addScaledVector(Yw, gap * Math.sin(half))
        worstClose = Math.max(worstClose, stepped.distanceTo(V(j.rimB)))
        worstEdge = Math.max(worstEdge,
          V(j.rimA).distanceTo(V(cell.position).addScaledVector(e, L / 2)))
      }
      ok(worstClose <= 1e-9,
        `${angleDeg}°/${gap}cm: every ramp starts one gap step off its ground cell (worst ${worstClose})`)
      ok(worstEdge <= 2e-9, `${angleDeg}°/${gap}cm: off that cell's own face (worst ${worstEdge})`)
    }
  }
}

// -----------------------------------------------------------------------------
// 4. THE CENTRAL CLAIM — every joint spans exactly `gap`.
// -----------------------------------------------------------------------------
console.log('4. every joint spans exactly the nominal gap')
{
  let worst = 0
  let checks = 0
  for (const angleDeg of [0, 5, 12.5, 30, 45, 60, 75]) {
    for (const gap of [0.4, 1, 2, 3.7, 8]) {
      const C = solveLattice({ lattice: { cols: 4, rows: 4 }, angleDeg, gap })
      ok(C.joints.length === 48, `${angleDeg}°/${gap}cm: 24 ramps × 2 ends = 48 joints`)
      for (const j of C.joints) {
        const d = Math.hypot(j.rimB[0] - j.rimA[0], j.rimB[1] - j.rimA[1], j.rimB[2] - j.rimA[2])
        worst = Math.max(worst, Math.abs(d - gap))
        checks++
      }
    }
  }
  // Not exactly 0 as it was on the ribbon: the two rim points come from
  // different constructions (a cell's edge and a ramp's segment end) and are
  // rounded independently at 1e-12, so ~8e-13 of the distance is rounding. That
  // is three orders below the 1e-9 the unit records carry and four below
  // anything physical.
  ok(worst <= 1e-9, `all ${checks} joint spans equal their gap (worst error ${worst})`)

  // The rims are PARALLEL, which is what makes the span constant ALONG the joint
  // rather than merely right at one point. Both rims run along the joint's run
  // axis, so the vector between them is the same at every point of it.
  const C = solveLattice({ lattice: { cols: 3, rows: 3 }, angleDeg: 45, gap: 3 })
  let worstPar = 0
  for (const j of C.joints) {
    worstPar = Math.max(worstPar, Math.abs(j.rimA[j.runAxis] - j.rimB[j.runAxis]))
    const d = V(j.rimB).sub(V(j.rimA))
    d.setComponent(j.runAxis, 0)
    near(d.length(), 3, 1e-9, `${j.id}: the cross-run separation is the gap`)
    // Both panels' own rim directions lie ON the run axis — the joint is a line
    // in one of the two world axes, never a diagonal.
    for (const run of [j.runA, j.runB]) {
      const off = V(run)
      off.setComponent(j.runAxis, 0)
      ok(off.length() < 1e-9, `${j.id}: a panel's rim direction is the run axis`)
    }
  }
  ok(worstPar <= 1e-9, 'the two rims of every joint sit at the same run coordinate — they are parallel')

  // The joints alternate: a ramp meets the floor in a VALLEY and the high level
  // on a RIDGE, whichever way it runs.
  ok(C.joints.filter((j) => j.end === 'lo').length === C.joints.filter((j) => j.end === 'hi').length,
    'exactly half the joints are a ramp\'s ground end')
}

// -----------------------------------------------------------------------------
// 5. HANDEDNESS — asserted directly, flipped and not, in all four directions.
// -----------------------------------------------------------------------------
console.log('5. every panel basis is right-handed')
{
  const Yw = new THREE.Vector3(0, 1, 0)
  for (const angleDeg of [0, 17, 30, 75]) {
    const C = solveLattice({
      lattice: { cols: 4, rows: 4 }, angleDeg,
      // Flip a diagonal of cells so both branches of the quaternion are used.
      overrides: { cells: [0, 1, 2, 3].map((k) => ({ i: k, j: k, flipped: true })), edges: [] },
    })
    let bad = 0
    let badNormal = 0
    let badW = 0
    let badAlong = 0
    const dirs = new Set()
    for (const p of C.panels) {
      const { X, Y, Z } = axesOf(p)
      if (X.clone().cross(Y).distanceTo(Z) > QUAT_TOL) bad++
      // The emitted `normal` IS local +Y through the placement — already negated
      // for a flipped panel, so no consumer has to remember to do it.
      if (Y.distanceTo(V(p.normal)) > QUAT_TOL) badNormal++
      // THE SIGN THAT MATTERS: local X is `Ŷ × ê`, never its negative. On a cell
      // there is no ê and the width axis is world +X.
      const e = p.kind === 'ramp' ? rampDir(p) : new THREE.Vector3(0, 0, 1)
      dirs.add(`${p.kind}:${e.x},${e.z}`)
      const wanted = new THREE.Vector3().crossVectors(Yw, e)
      if (X.distanceTo(wanted) > QUAT_TOL) badW++
      // Local +Z runs along the reference segment (reversed when flipped).
      const along = V(p.refEnd).sub(V(p.refStart)).normalize()
      if (Math.abs(Math.abs(Z.dot(along)) - 1) > QUAT_TOL) badAlong++
    }
    ok(dirs.size === 5, `${angleDeg}°: four ramp directions plus the cells (${[...dirs].length})`)
    ok(bad === 0, `${angleDeg}°: X × Y = Z for every panel, flipped and not`)
    ok(badNormal === 0, `${angleDeg}°: the emitted normal is the LIT normal (local +Y)`)
    ok(badW === 0, `${angleDeg}°: local X is Ŷ × ê on every ramp, in all four directions`)
    ok(badAlong === 0, `${angleDeg}°: local Z runs along the reference segment`)
  }

  // The analytic frame itself, checked exactly rather than through a rounded
  // quaternion: w × n = u for all four ê, which is the identity V4_SPEC §9.3
  // says fixes w rather than leaving it to be chosen.
  for (const angleDeg of [0, 30, 75]) {
    const t = angleDeg * RAD
    for (const e of [[1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1]]) {
      const E = new THREE.Vector3(...e)
      const u = E.clone().multiplyScalar(Math.cos(t)).addScaledVector(Yw, Math.sin(t))
      const n = E.clone().multiplyScalar(-Math.sin(t)).addScaledVector(Yw, Math.cos(t))
      const w = new THREE.Vector3().crossVectors(Yw, E)
      near(w.clone().cross(n).distanceTo(u), 0, 1e-12, `ê = [${e}] at ${angleDeg}°: w × n = u`)
    }
  }
  // And the two directions V4_SPEC §9.3 names by hand.
  const plus = solveLattice({ lattice: { cols: 3, rows: 3 } })
  const M = byId(plus)
  nearV(axesOf(M.get('Ei0j0z')).X.toArray(), [1, 0, 0], QUAT_TOL, 'ê = +Z gives w = +X — the ribbon')
  nearV(axesOf(M.get('Ei0j0x')).X.toArray(), [0, 0, -1], QUAT_TOL, 'ê = +X gives w = −Z, not +Z')
}

// -----------------------------------------------------------------------------
// 6. A CYCLE CLOSES — walk right round a lattice corner and come back.
//
// ground → ramp → high → ramp → ground → ramp → high → ramp → ground, using only
// the emitted rim points and a cell traversal, never the (i, j) formula the
// lattice was generated from. If the pitch were not uniform this is where the
// residual would appear.
// -----------------------------------------------------------------------------
console.log('6. a cycle closes with zero residual')
{
  for (const angleDeg of [0, 30, 62]) {
    for (const gap of [0.4, 2, 6]) {
      const C = solveLattice({ lattice: { cols: 3, rows: 3 }, angleDeg, gap })
      const M = byId(C)
      // Every joint, indexed by the unordered pair it joins.
      const jointOf = new Map()
      for (const j of C.joints) {
        jointOf.set(`${j.a}|${j.b}`, j)
        jointOf.set(`${j.b}|${j.a}`, j)
      }
      // The ramp on a lattice edge, whichever way it runs.
      const rampBetween = (a, b) => {
        for (const p of C.panels) {
          if (p.kind !== 'ramp') continue
          const lo = `Ci${p.i}j${p.j}`
          const hi = p.axis === 'x' ? `Ci${p.i + 1}j${p.j}` : `Ci${p.i}j${p.j + 1}`
          if ((lo === a && hi === b) || (lo === b && hi === a)) return p
        }
        return null
      }
      // The four cells round the corner between (0,0),(1,0),(1,1),(0,1).
      const loop = [['Ci0j0', 'Ci1j0'], ['Ci1j0', 'Ci1j1'], ['Ci1j1', 'Ci0j1'], ['Ci0j1', 'Ci0j0']]
      const cursor = new THREE.Vector3(0, 0, 0)
      let steps = 0
      for (const [from, to] of loop) {
        const ramp = rampBetween(from, to)
        const jA = jointOf.get(`${from}|${ramp.id}`)
        const jB = jointOf.get(`${ramp.id}|${to}`)
        // The rim point on each cell's side of its own joint.
        const enter = V(jA.a === from ? jA.rimA : jA.rimB)
        const land = V(jB.a === to ? jB.rimA : jB.rimB)
        // Cross the ramp (two gap steps and 60 of panel), then cross the cell.
        cursor.add(land.clone().sub(enter))
        const dir = new THREE.Vector3().subVectors(V(M.get(to).position), V(M.get(from).position))
        dir.y = 0
        dir.normalize()
        cursor.addScaledVector(dir, L)
        steps++
      }
      ok(steps === 4, `${angleDeg}°/${gap}cm: four ramps and four cells walked`)
      ok(cursor.length() <= 1e-9,
        `${angleDeg}°/${gap}cm: the cycle returns to the same point (residual ${cursor.length()})`)
    }
  }
}

// -----------------------------------------------------------------------------
// 7. THE CHECKERBOARD.
// -----------------------------------------------------------------------------
console.log('7. the level field, and what it forces')
{
  for (const phase of [0, 1]) {
    const C = solveLattice({ lattice: { cols: 5, rows: 4 }, pattern: { kind: 'trapezoid', phase } })
    const M = byId(C)
    let bad = 0
    for (let i = 0; i < 5; i++) {
      for (let j = 0; j < 4; j++) {
        const want = (i + j + phase) % 2
        if (C.lattice.levels[i][j] !== want) bad++
        const cell = M.get(`Ci${i}j${j}`)
        if (cell.level !== want) bad++
        if (cell.role !== (want === 1 ? 'high' : 'ground')) bad++
      }
    }
    ok(bad === 0, `phase ${phase}: level(i,j) = (i + j + phase) mod 2 everywhere`)

    // EVERY edge joins one ground and one high cell — no edge ever joins two of
    // the same level. That is the property the checkerboard exists to have, and
    // the reason the pitch can be uniform (V4_SPEC §9.1).
    let sameLevel = 0
    let edges = 0
    for (const p of C.panels) {
      if (p.kind !== 'ramp' || p.anchor) continue
      const a = levelAt(p.i, p.j, phase)
      const b = p.axis === 'x' ? levelAt(p.i + 1, p.j, phase) : levelAt(p.i, p.j + 1, phase)
      if (a === b) sameLevel++
      edges++
    }
    ok(edges === 4 * 4 + 5 * 3, `phase ${phase}: 16 x-edges and 15 z-edges (got ${edges})`)
    ok(sameLevel === 0, `phase ${phase}: no edge joins two same-level cells`)

    // ...and every JOINT is a ramp against a cell, never cell-to-cell.
    let wrongKind = 0
    for (const j of C.joints) {
      if (!((j.panelA === 'cell') !== (j.panelB === 'cell'))) wrongKind++
    }
    ok(wrongKind === 0, `phase ${phase}: every joint is one ramp against one cell`)
  }

  // Phase 1 is phase 0 with the levels swapped, panel for panel.
  const a = solveLattice({ lattice: { cols: 4, rows: 4 }, pattern: { phase: 0 } })
  const b = solveLattice({ lattice: { cols: 4, rows: 4 }, pattern: { phase: 1 } })
  let swapped = 0
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      if (byId(a).get(`Ci${i}j${j}`).level === byId(b).get(`Ci${i}j${j}`).level) swapped++
    }
  }
  ok(swapped === 0, 'phase 1 inverts every cell\'s level')
  ok(a.panels.length === b.panels.length, 'and changes nothing about how many panels there are')

  ok(Object.keys(ROLE_GLYPH).length === 4, 'four roles, four glyphs')
}

// -----------------------------------------------------------------------------
// 8. EDITING IS INERT — every other panel keeps the bits it had.
// -----------------------------------------------------------------------------
console.log('8. editing')
{
  const base = solveLattice({ lattice: { cols: 3, rows: 5 }, angleDeg: 30 })

  // An INTERIOR cell, so the network's own extents do not move with it. (The
  // offsets and the grounding are taken over PRESENT material, so removing the
  // panel that happens to be lowest or nearest legitimately moves everything —
  // lattice.js documents that as the one non-local case, inherited from the
  // ribbon's grounding.)
  const cut = solveLattice({
    lattice: { cols: 3, rows: 5 }, angleDeg: 30,
    overrides: { cells: [{ i: 1, j: 2, present: false }], edges: [] },
  })
  ok(cut.panels.length === base.panels.length, 'the panel table keeps its shape')
  const cutM = byId(cut)
  // ONE PANEL, NOT FIVE. A ramp needs one cell, not two (§9.3), so removing a
  // flat leaves its four ramps cantilevered off their far ends rather than
  // taking them with it. That is what makes the design editable panel by panel,
  // and it is the same shape as a wall anchor — a ramp with nothing at one end.
  const goneRamps = base.panels.filter((p) => p.present && !cutM.get(p.id).present).map((p) => p.id)
  ok(goneRamps.length === 1 && goneRamps[0] === 'Ci1j2',
    `removing a cell removes exactly that one panel (got ${goneRamps.length}: ${goneRamps})`)
  for (const id of ['Ei0j2x', 'Ei1j2x', 'Ei1j1z', 'Ei1j2z']) {
    ok(cutM.get(id).present, `its ramp ${id} survives, hanging off its other cell`)
  }

  // But a ramp with NEITHER cell would float, and that one really does go.
  const both = byId(solveLattice({
    lattice: { cols: 3, rows: 5 }, angleDeg: 30,
    overrides: { cells: [{ i: 0, j: 0, present: false }, { i: 0, j: 1, present: false }], edges: [] },
  }))
  ok(!both.get('Ei0j0z').present, 'a ramp between two removed cells has nothing to hang from, and goes')
  ok(both.get('Ei0j1z').present, 'while one that still has a cell stays')

  let moved = 0
  for (const p of base.panels) {
    if (p.id === 'Ci1j2') continue
    const q = cutM.get(p.id)
    if (JSON.stringify(p.position) !== JSON.stringify(q.position)) moved++
    if (JSON.stringify(p.quaternion) !== JSON.stringify(q.quaternion)) moved++
    if (JSON.stringify(p.corners) !== JSON.stringify(q.corners)) moved++
  }
  ok(moved === 0, 'and every other panel is BIT-IDENTICAL — the lattice is generated, not chained')

  // The joints that touched it, and no others, are gone.
  const survivors = new Set(cut.joints.map((j) => j.id))
  const lost = base.joints.filter((j) => !survivors.has(j.id))
  // Four joints, not eight: each of the four ramps loses the end that met this
  // cell and keeps the end that meets its other one.
  ok(lost.length === 4, `four joints disappear — one per orphaned ramp end (got ${lost.length})`)
  ok(cut.joints.every((j) => j.a !== 'Ci1j2' && j.b !== 'Ci1j2'), 'no surviving joint mentions the cell')
  ok(cut.joints.every((j) => !goneRamps.includes(j.a) && !goneRamps.includes(j.b)),
    'and none mentions its ramps')

  // Switching an EDGE off: the ramp goes, the cells stay, nothing moves.
  const openEdge = solveLattice({
    lattice: { cols: 3, rows: 5 }, angleDeg: 30,
    overrides: { cells: [], edges: [{ i: 1, j: 2, axis: 'z', present: false }] },
  })
  const eM = byId(openEdge)
  const goneEdge = base.panels.filter((p) => p.present && !eM.get(p.id).present).map((p) => p.id)
  ok(goneEdge.length === 1 && goneEdge[0] === 'Ei1j2z',
    `switching an edge off removes exactly its ramp (got ${goneEdge})`)
  let movedEdge = 0
  for (const p of base.panels) {
    if (p.id === 'Ei1j2z') continue
    if (JSON.stringify(p.position) !== JSON.stringify(eM.get(p.id).position)) movedEdge++
  }
  ok(movedEdge === 0, 'and moves nothing at all')
  ok(base.joints.length - openEdge.joints.length === 2, 'its two joints go with it')
  ok(eM.get('Ci1j2').present && eM.get('Ci1j3').present, 'while both its cells stay')

  // Strip a 2 × 2 down to one corner cell. Its two ramps hang on, because each
  // still has a cell — so this is also the cheapest check that a cantilevered
  // ramp really is emitted rather than quietly dropped.
  const cornerCfg = {
    lattice: { cols: 2, rows: 2 },
    overrides: {
      cells: [{ i: 0, j: 1, present: false }, { i: 1, j: 0, present: false },
        { i: 1, j: 1, present: false }],
      edges: [],
    },
  }
  const hanging = solveLattice(cornerCfg)
  ok(hanging.panels.filter((p) => p.present).length === 3,
    'one cell and the two ramps still hanging off it')
  ok(hanging.joints.length === 2, 'each cantilevered ramp keeps exactly one joint')

  // Switch those two ramps off as well and the single panel stands alone —
  // every panel really is independently removable.
  const only = solveLattice({
    ...cornerCfg,
    overrides: {
      ...cornerCfg.overrides,
      edges: [{ i: 0, j: 0, axis: 'x', present: false }, { i: 0, j: 0, axis: 'z', present: false }],
    },
  })
  ok(only.panels.filter((p) => p.present).length === 1, 'one cell left of four')
  ok(only.joints.length === 0, 'and nothing to join it to')
  near(only.bounds.size[2], L, 1e-9, 'and the box is one panel deep')
  near(only.bounds.size[0], W, 1e-9, 'and one panel wide')
  ok(latticeBounds([]).size[2] === 0, 'an empty network has a zero box rather than an infinite one')

  // A flip does not move the panel, and inverts its lit normal.
  const flip = solveLattice({
    lattice: { cols: 3, rows: 5 }, angleDeg: 30,
    overrides: { cells: [{ i: 1, j: 2, flipped: true }], edges: [] },
  })
  const fM = byId(flip)
  const plain = byId(base).get('Ci1j2')
  const flipped = fM.get('Ci1j2')
  ok(JSON.stringify(plain.position) === JSON.stringify(flipped.position), 'a flip does not move it')
  nearV(flipped.normal, plain.normal.map((v) => -v), 1e-9, 'the lit normal is inverted')
  const off = (p) => [0, 1, 2].map((k) => p.obb.center[k] - p.position[k])
  nearV(off(flipped), off(plain).map((v) => -v), 1e-9,
    'and the box moves to the other side of the reference plane')
  near(Math.hypot(...off(plain)), T / 2, 1e-9, 'half a housing thickness behind the lit face')
}

// -----------------------------------------------------------------------------
// 9. THE WALL ANCHOR.
// -----------------------------------------------------------------------------
console.log('9. the wall anchor')
{
  const free = solveLattice({ lattice: { cols: 3, rows: 5 }, placement: { wallAnchor: 'free' } })
  const braced = solveLattice({ lattice: { cols: 3, rows: 5 }, placement: { wallAnchor: 'braced' } })
  ok(free.panels.filter((p) => p.anchor).length === 0, 'free adds no anchor ramps')

  const anchors = braced.panels.filter((p) => p.anchor)
  const highAtWall = [0, 1, 2, 3, 4].filter((j) => levelAt(0, j, 0) === 1)
  ok(highAtWall.length === 2, 'the default 3 × 5 has two high cells in column 0')
  ok(anchors.length === highAtWall.length,
    `braced adds exactly one per present high cell at i = 0 (got ${anchors.length})`)
  ok(anchors.every((p) => p.present && p.i === -1 && p.axis === 'x'), 'each is an x-axis ramp at i = −1')
  ok(anchors.map((p) => p.j).join() === highAtWall.join(), `at rows ${highAtWall}`)
  ok(braced.panels.length === free.panels.length + 2, 'and nothing else appears')

  // Each has ONE joint — to its high cell — and it is a ridge like any other.
  const anchorJoints = braced.joints.filter((j) => j.anchor)
  ok(anchorJoints.length === 2, 'two anchor joints')
  ok(anchorJoints.every((j) => j.end === 'hi'), 'each is a ramp\'s high end')
  ok(anchorJoints.every((j) => j.b.startsWith('Ci0j')), 'joined to a column-0 cell')
  let worstSpan = 0
  for (const j of anchorJoints) {
    worstSpan = Math.max(worstSpan,
      Math.abs(Math.hypot(j.rimB[0] - j.rimA[0], j.rimB[1] - j.rimA[1], j.rimB[2] - j.rimA[2]) - 2))
  }
  ok(worstSpan <= 1e-9, `and spans exactly the gap (worst ${worstSpan})`)

  // IT LANDS AT GROUND LEVEL: its toe sits at the same height as every other
  // ramp's ground end, which is one gap step above the floor plane.
  const ungrounded = solveLattice({
    lattice: { cols: 3, rows: 5 },
    placement: { wallAnchor: 'braced', groundToFloor: false, wallOffsetCm: 0, windowOffsetCm: 0 },
  })
  const toes = ungrounded.panels.filter((p) => p.anchor).map((p) => p.refStart[1])
  const ordinaryLows = ungrounded.panels
    .filter((p) => p.kind === 'ramp' && !p.anchor)
    .map((p) => Math.min(p.refStart[1], p.refEnd[1]))
  near(Math.max(...toes), Math.min(...ordinaryLows), 1e-9,
    'the anchor toe sits at the same height as every other ramp\'s ground end')
  near(toes[0], 2 * Math.sin(15 * RAD), 1e-9, 'which is one gap step above the ground plane')

  // A removed high cell takes its anchor with it.
  const cutHigh = solveLattice({
    lattice: { cols: 3, rows: 5 },
    placement: { wallAnchor: 'braced' },
    overrides: { cells: [{ i: 0, j: 1, present: false }], edges: [] },
  })
  ok(cutHigh.panels.filter((p) => p.anchor && p.present).length === 1,
    'removing a high cell at the wall removes its anchor')
  ok(cutHigh.joints.filter((j) => j.anchor).length === 1, 'and its joint')

  // A network whose wall column is all GROUND gets no anchors: those cells are
  // already on the floor.
  const oneRow = solveLattice({ lattice: { cols: 3, rows: 1 }, placement: { wallAnchor: 'braced' } })
  ok(oneRow.panels.filter((p) => p.anchor).length === 0,
    'a single ground cell at the wall needs nothing to prop it')
}

// -----------------------------------------------------------------------------
// 10. OFFSETS — what a tape measure reads.
// -----------------------------------------------------------------------------
console.log('10. the offsets measure material, not indices')
{
  for (const wallAnchor of ['free', 'braced']) {
    for (const k of [0, 12.5, 200]) {
      const C = solveLattice({
        lattice: { cols: 3, rows: 5 },
        placement: {
          wallOffsetCm: k, windowOffsetCm: k, wallAnchor,
          groundToFloor: true, groundClearanceCm: 0,
        },
      })
      const pres = C.panels.filter((p) => p.present)
      const minX = Math.min(...pres.flatMap((p) => p.corners.map((c) => c[0])))
      const minZ = Math.min(...pres.flatMap((p) => p.corners.map((c) => c[2])))
      const minY = Math.min(...pres.flatMap((p) => p.corners.map((c) => c[1])))
      near(minX, k, 1e-9, `${wallAnchor} at ${k}cm: the nearest material to the wall sits at ${k}`)
      near(minZ, k, 1e-9, `${wallAnchor} at ${k}cm: and to the window`)
      near(minY, 0, 1e-9, `${wallAnchor} at ${k}cm: grounded, the lowest material sits on the floor`)
    }
  }

  // BRACED IS DIFFERENT FROM FREE, and that is the point of §9.5: the anchor
  // toes become the minimum x, so the same offset puts the CELLS somewhere else.
  const free = solveLattice({ lattice: { cols: 3, rows: 5 }, placement: { wallOffsetCm: 0, wallAnchor: 'free' } })
  const braced = solveLattice({ lattice: { cols: 3, rows: 5 }, placement: { wallOffsetCm: 0, wallAnchor: 'braced' } })
  const cellX = (C) => byId(C).get('Ci0j0').position[0]
  ok(braced.lattice.shiftCm[0] > free.lattice.shiftCm[0],
    `bracing pushes the cells out from the wall (${free.lattice.shiftCm[0]} → ${braced.lattice.shiftCm[0]})`)
  near(cellX(braced) - cellX(free), braced.lattice.shiftCm[0] - free.lattice.shiftCm[0], 1e-9,
    'by exactly the anchor toe\'s reach')

  // Grounding off leaves the housings hanging below the floor, by exactly one
  // thickness (an unflipped flat cell's housing goes straight down).
  const off = solveLattice({ lattice: { cols: 3, rows: 5 }, placement: { groundToFloor: false } })
  const lowest = Math.min(...off.panels.filter((p) => p.present).flatMap((p) => p.corners.map((c) => c[1])))
  near(lowest, -T, 1e-9, 'ungrounded, the ground cells\' housings hang a full thickness below y = 0')
  ok(off.lattice.shiftCm[1] === 0, 'and no shift is applied')
  // Grounding is a translation in y and nothing else.
  const on = solveLattice({ lattice: { cols: 3, rows: 5 } })
  let bad = 0
  for (let k = 0; k < on.panels.length; k++) {
    if (on.panels[k].position[0] !== off.panels[k].position[0]) bad++
    if (on.panels[k].position[2] !== off.panels[k].position[2]) bad++
  }
  ok(bad === 0, 'grounding moves nothing but y')

  // --- the ground clearance (V4_SPEC §9.12) --------------------------------
  // `groundToFloor` puts the lowest material at `groundClearanceCm`, not at 0.
  // Exact, because the spacers under it are cut to that number — a network
  // sitting 14.97cm up is a network whose posts do not fit. Checked over the
  // angle and gap sweep because the identity of the lowest panel changes with
  // both, and checked at 0 as well so the assertion cannot pass against a
  // constant.
  for (const angleDeg of [0, 30, 60, 75]) {
    for (const gap of [0.4, 2, 8]) {
      for (const clearance of [0, 15, 50]) {
        const C = solveLattice({
          lattice: { cols: 3, rows: 5 }, angleDeg, gap,
          placement: { groundToFloor: true, groundClearanceCm: clearance },
        })
        const y = Math.min(...C.panels.filter((p) => p.present).flatMap((p) => p.corners.map((c) => c[1])))
        near(y, clearance, 1e-9,
          `θ=${angleDeg}° gap=${gap} clearance=${clearance}: the lowest material sits exactly there`)
      }
    }
  }
  // And it is a pure translation: same box, moved.
  const at0 = solveLattice({ lattice: { cols: 3, rows: 5 }, placement: { groundClearanceCm: 0 } })
  const at15 = solveLattice({ lattice: { cols: 3, rows: 5 }, placement: { groundClearanceCm: 15 } })
  near(at15.bounds.size[1], at0.bounds.size[1], 1e-9, 'the clearance does not change the height of the box')
  near(at15.bounds.min[1] - at0.bounds.min[1], 15, 1e-9, 'it moves its floor by exactly the clearance')
  near(at15.lattice.shiftCm[1] - at0.lattice.shiftCm[1], 15, 1e-9, 'and shows up wholly in shiftCm')
  // With grounding OFF the clearance is inert — there is nothing to measure it
  // from, and pretending otherwise would float the design on a switch that says
  // it is off.
  const offA = solveLattice({ lattice: { cols: 3, rows: 5 }, placement: { groundToFloor: false, groundClearanceCm: 0 } })
  const offB = solveLattice({ lattice: { cols: 3, rows: 5 }, placement: { groundToFloor: false, groundClearanceCm: 50 } })
  ok(JSON.stringify(offA.panels) === JSON.stringify(offB.panels),
    'ungrounded, the clearance changes nothing at all')
}

// -----------------------------------------------------------------------------
// 11. THE CORNER HOLES — four ramps meet and leave a diamond.
// -----------------------------------------------------------------------------
console.log('11. the corner holes')
{
  for (const angleDeg of [0, 30, 60]) {
    const C = solveLattice({ lattice: { cols: 3, rows: 3 }, angleDeg })
    const M = byId(C)
    // The ramps off one cell, from the joint table.
    const mine = new Set()
    for (const j of C.joints) {
      if (j.a === 'Ci1j1') mine.add(j.b)
      if (j.b === 'Ci1j1') mine.add(j.a)
    }
    ok(mine.size === 4, `${angleDeg}°: the interior cell has four ramps (got ${mine.size})`)

    // Their REFERENCE FACES are pairwise disjoint in plan: each sits in its own
    // corridor beyond one edge, and the corner beyond both belongs to neither.
    const planBox = (p) => {
      const face = p.corners.slice(0, 4)
      return {
        x0: Math.min(...face.map((c) => c[0])), x1: Math.max(...face.map((c) => c[0])),
        z0: Math.min(...face.map((c) => c[2])), z1: Math.max(...face.map((c) => c[2])),
      }
    }
    const list = [...mine].sort().map((id) => planBox(M.get(id)))
    let overlaps = 0
    for (let a = 0; a < list.length; a++) {
      for (let b = a + 1; b < list.length; b++) {
        const A = list[a]
        const B = list[b]
        if (A.x1 > B.x0 + 1e-9 && B.x1 > A.x0 + 1e-9 && A.z1 > B.z0 + 1e-9 && B.z1 > A.z0 + 1e-9) {
          overlaps++
        }
      }
    }
    ok(overlaps === 0, `${angleDeg}°: the four ramps are pairwise disjoint in plan`)
  }
}

// -----------------------------------------------------------------------------
// 12. The OBB — derived from PANEL_PROFILE, never a written-down 4.1.
// -----------------------------------------------------------------------------
console.log('12. the collision box')
{
  const C = solveLattice({ lattice: { cols: 3, rows: 3 }, angleDeg: 30 })
  for (const p of C.panels) {
    ok(p.obb.halfExtents[0] === W / 2, `${p.id}: half width along local X`)
    ok(p.obb.halfExtents[1] === PANEL_PROFILE.overallThickness / 2,
      `${p.id}: half the DERIVED housing thickness along local Y`)
    ok(p.obb.halfExtents[2] === L / 2, `${p.id}: half length along local Z`)
    ok(JSON.stringify(p.obb.quaternion) === JSON.stringify(p.quaternion),
      `${p.id}: the box shares the panel's rotation`)
  }
  // The box encloses the solid exactly: every corner sits on its surface. 1e-7,
  // not 1e-9, and the arithmetic is the reason rather than slack — the
  // quaternion's components are rounded at 1e-9 (~2e-9 rad) and a corner sits
  // ~42.5cm from the box centre, so the round trip through an inverted rounded
  // quaternion carries ~1e-7cm. A nanometre is not a fit problem.
  let worst = 0
  for (const p of C.panels) {
    const q = new THREE.Quaternion(...p.obb.quaternion)
    const c = V(p.obb.center)
    for (const corner of p.corners) {
      const local = V(corner).sub(c).applyQuaternion(q.clone().invert())
      worst = Math.max(
        worst,
        Math.abs(Math.abs(local.x) - p.obb.halfExtents[0]),
        Math.abs(Math.abs(local.y) - p.obb.halfExtents[1]),
        Math.abs(Math.abs(local.z) - p.obb.halfExtents[2]),
      )
    }
  }
  ok(worst < 1e-7, `the 8 solid corners are exactly every box's corners (worst ${worst})`)
}

// -----------------------------------------------------------------------------
// 13. Determinism.
// -----------------------------------------------------------------------------
console.log('13. determinism')
{
  const cfg = {
    lattice: { cols: 3, rows: 4, panelType: '2x2' },
    angleDeg: 37.5,
    gap: 1.3,
    pattern: { kind: 'trapezoid', phase: 1 },
    placement: { wallOffsetCm: 3.25, windowOffsetCm: 9.75, groundToFloor: true, wallAnchor: 'braced' },
    overrides: {
      cells: [{ i: 2, j: 1, flipped: true }, { i: 1, j: 3, present: false }],
      edges: [{ i: 0, j: 0, axis: 'x', present: false }],
    },
  }
  const a = JSON.stringify(solveLattice(cfg))
  ok(a === JSON.stringify(solveLattice(cfg)), 'two calls on the same config produce identical JSON')
  ok(JSON.stringify(solveLattice(normalizeConfig(cfg))) === a,
    'and normalizing first changes nothing — solveLattice normalizes for itself')
  ok(JSON.stringify(solveLattice(JSON.parse(JSON.stringify(cfg)))) === a,
    'and a JSON round-trip of the config changes nothing')
  ok(!a.includes('-0,') && !a.includes('-0]'), 'no −0 survives into the output')
  ok(!a.includes('NaN'), 'and no NaN')

  // DEFAULT_CONFIG carries a `name` and `{}` does not, and solveLattice hands
  // back the config it solved — so the GEOMETRY is what must match.
  const geom = (C) => JSON.stringify({ lattice: C.lattice, panels: C.panels, joints: C.joints, bounds: C.bounds })
  ok(geom(solveLattice(DEFAULT_CONFIG)) === geom(solveLattice({})),
    'the default config and an empty one produce identical geometry')

  // Degenerate lattices do not throw.
  const one = solveLattice({ lattice: { cols: 1, rows: 1 } })
  ok(one.panels.length === 1 && one.joints.length === 0, 'a one-cell lattice has no joints at all')
  const strip = solveLattice({ lattice: { cols: 1, rows: 2 } })
  ok(strip.panels.length === 3 && strip.joints.length === 2, 'a 1 × 2 lattice is one ramp and two joints')
  const wide = solveLattice({ lattice: { cols: 2, rows: 1 } })
  ok(wide.panels.length === 3 && wide.joints.length === 2, 'and so is a 2 × 1, turned 90°')
  ok(wide.panels.find((p) => p.kind === 'ramp').axis === 'x', 'whose one ramp runs along x')
}

// -----------------------------------------------------------------------------
// G. GROWING AT AN EDGE — the re-origin, and what it has to carry with it
//
// The plan editor grows the rectangle by clicking a slot outside it. That is a
// UI action, but the INVARIANT it has to preserve is a core one and belongs
// here: growing at the wall or window side shifts every index by one, and
// `level = (i + j + phase) mod 2` would invert the whole checkerboard unless
// the phase absorbs the shift. Getting that wrong turns every ground cell into
// a high one — a change that looks deliberate and is not.
// -----------------------------------------------------------------------------
console.log('G. growing at an edge preserves the design')
{
  // Grow at the FAR edge: indices do not move, so nothing else may either.
  const base = solveLattice({ lattice: { cols: 3, rows: 5 }, angleDeg: 30 })
  const far = solveLattice({
    lattice: { cols: 4, rows: 5 }, angleDeg: 30,
    overrides: {
      cells: [1, 2, 3, 4].map((j) => ({ i: 3, j, present: false })),
      edges: [],
    },
  })
  const farM = byId(far)
  let moved = 0
  for (const p of base.panels) {
    const q = farM.get(p.id)
    if (!q || JSON.stringify(p.position) !== JSON.stringify(q.position)) moved++
  }
  ok(moved === 0, 'growing at the FAR edge moves nothing — every old panel is bit-identical')
  ok(farM.get('Ci3j0').present && !farM.get('Ci3j1').present,
    'and only the clicked cell of the new column is present')

  // Grow at the WALL edge: every index shifts by one AND the phase flips, which
  // together must leave each original cell on the LEVEL it had.
  const wall = solveLattice({
    lattice: { cols: 4, rows: 5 }, angleDeg: 30,
    pattern: { kind: 'trapezoid', phase: 1 },
    overrides: { cells: [1, 2, 3, 4].map((j) => ({ i: 0, j, present: false })), edges: [] },
  })
  const wallM = byId(wall)
  let levelsHeld = 0
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 5; j++) {
      const was = byId(base).get(`Ci${i}j${j}`)
      const now = wallM.get(`Ci${i + 1}j${j}`)
      if (was.level === now.level && was.role === now.role) levelsHeld++
    }
  }
  ok(levelsHeld === 15,
    `all 15 original cells keep their level after the re-origin (got ${levelsHeld})`)
  ok(wallM.get('Ci0j0').level !== byId(base).get('Ci0j0').level,
    'while the NEW cell at the wall is the opposite level, as the checkerboard requires')

  // Without the phase flip the whole field inverts — the non-vacuous negative
  // that proves the check above is testing something.
  const unflipped = byId(solveLattice({
    lattice: { cols: 4, rows: 5 }, angleDeg: 30,
    pattern: { kind: 'trapezoid', phase: 0 },
    overrides: { cells: [1, 2, 3, 4].map((j) => ({ i: 0, j, present: false })), edges: [] },
  }))
  let inverted = 0
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 5; j++) {
      if (byId(base).get(`Ci${i}j${j}`).level !== unflipped.get(`Ci${i + 1}j${j}`).level) inverted++
    }
  }
  ok(inverted === 15, 'and omitting the flip inverts every one of them — the flip is load-bearing')
}

console.log(`\ntest-v4-lattice: ${passed} checks passed, ${failed} failed`)
process.exit(failed ? 1 : 0)
