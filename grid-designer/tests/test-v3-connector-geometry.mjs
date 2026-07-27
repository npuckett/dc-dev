/**
 * tests/test-v3-connector-geometry.mjs — the connector's cross-section and solid.
 *
 * Same guarantees test-geometry.mjs holds panelGeometry.js to, because the same
 * thing is at stake: this mesh goes to a printer. Closed 2-manifold, consistent
 * outward orientation, and a signed volume checked against the profile area
 * computed independently — a shell wound inside-out has NEGATIVE signed volume
 * and looks perfectly fine on screen right up until it is sliced.
 *
 * The profile checks matter just as much. The hook is bounded by the panel's
 * TAPER rather than by a parallel slot face, and that is the one place this
 * design departs from v1 on purpose — §2 asserts it against config.js directly
 * rather than against a copied constant.
 */

import * as THREE from 'three'
import {
  connectorProfile,
  connectorEndProfiles,
  connectorOBB,
  polygonArea,
  profileSelfIntersects,
  bezelDepthAt,
  flangeDepthAt,
  solveConnectors,
  CONNECTOR_PROFILE,
  CONNECTOR_LIMITS,
} from '../src/core/v3/connectors.js'
import { buildConnectorGeometry, connectorTransform } from '../src/geometry/connectorGeometry.js'
import { PANEL_PROFILE, PANEL_METRICS } from '../src/config.js'
import { buildPreset, PRESET_IDS } from '../src/core/v3/presets.js'
import { solveLayout } from '../src/core/v3/placement.js'

let passed = 0
let failed = 0
const ok = (c, m) => { if (c) passed++; else { failed++; console.log(`  FAIL: ${m}`) } }
const near = (a, b, tol, m) => ok(Math.abs(a - b) <= tol, `${m} (got ${a}, want ${b} ±${tol})`)

/** A station stub — the geometry only ever reads these five fields. */
const stub = (over = {}) => ({
  lengthCm: 10,
  spanStartCm: 3,
  spanEndCm: 3,
  foldDeg: 0,
  mid: [0, 0, 0],
  frame: { p: [1, 0, 0], q: [0, 1, 0], r: [0, 0, 1] },
  ...over,
})

console.log('=== test-v3-connector-geometry ===')

// -----------------------------------------------------------------------------
// 1. The profile is a well-formed, counter-clockwise simple polygon.
// -----------------------------------------------------------------------------
console.log('1. profile well-formedness')
{
  let notCCW = 0
  let degenerate = 0
  let wrongCount = 0
  for (const spanCm of [1.5, 3, 6, 12]) {
    for (const foldDeg of [-30, -10, 0, 10, 30]) {
      const { points } = connectorProfile({ spanCm, foldDeg })
      if (points.length !== 16) wrongCount++
      if (polygonArea(points) <= 0) notCCW++
      if (profileSelfIntersects(points)) degenerate++
      for (const [p, q] of points) if (!Number.isFinite(p) || !Number.isFinite(q)) degenerate++
    }
  }
  ok(wrongCount === 0, 'the outline is always 16 points, so a loft between two of them cannot tear')
  ok(notCCW === 0, 'the outline is counter-clockwise at every span and fold')
  ok(degenerate === 0, 'no self-intersection or non-finite point in the working range')

  // Span appears in the outline exactly where it should: the part gets wider by
  // exactly the extra gap, because the channels themselves do not change.
  const a = connectorProfile({ spanCm: 3, foldDeg: 0 }).extents
  const b = connectorProfile({ spanCm: 8, foldDeg: 0 }).extents
  near(b.width - a.width, 5, 1e-9, 'widening the gap by 5cm widens the part by exactly 5cm')
  near(a.height, b.height, 1e-9, 'the gap does not change the part height at zero fold')
}

// -----------------------------------------------------------------------------
// 2. The grip is derived from the REAL panel rim, not from v1's parallel slot.
//
// This is the design's one deliberate departure from the v1 part, so it is
// asserted against config.js rather than against a copied number.
// -----------------------------------------------------------------------------
console.log('2. the grip matches the panel rim')
{
  // The clamp's inner faces must lie ON the panel's own rim surfaces. Both are
  // read from PANEL_PROFILE here, exactly as connectors.js reads them, so this
  // check follows a refinement of the panel instead of pinning it.
  near(bezelDepthAt(0), PANEL_PROFILE.bezelDrop, 1e-12, 'the bezel is lowest at the panel edge')
  near(bezelDepthAt(PANEL_PROFILE.bezelWidth), 0, 1e-12, 'and reaches the front-most plane at its inner edge')
  near(flangeDepthAt(0), PANEL_PROFILE.outerWallDepth, 1e-12, 'the flange starts at the back outer corner')
  near(flangeDepthAt(PANEL_PROFILE.flangeWidth), PANEL_METRICS.flangeInnerDepth, 1e-12,
    'and falls to the flange inner depth')

  const { points } = connectorProfile({ spanCm: 4, foldDeg: 0 })
  const rimP = -2   // panel A's edge, at span/2
  const fg = Math.min(CONNECTOR_PROFILE.frontGripCm, PANEL_PROFILE.bezelWidth)
  const bg = Math.min(CONNECTOR_PROFILE.backGripCm, PANEL_PROFILE.flangeWidth)

  // Indices are fixed by connectorProfile's construction: A's clamp is 0–5.
  near(points[0][0], rimP - fg, 1e-9, 'the front lip reaches frontGrip inboard')
  near(points[0][1], -bezelDepthAt(fg), 1e-9, 'and its inner face sits on the bezel')
  near(points[1][0], rimP, 1e-9, 'the clamp meets the panel at its outer edge')
  near(points[1][1], -PANEL_PROFILE.bezelDrop, 1e-9, 'at the front outer corner')
  near(points[2][1], -PANEL_PROFILE.outerWallDepth, 1e-9, 'and runs down the full outer wall')
  near(points[3][0], rimP - bg, 1e-9, 'the flange lip reaches backGrip inboard')
  near(points[3][1], -flangeDepthAt(bg), 1e-9, 'and its inner face sits on the flange')

  // The clamp closes across the outer wall, which is the whole grip.
  near(Math.abs(points[2][1] - points[1][1]), PANEL_METRICS.outerWallHeight, 1e-9,
    'the clamp spans exactly the outer wall height')

  // Both lips must stay on the surfaces they grip, or the clamp overhangs the
  // diffuser at the front or fouls the taper at the back.
  ok(CONNECTOR_PROFILE.frontGripCm <= PANEL_PROFILE.bezelWidth,
    'the front lip stays on the bezel and never covers the diffuser')
  ok(CONNECTOR_PROFILE.backGripCm <= PANEL_PROFILE.flangeWidth,
    'the flange lip stays on the flange and never fouls the taper')
  ok(CONNECTOR_PROFILE.backGripCm > CONNECTOR_PROFILE.frontGripCm,
    'the grip is mostly on the BACK — the flange bears the load, the bezel lip only locates it')
}

// -----------------------------------------------------------------------------
// 3. The feasibility boundary is real, and it tightens as the gap narrows.
//
// A non-vacuous negative: the gate is worthless without cases it rejects.
// -----------------------------------------------------------------------------
console.log('3. the self-intersection gate')
{
  const bad = (spanCm, foldDeg) => profileSelfIntersects(connectorProfile({ spanCm, foldDeg }).points)
  ok(!bad(3, 35), 'a 3cm gap folds 35° fine')
  ok(bad(0.4, 40), 'a 0.4cm gap at 40° puts the two hooks through each other')
  ok(!bad(0.4, 5), 'the same 0.4cm gap is fine when nearly flat')

  // Monotone: a wider gap never makes a fold worse.
  let nonMonotone = 0
  for (const foldDeg of [10, 20, 30, 40]) {
    let seenGood = false
    for (const spanCm of [0.4, 0.6, 0.8, 1, 1.5, 2, 2.5, 3, 5]) {
      const isBad = bad(spanCm, foldDeg)
      if (!isBad) seenGood = true
      else if (seenGood) nonMonotone++
    }
  }
  ok(nonMonotone === 0, 'widening the gap never re-introduces a collision')

  // The limits are consistent with the gate at the gap they claim to allow.
  ok(!bad(CONNECTOR_LIMITS.minSpanCm, 10),
    `minSpanCm (${CONNECTOR_LIMITS.minSpanCm}cm) is buildable at a shallow fold`)

  // --- WHICH dimension sets the boundary, and which are free -------------
  // The clamps meet at their DEEPEST corners as a joint folds, so the fold
  // capacity is set by how far the part stands off the panel — `jawCm` — and
  // not by how far either lip reaches along the rim. Re-measured for the rim
  // clamp; under the previous hook design the budget was the hook's depth.
  const maxFold = (spanCm, profile) => {
    let last = 0
    for (let f = 0; f <= 90; f += 0.5) {
      if (profileSelfIntersects(connectorProfile({ spanCm, foldDeg: f, profile }).points)) break
      last = f
    }
    return last
  }
  const spans = [0.5, 1, 1.5, 2]
  const baseline = spans.map((s) => maxFold(s, {}))

  const same = (profile, label) => {
    const got = spans.map((s) => maxFold(s, profile))
    ok(got.every((v, k) => v === baseline[k]),
      `${label} does not change the fold boundary (${got.join('/')} vs ${baseline.join('/')})`)
  }
  same({ frontGripCm: 0.4 }, 'shortening the bezel lip')
  same({ backGripCm: 1.2 }, 'shortening the flange lip')
  same({ backGripCm: 3.0 }, 'lengthening the flange lip to the full flange')

  // ...and the check is not simply insensitive: the standoff DOES move it.
  const thicker = spans.map((s) => maxFold(s, { jawCm: CONNECTOR_PROFILE.jawCm * 2 }))
  ok(thicker.every((v, k) => v < baseline[k]),
    `a thicker jaw costs fold (${thicker.join('/')} vs ${baseline.join('/')})`)
}

// -----------------------------------------------------------------------------
// 4. The solid is closed, outward-oriented, and the right size.
// -----------------------------------------------------------------------------
console.log('4. the solid')
{
  const cases = [
    stub(),
    stub({ foldDeg: 25 }),
    stub({ foldDeg: -25 }),
    stub({ spanStartCm: 2, spanEndCm: 5 }),           // a twisted joint
    stub({ spanStartCm: 12, spanEndCm: 11, lengthCm: 4 }),
  ]

  for (const st of cases) {
    const label = `span ${st.spanStartCm}→${st.spanEndCm}, fold ${st.foldDeg}, len ${st.lengthCm}`
    const g = buildConnectorGeometry(st)
    const pos = g.getAttribute('position')
    const idx = g.getIndex()

    // Closed 2-manifold: every undirected edge used exactly twice, and every
    // directed edge exactly once — the orientation-consistency check.
    const dir = new Map()
    const und = new Map()
    const key = (i) => {
      const k = i * 3
      return `${pos.array[k].toFixed(6)},${pos.array[k + 1].toFixed(6)},${pos.array[k + 2].toFixed(6)}`
    }
    for (let t = 0; t < idx.count; t += 3) {
      const v = [key(idx.array[t]), key(idx.array[t + 1]), key(idx.array[t + 2])]
      for (let e = 0; e < 3; e++) {
        const a = v[e]
        const b = v[(e + 1) % 3]
        dir.set(`${a}|${b}`, (dir.get(`${a}|${b}`) ?? 0) + 1)
        const u = a < b ? `${a}|${b}` : `${b}|${a}`
        und.set(u, (und.get(u) ?? 0) + 1)
      }
    }
    ok([...und.values()].every((c) => c === 2), `${label}: every edge borders exactly 2 triangles`)
    ok([...dir.values()].every((c) => c === 1), `${label}: consistently wound (no directed edge twice)`)

    // Signed volume, against the profile area × length computed independently.
    // A prism between two parallel end profiles has volume = mean area × length.
    let vol = 0
    for (let t = 0; t < idx.count; t += 3) {
      const p = [0, 1, 2].map((e) => {
        const k = idx.array[t + e] * 3
        return new THREE.Vector3(pos.array[k], pos.array[k + 1], pos.array[k + 2])
      })
      vol += p[0].dot(new THREE.Vector3().crossVectors(p[1], p[2])) / 6
    }
    const { start, end } = connectorEndProfiles(st)
    const expected = ((polygonArea(start.points) + polygonArea(end.points)) / 2) * st.lengthCm
    ok(vol > 0, `${label}: signed volume is positive — the shell faces outward`)
    // Exact for a linear loft: the swept area varies linearly with z, so the
    // mean of the two end areas is the true average cross-section.
    near(vol, expected, Math.abs(expected) * 1e-6 + 1e-6, `${label}: volume = mean profile area × length`)
  }

  // A degenerate length still builds rather than producing a torn shell.
  const flat = buildConnectorGeometry(stub({ lengthCm: 4 }))
  ok(flat.getIndex().count > 0, 'a short part still produces triangles')
}

// -----------------------------------------------------------------------------
// 5. The mesh and its collision box agree.
// -----------------------------------------------------------------------------
console.log('5. mesh and OBB agree')
{
  for (const st of [stub(), stub({ foldDeg: 30 }), stub({ spanStartCm: 2, spanEndCm: 7 })]) {
    const g = buildConnectorGeometry(st)
    const obb = connectorOBB(st)
    const { position, quaternion } = connectorTransform(st)

    // Every mesh vertex, placed in the world, must sit inside the OBB.
    const inv = quaternion.clone().invert()
    const centre = new THREE.Vector3(...obb.center)
    const pos = g.getAttribute('position')
    let outside = 0
    // Tightness is about the CLOSEST approach to each face, so track the
    // smallest slack per axis: some vertex must touch, or the box is oversized
    // and would manufacture collisions that are not there.
    const minSlack = [Infinity, Infinity, Infinity]
    for (let i = 0; i < pos.count; i++) {
      const w = new THREE.Vector3(pos.getX(i), pos.getY(i), pos.getZ(i))
        .applyQuaternion(quaternion).add(position)
      const local = w.clone().sub(centre).applyQuaternion(inv)
      const c = [Math.abs(local.x), Math.abs(local.y), Math.abs(local.z)]
      for (let k = 0; k < 3; k++) {
        if (c[k] > obb.halfExtents[k] + 1e-6) outside++
        minSlack[k] = Math.min(minSlack[k], obb.halfExtents[k] - c[k])
      }
    }
    ok(outside === 0, `fold ${st.foldDeg}: every vertex is inside the collision box`)
    ok(minSlack.every((s) => s < 1e-5), `fold ${st.foldDeg}: the box is tight on all three axes`)

    // Right-handed, or every SAT result downstream is garbage (HANDOFF §6).
    const q = new THREE.Quaternion(...obb.quaternion)
    const ex = new THREE.Vector3(1, 0, 0).applyQuaternion(q)
    const ey = new THREE.Vector3(0, 1, 0).applyQuaternion(q)
    const ez = new THREE.Vector3(0, 0, 1).applyQuaternion(q)
    near(new THREE.Vector3().crossVectors(ex, ey).dot(ez), 1, 1e-9,
      `fold ${st.foldDeg}: the OBB basis is right-handed`)
  }
}

// -----------------------------------------------------------------------------
// 6. Every station of every preset produces a buildable part.
// -----------------------------------------------------------------------------
console.log('6. preset sweep')
{
  let total = 0
  let infeasible = 0
  let worstFold = 0
  let narrowest = Infinity
  for (const id of PRESET_IDS) {
    const cfg = buildPreset(id)
    const C = solveConnectors(cfg, solveLayout(cfg))
    let bad = 0
    for (const st of C.stations) {
      total++
      const { start, end } = connectorEndProfiles(st)
      if (profileSelfIntersects(start.points) || profileSelfIntersects(end.points)) { infeasible++; continue }
      worstFold = Math.max(worstFold, Math.abs(st.foldDeg))
      narrowest = Math.min(narrowest, st.spanMinCm)
      const g = buildConnectorGeometry(st)
      if (g.getIndex().count === 0) bad++
      if (!g.boundingBox || !Number.isFinite(g.boundingBox.min.x)) bad++
    }
    ok(bad === 0, `${id}: every feasible station builds a finite solid`)
  }
  console.log(`   ${total} stations, worst |fold| ${worstFold.toFixed(1)}°, narrowest gap ${narrowest.toFixed(2)}cm`)
  console.log(`   ${infeasible} infeasible (hooks colliding)`)
  ok(total > 200, `the sweep is substantial (${total} stations)`)

  // Determinism: the same station must produce the same vertex buffer.
  const cfg = buildPreset('drift')
  const st = solveConnectors(cfg).stations[0]
  const g1 = buildConnectorGeometry(st)
  const g2 = buildConnectorGeometry(st)
  ok(g1.getAttribute('position').array.join(',') === g2.getAttribute('position').array.join(','),
    'the same station builds a byte-identical mesh')
}

// -----------------------------------------------------------------------------
// 7. THE POINT OF THE WHOLE PART: it grips the rim without cutting into it.
//
// §2 checks the outline's coordinates. This checks the CONSEQUENCE, which is a
// different claim: that the panel's material and the connector's material are
// disjoint, and that the connector nonetheless closes around the rim on both
// sides. Either half alone is satisfiable by a part that does nothing — a
// connector floating in the gap has no interference at all.
// -----------------------------------------------------------------------------
console.log('7. the clamp grips the rim, and does not cut into it')
{
  /** Even-odd point-in-polygon. */
  const inside = (pts, x, y) => {
    let n = false
    for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
      const [xi, yi] = pts[i]
      const [xj, yj] = pts[j]
      if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) n = !n
    }
    return n
  }

  const fg = Math.min(CONNECTOR_PROFILE.frontGripCm, PANEL_PROFILE.bezelWidth)
  const bg = Math.min(CONNECTOR_PROFILE.backGripCm, PANEL_PROFILE.flangeWidth)
  const EPS = 0.02

  let interference = 0
  let noFrontGrip = 0
  let noBackGrip = 0
  let cases = 0

  for (const foldDeg of [-25, -10, 0, 10, 25]) {
    for (const spanCm of [1.5, 3, 6, 12]) {
      const { points } = connectorProfile({ spanCm, foldDeg })
      const phi = (foldDeg * Math.PI) / 180 / 2
      const sides = [
        { rim: [-spanCm / 2, 0], inward: [-Math.cos(phi), -Math.sin(phi)], deeper: [Math.sin(phi), -Math.cos(phi)] },
        { rim: [spanCm / 2, 0], inward: [Math.cos(phi), -Math.sin(phi)], deeper: [-Math.sin(phi), -Math.cos(phi)] },
      ]
      const at = (s, i, d) => [
        s.rim[0] + i * s.inward[0] + d * s.deeper[0],
        s.rim[1] + i * s.inward[1] + d * s.deeper[1],
      ]

      for (const side of sides) {
        cases++
        for (let k = 1; k <= 8; k++) {
          const iF = (fg * k) / 9
          const iB = (bg * k) / 9

          // (a) PANEL MATERIAL — between the bezel and the flange, at any inboard
          //     offset the clamp reaches — must never be inside the clamp.
          for (let m = 1; m <= 6; m++) {
            const i = (Math.max(fg, bg) * k) / 9
            const top = -bezelDepthAt(i)
            const bot = -flangeDepthAt(i)
            const [x, y] = at(side, i, -(top + ((bot - top) * m) / 7))
            if (inside(points, x, y)) interference++
          }

          // (b) THE GRIP — material just off the bezel and just off the flange
          //     must BE the clamp, or it is holding nothing.
          const [fx, fy] = at(side, iF, bezelDepthAt(iF) - EPS)
          if (!inside(points, fx, fy)) noFrontGrip++
          const [bx, by] = at(side, iB, flangeDepthAt(iB) + EPS)
          if (!inside(points, bx, by)) noBackGrip++
        }
      }
    }
  }

  ok(interference === 0, `no clamp material inside the panel rim, over ${cases} panel/fold/span cases`)
  ok(noFrontGrip === 0, 'the bezel lip covers the bezel over its full reach')
  ok(noBackGrip === 0, 'the flange lip covers the flange over its full reach')

  // Non-vacuous: the clamp must actually be somewhere, not an empty polygon.
  const area = polygonArea(connectorProfile({ spanCm: 3, foldDeg: 0 }).points)
  ok(area > 1, `the clamp has real section area (${area.toFixed(2)} cm²)`)
}

console.log(`\ntest-v3-connector-geometry: ${passed} checks passed, ${failed} failed`)
process.exit(failed ? 1 : 0)
