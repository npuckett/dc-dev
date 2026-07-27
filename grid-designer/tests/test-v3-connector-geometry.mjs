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
  backHalfProfile,
  frontBarProfile,
  frontBarBand,
  connectorEndProfiles,
  connectorOBB,
  polygonArea,
  profileSelfIntersects,
  bezelDepthAt,
  flangeDepthAt,
  splitPlaneRange,
  solveConnectors,
  CONNECTOR_PROFILE,
  CONNECTOR_LIMITS,
} from '../src/core/v3/connectors.js'
import { buildConnectorGeometry, buildFrontBarGeometry, connectorTransform } from '../src/geometry/connectorGeometry.js'
import { PANEL_PROFILE, PANEL_METRICS } from '../src/config.js'
import { buildPreset, PRESET_IDS } from '../src/core/v3/presets.js'
import { buildReport } from '../src/core/v3/report.js'
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
console.log('1. both sections are well-formed, counter-clockwise polygons')
{
  let notCCW = 0
  let degenerate = 0
  let wrongCount = 0
  for (const spanCm of [1.5, 3, 6, 12]) {
    for (const foldDeg of [-30, -10, 0, 10, 30]) {
      const { points } = backHalfProfile({ spanCm, foldDeg })
      if (points.length !== 8) wrongCount++
      if (polygonArea(points) <= 0) notCCW++
      if (profileSelfIntersects(points)) degenerate++
      for (const [p, q] of points) if (!Number.isFinite(p) || !Number.isFinite(q)) degenerate++
    }
  }
  ok(wrongCount === 0, 'the back half is always an 8-point outline, so a loft cannot tear')
  ok(notCCW === 0, 'and counter-clockwise at every span and fold — a clockwise one lofts inside-out')
  ok(degenerate === 0, 'no self-intersection or non-finite point in the working range')

  for (const w of [3, 5, 9]) {
    const bar = frontBarProfile(w)
    ok(bar.points.length === 4, `the front bar at ${w}cm is a rectangle`)
    ok(polygonArea(bar.points) > 0, `and counter-clockwise`)
    near(bar.extents.width, w, 1e-12, `and exactly ${w}cm wide`)
  }

  const a = backHalfProfile({ spanCm: 3, foldDeg: 0 }).extents
  const b = backHalfProfile({ spanCm: 8, foldDeg: 0 }).extents
  near(b.width - a.width, 5, 1e-9, 'widening the gap by 5cm widens the back half by exactly 5cm')
  near(a.height, b.height, 1e-9, 'the gap does not change its height at zero fold')
}

// -----------------------------------------------------------------------------
// 2. Each piece grips the surface it is supposed to, derived from PANEL_PROFILE.
// -----------------------------------------------------------------------------
console.log('2. the pieces match the panel rim')
{
  near(bezelDepthAt(0), PANEL_PROFILE.bezelDrop, 1e-12, 'the bezel is lowest at the panel edge')
  near(bezelDepthAt(PANEL_PROFILE.bezelWidth), 0, 1e-12, 'and reaches the front plane at its inner edge')
  near(flangeDepthAt(0), PANEL_PROFILE.outerWallDepth, 1e-12, 'the flange starts at the back outer corner')
  near(flangeDepthAt(PANEL_PROFILE.flangeWidth), PANEL_METRICS.flangeInnerDepth, 1e-12,
    'and falls to the flange inner depth')

  // BACK HALF: lips on the flange, top face at the split plane.
  const bg = Math.min(CONNECTOR_PROFILE.backGripCm, PANEL_PROFILE.flangeWidth)
  const { points } = backHalfProfile({ spanCm: 4, foldDeg: 0 })
  const sh = CONNECTOR_PROFILE.shimCm
  near(points[7][0], -2 - bg, 1e-9, 'the flange lip reaches backGrip inboard')
  near(points[7][1], -(flangeDepthAt(bg) + sh), 1e-9,
    'and its top face sits exactly one shim below the flange — never on it')
  near(points[6][1], -(PANEL_PROFILE.outerWallDepth + sh), 1e-9,
    'it runs up the outer wall a shim clear of the back corner')
  near(points[5][1], -CONNECTOR_PROFILE.splitDepthCm, 1e-9,
    'to the split plane — which is a face against the OTHER HALF, so no shim there')
  near(points[5][0], -2 + sh, 1e-9, 'and stands a shim off the panel edge')
  ok(CONNECTOR_PROFILE.backGripCm <= PANEL_PROFILE.flangeWidth,
    'the flange lip stays on the flange and never fouls the taper')

  // THE SPLIT PLANE has only the outer wall to live in.
  const [lo, hi] = splitPlaneRange()
  near(lo, PANEL_PROFILE.bezelDrop, 1e-12, 'it must clear the bezel the front bar grips')
  near(hi, PANEL_PROFILE.outerWallDepth, 1e-12, 'and stay above the flange the back half grips')
  near(hi - lo, PANEL_METRICS.outerWallHeight, 1e-12,
    'so its whole latitude is the outer wall height')
  ok(CONNECTOR_PROFILE.splitDepthCm > lo && CONNECTOR_PROFILE.splitDepthCm < hi,
    'and the shipped split sits inside it')

  // FRONT BAR: its underside stays within the bezel's own drop, so it bears on
  // the bezel rather than hovering above the panel or cutting into it.
  const bar = frontBarProfile(5)
  near(bar.extents.qMin, sh, 1e-12,
    'the bar floats exactly one shim above the panel front plane — the bezel peak')
  near(bar.extents.height, CONNECTOR_PROFILE.frontThicknessCm, 1e-12, 'and is its own thickness thick')
  ok(sh >= 0.05 && sh <= 0.1,
    `the shim is in the 0.5–1mm band a rubber shim needs (${(sh * 10).toFixed(2)}mm)`)
}

// -----------------------------------------------------------------------------
// 3. The feasibility boundary is real, and it tightens as the gap narrows.
//
// A non-vacuous negative: the gate is worthless without cases it rejects.
// -----------------------------------------------------------------------------
console.log('3. the self-intersection gate')
{
  const bad = (spanCm, foldDeg) => profileSelfIntersects(backHalfProfile({ spanCm, foldDeg }).points)
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

  // --- WHAT SETS THE BOUNDARY, in closed form ----------------------------
  // With the two-piece design the back half closes when the two panels' own
  // BACK OUTER CORNERS meet — i.e. when the gap has shut at the outer-wall
  // depth. So the limit is
  //
  //     maxFold = 2·asin( gap / (2·outerWallDepth) )
  //
  // which contains no connector dimension at all. That is the finding: the fold
  // limit belongs to the PANEL, and no redesign of the part buys more of it.
  // (Past that point the panels themselves interpenetrate, which collide.js
  // already reports — the two agree because they are the same event.)
  const maxFold = (spanCm, profile) => {
    let last = 0
    for (let f = 0; f <= 90; f += 0.5) {
      if (profileSelfIntersects(backHalfProfile({ spanCm, foldDeg: f, profile }).points)) break
      last = f
    }
    return last
  }
  // The two back corners sit a shim off the panel in BOTH axes, so their
  // separation is  gap − 2·shim·cos(φ) − 2·(outerWallDepth + shim)·sin(φ)
  // with φ = fold/2. Solved by bisection here — derived from PANEL_PROFILE and
  // the shim alone, owing nothing to the profile code it checks.
  const sh = CONNECTOR_PROFILE.shimCm
  const sep = (gap, phi) =>
    gap - 2 * sh * Math.cos(phi) - 2 * (PANEL_PROFILE.outerWallDepth + sh) * Math.sin(phi)
  const closedForm = (gap) => {
    if (sep(gap, Math.PI / 4) > 0) return 90
    let lo = 0
    let hi = Math.PI / 4
    for (let k = 0; k < 60; k++) {
      const mid = (lo + hi) / 2
      if (sep(gap, mid) > 0) lo = mid; else hi = mid
    }
    return ((lo + hi) / 2) * 2 * 180 / Math.PI
  }
  let offBy = 0
  for (const gap of [0.5, 0.8, 1, 1.5, 2]) {
    const got = maxFold(gap, {})
    if (Math.abs(got - Math.min(90, closedForm(gap))) > 0.6) offBy++
  }
  ok(offBy === 0, 'the fold limit is exactly where the two shimmed back corners meet, at every gap')

  // And it really is independent of the connector's own dimensions.
  const spans = [0.5, 1, 1.5]
  const baseline = spans.map((s) => maxFold(s, {}))
  const same = (profile, label) => {
    const got = spans.map((s) => maxFold(s, profile))
    ok(got.every((v, k) => v === baseline[k]), `${label} cannot buy fold (${got.join('/')})`)
  }
  same({ splitDepthCm: 1.0 }, 'a deeper split')
  same({ splitDepthCm: 0.3 }, 'a shallower split')
  same({ backGripCm: 3.0 }, 'a longer flange lip')
  same({ backFloorCm: 0.9 }, 'a deeper floor')
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
console.log('6. every piece of every station renders')
{
  // The whole kit, both families, across every preset — closed, outward-wound,
  // finite and placed. This is the guard that "all the connectors render
  // properly" stays true: a single torn or inside-out piece is invisible in the
  // viewport and fatal at the slicer.
  let total = 0
  const bad = []
  let worstOffset = 0

  for (const id of PRESET_IDS) {
    const conn = buildReport(buildPreset(id)).connectors
    for (const st of conn.stations) {
      const pieces = [['back half', buildConnectorGeometry(st)]]
      pieces.push(['front bar', st.barWidthCm ? buildFrontBarGeometry(st.barWidthCm, st.lengthCm) : null])

      for (const [kind, geo] of pieces) {
        total++
        if (!geo) { bad.push(`${id} ${st.id} ${kind}: no geometry`); continue }
        const pos = geo.getAttribute('position')
        const idx = geo.getIndex()
        if (!idx || idx.count === 0) { bad.push(`${id} ${st.id} ${kind}: no triangles`); continue }

        let nonFinite = 0
        for (let i = 0; i < pos.count; i++) {
          if (![pos.getX(i), pos.getY(i), pos.getZ(i)].every(Number.isFinite)) nonFinite++
        }
        if (nonFinite) { bad.push(`${id} ${st.id} ${kind}: ${nonFinite} non-finite vertices`); continue }

        const key = (i) => {
          const k = i * 3
          return `${pos.array[k].toFixed(5)},${pos.array[k + 1].toFixed(5)},${pos.array[k + 2].toFixed(5)}`
        }
        const und = new Map()
        const dir = new Map()
        for (let t = 0; t < idx.count; t += 3) {
          const v = [key(idx.array[t]), key(idx.array[t + 1]), key(idx.array[t + 2])]
          for (let e = 0; e < 3; e++) {
            const x = v[e]
            const y = v[(e + 1) % 3]
            dir.set(`${x}|${y}`, (dir.get(`${x}|${y}`) ?? 0) + 1)
            const u = x < y ? `${x}|${y}` : `${y}|${x}`
            und.set(u, (und.get(u) ?? 0) + 1)
          }
        }
        if (![...und.values()].every((c) => c === 2)) bad.push(`${id} ${st.id} ${kind}: not closed`)
        if (![...dir.values()].every((c) => c === 1)) bad.push(`${id} ${st.id} ${kind}: inconsistent winding`)

        let vol = 0
        for (let t = 0; t < idx.count; t += 3) {
          const q = [0, 1, 2].map((e) => {
            const k = idx.array[t + e] * 3
            return new THREE.Vector3(pos.array[k], pos.array[k + 1], pos.array[k + 2])
          })
          vol += q[0].dot(new THREE.Vector3().crossVectors(q[1], q[2])) / 6
        }
        if (vol <= 0) bad.push(`${id} ${st.id} ${kind}: inside-out`)

        const { position, quaternion } = connectorTransform(st)
        geo.computeBoundingBox()
        const centre = geo.boundingBox.getCenter(new THREE.Vector3()).applyQuaternion(quaternion).add(position)
        const off = centre.distanceTo(new THREE.Vector3(...st.mid))
        worstOffset = Math.max(worstOffset, off)
        if (off > 12) bad.push(`${id} ${st.id} ${kind}: sits ${off.toFixed(1)}cm from its station`)
        geo.dispose()
      }
    }
  }
  ok(bad.length === 0, bad.length
    ? `${bad.length} bad pieces, first: ${bad[0]}`
    : `all ${total} pieces are closed, outward-wound, finite and placed`)
  ok(total > 800, `the sweep is substantial (${total} pieces)`)
  // Both pieces really are being built, not just the back half twice.
  ok(worstOffset > 0.1, 'the two pieces sit at different depths, as they must')
  console.log(`   ${total} pieces, worst centre offset ${worstOffset.toFixed(2)}cm`)

  const cfg = buildPreset('drift')
  const st = buildReport(cfg).connectors.stations[0]
  const g1 = buildConnectorGeometry(st)
  const g2 = buildConnectorGeometry(st)
  ok(g1.getAttribute('position').array.join(',') === g2.getAttribute('position').array.join(','),
    'the same station builds a byte-identical mesh')
}

console.log('7. each piece grips without cutting into the panel')
{
  const inside = (pts, x, y) => {
    let n = false
    for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
      const [xi, yi] = pts[i]
      const [xj, yj] = pts[j]
      if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) n = !n
    }
    return n
  }
  const bg = Math.min(CONNECTOR_PROFILE.backGripCm, PANEL_PROFILE.flangeWidth)
  // Probe beyond the shim: the clamp is deliberately held off the panel, so a
  // probe inside the shim gap would find void and report no grip.
  const EPS = CONNECTOR_PROFILE.shimCm * 2
  let interference = 0
  let noGrip = 0
  let cases = 0

  for (const foldDeg of [-25, -10, 0, 10, 25]) {
    for (const spanCm of [1.5, 3, 6, 12]) {
      const { points } = backHalfProfile({ spanCm, foldDeg })
      const phi = (foldDeg * Math.PI) / 180 / 2
      const sides = [
        { rim: [-spanCm / 2, 0], inward: [-Math.cos(phi), -Math.sin(phi)], deeper: [Math.sin(phi), -Math.cos(phi)] },
        { rim: [spanCm / 2, 0], inward: [Math.cos(phi), -Math.sin(phi)], deeper: [-Math.sin(phi), -Math.cos(phi)] },
      ]
      const at = (s, i, d) => [s.rim[0] + i * s.inward[0] + d * s.deeper[0], s.rim[1] + i * s.inward[1] + d * s.deeper[1]]

      for (const side of sides) {
        cases++
        for (let k = 1; k <= 8; k++) {
          const i = (bg * k) / 9
          // (a) PANEL MATERIAL — anything between the bezel and the flange at an
          //     offset the back half reaches must never be inside it.
          for (let m = 1; m <= 6; m++) {
            const top = -bezelDepthAt(i)
            const bot = -flangeDepthAt(i)
            const [x, y] = at(side, i, -(top + ((bot - top) * m) / 7))
            if (inside(points, x, y)) interference++
          }
          // (b) THE GRIP — material just behind the flange must BE the back half.
          const [gx, gy] = at(side, i, flangeDepthAt(i) + CONNECTOR_PROFILE.shimCm + EPS)
          if (!inside(points, gx, gy)) noGrip++
        }
      }
    }
  }
  ok(interference === 0, `no back-half material inside the panel rim, over ${cases} panel/fold/span cases`)
  ok(noGrip === 0, 'the flange lip covers the flange over its full reach')

  // THE FRONT BAR bears on both bezels for every gap in its band, and never
  // reaches past the bezel onto the diffuser. That IS the band's definition, so
  // it is checked at both ends of it rather than at a comfortable middle.
  let badBar = 0
  for (const width of [4, 6, 9]) {
    const [lo, hi] = frontBarBand(width)
    for (const gap of [lo, (lo + hi) / 2, hi]) {
      const lip = (width - gap) / 2
      if (lip < CONNECTOR_PROFILE.frontMinLipCm - 1e-9) badBar++
      if (lip > PANEL_PROFILE.bezelWidth + 1e-9) badBar++
    }
  }
  ok(badBar === 0, 'across every band, the bar keeps a bearing lip and never overhangs the diffuser')

  const area = polygonArea(backHalfProfile({ spanCm: 3, foldDeg: 0 }).points)
  ok(area > 1, `the back half has real section area (${area.toFixed(2)} cm²)`)
}

console.log(`\ntest-v3-connector-geometry: ${passed} checks passed, ${failed} failed`)
process.exit(failed ? 1 : 0)
