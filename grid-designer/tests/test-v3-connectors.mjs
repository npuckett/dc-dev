/**
 * tests/test-v3-connectors.mjs — headless checks for core/v3/connectors.js.
 *
 * The control throughout is the FLAT form (amplitude 0), where every answer is
 * known exactly: span = the nominal gap, dihedral = 0, twist = 0. Everything
 * curved is then checked as a departure from that, which is the only way sign
 * and frame conventions ever get caught (README, "Testing").
 *
 * The claim this package exists to support — that a SHORT part sees far less of
 * a joint's variation than the joint has — is asserted directly in §5, against
 * the alternative of one part spanning the whole joint. A check that short parts
 * fit well is worthless without the long-part comparison that fails.
 */

import * as THREE from 'three'
import {
  solveConnectors,
  stationCount,
  connectorProfile,
  SPAN_SAMPLES,
  CROWDED_CODE,
  CONNECTOR_PROFILE,
  taperDepthAt,
} from '../src/core/v3/connectors.js'
import { solveLayout, jointEdgePoint } from '../src/core/v3/placement.js'
import { DEFAULT_CONFIG, DEFAULT_CONNECTORS, normalizeConfig, validateConfig } from '../src/core/v3/schema.js'
import { PANEL_PROFILE } from '../src/config.js'
import { buildPreset, PRESET_IDS } from '../src/core/v3/presets.js'

let passed = 0
let failed = 0
const ok = (c, m) => { if (c) passed++; else { failed++; console.log(`  FAIL: ${m}`) } }
const near = (a, b, tol, m) => ok(Math.abs(a - b) <= tol, `${m} (got ${a}, want ${b} ±${tol})`)

const cfgOf = (over = {}) => normalizeConfig({
  ...DEFAULT_CONFIG,
  ...over,
  form: { ...DEFAULT_CONFIG.form, ...(over.form ?? {}) },
  connectors: { ...DEFAULT_CONNECTORS, ...(over.connectors ?? {}) },
})

console.log('=== test-v3-connectors ===')

// -----------------------------------------------------------------------------
// 1. The flat control — every quantity has a closed-form answer.
// -----------------------------------------------------------------------------
console.log('1. flat form: the known-answer control')
{
  for (const gap of [1, 2, 3.5]) {
    const cfg = cfgOf({ gap, form: { amplitude: 0 } })
    const C = solveConnectors(cfg)
    ok(C.stations.length > 0, `gap ${gap}: stations exist`)
    let worstSpan = 0
    let worstDih = 0
    let worstTwist = 0
    let worstSpread = 0
    for (const st of C.stations) {
      worstSpan = Math.max(worstSpan, Math.abs(st.spanCm - gap))
      worstDih = Math.max(worstDih, st.dihedralDeg)
      worstTwist = Math.max(worstTwist, st.twistDeg)
      worstSpread = Math.max(worstSpread, st.spanSpreadCm)
    }
    near(worstSpan, 0, 1e-9, `gap ${gap}: every span equals the nominal gap`)
    near(worstDih, 0, 1e-9, `gap ${gap}: every dihedral is 0`)
    near(worstTwist, 0, 1e-9, `gap ${gap}: every twist is 0`)
    near(worstSpread, 0, 1e-9, `gap ${gap}: no span varies along a part`)
  }

  // On the flat sheet the frames are the world axes, so they can be named.
  const C = solveConnectors(cfgOf({ form: { amplitude: 0 } }))
  let badNormal = 0
  let badOrtho = 0
  let badInward = 0
  for (const st of C.stations) {
    for (const f of [st.aFrame, st.bFrame]) {
      const n = new THREE.Vector3(...f.normal)
      const run = new THREE.Vector3(...f.run)
      const inward = new THREE.Vector3(...f.inward)
      if (Math.abs(n.y - 1) > 1e-9) badNormal++
      // run, inward and normal are a right-angled triple on every station.
      if (Math.abs(run.dot(n)) > 1e-9 || Math.abs(inward.dot(n)) > 1e-9 ||
          Math.abs(run.dot(inward)) > 1e-9) badOrtho++
      if (Math.abs(inward.length() - 1) > 1e-9 || Math.abs(run.length() - 1) > 1e-9) badOrtho++
    }
    // The two panels lie on opposite sides of the joint, so their inward
    // directions must OPPOSE. This is the check that catches a dropped sign.
    const ia = new THREE.Vector3(...st.aFrame.inward)
    const ib = new THREE.Vector3(...st.bFrame.inward)
    if (ia.dot(ib) > -1 + 1e-9) badInward++
  }
  ok(badNormal === 0, 'flat: every lit normal is +Y')
  ok(badOrtho === 0, 'flat: (run, inward, normal) is an orthonormal triple everywhere')
  ok(badInward === 0, 'flat: the two panels reach in from opposite sides of every joint')
}

// -----------------------------------------------------------------------------
// 2. Station counts and positions, closed form.
// -----------------------------------------------------------------------------
console.log('2. station count and placement')
{
  const d = { spacingCm: 50, minPerJoint: 2 }
  ok(stationCount(60, d) === 2, '60cm edge at 50cm spacing → 2 parts')
  ok(stationCount(121, d) === 3, '121cm plate edge at 50cm spacing → 3 parts')
  ok(stationCount(58, d) === 2, '58cm partial edge → 2 parts')
  ok(stationCount(50, d) === 2, 'exactly 50cm → 1 by spacing, raised to 2 by the floor')
  ok(stationCount(10, { spacingCm: 50, minPerJoint: 1 }) === 1, 'a short edge takes 1 when the floor allows')
  ok(stationCount(200, { spacingCm: 50, minPerJoint: 1 }) === 4, 'spacing is a ceiling, not a rounding')
  ok(stationCount(121, { spacingCm: 50, minPerJoint: 5 }) === 5, 'minPerJoint raises the count')

  // Stations are centred and symmetric within their joint.
  const cfg = cfgOf({ form: { amplitude: 0 } })
  const C = solveConnectors(cfg)
  const L = solveLayout(cfg)
  let asym = 0
  let outside = 0
  for (const pj of C.perJoint) {
    const edge = L.adjacency[pj.jointIndex]
    const mine = C.stations.filter((s) => s.jointIndex === pj.jointIndex)
    ok(mine.length === pj.count, `joint ${pj.jointIndex}: ${pj.count} stations emitted`)
    const mid = (edge.edge.from + edge.edge.to) / 2
    // Symmetric about the joint's midpoint: s[k] + s[count-1-k] = 2·mid.
    for (let k = 0; k < mine.length; k++) {
      const j = mine.length - 1 - k
      if (Math.abs(mine[k].s + mine[j].s - 2 * mid) > 1e-9) asym++
      const half = mine[k].lengthCm / 2
      if (mine[k].s - half < edge.edge.from - 1e-9 || mine[k].s + half > edge.edge.to + 1e-9) outside++
    }
  }
  ok(asym === 0, 'stations are symmetric about each joint midpoint')
  ok(outside === 0, 'no part runs off the end of its joint')
}

// -----------------------------------------------------------------------------
// 3. Parts never overlap each other, at any setting.
// -----------------------------------------------------------------------------
console.log('3. no two parts on a joint overlap')
{
  let overlaps = 0
  let shortened = 0
  for (const lengthCm of [4, 10, 30]) {
    for (const spacingCm of [10, 50, 200]) {
      for (const minPerJoint of [1, 2, 6]) {
        const C = solveConnectors(cfgOf({ connectors: { lengthCm, spacingCm, minPerJoint } }))
        const byJoint = new Map()
        for (const st of C.stations) {
          if (!byJoint.has(st.jointIndex)) byJoint.set(st.jointIndex, [])
          byJoint.get(st.jointIndex).push(st)
        }
        for (const list of byJoint.values()) {
          for (let k = 1; k < list.length; k++) {
            const prevEnd = list[k - 1].s + list[k - 1].lengthCm / 2
            const thisStart = list[k].s - list[k].lengthCm / 2
            // In the crowded case consecutive parts butt EXACTLY end to end
            // (partLength = span/count is precisely the station pitch), so the
            // clearance is zero by construction and the only thing left to
            // measure is `s` and `lengthCm` each being rounded to 1e-9 on
            // output. A 1e-9 tolerance here would be testing that rounding —
            // measured -1.0e-9 cm, i.e. 10 picometres. 1e-6 cm is still ten
            // thousand times finer than anything printable.
            if (thisStart < prevEnd - 1e-6) overlaps++
          }
          if (list[0].lengthCm < lengthCm - 1e-9) shortened++
        }
        // Every shortened joint must SAY so — the report-don't-veto contract.
        const crowdedJoints = new Set(C.warnings.filter((w) => w.code === CROWDED_CODE).map((w) => w.joint))
        for (const [ji, list] of byJoint) {
          if (list[0].lengthCm < lengthCm - 1e-9 && !crowdedJoints.has(ji)) overlaps++
        }
      }
    }
  }
  ok(overlaps === 0, 'no overlapping parts and no silent shortening across 27 settings')
  ok(shortened > 0, `crowding is actually reachable (${shortened} shortened joints seen) — a non-vacuous check`)

  // 6 parts on a 60cm edge is the crowding case, and it must still be reported.
  const C = solveConnectors(cfgOf({ connectors: { lengthCm: 30, spacingCm: 200, minPerJoint: 6 } }))
  const w = C.warnings.filter((x) => x.code === CROWDED_CODE)
  ok(w.length > 0, 'W_CONNECTOR_CROWDED is raised when parts do not fit at full length')
  ok(w.every((x) => x.placedLengthCm < x.requestedLengthCm), 'the warning carries what was actually placed')
}

// -----------------------------------------------------------------------------
// 4. The station geometry agrees with placement.js, independently recomputed.
// -----------------------------------------------------------------------------
console.log('4. frames agree with an independent recomputation')
{
  const cfg = cfgOf({ form: { amplitude: 90 }, gap: 2 })
  const L = solveLayout(cfg)
  const C = solveConnectors(cfg, L)
  const byId = new Map(L.tiles.map((t) => [t.id, t]))
  let badSpan = 0
  let badMid = 0
  let badPoint = 0
  let badDihedral = 0
  for (const st of C.stations) {
    const edge = L.adjacency[st.jointIndex]
    const A = byId.get(st.a)
    const B = byId.get(st.b)
    const pa = jointEdgePoint(A, edge, true, st.s)
    const pb = jointEdgePoint(B, edge, false, st.s)
    if (Math.abs(pa.distanceTo(pb) - st.spanCm) > 1e-6) badSpan++
    if (pa.distanceTo(new THREE.Vector3(...st.aFrame.point)) > 1e-6) badPoint++
    if (pb.distanceTo(new THREE.Vector3(...st.bFrame.point)) > 1e-6) badPoint++
    const mid = pa.clone().add(pb).multiplyScalar(0.5)
    if (mid.distanceTo(new THREE.Vector3(...st.mid)) > 1e-6) badMid++
    const nA = new THREE.Vector3(...A.normal)
    const nB = new THREE.Vector3(...B.normal)
    const dih = Math.acos(Math.min(1, Math.max(-1, nA.dot(nB)))) * 180 / Math.PI
    if (Math.abs(dih - st.dihedralDeg) > 1e-6) badDihedral++
  }
  ok(badSpan === 0, 'spanCm is the rim-to-rim distance at the station')
  ok(badPoint === 0, 'aFrame/bFrame points are the rim points')
  ok(badMid === 0, 'mid is the midpoint of the two rims')
  ok(badDihedral === 0, 'dihedralDeg is the angle between the two lit normals')

  // --- the SIGN of the fold, checked against an independent physical fact ---
  // A convex joint is a ridge: the lit faces diverge and the HOUSINGS pinch. So
  // sign(foldDeg) must agree with sign(lit separation − housing-back separation),
  // measured on the panel solids and owing nothing to the frame construction
  // that produced foldDeg. A left-handed frame or a dropped negation flips one
  // and not the other. HANDOFF §6 records what that class of bug costs here.
  //
  // Both separations are measured IN THE CROSS-SECTION PLANE — the r̂ component
  // removed — because `foldDeg` is by definition the fold ABOUT r̂, which is what
  // the part's cross-section is cut perpendicular to. The raw 3D separation also
  // carries the twist, and on a joint where twist dominates fold the two
  // disagree legitimately: measured 9 such stations, every one of them with
  // twist larger than fold. Stations flatter than half a degree are skipped for
  // the same reason — at 0.18° of fold against 6.6° of twist there is no fold
  // sign left to check.
  {
    const TH = PANEL_PROFILE.overallThickness
    let disagree = 0
    let convex = 0
    let concave = 0
    for (const st of C.stations) {
      if (Math.abs(st.foldDeg) < 0.5) continue
      const edge = L.adjacency[st.jointIndex]
      const A = byId.get(st.a)
      const B = byId.get(st.b)
      const pa = jointEdgePoint(A, edge, true, st.s)
      const pb = jointEdgePoint(B, edge, false, st.s)
      const qa = pa.clone().addScaledVector(new THREE.Vector3(...A.normal), -TH)
      const qb = pb.clone().addScaledVector(new THREE.Vector3(...B.normal), -TH)
      const rHat = new THREE.Vector3(...st.frame.r)
      const acrossJoint = (u, v) => {
        const d = v.clone().sub(u)
        return d.addScaledVector(rHat, -d.dot(rHat)).length()
      }
      const pinch = acrossJoint(pa, pb) - acrossJoint(qa, qb)  // >0 when the back pinches
      if (st.foldDeg > 0) convex++; else concave++
      if (Math.sign(pinch) !== Math.sign(st.foldDeg)) disagree++
    }
    ok(disagree === 0, 'the fold sign agrees with which side of the joint pinches, at every station')
    ok(convex > 0, `convex joints occur (${convex})`)
    // A drift is convex almost everywhere, so concave stations are rare and this
    // is coverage information as much as an assertion: if it ever fails, the
    // presets stopped exercising the other sign, not the code.
    ok(concave > 0, `concave joints also occur, so the sign check is non-vacuous (${concave})`)
  }

  // The same convention, asserted on the PROFILE alone — no layout, no presets,
  // closed form. Positive fold must bring the hooks (the housing side) closer
  // together than the rims, and negative fold must spread them.
  {
    const SPAN = 6
    // The lowest pair of points in the outline are the two hooks' INBOARD
    // corners — deeper than the mouth corners by the taper, which is exactly the
    // wedge grip this design is built around.
    const hookGapAt = (foldDeg) => {
      const pts = connectorProfile({ spanCm: SPAN, foldDeg }).points
      const lowest = Math.min(...pts.map(([, q]) => q))
      const bottom = pts.filter(([, q]) => Math.abs(q - lowest) < 0.35)
      ok(bottom.length === 2, `fold ${foldDeg}: exactly two inboard hook corners at the bottom`)
      return Math.max(...bottom.map(([p]) => p)) - Math.min(...bottom.map(([p]) => p))
    }
    const flat = hookGapAt(0)
    ok(hookGapAt(20) < flat, 'positive fold (convex) draws the hooks together — the housings pinch')
    ok(hookGapAt(-20) > flat, 'negative fold (concave) spreads the hooks apart')

    // Closed form, derived from the constants rather than recorded as a golden
    // number. Each inboard hook corner sits `back` along the panel's inward
    // direction and `nHook` along its normal, both of which rotate by half the
    // fold — so the separation is span + 2·back·cos(φ) + 2·nHook·sin(φ), with
    // nHook negative (it is below the rim plane).
    const back = CONNECTOR_PROFILE.gripCm + CONNECTOR_PROFILE.wallCm
    const nHook = -(PANEL_PROFILE.outerThickness + taperDepthAt(back) + CONNECTOR_PROFILE.hookCm)
    const predicted = (foldDeg) => {
      const phi = (foldDeg * Math.PI) / 180 / 2
      return SPAN + 2 * back * Math.cos(phi) + 2 * nHook * Math.sin(phi)
    }
    near(flat, predicted(0), 1e-9, 'flat: the bottom width is span + 2·(grip + wall)')
    near(hookGapAt(20), predicted(20), 1e-9, 'convex: the hooks close by the derived amount')
    near(hookGapAt(-20), predicted(-20), 1e-9, 'concave: the hooks open by the derived amount')
  }

  // The part straddles its footprint, so spanMin/Max must BRACKET the centre.
  let badBracket = 0
  for (const st of C.stations) {
    if (st.spanMinCm > st.spanCm + 1e-9 || st.spanMaxCm < st.spanCm - 1e-9) badBracket++
    if (st.spanSpreadCm < -1e-12) badBracket++
  }
  ok(badBracket === 0, 'spanMin ≤ span ≤ spanMax at every station')
  ok(SPAN_SAMPLES >= 3, 'the footprint is sampled at its ends and centre at least')
}

// -----------------------------------------------------------------------------
// 5. THE CLAIM: a short part sees far less variation than its joint has.
//
// This is the measurement the whole design rests on, so it is asserted as a
// COMPARISON — the same layout, parts sized 10cm against parts spanning the
// entire joint — rather than as an absolute bound that could pass vacuously.
// -----------------------------------------------------------------------------
console.log('5. short parts linearize the joint (the design claim)')
{
  for (const preset of ['dune', 'crest']) {
    const base = buildPreset(preset)
    const L = solveLayout(base)

    const short = solveConnectors({ ...base, connectors: { ...DEFAULT_CONNECTORS, lengthCm: 10 } }, L)
    // One part per joint at CONNECTOR_LENGTH_MAX — the longest the tool allows.
    const long = solveConnectors(
      { ...base, connectors: { ...DEFAULT_CONNECTORS, lengthCm: 30, spacingCm: 200, minPerJoint: 1 } },
      L,
    )
    // And the limit case the design is actually arguing against: ONE rigid part
    // per joint, as long as the joint itself. Measured directly off the layout,
    // since the schema caps `lengthCm` well below a 121cm plate edge.
    const byId = new Map(L.tiles.map((t) => [t.id, t]))
    let worstWhole = 0
    for (const edge of L.adjacency) {
      const A = byId.get(edge.a)
      const B = byId.get(edge.b)
      if (!A?.position || !B?.position) continue
      let lo = Infinity
      let hi = -Infinity
      for (let m = 0; m <= 20; m++) {
        const s = edge.edge.from + (edge.edge.to - edge.edge.from) * (m / 20)
        const d = jointEdgePoint(A, edge, true, s).distanceTo(jointEdgePoint(B, edge, false, s))
        lo = Math.min(lo, d)
        hi = Math.max(hi, d)
      }
      worstWhole = Math.max(worstWhole, hi - lo)
    }

    const worstShort = Math.max(...short.stations.map((s) => s.spanSpreadCm))
    const worstLong = Math.max(...long.stations.map((s) => s.spanSpreadCm))
    ok(worstShort < worstLong,
      `${preset}: a 10cm part sees less span variation than a 30cm one ` +
      `(${worstShort.toFixed(2)}cm vs ${worstLong.toFixed(2)}cm)`)
    ok(worstShort < worstWhole / 3,
      `${preset}: a 10cm part sees under a third of what a joint-length part would ` +
      `(${worstShort.toFixed(2)}cm vs ${worstWhole.toFixed(2)}cm)`)
    ok(worstShort < 3,
      `${preset}: worst variation inside one 10cm part stays under 3cm (${worstShort.toFixed(2)}cm)`)
    console.log(
      `   ${preset}: 10cm ${worstShort.toFixed(2)}cm | 30cm ${worstLong.toFixed(2)}cm | ` +
      `whole joint ${worstWhole.toFixed(2)}cm`)
  }
}

// -----------------------------------------------------------------------------
// 6. Determinism, and every preset produces a usable set.
// -----------------------------------------------------------------------------
console.log('6. determinism and preset sweep')
{
  for (const id of PRESET_IDS) {
    const cfg = buildPreset(id)
    const a = JSON.stringify(solveConnectors(cfg))
    const b = JSON.stringify(solveConnectors(cfg))
    ok(a === b, `${id}: byte-identical on re-solve`)

    const C = solveConnectors(cfg)
    const L = solveLayout(cfg)
    ok(C.perJoint.length === L.adjacency.length, `${id}: every joint is connected`)
    ok(C.stations.length >= 2 * L.adjacency.length, `${id}: at least 2 parts per joint at the defaults`)

    let bad = 0
    for (const st of C.stations) {
      if (!Number.isFinite(st.spanCm) || st.spanCm <= 0) bad++
      if (!Number.isFinite(st.dihedralDeg) || st.dihedralDeg < 0 || st.dihedralDeg > 180) bad++
      if (!Number.isFinite(st.twistDeg) || st.twistDeg < 0) bad++
      if (!st.mid.every(Number.isFinite)) bad++
      for (const f of [st.aFrame, st.bFrame]) {
        if (Math.abs(new THREE.Vector3(...f.normal).length() - 1) > 1e-6) bad++
        if (Math.abs(new THREE.Vector3(...f.run).length() - 1) > 1e-6) bad++
        if (Math.abs(new THREE.Vector3(...f.inward).length() - 1) > 1e-6) bad++
      }
    }
    ok(bad === 0, `${id}: every station is finite with unit frames`)

    const ids = new Set(C.stations.map((s) => s.id))
    ok(ids.size === C.stations.length, `${id}: station ids are unique`)
  }

  // Solving with an injected layout must match solving without one.
  const cfg = buildPreset('drift')
  ok(JSON.stringify(solveConnectors(cfg)) === JSON.stringify(solveConnectors(cfg, solveLayout(cfg))),
    'injecting the layout changes nothing')
}

// -----------------------------------------------------------------------------
// 7. Schema: additive, backward compatible, and reports rather than clamps.
// -----------------------------------------------------------------------------
console.log('7. schema block')
{
  // A config written before connectors existed still normalizes and validates.
  const legacy = { ...DEFAULT_CONFIG }
  delete legacy.connectors
  const n = normalizeConfig(legacy)
  ok(n.connectors.lengthCm === DEFAULT_CONNECTORS.lengthCm, 'an omitted connectors block defaults')
  ok(validateConfig(legacy).valid, 'a pre-connectors config still validates')
  ok(n.version === 3, 'adding connectors did not bump the schema version')

  // Idempotence, the file's standing contract.
  ok(JSON.stringify(normalizeConfig(n)) === JSON.stringify(n), 'normalizeConfig stays idempotent')

  // Out of range is REPORTED, not silently corrected.
  const codes = (over) => validateConfig({ ...DEFAULT_CONFIG, connectors: { ...DEFAULT_CONNECTORS, ...over } })
    .errors.map((e) => e.code)
  ok(codes({ lengthCm: 999 }).includes('E_RANGE'), 'lengthCm out of range raises E_RANGE')
  ok(codes({ lengthCm: 0 }).includes('E_RANGE'), 'lengthCm below range raises E_RANGE')
  ok(codes({ spacingCm: 1 }).includes('E_RANGE'), 'spacingCm out of range raises E_RANGE')
  ok(codes({ binAngleDeg: 0 }).includes('E_RANGE'), 'binAngleDeg out of range raises E_RANGE')
  ok(codes({ binSpanCm: 99 }).includes('E_RANGE'), 'binSpanCm out of range raises E_RANGE')
  ok(codes({ lengthCm: 'ten' }).includes('E_SHAPE'), 'a non-numeric lengthCm raises E_SHAPE')
  ok(codes({ minPerJoint: 2.5 }).includes('E_SHAPE'), 'a fractional minPerJoint raises E_SHAPE — parts are counted')
  ok(codes({ minPerJoint: 2 }).length === 0, 'a legal block raises nothing')

  // And normalize still clamps rather than throwing, so a slider drag survives.
  const clamped = normalizeConfig({ ...DEFAULT_CONFIG, connectors: { lengthCm: 999, minPerJoint: 99 } })
  ok(clamped.connectors.lengthCm === 30, 'lengthCm clamps to its max')
  ok(clamped.connectors.minPerJoint === 6, 'minPerJoint clamps to its max')

  // minPerJoint 1 is legal but says why it is a bad idea.
  const warns = validateConfig({ ...DEFAULT_CONFIG, connectors: { ...DEFAULT_CONNECTORS, minPerJoint: 1 } })
  ok(warns.valid, 'minPerJoint 1 is legal')
  ok(warns.warnings.some((w) => w.code === 'W_SINGLE_CONNECTOR_JOINTS'), 'a one-part joint is flagged as a hinge')
}

console.log(`\ntest-v3-connectors: ${passed} checks passed, ${failed} failed`)
process.exit(failed ? 1 : 0)
