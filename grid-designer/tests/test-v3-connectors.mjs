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
  backHalfProfile,
  frontBarProfile,
  solveFrontBars,
  SPAN_SAMPLES,
  CROWDED_CODE,
  CONNECTOR_PROFILE,
  flangeDepthAt,
  BLOCKED_CODE,
  blockedSpansOnJoint,
  clearSpans,
  connectorStationFlags,
  supplyProudAt,
  supplyReliefCm,
  fastenerGapNeededCm,
  gapAtDepthCm,
  frontBarBand,
} from '../src/core/v3/connectors.js'
import { solveLayout, jointEdgePoint } from '../src/core/v3/placement.js'
import { DEFAULT_CONFIG, DEFAULT_CONNECTORS, normalizeConfig, validateConfig } from '../src/core/v3/schema.js'
import { PANEL_PROFILE, POWER_SUPPLY, poweredEdgeBlockedSpan } from '../src/config.js'
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
  // closed form. Positive fold must bring the FLANGE LIPS (the deep side of the
  // clamp) closer together than the rims, and negative fold must spread them.
  {
    const SPAN = 6
    const bg = Math.min(CONNECTOR_PROFILE.backGripCm, PANEL_PROFILE.flangeWidth)
    const lipDepth = flangeDepthAt(bg)
    // The two flange lip contact points, taken by INDEX: backHalfProfile emits
    // [A floor, B floor, B lip, B wall, B split, A split, A wall, A lip], so the
    // lips are 2 and 7.
    const lipGapAt = (foldDeg) => {
      const pts = backHalfProfile({ spanCm: SPAN, foldDeg }).points
      return Math.abs(pts[2][0] - pts[7][0])
    }
    const flat = lipGapAt(0)
    ok(lipGapAt(20) < flat, 'positive fold (convex) draws the flange lips together — the backs pinch')
    ok(lipGapAt(-20) > flat, 'negative fold (concave) spreads the flange lips apart')

    // Closed form from the constants, not a golden number: each lip sits `bg`
    // along its panel's inward direction and `lipDepth` along its normal, both
    // of which rotate by half the fold.
    const predicted = (foldDeg) => {
      const phi = (foldDeg * Math.PI) / 180 / 2
      return SPAN + 2 * bg * Math.cos(phi) - 2 * lipDepth * Math.sin(phi)
    }
    near(flat, predicted(0), 1e-9, 'flat: the lip separation is span + 2·backGrip')
    near(lipGapAt(20), predicted(20), 1e-9, 'convex: the lips close by the derived amount')
    near(lipGapAt(-20), predicted(-20), 1e-9, 'concave: the lips open by the derived amount')
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
    ok(C.perJoint.length === L.adjacency.length, `${id}: every joint is accounted for`)
    // NOT "2 parts per joint": the power supply blocks the middle 50cm of one
    // edge per panel, and a 10cm part does not fit the 5cm left at each end. So
    // the default build genuinely leaves joints unconnected — see §8.
    const unpowered = solveConnectors({ ...cfg, connectors: { ...DEFAULT_CONNECTORS, powerEdge: 'none' } })
    ok(unpowered.stations.length >= 2 * L.adjacency.length,
      `${id}: 2 parts per joint once the power supply is ignored`)

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

// -----------------------------------------------------------------------------
// 8. THE POWER SUPPLY — an edge a flange-gripping part cannot use.
//
// A 50cm box on the back of one 60cm edge, sitting directly on the 3cm flange
// the connector grips. Only ~5cm at each end of that edge is usable, so a 10cm
// part does not fit a powered edge AT ALL. That is the single most consequential
// thing the corrected panel geometry brought with it, so it is asserted as a
// measured consequence rather than described.
// -----------------------------------------------------------------------------
console.log('8. the power supply')
{
  const withPS = (over) => solveConnectors(cfgOf({ connectors: { ...over } }))

  // --- RELIEF (the default) --------------------------------------------------
  // The supply is flush with the flange to within 1mm, so a relief in the back
  // half's lip clears it and the joint is connected — but the lip then bears on
  // the supply HOUSING rather than the panel frame, which is reported.
  const off = withPS({ powerEdge: 'none' })
  const relief = withPS({ powerEdge: 'low', supplyMode: 'relief' })
  ok(relief.stations.length === off.stations.length,
    'in relief mode the supply costs no parts at all')
  const bearing = relief.stations.filter((st) => st.bearsOnPowerSupply).length
  ok(bearing > 0, `but ${bearing} parts land over a supply and are flagged`)
  ok(off.stations.filter((st) => st.bearsOnPowerSupply).length === 0,
    'and none are flagged once the supply is gone — the flag tracks the supply')

  // The relief is small, and derived rather than assumed.
  const relCm = supplyReliefCm(CONNECTOR_PROFILE.backGripCm)
  ok(relCm > 0, 'the supply does stand proud of the flange, so a relief is needed')
  ok(relCm <= 0.1 + 1e-9, `and only ${(relCm * 10).toFixed(1)}mm of it — a printable pocket`)
  near(supplyProudAt(0), 0, 1e-12, 'it is flush at the panel edge')
  near(supplyProudAt(PANEL_PROFILE.flangeWidth), PANEL_PROFILE.flangeDrop, 1e-12,
    'and proud by exactly the flange drop at the flange inner edge')

  // --- BLOCK (the stricter reading) -----------------------------------------
  const block = withPS({ powerEdge: 'low', supplyMode: 'block' })
  ok(block.stations.length < relief.stations.length,
    `treating the supply as solid costs parts (${relief.stations.length} → ${block.stations.length})`)
  const blocked = block.warnings.filter((w) => w.code === BLOCKED_CODE)
  ok(blocked.length > 0, `and leaves joints with nothing (${blocked.length})`)

  // Every blocked joint really does touch a powered edge and really has no room.
  const cfg = cfgOf({ connectors: { supplyMode: 'block' } })
  const L = solveLayout(cfg)
  const byId = new Map(L.tiles.map((t) => [t.id, t]))
  let wrong = 0
  for (const w of blocked) {
    const edge = L.adjacency[w.joint]
    const spans = blockedSpansOnJoint(edge, byId.get(edge.a), byId.get(edge.b), 'low')
    if (spans.length === 0) wrong++
    if (clearSpans(spans, edge.edge.from, edge.edge.to).some(([x, y]) => y - x >= cfg.connectors.lengthCm)) wrong++
  }
  ok(wrong === 0, 'every blocked joint touches a powered edge and has no stretch long enough')

  const blockedAt = (lengthCm) =>
    withPS({ lengthCm, powerEdge: 'low', supplyMode: 'block' })
      .warnings.filter((w) => w.code === BLOCKED_CODE).length
  ok(blockedAt(5) === 0, 'blocking: a 5cm part fits the clear end exactly')
  ok(blockedAt(6) > 0, 'blocking: a 6cm part does not')
  console.log(`   relief: ${relief.stations.length} parts, ${bearing} on a supply, ` +
    `${(relCm * 10).toFixed(1)}mm relief · block: ${block.stations.length} parts, ${blocked.length} joints empty`)

  // Where the supply sits is a real choice: 'high' affects a different set.
  const lo = new Set(withPS({ powerEdge: 'low', supplyMode: 'block' })
    .warnings.filter((w) => w.code === BLOCKED_CODE).map((w) => w.joint))
  const hi = new Set(withPS({ powerEdge: 'high', supplyMode: 'block' })
    .warnings.filter((w) => w.code === BLOCKED_CODE).map((w) => w.joint))
  ok(hi.size > 0, 'the "high" convention blocks joints too')
  ok([...lo].some((j) => !hi.has(j)) || [...hi].some((j) => !lo.has(j)),
    'and a different set of them — panel orientation genuinely matters')

  const [from, to] = poweredEdgeBlockedSpan(60)
  near(from, (60 - POWER_SUPPLY.length) / 2, 1e-12, 'the blocked span is centred on the edge')
  near(to - from, POWER_SUPPLY.length, 1e-12, 'and is exactly as long as the supply')
}

// -----------------------------------------------------------------------------
// 9. THE FASTENER — three small bolts, and the gap they need.
// -----------------------------------------------------------------------------
console.log('9. the fastener needs gap, measured at depth')
{
  const need = fastenerGapNeededCm()
  near(need, Math.max(CONNECTOR_PROFILE.bolt.headCm, CONNECTOR_PROFILE.bolt.insertOdCm)
    + 2 * CONNECTOR_PROFILE.wallCm, 1e-12, 'the need is the head or insert, plus a wall each side')
  ok(need <= 1.0 + 1e-9, `three ${CONNECTOR_PROFILE.bolt.name}s need only ${need.toFixed(2)}cm of gap`)

  // The gap CLOSES with depth on a convex joint, and the insert sits deep.
  ok(gapAtDepthCm(3, 35, 1.7) < 3, 'a convex fold narrows the gap at depth')
  ok(gapAtDepthCm(3, -35, 1.7) > 3, 'a concave fold opens it')
  near(gapAtDepthCm(3, 0, 1.7), 3, 1e-12, 'and a flat joint does neither')

  // The flag fires where it should and is silent where it should not.
  const stub = (o) => ({ id: 'T', jointIndex: 0, lengthCm: 10, spanStartCm: 3, spanEndCm: 3,
    spanMinCm: 3, spanMaxCm: 3, spanSpreadCm: 0, foldDeg: 0, ...o })
  ok(!connectorStationFlags(stub({})).includes('W_FASTENER_PINCHED'), 'a 3cm gap takes the fastener')
  ok(connectorStationFlags(stub({ spanMinCm: 0.7, spanMaxCm: 0.7 })).includes('W_FASTENER_PINCHED'),
    'a 0.7cm gap does not')
  ok(connectorStationFlags(stub({ spanMinCm: 1.2, spanMaxCm: 1.2, foldDeg: 30 }))
    .includes('W_FASTENER_PINCHED'),
    'and a 1.2cm gap that would pass at the face fails once the fold closes it at depth')
}

// -----------------------------------------------------------------------------
// 10. THE UNIVERSAL FRONT BAR.
// -----------------------------------------------------------------------------
console.log('10. one bar serves a band of gaps')
{
  const band = frontBarBand(6)
  near(band[0], 6 - 2 * PANEL_PROFILE.bezelWidth, 1e-12, 'the narrowest gap gives the bar a full bezel lip')
  near(band[1], 6 - 2 * CONNECTOR_PROFILE.frontMinLipCm, 1e-12, 'the widest leaves the minimum bearing')
  const width = band[1] - band[0]
  near(width, 2 * (PANEL_PROFILE.bezelWidth - CONNECTOR_PROFILE.frontMinLipCm), 1e-12,
    'so the band is set by the bezel width, not by the bar')

  // Greedy cover is exact for fixed-width intervals on a line.
  const bars = solveFrontBars([2, 2.5, 3, 5.5, 6])
  ok(bars.length === 2, `five gaps spanning 4cm need 2 bars (got ${bars.length})`)
  let uncovered = 0
  for (const g of [2, 2.5, 3, 5.5, 6]) {
    if (!bars.some((b) => { const [lo, hi] = frontBarBand(b.widthCm); return g >= lo - 1e-9 && g <= hi + 1e-9 })) uncovered++
  }
  ok(uncovered === 0, 'and every gap is covered by one of them')
  ok(solveFrontBars([3, 3, 3]).length === 1, 'identical gaps need one bar')

  // The bar really is a plain rectangle — that is what makes it universal.
  const prof = frontBarProfile(5)
  ok(prof.points.length === 4, 'the bar section is a rectangle')
  near(prof.extents.width, 5, 1e-12, 'as wide as asked')
  const other = frontBarProfile(5)
  ok(JSON.stringify(prof.points) === JSON.stringify(other.points),
    'and depends on nothing but its width — no fold, no station')
}

console.log(`\ntest-v3-connectors: ${passed} checks passed, ${failed} failed`)
process.exit(failed ? 1 : 0)
