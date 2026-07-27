/**
 * tests/test-v4-connectors.mjs — headless checks for core/v4/connectors.js.
 *
 * Plain node script, no test framework. Exits non-zero on any failure.
 *   node tests/test-v4-connectors.mjs
 *
 * Three things this file exists to pin down, all of which would fail silently:
 *
 *   THE STATION SHAPE. v3's part machinery (`connectorEndProfiles`,
 *        `connectorOBB`, `connectorStationFlags`) and the geometry module all
 *        consume these objects. §1 compares a v4 station's key set against a
 *        REAL v3 station, field for field and nested object for nested object,
 *        rather than against a list typed out here — a list would drift the
 *        moment v3 gained a field.
 *   THE FOLD'S SIGN. §2 checks the v3 p̂/q̂/r̂ construction against the closed form
 *        (−θ at a ramp's ground end, +θ at its high end), AND checks the sign is
 *        the same in all four ramp directions. A magnitude test alone passes
 *        happily with the sign inverted, which would print every back half with
 *        its hooks the wrong way.
 *   THE JOINT NO LONGER RUNNING ALONG X. §3. On the ribbon every joint ran along
 *        world +X and both panels agreed about which way. On the network the run
 *        axis is X or Z depending on the lattice edge, and the two panels can
 *        number their width axes in OPPOSITE senses. Anything that quietly
 *        assumed local X, or a signed twist, is wrong here and nowhere else.
 */

import * as THREE from 'three'
import {
  solveConnectorsV4,
  jointFrame,
  blockedSpansOnJointV4,
  poweredRimOf,
  FLIP_MISMATCH_CODE,
} from '../src/core/v4/connectors.js'
import { solveLattice } from '../src/core/v4/lattice.js'
import { DEFAULT_CONFIG } from '../src/core/v4/schema.js'
import {
  solveConnectors,
  connectorOBB,
  connectorEndProfiles,
  connectorStationFlags,
  stationCount,
  CONNECTOR_LIMITS,
  BLOCKED_CODE,
  REDUCED_CODE,
  CROWDED_CODE,
} from '../src/core/v3/connectors.js'
import { DEFAULT_CONFIG as V3_DEFAULT_CONFIG } from '../src/core/v3/schema.js'
import { POWER_SUPPLY } from '../src/config.js'

let passed = 0
let failed = 0
const ok = (c, m) => { if (c) passed++; else { failed++; console.log(`  FAIL: ${m}`) } }
const near = (a, b, tol, m) => ok(Math.abs(a - b) <= tol, `${m} (got ${a}, want ${b} ±${tol})`)

const solve = (cfg = {}) => {
  const C = solveLattice(cfg)
  return { lattice: C, ...solveConnectorsV4(cfg, C) }
}
/** The default 3 × 5 network: 15 cells, 22 ramps, 44 joints. */
const RIBBON = { lattice: { cols: 1, rows: 5, panelType: '2x2' } }

console.log('=== test-v4-connectors ===')

// -----------------------------------------------------------------------------
// 1. THE STATION SHAPE, against a real v3 station.
// -----------------------------------------------------------------------------
console.log('1. a v4 station is shaped exactly like a v3 station')
{
  const v3 = solveConnectors(V3_DEFAULT_CONFIG).stations[0]
  const v4 = solve().stations[0]
  ok(v3 && v4, 'both solvers produce stations to compare')

  const keys = (o) => Object.keys(o).sort().join(',')
  ok(keys(v4) === keys(v3), `top-level keys match\n    v3: ${keys(v3)}\n    v4: ${keys(v4)}`)
  ok(keys(v4.frame) === keys(v3.frame), 'frame keys match')
  ok(keys(v4.aFrame) === keys(v3.aFrame), 'aFrame keys match')
  ok(keys(v4.bFrame) === keys(v3.bFrame), 'bFrame keys match')

  // Types match too — a key of the right name holding the wrong kind of thing is
  // the failure mode a key-set comparison misses.
  const kind = (v) => (Array.isArray(v) ? `array[${v.length}]` : typeof v)
  let badType = 0
  for (const k of Object.keys(v3)) {
    if (k === 'frame' || k === 'aFrame' || k === 'bFrame') continue
    if (kind(v3[k]) !== kind(v4[k])) { badType++; console.log(`    (${k}: v3 ${kind(v3[k])} vs v4 ${kind(v4[k])})`) }
  }
  ok(badType === 0, 'every station field holds the same kind of value as v3\'s')

  // And v3's part machinery consumes it without complaint.
  const profiles = connectorEndProfiles(v4)
  ok(profiles.start.points.length === 8 && profiles.end.points.length === 8,
    'connectorEndProfiles lofts a back half from a v4 station')
  const box = connectorOBB(v4)
  ok(box.center.length === 3 && box.halfExtents.length === 3 && box.quaternion.length === 3 + 1,
    'connectorOBB builds a box from a v4 station')
  ok(box.halfExtents.every((h) => h > 0), 'and the box has positive extents')
  ok(Array.isArray(connectorStationFlags(v4, CONNECTOR_LIMITS)), 'connectorStationFlags judges it')

  // The station frame is right-handed, which connectorOBB relies on — checked
  // over a lattice big enough to carry all four ramp directions.
  let bad = 0
  for (const st of solve({ lattice: { cols: 4, rows: 4 }, angleDeg: 40 }).stations) {
    const p = new THREE.Vector3(...st.frame.p)
    const q = new THREE.Vector3(...st.frame.q)
    const r = new THREE.Vector3(...st.frame.r)
    if (p.clone().cross(q).distanceTo(r) > 1e-9) bad++
    if (Math.abs(p.length() - 1) > 1e-9 || Math.abs(q.length() - 1) > 1e-9 ||
        Math.abs(r.length() - 1) > 1e-9) bad++
    // q̂ must agree with the lit side — that is what fixes the fold's sign.
    const n = new THREE.Vector3(...st.aFrame.normal).add(new THREE.Vector3(...st.bFrame.normal))
    if (n.lengthSq() > 1e-12 && q.dot(n) < 0) bad++
  }
  ok(bad === 0, '(p̂, q̂, r̂) is right-handed, orthonormal, and lit-side-up at every station')

  // The two panels reach in from opposite sides of every joint — the check that
  // catches a dropped sign on `inward`.
  let badInward = 0
  for (const st of solve({ lattice: { cols: 4, rows: 4 }, angleDeg: 25 }).stations) {
    const ia = new THREE.Vector3(...st.aFrame.inward)
    const ib = new THREE.Vector3(...st.bFrame.inward)
    if (Math.abs(ia.length() - 1) > 1e-9 || Math.abs(ib.length() - 1) > 1e-9) badInward++
    if (ia.dot(ib) > 0) badInward++
    // ...and each is perpendicular to its own panel's normal.
    if (Math.abs(ia.dot(new THREE.Vector3(...st.aFrame.normal))) > 1e-9) badInward++
    if (Math.abs(ib.dot(new THREE.Vector3(...st.bFrame.normal))) > 1e-9) badInward++
  }
  ok(badInward === 0, 'the two panels reach in from opposite sides, each in its own plane')
}

// -----------------------------------------------------------------------------
// 2. THE FOLD — v3's formula against the closed form, magnitude AND sign.
// -----------------------------------------------------------------------------
console.log('2. foldDeg')
{
  let worst = 0
  let checks = 0
  const dirs = new Set()
  for (const angleDeg of [0, 3, 12.5, 30, 45, 60, 75]) {
    for (const gap of [0.5, 2, 6]) {
      const C = solveLattice({ lattice: { cols: 3, rows: 3 }, angleDeg, gap })
      const M = new Map(C.panels.map((p) => [p.id, p]))
      for (const j of C.joints) {
        const f = jointFrame(j)
        // CLOSED FORM: a ramp meets the floor in a valley and the high level on
        // a ridge, whichever way it runs. (The ribbon's α_A − α_B does not
        // survive: a descending ramp is built rising out of its own ground cell,
        // so the A/B ordering no longer tracks the tilt's sign.)
        const expect = j.end === 'lo' ? -angleDeg : angleDeg
        worst = Math.max(worst, Math.abs(f.foldDeg - expect))
        const ramp = M.get(j.panelA === 'ramp' ? j.a : j.b)
        dirs.add(`${ramp.axis}${ramp.role}${j.end}`)
        checks++
      }
    }
  }
  ok(worst <= 1e-9,
    `over ${checks} joints the p̂/q̂/r̂ formula equals ∓θ by end (worst ${worst})`)
  ok(dirs.size === 8, `and all four ramp directions × both ends are covered (${dirs.size})`)

  // THE SIGN, named explicitly on a 1-column lattice — the ribbon, where the
  // answer is already recorded.
  const C = solveLattice({ ...RIBBON, angleDeg: 30 })
  const fold = Object.fromEntries(C.joints.map((j) => [j.id, jointFrame(j).foldDeg]))
  near(fold['Ci0j0>Ei0j0z'], -30, 1e-9, 'the first ramp\'s ground end is CONCAVE — a valley, negative')
  near(fold['Ei0j0z>Ci0j1'], 30, 1e-9, 'its high end is CONVEX — a ridge, positive')
  near(fold['Ei0j1z>Ci0j1'], 30, 1e-9, 'the descending ramp meets the SAME ridge, also positive')
  near(fold['Ci0j2>Ei0j1z'], -30, 1e-9, 'and its own ground end is the valley at the floor')
  ok(C.joints.every((j) => Math.abs(Math.abs(jointFrame(j).foldDeg) - 30) < 1e-9),
    'every fold has magnitude θ, not 2θ — that is what the flats buy')

  // In two dimensions the same holds in every direction: the tops of the high
  // cells are ridges and their feet are valleys, in x as in z.
  const wide = solveLattice({ lattice: { cols: 3, rows: 3 }, angleDeg: 30 })
  const senses = wide.joints.map((j) => (jointFrame(j).foldDeg > 0 ? '+' : '-')).join('')
  const wanted = wide.joints.map((j) => (j.end === 'hi' ? '+' : '-')).join('')
  ok(senses === wanted, `sense follows the end, never the direction (got ${senses})`)

  // dihedralDeg is |foldDeg| exactly (V4_SPEC §3), not an independent acos.
  for (const st of solve({ lattice: { cols: 3, rows: 3 }, angleDeg: 30 }).stations) {
    ok(st.dihedralDeg === Math.abs(st.foldDeg), `${st.id}: dihedralDeg is |foldDeg| exactly`)
  }
  // A flat design folds by nothing at all — and atan2 gets that exactly right
  // where an acos would return the square root of its own rounding noise.
  for (const st of solve({ lattice: { cols: 2, rows: 2 }, angleDeg: 0 }).stations) {
    ok(st.foldDeg === 0 && st.dihedralDeg === 0, `${st.id}: a flat joint folds by exactly 0`)
  }
}

// -----------------------------------------------------------------------------
// 3. THE RUN AXIS — a joint runs along Ŷ × ê, which is X or Z.
// -----------------------------------------------------------------------------
console.log('3. joints no longer run along world X')
{
  const K = solve({ lattice: { cols: 3, rows: 3 }, angleDeg: 30 })
  const M = new Map(K.lattice.panels.map((p) => [p.id, p]))
  const axes = new Set(K.stations.map((st) => st.axis))
  ok(axes.size === 2 && axes.has('x') && axes.has('z'),
    `a 2-D network carries stations on both world axes (got ${[...axes]})`)

  // The run axis is the one PERPENDICULAR to the lattice edge: a z-axis edge's
  // ramp is joined along world X, an x-axis edge's along world Z. Getting this
  // backwards would put every station 90° out and would not otherwise show.
  let badAxis = 0
  for (const j of K.lattice.joints) {
    const ramp = M.get(j.panelA === 'ramp' ? j.a : j.b)
    const wantAxis = ramp.axis === 'z' ? 0 : 2
    if (j.runAxis !== wantAxis) badAxis++
    // The run interval is the panel's 60cm, centred on the rim.
    if (Math.abs((j.runTo - j.runFrom) - 60) > 1e-9) badAxis++
    if (Math.abs(j.rimA[j.runAxis] - (j.runFrom + 30)) > 1e-9) badAxis++
  }
  ok(badAxis === 0, 'every joint runs along the axis perpendicular to its lattice edge')

  // THE ANTIPARALLEL CASE, which is the reason the twist is measured between
  // LINES. A descending ramp's width axis is the negative of its cells'.
  let antiparallel = 0
  let parallel = 0
  for (const j of K.lattice.joints) {
    const d = new THREE.Vector3(...j.runA).dot(new THREE.Vector3(...j.runB))
    if (d < -0.5) antiparallel++
    else if (d > 0.5) parallel++
  }
  ok(antiparallel > 0,
    `some joints DO have their two width axes numbered opposite ways (${antiparallel} of ` +
    `${K.lattice.joints.length})`)
  ok(antiparallel + parallel === K.lattice.joints.length,
    'and every pair is either parallel or antiparallel — never skew')
  ok(K.stations.every((st) => st.twistDeg === 0),
    'yet nothing twists: the twist is between rim LINES, and a signed dot would read 180°')

  // The 1-column lattice is the ribbon, where every joint runs along world X.
  const ribbon = solve(RIBBON)
  ok(ribbon.stations.every((st) => st.axis === 'x'), 'the cols:1 lattice keeps every joint on world X')
  ok(ribbon.stations.every((st) => st.twistDeg === 0), 'and still nothing twists')
}

// -----------------------------------------------------------------------------
// 4. Spans — constant, parallel, and exactly the gap.
// -----------------------------------------------------------------------------
console.log('4. spans')
{
  for (const gap of [0.4, 1, 2, 5.5, 8]) {
    for (const angleDeg of [0, 30, 60]) {
      const K = solve({ lattice: { cols: 3, rows: 3 }, gap, angleDeg })
      let worst = 0
      let worstSpread = 0
      let worstTwist = 0
      for (const st of K.stations) {
        worst = Math.max(worst, Math.abs(st.spanCm - gap), Math.abs(st.spanMinCm - gap),
          Math.abs(st.spanMaxCm - gap), Math.abs(st.spanStartCm - gap), Math.abs(st.spanEndCm - gap))
        worstSpread = Math.max(worstSpread, st.spanSpreadCm)
        worstTwist = Math.max(worstTwist, st.twistDeg)
      }
      near(worst, 0, 1e-9, `gap ${gap} at ${angleDeg}°: every span number is the gap`)
      ok(worstSpread === 0, `gap ${gap} at ${angleDeg}°: nothing wedges along a part`)
      ok(worstTwist === 0, `gap ${gap} at ${angleDeg}°: nothing twists`)
    }
  }
  // A v4 part never wedges, so it never trips the twist limit — which is the
  // whole reason `connectors.lengthCm` was a delicate knob in v3 and is not here.
  const long = solve({ connectors: { ...DEFAULT_CONFIG.connectors, lengthCm: 30 } })
  ok(long.stations.every((st) => !connectorStationFlags(st, CONNECTOR_LIMITS).includes('W_CONNECTOR_TWIST')),
    'even a 30cm part never trips W_CONNECTOR_TWIST on a v4 joint')
}

// -----------------------------------------------------------------------------
// 5. Station counts and placement along the rim.
// -----------------------------------------------------------------------------
console.log('5. how many parts, and where')
{
  const K = solve()
  ok(K.lattice.joints.length === 44, `the default 3 × 5 has 22 ramps × 2 ends = 44 joints`)
  ok(K.stations.length === 88, `× 2 parts = 88 stations (got ${K.stations.length})`)
  ok(K.perJoint.length === 44, 'one perJoint row per joint')
  ok(K.perJoint.every((pj) => pj.count === 2 && pj.materialLength === 60),
    'each 60cm rim carries 2 parts at the default 50cm spacing')
  ok(stationCount(60, { spacingCm: 50, minPerJoint: 2 }) === 2, 'which is what stationCount says')

  // Centres are symmetric within the rim, in WORLD coordinates on the run axis.
  let badCentres = 0
  for (const j of K.lattice.joints) {
    const mine = K.stations.filter((s) => s.jointIndex === j.jointIndex).map((s) => s.s)
    if (mine.length !== 2) { badCentres++; continue }
    if (Math.abs(mine[0] - (j.runFrom + 15)) > 1e-9) badCentres++
    if (Math.abs(mine[1] - (j.runFrom + 45)) > 1e-9) badCentres++
  }
  ok(badCentres === 0, 'two parts per joint, at runFrom + 15 and + 45 — evenly spaced and symmetric')
  ok(K.stations.every((s) => s.of === 2), 'each knows the joint total')

  // The rim runs in WORLD coordinates, so an offset moves the stations with it —
  // and now in BOTH axes, which the ribbon could not test.
  const off = solve({ placement: { wallOffsetCm: 25, windowOffsetCm: 40, groundToFloor: true, wallAnchor: 'free' } })
  let bad = 0
  for (const j of off.lattice.joints) {
    const base = K.lattice.joints.find((b) => b.id === j.id)
    const delta = j.runFrom - base.runFrom
    if (Math.abs(delta - (j.runAxis === 0 ? 25 : 40)) > 1e-9) bad++
  }
  ok(bad === 0, 'x-axis joints follow the wall offset and z-axis joints the window offset')

  // minPerJoint is a floor, spacing is a ceiling.
  ok(solve({ connectors: { ...DEFAULT_CONFIG.connectors, minPerJoint: 4 } })
    .perJoint.every((pj) => pj.count === 4), 'minPerJoint raises the count')
  ok(solve({ connectors: { ...DEFAULT_CONFIG.connectors, spacingCm: 20 } })
    .perJoint.every((pj) => pj.count === 3), '20cm spacing on a 60cm rim gives 3')

  // Crowding: six 30cm parts cannot fit a 60cm rim at full length, so they are
  // SHORTENED and the cost is reported rather than the parts refused.
  const crowded = solve({ ...RIBBON,
    connectors: { ...DEFAULT_CONFIG.connectors, lengthCm: 30, minPerJoint: 6 } })
  ok(crowded.perJoint.every((pj) => pj.count === 6), 'six parts are placed on a 60cm rim')
  ok(crowded.perJoint.every((pj) => pj.lengthCm === 10), 'shortened from 30cm to 10cm to fit')
  ok(crowded.warnings.filter((w) => w.code === CROWDED_CODE).length === 8,
    'and every joint says so, rather than dropping a structural requirement')
}

// -----------------------------------------------------------------------------
// 6. The power supply.
// -----------------------------------------------------------------------------
console.log('6. the power supply')
{
  ok(poweredRimOf('low') === 'start' && poweredRimOf('high') === 'end' && poweredRimOf('none') === null,
    'poweredRimOf picks the rim by policy')

  const C = solveLattice({})
  const blocked = blockedSpansOnJointV4(C.joints[0], 'low')
  ok(blocked.length === 1, 'one of the two panels contributes the blocked span, never both')
  ok(JSON.stringify(blocked[0]) === JSON.stringify([C.joints[0].runFrom + 5, C.joints[0].runFrom + 55]),
    `the middle 50cm of a 60cm rim (got ${JSON.stringify(blocked[0])})`)
  ok(blocked[0][1] - blocked[0][0] === POWER_SUPPLY.length, 'which is exactly the supply box')
  ok(blockedSpansOnJointV4(C.joints[0], 'none').length === 0, '"none" blocks nothing')

  // 'relief' (the default): parts are placed, and flagged for what they bear on.
  const relief = solve()
  ok(relief.stations.every((s) => s.bearsOnPowerSupply),
    'in relief mode every station is placed and flagged as bearing on the supply')
  ok(relief.warnings.length === 0, 'and no joint is reported as blocked or reduced')
  ok(relief.stations.every((s) =>
    connectorStationFlags(s, CONNECTOR_LIMITS).includes('W_BEARS_ON_POWER_SUPPLY')),
    'which is what W_BEARS_ON_POWER_SUPPLY means')

  // 'block': only ~5cm survives at each end, too short for a 10cm part.
  const block = solve({ connectors: { ...DEFAULT_CONFIG.connectors, supplyMode: 'block' } })
  ok(block.stations.length === 0, 'in block mode a 10cm part has nowhere to go')
  ok(block.warnings.filter((w) => w.code === BLOCKED_CODE).length === 44,
    'and every one of the 44 joints reports W_JOINT_BLOCKED_BY_POWER_SUPPLY')
  ok(block.perJoint.every((pj) => pj.count === 0 && pj.blockedCm === 50), 'with 50cm blocked per joint')

  // A 4cm part fits the two clear ends, so the joint is REDUCED, not blocked.
  const short = solve({ connectors: { ...DEFAULT_CONFIG.connectors, supplyMode: 'block', lengthCm: 4 } })
  ok(short.perJoint.every((pj) => pj.count === 2), 'a 4cm part fits the 5cm clear ends, one each')
  ok(short.stations.every((s) => !s.bearsOnPowerSupply),
    'and none of them bears on the supply — they are outside it')
  let badShort = 0
  for (const j of short.lattice.joints) {
    const mine = short.stations.filter((s) => s.jointIndex === j.jointIndex).map((s) => s.s)
    if (Math.abs(mine[0] - (j.runFrom + 2.5)) > 1e-9) badShort++
    if (Math.abs(mine[1] - (j.runFrom + 57.5)) > 1e-9) badShort++
  }
  ok(badShort === 0, 'sitting in the middle of each clear stretch')
  ok(short.warnings.filter((w) => w.code === REDUCED_CODE).length === 0,
    'two parts is what was wanted, so nothing is reported as reduced')

  // 'none' removes the supply entirely — the measurement of what it costs.
  const none = solve({ connectors: { ...DEFAULT_CONFIG.connectors, powerEdge: 'none', supplyMode: 'block' } })
  ok(none.stations.length === 88 && none.stations.every((s) => !s.bearsOnPowerSupply),
    'powerEdge "none" gives the full 88 parts and no supply flag')

  // 'high' blocks the other panel's rim — the same interval, so the same outcome.
  const high = solve({ connectors: { ...DEFAULT_CONFIG.connectors, powerEdge: 'high', supplyMode: 'block' } })
  ok(high.stations.length === 0, 'the "high" policy blocks the same stretch from the other side')
}

// -----------------------------------------------------------------------------
// 7. A joint whose panels face opposite ways gets NO part.
//
// AND A FINDING WORTH RECORDING: on the network only CELLS can be flipped
// (V4_SPEC §9.7 gives an edge a `present` and nothing else), and a ramp is never
// flipped. So flipping a cell mismatches EVERY joint it has, and there is no
// "flip a whole run" move as there was on the ribbon. A flip is only free on a
// cell that is joined to nothing.
// -----------------------------------------------------------------------------
console.log('7. flip mismatch')
{
  const K = solve({ overrides: { cells: [{ i: 1, j: 2, flipped: true }], edges: [] } })
  const bad = K.warnings.filter((w) => w.code === FLIP_MISMATCH_CODE)
  ok(bad.length === 4, `an interior cell has four joints, and flipping it mismatches all four (got ${bad.length})`)
  ok(bad.every((w) => w.a === 'Ci1j2' || w.b === 'Ci1j2'), 'each names the flipped cell')
  ok(K.stations.length === 88 - 4 * 2, 'the four joints lose their two parts each (88 → 80)')
  ok(K.perJoint.length === 44, 'but every joint still has a row')
  ok(K.perJoint.filter((pj) => pj.count === 0).length === 4, 'four rows at count 0')

  // A CORNER cell has two joints, so flipping it costs two.
  const corner = solve({ overrides: { cells: [{ i: 0, j: 0, flipped: true }], edges: [] } })
  ok(corner.warnings.filter((w) => w.code === FLIP_MISMATCH_CODE).length === 2,
    'a corner cell has two ramps, so a flip there costs two joints')

  // Switch its edges off first and the flip is free — the only way to flip
  // without breaking a joint, and worth stating because it is the ONLY way.
  const isolated = solve({
    overrides: {
      cells: [{ i: 0, j: 0, flipped: true }],
      edges: [{ i: 0, j: 0, axis: 'x', present: false }, { i: 0, j: 0, axis: 'z', present: false }],
    },
  })
  ok(isolated.warnings.filter((w) => w.code === FLIP_MISMATCH_CODE).length === 0,
    'a cell joined to nothing can be flipped for free')

  // Flipping BOTH cells of a ramp still mismatches, because the RAMP is not
  // flipped and cannot be — the ribbon's "flip a run" move does not survive.
  const both = solve({
    overrides: { cells: [{ i: 0, j: 0, flipped: true }, { i: 1, j: 0, flipped: true }], edges: [] },
  })
  const bothBad = both.warnings.filter((w) => w.code === FLIP_MISMATCH_CODE)
  ok(bothBad.filter((w) => w.a === 'Ei0j0x' || w.b === 'Ei0j0x').length === 2,
    'flipping both cells of an edge mismatches BOTH the shared ramp\'s joints, not neither — a ramp ' +
    'has no flip, so the ribbon\'s "flip a run" move does not survive')
  ok(bothBad.length === 5,
    `and every other joint on both cells goes with them (2 + 3 joints, got ${bothBad.length})`)
}

// -----------------------------------------------------------------------------
// 8. Anchors, absent panels, and determinism.
// -----------------------------------------------------------------------------
console.log('8. anchors, holes and determinism')
{
  const braced = solve({ placement: { wallAnchor: 'braced', groundToFloor: true, wallOffsetCm: 0, windowOffsetCm: 0 } })
  ok(braced.lattice.joints.length === 46, 'bracing adds one joint per anchor ramp (44 → 46)')
  ok(braced.stations.length === 92, 'and two parts on each (88 → 92)')
  const anchorStations = braced.stations.filter((st) => {
    const j = braced.lattice.joints.find((x) => x.jointIndex === st.jointIndex)
    return j.anchor
  })
  ok(anchorStations.length === 4, 'four of them sit on anchor joints')
  ok(anchorStations.every((st) => st.axis === 'z'),
    'which run along world Z — an anchor is an x-axis ramp')
  ok(anchorStations.every((st) => Math.abs(st.foldDeg - 30) < 1e-9),
    'and each is an ordinary ridge at θ')

  // A cell switched off takes only the joints that MET it. Its four ramps stay,
  // cantilevered off their far cells (§9.3), so this is four joints and not
  // eight — one lost end per orphaned ramp.
  const holed = solve({ overrides: { cells: [{ i: 1, j: 2, present: false }], edges: [] } })
  ok(holed.lattice.joints.length === 44 - 4, 'a removed interior cell costs four joints')
  ok(holed.stations.length === (44 - 4) * 2, 'and eight parts with them')
  ok(holed.warnings.length === 0, 'without warning about it — a hole is a design, not a fault')

  ok(new Set(solve().stations.map((s) => s.id)).size === 88, 'every station id is unique')
  ok(solve().stations.every((s) => s.id === `J${s.jointIndex}S${s.index}`), 'ids follow the v3 scheme')

  const cfg = {
    lattice: { cols: 3, rows: 4, panelType: '2x2' },
    angleDeg: 41.5,
    gap: 1.7,
    placement: { wallOffsetCm: 5, windowOffsetCm: 5, groundToFloor: true, wallAnchor: 'braced' },
    overrides: { cells: [{ i: 1, j: 1, present: false }], edges: [{ i: 0, j: 0, axis: 'z', present: false }] },
  }
  const a = JSON.stringify(solveConnectorsV4(cfg))
  ok(a === JSON.stringify(solveConnectorsV4(cfg)), 'two calls produce identical JSON')
  ok(a === JSON.stringify(solveConnectorsV4(cfg, solveLattice(cfg))),
    'and passing the network in explicitly changes nothing')
  ok(!a.includes('NaN') && !a.includes('null,'), 'no NaN and no holes in the output')
}

console.log(`\ntest-v4-connectors: ${passed} checks passed, ${failed} failed`)
process.exit(failed ? 1 : 0)
