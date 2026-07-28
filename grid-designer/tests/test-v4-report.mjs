/**
 * tests/test-v4-report.mjs — headless checks for core/v4/report.js.
 *
 * Plain node script, no test framework. Exits non-zero on any failure.
 *   node tests/test-v4-report.mjs
 *
 * `envelope.maxAngleDeg` and `envelope.minGapCm` are the numbers this whole
 * pivot exists to produce (V4_SPEC §3), so §1 checks that each is genuinely a
 * BOUNDARY: 0.05 inside it the design is clean and 0.05 outside it is not. A
 * test that only asserts the field is a number would pass against a constant,
 * against a stale cache, and against a bisection that converged on the wrong
 * side — which is to say it would assert nothing at all.
 *
 * §2 also pins the fact that the envelope is a property of (gap, θ) ALONE — the
 * 1 × 2 lattice and the 4 × 4 one report the same boundary, because every joint
 * on a network has the same span and the same |fold|. That is what makes the
 * number usable as a permission before a design exists.
 *
 * §4's real subject is the ADJACENCY FILTER, and it is checked from both sides.
 * The raw OBB pass on a healthy network is NOT empty — the housings converge
 * under every convex fold — and the filter's job is to recognise the JOINED ones
 * as the joint's business. What it must NOT swallow is a pair that shares no
 * joint, and §5's corner ramps are exactly that case, arriving for real rather
 * than through a corrupted chain.
 */

import {
  buildReportV4,
  solveEnvelope,
  isClean,
  solveFrontBarLimit,
  obbClearance,
  solveCornerClearance,
  ENVELOPE_HARD_FLAGS,
  FRONT_BAR_CODE,
} from '../src/core/v4/report.js'
import { solveLattice } from '../src/core/v4/lattice.js'
import { solveConnectorsV4, FLIP_MISMATCH_CODE } from '../src/core/v4/connectors.js'
import { findCollisions } from '../src/core/v3/collide.js'
import { ANGLE_MAX, GAP_MAX, GAP_MIN, DEFAULT_CONFIG } from '../src/core/v4/schema.js'
import { PANEL_DIMENSIONS, PANEL_PROFILE } from '../src/config.js'

let passed = 0
let failed = 0
const ok = (c, m) => { if (c) passed++; else { failed++; console.log(`  FAIL: ${m}`) } }
const near = (a, b, tol, m) => ok(Math.abs(a - b) <= tol, `${m} (got ${a}, want ${b} ±${tol})`)

const L = PANEL_DIMENSIONS['2x2'].height
const W = PANEL_DIMENSIONS['2x2'].width
const T = PANEL_PROFILE.overallThickness
const RAD = Math.PI / 180
const clone = (o) => JSON.parse(JSON.stringify(o))
/** A 1 × 2 lattice: one ramp, two joints — the cheapest thing with a fold. */
const TINY = { lattice: { cols: 1, rows: 2, panelType: '2x2' } }

console.log('=== test-v4-report ===')

// -----------------------------------------------------------------------------
// 1. THE ENVELOPE — is the boundary really the boundary?
// -----------------------------------------------------------------------------
console.log('1. the envelope is a real boundary')
{
  // A gap where the limit is INTERIOR to the angle range. At 2cm the fastener,
  // pinched at the depth its insert sits at, runs out around 33.6° — well inside
  // the 0..75 band, so there is a boundary to find.
  for (const gap of [1.5, 2, 2.5, 3]) {
    const e = solveEnvelope({ ...TINY, gap, angleDeg: 0 })
    ok(e.maxAngleDeg !== null, `gap ${gap}: an angle limit exists`)
    ok(!e.angleAtRangeLimit, `gap ${gap}: and it is a real limit, not the top of the range`)
    ok(isClean({ ...TINY, gap, angleDeg: e.maxAngleDeg - 0.05 }),
      `gap ${gap}: at maxAngleDeg − 0.05° (${(e.maxAngleDeg - 0.05).toFixed(4)}°) the design is CLEAN`)
    ok(!isClean({ ...TINY, gap, angleDeg: e.maxAngleDeg + 0.05 }),
      `gap ${gap}: at maxAngleDeg + 0.05° (${(e.maxAngleDeg + 0.05).toFixed(4)}°) it is NOT`)
    // And the boundary itself is on the clean side — the number is a permission.
    ok(isClean({ ...TINY, gap, angleDeg: e.maxAngleDeg }), `gap ${gap}: maxAngleDeg itself is clean`)
  }

  for (const angleDeg of [15, 30, 45, 60]) {
    const e = solveEnvelope({ ...TINY, angleDeg, gap: GAP_MAX })
    ok(e.minGapCm !== null, `${angleDeg}°: a gap limit exists`)
    ok(!e.gapAtRangeLimit, `${angleDeg}°: and it is a real limit, not the bottom of the range`)
    ok(isClean({ ...TINY, angleDeg, gap: e.minGapCm + 0.05 }),
      `${angleDeg}°: at minGapCm + 0.05cm (${(e.minGapCm + 0.05).toFixed(4)}) the design is CLEAN`)
    ok(!isClean({ ...TINY, angleDeg, gap: e.minGapCm - 0.05 }),
      `${angleDeg}°: at minGapCm − 0.05cm (${(e.minGapCm - 0.05).toFixed(4)}) it is NOT`)
    ok(isClean({ ...TINY, angleDeg, gap: e.minGapCm }), `${angleDeg}°: minGapCm itself is clean`)
  }

  // The two boundaries are the SAME boundary seen from two directions: the
  // smallest gap that admits θ must admit exactly θ as its largest angle.
  const e = solveEnvelope({ ...TINY, angleDeg: 45, gap: GAP_MAX })
  near(solveEnvelope({ ...TINY, gap: e.minGapCm, angleDeg: 0 }).maxAngleDeg, 45, 0.01,
    'maxAngleDeg(minGapCm(45°)) comes back to 45° — one curve, two readings')
}

// -----------------------------------------------------------------------------
// 2. The envelope's shape, and what it says about a design.
// -----------------------------------------------------------------------------
console.log('2. headroom, monotonicity, and the flags that bind')
{
  const e = buildReportV4({}).envelope
  ok(e.clean, 'the default design (3 × 5, 30°, 2cm) is inside the connector envelope')
  near(e.angleDeg, 30, 0, 'it reports where the design is')
  near(e.gapCm, 2, 0, '...on both axes')
  // The ribbon's numbers, unchanged by the generalisation — HANDOFF §8.3 records
  // 33.60° and 1.90cm, and a 2-D network has to agree because every joint on it
  // is the same joint.
  near(e.maxAngleDeg, 33.6, 0.005, `maxAngleDeg is still 33.60° at a 2cm gap (${e.maxAngleDeg})`)
  near(e.minGapCm, 1.9, 0.005, `and minGapCm still 1.90cm at 30° (${e.minGapCm})`)
  near(e.angleHeadroomDeg, e.maxAngleDeg - 30, 1e-9, 'angle headroom is maxAngle − θ')
  near(e.gapHeadroomCm, 2 - e.minGapCm, 1e-9, 'gap headroom is gap − minGap')
  ok(e.limitedBy.length > 0 && e.limitedBy.every((f) => ENVELOPE_HARD_FLAGS.includes(f)),
    `limitedBy names hard flags (got ${e.limitedBy})`)
  ok(e.limitedBy.includes('W_FASTENER_PINCHED'),
    'and at a 2cm gap it is the FASTENER that binds, not the panels')
  ok(e.bisectionSteps === 60, 'a fixed 60 steps, never "until converged"')

  // THE ENVELOPE IS A PROPERTY OF (gap, θ) AND NOTHING ELSE. Every joint on a
  // network has the same span and the same |fold|, so the size of the lattice
  // cannot move it — which is what lets it be quoted as a permission before a
  // design exists.
  for (const lattice of [{ cols: 1, rows: 2 }, { cols: 1, rows: 5 }, { cols: 4, rows: 4 }]) {
    const f = solveEnvelope({ lattice, gap: 2, angleDeg: 30 })
    near(f.maxAngleDeg, e.maxAngleDeg, 1e-9,
      `a ${lattice.cols} × ${lattice.rows} lattice reports the same maxAngleDeg`)
    near(f.minGapCm, e.minGapCm, 1e-9, '...and the same minGapCm')
  }

  // A bigger gap buys more angle, and more angle costs gap. Monotone both ways.
  const maxAngles = [1.5, 2, 2.5, 3, 4].map((gap) => solveEnvelope({ ...TINY, gap, angleDeg: 0 }).maxAngleDeg)
  let mono = true
  for (let k = 1; k < maxAngles.length; k++) if (!(maxAngles[k] > maxAngles[k - 1])) mono = false
  ok(mono, `maxAngleDeg rises with gap (${maxAngles.map((a) => a.toFixed(2))})`)

  const minGaps = [10, 30, 50, 70].map((angleDeg) => solveEnvelope({ ...TINY, angleDeg, gap: GAP_MAX }).minGapCm)
  mono = true
  for (let k = 1; k < minGaps.length; k++) if (!(minGaps[k] > minGaps[k - 1])) mono = false
  ok(mono, `minGapCm rises with θ (${minGaps.map((g) => g.toFixed(3))})`)

  // A design OUTSIDE the envelope: the headroom goes negative and says so.
  const over = buildReportV4({ ...TINY, gap: 2, angleDeg: 45 })
  ok(!over.envelope.clean, '45° at a 2cm gap is outside the envelope')
  ok(over.envelope.angleHeadroomDeg < 0, `and the headroom is negative (${over.envelope.angleHeadroomDeg}°)`)
  ok(over.envelope.maxAngleDeg < 45, 'the largest clean angle is below where the design sits')
  ok(over.warnings.some((w) => w.code === 'W_OUTSIDE_ENVELOPE'), 'the report says so')
  ok(over.envelope.limitedBy.includes('W_FASTENER_PINCHED'), 'and names what is wrong now')

  // Infeasible at EVERY angle: a 0.4cm gap cannot take a fastener flat, so no
  // angle is the answer and reporting one would suggest otherwise.
  const tiny = solveEnvelope({ ...TINY, gap: GAP_MIN, angleDeg: 0 })
  ok(tiny.maxAngleDeg === null && tiny.angleHeadroomDeg === null,
    'at the smallest gap in the band there is no clean angle at all, and the report says null')
  ok(!tiny.clean && tiny.limitedBy.includes('W_FASTENER_PINCHED'),
    'and names the reason rather than a number')

  // A design with nothing to fold is vacuously clean at any angle.
  const lone = solveEnvelope({ lattice: { cols: 1, rows: 1, panelType: '2x2' } })
  ok(lone.clean && lone.maxAngleDeg === ANGLE_MAX && lone.angleAtRangeLimit,
    'a one-cell lattice has no joints, so no angle is forbidden — the range limit is reported as such')

  // The two deliberate exclusions from "clean" (report.js's header).
  ok(!ENVELOPE_HARD_FLAGS.includes('W_BEARS_ON_POWER_SUPPLY'),
    'W_BEARS_ON_POWER_SUPPLY is excluded — it does not move with θ or gap')
  ok(!ENVELOPE_HARD_FLAGS.includes(FLIP_MISMATCH_CODE),
    'and so is the flip mismatch — a flip is not an angle')
  const flipped = buildReportV4({ overrides: { cells: [{ i: 1, j: 2, flipped: true }], edges: [] } })
  ok(flipped.envelope.clean,
    'so a design with mismatched joints still reports its angle envelope honestly')
  ok(flipped.warnings.filter((w) => w.code === FLIP_MISMATCH_CODE).length === 4,
    'while the mismatch is reported against the four joints it belongs to')
}

// -----------------------------------------------------------------------------
// 3. Joints.
// -----------------------------------------------------------------------------
console.log('3. the joint table')
{
  const R = buildReportV4({})
  ok(R.joints.length === 44, `44 joints on the default 3 × 5 (got ${R.joints.length})`)
  ok(R.joints.every((j) => j.spanCm === 2), 'every span is the nominal gap, exactly')
  ok(R.joints.every((j) => j.twistDeg === 0), 'nothing twists')
  ok(R.joints.every((j) => j.dihedralDeg === Math.abs(j.foldDeg)), 'dihedral is |fold|')
  // A ramp meets the floor in a valley and the high level on a ridge, so exactly
  // half of EVERY network's joints are concave, at any θ.
  ok(R.joints.filter((j) => j.sense === 'concave').length === 22 &&
     R.joints.filter((j) => j.sense === 'convex').length === 22,
    'exactly half are valleys and half ridges')
  ok(R.joints.every((j) => j.sense === (j.end === 'lo' ? 'concave' : 'convex')),
    'and the sense follows the END, never the ramp\'s direction')
  ok(R.joints.every((j) => j.stationCount === 2), 'each carries two parts')
  ok(R.joints.every((j) => j.flags.includes('W_BEARS_ON_POWER_SUPPLY')),
    'and each is flagged for what its lips bear on')
  ok(R.joints.every((j) => !j.flags.some((f) => ENVELOPE_HARD_FLAGS.includes(f))),
    'with no hard flag anywhere — which is what "inside the envelope" means')
  ok(R.joints.every((j) => !j.flipMismatch), 'no flip mismatch on the default design')
  // Each joint names its lattice edge and which end of the ramp it is.
  ok(R.joints.every((j) => j.edge && (j.edge.axis === 'x' || j.edge.axis === 'z')),
    'every joint names the lattice edge its ramp lies on')
  ok(new Set(R.joints.map((j) => j.axis)).size === 2, 'and joints run on both world axes')

  // A flat design has no fold at all.
  const flat = buildReportV4({ angleDeg: 0 })
  ok(flat.joints.every((j) => j.sense === 'flat' && j.foldDeg === 0), 'a flat design folds by nothing')

  // Spans hold at every gap.
  for (const gap of [0.4, 1, 3.75, 8]) {
    const g = buildReportV4({ ...TINY, gap, angleDeg: 20 })
    ok(g.joints.every((j) => j.spanCm === gap), `gap ${gap}: every joint spans exactly it`)
  }

  // A mismatched joint carries no part, and says which one it is.
  const mism = buildReportV4({ overrides: { cells: [{ i: 0, j: 0, flipped: true }], edges: [] } })
  const bad = mism.joints.filter((j) => j.flipMismatch)
  ok(bad.length === 2 && bad.every((j) => j.stationCount === 0),
    'the corner cell\'s two joints carry no parts')
  // ...and therefore no STATION flag. They do still carry the front bar's, and
  // that is right: a joint's valley is too deep for a flat bar whether or not a
  // part is currently placed on it, and hiding the finding behind the flip would
  // make it reappear the moment the flip was undone.
  ok(bad.every((j) => j.flags.every((f) => f === FRONT_BAR_CODE)),
    'and no station flag — only the front bar\'s, which is a fact about the fold')
  ok(mism.joints.filter((j) => !j.flipMismatch).every((j) => j.stationCount === 2),
    'and every other joint is untouched')
}

// -----------------------------------------------------------------------------
// 4. Collisions — the ADJACENCY FILTER is the thing under test.
// -----------------------------------------------------------------------------
console.log('4. collisions')
{
  // A 1-column lattice is the ribbon, and it still collides with nothing.
  ok(buildReportV4({ lattice: { cols: 1, rows: 5, panelType: '2x2' } }).collisions.length === 0,
    'the cols:1 network (the ribbon) collides with nothing')
  for (const angleDeg of [0, 30, 60, 75]) {
    ok(buildReportV4({ lattice: { cols: 1, rows: 5, panelType: '2x2' }, angleDeg, gap: 1 })
      .collisions.length === 0,
      `${angleDeg}° at a 1cm gap: still nothing along a single column`)
  }

  // ...but that verdict is EARNED, not vacuous. The raw OBB pass DOES find
  // overlaps on a folded network: the housings converge under every CONVEX fold,
  // so the two boxes meeting at a ridge interpenetrate. Every one of them is a
  // JOINED pair, which is precisely why joined pairs are excluded and why
  // `sectionFouling` (which sees the real tapered section rather than a box) is
  // what judges a joint.
  for (const angleDeg of [30, 60, 75]) {
    const C0 = solveLattice({ lattice: { cols: 1, rows: 5, panelType: '2x2' }, angleDeg })
    const p0 = C0.panels.filter((p) => p.present)
    const raw = findCollisions(p0.map((p) => p.obb), { minDepthCm: 0.05 })
    const joined = new Set(C0.joints.map((j) => (j.a < j.b ? `${j.a}|${j.b}` : `${j.b}|${j.a}`)))
    ok(raw.length > 0, `${angleDeg}°: the unfiltered OBB pass DOES find overlaps (${raw.length})`)
    ok(raw.every((h) => {
      const a = p0[h.i].id
      const b = p0[h.j].id
      return joined.has(a < b ? `${a}|${b}` : `${b}|${a}`)
    }), `${angleDeg}°: and every one of them is a JOINED pair`)
  }

  // Panels placed on top of each other with no joint between them MUST be
  // reported...
  const cfg = { lattice: { cols: 1, rows: 5, panelType: '2x2' } }
  const C = solveLattice(cfg)
  const K = solveConnectorsV4(cfg, C)
  const idx = (id) => C.panels.findIndex((p) => p.id === id)
  const overlap = clone(C)
  overlap.panels[idx('Ci0j2')].obb = clone(overlap.panels[idx('Ci0j0')].obb)
  const hit = buildReportV4(cfg, overlap, K)
  ok(hit.collisions.length === 1, 'two unjoined panels in the same place are a collision')
  ok(hit.collisions[0].a === 'Ci0j0' && hit.collisions[0].b === 'Ci0j2', 'named by id')
  ok(hit.collisions[0].depthCm > 0, 'with a positive penetration depth')

  // ...and JOINED ones must not: their overlap is the joint, which
  // sectionFouling judges exactly and an OBB pair cannot.
  const adjacent = clone(C)
  adjacent.panels[idx('Ei0j0z')].obb = clone(adjacent.panels[idx('Ci0j0')].obb)
  ok(buildReportV4(cfg, adjacent, K).collisions.length === 0,
    'a ramp on top of the cell it is JOINED to is the joint\'s business, not a collision')

  // Two panels either side of a REMOVED one are not joined — nothing connects
  // them, so they are checked against each other.
  const cutCfg = { lattice: { cols: 1, rows: 5, panelType: '2x2' },
    overrides: { cells: [], edges: [{ i: 0, j: 1, axis: 'z', present: false }] } }
  const cutC = solveLattice(cutCfg)
  const cutK = solveConnectorsV4(cutCfg, cutC)
  const ci = (id) => cutC.panels.findIndex((p) => p.id === id)
  const bridged = clone(cutC)
  bridged.panels[ci('Ci0j2')].obb = clone(bridged.panels[ci('Ci0j1')].obb)
  ok(buildReportV4(cutCfg, bridged, cutK).collisions.some((c) => c.a === 'Ci0j1' && c.b === 'Ci0j2'),
    'cells either side of a switched-off edge are checked — nothing connects them')

  // An ABSENT panel is not in the world and cannot collide with anything.
  const ghost = clone(cutC)
  ghost.panels[ci('Ei0j1z')].obb = clone(ghost.panels[ci('Ci0j0')].obb)
  ok(buildReportV4(cutCfg, ghost, cutK).collisions.length === 0, 'an absent panel collides with nothing')

  // W_BELOW_FLOOR is reachable by turning grounding off.
  const sunk = buildReportV4({ placement: { groundToFloor: false } })
  const below = sunk.warnings.filter((w) => w.code === 'W_BELOW_FLOOR')
  ok(below.length > 0, 'ungrounded, the housings dip below the floor and are flagged')
  near(Math.min(...below.map((w) => w.yMinCm)), -T, 1e-9, 'by exactly one housing thickness')
  ok(buildReportV4({}).warnings.every((w) => w.code !== 'W_BELOW_FLOOR'),
    'and grounded, nothing is below the floor')

  // W_THROUGH_WALL is NOT reachable from a valid config — §9.5 translates the
  // network so its nearest material sits AT wallOffsetCm, which is non-negative,
  // and that holds braced as well as free. So it is checked against a
  // deliberately corrupted network, the same way v3 tests W_CONNECTOR_INFEASIBLE.
  for (const wallAnchor of ['free', 'braced']) {
    ok(buildReportV4({ placement: { wallOffsetCm: 0, windowOffsetCm: 0, groundToFloor: true, wallAnchor } })
      .warnings.every((w) => w.code !== 'W_THROUGH_WALL'),
      `${wallAnchor}: a network flush against the wall does not report going through it`)
  }
  const through = clone(C)
  through.panels[idx('Ci0j0')].corners = through.panels[idx('Ci0j0')].corners
    .map((c) => [c[0] - 5, c[1], c[2]])
  const wall = buildReportV4(cfg, through, K).warnings.filter((w) => w.code === 'W_THROUGH_WALL')
  ok(wall.length === 1 && wall[0].panel === 'Ci0j0' && wall[0].xMinCm === -5,
    'a panel reaching past x = 0 is reported, with how far')
}

// -----------------------------------------------------------------------------
// 5. THE CORNER CLEARANCE — the number with no ribbon analogue.
// -----------------------------------------------------------------------------
console.log('5. the corner clearance')
{
  // A one-column network's pairs are COLLINEAR — the two ramps a mid-column cell
  // carries, one to each side — and they sit a whole pitch apart. That is the
  // non-vacuous control: the quantity exists on the ribbon and says "miles of
  // room", and only a second axis brings it anywhere near zero.
  const ribbon = buildReportV4({ lattice: { cols: 1, rows: 5, panelType: '2x2' } })
  ok(ribbon.metrics.cornerClearance.pairCount === 3,
    `the ribbon's three mid-column cells each carry two collinear ramps ` +
    `(got ${ribbon.metrics.cornerClearance.pairCount})`)
  ok(ribbon.metrics.cornerClearance.worstCm > 50,
    `and they are ${ribbon.metrics.cornerClearance.worstCm}cm apart — nowhere near touching`)
  ok(ribbon.metrics.cornerClearance.touchingPairs === 0, 'so nothing is flagged')

  const R = buildReportV4({})
  const cc = R.metrics.cornerClearance
  ok(cc.pairCount > 0, `the default 3 × 5 has ${cc.pairCount} corner pairs`)
  ok(cc.worst && cc.worst.cell && cc.worst.a && cc.worst.b, 'and names the tightest one')
  ok(cc.pairs.every((p) => p.a < p.b), 'each pair is reported once, in id order')
  ok(cc.pairs[0].clearanceCm === cc.worstCm, 'sorted tightest first')

  // AT θ = 0 the ramps are coplanar arms of a cross and the clearance IS the
  // gap. A closed form to anchor the curve at one end.
  for (const gap of [1, 2, 4]) {
    near(buildReportV4({ gap, angleDeg: 0 }).metrics.cornerClearance.worstCm, gap, 1e-9,
      `flat, two ramps off a cell clear each other by exactly the gap (${gap}cm)`)
  }

  // It falls MONOTONICALLY with θ and crosses zero — that crossing is the 2-D
  // limit, and at a 2cm gap it is TIGHTER than the connector envelope's 33.6°.
  const curve = [0, 10, 20, 25, 30, 40, 50].map((a) => buildReportV4({ angleDeg: a }).metrics.cornerClearance.worstCm)
  let mono = true
  for (let k = 1; k < curve.length; k++) if (!(curve[k] < curve[k - 1])) mono = false
  ok(mono, `the clearance falls monotonically with θ (${curve.map((c) => c.toFixed(3))})`)
  ok(buildReportV4({ angleDeg: 25 }).metrics.cornerClearance.worstCm > 0, 'positive at 25°')
  ok(buildReportV4({ angleDeg: 30 }).metrics.cornerClearance.worstCm < 0, 'and negative at 30°')
  ok(buildReportV4({ angleDeg: 30 }).envelope.maxAngleDeg > 30,
    'while the CONNECTOR envelope still allows 30° — the 2-D limit binds first, and it is new')

  // UNLIKE the front bar, the gap buys this back: a wider joint moves the corner
  // limit directly. That difference is why they are reported separately.
  // Bisected on `solveCornerClearance` directly rather than on a whole report —
  // the envelope alone is 130 solves, and this asks the question 24 times.
  const crossing = (gap) => {
    let lo = 0
    let hi = ANGLE_MAX
    for (let k = 0; k < 24; k++) {
      const m = (lo + hi) / 2
      const cc = solveCornerClearance(solveLattice({ gap, angleDeg: m }))
      if (cc.worstCm > 0) lo = m
      else hi = m
    }
    return lo
  }
  const c1 = crossing(1)
  const c2 = crossing(2)
  ok(c2 > c1 + 5, `widening the gap 1cm → 2cm moves the corner limit ${c1.toFixed(1)}° → ${c2.toFixed(1)}°`)
  near(solveFrontBarLimit(1), solveFrontBarLimit(2), 1e-6,
    'where the front bar\'s limit does not move with the gap at all — the two are different animals')

  // The warning fires exactly when the boxes meet, and not before.
  const clear = buildReportV4({ angleDeg: 20 })
  ok(clear.metrics.cornerClearance.touchingPairs === 0, 'at 20° no corner pair is touching')
  ok(clear.warnings.every((w) => w.code !== 'W_CORNER_RAMPS_MEET'), 'and nothing is warned about')
  ok(R.warnings.filter((w) => w.code === 'W_CORNER_RAMPS_MEET').length === 1,
    'at 30° exactly one summary warning is raised')
  // Corner pairs are reported, but NOT as collisions. The OBB is 4.1cm deep
  // where the real panel at its corner is 1.2cm of outer wall, so an overlap of
  // box corners is not a claim this primitive can support — see report.js's note
  // by `cornerContacts`. The test pins both halves of that decision, because
  // either one alone would be a bug: silence, or 16 false alarms.
  ok(R.metrics.cornerClearance.touchingPairs === R.cornerContacts.length,
    'every touching corner pair is reported in cornerContacts')
  ok(R.collisions.length === 0,
    'and NONE of them is counted as a collision — a 3cm-thick box corner is not a panel')
  const cornerIds = new Set(R.cornerContacts.map((c) => `${c.a}|${c.b}`))
  ok(R.collisions.every((c) => !cornerIds.has(`${c.a}|${c.b}`)), 'the two lists are disjoint')
  // Non-vacuous, and the check that matters: the split must be a PARTITION of
  // the unjoined overlaps, not a filter that quietly drops some. Recompute the
  // raw pass here and assert the two lists account for every one of them.
  //
  // Worth recording why the plain `collisions` list comes back empty on every
  // lattice tried: on a checkerboard the only panels that can reach each other
  // are the ones sharing a joint (excluded — that overlap IS the joint) and the
  // two ramps off a shared cell (the corner pairs). Anything else is a full
  // pitch away. So an empty collision list is the expected result, which is
  // exactly why it cannot be left to stand as the only evidence.
  for (const cfg of [{ angleDeg: 30 }, { angleDeg: 75, gap: 1 }, { lattice: { cols: 3, rows: 3 }, angleDeg: 60, gap: 1 }]) {
    const rep = buildReportV4(cfg)
    const C = solveLattice(cfg)
    const live = C.panels.filter((p) => p.present)
    const joined = new Set(C.joints.map((j) => (j.a < j.b ? `${j.a}|${j.b}` : `${j.b}|${j.a}`)))
    const raw = findCollisions(live.map((p) => p.obb), { minDepthCm: 0.05 })
      .filter(({ i, j }) => {
        const [A, B] = [live[i].id, live[j].id]
        return !joined.has(A < B ? `${A}|${B}` : `${B}|${A}`)
      })
    ok(raw.length === rep.collisions.length + rep.cornerContacts.length,
      `${JSON.stringify(cfg)}: the two lists partition the ${raw.length} unjoined overlaps, none dropped`)
  }

  // IT IS A BOUNDING-BOX VERDICT, NOT A PANEL ONE. The boxes meet at the panels'
  // CORNERS, where the real section is only the outer wall; the back plates are
  // still metres apart. Measured here so the caveat in report.js's header is a
  // number rather than an assurance.
  const C = solveLattice({ lattice: { cols: 3, rows: 3 }, angleDeg: 30 })
  const M = new Map(C.panels.map((p) => [p.id, p]))
  const inset = PANEL_PROFILE.flangeWidth + PANEL_PROFILE.taperWidth
  const d0 = PANEL_PROFILE.outerWallDepth + PANEL_PROFILE.flangeDrop
  const backPlate = (p) => ({
    center: [0, 1, 2].map((k) => p.position[k] - p.normal[k] * ((d0 + T) / 2)),
    halfExtents: [(W - 2 * inset) / 2, (T - d0) / 2, (L - 2 * inset) / 2],
    quaternion: p.quaternion,
  })
  const A = M.get('Ei0j0x')
  const B = M.get('Ei1j0z')
  ok(obbClearance(A.obb, B.obb) < 0, 'at 30° the two full boxes overlap')
  ok(obbClearance(backPlate(A), backPlate(B)) > 5,
    `while their BACK PLATES are still ${obbClearance(backPlate(A), backPlate(B)).toFixed(2)}cm apart`)

  // obbClearance itself, against a case with a known answer: two axis-aligned
  // boxes a stated distance apart.
  const box = (x) => ({ center: [x, 0, 0], halfExtents: [1, 1, 1], quaternion: [0, 0, 0, 1] })
  near(obbClearance(box(0), box(5)), 3, 1e-12, 'obbClearance on two unit boxes 5 apart reads 3')
  near(obbClearance(box(0), box(2)), 0, 1e-12, 'exactly touching reads 0')
  near(obbClearance(box(0), box(1.5)), -0.5, 1e-12, 'and overlapping reads negative')
}

// -----------------------------------------------------------------------------
// 6. Metrics.
// -----------------------------------------------------------------------------
console.log('6. metrics')
{
  const R = buildReportV4({})
  const m = R.metrics
  const P = m.lattice.pitchCm

  near(P, 115.825227532, 1e-9, 'the pitch is reported')
  near(m.lattice.riseCm, 31.03527618, 1e-9, 'and the rise')
  ok(m.lattice.cols === 3 && m.lattice.rows === 5, 'and the lattice it came from')

  // Overall: the measuring box. A 3-wide network spans 2 pitches plus a panel.
  // Against the CLOSED FORM pitch, not the reported one: multiplying a 1e-9
  // record by four multiplies its rounding by four too.
  const Pexact = L + 2 * 2 * Math.cos(15 * RAD) + L * Math.cos(30 * RAD)
  near(m.overall.size[0], 2 * Pexact + W, 1e-9, 'three columns span 2·P + 60 in x')
  near(m.overall.size[2], 4 * Pexact + L, 1e-9, 'and five rows 4·P + 60 in z')
  near(m.overall.size[1], R.metrics.lattice.riseCm + T, 1e-9,
    'the height is the level rise plus one housing thickness')
  ok(m.overall.min[1] === 15,
    `and it sits at the 15cm ground clearance (got ${m.overall.min[1]})`)

  // Counts by role — V4_SPEC §9.6.
  ok(m.counts.cells === 15, '15 cells')
  ok(m.counts.groundCells === 8 && m.counts.highCells === 7,
    'of which 8 ground and 7 high — a checkerboard of odd size is never even')
  ok(m.counts.ramps === 2 * 5 + 3 * 4, '22 ramps: 10 x-edges and 12 z-edges')
  ok(m.counts.rampsRising === 11 && m.counts.rampsFalling === 11, 'half rising, half falling')
  ok(m.counts.anchorRamps === 0, 'no anchors when free')
  ok(m.counts.panels === 37, '37 panels in all')
  ok(m.counts.joints === 44, 'and 44 joints — two per ramp')
  ok(m.counts.absent === 0, 'nothing absent on a default design')

  // Braced adds one panel and one joint per high cell at the wall.
  const braced = buildReportV4({ placement: { wallAnchor: 'braced' } }).metrics.counts
  ok(braced.anchorRamps === 2 && braced.panels === 39 && braced.joints === 46,
    'braced adds two anchors, two panels and two joints')

  // Per column: each is one panel wide and runs the full depth.
  ok(m.columns.length === 3, 'one row per column')
  for (const c of m.columns) {
    near(c.widthCm, W, 1e-9, `column ${c.i} is one panel wide — its x-ramps belong to no column`)
    near(c.xMinCm, c.i * P, 1e-9, `column ${c.i} starts at i·P`)
    near(c.planRunCm, m.overall.size[2], 1e-9, `column ${c.i} runs the full depth`)
    ok(c.presentCells === 5 && c.cellSlots === 5, `column ${c.i} has all five cells`)
    ok(c.panelCount === 5 + 4, `and nine panels — five cells and four z-ramps`)
  }

  // Per row: each spans the full width.
  ok(m.rows.length === 5, 'one row per lattice row')
  for (const c of m.rows) {
    near(c.widthCm, m.overall.size[0], 1e-9, `row ${c.j} spans the full width`)
    near(c.planRunCm, L, 1e-9, `row ${c.j} is one panel deep — its z-ramps belong to no row`)
    near(c.zMinCm, c.j * P, 1e-9, `row ${c.j} starts at j·P`)
    ok(c.presentCells === 3 && c.panelCount === 3 + 2, `row ${c.j} is three cells and two x-ramps`)
  }

  // Material vs plan — the 2-D analogue of the ribbon's compression ratio.
  near(m.material.materialAreaCm2, 37 * L * W, 1e-9, '37 panels of sheet')
  near(m.material.planAreaCm2, m.overall.size[0] * m.overall.size[2], 1e-9, 'against the plan box')
  ok(m.material.coverageRatio > 1, 'and the network covers more plan than it is made of — the holes')

  // Per panel.
  ok(m.panels.length === 37, 'one row per panel')
  ok(m.panels.filter((p) => p.kind === 'cell').every((p) => p.riseCm === 0), 'the flats rise by nothing')
  let worst = 0
  for (const p of m.panels.filter((x) => x.kind === 'ramp')) {
    worst = Math.max(worst, Math.abs(p.planRunCm - L * Math.cos(Math.abs(p.tiltDeg) * RAD)))
    worst = Math.max(worst, Math.abs(Math.abs(p.riseCm) - L * Math.sin(Math.abs(p.tiltDeg) * RAD)))
  }
  ok(worst < 1e-8, `every ramp's plan run is L·cos θ and its rise L·sin θ (worst ${worst})`)
  ok(m.panels.filter((p) => p.level === 0).every((p) => p.role === 'ground'), 'level 0 reads as ground')
  ok(m.panels.filter((p) => p.level === 1).every((p) => p.role === 'high'), 'and level 1 as high')

  // A ragged edge: the counts and the columns follow it.
  const ragged = buildReportV4({
    overrides: { cells: [{ i: 2, j: 4, present: false }, { i: 2, j: 3, present: false }], edges: [] },
  }).metrics
  ok(ragged.counts.cells === 13, 'two cells removed')
  ok(ragged.columns[2].presentCells === 3, 'the last column is down to three')
  ok(ragged.columns[0].presentCells === 5 && ragged.columns[1].presentCells === 5,
    'and the others are untouched')
  // Only the CELLS go. Their ramps hang on wherever a cell survives at the far
  // end (§9.3) — the one exception being the z-edge (2,3), which joined the two
  // removed cells to each other and so has nothing left to hang from.
  ok(ragged.counts.absent === 2 + 1,
    'three panels absent — the two cells, plus the one ramp that joined only them')
  ok(ragged.counts.slots === 37, 'while the table keeps all 37 slots')

  // An empty network does not divide by zero or report an infinite box.
  const empty = buildReportV4({
    lattice: { cols: 2, rows: 2, panelType: '2x2' },
    overrides: { cells: [0, 1].flatMap((i) => [0, 1].map((j) => ({ i, j, present: false }))), edges: [] },
  })
  ok(empty.metrics.counts.panels === 0, 'nothing present')
  ok(empty.metrics.overall.size.every((s) => s === 0), 'and a zero box rather than an infinite one')
  ok(empty.metrics.material.coverageRatio === null, 'and no ratio rather than a NaN')
  ok(!JSON.stringify(empty).includes('NaN'), 'no NaN anywhere in the report')
}

// -----------------------------------------------------------------------------
// 7. Warnings, shape, and determinism.
// -----------------------------------------------------------------------------
console.log('7. warnings and determinism')
{
  const cfgs = [
    {},
    { angleDeg: 45 },
    { gap: 0.4 },
    { placement: { groundToFloor: false } },
    { placement: { wallAnchor: 'braced', wallOffsetCm: 10 } },
    { overrides: { cells: [{ i: 1, j: 2, flipped: true }], edges: [] } },
    { overrides: { cells: [], edges: [{ i: 0, j: 0, axis: 'x', present: false }] } },
    { connectors: { ...DEFAULT_CONFIG.connectors, supplyMode: 'block' } },
    { lattice: { cols: 1, rows: 1, panelType: '2x2' } },
    { lattice: { cols: 4, rows: 4, panelType: '2x2' }, angleDeg: 20, gap: 3 },
  ]
  for (const cfg of cfgs) {
    const R = buildReportV4(cfg)
    const label = JSON.stringify(cfg).slice(0, 45)
    ok(R.warnings.every((w) => typeof w.code === 'string' && w.code.length > 0),
      `${label}: every warning carries a code`)
    ok(['joints', 'envelope', 'collisions', 'metrics', 'warnings'].every((k) => k in R),
      `${label}: the report has all five sections`)
    const a = JSON.stringify(R)
    ok(a === JSON.stringify(buildReportV4(cfg)), `${label}: two calls produce identical JSON`)
    ok(!a.includes('NaN'), `${label}: no NaN anywhere in the report`)
  }

  // Passing the network and connectors in explicitly must change nothing.
  const cfg = { lattice: { cols: 2, rows: 3, panelType: '2x2' }, angleDeg: 22.5, gap: 1.9 }
  const C = solveLattice(cfg)
  const K = solveConnectorsV4(cfg, C)
  ok(JSON.stringify(buildReportV4(cfg, C, K)) === JSON.stringify(buildReportV4(cfg)),
    'supplying the network and stations yourself gives the same report')
}

// -----------------------------------------------------------------------------
// 8. THE FRONT BAR AND THE VALLEYS
//
// A separate limit from the envelope, deliberately — see report.js's header.
// The claims worth testing are that it is a real boundary, that it does NOT
// move with the gap (which is what says it belongs to the bar rather than to
// the joint), and that it is kept OUT of the envelope.
// -----------------------------------------------------------------------------
console.log('8. the front bar bites the bezels in a deep enough valley')
{
  const limit = solveFrontBarLimit(2)
  ok(limit > 0 && limit < ANGLE_MAX, `there is an interior limit (${limit.toFixed(3)}°)`)

  // Constant across the gap band. If this ever fails, the collision has started
  // depending on the gap and the header's reasoning no longer holds.
  for (const gap of [1, 1.5, 2, 3, 4, 6]) {
    near(solveFrontBarLimit(gap), limit, 1e-6, `gap ${gap}cm: the bar's limit is the same`)
  }

  // A real boundary, probed through the report rather than the solver, so the
  // wiring is covered too. EVERY ramp meets its ground cell in a valley, so a
  // network has exactly one valley per ramp — 22 of them on the default 3 × 5.
  const barWarnings = (angleDeg) =>
    buildReportV4({ angleDeg, gap: 2 }).warnings.filter((w) => w.code === FRONT_BAR_CODE)
  ok(barWarnings(limit - 0.05).length === 0, 'just inside the limit no joint is flagged')
  ok(barWarnings(limit + 0.05).length === 22, 'just outside it all 22 valleys are')

  const R = buildReportV4({ angleDeg: 30, gap: 2 })
  ok(R.envelope.frontBar.clears === false, 'the default 30° design does not clear the bar')
  near(R.envelope.frontBar.headroomDeg, limit - 30, 1e-6, 'and its headroom is negative by that much')
  ok(R.joints.filter((j) => j.sense === 'concave').every((j) => j.flags.includes(FRONT_BAR_CODE)),
    'every valley carries the flag')
  ok(R.joints.filter((j) => j.sense === 'convex').every((j) => !j.flags.includes(FRONT_BAR_CODE)),
    'and no ridge does — this is a concave-only failure')

  // THE POINT of keeping it separate: it must not move maxAngleDeg.
  ok(!ENVELOPE_HARD_FLAGS.includes(FRONT_BAR_CODE), 'the code is not an envelope flag')
  ok(R.envelope.maxAngleDeg > limit,
    `and maxAngleDeg (${R.envelope.maxAngleDeg.toFixed(2)}°) is unaffected by it`)

  // Non-vacuous negative: a design shallow enough to clear says so.
  const shallow = buildReportV4({ angleDeg: 8, gap: 2 })
  ok(shallow.envelope.frontBar.clears === true, 'an 8° wave clears the bar')
  ok(shallow.warnings.every((w) => w.code !== FRONT_BAR_CODE), 'and raises no bar warning')
}

// -----------------------------------------------------------------------------
// 9. THE SPACERS SECTION
//
// The section exists to CROSS-CHECK grounding against the spacer solver, so what
// is worth asserting is that the two numbers came from different places and
// still agree — and that the report notices when they cannot.
// -----------------------------------------------------------------------------
console.log('9. report.spacers')
{
  const R = buildReportV4({ ...DEFAULT_CONFIG })
  ok(R.spacers.grounded === true, 'the default design is grounded')
  ok(R.spacers.count === 64, `and takes 64 posts (got ${R.spacers.count})`)
  ok(R.spacers.cellCount === 8 && R.spacers.perCell.length === 8, 'under 8 floor-resting cells')
  ok(R.spacers.perCell.every((c) => c.count === 8), 'eight each — four edges, two per edge')
  ok(R.spacers.heightsCm.length === 1, 'EXACTLY ONE distinct height, which is the whole check')
  ok(R.spacers.heightsCm[0] === R.spacers.clearanceCm && R.spacers.clearanceCm === 15,
    'and it is the 15cm clearance the lattice was translated by')
  // The report's own box agrees with the posts under it.
  ok(R.metrics.overall.min[1] === R.spacers.heightsCm[0],
    'the measuring box sits on top of the posts, not somewhere else')

  // The clearance moves both together.
  const at5 = buildReportV4({ ...DEFAULT_CONFIG, placement: { ...DEFAULT_CONFIG.placement, groundClearanceCm: 5 } })
  ok(at5.spacers.heightsCm[0] === 5 && at5.metrics.overall.min[1] === 5,
    'at a 5cm clearance both read 5 — non-vacuous')
  ok(at5.spacers.count === R.spacers.count, 'and the count is unchanged: it is a length, not a number of posts')

  // The spacing rule is the connectors', in the report as in the solver.
  const dense = buildReportV4({
    ...DEFAULT_CONFIG,
    connectors: { ...DEFAULT_CONFIG.connectors, spacingCm: 10 },
  })
  ok(dense.spacers.perEdge === 6 && dense.spacers.count === 8 * 4 * 6,
    `a 10cm spacing puts 6 per edge and ${8 * 4 * 6} in total (got ${dense.spacers.perEdge}, ${dense.spacers.count})`)

  // Grounding off: no posts, and the section says so rather than going missing.
  const free = buildReportV4({
    ...DEFAULT_CONFIG,
    placement: { ...DEFAULT_CONFIG.placement, groundToFloor: false },
  })
  ok(free.spacers.grounded === false && free.spacers.count === 0, 'ungrounded, there are none')
  ok(free.spacers.heightsCm.length === 0, 'and no heights to report')
  ok(free.warnings.every((w) => w.code !== 'W_SPACER_MISMATCH'),
    'which is not a mismatch — there is nothing to mismatch with')
}

console.log(`\ntest-v4-report: ${passed} checks passed, ${failed} failed`)
process.exit(failed ? 1 : 0)
