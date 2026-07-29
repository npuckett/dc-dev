/**
 * tests/test-v4-wave.mjs — headless checks for core/v4/wave.js and the wave path
 * through core/v4/lattice.js.
 *
 * Plain node script, no test framework. Exits non-zero on any failure.
 *   node tests/test-v4-wave.mjs
 *
 * Closed-form expectations, not snapshots — with one deliberate exception, §8,
 * which IS a snapshot and has to be: its subject is that the TRAPEZOID solve did
 * not move, and the only honest way to check "nothing changed" is to compare
 * against what it was. The seven hashes there were taken from the tree at
 * `71f8b15`, the commit before the wave existed.
 *
 * The four claims worth naming, each of which would fail silently:
 *
 *   ALIGNMENT (§2). The x of column `i` is the same number for every `j`, and the
 *        z of row `j` the same for every `i`. This is the user's "nothing gets
 *        out of basic alignment" and it is what forces the separable model in the
 *        first place — if it ever fails, the mode has no reason to exist.
 *   CLOSURE (§3). Every 4-cycle of ramps comes back to the same height. Checked
 *        from the RAMPS' OWN RISES, never from differences of a height field
 *        (which telescope to zero for any field at all and would assert nothing),
 *        and checked against a CHECKERBOARD built from the same varying angles,
 *        which is off by ~9cm. That negative is the proof the test can fail.
 *   THE SPAN (§5). Every joint's rim-to-rim distance is still exactly `gap`, at
 *        every one of the differing angles. v4's whole premise; the wave changes
 *        the per-edge half-angle step, which is precisely the machinery that
 *        keeps this true, so it is the machinery most likely to be got wrong.
 *   THE TRAPEZOID DID NOT MOVE (§8).
 */

import {
  solveWave,
  solveAxis,
  planAdvanceCm,
  riseCm,
  scrunchFactor,
  angleForAdvance,
  worstCycleResidual,
  SCRUNCH_UNREACHABLE_CODE,
} from '../src/core/v4/wave.js'
import { solveLattice, latticeStep } from '../src/core/v4/lattice.js'
import { buildReportV4 } from '../src/core/v4/report.js'
import {
  normalizeConfig,
  validateConfig,
  ANGLE_MAX,
  SCRUNCH_MAX,
  DEFAULT_WAVE,
  PATTERN_KINDS,
} from '../src/core/v4/schema.js'
import { PANEL_DIMENSIONS } from '../src/config.js'

let passed = 0
let failed = 0
const ok = (c, m) => { if (c) passed++; else { failed++; console.log(`  FAIL: ${m}`) } }
const near = (a, b, tol, m) => ok(Math.abs(a - b) <= tol, `${m} (got ${a}, want ${b} ±${tol})`)

const RAD = Math.PI / 180
const L = PANEL_DIMENSIONS['2x2'].height

/** A wave config, spelled out so every test says what it is testing. */
const wave = (over = {}, w = {}) => ({
  lattice: { cols: 4, rows: 5, panelType: '2x2' },
  angleDeg: 25,
  gap: 2,
  placement: { wallOffsetCm: 0, windowOffsetCm: 0, groundToFloor: false, wallAnchor: 'free' },
  ...over,
  pattern: { kind: 'wave', wave: { ...DEFAULT_WAVE, ...w } },
})

const cellOf = (C, i, j) => C.panels.find((p) => p.id === `Ci${i}j${j}`)

console.log('=== test-v4-wave ===')

// -----------------------------------------------------------------------------
// 1. E and R, and their agreement with the checkerboard's own step.
//
// The wave is written in E/R form and `latticeStep` in pitch/rise form. They are
// the same two numbers, and asserting it here is what keeps a deliberate
// restatement from drifting — the same discipline schema.js applies to the gap
// band.
// -----------------------------------------------------------------------------
console.log('1. the plan advance and the rise')
{
  for (const angleDeg of [0, 12.5, 25, 30, 47.5, 75]) {
    for (const gap of [0.4, 2, 5, 8]) {
      const step = latticeStep({ gapCm: gap, angleDeg, lengthCm: L })
      near(L + planAdvanceCm(angleDeg, gap, L), step.pitchCm, 1e-12,
        `${angleDeg}°/${gap}cm: L + E(θ) is the checkerboard's cell pitch`)
      near(riseCm(angleDeg, gap, L), step.riseCm, 1e-12,
        `${angleDeg}°/${gap}cm: R(θ) is its level rise`)
    }
  }
  // Strictly decreasing in θ — the property the bisection rests on.
  let mono = true
  let prev = Infinity
  for (let a = 0; a <= 90; a += 0.25) {
    const e = planAdvanceCm(a, 2, L)
    if (e >= prev) mono = false
    prev = e
  }
  ok(mono, 'E(θ) is strictly decreasing on [0°, 90°], so the bisection is well posed')

  // The brief's own measured numbers, restated as a check on E and R rather than
  // taken on trust — if these drift, every quantity in this file drifts with them.
  near(riseCm(30, 2, L), 31.03527618, 1e-8, 'R(30°, 2cm) is the shipped 31.035276')
  near(riseCm(35, 2, L), 35.617409379, 1e-8, 'R(35°, 2cm)')
  near(riseCm(40, 2, L), 39.935337154, 1e-8, 'R(40°, 2cm)')
}

// -----------------------------------------------------------------------------
// 2. ALIGNMENT — the plan grid stays a product grid.
// -----------------------------------------------------------------------------
console.log('2. nothing gets out of basic alignment')
{
  for (const [sx, sz, ax, az] of [[0, 0, 1, 1], [0.3, 0, 1, 1], [0, 0.4, 1, 1],
    [0.25, 0.4, 1, 0.5], [0.5, 0.5, 0.3, 1], [SCRUNCH_MAX, 0.1, 1, 1]]) {
    const C = solveLattice(wave({ lattice: { cols: 5, rows: 6, panelType: '2x2' } },
      { scrunchX: sx, scrunchZ: sz, attractorX: ax, attractorZ: az }))
    const tag = `x${sx}/z${sz} @ ${ax}/${az}`
    let worstX = 0
    let worstZ = 0
    for (let i = 0; i < 5; i++) {
      const x0 = cellOf(C, i, 0).position[0]
      for (let j = 0; j < 6; j++) worstX = Math.max(worstX, Math.abs(cellOf(C, i, j).position[0] - x0))
    }
    for (let j = 0; j < 6; j++) {
      const z0 = cellOf(C, 0, j).position[2]
      for (let i = 0; i < 5; i++) worstZ = Math.max(worstZ, Math.abs(cellOf(C, i, j).position[2] - z0))
    }
    ok(worstX < 1e-9, `${tag}: column i has one x for every j (worst ${worstX})`)
    ok(worstZ < 1e-9, `${tag}: row j has one z for every i (worst ${worstZ})`)
  }

  // Non-vacuous: the x values are not all the same number either. A solve that
  // put every cell at one x would pass the check above trivially.
  const C = solveLattice(wave({}, { scrunchX: 0.3, scrunchZ: 0.2 }))
  ok(new Set([0, 1, 2, 3].map((i) => cellOf(C, i, 0).position[0])).size === 4,
    'and the four columns are at four DIFFERENT x — the check is not passing on a constant')
}

// -----------------------------------------------------------------------------
// 3. CLOSURE — every 4-cycle comes back, and a checkerboard would not.
// -----------------------------------------------------------------------------
console.log('3. the cycles close, and the checkerboard would not')
{
  const cfg = normalizeConfig(wave({ lattice: { cols: 6, rows: 7, panelType: '2x2' } },
    { scrunchX: 0.4, scrunchZ: 0.3, attractorX: 0.8, attractorZ: 1 }))
  const W = solveWave(cfg, L)
  ok(new Set(W.x.angleDeg).size === 5 && new Set(W.z.angleDeg).size === 6,
    `5 different x-angles and 6 different z-angles (got ${new Set(W.x.angleDeg).size}, ` +
    `${new Set(W.z.angleDeg).size}) — the cycle check is worthless at one angle`)

  // The wave's own step: the ramp on edge (i,j,axis) climbs σ·R(θ) and that
  // depends on ONE index, which is the separability.
  const waveStep = (i, j, axis) =>
    axis === 'x' ? W.x.sign[i] * W.x.riseCm[i] : W.z.sign[j] * W.z.riseCm[j]
  const res = worstCycleResidual(waveStep, 6, 7)
  ok(res < 1e-9, `worst 4-cycle residual is ${res.toExponential(2)}cm — the network closes`)

  // THE NEGATIVE. Take the SAME angles and put them on a two-level checkerboard,
  // where the direction of each ramp flips with the level of its low cell. Now
  // the four steps are +Rx, −Rz, +Rx, −Rz and the loop is 2(Rx − Rz).
  const boardStep = (i, j, axis) => {
    const level = (i + j) % 2
    const sign = level === 0 ? 1 : -1
    return sign * (axis === 'x' ? W.x.riseCm[i] : W.z.riseCm[j])
  }
  const boardRes = worstCycleResidual(boardStep, 6, 7)
  ok(boardRes > 1, `the same angles on a CHECKERBOARD leave the loop ${boardRes.toFixed(3)}cm open`)

  // The impossibility proof's own numbers, at the angles it quotes.
  near(2 * Math.abs(riseCm(30, 2, L) - riseCm(35, 2, L)), 9.164, 1e-3,
    'x at 30° against z at 35° opens a checkerboard cycle by 9.164cm')
  near(2 * Math.abs(riseCm(30, 2, L) - riseCm(40, 2, L)), 17.800, 1e-3,
    'and against 40° by 17.800cm')

  // The same closure asserted on the BUILT GEOMETRY rather than on the model:
  // every cell's emitted height equals f(i) + g(j) plus the one rigid shift.
  const C = solveLattice(cfg)
  let worst = 0
  for (let i = 0; i < 6; i++) {
    for (let j = 0; j < 7; j++) {
      const want = W.x.heightCm[i] + W.z.heightCm[j] + C.lattice.shiftCm[1]
      worst = Math.max(worst, Math.abs(cellOf(C, i, j).position[1] - want))
    }
  }
  ok(worst < 1e-8, `and every built cell sits at f(i) + g(j) (worst ${worst.toExponential(2)})`)
}

// -----------------------------------------------------------------------------
// 4. THE SCRUNCH IS LINEAR IN THE PLAN ADVANCE — not in the angle.
// -----------------------------------------------------------------------------
console.log('4. the compression is linear in the plan advance')
{
  for (const [scrunch, attractor] of [[0.3, 1], [0.45, 0.5], [0.2, 0.75], [0.1, 0]]) {
    const a = solveAxis({
      edgeCount: 8, baseAngleDeg: 20, gapCm: 2.5, lengthCm: L, scrunch, attractor, axis: 'x',
    })
    const base = planAdvanceCm(20, 2.5, L)
    let worst = 0
    for (let k = 0; k < 8; k++) {
      const t = k / 7
      const want = base * (1 - scrunchFactor(t, scrunch, attractor))
      worst = Math.max(worst, Math.abs(a.advanceCm[k] - want))
    }
    ok(worst < 1e-9,
      `scrunch ${scrunch} @ ${attractor}: E(θ(k)) hits its linear target (worst ${worst.toExponential(2)})`)

    // Non-vacuous the other way: the ANGLES are NOT linear. If they were, the
    // model would be the naive one this brief explicitly is not.
    if (scrunch > 0 && attractor === 1) {
      const d = a.angleDeg.slice(1).map((v, k) => v - a.angleDeg[k])
      const spread = Math.max(...d) - Math.min(...d)
      ok(spread > 0.1, `scrunch ${scrunch}: the angle steps are NOT equal (spread ${spread.toFixed(3)}°)`)
    }
  }

  // The attractor is where it stops. Past it, every edge is at full scrunch.
  const a = solveAxis({
    edgeCount: 9, baseAngleDeg: 20, gapCm: 2, lengthCm: L, scrunch: 0.3, attractor: 0.5, axis: 'x',
  })
  ok(a.factor.slice(4).every((f) => Math.abs(f - 0.3) < 1e-12),
    'past the attractor every edge sits at full scrunch')
  ok(a.factor.slice(0, 4).every((f) => f < 0.3), 'and short of it, none of them does')
}

// -----------------------------------------------------------------------------
// 5. THE SPAN — every joint is still exactly `gap`, at every differing angle.
// -----------------------------------------------------------------------------
console.log('5. the joints still span exactly the gap')
{
  for (const gap of [0.4, 2, 4.5, 8]) {
    const C = solveLattice(wave({ gap, lattice: { cols: 5, rows: 5, panelType: '2x2' } },
      { scrunchX: 0.3, scrunchZ: 0.35, attractorX: 1, attractorZ: 0.6 }))
    let worst = 0
    for (const j of C.joints) {
      const d = Math.hypot(j.rimB[0] - j.rimA[0], j.rimB[1] - j.rimA[1], j.rimB[2] - j.rimA[2])
      worst = Math.max(worst, Math.abs(d - gap))
    }
    ok(C.joints.length > 0 && worst < 1e-9,
      `gap ${gap}: all ${C.joints.length} joints span it (worst error ${worst.toExponential(2)})`)
  }

  // And the fold at each joint is the EDGE's angle, not the base angle — which is
  // the whole point of the mode and the thing a shared-θ bug would hide.
  const R = buildReportV4(wave({}, { scrunchX: 0.3, scrunchZ: 0.3 }))
  const folds = new Set(R.joints.map((j) => Math.round(Math.abs(j.foldDeg) * 1e6)))
  ok(folds.size >= 5, `the design carries ${folds.size} distinct folds, not one`)
  ok(R.envelope.angleIsPerJoint === true && R.envelope.perJoint.jointCount === R.joints.length,
    'and the envelope reports per joint rather than quoting a single-angle boundary')
}

// -----------------------------------------------------------------------------
// 6. THE BASE ANGLE IS AT THE FRONT, AND THE RUN ONLY EVER TIGHTENS.
// -----------------------------------------------------------------------------
console.log('6. base angle at the front, monotone from there')
{
  for (const scrunch of [0, 0.1, 0.35, 0.6, SCRUNCH_MAX]) {
    const a = solveAxis({
      edgeCount: 7, baseAngleDeg: 22.5, gapCm: 2, lengthCm: L, scrunch, attractor: 1, axis: 'x',
    })
    ok(a.angleDeg[0] === 22.5, `scrunch ${scrunch}: θ(0) is EXACTLY the base angle`)
    ok(a.angleDeg.every((v, k) => k === 0 || v >= a.angleDeg[k - 1] - 1e-12),
      `scrunch ${scrunch}: the angles never come back down`)
    ok(a.advanceCm.every((v, k) => k === 0 || v <= a.advanceCm[k - 1] + 1e-12),
      `scrunch ${scrunch}: the plan pitch never grows`)
    ok(a.angleDeg.every((v) => v >= 22.5 - 1e-12 && v <= ANGLE_MAX + 1e-12),
      `scrunch ${scrunch}: every angle is in [base, ANGLE_MAX]`)
  }

  // attractor 0 is the documented exception: everything is already past the
  // attractor, so the whole axis compresses uniformly and edge 0 is NOT at base.
  const flat = solveAxis({
    edgeCount: 5, baseAngleDeg: 22.5, gapCm: 2, lengthCm: L, scrunch: 0.25, attractor: 0, axis: 'x',
  })
  ok(new Set(flat.angleDeg.map((v) => Math.round(v * 1e9))).size === 1 && flat.angleDeg[0] > 22.5,
    'attractor 0 compresses the whole axis uniformly, above the base angle')

  // One edge has no run to ramp along, so it stays at the base angle.
  const one = solveAxis({
    edgeCount: 1, baseAngleDeg: 22.5, gapCm: 2, lengthCm: L, scrunch: 0.5, attractor: 1, axis: 'x',
  })
  ok(one.angleDeg[0] === 22.5, 'a single edge has nothing to ramp toward and sits at the base angle')
}

// -----------------------------------------------------------------------------
// 7. ZERO SCRUNCH — uniform, and still NOT the checkerboard.
// -----------------------------------------------------------------------------
console.log('7. zero scrunch is uniform, and is an egg-crate rather than a checkerboard')
{
  const W = solveWave(normalizeConfig(wave()), L)
  ok(W.x.angleDeg.every((v) => v === 25) && W.z.angleDeg.every((v) => v === 25),
    'every edge sits at the base angle, exactly')

  const C = solveLattice(wave())
  const R = riseCm(25, 2, L)
  // Three storeys — 0, R, 2R — which is the honest shape of the separable family
  // and must not be "fixed" back to two. wave.js's header has the proof.
  ok(C.lattice.wave.storeyCount === 3,
    `three storeys, not two (got ${C.lattice.wave.storeyCount})`)
  near(cellOf(C, 1, 1).position[1] - cellOf(C, 0, 0).position[1], 2 * R, 1e-9,
    'cell (1,1) is TWO rises up, where the checkerboard would put it back on the floor')

  // ...and the trapezoid at the same knobs really does put it back down, so the
  // difference is measured rather than asserted.
  const T = solveLattice({ ...wave(), pattern: { kind: 'trapezoid', phase: 0 } })
  near(cellOf(T, 1, 1).position[1] - cellOf(T, 0, 0).position[1], 0, 1e-9,
    'while the checkerboard puts (1,1) back level with (0,0)')

  ok(PATTERN_KINDS.includes('wave') && PATTERN_KINDS.includes('trapezoid'),
    'both kinds are declared in the schema')
  ok(normalizeConfig({}).pattern.kind === 'trapezoid' && normalizeConfig({}).pattern.wave === undefined,
    'and trapezoid is still the default, with no wave block written into it')
  ok(normalizeConfig(wave()).pattern.wave !== undefined, 'while a wave config carries all four knobs')
}

// -----------------------------------------------------------------------------
// 8. THE TRAPEZOID DID NOT MOVE — the single most important check here.
//
// FNV-1a over `JSON.stringify(solveLattice(cfg))`, taken from the tree at
// `71f8b15`. A hash rather than a stored blob because the blob is 1.5MB and the
// question is binary; a hash AND a length because the two together are what
// makes a collision not worth worrying about.
// -----------------------------------------------------------------------------
console.log('8. every trapezoid solve is byte-identical to 71f8b15')
{
  const fnv1a = (s) => {
    let h = 0x811c9dc5
    for (let i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i)
      h = Math.imul(h, 0x01000193) >>> 0
    }
    return h >>> 0
  }

  const FROZEN = [
    // EVERY config pins `yOffsetCm` explicitly. These hashes are the frozen
    // record of the trapezoid geometry at 71f8b15, and they must not move when
    // an unrelated DEFAULT does — which is exactly what happened when the
    // default offset went 15 → 0 and six of the seven fired at once. A
    // regression check that trips on a deliberate, unrelated change is a check
    // whose numbers get rubber-stamped.
    ['default', 888684285, 54139, { placement: { yOffsetCm: 15 } }],
    ['ribbon', 647362578, 10538,
      { lattice: { cols: 1, rows: 5, panelType: '2x2' }, angleDeg: 30, gap: 2, placement: { yOffsetCm: 15 } }],
    ['4x4-45-phase1', 243507324, 60547,
      { lattice: { cols: 4, rows: 4, panelType: '2x2' }, angleDeg: 45, gap: 3.5, pattern: { kind: 'trapezoid', phase: 1 }, placement: { yOffsetCm: 15 } }],
    ['flat', 2459399168, 13457,
      { lattice: { cols: 2, rows: 3, panelType: '2x2' }, angleDeg: 0, gap: 0.4, placement: { yOffsetCm: 15 } }],
    ['braced-62.5-8', 2080408362, 60248,
      { lattice: { cols: 3, rows: 5, panelType: '2x2' }, angleDeg: 62.5, gap: 8, placement: { wallAnchor: 'braced', groundToFloor: true, yOffsetCm: 15, wallOffsetCm: 12, windowOffsetCm: 7 } }],
    ['overrides', 2362091016, 30377,
      { lattice: { cols: 5, rows: 2, panelType: '2x2' }, angleDeg: 20, gap: 1.25, placement: { yOffsetCm: 15 }, overrides: { cells: [{ i: 1, j: 0, present: false }, { i: 3, j: 1, flipped: true }], edges: [{ i: 2, j: 1, axis: 'x', present: false }] } }],
    ['6x6-braced', 3464671071, 156438,
      { lattice: { cols: 6, rows: 6, panelType: '2x2' }, angleDeg: 33.3, gap: 2.4, placement: { wallAnchor: 'braced', yOffsetCm: 15 } }],
  ]

  // Hash the GEOMETRY, not the whole solve. The record also carries the
  // normalized `config` it was solved from, so hashing that too made this fire
  // on any additive schema field — `room.wallThicknessCm` moved all seven by
  // exactly 31 bytes while not a single panel changed. That is a false alarm,
  // and a regression check that cries wolf gets its numbers rubber-stamped,
  // which is the one thing it must never train anyone to do. What this section
  // is for is that the SOLVER's output is frozen; the config envelope has
  // schema.js's own idempotence tests.
  const geometryOf = (o) =>
    JSON.stringify({ lattice: o.lattice, panels: o.panels, joints: o.joints, bounds: o.bounds })

  for (const [name, hash, length, cfg] of FROZEN) {
    const s = geometryOf(solveLattice(cfg))
    ok(s.length === length, `${name}: solve is ${length} bytes (got ${s.length})`)
    ok(fnv1a(s) === hash, `${name}: solve hashes to ${hash} (got ${fnv1a(s)})`)
  }

  // The hash is not a constant: change one knob and it must move.
  const moved = geometryOf(solveLattice({ angleDeg: 30.5 }))
  ok(fnv1a(moved) !== FROZEN[0][1], 'and a different angle hashes differently — the check can fail')

  // The trapezoid record has no wave-only fields on it.
  const T = solveLattice({})
  ok(T.lattice.wave === undefined && T.panels.every((p) => p.heightCm === undefined),
    'no wave block and no heightCm anywhere on a checkerboard solve')
}

// -----------------------------------------------------------------------------
// 9. DETERMINISM — the standing rule in this core.
// -----------------------------------------------------------------------------
console.log('9. determinism')
{
  const cfg = wave({ lattice: { cols: 5, rows: 5, panelType: '2x2' } },
    { scrunchX: 0.31, scrunchZ: 0.17, attractorX: 0.7, attractorZ: 1 })
  const a = JSON.stringify(solveLattice(cfg))
  const b = JSON.stringify(solveLattice(JSON.parse(JSON.stringify(cfg))))
  ok(a === b, 'two solves of the same wave are byte-identical')

  const n1 = JSON.stringify(normalizeConfig(cfg))
  const n2 = JSON.stringify(normalizeConfig(normalizeConfig(cfg)))
  ok(n1 === n2, 'and normalizeConfig is idempotent on a wave config')

  const r1 = JSON.stringify(buildReportV4(cfg))
  const r2 = JSON.stringify(buildReportV4(cfg))
  ok(r1 === r2, 'and so is the report')
}

// -----------------------------------------------------------------------------
// 10. THE SCRUNCH THE ANGLE BAND CANNOT DELIVER.
// -----------------------------------------------------------------------------
console.log('10. an unreachable scrunch is reported, not silently clamped')
{
  const floorCm = planAdvanceCm(ANGLE_MAX, 2, L)
  const base = planAdvanceCm(25, 2, L)
  const reachable = 1 - floorCm / base
  ok(reachable > 0.5 && reachable < SCRUNCH_MAX,
    `at 25°/2cm the steepest reachable scrunch is ${(reachable * 100).toFixed(1)}%, inside the band ` +
    '— so the band\'s top is a place the solver reports from')

  const C = solveLattice(wave({ lattice: { cols: 8, rows: 2, panelType: '2x2' } },
    { scrunchX: SCRUNCH_MAX, attractorX: 1 }))
  const w = C.lattice.wave.warnings.filter((x) => x.code === SCRUNCH_UNREACHABLE_CODE)
  ok(w.length > 0, `${w.length} edges are pinned at ANGLE_MAX and each is named`)
  ok(w.every((x) => x.axis === 'x' && Number.isInteger(x.index)), 'and each names its axis and index')
  ok(C.lattice.wave.x.angleDeg.every((v) => v <= ANGLE_MAX + 1e-12), 'nothing exceeds ANGLE_MAX')
  ok(buildReportV4(wave({ lattice: { cols: 8, rows: 2, panelType: '2x2' } }, { scrunchX: SCRUNCH_MAX }))
    .warnings.some((x) => x.code === SCRUNCH_UNREACHABLE_CODE),
    'and the warning reaches the report')

  // A reachable scrunch raises nothing.
  ok(solveLattice(wave({}, { scrunchX: 0.2, scrunchZ: 0.2 })).lattice.wave.warnings.length === 0,
    'while a reachable one is silent')

  // The bisection itself: the angle it returns really does advance by the target.
  for (const target of [55, 45, 30, 20]) {
    const { angleDeg, reached } = angleForAdvance(target, 10, 2, L)
    if (!reached) continue
    near(planAdvanceCm(angleDeg, 2, L), target, 1e-9, `angleForAdvance(${target}) inverts E exactly`)
  }
}

// -----------------------------------------------------------------------------
// 11. THE SCHEMA — the four knobs are range-checked, and the block is conditional.
// -----------------------------------------------------------------------------
console.log('11. schema')
{
  const raises = (cfg, code, path) =>
    validateConfig(cfg).errors.some((e) => e.code === code && e.path === path)

  for (const key of ['scrunchX', 'scrunchZ']) {
    ok(raises(wave({}, { [key]: 2 }), 'E_RANGE', `pattern.wave.${key}`), `${key} above the band is E_RANGE`)
    ok(raises(wave({}, { [key]: -0.1 }), 'E_RANGE', `pattern.wave.${key}`), `${key} below it is too`)
  }
  for (const key of ['attractorX', 'attractorZ']) {
    ok(raises(wave({}, { [key]: 1.5 }), 'E_RANGE', `pattern.wave.${key}`), `${key} above the band is E_RANGE`)
    ok(raises(wave({}, { [key]: -1 }), 'E_RANGE', `pattern.wave.${key}`), `${key} below it is too`)
  }
  ok(raises({ ...wave(), pattern: { kind: 'wave', wave: 7 } }, 'E_SHAPE', 'pattern.wave'),
    'a wave block that is not an object is E_SHAPE')
  ok(validateConfig(wave()).valid, 'and a well-formed wave config validates')

  // Clamping, and the conditional emission.
  const n = normalizeConfig(wave({}, { scrunchX: 9, attractorZ: -3 }))
  ok(n.pattern.wave.scrunchX === SCRUNCH_MAX && n.pattern.wave.attractorZ === 0,
    'normalizeConfig clamps where validateConfig reports')
  const dropped = normalizeConfig({ pattern: { kind: 'trapezoid', wave: { scrunchX: 0.5 } } })
  ok(dropped.pattern.wave === undefined, 'a wave block on a trapezoid is dropped')
  ok(validateConfig({ pattern: { kind: 'trapezoid', wave: { scrunchX: 0.5 } } })
    .warnings.some((x) => x.code === 'W_WAVE_SETTINGS_IGNORED'), 'and reported as ignored first')
  ok(validateConfig(wave({ pattern: undefined }, {})).valid, 'a wave with default knobs is valid')
}

// -----------------------------------------------------------------------------
// 12. THE THINGS THE WAVE DOES NOT DO, stated rather than implied.
// -----------------------------------------------------------------------------
console.log('12. the wall anchor, and the floor')
{
  const R = buildReportV4(wave({ placement: { wallAnchor: 'braced', groundToFloor: true, yOffsetCm: 15, wallOffsetCm: 0, windowOffsetCm: 0 } },
    { scrunchX: 0.2, scrunchZ: 0.2 }))
  ok(R.metrics.counts.anchorRamps === 0, 'no anchor ramps are built on a wave')
  ok(R.warnings.some((w) => w.code === 'W_WAVE_NO_ANCHOR'), 'and the request is reported as declined')
  // The checkerboard still builds them, so the absence is the wave's and not a
  // regression in the anchor itself.
  ok(buildReportV4({ placement: { wallAnchor: 'braced' } }).metrics.counts.anchorRamps > 0,
    'while the checkerboard still builds its own')

  // The y offset lands the wave's lowest material exactly on the number, the
  // same as it does a checkerboard — a scrunched wave has an UNEVEN floor, so
  // this is where a per-cell lift masquerading as a translation would show.
  ok(R.metrics.overall.min[1] === 15, 'the scrunched wave is grounded at exactly the 15cm offset')
  const even = buildReportV4(wave({ placement: { wallAnchor: 'free', groundToFloor: true, yOffsetCm: 15, wallOffsetCm: 0, windowOffsetCm: 0 } }))
  ok(even.metrics.overall.min[1] === 15, 'and so is the unscrunched one, whose floor is level')
}

console.log(`\n${passed} passed, ${failed} failed`)
process.exit(failed ? 1 : 0)
