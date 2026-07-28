/**
 * tests/test-v4-spacers.mjs — headless checks for core/v4/spacers.js and the
 * ground clearance it holds open.
 *
 * Plain node script, no test framework. Exits non-zero on any failure.
 *   node tests/test-v4-spacers.mjs
 *
 * Closed-form expectations throughout, and every one of them is non-vacuous in
 * both directions — a clearance test that only checks 15 would pass against a
 * hardcoded 15, so §1 checks 0 as well and checks that the design at 0 is
 * genuinely different.
 *
 * The claims worth naming, all of which would fail silently:
 *
 *   THE CLEARANCE IS EXACT. §1. `groundClearanceCm` is not a nudge — the minimum
 *        y over every present panel's OBB corners is the clearance to 1e-9, at
 *        several angles and gaps. It has to be exact because the spacers are cut
 *        to it: a network sitting 14.97cm up is a network whose posts do not fit.
 *   THE SPACERS AGREE WITH THE GROUNDING. §2. Height measured off the panel,
 *        clearance applied by the lattice, two modules, one number.
 *   "MATCHING THE PATTERN" IS A REUSED RULE. §6. Moving `connectors.spacingCm`
 *        moves the spacer count exactly as it moves the connector count. This is
 *        the check that distinguishes a shared rule from a coincidence, and it
 *        is the one the brief actually asks for.
 */

import { solveSpacers, SPACER_SECTION_CM, SPACER_MISMATCH_CODE } from '../src/core/v4/spacers.js'
import { solveLattice } from '../src/core/v4/lattice.js'
import { solveConnectorsV4 } from '../src/core/v4/connectors.js'
import { buildReportV4 } from '../src/core/v4/report.js'
import { stationCount } from '../src/core/v3/connectors.js'
import {
  DEFAULT_CONFIG,
  normalizeConfig,
  validateConfig,
  GROUND_CLEARANCE_MIN,
  GROUND_CLEARANCE_MAX,
} from '../src/core/v4/schema.js'
import { PANEL_DIMENSIONS, PANEL_PROFILE } from '../src/config.js'

let passed = 0
let failed = 0
const ok = (c, m) => { if (c) passed++; else { failed++; console.log(`  FAIL: ${m}`) } }
const near = (a, b, tol, m) => ok(Math.abs(a - b) <= tol, `${m} (got ${a}, want ${b} ±${tol})`)

const L = PANEL_DIMENSIONS['2x2'].height

const solve = (patch = {}) => {
  const cfg = normalizeConfig({ ...DEFAULT_CONFIG, ...patch })
  const C = solveLattice(cfg)
  return { cfg, C, S: solveSpacers(cfg, C) }
}

/** The lowest corner of any PRESENT panel's OBB — what grounding aims. */
function minY(C) {
  let y = Infinity
  for (const p of C.panels) {
    if (!p.present) continue
    for (const c of p.corners) if (c[1] < y) y = c[1]
  }
  return y
}

console.log('=== test-v4-spacers ===')

// -----------------------------------------------------------------------------
// 1. THE CLEARANCE — exact, at several angles and gaps, and non-vacuous at 0
// -----------------------------------------------------------------------------
console.log('1. groundClearanceCm puts the lowest material exactly there')
{
  ok(normalizeConfig({}).placement.groundClearanceCm === 15,
    'the default clearance is 15cm — the brief, not a nudge')

  for (const angleDeg of [0, 12, 30, 45, 60]) {
    for (const gap of [0.4, 2, 5]) {
      const at15 = solve({ angleDeg, gap, placement: { ...DEFAULT_CONFIG.placement, groundClearanceCm: 15 } })
      near(minY(at15.C), 15, 1e-9, `θ=${angleDeg}° gap=${gap}: lowest OBB corner is exactly 15`)

      const at0 = solve({ angleDeg, gap, placement: { ...DEFAULT_CONFIG.placement, groundClearanceCm: 0 } })
      near(minY(at0.C), 0, 1e-9, `θ=${angleDeg}° gap=${gap}: at clearance 0 it is exactly 0`)
    }
  }

  // Non-vacuous the other way: the two designs are NOT the same object shifted
  // by nothing — every panel moved by exactly 15 in y and by nothing else.
  const a = solve({ placement: { ...DEFAULT_CONFIG.placement, groundClearanceCm: 0 } })
  const b = solve({ placement: { ...DEFAULT_CONFIG.placement, groundClearanceCm: 15 } })
  let worstY = 0
  let worstXZ = 0
  for (let k = 0; k < a.C.panels.length; k++) {
    const p = a.C.panels[k]
    const q = b.C.panels[k]
    worstY = Math.max(worstY, Math.abs((q.position[1] - p.position[1]) - 15))
    worstXZ = Math.max(worstXZ, Math.abs(q.position[0] - p.position[0]), Math.abs(q.position[2] - p.position[2]))
  }
  near(worstY, 0, 1e-9, 'every panel rose by exactly 15 — a rigid translation, not a re-solve')
  near(worstXZ, 0, 1e-9, 'and moved by nothing in x or z')

  // The height of the box is unchanged: raising the floor raises everything.
  near(b.C.bounds.size[1], a.C.bounds.size[1], 1e-9, 'the measuring box is the same height either way')
  near(b.C.bounds.min[1] - a.C.bounds.min[1], 15, 1e-9, 'its floor is 15 higher')

  // Grounding OFF ignores the clearance entirely — nothing to measure from.
  const free = solve({
    placement: { ...DEFAULT_CONFIG.placement, groundToFloor: false, groundClearanceCm: 15 },
  })
  ok(minY(free.C) < 0, 'with grounding off the clearance does nothing and panels dip below y = 0')
  ok(free.S.spacers.length === 0 && free.S.grounded === false,
    'and no spacers are solved — a post to a cell under the floor is not a part')
}

// -----------------------------------------------------------------------------
// 2. EVERY SPACER SPANS THE CLEARANCE, FOOT ON THE FLOOR
// -----------------------------------------------------------------------------
console.log('2. every post is the clearance tall and stands on y = 0')
{
  for (const clearance of [0, 5, 15, 27.5, GROUND_CLEARANCE_MAX]) {
    const { S } = solve({ placement: { ...DEFAULT_CONFIG.placement, groundClearanceCm: clearance } })
    const heights = new Set(S.spacers.map((sp) => sp.heightCm))
    ok(heights.size === 1, `clearance ${clearance}: exactly one distinct spacer height (got ${heights.size})`)
    near([...heights][0], clearance, 1e-9, `clearance ${clearance}: and it is the clearance`)
    ok(S.heightsCm.length === 1 && S.heightsCm[0] === clearance,
      `clearance ${clearance}: the solver reports that single height`)

    // The foot: the OBB's bottom face. center.y − halfExtent.y = 0.
    let worstFoot = 0
    let worstTop = 0
    for (const sp of S.spacers) {
      worstFoot = Math.max(worstFoot, Math.abs(sp.obb.center[1] - sp.obb.halfExtents[1]))
      worstTop = Math.max(worstTop, Math.abs((sp.obb.center[1] + sp.obb.halfExtents[1]) - sp.position[1]))
    }
    near(worstFoot, 0, 1e-9, `clearance ${clearance}: every foot is on the floor`)
    near(worstTop, 0, 1e-9, `clearance ${clearance}: and every top is on the underside it holds up`)
  }

  // The section is derived from the measured panel, not written down.
  ok(SPACER_SECTION_CM === PANEL_PROFILE.overallThickness,
    'the post section is the panel housing thickness, derived rather than chosen')
}

// -----------------------------------------------------------------------------
// 3. THE COUNT — computed independently from stationCount
// -----------------------------------------------------------------------------
console.log('3. the count is stationCount × 4 edges × qualifying cells')
{
  for (const spacingCm of [10, 20, 31, 50, 200]) {
    for (const minPerJoint of [1, 2, 4]) {
      const { cfg, C, S } = solve({
        connectors: { ...DEFAULT_CONFIG.connectors, spacingCm, minPerJoint },
      })
      // Independently: the qualifying cells are the present ones at the lowest
      // level, and a cell edge is L long.
      const flats = C.panels.filter((p) => p.kind === 'cell' && p.present)
      const lowestLevel = Math.min(...flats.map((p) => p.level))
      const qualifying = flats.filter((p) => p.level === lowestLevel).length
      const perEdge = stationCount(L, { spacingCm, minPerJoint })
      const want = qualifying * 4 * perEdge
      ok(S.spacers.length === want,
        `spacing ${spacingCm} min ${minPerJoint}: ${S.spacers.length} spacers, want ` +
        `${qualifying} cells × 4 edges × ${perEdge} = ${want}`)
      ok(S.perEdge === perEdge, `spacing ${spacingCm} min ${minPerJoint}: perEdge reported as ${perEdge}`)
      ok(S.perCell.length === qualifying && S.perCell.every((c) => c.count === 4 * perEdge),
        `spacing ${spacingCm} min ${minPerJoint}: perCell is ${qualifying} entries of ${4 * perEdge}`)
      ok(cfg.connectors.spacingCm === spacingCm, 'and the knob survived normalization')
    }
  }

  // The default design, stated as a number so a change to it is visible here.
  const { S } = solve()
  ok(S.spacers.length === 64,
    `the default 3 × 5 network takes 64 spacers — 8 floor cells × 4 edges × 2 (got ${S.spacers.length})`)

  // Stations are evenly spaced and symmetric WITHIN the edge, exactly as
  // solveConnectorsV4 places its own: centres at (m + 0.5)/n.
  const one = S.spacers.filter((sp) => sp.cell === S.cells[0] && sp.edge === '+x').sort((a, b) => a.s - b.s)
  ok(one.length >= 2, 'the first cell has at least two stations on its +x edge')
  const lo = Math.min(...one.map((sp) => sp.s))
  const hi = Math.max(...one.map((sp) => sp.s))
  const mid = one.reduce((n, sp) => n + sp.s, 0) / one.length
  const edgeMid = (lo + hi) / 2
  near(mid, edgeMid, 1e-9, 'the stations are symmetric about the middle of their edge')
  near(hi - lo, L * (one.length - 1) / one.length, 1e-9, 'and evenly spaced at the (m+0.5)/n rule')
}

// -----------------------------------------------------------------------------
// 4. ONLY THE CELLS RESTING ON THE FLOOR
// -----------------------------------------------------------------------------
console.log('4. spacers appear only under the lowest-level cells')
{
  const { C, S } = solve()
  const byId = new Map(C.panels.map((p) => [p.id, p]))
  ok(S.cells.every((id) => byId.get(id).kind === 'cell'), 'every spacered thing is a flat cell')
  ok(S.cells.every((id) => byId.get(id).level === 0),
    'and every one of them is a ground cell — no post under a cell the ramps hold up')
  const highs = C.panels.filter((p) => p.kind === 'cell' && p.present && p.level === 1)
  ok(highs.length > 0 && !S.cells.some((id) => byId.get(id).level === 1),
    `there ARE ${highs.length} high cells and not one of them is spacered — non-vacuous`)

  // The phase swaps which cells are on the floor; the count follows it.
  const swapped = solve({ pattern: { kind: 'trapezoid', phase: 1 } })
  ok(swapped.S.cells.length === 7 && S.cells.length === 8,
    `phase 1 puts 7 cells on the floor where phase 0 puts 8 (got ${swapped.S.cells.length} / ${S.cells.length})`)

  // REMOVING a floor cell removes exactly its spacers and no others.
  const victim = S.cells[0]
  const v = byId.get(victim)
  const cut = solve({
    overrides: { cells: [{ i: v.i, j: v.j, present: false, flipped: false }], edges: [] },
  })
  const before = new Set(S.spacers.map((sp) => sp.id))
  const after = new Set(cut.S.spacers.map((sp) => sp.id))
  const gone = [...before].filter((id) => !after.has(id))
  const added = [...after].filter((id) => !before.has(id))
  ok(gone.length === S.perCell.find((c) => c.cell === victim).count,
    `removing ${victim} removed exactly its ${gone.length} spacers`)
  ok(gone.every((id) => id.startsWith(`Sp${victim}`)), 'and they are all its own')
  ok(added.length === 0, 'and no other spacer moved or appeared')

  // Removing a HIGH cell touches no spacer at all.
  const hi = highs[0]
  const cutHigh = solve({
    overrides: { cells: [{ i: hi.i, j: hi.j, present: false, flipped: false }], edges: [] },
  })
  ok(JSON.stringify(cutHigh.S.spacers) === JSON.stringify(S.spacers),
    'removing a HIGH cell leaves every spacer bit-identical')
}

// -----------------------------------------------------------------------------
// 5. DETERMINISM
// -----------------------------------------------------------------------------
console.log('5. two solves are byte-identical')
{
  const a = solve()
  const b = solve()
  ok(JSON.stringify(a.S) === JSON.stringify(b.S), 'the default design solves identically twice')

  const patch = {
    angleDeg: 41.5,
    gap: 3.25,
    lattice: { cols: 4, rows: 4, panelType: '2x2' },
    placement: { ...DEFAULT_CONFIG.placement, groundClearanceCm: 22.5, wallAnchor: 'braced' },
  }
  ok(JSON.stringify(solve(patch).S) === JSON.stringify(solve(patch).S),
    'and so does an awkward one, braced, at a non-round clearance')
  ok(JSON.stringify(a.S) !== JSON.stringify(solve(patch).S),
    'while the two designs do NOT solve the same — the comparison has teeth')
}

// -----------------------------------------------------------------------------
// 6. THE SPACING RULE IS THE CONNECTORS' OWN
// -----------------------------------------------------------------------------
console.log('6. connectors.spacingCm moves the spacer count the way it moves the connector count')
{
  // A joint's rim and a cell's edge are both L long, so if the two really share
  // `stationCount` the PER-STRETCH counts have to track each other exactly.
  // Powered joints are split by the driver in 'block' mode, so 'relief' (the
  // default) is what makes the comparison a like-for-like one.
  const seen = []
  for (const spacingCm of [10, 15, 20, 30, 50, 120, 200]) {
    const cfg = normalizeConfig({ ...DEFAULT_CONFIG, connectors: { ...DEFAULT_CONFIG.connectors, spacingCm } })
    const C = solveLattice(cfg)
    const K = solveConnectorsV4(cfg, C)
    const S = solveSpacers(cfg, C)

    const perJoint = K.perJoint[0].count
    const spacerPerEdge = S.perEdge
    ok(spacerPerEdge === perJoint,
      `spacing ${spacingCm}: ${spacerPerEdge} spacers per edge and ${perJoint} connectors per joint`)
    seen.push(spacerPerEdge)

    // And the totals move together, not just the per-edge figure.
    const edges = S.perCell.length * 4
    ok(S.spacers.length === edges * spacerPerEdge, `spacing ${spacingCm}: the totals follow`)
  }
  ok(new Set(seen).size > 1,
    `the count actually MOVED across the sweep (${seen.join(', ')}) — otherwise this proves nothing`)

  // minPerJoint is the other half of the shared rule.
  for (const minPerJoint of [1, 2, 3, 6]) {
    const cfg = normalizeConfig({
      ...DEFAULT_CONFIG,
      connectors: { ...DEFAULT_CONFIG.connectors, spacingCm: 200, minPerJoint },
    })
    const C = solveLattice(cfg)
    const S = solveSpacers(cfg, C)
    ok(S.perEdge === minPerJoint,
      `minPerJoint ${minPerJoint}: the floor applies to the spacers too (got ${S.perEdge})`)
  }
}

// -----------------------------------------------------------------------------
// 7. THE REPORT, AND THE MISMATCH WARNING
// -----------------------------------------------------------------------------
console.log('7. report.spacers, and W_SPACER_MISMATCH')
{
  const cfg = normalizeConfig(DEFAULT_CONFIG)
  const C = solveLattice(cfg)
  const K = solveConnectorsV4(cfg, C)
  const S = solveSpacers(cfg, C)
  const R = buildReportV4(cfg, C, K, S)

  ok(R.spacers.count === S.spacers.length, 'the report counts what the solver emitted')
  ok(R.spacers.clearanceCm === 15, 'and quotes the clearance')
  ok(R.spacers.heightsCm.length === 1 && R.spacers.heightsCm[0] === 15,
    'exactly one distinct height, equal to the clearance')
  ok(R.spacers.perCell.length === 8 && R.spacers.cellCount === 8, 'eight floor-resting cells')
  ok(!R.warnings.some((w) => w.code === SPACER_MISMATCH_CODE),
    'a healthy grounded design raises no mismatch')

  // A design with NO ground cells present: the lowest flat cells are then the
  // high ones, and something else (a ramp toe) is the lowest material — so the
  // posts are the wrong length and the warning has to fire. This is the whole
  // reason `heightCm` is measured rather than restated.
  const highsOnly = {
    ...DEFAULT_CONFIG,
    lattice: { cols: 2, rows: 2, panelType: '2x2' },
    overrides: {
      cells: [{ i: 0, j: 0, present: false, flipped: false }, { i: 1, j: 1, present: false, flipped: false }],
      edges: [],
    },
  }
  const cfg2 = normalizeConfig(highsOnly)
  const C2 = solveLattice(cfg2)
  const S2 = solveSpacers(cfg2, C2)
  const R2 = buildReportV4(cfg2, C2, solveConnectorsV4(cfg2, C2), S2)
  const groundLeft = C2.panels.filter((p) => p.kind === 'cell' && p.present && p.level === 0).length
  ok(groundLeft === 0, 'the probe design really has no ground cells left')
  const w = R2.warnings.find((x) => x.code === SPACER_MISMATCH_CODE)
  ok(Boolean(w), 'and W_SPACER_MISMATCH fires — the posts do not span the clearance')
  ok(w && w.count === S2.spacers.length && Math.abs(w.worstHeightCm - 15) > 1e-6,
    'naming how many and the worst height')
}

// -----------------------------------------------------------------------------
// 8. THE SCHEMA — clamped in normalize, REPORTED in validate
// -----------------------------------------------------------------------------
console.log('8. groundClearanceCm is range-checked, not silently corrected')
{
  ok(normalizeConfig({ placement: { groundClearanceCm: 999 } }).placement.groundClearanceCm
    === GROUND_CLEARANCE_MAX, 'normalizeConfig clamps a wild value')
  ok(normalizeConfig({ placement: { groundClearanceCm: -20 } }).placement.groundClearanceCm
    === GROUND_CLEARANCE_MIN, 'at both ends')

  const over = validateConfig({ ...DEFAULT_CONFIG, placement: { ...DEFAULT_CONFIG.placement, groundClearanceCm: 999 } })
  ok(!over.valid && over.errors.some((e) => e.code === 'E_RANGE' && e.path === 'placement.groundClearanceCm'),
    'validateConfig REPORTS the out-of-range value rather than clamping it')
  const under = validateConfig({ ...DEFAULT_CONFIG, placement: { ...DEFAULT_CONFIG.placement, groundClearanceCm: -1 } })
  ok(!under.valid && under.errors.some((e) => e.code === 'E_RANGE'), 'and below the floor too')
  const nan = validateConfig({ ...DEFAULT_CONFIG, placement: { ...DEFAULT_CONFIG.placement, groundClearanceCm: 'tall' } })
  ok(!nan.valid && nan.errors.some((e) => e.code === 'E_SHAPE'), 'a non-number is a shape error')
  ok(validateConfig(DEFAULT_CONFIG).valid, 'and the shipped default still validates — non-vacuous')

  // Idempotence, the contract normalizeConfig owes every consumer.
  const once = normalizeConfig({ placement: { groundClearanceCm: 17.25 } })
  ok(JSON.stringify(normalizeConfig(once)) === JSON.stringify(once), 'normalizeConfig stays idempotent')
}

console.log(`\ntest-v4-spacers: ${passed} checks passed, ${failed} failed`)
process.exit(failed ? 1 : 0)
