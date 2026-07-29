/**
 * tests/test-v4-obstacles.mjs — headless checks for core/v4/obstacles.js.
 *
 * Plain node script, no test framework. Exits non-zero on any failure.
 *   node tests/test-v4-obstacles.mjs
 *
 * The subject is a STRUCTURAL COLUMN measured on site, so the thing most worth
 * testing is not the SAT — collide.js already has 76 checks on that — but the
 * two ways a correct collision test still gives you the wrong answer:
 *
 *   1. the box is in the wrong place, because `(x, z)` was read as a centre
 *      when it meant a corner (a 25cm error on a 50cm column), and
 *   2. the verdict is silently absolute — a design that misses the column
 *      reports the same "no hits" whether it clears by 90cm or by 2mm.
 *
 * So §1 pins the extents against hand-arithmetic under BOTH anchors, and §3
 * checks the clearance is a real distance that falls as the network grows
 * toward the column, not just a null-vs-number flag.
 */

import {
  obstacleExtents,
  obstacleOBB,
  obstacleClearanceCm,
  solveObstacles,
  OBSTACLE_HIT_CODE,
} from '../src/core/v4/obstacles.js'
import { solveLattice } from '../src/core/v4/lattice.js'
import { buildReportV4 } from '../src/core/v4/report.js'
import { normalizeConfig, DEFAULT_CONFIG, DEFAULT_OBSTACLES, OBSTACLE_SIZE_MIN } from '../src/core/v4/schema.js'
import { obbPenetration } from '../src/core/v3/collide.js'

let passed = 0
let failed = 0
const ok = (c, m) => { if (c) passed++; else { failed++; console.log(`  FAIL: ${m}`) } }
const near = (a, b, tol, m) => ok(Math.abs(a - b) <= tol, `${m} (got ${a}, want ${b} ±${tol})`)

/** The room's column, as measured: 380 in x, 285 in z, 50cm square. */
const COLUMN = { id: 'column', label: 'column', xCm: 380, zCm: 285, widthCm: 50, depthCm: 50, heightCm: 300, anchor: 'corner' }

console.log('=== test-v4-obstacles ===')

// -----------------------------------------------------------------------------
// 1. WHERE THE BOX ACTUALLY IS
// -----------------------------------------------------------------------------
console.log('1. the anchor decides the extents, and it is 25cm either way')
{
  const c = obstacleExtents(COLUMN)
  // corner: (380, 285) is the MINIMUM corner, so the box runs to (430, 335).
  ok(c.min[0] === 380 && c.min[2] === 285, 'corner anchor: min is the quoted point')
  ok(c.max[0] === 430 && c.max[2] === 335, 'and it runs 50cm to (430, 335)')
  ok(c.centre[0] === 405 && c.centre[2] === 310, 'centre is halfway')
  ok(c.min[1] === 0, 'it stands ON the floor')
  ok(c.max[1] === 300, 'and runs to its full height')

  const m = obstacleExtents({ ...COLUMN, anchor: 'centre' })
  ok(m.min[0] === 355 && m.min[2] === 260, 'centre anchor: the box is offset by half its size')
  ok(m.max[0] === 405 && m.max[2] === 310, '...on both faces')
  ok(m.centre[0] === 380 && m.centre[2] === 285, 'and the quoted point IS the centre')

  // The whole reason `anchor` exists rather than being assumed.
  ok(c.min[0] - m.min[0] === 25, 'the two readings differ by half the width — 25cm on this column')

  const box = obstacleOBB(COLUMN)
  ok(JSON.stringify(box.quaternion) === JSON.stringify([0, 0, 0, 1]),
    'the OBB is unrotated — a column is axis-aligned, which is why the SAT is exact here')
  ok(JSON.stringify(box.halfExtents) === JSON.stringify([25, 150, 25]), 'and its half-extents are the size halved')
}

// -----------------------------------------------------------------------------
// 2. WHAT IT HITS — and a non-vacuous negative on both sides
// -----------------------------------------------------------------------------
console.log('2. hits are found, and absence of hits is not an artifact')
{
  const cfg = (cols) => normalizeConfig({ ...DEFAULT_CONFIG, lattice: { cols, rows: 5, panelType: '2x2' } })

  // The default 3 × 5 network stops well short of x = 380, so it MUST be clear.
  const clear = solveObstacles(solveLattice(cfg(3)), [COLUMN])[0]
  ok(clear.hitCount === 0, 'the default 3 × 5 network does not reach the column')
  ok(clear.nearestPanel !== null, 'so it reports which panel is nearest')
  ok(clear.nearestClearanceCm > 50, `and a real clearance (${clear.nearestClearanceCm.toFixed(1)}cm)`)

  // Grow it and it reaches. THIS is the non-vacuous half: a test that only ever
  // saw "no hits" would pass against a function that always returns none.
  const hit4 = solveObstacles(solveLattice(cfg(4)), [COLUMN])[0]
  ok(hit4.hitCount === 2, `a 4-column network runs 2 panels through it (got ${hit4.hitCount})`)
  ok(hit4.nearestClearanceCm === null, 'and a design already through it quotes no clearance')
  ok(hit4.hits.every((h) => h.depthCm > 0), 'every hit has positive penetration depth')
  ok(hit4.hits[0].depthCm >= hit4.hits[1].depthCm, 'and they are ordered deepest first')

  const hit5 = solveObstacles(solveLattice(cfg(5)), [COLUMN])[0]
  ok(hit5.hitCount > hit4.hitCount, 'a wider network hits it more')

  // Every reported hit really does overlap, checked straight against the SAT
  // rather than trusting this module's own bookkeeping.
  const L = solveLattice(cfg(5))
  const box = obstacleOBB(COLUMN)
  const byId = new Map(L.panels.map((p) => [p.id, p]))
  ok(hit5.hits.every((h) => obbPenetration(box, byId.get(h.id).obb) !== null),
    'each hit is confirmed by obbPenetration directly')
  const hitIds = new Set(hit5.hits.map((h) => h.id))
  ok(L.panels.filter((p) => p.present && !hitIds.has(p.id))
    .every((p) => obbPenetration(box, p.obb) === null),
    'and NO unreported panel overlaps — the list is complete, not just correct')

  // Moving the column away must clear it. If this passed regardless, the box
  // position would not be reaching the test at all.
  const moved = solveObstacles(L, [{ ...COLUMN, xCm: 1200 }])[0]
  ok(moved.hitCount === 0, 'moving the column out of the room clears every hit')
}

// -----------------------------------------------------------------------------
// 3. CLEARANCE IS A DISTANCE, NOT A FLAG
// -----------------------------------------------------------------------------
console.log('3. clearance behaves like a distance')
{
  const L = solveLattice(normalizeConfig({ ...DEFAULT_CONFIG, lattice: { cols: 3, rows: 5, panelType: '2x2' } }))
  const far = solveObstacles(L, [{ ...COLUMN, xCm: 1000 }])[0].nearestClearanceCm
  const near_ = solveObstacles(L, [{ ...COLUMN, xCm: 400 }])[0].nearestClearanceCm
  ok(far > near_, `moving the column closer reduces the clearance (${far.toFixed(0)} → ${near_.toFixed(0)}cm)`)

  // A LOWER BOUND, by construction (see the module header) — so it must never
  // exceed the true centre-to-centre separation, and must be 0 on contact.
  const panel = L.panels.find((p) => p.present)
  ok(obstacleClearanceCm({ ...COLUMN, xCm: panel.position[0] - 25, zCm: panel.position[2] - 25 }, panel) === 0,
    'a column sitting on a panel has zero clearance, not a negative one')

  // Shifting the column by a known amount along a free axis moves the clearance
  // by exactly that amount — the check that it is measuring, not estimating.
  const a = solveObstacles(L, [{ ...COLUMN, xCm: 600, zCm: 285 }])[0].nearestClearanceCm
  const b = solveObstacles(L, [{ ...COLUMN, xCm: 700, zCm: 285 }])[0].nearestClearanceCm
  near(b - a, 100, 1e-6, 'a 100cm shift straight away from the network adds exactly 100cm')
}

// -----------------------------------------------------------------------------
// 4. THE REPORT, AND THE CONFIG CONTRACT
// -----------------------------------------------------------------------------
console.log('4. the report surfaces it, and never enforces it')
{
  const wide = buildReportV4({ ...DEFAULT_CONFIG, lattice: { cols: 4, rows: 5, panelType: '2x2' } })
  const w = wide.warnings.filter((x) => x.code === OBSTACLE_HIT_CODE)
  ok(w.length === 1, 'one warning per fouled obstacle')
  ok(w[0].panels.length === wide.obstacles[0].hitCount, 'naming every panel that runs through it')
  ok(w[0].message.includes('380') && w[0].message.includes('285'),
    'and quoting where the thing actually is')

  // REPORT, DO NOT VETO. The panels are still placed and still counted.
  const plain = buildReportV4({ lattice: { cols: 4, rows: 5, panelType: '2x2' }, obstacles: [] })
  ok(plain.metrics.counts.panels === wide.metrics.counts.panels,
    'a fouled design has exactly the same panels as one with no column at all')
  ok(JSON.stringify(plain.collisions) === JSON.stringify(wide.collisions),
    'and the column changes nothing about the collision pass')

  const quiet = buildReportV4({ ...DEFAULT_CONFIG, lattice: { cols: 3, rows: 5, panelType: '2x2' } })
  ok(quiet.warnings.every((x) => x.code !== OBSTACLE_HIT_CODE), 'a clear design raises no obstacle warning')

  // The config contract, three rules — see schema.js's `mergeObstacles`.
  ok(normalizeConfig({}).obstacles.length === DEFAULT_OBSTACLES.length,
    'an absent obstacles key defaults to the room as measured')
  ok(normalizeConfig({ obstacles: [] }).obstacles.length === 0,
    'while an explicit empty list means there is nothing there — a different statement')

  // A PARTIAL list still gets the whole room. This is the regression that made
  // the heating gap and all five mullions invisible to anyone with a working
  // config saved before they were measured: the list was taken verbatim, so a
  // save that predated an element could withhold it forever. Obstacles are site
  // measurements, not design choices, and a design must not be able to do that.
  const partial = normalizeConfig({ obstacles: [{ id: 'column', xCm: 380, zCm: 285, widthCm: 50, depthCm: 50 }] })
  ok(partial.obstacles.length === DEFAULT_OBSTACLES.length,
    `a save carrying only the column still loads the whole room (got ${partial.obstacles.length})`)
  // The caps are `mullion-N-cap`, so count the mullions themselves explicitly —
  // a bare startsWith('mullion') now matches ten things and would pass by
  // accident if the mullions vanished and only their caps survived.
  const mullionCount = partial.obstacles.filter((o) => /^mullion-\d+$/.test(o.id)).length
  const capCount = partial.obstacles.filter((o) => /^mullion-\d+-cap$/.test(o.id)).length
  ok(partial.obstacles.some((o) => o.id === 'heating') && mullionCount === 5 && capCount === 5,
    `including the heating gap, all five mullions and their caps (${mullionCount} + ${capCount})`)
  ok(['sill', 'sill-sidewalk', 'glass'].every((id) => partial.obstacles.some((o) => o.id === id)),
    '...and the sill, the sidewalk sill and the glass')
  // ...and a tuned entry is not overwritten by the default it merges onto.
  const tuned = normalizeConfig({ obstacles: [{ id: 'column', xCm: 999, zCm: 285, widthCm: 50, depthCm: 50, heightCm: 300 }] })
  ok(tuned.obstacles.find((o) => o.id === 'column').xCm === 999,
    'while an entry the user tuned survives the merge')
  // An id that is not a default is kept rather than silently dropped.
  const extra = normalizeConfig({ obstacles: [{ id: 'duct', xCm: 10, zCm: 10, widthCm: 20, depthCm: 20, heightCm: 50 }] })
  ok(extra.obstacles.some((o) => o.id === 'duct') && extra.obstacles.length === DEFAULT_OBSTACLES.length + 1,
    'and an element added by hand is appended, not dropped')
  const round = normalizeConfig(normalizeConfig({}))
  ok(JSON.stringify(round.obstacles) === JSON.stringify(normalizeConfig({}).obstacles),
    'and normalizing twice changes nothing')

  // Garbage in an obstacle record is clamped, not obeyed, and never throws.
  const junk = normalizeConfig({ obstacles: [{ id: 'a', xCm: 'x', widthCm: -99 }, null, { id: 'a' }] })
  // The room's own elements come through the merge too, so the junk is counted
  // by what it added rather than by the whole length.
  const added = junk.obstacles.filter((o) => !DEFAULT_OBSTACLES.some((d) => d.id === o.id))
  ok(added.length === 2, `non-objects are dropped (got ${added.length} added)`)
  ok(added[0].xCm === 0 && added[0].widthCm === OBSTACLE_SIZE_MIN,
    `and bad numbers are replaced or clamped (widthCm ${added[0].widthCm}, floor ${OBSTACLE_SIZE_MIN})`)
  // The floor has to stay below the thinnest real element or it rewrites it —
  // it already did that to the 0.5cm glass once.
  ok(OBSTACLE_SIZE_MIN <= 0.5, 'and the floor is thin enough not to fatten the glazing')
  ok(junk.obstacles[0].id !== junk.obstacles[1].id, 'a duplicate id is renamed rather than losing a column')
}

// -----------------------------------------------------------------------------
// 5. THE CORNER — the facade turns 90°, and the section must turn with it
// -----------------------------------------------------------------------------
console.log('5. the return facade is the main one rotated about the corner')
{
  const all = normalizeConfig({}).obstacles
  const ext = (id) => obstacleExtents(all.find((o) => o.id === id))
  const CORNER = [-81.3, -59.7]

  // Both facades start AT the corner. The corner post is shared and appears in
  // both lists, so their first mullions overlap in a 6.35 × 6.35 column there.
  const m1 = ext('mullion-1')
  const r1 = ext('mullion-1-return')
  ok(m1.min[0] === CORNER[0] && m1.min[2] === CORNER[1], 'the main facade starts at the corner')
  ok(r1.min[0] === CORNER[0] && r1.min[2] === CORNER[1], 'and so does the return')

  // THE SECTION TURNS WITH THE FACADE. This is the failure the whole shared
  // generator exists to prevent: naming the section's dimensions after world
  // axes lays the return's mullions on their side — 6.35 deep and 19.05 across
  // — which looks entirely plausible in plan and is wrong in section.
  near(m1.max[0] - m1.min[0], 6.35, 1e-9, 'main mullion is 6.35 ACROSS in x')
  near(m1.max[2] - m1.min[2], 19.05, 1e-9, '...and 19.05 DEEP in z')
  near(r1.max[0] - r1.min[0], 19.05, 1e-9, 'the return mullion is 19.05 DEEP in x')
  near(r1.max[2] - r1.min[2], 6.35, 1e-9, '...and 6.35 ACROSS in z — the section turned')

  // Same spacing pattern, measured out from the corner on both.
  const centresOf = (re, axis) => all
    .filter((o) => re.test(o.id))
    .map((o) => { const e = obstacleExtents(o); return (e.min[axis] + e.max[axis]) / 2 })
  const mainC = centresOf(/^mullion-\d+$/, 0)
  const retC = centresOf(/^mullion-\d+-return$/, 2)
  const gaps = (v) => v.slice(1).map((x, i) => +(x - v[i]).toFixed(6))
  ok(mainC.length === 5 && retC.length === 5, 'five mullions on each facade')
  ok(JSON.stringify(gaps(mainC)) === JSON.stringify([129.5, 152.4, 152.4, 152.4]),
    `main spacings are 129.5 then 152.4 × 3 (${gaps(mainC)})`)
  ok(JSON.stringify(gaps(retC)) === JSON.stringify(gaps(mainC)),
    `and the return repeats them exactly from the corner (${gaps(retC)})`)

  // Heights are untouched by the turn — a corner changes plan, not section.
  for (const id of ['mullion-1', 'mullion-1-return']) {
    ok(ext(id).min[1] === -25 && ext(id).max[1] === 375, `${id} spans y −25 → 375`)
  }
  ok(ext('sill-return').max[1] === -25 && ext('sill-sidewalk-return').max[1] === -65,
    'the return sill and sidewalk sill sit at the same levels as the main ones')

  // Glass stays 0.5 thick — across x on the return, across z on the main. The
  // thickness must land on the axis the facade faces, not on a fixed one.
  near(ext('glass').max[2] - ext('glass').min[2], 0.5, 1e-9, 'main glass is 0.5 thick in z')
  near(ext('glass-return').max[0] - ext('glass-return').min[0], 0.5, 1e-9,
    'and the return glass 0.5 thick in x')

  // The caps sit OUTSIDE their own glazing plane, which is a different
  // direction on each facade.
  ok(ext('mullion-1-cap').max[2] <= ext('glass').min[2], 'main caps are outside the main glass')
  ok(ext('mullion-1-return-cap').max[0] <= ext('glass-return').min[0],
    'and the return caps outside the return glass — the "outside" direction turned too')
}

// -----------------------------------------------------------------------------
// 6. THE HEATING RUNS — they tile, and they stop at the floor
// -----------------------------------------------------------------------------
console.log('6. the two heating runs meet at the corner without overlapping')
{
  const all = normalizeConfig({}).obstacles
  const ext = (id) => obstacleExtents(all.find((o) => o.id === id))
  const a = ext('heating')
  const b = ext('heating-return')

  // BOTH STOP AT THE FLOOR. This is what lets the return run reach the wall's
  // room-side face at x = 0: the wall is drawn from y = 0 up, so a trench below
  // the floor and a wall above it never meet. If a heating run ever climbs past
  // y = 0 again it starts intersecting the wall, silently.
  ok(a.max[1] === 0 && b.max[1] === 0, 'both runs top out at the floor, y = 0')
  ok(a.min[1] === -25 && b.min[1] === -25, 'and both bottom on the mullion base at −25')
  ok(b.max[0] === 0, 'the return run reaches the wall room-side face at x = 0')

  // The main run starts at the RETURN mullions' inner face, not at the corner —
  // the return elevation occupies that ground.
  near(a.min[0], ext('mullion-1-return').max[0], 1e-9,
    'the main run starts exactly at the return mullions\' inner face')

  // They TILE: gap 1 covers the corner square across gap 2's whole x range, so
  // gap 2 starting at z = 0 leaves neither an overlap nor a missed strip.
  ok(a.max[2] === 0 && b.min[2] === 0, 'they meet at z = 0')
  ok(a.min[0] <= b.min[0] && a.max[0] >= b.max[0],
    'and gap 1 spans gap 2\'s full x range, so the corner square is covered once')
  ok(obbPenetration(obstacleOBB(all.find((o) => o.id === 'heating')),
    obstacleOBB(all.find((o) => o.id === 'heating-return'))) === null,
    'the two runs do not interpenetrate — checked with the SAT, not by eye')

  // The return run ends where the return glazing does, and is derived from the
  // same spacings — so correcting a mullion spacing moves both together.
  near(b.max[2], ext('glass-return').max[2], 1e-9,
    'the return run ends exactly where the return glazing does')
  ok(b.max[0] - b.min[0] > a.max[2] - a.min[2],
    `and it is slightly wider than the main run (${(b.max[0] - b.min[0]).toFixed(2)} vs ${(a.max[2] - a.min[2]).toFixed(2)})`)
}

console.log(`\ntest-v4-obstacles: ${passed} checks passed, ${failed} failed`)
process.exit(failed ? 1 : 0)
