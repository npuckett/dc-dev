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
import { normalizeConfig, DEFAULT_CONFIG, DEFAULT_OBSTACLES, OBSTACLE_SIZE_MIN, stairBands } from '../src/core/v4/schema.js'
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
  // ONE WARNING PER FOULED OBSTACLE — however many that is. It used to be
  // exactly one, back when the column was the only thing in the room; the stair
  // and the relocated column both foul now, so the check is the INVARIANT
  // (one warning each, naming every panel) rather than the count of the day.
  const wide = buildReportV4({ ...DEFAULT_CONFIG, lattice: { cols: 4, rows: 5, panelType: '2x2' } })
  const w = wide.warnings.filter((x) => x.code === OBSTACLE_HIT_CODE)
  const fouled = wide.obstacles.filter((o) => o.hitCount > 0)
  ok(fouled.length > 0, `something is fouled, so the check is not vacuous (${fouled.length})`)
  ok(w.length === fouled.length, `one warning per fouled obstacle (${w.length} vs ${fouled.length})`)
  for (const o of fouled) {
    const mine = w.find((x) => x.message.includes(o.label) || x.obstacle === o.id)
    ok(mine !== undefined, `${o.id} gets a warning of its own`)
  }
  const colW = w.find((x) => x.obstacle === 'column' || x.message.includes('column'))
  const col = wide.obstacles.find((o) => o.id === 'column')
  ok(colW && colW.message.includes(String(col.extents.min[0])),
    'and quoting where the thing actually is')

  // REPORT, DO NOT VETO. The panels are still placed and still counted.
  const plain = buildReportV4({ lattice: { cols: 4, rows: 5, panelType: '2x2' }, obstacles: [] })
  ok(plain.metrics.counts.panels === wide.metrics.counts.panels,
    'a fouled design has exactly the same panels as one with no column at all')
  ok(JSON.stringify(plain.collisions) === JSON.stringify(wide.collisions),
    'and the column changes nothing about the collision pass')

  // A design clear of everything raises nothing. The default no longer is —
  // the stair landing sits on the network's peak — so this uses a lattice small
  // enough to be genuinely clear, which keeps the negative meaningful.
  // A SHALLOW one: at 30° the network peaks at 35.135 and the stair landing's
  // underside is at 35, so every default-angle design fouls it by 1.35mm. That
  // is a real finding about the two heights, not something to test around — so
  // the probe drops the angle until the network is genuinely clear.
  const quiet = buildReportV4({ lattice: { cols: 1, rows: 2, panelType: '2x2' }, angleDeg: 10 })
  ok(quiet.obstacles.every((o) => o.hitCount === 0), 'the probe design really is clear')
  ok(quiet.warnings.every((x) => x.code !== OBSTACLE_HIT_CODE), 'a clear design raises no obstacle warning')

  // The config contract, three rules — see schema.js's `mergeObstacles`.
  ok(normalizeConfig({}).obstacles.length === DEFAULT_OBSTACLES.length,
    'an absent obstacles key defaults to the room as measured')
  ok(normalizeConfig({ obstacles: [] }).obstacles.length === 0,
    'while an explicit empty list means there is nothing there — a different statement')

  // A SAVED LIST CANNOT OVERRIDE THE ROOM. Every element is derived, so a
  // browser that saved before a correction must not keep serving the old
  // geometry — that is exactly what happened, twice, and it looked like the
  // changes had not been applied at all.
  const stale = normalizeConfig({ obstacles: [
    { id: 'column', xCm: 999, zCm: 999, widthCm: 50, depthCm: 50 },
    { id: 'stair-f1-step-16', xCm: 0, zCm: 0, widthCm: 10, depthCm: 10 },
  ] })
  ok(stale.obstacles.length === DEFAULT_OBSTACLES.length,
    `a stale saved list gives exactly the room as measured (${stale.obstacles.length})`)
  ok(stale.obstacles.find((o) => o.id === 'column').xCm !== 999,
    'a saved entry cannot override a derived one')
  ok(!stale.obstacles.some((o) => o.id === 'stair-f1-step-16'),
    'and a deleted element cannot come back as an unknown id')
  ok(stale.obstacles.some((o) => o.id === 'heating') &&
    stale.obstacles.filter((o) => /^mullion-\d+$/.test(o.id)).length === 5,
    'the whole room is present regardless of what was saved')

  // Garbage never throws. Nothing survives to be clamped now that a saved list
  // cannot override the room, so the check is that the room comes through
  // intact and the junk is simply gone.
  const junk = normalizeConfig({ obstacles: [{ id: 'a', xCm: 'x', widthCm: -99 }, null, { id: 'a' }] })
  ok(junk.obstacles.length === DEFAULT_OBSTACLES.length, 'garbage in an obstacle list is discarded')
  ok(junk.obstacles.every((o) => Number.isFinite(o.xCm) && o.widthCm >= OBSTACLE_SIZE_MIN),
    'and every surviving record is well formed')
  ok(OBSTACLE_SIZE_MIN <= 0.5, 'the size floor stays thin enough not to fatten the glazing')
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

// -----------------------------------------------------------------------------
// 7. THE STAIRCASE — a switchback about a well, with the column in it
// -----------------------------------------------------------------------------
console.log('7. the switchback stair and the well it turns about')
{
  const all = normalizeConfig({}).obstacles
  const ext = (id) => obstacleExtents(all.find((o) => o.id === id))
  const M1 = -81.3 + 6.35 / 2
  const M2 = M1 + 129.5
  const M5 = M1 + 129.5 + 152.4 * 3

  // The two measured alignments: the stair spans mullion 2 to mullion 5. That
  // span is now carried by the BAND (the outer edge), with the treads inset by
  // the band width so they abut it rather than run through it.
  const BW = 6 // STAIR_BAND_X_CM
  const bands = stairBands()
  const bandX = (id) => bands.ribbons.find((r) => r.id === id).xCenter
  near(bandX('stair-f2-band') - BW / 2, M2, 1e-9, "the stair's outer band sits on mullion 2")
  near(bandX('stair-f1-band') + BW / 2, M5, 1e-9, "and flight 1's outer band on mullion 5")
  const f2 = ext('stair-f2-step-1')
  const f1 = ext('stair-f1-step-1')
  near(f2.min[0], M2 + BW, 1e-9, "the tread is inset one band width from mullion 2")
  near(f1.max[0], M5 - BW, 1e-9, "and from mullion 5 — it abuts the band")
  near(f1.max[0] - f1.min[0], f2.max[0] - f2.min[0], 1e-9, 'both flights are the same width')

  // THE WELL, and the column standing in it. The column's own position was an
  // estimate; these three constraints — span, equal flights, centred column —
  // pin it 125cm from where that estimate put it, so it is DERIVED from the
  // stair now. If it ever drifts out of the well, this fails.
  const well0 = f2.max[0]
  const well1 = f1.min[0]
  const col = ext('column')
  ok(well1 > well0, `there is a well between the flights (${(well1 - well0).toFixed(1)}cm)`)
  near((well0 + well1) / 2, (col.min[0] + col.max[0]) / 2, 1e-9,
    'and the column is centred in it, in x')
  ok(col.min[0] > well0 && col.max[0] < well1,
    'the column stands clear of both flights — it goes around it without touching')
  near(col.min[2], 271, 1e-9, 'and its measured z face is at 271')

  // The landing reaches further −X than the flights, and that overhang is where
  // the column penetrates it.
  // The landing spans the flights AND the well — nothing more. Its −X edge is
  // flight 2's, not an overhang past it: it is a staircase.
  const land = ext('stair-landing')
  near(land.min[0], f2.min[0], 1e-9, "the landing's −X edge is flight 2's −X edge")
  near(land.max[0], f1.max[0], 1e-9, "and its +X edge is flight 1's")
  const landW = land.max[0] - land.min[0]
  const flightW = (f1.max[0] - f1.min[0]) * 2
  ok(landW > flightW,
    `so it is longer than the two flight widths (${landW.toFixed(1)} vs ${flightW.toFixed(1)})`)
  near(landW - flightW, well1 - well0, 1e-9,
    'and the difference is exactly the well — which is where the column comes through')
  near(land.min[2], 96 + BW, 1e-9, "the landing's near Z edge is inset a band width from 96")

  // THE FLIGHTS WRAP THE COLUMN, NOT THE LANDING. At z 271 the column starts
  // past the landing's back edge at 256, so nothing penetrates the slab — it is
  // the well between the flights that goes around it, in x. Checked with the
  // SAT against every stair piece, so a landing that grew deeper again could
  // not quietly start fouling it.
  const colBox = obstacleOBB(all.find((o) => o.id === 'column'))
  const stairPieces = all.filter((o) => o.id.startsWith('stair'))
  const fouling = stairPieces.filter((o) => obbPenetration(colBox, obstacleOBB(o)) !== null)
  ok(fouling.length === 0,
    `no stair piece touches the column (${fouling.map((o) => o.id).join(', ') || 'none'})`)
  ok(col.min[2] > land.max[2],
    `and the column starts past the landing's back edge (${col.min[2]} vs ${land.max[2]})`)

  // Levels. Flight 1 is fully anchored at BOTH ends — sidewalk at −65, landing
  // at 215 — so the run between them is derived, not chosen, and it comes out on
  // standard proportions exactly: 16 risers of 17.5 and 15 treads of 28. Only
  // one riser count makes both numbers whole, which is the check worth having.
  ok(f1.min[1] === -65, 'flight 1 starts at sidewalk level, y = −65')
  // THE LANDING FOLLOWS THE STAIR, not the other way round. Its height was an
  // estimate (215); the riser is a standard 17.5, so 16 of them off the floor
  // is the exact number and the landing sits on the top tread by construction.
  // Asserting 215 would pin the estimate and let the stair drift off it.
  const f1top = ext('stair-f1-step-15')
  const riser = 17.5
  near(land.max[1], -65 + 15 * riser, 1e-9,
    `the landing is 15 standard risers off the floor (${land.max[1]})`)
  // ALIGNED TO THE LAST RISER, top AND bottom. Its depth is one riser, not a
  // guessed fascia — 40 was an assumption and it hung the slab 22.5cm below the
  // stair it belongs to.
  near(land.min[1], f1top.min[1], 1e-9, "the landing's underside is the last riser's underside")
  near(land.max[1] - land.min[1], riser, 1e-9, 'so the landing is exactly one riser deep')
  near(63 - 2 * riser, 28, 1e-9, '2R + G = 63 gives exactly the 28 tread used')
  // THE LANDING IS FLIGHT 1'S TOP TREAD, and flight 2 rises OFF it — the two
  // flights share that level rather than stacking a redundant tread on it.
  near(ext('stair-f2-step-1').min[1], land.max[1], 1e-9,
    "flight 2's lowest step starts on the landing")
  near(ext('stair-f2-step-1').max[1] - land.max[1], riser, 1e-9,
    '...and rises one standard riser off it')
  // FLUSH, not one riser short: the top tread's surface IS the landing level.
  near(f1top.max[1], land.max[1], 1e-9, "the top tread is flush with the landing")
  near(f1top.max[1] - f1top.min[1], riser, 1e-9, 'and it is one riser tall like the rest')
  near(f1.max[1] - f1.min[1], riser, 1e-9, "step 1 is one standard rise off the lower floor")
  near(f1.min[1], -65, 1e-9, '...starting on the floor itself')
  near(land.max[2] - land.min[2], 160 - BW, 1e-9, 'the landing is 160 deep less the front band inset')
  near(f1.max[2] - f1top.min[2], 15 * 28, 1e-9,
    'and the flight runs 15 treads × 28 = 420')
  // THE FLIGHT MUST NOT INTRUDE INTO THE LANDING. Flush in height, adjacent in
  // plan — the top tread starts exactly where the landing stops.
  near(f1top.min[2], land.max[2], 1e-9,
    'the top tread begins exactly at the landing back edge')
}

console.log(`\ntest-v4-obstacles: ${passed} checks passed, ${failed} failed`)
process.exit(failed ? 1 : 0)
