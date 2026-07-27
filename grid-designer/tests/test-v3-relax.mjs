/**
 * tests/test-v3-relax.mjs — the relaxation post-pass.
 *
 * The thing being guarded is not "does it converge" — it is that the relaxation
 * stays HONEST. Three properties matter more than the numbers:
 *
 *   1. It is OFF unless asked, and its existence changes nothing about
 *      `surface-fit` or `chain`. Every other suite describes the same code path
 *      it always did.
 *   2. It moves the PLACEMENTS and nothing else. The form and the tiling come
 *      out identical, because moving those would quietly redesign the drift the
 *      user authored.
 *   3. IT CAN FAIL, AND SAYS SO. A relaxation that always succeeds has stopped
 *      being a measurement. §5 asserts there is a preset it cannot fully solve
 *      and that it reports exactly which joints — if that ever passes vacuously,
 *      the tool has started lying about buildability.
 */

import * as THREE from 'three'
import { solveLayout } from '../src/core/v3/placement.js'
import { relaxLayout, jointEnvelope, RESOLVED_TOLERANCE_CM } from '../src/core/v3/relax.js'
import { buildReport } from '../src/core/v3/report.js'
import { normalizeConfig, DEFAULT_RELAX } from '../src/core/v3/schema.js'
import { buildPreset, PRESET_IDS } from '../src/core/v3/presets.js'
import { fastenerGapNeededCm, foldLimitDeg } from '../src/core/v3/connectors.js'

let passed = 0
let failed = 0
const ok = (c, m) => { if (c) passed++; else { failed++; console.log(`  FAIL: ${m}`) } }
const near = (a, b, tol, m) => ok(Math.abs(a - b) <= tol, `${m} (got ${a}, want ${b} ±${tol})`)

const withRelax = (id, relax) => {
  const base = buildPreset(id)
  return { ...base, placement: { ...base.placement, relax: { ...DEFAULT_RELAX, ...relax } } }
}

console.log('=== test-v3-relax ===')

// -----------------------------------------------------------------------------
// 1. Off by default, and its existence changes nothing.
// -----------------------------------------------------------------------------
console.log('1. off unless asked')
{
  ok(DEFAULT_RELAX.enabled === false, 'relax is off in the shipped defaults')
  for (const id of PRESET_IDS) {
    const plain = solveLayout(buildPreset(id))
    ok(plain.relax === undefined, `${id}: a normal solve carries no relax report`)
  }
  // Turning it on and off again returns the original layout exactly.
  const off = JSON.stringify(solveLayout(withRelax('drift', { enabled: false })))
  ok(off === JSON.stringify(solveLayout(buildPreset('drift'))),
    'an explicitly-disabled relax is byte-identical to no relax at all')
}

// -----------------------------------------------------------------------------
// 2. Deterministic, and it does not mutate what it is given.
// -----------------------------------------------------------------------------
console.log('2. deterministic and non-mutating')
{
  for (const id of PRESET_IDS) {
    const cfg = withRelax(id, { enabled: true })
    ok(JSON.stringify(solveLayout(cfg)) === JSON.stringify(solveLayout(cfg)),
      `${id}: byte-identical on re-solve`)
  }
  // relaxLayout takes a layout and must leave it untouched.
  const cfg = normalizeConfig(withRelax('modular', { enabled: true }))
  const base = solveLayout(buildPreset('modular'))
  const before = JSON.stringify(base)
  relaxLayout(base, cfg)
  ok(JSON.stringify(base) === before, 'relaxLayout does not mutate the layout it is handed')
}

// -----------------------------------------------------------------------------
// 3. It moves the placements and NOTHING else.
// -----------------------------------------------------------------------------
console.log('3. placements only — never the form or the tiling')
{
  for (const id of ['modular', 'crest']) {
    const plain = solveLayout(buildPreset(id))
    const relaxed = solveLayout(withRelax(id, { enabled: true }))

    const strip = (L) => JSON.stringify(L.tiles.map((t) => ({ id: t.id, type: t.type, axis: t.axis, cells: t.cells, uv: t.uv })))
    ok(strip(plain) === strip(relaxed), `${id}: the tiling is untouched — same tiles, types and cells`)
    ok(plain.pattern === relaxed.pattern, `${id}: the tiling pattern is untouched`)
    ok(JSON.stringify(plain.adjacency) === JSON.stringify(relaxed.adjacency),
      `${id}: adjacency is material and cannot move`)

    // The FORM is config, and the relaxation never sees a draft of it.
    const cfg = withRelax(id, { enabled: true })
    ok(JSON.stringify(cfg.form) === JSON.stringify(buildPreset(id).form),
      `${id}: the authored form is unchanged`)

    // But the placements DO move.
    const moved = plain.tiles.some((t, k) =>
      t.position && JSON.stringify(t.position) !== JSON.stringify(relaxed.tiles[k].position))
    ok(moved, `${id}: tiles actually move`)
  }
}

// -----------------------------------------------------------------------------
// 4. It reduces violations, and the geometry survives.
// -----------------------------------------------------------------------------
console.log('4. it reduces violations without breaking the geometry')
{
  const env = jointEnvelope()
  const countOutside = (L) => {
    let n = 0
    let worst = 0
    const byId = new Map(L.tiles.map((t) => [t.id, t]))
    for (const edge of L.adjacency) {
      const A = byId.get(edge.a)
      const B = byId.get(edge.b)
      if (!A?.position || !B?.position) continue
      const s = (edge.edge.from + edge.edge.to) / 2
      const pa = new THREE.Vector3(...A.position)
      const pb = new THREE.Vector3(...B.position)
      void pa; void pb
      const gap = gapOf(L, edge, A, B, s)
      if (gap < env.minGapCm - RESOLVED_TOLERANCE_CM) { n++; worst = Math.max(worst, env.minGapCm - gap) }
    }
    return { n, worst }
  }

  for (const id of ['modular', 'crest']) {
    const plain = countOutside(solveLayout(buildPreset(id)))
    const relaxed = solveLayout(withRelax(id, { enabled: true }))
    const after = countOutside(relaxed)
    ok(after.n <= plain.n, `${id}: no more joints outside the envelope than before (${plain.n} → ${after.n})`)
    ok(after.worst <= plain.worst + 1e-9,
      `${id}: the worst deficit does not grow (${(plain.worst * 10).toFixed(2)}mm → ${(after.worst * 10).toFixed(2)}mm)`)

    // Frames must survive the lerping: repeated blending drifts them off
    // orthonormal, and a skewed frame corrupts every downstream measurement.
    let badFrame = 0
    for (const t of relaxed.tiles) {
      if (!t.position) continue
      const eu = new THREE.Vector3(...t.eu)
      const ev = new THREE.Vector3(...t.ev)
      const n = new THREE.Vector3(...t.normal)
      if (Math.abs(eu.length() - 1) > 1e-6 || Math.abs(ev.length() - 1) > 1e-6 || Math.abs(n.length() - 1) > 1e-6) badFrame++
      if (Math.abs(eu.dot(n)) > 1e-6 || Math.abs(ev.dot(n)) > 1e-6 || Math.abs(eu.dot(ev)) > 1e-6) badFrame++
      if (Math.abs(new THREE.Vector3().crossVectors(ev, eu).dot(n) - 1) > 1e-5) badFrame++
    }
    ok(badFrame === 0, `${id}: every frame is still orthonormal and right-handed after relaxing`)

    // And it still rests on the floor.
    const minY = Math.min(...relaxed.tiles.filter((t) => t.position).map((t) => t.minY))
    near(minY, 0, 1e-6, `${id}: the assembly is settled back onto the floor`)

    // The report still builds on it.
    const R = buildReport(withRelax(id, { enabled: true }), relaxed)
    ok(R.joints.length === relaxed.adjacency.length, `${id}: the report still measures every joint`)
    ok(Number.isFinite(R.fit.shapeResidualSigmaCm), `${id}: shape residual is finite`)
  }
}

/** Gap at a joint's midpoint, recomputed independently of relax.js. */
function gapOf(L, edge, A, B, s) {
  const uc = (t) => t.uv.u0 + t.uv.uLen / 2
  const vc = (t) => t.uv.v0 + t.uv.vLen / 2
  const pt = (tile, isA) => {
    const runAxis = edge.axis
    const sepAxis = runAxis === 'u' ? 'v' : 'u'
    const sepB = isA ? edge.edge.a : edge.edge.b
    const sepC = sepAxis === 'u' ? uc(tile) : vc(tile)
    const runC = runAxis === 'u' ? uc(tile) : vc(tile)
    const eSep = sepAxis === 'u' ? tile.eu : tile.ev
    const eRun = runAxis === 'u' ? tile.eu : tile.ev
    return new THREE.Vector3(
      tile.position[0] + (sepB - sepC) * eSep[0] + (s - runC) * eRun[0],
      tile.position[1] + (sepB - sepC) * eSep[1] + (s - runC) * eRun[1],
      tile.position[2] + (sepB - sepC) * eSep[2] + (s - runC) * eRun[2],
    )
  }
  return pt(A, true).distanceTo(pt(B, false))
}

// -----------------------------------------------------------------------------
// 5. IT CAN FAIL, AND IT SAYS SO. The property that keeps it a measurement.
// -----------------------------------------------------------------------------
console.log('5. it reports what it could not fix')
{
  // A relaxation given almost no room to work MUST leave joints outside and say
  // which. Using `iterations: 1` rather than a marginal design keeps this
  // property stable: it tests the REPORTING, not a particular preset's luck.
  const starved = solveLayout(withRelax('modular', { enabled: true, iterations: 1 })).relax
  ok(starved.unresolved.length > 0,
    `with one iteration there are joints it cannot reach (${starved.unresolved.length}) — ` +
    'a relaxation that always succeeds has stopped being a measurement')
  ok(starved.resolvedCount + starved.unresolved.length === starved.jointCount,
    'resolved and unresolved account for every joint')
  ok(starved.unresolved.every((u) => u.gapShortCm > 0 || u.foldOverDeg > 0),
    'every unresolved joint names what it is still short by')
  ok(starved.unresolved.every((u) => typeof u.a === 'string' && typeof u.b === 'string'),
    'and which two tiles it is between, so it can be found in the model')

  // Given room, it does the job — and reports the price.
  const full = solveLayout(withRelax('modular', { enabled: true })).relax
  ok(full.unresolved.length < starved.unresolved.length,
    `and at the shipped settings it resolves them (${starved.unresolved.length} → ${full.unresolved.length})`)
  ok(full.worstDisplacementCm > 0, `it reports how far it moved things (${full.worstDisplacementCm}cm)`)
  ok(full.meanDisplacementCm <= full.worstDisplacementCm, 'mean displacement never exceeds the worst')
  ok(full.moved.length > 0 && full.moved[0].displacementCm === full.worstDisplacementCm,
    'the moved list is worst-first')

  // A preset already inside the envelope must need no movement at all.
  const easy = solveLayout(withRelax('closed', { enabled: true })).relax
  ok(easy.unresolved.length === 0, 'closed is already inside the envelope')
  near(easy.worstDisplacementCm, 0, 1e-6, 'and is therefore not moved at all')
  console.log(`   modular: ${starved.unresolved.length} unresolved starved, ` +
    `${full.unresolved.length} at the shipped settings, for ${full.worstDisplacementCm}cm of movement`)
}

// -----------------------------------------------------------------------------
// 6. The envelope is the CONNECTOR's, not a restatement of it.
// -----------------------------------------------------------------------------
console.log('6. the envelope comes from the connector')
{
  const env = jointEnvelope()
  near(env.minGapCm, fastenerGapNeededCm(), 1e-12, 'the minimum gap is the fastener\'s own requirement')
  for (const gap of [0.5, 1, 2, 4]) {
    near(env.foldLimitDeg(gap), foldLimitDeg(gap), 1e-12, `the fold limit at ${gap}cm is the panel's own`)
  }
  // Widen the fastener and the envelope must follow, or it has been restated.
  const wide = jointEnvelope({ bolt: { headCm: 1.2, insertOdCm: 0.4, shankCm: 0.3, insertLenCm: 0.57 } })
  ok(wide.minGapCm > env.minGapCm, 'a bigger bolt demands a bigger gap, through the same function')
}

// -----------------------------------------------------------------------------
// 7. Knobs behave monotonically.
// -----------------------------------------------------------------------------
console.log('7. the knobs do what they say')
{
  const at = (relax) => solveLayout(withRelax('modular', { enabled: true, ...relax })).relax
  const stiff = at({ stiffness: 0.9, targetWeight: 0.05 })
  const slack = at({ stiffness: 0.1, targetWeight: 0.05 })
  ok(stiff.unresolved.length <= slack.unresolved.length,
    `more stiffness resolves at least as many joints (${slack.unresolved.length} → ${stiff.unresolved.length})`)

  const held = at({ targetWeight: 0.6 })
  const free = at({ targetWeight: 0.02 })
  ok(held.worstDisplacementCm <= free.worstDisplacementCm + 1e-9,
    `holding tiles harder moves them less (${free.worstDisplacementCm} → ${held.worstDisplacementCm})`)
  ok(free.unresolved.length <= held.unresolved.length,
    'and letting them go resolves at least as many')

  console.log(`   modular: ${slack.unresolved.length} unresolved at low stiffness, ` +
    `${stiff.unresolved.length} at high; worst move ${free.worstDisplacementCm}cm free vs ` +
    `${held.worstDisplacementCm}cm held`)
}

console.log(`\ntest-v3-relax: ${passed} checks passed, ${failed} failed`)
process.exit(failed ? 1 : 0)
