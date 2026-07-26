/**
 * tests/test-v3-presets.mjs — headless checks for core/v3/presets.js.
 *
 * The presets encode measured claims about buildability, so these checks
 * re-measure those claims rather than restating them. If the core geometry
 * changes, the assertions here are what tell you the preset prose has gone
 * stale — that is the point of pinning real numbers.
 */

import { PRESETS, PRESET_IDS, DEFAULT_PRESET, buildPreset } from '../src/core/v3/presets.js'
import { buildReport } from '../src/core/v3/report.js'
import { validateConfig } from '../src/core/v3/schema.js'

let passed = 0
let failed = 0
const ok = (c, m) => { if (c) passed++; else { failed++; console.log(`  FAIL: ${m}`) } }

console.log('=== test-v3-presets ===')

const measured = {}
for (const id of PRESET_IDS) {
  const cfg = buildPreset(id)
  const R = buildReport(cfg)
  measured[id] = {
    cfg,
    R,
    peak: R.bounds.size[1],
    worst: R.summary.worst,
    flagged: R.summary.flagged,
    joints: R.joints.length,
    collisions: R.collisions.length,
    plates: R.fit.plateCount,
    tiles: R.fit.tileCount,
    edges: R.support.edges,
  }
}

// -----------------------------------------------------------------------------
console.log('1. every preset is valid, deterministic and well-formed')
for (const id of PRESET_IDS) {
  const v = validateConfig(buildPreset(id))
  ok(v.valid, `${id}: validates (${v.errors.map((e) => `${e.code} ${e.path}`).join(', ')})`)
  ok(JSON.stringify(buildPreset(id)) === JSON.stringify(buildPreset(id)), `${id}: deterministic`)
  ok(buildPreset(id).version === 3, `${id}: version 3`)
  ok(buildPreset(id).meta.preset === id, `${id}: records its own id in meta`)
  ok(buildPreset(id).form.footprint.width > 0, `${id}: footprint derived, not left empty`)
}
ok(buildPreset('nope') === null, 'an unknown id returns null rather than throwing')
ok(PRESET_IDS.includes(DEFAULT_PRESET), 'the default preset id exists')
ok(PRESETS.length === PRESET_IDS.length, 'PRESETS and PRESET_IDS agree')
ok(PRESETS.every((p) => p.label && p.trade), 'every preset states a label and its trade-off')

// -----------------------------------------------------------------------------
console.log('2. the plate-pattern rule: at least 4 plates, placed by a rule')
for (const id of PRESET_IDS) {
  ok(measured[id].plates >= 4,
    `${id}: at least 4 plates (${measured[id].plates}) — a design using one or two reads as squares with mistakes in it`)
}

// -----------------------------------------------------------------------------
console.log('3. every preset but `crest` is collision-free')
for (const id of PRESET_IDS) {
  if (id === 'crest') continue
  ok(measured[id].collisions === 0,
    `${id}: no interpenetrating panels (${measured[id].collisions})`)
}
// crest is deliberately NOT closed. If it ever becomes clean, the preset has
// lost its reason to exist and the prose is wrong.
ok(measured.crest.collisions > 0 || measured.crest.worst > 15,
  'crest is still deliberately unbuildable — it exists to show what v2 height costs')

// -----------------------------------------------------------------------------
console.log('4. each preset actually delivers the thing it claims')
{
  // `closed` claims zero flagged joints.
  ok(measured.closed.flagged === 0,
    `closed: not one joint out of tolerance (${measured.closed.flagged}/${measured.closed.joints})`)

  // `shelf` claims the graded edges land. Both edges within ~3cm, most tiles down.
  for (const e of measured.shelf.edges) {
    ok(e.maxClearanceCm < 3.5, `shelf: ${e.edge} edge within 3.5cm of the floor (${e.maxClearanceCm.toFixed(2)})`)
    ok(e.grounded >= e.tiles - 1, `shelf: at most one ${e.edge}-edge tile off the floor (${e.grounded}/${e.tiles})`)
  }

  // `shelf` must ground BETTER than the broad-facet presets — that is its whole
  // justification for using one facet per panel.
  const worstEdge = (id) => Math.max(...measured[id].edges.map((e) => e.maxClearanceCm))
  ok(worstEdge('shelf') < worstEdge('closed'),
    `shelf grounds better than closed (${worstEdge('shelf').toFixed(1)}cm vs ${worstEdge('closed').toFixed(1)}cm)`)
  ok(worstEdge('shelf') < worstEdge('dune'),
    `shelf grounds better than dune (${worstEdge('shelf').toFixed(1)}cm vs ${worstEdge('dune').toFixed(1)}cm)`)

  // `modular` claims exact plate modularity: 2·size + gap == plateLength.
  const m = measured.modular.cfg
  ok(2 * m.cell.size + m.gap === m.cell.plateLength,
    `modular: 2·${m.cell.size} + ${m.gap} = ${m.cell.plateLength} exactly`)
  ok(!measured.modular.R.warnings.some((w) => w.code === 'W_PLATE_LENGTH'),
    'modular: no plate-length mismatch warning')
  // And every other preset trades that away, deliberately.
  for (const id of PRESET_IDS) {
    if (id === 'modular') continue
    const c = measured[id].cfg
    ok(2 * c.cell.size + c.gap !== c.cell.plateLength,
      `${id}: knowingly gives up exact plate modularity (${2 * c.cell.size + c.gap} vs ${c.cell.plateLength})`)
  }

  // `dune` claims to be the tall one among the closed presets.
  const closedIds = PRESET_IDS.filter((id) => id !== 'crest')
  const tallest = closedIds.reduce((a, b) => (measured[a].peak >= measured[b].peak ? a : b))
  ok(tallest === 'dune', `dune is the tallest collision-free preset (got ${tallest})`)
}

// -----------------------------------------------------------------------------
console.log('5. the trade-off is monotone where it claims to be')
{
  // Height costs joint deviation: across the closed presets, the taller ones
  // must not have tighter worst-case joints than the shortest.
  ok(measured.dune.worst > measured.closed.worst,
    `taller costs joint deviation (dune ${measured.dune.worst.toFixed(2)}cm > closed ${measured.closed.worst.toFixed(2)}cm)`)
  ok(measured.crest.worst > measured.dune.worst,
    `crest costs more still (${measured.crest.worst.toFixed(2)}cm > ${measured.dune.worst.toFixed(2)}cm)`)
}

console.log('\n   preset   peak  worst  flagged  coll  plates  worst edge clearance')
for (const id of PRESET_IDS) {
  const m = measured[id]
  const we = Math.max(...m.edges.map((e) => e.maxClearanceCm))
  console.log(`   ${id.padEnd(8)} ${m.peak.toFixed(0).padStart(4)}  ${m.worst.toFixed(2).padStart(5)}   ${String(m.flagged).padStart(2)}/${String(m.joints).padStart(2)}    ${String(m.collisions).padStart(2)}   ${m.plates}/${m.tiles}      ${we.toFixed(1)}cm`)
}

console.log(`\ntest-v3-presets: ${passed} checks passed, ${failed} failed`)
process.exit(failed ? 1 : 0)
