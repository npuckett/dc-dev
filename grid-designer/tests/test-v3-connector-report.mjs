/**
 * tests/test-v3-connector-report.mjs — the connector section of buildReport.
 *
 * Two things are being guarded. First that the flags MEAN something: every rule
 * is exercised with a case that trips it and a case that does not, because a
 * clean report is worthless if the rules cannot fire. Second that the kit is a
 * real partition — every part accounted for exactly once, and every station
 * genuinely within the bin it was assigned to.
 *
 * The clash rule needs saying out loud: a connector's bounding box ALWAYS
 * overlaps the two panels it grips, because the channel closes around the rim
 * and the slot is a void inside the box. Those two pairs are excluded by
 * construction, so what the rule detects is a part fouling a THIRD panel, or two
 * parts fouling each other.
 */

import { buildReport } from '../src/core/v3/report.js'
import {
  solveConnectors,
  connectorStationFlags,
  connectorOBB,
  CONNECTOR_LIMITS,
} from '../src/core/v3/connectors.js'
import { solveLayout, tileOBB } from '../src/core/v3/placement.js'
import { obbPenetration } from '../src/core/v3/collide.js'
import { DEFAULT_CONFIG, normalizeConfig } from '../src/core/v3/schema.js'
import { buildPreset, PRESET_IDS } from '../src/core/v3/presets.js'

let passed = 0
let failed = 0
const ok = (c, m) => { if (c) passed++; else { failed++; console.log(`  FAIL: ${m}`) } }
const near = (a, b, tol, m) => ok(Math.abs(a - b) <= tol, `${m} (got ${a}, want ${b} ±${tol})`)

const withConnectors = (base, conn) => ({
  ...base,
  connectors: { lengthCm: 10, spacingCm: 50, minPerJoint: 2, binSpanCm: 0.5, binAngleDeg: 5, ...conn },
})

console.log('=== test-v3-connector-report ===')

// -----------------------------------------------------------------------------
// 1. Every preset reports a connector set, and the summary agrees with the rows.
// -----------------------------------------------------------------------------
console.log('1. the summary is consistent with the stations')
{
  for (const id of PRESET_IDS) {
    const R = buildReport(buildPreset(id))
    const c = R.connectors
    ok(c.summary.count === c.stations.length, `${id}: summary count matches the station rows`)
    ok(c.summary.jointCount === c.perJoint.length, `${id}: every joint appears once in perJoint`)
    ok(c.summary.flagged === c.stations.filter((s) => s.flags.length > 0).length,
      `${id}: flagged count matches the rows`)
    // partTypes is now BOTH families: one back half per geometry bin plus the
    // handful of universal front bars.
    ok(c.summary.backHalfTypes === c.kit.length, `${id}: backHalfTypes matches the kit length`)
    ok(c.summary.frontBarTypes === c.bars.length, `${id}: frontBarTypes matches the bar list`)
    ok(c.summary.partTypes === c.kit.length + c.bars.length, `${id}: partTypes is both families`)
    ok(c.bars.length <= c.kit.length, `${id}: there are never more bar types than back-half types`)
    ok(c.summary.clashes === c.clashes.length, `${id}: clash count matches the rows`)

    const worstSpread = Math.max(...c.stations.map((s) => s.spanSpreadCm))
    near(c.summary.worstSpanSpreadCm, worstSpread, 1e-9, `${id}: worstSpanSpreadCm is the max over stations`)
  }
}

// -----------------------------------------------------------------------------
// 2. The kit is a partition, and every station really is inside its bin.
// -----------------------------------------------------------------------------
console.log('2. the kit partitions the stations')
{
  for (const id of PRESET_IDS) {
    for (const [binSpanCm, binAngleDeg] of [[0.1, 1], [0.5, 5], [2, 15]]) {
      const R = buildReport(withConnectors(buildPreset(id), { binSpanCm, binAngleDeg }))
      const c = R.connectors
      const label = `${id} @ ${binSpanCm}cm/${binAngleDeg}°`

      const assigned = c.kit.flatMap((k) => k.stationIds)
      ok(assigned.length === c.stations.length, `${label}: every station is in exactly one part type`)
      ok(new Set(assigned).size === assigned.length, `${label}: no station is in two part types`)
      ok(c.kit.reduce((n, k) => n + k.count, 0) === c.stations.length, `${label}: counts sum to the total`)

      // The representative is the BIN CENTRE, so no member can be further from
      // it than half a bin. This is what makes the forced deviation bounded and
      // independent of which station happened to be seen first.
      const byId = new Map(c.stations.map((s) => [s.id, s]))
      let outOfBin = 0
      for (const part of c.kit) {
        for (const sid of part.stationIds) {
          const st = byId.get(sid)
          const lo = Math.min(st.spanStartCm, st.spanEndCm)
          const hi = Math.max(st.spanStartCm, st.spanEndCm)
          if (Math.abs(lo - part.spanStartCm) > binSpanCm / 2 + 1e-9) outOfBin++
          if (Math.abs(hi - part.spanEndCm) > binSpanCm / 2 + 1e-9) outOfBin++
          if (Math.abs(st.foldDeg - part.foldDeg) > binAngleDeg / 2 + 1e-9) outOfBin++
        }
        if (part.worstSpanErrorCm > binSpanCm / 2 + 1e-9) outOfBin++
        if (part.worstFoldErrorDeg > binAngleDeg / 2 + 1e-9) outOfBin++
      }
      ok(outOfBin === 0, `${label}: no station sits further than half a bin from its part`)

      // Sorted biggest-first, which is how a kit is read.
      let unsorted = 0
      for (let k = 1; k < c.kit.length; k++) if (c.kit[k].count > c.kit[k - 1].count) unsorted++
      ok(unsorted === 0, `${label}: the kit is ordered by count`)
    }
  }
}

// -----------------------------------------------------------------------------
// 3. Loosening the bins can only ever reduce the number of part types.
//
// The whole point of the knob, stated as a monotonicity property rather than as
// a golden table that would go stale.
// -----------------------------------------------------------------------------
console.log('3. the bin knob is monotone and has real range')
{
  const bins = [[0.1, 1], [0.25, 2.5], [0.5, 5], [1, 10], [2, 15]]
  for (const id of PRESET_IDS) {
    const types = bins.map(([binSpanCm, binAngleDeg]) =>
      buildReport(withConnectors(buildPreset(id), { binSpanCm, binAngleDeg })).connectors.summary.partTypes)
    let rising = 0
    for (let k = 1; k < types.length; k++) if (types[k] > types[k - 1]) rising++
    ok(rising === 0, `${id}: part types never increase as the bins loosen (${types.join(' → ')})`)
    ok(types[0] > types[types.length - 1], `${id}: the knob actually does something (${types[0]} → ${types.at(-1)})`)
    console.log(`   ${id.padEnd(8)} ${types.map((t) => String(t).padStart(4)).join('')}   of ` +
      `${buildReport(buildPreset(id)).connectors.summary.count} parts`)
  }
}

// -----------------------------------------------------------------------------
// 4. Every station rule fires, and every station rule can be silent.
// -----------------------------------------------------------------------------
console.log('4. the station rules are non-vacuous')
{
  const stub = (over) => ({
    id: 'T', jointIndex: 0, lengthCm: 10,
    spanStartCm: 3, spanEndCm: 3, spanMinCm: 3, spanMaxCm: 3, spanSpreadCm: 0,
    foldDeg: 0, ...over,
  })

  ok(connectorStationFlags(stub({})).length === 0, 'a benign station raises nothing')

  ok(connectorStationFlags(stub({ spanMinCm: 0.2 })).includes('W_CONNECTOR_PINCH'),
    'PINCH fires below minSpanCm')
  ok(!connectorStationFlags(stub({ spanMinCm: CONNECTOR_LIMITS.minSpanCm })).includes('W_CONNECTOR_PINCH'),
    'PINCH is silent exactly at the limit')

  ok(connectorStationFlags(stub({ spanMaxCm: 12 })).includes('W_CONNECTOR_SPAN'),
    'SPAN fires above maxSpanCm')
  ok(!connectorStationFlags(stub({ spanMaxCm: CONNECTOR_LIMITS.maxSpanCm })).includes('W_CONNECTOR_SPAN'),
    'SPAN is silent exactly at the limit')

  ok(connectorStationFlags(stub({ spanSpreadCm: 2 })).includes('W_CONNECTOR_TWIST'),
    'TWIST fires above maxSpanSpreadCm')
  ok(!connectorStationFlags(stub({ spanSpreadCm: CONNECTOR_LIMITS.maxSpanSpreadCm })).includes('W_CONNECTOR_TWIST'),
    'TWIST is silent exactly at the limit')

  // INFEASIBLE is the one no real design reaches — see connectors.js. Tested
  // against a synthetic station precisely because of that, so the rule cannot
  // rot unnoticed behind a correlation that happens to hold today.
  const tight = stub({ spanStartCm: 0.4, spanEndCm: 0.4, spanMinCm: 0.4, spanMaxCm: 0.4, foldDeg: 40 })
  ok(connectorStationFlags(tight).includes('W_CONNECTOR_INFEASIBLE'),
    'INFEASIBLE fires when a tight gap folds hard enough to cross the hooks')
  ok(!connectorStationFlags(stub({ ...tight, foldDeg: 5 })).includes('W_CONNECTOR_INFEASIBLE'),
    'the same tight gap is feasible when nearly flat')

  // And it stays silent everywhere real, which is the property worth recording.
  let anyInfeasible = 0
  for (const id of PRESET_IDS) anyInfeasible += buildReport(buildPreset(id)).connectors.summary.infeasible
  ok(anyInfeasible === 0, 'no preset contains an infeasible station')
}

// -----------------------------------------------------------------------------
// 5. Clash detection: fires, excludes the gripped panels, and matches the SAT.
// -----------------------------------------------------------------------------
console.log('5. clash detection')
{
  // Crowding a joint with parts drives connectors on DIFFERENT joints into each
  // other near the panel corners. Measured on `modular`, 6 per joint.
  const crowded = buildReport(withConnectors(buildPreset('modular'), { spacingCm: 10 }))
  ok(crowded.connectors.summary.clashes > 0,
    `crowding produces clashes (${crowded.connectors.summary.clashes}) — the rule is non-vacuous`)
  ok(crowded.connectors.clashes.every((c) => c.depthCm > 0.05), 'every reported clash is deeper than the threshold')
  ok(crowded.connectors.stations.filter((s) => s.flags.includes('W_CONNECTOR_CLASH')).length > 0,
    'clashing stations carry the flag')

  // Sorted deepest-first.
  let unsorted = 0
  for (let k = 1; k < crowded.connectors.clashes.length; k++) {
    if (crowded.connectors.clashes[k].depthCm > crowded.connectors.clashes[k - 1].depthCm + 1e-9) unsorted++
  }
  ok(unsorted === 0, 'clashes are ordered deepest first')

  // A part is NEVER reported against a panel it grips — it necessarily overlaps
  // those two, so reporting them would flag every design ever.
  const byStation = new Map(crowded.connectors.stations.map((s) => [s.id, s]))
  let selfReported = 0
  for (const c of crowded.connectors.clashes) {
    if (c.kind !== 'panel') continue
    const st = byStation.get(c.station)
    if (c.against === st.a || c.against === st.b) selfReported++
  }
  ok(selfReported === 0, 'a part is never flagged against a panel it grips')

  // And that exclusion is load-bearing, not defensive: run the SAT with the
  // exclusion removed and it must light up everywhere.
  {
    const cfg = normalizeConfig(buildPreset('modular'))
    const L = solveLayout(cfg)
    const C = solveConnectors(cfg, L)
    const placed = L.tiles.filter((t) => t.position)
    const boxes = placed.map(tileOBB)
    const index = new Map(placed.map((t, i) => [t.id, i]))
    let gripOverlaps = 0
    for (const st of C.stations) {
      const box = connectorOBB(st)
      for (const id of [st.a, st.b]) {
        const pen = obbPenetration(box, boxes[index.get(id)])
        if (pen && pen.depthCm > 0.05) gripOverlaps++
      }
    }
    ok(gripOverlaps === C.stations.length * 2,
      `every part overlaps both panels it grips (${gripOverlaps}) — which is why they are excluded`)
  }
}

// -----------------------------------------------------------------------------
// 6. Single-connector joints are surfaced, and are reachable only deliberately.
// -----------------------------------------------------------------------------
console.log('6. single-connector joints')
{
  // With the power supply modelled, "under-connected" joints are expected — a
  // blocked joint carries none at all. The floor only binds where there is rim
  // to put a part on, so the clean case is the one with the supply ignored.
  const R2 = buildReport(withConnectors(buildPreset('drift'), { minPerJoint: 2, powerEdge: 'none' }))
  ok(R2.connectors.summary.singleConnectorJoints === 0,
    'ignoring the power supply, every joint reaches the two-part floor')
  ok(!R2.connectors.warnings.some((w) => w.code === 'W_JOINT_SINGLE_CONNECTOR'), 'and no warning')

  // In the DEFAULT relief mode the supply costs no parts, so nothing falls below
  // the floor — that is the whole point of the correction. Only the stricter
  // 'block' reading empties joints.
  const RR = buildReport(withConnectors(buildPreset('drift'), { minPerJoint: 2, powerEdge: 'low' }))
  ok(RR.connectors.summary.singleConnectorJoints === 0,
    'relief mode leaves every joint at the floor')
  ok(RR.connectors.summary.bearsOnSupply > 0,
    `but flags the parts that bear on a supply (${RR.connectors.summary.bearsOnSupply})`)
  const RB = buildReport(withConnectors(buildPreset('drift'), { minPerJoint: 2, powerEdge: 'low', supplyMode: 'block' }))
  ok(RB.connectors.summary.singleConnectorJoints > 0,
    `the stricter reading drops joints below the floor (${RB.connectors.summary.singleConnectorJoints})`)

  const R1 = buildReport(withConnectors(buildPreset('drift'), { minPerJoint: 1, spacingCm: 200 }))
  ok(R1.connectors.summary.singleConnectorJoints > 0,
    `asking for one part per joint produces hinges (${R1.connectors.summary.singleConnectorJoints})`)
  ok(R1.connectors.warnings.filter((w) => w.code === 'W_JOINT_SINGLE_CONNECTOR').length ===
    R1.connectors.summary.singleConnectorJoints, 'one warning per hinged joint')
}

// -----------------------------------------------------------------------------
// 7. Determinism, and the rest of the report is untouched.
// -----------------------------------------------------------------------------
console.log('7. determinism and non-interference')
{
  for (const id of PRESET_IDS) {
    const cfg = buildPreset(id)
    ok(JSON.stringify(buildReport(cfg)) === JSON.stringify(buildReport(cfg)), `${id}: byte-identical on re-run`)
  }

  // Changing a connector knob must not move a single panel or joint number —
  // connectors are measured FROM the layout, they never feed back into it.
  const base = buildReport(withConnectors(buildPreset('drift'), {}))
  const other = buildReport(withConnectors(buildPreset('drift'), { lengthCm: 25, spacingCm: 15, minPerJoint: 4 }))
  const strip = (R) => JSON.stringify({ ...R, connectors: null })
  ok(strip(base) === strip(other), 'connector settings do not perturb the joints, fit, collisions or support')
  ok(other.connectors.summary.count > base.connectors.summary.count, 'but they do change the connector set')
}

console.log(`\ntest-v3-connector-report: ${passed} checks passed, ${failed} failed`)
process.exit(failed ? 1 : 0)
