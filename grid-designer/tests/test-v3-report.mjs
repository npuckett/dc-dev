/**
 * tests/test-v3-report.mjs — headless checks for core/v3/report.js.
 *
 * The report is the tool's primary output, so these checks are mostly about it
 * being HONEST: silent when there is nothing to say, non-vacuous when there is,
 * and measuring the same thing an independent recomputation measures.
 */

import { buildReport, COLLISION_MIN_DEPTH_CM } from '../src/core/v3/report.js'
import { solveLayout } from '../src/core/v3/placement.js'
import { DEFAULT_CONFIG, normalizeConfig } from '../src/core/v3/schema.js'

let passed = 0
let failed = 0
const ok = (c, m) => { if (c) passed++; else { failed++; console.log(`  FAIL: ${m}`) } }
const near = (a, b, tol, m) => ok(Math.abs(a - b) <= tol, `${m} (got ${a}, want ${b} ±${tol})`)

const cfgOf = (over = {}) => normalizeConfig({
  ...DEFAULT_CONFIG,
  sheet: { cols: 6, rows: 8 },
  ...over,
  form: { ...DEFAULT_CONFIG.form, toeSharpX: 1, toeSharpZ: 1, ...(over.form ?? {}) },
  placement: { ...DEFAULT_CONFIG.placement, ...(over.placement ?? {}) },
})

console.log('=== test-v3-report ===')

// -----------------------------------------------------------------------------
// 1. A flat sheet has nothing to report. The control for everything else.
// -----------------------------------------------------------------------------
console.log('1. flat sheet reports clean')
{
  const cfg = cfgOf({ form: { amplitude: 0 } })
  const R = buildReport(cfg)
  ok(R.joints.length > 0, `flat sheet still has joints to measure (${R.joints.length})`)
  near(R.summary.worst, 0, 1e-6, 'flat: worst gap deviation is zero')
  near(R.summary.mean, 0, 1e-6, 'flat: mean gap deviation is zero')
  ok(R.summary.flagged === 0, 'flat: no joint flagged')
  ok(R.summary.pinched === 0, 'flat: no joint pinched')
  ok(R.collisions.length === 0, 'flat: no collisions')
  near(R.summary.worstDihedralDeg, 0, 1e-6, 'flat: no fold anywhere')
  near(R.summary.worstSkewDeg, 0, 1e-6, 'flat: no skew anywhere')
  near(R.fit.shapeResidualSigmaCm, 0, 1e-6, 'flat: panels sit exactly on the target')
  ok(R.violations.length === 0, 'flat: no violations')
  // Every joint's gap is the nominal gap.
  ok(R.joints.every((j) => Math.abs(j.gapMin - cfg.gap) < 1e-6), 'flat: every gapMin is nominal')
  ok(R.joints.every((j) => Math.abs(j.gapMax - cfg.gap) < 1e-6), 'flat: every gapMax is nominal')
}

// -----------------------------------------------------------------------------
// 2. Gap deviations agree with an INDEPENDENT recomputation from tile frames.
// -----------------------------------------------------------------------------
console.log('2. joint gaps match an independent measurement')
{
  const cfg = cfgOf({ form: { amplitude: 80 } })
  const L = solveLayout(cfg)
  const R = buildReport(cfg, L)
  const byId = new Map(L.tiles.map((t) => [t.id, t]))
  let worstDisagreement = 0
  for (const j of R.joints) {
    const edge = L.adjacency[j.index]
    const A = byId.get(j.a)
    const B = byId.get(j.b)
    const runAxis = edge.axis
    const sepAxis = runAxis === 'u' ? 'v' : 'u'
    const pt = (T, isA, f) => {
      const sepB = isA ? edge.edge.a : edge.edge.b
      const uc = T.uv.u0 + T.uv.uLen / 2
      const vc = T.uv.v0 + T.uv.vLen / 2
      const sc = sepAxis === 'u' ? uc : vc
      const rc = runAxis === 'u' ? uc : vc
      const eS = sepAxis === 'u' ? T.eu : T.ev
      const eR = runAxis === 'u' ? T.eu : T.ev
      const s = edge.edge.from + f * (edge.edge.to - edge.edge.from)
      return [0, 1, 2].map((k) => T.position[k] + (sepB - sc) * eS[k] + (s - rc) * eR[k])
    }
    let mn = Infinity
    let mx = -Infinity
    for (const f of [0, 0.25, 0.5, 0.75, 1]) {
      const a = pt(A, true, f)
      const b = pt(B, false, f)
      const d = Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])
      mn = Math.min(mn, d)
      mx = Math.max(mx, d)
    }
    worstDisagreement = Math.max(worstDisagreement,
      Math.abs(mn - j.gapMin), Math.abs(mx - j.gapMax))
  }
  ok(worstDisagreement < 1e-6,
    `independently recomputed gaps agree (worst disagreement ${worstDisagreement.toExponential(2)})`)
}

// -----------------------------------------------------------------------------
// 3. Holonomy: in chain mode the tree edges are exact and the cycle edges carry
//    everything. This split is the whole argument for the default mode.
// -----------------------------------------------------------------------------
console.log('3. holonomy split is real in chain mode and absent in surface-fit')
{
  for (const tree of ['bfs-corner', 'comb-v']) {
    const R = buildReport(cfgOf({ form: { amplitude: 80 }, placement: { mode: 'chain', tree } }))
    ok(R.holonomy.mode === 'chain', `${tree}: holonomy reported for chain mode`)
    ok(R.holonomy.treeEdges.count > 0 && R.holonomy.cycleEdges.count > 0,
      `${tree}: both edge classes present`)
    ok(R.holonomy.treeEdges.worst < 1e-6,
      `${tree}: tree edges exact (${R.holonomy.treeEdges.worst.toExponential(2)}cm)`)
    ok(R.holonomy.cycleEdges.worst > 1,
      `${tree}: cycle edges carry the closure error (${R.holonomy.cycleEdges.worst.toFixed(2)}cm)`)
    ok(R.holonomy.worstJoint && R.holonomy.worstJoint.treeEdge === false,
      `${tree}: the worst joint named is a cycle edge`)
    console.log(`   ${tree}: tree worst ${R.holonomy.treeEdges.worst.toExponential(1)}cm, cycle worst ${R.holonomy.cycleEdges.worst.toFixed(2)}cm`)
  }
  // surface-fit has no tree, so the split must be null rather than a fake zero.
  const S = buildReport(cfgOf({ form: { amplitude: 80 } }))
  ok(S.holonomy.mode === 'surface-fit', 'surface-fit named in the holonomy block')
  ok(S.holonomy.treeEdges === null && S.holonomy.cycleEdges === null,
    'surface-fit reports no tree/cycle split rather than a misleading zero')
}

// -----------------------------------------------------------------------------
// 4. surface-fit tracks the target; chain does not. Same claim as the placement
//    suite, but measured through the report's own numbers.
// -----------------------------------------------------------------------------
console.log('4. fit block distinguishes the two modes')
{
  const S = buildReport(cfgOf({ form: { amplitude: 100 } }))
  const C = buildReport(cfgOf({ form: { amplitude: 100 }, placement: { mode: 'chain' } }))
  ok(S.fit.shapeResidualSigmaCm < C.fit.shapeResidualSigmaCm,
    `surface-fit residual σ ${S.fit.shapeResidualSigmaCm.toFixed(2)}cm < chain ${C.fit.shapeResidualSigmaCm.toFixed(2)}cm`)
  ok(S.fit.tileCount > 0 && S.fit.facetCount > 0, 'fit block carries tile and facet counts')
  ok(S.fit.worstPlateSagittaCm <= S.fit.plateFitToleranceCm + 1e-9,
    `no plate exceeds its fit tolerance (${S.fit.worstPlateSagittaCm} <= ${S.fit.plateFitToleranceCm})`)
}

// -----------------------------------------------------------------------------
// 5. Collisions: silent when flat, non-vacuous when steep, ordered by depth.
// -----------------------------------------------------------------------------
console.log('5. collisions')
{
  ok(buildReport(cfgOf({ form: { amplitude: 0 } })).collisions.length === 0,
    'flat sheet: no collisions')
  const steep = buildReport(cfgOf({ form: { amplitude: 160, angularity: 0 } }))
  ok(steep.collisions.length > 0,
    `steep smooth drift does collide, so the check is not vacuous (${steep.collisions.length})`)
  ok(steep.collisions.every((c) => c.depthCm >= COLLISION_MIN_DEPTH_CM),
    'every reported collision is at least the minimum depth')
  const depths = steep.collisions.map((c) => c.depthCm)
  ok(depths.every((d, i) => i === 0 || depths[i - 1] >= d), 'collisions ordered deepest first')
  // Faceting is the mitigation; assert it actually mitigates.
  const faceted = buildReport(cfgOf({ form: { amplitude: 160, angularity: 1, facetCells: 3 } }))
  ok(faceted.collisions.length < steep.collisions.length,
    `faceting reduces collisions (${steep.collisions.length} → ${faceted.collisions.length})`)
  console.log(`   amplitude 160: ${steep.collisions.length} pairs smooth, ${faceted.collisions.length} faceted`)
}

// -----------------------------------------------------------------------------
// 6. Flags fire against the configured tolerance, both ways.
// -----------------------------------------------------------------------------
console.log('6. tolerance flags')
{
  const tight = buildReport(cfgOf({ form: { amplitude: 80 }, gapTolerance: 0.2 }))
  const loose = buildReport(cfgOf({ form: { amplitude: 80 }, gapTolerance: 100 }))
  ok(tight.summary.flagged > loose.summary.flagged,
    `a tighter tolerance flags more joints (${tight.summary.flagged} vs ${loose.summary.flagged})`)
  ok(loose.summary.flagged === 0, 'an absurdly loose tolerance flags nothing')
  ok(tight.joints.filter((j) => j.flags.includes('W_GAP_OUT_OF_TOLERANCE')).length === tight.summary.flagged,
    'flagged count matches the per-joint flags')
}

// -----------------------------------------------------------------------------
// 7. Determinism and structural sanity.
// -----------------------------------------------------------------------------
console.log('7. determinism and structure')
{
  for (const mode of ['surface-fit', 'chain']) {
    for (const amp of [0, 60, 140]) {
      const cfg = cfgOf({ form: { amplitude: amp }, placement: { mode } })
      ok(JSON.stringify(buildReport(cfg)) === JSON.stringify(buildReport(cfg)),
        `${mode} amp ${amp}: repeat reports byte-identical`)
    }
  }
  const R = buildReport(cfgOf({ form: { amplitude: 90 } }))
  ok(R.joints.every((j) => j.gapMin <= j.gapMax + 1e-9), 'gapMin never exceeds gapMax')
  ok(R.joints.every((j) => j.dihedralDeg >= 0 && j.dihedralDeg <= 180), 'dihedral within [0,180]')
  ok(R.joints.every((j) => j.skewDeg >= 0 && j.skewDeg <= 180), 'skew within [0,180]')
  ok(R.joints.every((j) => Number.isFinite(j.deviationCm)), 'no non-finite deviation')
  ok(R.joints.every((j) => typeof j.treeEdge === 'boolean'), 'every joint labelled tree/non-tree')
  ok(R.bounds && R.support && Array.isArray(R.warnings), 'report carries bounds, support and warnings')
  // buildReport must accept a caller-supplied layout and not re-solve it.
  const L = solveLayout(cfgOf({ form: { amplitude: 90 } }))
  ok(JSON.stringify(buildReport(cfgOf({ form: { amplitude: 90 } }), L)) === JSON.stringify(R),
    'passing a layout in gives the same report as solving internally')
}

console.log(`\ntest-v3-report: ${passed} checks passed, ${failed} failed`)
process.exit(failed ? 1 : 0)
