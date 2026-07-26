/**
 * tests/test-v3-placement.mjs — headless checks for core/v3/placement.js.
 *
 * Conventions inherited from the v2 suite and worth keeping:
 *   - closed-form expectations DERIVED HERE from the constants, never golden
 *     numbers. Sign and frame conventions are the classic bug source in this
 *     project and only a derivation catches them.
 *   - determinism asserted by JSON.stringify equality across repeat solves.
 */

import {
  solveLayout,
  layoutBounds,
  panelSolidCorners,
  tileOBB,
  TOE_FLAT_DEG,
} from '../src/core/v3/placement.js'
import { DEFAULT_CONFIG, normalizeConfig } from '../src/core/v3/schema.js'
import { normalizeForm, driftHeight } from '../src/core/v3/form.js'
import { findCollisions } from '../src/core/v3/collide.js'
import { PANEL_PROFILE } from '../src/config.js'

let passed = 0
let failed = 0
const ok = (cond, msg) => {
  if (cond) passed++
  else { failed++; console.log(`  FAIL: ${msg}`) }
}
const near = (a, b, tol, msg) => ok(Math.abs(a - b) <= tol, `${msg} (got ${a}, want ${b} ±${tol})`)

const cfgOf = (over = {}) => normalizeConfig({
  ...DEFAULT_CONFIG,
  sheet: { cols: 6, rows: 8 },
  ...over,
  form: { ...DEFAULT_CONFIG.form, ...(over.form ?? {}) },
  placement: { ...DEFAULT_CONFIG.placement, ...(over.placement ?? {}) },
})

const minCell = (t, k) => Math.min(...t.cells.map((c) => c[k]))

/**
 * Realised edge-to-edge separation on an adjacency record, sampled along the
 * shared material edge. Recomputed here from the tile frames rather than read
 * out of placement.js, so this is an independent measurement.
 */
function edgeGapDeviation(layout, edge, gap) {
  const byId = new Map(layout.tiles.map((t) => [t.id, t]))
  const A = byId.get(edge.a)
  const B = byId.get(edge.b)
  if (!A?.position || !B?.position) return null
  const runAxis = edge.axis
  const sepAxis = runAxis === 'u' ? 'v' : 'u'
  const seg = (T) => {
    const isA = edge.a === T.id
    const sepB = isA ? edge.edge.a : edge.edge.b
    const uc = T.uv.u0 + T.uv.uLen / 2
    const vc = T.uv.v0 + T.uv.vLen / 2
    const sepC = sepAxis === 'u' ? uc : vc
    const runC = runAxis === 'u' ? uc : vc
    const eSep = sepAxis === 'u' ? T.eu : T.ev
    const eRun = runAxis === 'u' ? T.eu : T.ev
    return [0, 0.5, 1].map((f) => {
      const s = edge.edge.from + f * (edge.edge.to - edge.edge.from)
      return [0, 1, 2].map((k) => T.position[k] + (sepB - sepC) * eSep[k] + (s - runC) * eRun[k])
    })
  }
  const pa = seg(A)
  const pb = seg(B)
  return Math.max(...pa.map((p, k) =>
    Math.abs(Math.hypot(p[0] - pb[k][0], p[1] - pb[k][1], p[2] - pb[k][2]) - gap)))
}

const allGapDevs = (L, cfg) =>
  L.adjacency.map((e) => edgeGapDeviation(L, e, cfg.gap)).filter((v) => v !== null)

console.log('=== test-v3-placement ===')

// -----------------------------------------------------------------------------
// 1. The flat form: the one case whose answer is known exactly.
// -----------------------------------------------------------------------------
console.log('1. flat form reproduces an exact flat grid')
for (const mode of ['surface-fit', 'chain']) {
  const cfg = cfgOf({ form: { amplitude: 0 }, placement: { mode } })
  const L = solveLayout(cfg)
  const pitch = cfg.cell.size + cfg.gap

  // Closed form: the sheet spans cols cells plus the gaps between them.
  const expW = cfg.sheet.cols * pitch - cfg.gap
  const expD = cfg.sheet.rows * pitch - cfg.gap
  near(L.bounds.size[0], expW, 1e-6, `${mode}: flat width = cols·pitch − gap`)
  near(L.bounds.size[2], expD, 1e-6, `${mode}: flat depth = rows·pitch − gap`)
  near(L.bounds.size[1], PANEL_PROFILE.overallThickness, 1e-6, `${mode}: flat height = housing thickness`)

  ok(L.tiles.every((t) => Math.abs(t.normal[1] - 1) < 1e-9), `${mode}: every normal is +Y on a flat form`)
  const ys = new Set(L.tiles.map((t) => t.position[1]))
  ok(ys.size === 1, `${mode}: all lit faces coplanar on a flat form (got ${ys.size} heights)`)
  near([...ys][0], PANEL_PROFILE.overallThickness, 1e-6,
    `${mode}: flat lit face sits one housing thickness up (v2's SHORE_Y)`)
  ok(L.tiles.every((t) => t.grounded), `${mode}: every tile grounded on a flat form`)

  const devs = allGapDevs(L, cfg)
  near(Math.max(...devs), 0, 1e-6, `${mode}: every joint is exactly gap on a flat form`)
  ok(L.violations.length === 0, `${mode}: flat form has no violations`)
}

// -----------------------------------------------------------------------------
// 2. Determinism.
// -----------------------------------------------------------------------------
console.log('2. determinism')
for (const mode of ['surface-fit', 'chain']) {
  for (const amp of [0, 60, 120]) {
    const cfg = cfgOf({ form: { amplitude: amp }, placement: { mode } })
    ok(JSON.stringify(solveLayout(cfg)) === JSON.stringify(solveLayout(cfg)),
      `${mode} amp ${amp}: repeat solves byte-identical`)
  }
}

// -----------------------------------------------------------------------------
// 3. chain mode: tree edges are EXACT by construction. This is the defining
//    property of the spanning-tree walk — if it fails the algorithm is wrong.
// -----------------------------------------------------------------------------
console.log('3. chain mode — tree edges exact, error confined to cycle edges')
for (const tree of ['bfs-corner', 'comb-v', 'comb-u']) {
  for (const amp of [40, 120]) {
    const cfg = cfgOf({ form: { amplitude: amp }, placement: { mode: 'chain', tree } })
    const L = solveLayout(cfg)
    const treeSet = new Set(L.tree.treeEdges)
    let worstTree = 0
    let worstNon = 0
    L.adjacency.forEach((e, idx) => {
      const d = edgeGapDeviation(L, e, cfg.gap)
      if (d === null) return
      if (treeSet.has(idx)) worstTree = Math.max(worstTree, d)
      else worstNon = Math.max(worstNon, d)
    })
    ok(worstTree < 1e-6, `${tree} amp ${amp}: tree-edge gap exact (worst dev ${worstTree.toExponential(2)}cm)`)
    if (amp > 0) {
      ok(worstNon > worstTree,
        `${tree} amp ${amp}: cycle-closing edges carry the error, not the tree edges`)
    }
  }
}

// -----------------------------------------------------------------------------
// 4. Holonomy is real geometry: it vanishes with curvature and grows with it.
//    A developable surface can be tiled exactly; a doubly-curved one cannot.
// -----------------------------------------------------------------------------
console.log('4. error vanishes as the surface flattens')
for (const mode of ['surface-fit', 'chain']) {
  let prev = -1
  const curve = []
  for (const amp of [0, 5, 20, 60, 120]) {
    const cfg = cfgOf({ form: { amplitude: amp }, placement: { mode } })
    const worst = Math.max(...allGapDevs(solveLayout(cfg), cfg), 0)
    curve.push(`${amp}:${worst.toFixed(2)}`)
    ok(worst >= prev - 1e-9, `${mode}: joint error non-decreasing with amplitude at amp ${amp}`)
    prev = worst
  }
  const flatWorst = Math.max(...allGapDevs(solveLayout(cfgOf({ form: { amplitude: 0 }, placement: { mode } })), cfgOf()))
  near(flatWorst, 0, 1e-6, `${mode}: zero curvature ⇒ zero joint error`)
  console.log(`   ${mode} worst joint deviation by amplitude: ${curve.join('  ')}`)
}

// -----------------------------------------------------------------------------
// 5. The brief: the sheet's wall+window corner is pinned to the origin and the
//    grounded edges are the closest thing to the floor.
// -----------------------------------------------------------------------------
console.log('5. brief — corner anchored, graded edges nearest the floor, not flat')
{
  const cfg = cfgOf({ form: { toeSharpX: 1, toeSharpZ: 1 } })
  const L = solveLayout(cfg)
  const lowest = Math.min(...L.tiles.flatMap((t) => t.corners.map((c) => c[1])))
  near(lowest, 0, 1e-6, 'the assembly rests on the floor (lowest solid point at y=0)')

  ok(L.support.edges.length === 2, 'both graded edges are reported')
  for (const e of L.support.edges) {
    ok(e.tiles > 0, `${e.edge} edge has tiles`)
    ok(e.maxClearanceCm >= 0, `${e.edge} edge clearance is non-negative`)
  }
  // The corner tile is grounded and genuinely pitched — "grounded but not flat".
  const corner = L.tiles.find((t) => minCell(t, 0) === 0 && minCell(t, 1) === 0)
  ok(corner.grounded, 'the wall+window corner tile is grounded')
  ok(corner.tiltDeg > TOE_FLAT_DEG,
    `the corner tile is pitched, not flat (${corner.tiltDeg.toFixed(1)}° > ${TOE_FLAT_DEG}°)`)

  // A flat form SHOULD trip the flat-toe warning — that is the control proving
  // W_TOE_FLAT is not vacuous.
  const flat = solveLayout(cfgOf({ form: { amplitude: 0 } }))
  ok(flat.warnings.some((w) => w.code === 'W_TOE_FLAT'), 'a flat form trips W_TOE_FLAT')
  ok(!L.warnings.some((w) => w.code === 'W_TOE_FLAT'), 'a real drift does not trip W_TOE_FLAT')
}

// -----------------------------------------------------------------------------
// 6. surface-fit tracks the authored drift; chain does not. This is the whole
//    reason surface-fit is the default.
// -----------------------------------------------------------------------------
console.log('6. surface-fit follows the authored form more closely than chain')
{
  const base = { form: { amplitude: 120, toeSharpX: 1, toeSharpZ: 1 } }
  const cfgS = cfgOf({ ...base, placement: { mode: 'surface-fit' } })
  const cfgC = cfgOf({ ...base, placement: { mode: 'chain' } })
  const form = normalizeForm(cfgS.form)
  /**
   * SHAPE fidelity, not absolute height. Measure the tile's UNDERSIDE (the lit
   * face pushed back one housing thickness along −n̂, which is the part that is
   * meant to sit on the surface) and report the spread of its residual, not the
   * mean. Both layouts are settled onto the floor by a global rigid lift, so a
   * mean-based metric mostly measures that lift and can rank a layout that has
   * the wrong shape but a lucky average above one that tracks the form exactly.
   */
  const dev = (L) => {
    const res = L.tiles.filter((t) => t.position).map((t) => {
      const ux = t.position[0] - t.normal[0] * PANEL_PROFILE.overallThickness
      const uy = t.position[1] - t.normal[1] * PANEL_PROFILE.overallThickness
      const uz = t.position[2] - t.normal[2] * PANEL_PROFILE.overallThickness
      return uy - driftHeight(form, ux, uz)
    })
    const mean = res.reduce((a, b) => a + b, 0) / res.length
    return Math.sqrt(res.reduce((a, b) => a + (b - mean) ** 2, 0) / res.length)
  }
  const S = solveLayout(cfgS)
  const C = solveLayout(cfgC)
  ok(dev(S) < dev(C),
    `surface-fit tracks the authored shape (residual σ ${dev(S).toFixed(1)}cm) better than chain (${dev(C).toFixed(1)}cm)`)

  const edgeMax = (L) => Math.max(...L.support.edges.map((e) => e.maxClearanceCm))
  ok(edgeMax(S) < edgeMax(C),
    `surface-fit holds the graded edges nearer the floor (${edgeMax(S).toFixed(1)}cm vs ${edgeMax(C).toFixed(1)}cm)`)
  console.log(`   surface-fit shape residual σ ${dev(S).toFixed(2)}cm, worst graded-edge clearance ${edgeMax(S).toFixed(2)}cm`)
  console.log(`   chain       shape residual σ ${dev(C).toFixed(2)}cm, worst graded-edge clearance ${edgeMax(C).toFixed(2)}cm`)
}

// -----------------------------------------------------------------------------
// 7. Panel solids — closed-form edge lengths, both tile types, both plate axes.
// -----------------------------------------------------------------------------
console.log('7. panel solids and OBBs')
{
  const L = solveLayout(cfgOf())
  const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])
  let sq = 0
  let pl = 0
  for (const t of L.tiles) {
    const c = panelSolidCorners(t)
    ok(c.length === 8, `${t.id}: 8 solid corners`)
    ok(new Set(c.map((p) => p.join(','))).size === 8, `${t.id}: corners distinct`)
    // Lit face is corners 0..3 in material order; edges are uLen and vLen.
    near(dist(c[0], c[1]), t.uv.uLen, 1e-6, `${t.id}: lit-face edge = uLen`)
    near(dist(c[1], c[2]), t.uv.vLen, 1e-6, `${t.id}: lit-face edge = vLen`)
    // Housing offset is exactly the profile thickness, along −n̂.
    near(dist(c[0], c[4]), PANEL_PROFILE.overallThickness, 1e-6, `${t.id}: housing depth`)

    const o = tileOBB(t)
    near(o.halfExtents[1], PANEL_PROFILE.overallThickness / 2, 1e-9, `${t.id}: OBB half-thickness`)
    // The OBB is re-centred on the SOLID, half a thickness behind the lit face.
    near(dist(o.center, t.position), PANEL_PROFILE.overallThickness / 2, 1e-6,
      `${t.id}: OBB centre offset from lit face`)
    near(o.halfExtents[0] * 2, t.width, 1e-9, `${t.id}: OBB width matches tile`)
    near(o.halfExtents[2] * 2, t.length, 1e-9, `${t.id}: OBB length matches tile`)
    if (t.type === '2x2') sq++
    else pl++
  }
  ok(sq > 0 && pl > 0, `both tile types exercised (${sq} squares, ${pl} plates)`)
}

// -----------------------------------------------------------------------------
// 8. Panels must not interpenetrate — the check v2 never needed.
// -----------------------------------------------------------------------------
console.log('8. panel interpenetration is detected and grows with curvature')
{
  // NOTE: panels colliding on a steep drift is a REAL PHYSICAL RESULT, not a
  // bug. On a convex region adjacent tiles tilt away from each other, so their
  // lit faces open while their HOUSINGS converge — and past some curvature the
  // housings interpenetrate. That is a genuine buildability limit and one of
  // the main things this tool exists to surface. So the assertion here is that
  // the DETECTOR behaves (silent when flat, responsive to curvature), not that
  // any particular design happens to be collision-free.
  const count = (amp) => {
    const cfg = cfgOf({ form: { amplitude: amp, toeSharpX: 1, toeSharpZ: 1 } })
    const L = solveLayout(cfg)
    return findCollisions(L.tiles.filter((t) => t.position).map(tileOBB), { minDepthCm: 0.05 }).length
  }
  ok(count(0) === 0, 'a flat sheet has no interpenetration')
  ok(count(20) === 0, 'a gentle drift has no interpenetration')
  const steep = count(120)
  ok(steep > 0, `a steep drift does interpenetrate, so the check is not vacuous (${steep} pairs)`)
  ok(count(80) <= steep, 'interpenetration grows with curvature')

  // Locate the buildability threshold — the number a designer actually needs.
  let onset = null
  for (const amp of [10, 20, 30, 40, 50, 60, 80, 100, 120]) {
    if (onset === null && count(amp) > 0) onset = amp
  }
  console.log(`   first interpenetration at amplitude ${onset}cm (6×8 sheet, linear toe)`)
}

// -----------------------------------------------------------------------------
// 9. Robustness: no NaN anywhere across a wide parameter sweep.
// -----------------------------------------------------------------------------
console.log('9. no NaN across a parameter sweep')
{
  let checked = 0
  for (const amp of [0, 250]) {
    for (const shear of [-0.4, 0, 0.4]) {
      for (const ts of [0.45, 1.0]) {
        for (const mode of ['surface-fit', 'chain']) {
          for (const [cols, rows] of [[4, 5], [6, 8], [8, 10]]) {
            const cfg = cfgOf({
              sheet: { cols, rows },
              form: { amplitude: amp, ridgeShear: shear, toeSharpX: ts, toeSharpZ: ts },
              placement: { mode },
            })
            const L = solveLayout(cfg)
            const bad = L.tiles.some((t) => t.position && (
              t.position.some((v) => !Number.isFinite(v)) ||
              t.quaternion.some((v) => !Number.isFinite(v)) ||
              !Number.isFinite(t.minY)))
            ok(!bad, `no NaN at amp ${amp} shear ${shear} toe ${ts} ${mode} ${cols}x${rows}`)
            checked++
          }
        }
      }
    }
  }
  console.log(`   swept ${checked} configurations`)
}

// -----------------------------------------------------------------------------
// 10. layoutBounds degenerates safely.
// -----------------------------------------------------------------------------
console.log('10. layoutBounds edge cases')
{
  const empty = layoutBounds({ tiles: [] })
  ok(empty.size.every((v) => v === 0), 'empty layout gives a zero box rather than Infinity')
  const L = solveLayout(cfgOf())
  const b = layoutBounds(L)
  ok(b.size.every((v) => v > 0), 'a real layout has positive extent')
  ok(b.max.every((v, k) => v >= b.min[k]), 'bounds max ≥ min componentwise')
}

console.log(`\ntest-v3-placement: ${passed} checks passed, ${failed} failed`)
process.exit(failed ? 1 : 0)
