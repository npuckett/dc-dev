/**
 * tests/test-v3-target.mjs — headless checks for core/v3/target.js.
 *
 * The target is what the panels are ASKED to be. Its job is to be reachable by
 * rigid flat panels, which a smooth drift is not. These checks establish that
 * the faceting actually does that rather than just changing the numbers.
 */

import { buildTarget, buildUnroll } from '../src/core/v3/target.js'
import { DEFAULT_CONFIG, normalizeConfig } from '../src/core/v3/schema.js'
import { normalizeForm, driftHeight } from '../src/core/v3/form.js'
import { solveLayout } from '../src/core/v3/placement.js'

let passed = 0
let failed = 0
const ok = (c, m) => { if (c) passed++; else { failed++; console.log(`  FAIL: ${m}`) } }
const near = (a, b, tol, m) => ok(Math.abs(a - b) <= tol, `${m} (got ${a}, want ${b} ±${tol})`)

const cfgOf = (over = {}) => normalizeConfig({
  ...DEFAULT_CONFIG,
  sheet: { cols: 6, rows: 8 },
  ...over,
  form: { ...DEFAULT_CONFIG.form, toeSharpX: 1, toeSharpZ: 1, ...(over.form ?? {}) },
})

console.log('=== test-v3-target ===')

// -----------------------------------------------------------------------------
// 1. The unroll is an arc-length map, and the graded edges are fixed points.
// -----------------------------------------------------------------------------
console.log('1. arc-length unroll')
{
  // Fixed-point checks are a property of the SMOOTH surface: H is identically
  // zero along both graded lines, so arc length there equals material length
  // and u must map to x unchanged. Faceting perturbs this (see below), because a
  // planar facet at the corner cannot be zero along both edges AND be tilted.
  const cfg = cfgOf({ form: { amplitude: 100, angularity: 0 } })
  const T = buildTarget(cfg)

  for (const v of [0, 100, 300, 480]) near(T.planAt(0, v).x, 0, 1e-6, `planAt(0,${v}).x = 0`)
  for (const u of [0, 100, 200, 360]) near(T.planAt(u, 0).z, 0, 1e-6, `planAt(${u},0).z = 0`)
  for (const u of [60, 180, 360]) near(T.planAt(u, 0).x, u, 1e-6, `window line: x tracks u at u=${u}`)
  for (const v of [60, 240, 480]) near(T.planAt(0, v).z, v, 1e-6, `wall line: z tracks v at v=${v}`)

  // A faceted target shifts the graded lines slightly, and the amount matters:
  // it is how far faceting moves the edges the brief pins to the floor.
  const TF = buildTarget(cfgOf({ form: { amplitude: 100, angularity: 1, facetCells: 2 } }))
  let worstShift = 0
  for (const u of [60, 180, 360]) worstShift = Math.max(worstShift, Math.abs(TF.planAt(u, 0).x - u))
  for (const v of [60, 240, 480]) worstShift = Math.max(worstShift, Math.abs(TF.planAt(0, v).z - v))
  // Bound set from the measured value (4.4cm at amplitude 100, facetCells 2),
  // not from a wish. It is one of the real costs of faceting and belongs on the
  // record: a planar facet at the wall/window corner cannot be zero along both
  // graded lines and still be tilted, so it trades a few cm of edge position
  // for the coplanarity that keeps the joints closed.
  ok(worstShift < 6, `faceting shifts the graded lines by under 6cm (worst ${worstShift.toFixed(2)}cm)`)
  console.log(`   faceted graded-line shift: ${worstShift.toFixed(2)}cm`)

  // A curved interior must PULL IN: the plan shadow is shorter than the sheet.
  const far = T.planAt(360, 480)
  ok(far.x <= 360 + 1e-6 && far.z <= 480 + 1e-6,
    `curved sheet's shadow does not exceed its material extent (got ${far.x.toFixed(1)}, ${far.z.toFixed(1)})`)

  // On a FLAT form the unroll is the identity.
  const flat = buildTarget(cfgOf({ form: { amplitude: 0 } }))
  for (const [u, v] of [[0, 0], [120, 240], [360, 480]]) {
    const p = flat.planAt(u, v)
    near(p.x, u, 1e-6, `flat unroll identity in u at (${u},${v})`)
    near(p.z, v, 1e-6, `flat unroll identity in v at (${u},${v})`)
  }

  // buildUnroll is exported and independently usable.
  const id = buildUnroll(() => [0, 0], 300, 400)
  near(id(150, 200).x, 150, 1e-6, 'zero-gradient unroll is the identity in x')
  near(id(150, 200).z, 200, 1e-6, 'zero-gradient unroll is the identity in z')
}

// -----------------------------------------------------------------------------
// 2. angularity = 0 reproduces the smooth drift exactly.
// -----------------------------------------------------------------------------
console.log('2. angularity 0 is the smooth drift')
{
  const cfg = cfgOf({ form: { amplitude: 100, angularity: 0 } })
  const form = normalizeForm(cfg.form)
  const T = buildTarget(cfg)
  for (const [u, v] of [[30, 30], [150, 200], [300, 400], [360, 480]]) {
    const { point } = T.frameAtMaterial(u, v)
    const [x, y, z] = point
    near(y, driftHeight(form, x, z), 1e-9, `smooth target height at material (${u},${v})`)
  }
}

// -----------------------------------------------------------------------------
// 3. THE POINT OF THE MODULE: at angularity 1, tiles sharing a facet are
//    COPLANAR, so the joints between them do not wedge open.
// -----------------------------------------------------------------------------
console.log('3. angularity 1 makes tiles within a facet coplanar')
for (const facetCells of [2, 3]) {
  const cfg = cfgOf({ form: { amplitude: 100, angularity: 1, facetCells } })
  const T = buildTarget(cfg)
  const L = solveLayout(cfg)

  // Group placed tiles by the facet their material centre falls in.
  const groups = new Map()
  for (const t of L.tiles) {
    if (!t.position) continue
    const u = t.uv.u0 + t.uv.uLen / 2
    const v = t.uv.v0 + t.uv.vLen / 2
    const k = T.facetIndexAt(u, v)
    if (!groups.has(k)) groups.set(k, [])
    groups.get(k).push(t)
  }
  let multi = 0
  let worstNormalSpread = 0
  for (const [, ts] of groups) {
    if (ts.length < 2) continue
    multi++
    // Every tile in the facet must share the facet plane's normal.
    for (let a = 1; a < ts.length; a++) {
      const d = Math.abs(ts[0].normal[0] * ts[a].normal[0] +
        ts[0].normal[1] * ts[a].normal[1] +
        ts[0].normal[2] * ts[a].normal[2])
      worstNormalSpread = Math.max(worstNormalSpread, Math.abs(1 - d))
    }
  }
  ok(multi > 0, `facetCells ${facetCells}: some facets hold more than one tile (${multi})`)
  // 1e-7, not 1e-9: placement rounds every emitted number to 1e-9 for
  // determinism, so a spread of order 1e-9 is that rounding, not a tilt.
  ok(worstNormalSpread < 1e-7,
    `facetCells ${facetCells}: co-facet tiles share a normal (worst deviation ${worstNormalSpread.toExponential(2)})`)
  console.log(`   facetCells ${facetCells}: ${groups.size} facets, ${multi} with >1 tile, normal spread ${worstNormalSpread.toExponential(1)}`)
}

// -----------------------------------------------------------------------------
// 4. Faceting reduces panel interpenetration — the buildability payoff.
// -----------------------------------------------------------------------------
console.log('4. faceting relieves housing interpenetration')
{
  const collisionsAt = async (form) => {
    const { tileOBB } = await import('../src/core/v3/placement.js')
    const { findCollisions } = await import('../src/core/v3/collide.js')
    const L = solveLayout(cfgOf({ form }))
    return findCollisions(L.tiles.filter((t) => t.position).map(tileOBB), { minDepthCm: 0.05 }).length
  }
  const smooth = await collisionsAt({ amplitude: 120, angularity: 0 })
  const faceted = await collisionsAt({ amplitude: 120, angularity: 1, facetCells: 3 })
  ok(faceted < smooth,
    `faceting cuts interpenetrating pairs (${smooth} smooth → ${faceted} faceted)`)
  console.log(`   amplitude 120: ${smooth} colliding pairs smooth, ${faceted} faceted (facetCells 3)`)
}

// -----------------------------------------------------------------------------
// 5. Determinism and robustness.
// -----------------------------------------------------------------------------
console.log('5. determinism and robustness')
{
  const snap = (cfg) => {
    const T = buildTarget(cfg)
    return JSON.stringify([[0, 0], [60, 120], [300, 400]].map(([u, v]) => {
      const f = T.frameAtMaterial(u, v)
      return [f.point, [f.normal.x, f.normal.y, f.normal.z]]
    }))
  }
  for (const ang of [0, 0.5, 1]) {
    const cfg = cfgOf({ form: { amplitude: 90, angularity: ang } })
    ok(snap(cfg) === snap(cfg), `angularity ${ang}: repeat builds byte-identical`)
  }
  // No NaN across extremes, including a zero-amplitude faceted form (every
  // facet plane degenerate) and maximum shear.
  let bad = 0
  for (const amplitude of [0, 250]) {
    for (const angularity of [0, 0.5, 1]) {
      for (const facetCells of [1, 4]) {
        for (const ridgeShear of [-0.4, 0.4]) {
          const T = buildTarget(cfgOf({ form: { amplitude, angularity, facetCells, ridgeShear } }))
          for (const [u, v] of [[0, 0], [180, 240], [360, 480]]) {
            const f = T.frameAtMaterial(u, v)
            if (!f.point.every(Number.isFinite)) bad++
            if (!Number.isFinite(f.normal.x + f.normal.y + f.normal.z)) bad++
            if (Math.abs(f.normal.length() - 1) > 1e-6) bad++
            if (f.normal.y <= 0) bad++
          }
        }
      }
    }
  }
  ok(bad === 0, `${bad} non-finite / non-unit / downward normals across the sweep`)

  // facetCells is an integer count and clamps.
  ok(buildTarget(cfgOf({ form: { facetCells: 99 } })).facetCells <= 4, 'facetCells clamps high')
  ok(buildTarget(cfgOf({ form: { facetCells: 0 } })).facetCells >= 1, 'facetCells clamps low')
  ok(buildTarget(cfgOf({ form: { facetCells: 1 } })).facetCount === 6 * 8,
    'facetCells 1 gives one facet per cell')
}

console.log(`\ntest-v3-target: ${passed} checks passed, ${failed} failed`)
process.exit(failed ? 1 : 0)
