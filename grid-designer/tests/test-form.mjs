/**
 * tests/test-form.mjs — headless checks for core/form.js (v3 drift heightfield).
 *
 * Plain node script, no test framework. Exits non-zero on any failure.
 *   node tests/test-form.mjs
 *
 * What this file is defending, in order of how badly it would hurt to regress:
 *   1. The two grounded edges (V3_SPEC.md §2.1): H(0, z) = H(x, 0) = 0 with
 *      NON-ZERO slope at both. A flat toe there is exactly what the brief
 *      forbids ("start from the ground but not flat"), and it is easy to get
 *      the zero right while silently losing the slope.
 *   2. The ridgeShear cross-term in ∂H/∂x. `driftGradient` is checked against a
 *      central difference at a wide scatter of interior points, several with
 *      ridgeShear ≠ 0 — this is the check that would fail if that term were
 *      ever dropped (V3_SPEC.md §2.4: "the kind of sign bug that has bitten
 *      this project before").
 *   3. Everything else: the profile's peak/value-1 property, normal unit
 *      length, footprint zeroing, determinism, normalizeForm's clamps, and the
 *      mesh sampler's index bookkeeping.
 */

import { DEFAULT_FORM, driftFrame, driftGradient, driftHeight, driftNormal, normalizeForm, sampleDriftMesh } from '../src/core/form.js'

let passed = 0
const failures = []

function check(name, condition, detail = '') {
  if (condition) {
    passed++
  } else {
    failures.push(`${name}${detail ? ` — ${detail}` : ''}`)
    console.error(`FAIL  ${name}${detail ? ` — ${detail}` : ''}`)
  }
}

const near = (a, b, eps = 1e-6) => Math.abs(a - b) <= eps

/** Deterministic PRNG (mulberry32) — no Math.random, so a failing scattered
 *  test always reproduces with the same inputs on the next run. */
function mulberry32(seed) {
  let a = seed >>> 0
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const F = normalizeForm(DEFAULT_FORM)

// =============================================================================
// 1. Grounded edges: H(0, z) = H(x, 0) = 0 for a spread of values
// =============================================================================
{
  const zs = [0, 1, 10, 50, 107.3, 215, 300, 429.9, 430, 500]
  check(
    'H(0, z) = 0 across a spread of z, including outside the footprint',
    zs.every((z) => driftHeight(F, 0, z) === 0),
    JSON.stringify(zs.map((z) => driftHeight(F, 0, z))),
  )
  const xs = [0, 1, 10, 50, 91.7, 200, 300, 364.9, 365, 500]
  check(
    'H(x, 0) = 0 across a spread of x, including outside the footprint',
    xs.every((x) => driftHeight(F, x, 0) === 0),
    JSON.stringify(xs.map((x) => driftHeight(F, x, 0))),
  )
  // Same, across a scatter of other form parameterisations (crest/toe/shear).
  const forms = [
    normalizeForm({ crestX: 0.2, crestZ: 0.8, ridgeShear: -0.35, toeSharpX: 0.45, toeSharpZ: 1.0 }),
    normalizeForm({ crestX: 0.8, crestZ: 0.2, ridgeShear: 0.35, toeSharpX: 1.0, toeSharpZ: 0.45 }),
    normalizeForm({ crestX: 0.5, crestZ: 0.5, ridgeShear: 0, toeSharpX: 0.6, toeSharpZ: 0.6 }),
  ]
  check(
    'H(0, z) = H(x, 0) = 0 holds across a scatter of other form parameterisations',
    forms.every((f) => zs.every((z) => driftHeight(f, 0, z) === 0) && xs.every((x) => driftHeight(f, x, 0) === 0)),
  )
}

// =============================================================================
// 2. Non-zero slope at both toes, across the whole toeSharp clamp range
//    [0.45, 1.0] — the brief's "not flat" made mechanical. Amplitude is fixed
//    non-zero here so the slope magnitude is meaningful, not just non-zero.
// =============================================================================
{
  const toeSharps = [0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  const smallFractions = [0.0005, 0.001, 0.005] // fraction of width/depth — "small x/z > 0"
  const MIN_SLOPE = 0.1 // observed range at these settings is ~0.7 to ~10.7; see task notes

  let worstX = Infinity
  for (const toeSharpX of toeSharps) {
    const form = normalizeForm({ ...DEFAULT_FORM, toeSharpX })
    const zAtRidge = form.crestZ * form.footprint.depth // near the z-axis crest, so Bz is not ~0
    for (const frac of smallFractions) {
      const x = frac * form.footprint.width
      const [dHdx] = driftGradient(form, x, zAtRidge)
      worstX = Math.min(worstX, dHdx)
    }
  }
  check(
    `wall edge (x→0+): dH/dx stays meaningfully positive (> ${MIN_SLOPE}) across toeSharpX ∈ [0.45, 1.0]`,
    worstX > MIN_SLOPE,
    `worst observed dH/dx = ${worstX}`,
  )

  let worstZ = Infinity
  for (const toeSharpZ of toeSharps) {
    const form = normalizeForm({ ...DEFAULT_FORM, toeSharpZ })
    const xAtRidge = form.crestX * form.footprint.width
    for (const frac of smallFractions) {
      const z = frac * form.footprint.depth
      const [, dHdz] = driftGradient(form, xAtRidge, z)
      worstZ = Math.min(worstZ, dHdz)
    }
  }
  check(
    `window edge (z→0+): dH/dz stays meaningfully positive (> ${MIN_SLOPE}) across toeSharpZ ∈ [0.45, 1.0]`,
    worstZ > MIN_SLOPE,
    `worst observed dH/dz = ${worstZ}`,
  )
}

// =============================================================================
// 3. B(τ; p, a) peaks at exactly τ = p with value 1 — checked indirectly
//    through driftHeight (B itself is module-private by design, per §2.4's
//    exact export list), by isolating one axis at a time:
//      - fix ridgeShear = 0 so tc(s) ≡ crestZ, a constant
//      - fix z = crestZ · depth exactly, so the z-factor sits AT ITS OWN peak
//        (t = tc(s) = crestZ) and contributes exactly 1 for every x — this
//        isolates H(x, z*) = amplitude · Bx(s)
//      - symmetrically for the z-axis factor with x fixed at crestX · width
// =============================================================================
{
  const form = normalizeForm({ ...DEFAULT_FORM, ridgeShear: 0 })
  const { width, depth } = form.footprint

  // --- x-axis (crestX) ---
  {
    const zStar = form.crestZ * depth
    const N = 4001
    let bestS = -1
    let bestVal = -Infinity
    for (let i = 0; i <= N; i++) {
      const s = i / N
      const val = driftHeight(form, s * width, zStar)
      if (val > bestVal) {
        bestVal = val
        bestS = s
      }
    }
    check(
      'x-axis profile: sampled peak location lands on crestX to grid resolution',
      near(bestS, form.crestX, 1 / N + 1e-9),
      `bestS=${bestS} vs crestX=${form.crestX}`,
    )
    // Tolerance is loose (not 1e-6) because this is a discrete grid search: the
    // nearest sample to the true peak is off by up to half a grid step, and B
    // is quadratically flat there, so the sampled max undershoots amplitude by
    // O((1/N)^2) — the EXACT evaluation just below checks the true value.
    check('x-axis profile: sampled peak value ≈ amplitude (both factors ≈ 1 there)', near(bestVal, form.amplitude, 1e-4), String(bestVal))
    check(
      'x-axis profile: exact value at x = crestX·width is amplitude',
      near(driftHeight(form, form.crestX * width, zStar), form.amplitude, 1e-6),
    )
    // Derivative sign change across the peak: + before, − after.
    const eps = 1e-4
    const [dBefore] = driftGradient(form, (form.crestX - eps) * width, zStar)
    const [dAfter] = driftGradient(form, (form.crestX + eps) * width, zStar)
    check(
      'x-axis profile: dH/dx changes sign across the peak (+ before, − after)',
      dBefore > 0 && dAfter < 0,
      `before=${dBefore} after=${dAfter}`,
    )
  }

  // --- z-axis (crestZ) ---
  {
    const xStar = form.crestX * width
    const N = 4001
    let bestT = -1
    let bestVal = -Infinity
    for (let i = 0; i <= N; i++) {
      const t = i / N
      const val = driftHeight(form, xStar, t * depth)
      if (val > bestVal) {
        bestVal = val
        bestT = t
      }
    }
    check(
      'z-axis profile: sampled peak location lands on crestZ to grid resolution',
      near(bestT, form.crestZ, 1 / N + 1e-9),
      `bestT=${bestT} vs crestZ=${form.crestZ}`,
    )
    // See the x-axis note above for why this tolerance is loose.
    check('z-axis profile: sampled peak value ≈ amplitude', near(bestVal, form.amplitude, 1e-4), String(bestVal))
    const eps = 1e-4
    const [, dBefore] = driftGradient(form, xStar, (form.crestZ - eps) * depth)
    const [, dAfter] = driftGradient(form, xStar, (form.crestZ + eps) * depth)
    check(
      'z-axis profile: dH/dz changes sign across the peak (+ before, − after)',
      dBefore > 0 && dAfter < 0,
      `before=${dBefore} after=${dAfter}`,
    )
  }
}

// =============================================================================
// 4. driftGradient vs central difference — the crux test. ≥ 40 scattered
//    interior points, several with ridgeShear ≠ 0, so this fails loudly if the
//    ∂Bz/∂tc · dtc/ds cross-term is ever dropped from ∂H/∂x.
//
//    Step size h = 1e-4 cm: measured during development (2000 random samples
//    of the underlying profile primitive, and 200 random samples of the full
//    driftHeight surface) the analytic-vs-central-difference error is ~1e-9 to
//    ~1e-10 at h = 1e-4 — comfortably inside the tolerance. h = 1e-2 is already
//    ~1e-8 (truncation-error dominated); h = 1e-6 climbs back to ~1e-8 in the
//    other direction (floating-point cancellation in the (f(x+h) − f(x−h))
//    subtraction, since H is O(10-250) but h·H'' is tiny). 1e-4 sits in the
//    flat middle of that error-vs-h curve.
// =============================================================================
{
  const h = 1e-4
  const rand = mulberry32(20260726)
  const points = []
  for (let i = 0; i < 50; i++) {
    const form = normalizeForm({
      amplitude: 20 + rand() * 220,
      crestX: 0.2 + rand() * 0.6,
      crestZ: 0.2 + rand() * 0.6,
      // Bias toward non-zero ridgeShear: only every 5th sample gets exactly 0.
      ridgeShear: i % 5 === 0 ? 0 : -0.4 + rand() * 0.8,
      toeSharpX: 0.45 + rand() * 0.55,
      toeSharpZ: 0.45 + rand() * 0.55,
      footprint: { width: 300 + rand() * 150, depth: 300 + rand() * 250 },
    })
    // Interior, away from the s=0/1, t=0/1 kinks so the central difference is
    // sampling a smooth region on both sides.
    const s = 0.1 + rand() * 0.8
    const t = 0.1 + rand() * 0.8
    const x = s * form.footprint.width
    const z = t * form.footprint.depth
    points.push({ form, x, z })
  }
  check('gradient check: generated at least 40 scattered points', points.length >= 40, String(points.length))
  check(
    'gradient check: includes several points with ridgeShear ≠ 0',
    points.filter((pt) => pt.form.ridgeShear !== 0).length >= 30,
    String(points.filter((pt) => pt.form.ridgeShear !== 0).length),
  )

  let worstErrX = 0
  let worstErrZ = 0
  let worstDetail = ''
  for (const { form, x, z } of points) {
    const [dHdx, dHdz] = driftGradient(form, x, z)
    const numDHdx = (driftHeight(form, x + h, z) - driftHeight(form, x - h, z)) / (2 * h)
    const numDHdz = (driftHeight(form, x, z + h) - driftHeight(form, x, z - h)) / (2 * h)
    const errX = Math.abs(dHdx - numDHdx)
    const errZ = Math.abs(dHdz - numDHdz)
    if (errX > worstErrX) {
      worstErrX = errX
      worstDetail = `x: analytic=${dHdx} numeric=${numDHdx} ridgeShear=${form.ridgeShear}`
    }
    if (errZ > worstErrZ) {
      worstErrZ = errZ
      worstDetail = `z: analytic=${dHdz} numeric=${numDHdz} ridgeShear=${form.ridgeShear}`
    }
  }
  check(`gradient check: dH/dx matches central difference to 1e-5 (worst = ${worstErrX})`, worstErrX < 1e-5, worstDetail)
  check(`gradient check: dH/dz matches central difference to 1e-5 (worst = ${worstErrZ})`, worstErrZ < 1e-5, worstDetail)
}

// =============================================================================
// 5. driftNormal: unit length, points generally +Y
// =============================================================================
{
  const rand = mulberry32(99)
  let allUnit = true
  let allUpish = true
  let worstLen = 0
  for (let i = 0; i < 60; i++) {
    const form = normalizeForm({
      amplitude: rand() * 250,
      crestX: 0.15 + rand() * 0.7,
      crestZ: 0.15 + rand() * 0.7,
      ridgeShear: -0.4 + rand() * 0.8,
      toeSharpX: 0.45 + rand() * 0.55,
      toeSharpZ: 0.45 + rand() * 0.55,
    })
    const x = rand() * form.footprint.width
    const z = rand() * form.footprint.depth
    const n = driftNormal(form, x, z)
    const len = Math.sqrt(n.x * n.x + n.y * n.y + n.z * n.z)
    worstLen = Math.max(worstLen, Math.abs(len - 1))
    if (!near(len, 1, 1e-9)) allUnit = false
    if (!(n.y > 0)) allUpish = false
  }
  check('driftNormal: unit length across a scatter of forms/points', allUnit, `worst |len-1| = ${worstLen}`)
  check('driftNormal: points generally +Y (n.y > 0) everywhere sampled', allUpish)

  // Flat spot (outside the footprint, or amplitude = 0): normal is exactly +Y.
  const flatForm = normalizeForm({ ...DEFAULT_FORM, amplitude: 0 })
  const n = driftNormal(flatForm, 100, 100)
  check('driftNormal: exactly +Y when amplitude = 0 (no slope anywhere)', n.x === 0 && n.y === 1 && n.z === 0, JSON.stringify(n))

  // driftFrame bundles the same point + normal.
  const frame = driftFrame(F, 150, 200)
  check(
    'driftFrame: point.y matches driftHeight, normal matches driftNormal',
    near(frame.point.y, driftHeight(F, 150, 200)) &&
      frame.point.x === 150 &&
      frame.point.z === 200 &&
      JSON.stringify(frame.normal) === JSON.stringify(driftNormal(F, 150, 200)),
    JSON.stringify(frame),
  )
}

// =============================================================================
// 6. H = 0 outside the footprint, and no NaN anywhere on a dense sweep —
//    corners, exact boundaries, and well outside on all four sides.
// =============================================================================
{
  const extremeForms = [
    normalizeForm({ amplitude: 0 }),
    normalizeForm({ amplitude: 250 }),
    normalizeForm({ crestX: 0.15, crestZ: 0.15, ridgeShear: -0.4, toeSharpX: 0.45, toeSharpZ: 0.45 }),
    normalizeForm({ crestX: 0.85, crestZ: 0.85, ridgeShear: 0.4, toeSharpX: 1.0, toeSharpZ: 1.0 }),
    normalizeForm(DEFAULT_FORM),
  ]

  for (const form of extremeForms) {
    const { width, depth } = form.footprint
    const xs = [-100, -1, 0, width * 0.25, width * 0.5, width * 0.75, width, width + 1, width + 100]
    const zs = [-100, -1, 0, depth * 0.25, depth * 0.5, depth * 0.75, depth, depth + 1, depth + 100]

    let allFinite = true
    let outsideZero = true
    for (const x of xs) {
      for (const z of zs) {
        const h = driftHeight(form, x, z)
        if (!Number.isFinite(h)) allFinite = false
        const [gx, gz] = driftGradient(form, x, z)
        if (!Number.isFinite(gx) || !Number.isFinite(gz)) allFinite = false
        const outside = x <= 0 || x >= width || z <= 0 || z >= depth
        if (outside && h !== 0) outsideZero = false
      }
    }
    check('dense sweep: driftHeight/driftGradient are finite everywhere (no NaN)', allFinite, JSON.stringify(form))
    check('dense sweep: H = 0 everywhere outside (or exactly on) the footprint boundary', outsideZero, JSON.stringify(form))
  }

  // Corners explicitly.
  const form = F
  const { width, depth } = form.footprint
  const corners = [
    [0, 0],
    [width, 0],
    [0, depth],
    [width, depth],
  ]
  check(
    'corners: all four footprint corners give H = 0',
    corners.every(([x, z]) => driftHeight(form, x, z) === 0),
    JSON.stringify(corners.map(([x, z]) => driftHeight(form, x, z))),
  )
  check(
    'corners: driftNormal is finite (no NaN) at all four corners',
    corners.every(([x, z]) => {
      const n = driftNormal(form, x, z)
      return Number.isFinite(n.x) && Number.isFinite(n.y) && Number.isFinite(n.z)
    }),
  )
}

// =============================================================================
// 7. Determinism: repeated calls are byte-identical
// =============================================================================
{
  const x = 137.4, z = 261.9
  check(
    'driftHeight is deterministic',
    JSON.stringify(driftHeight(F, x, z)) === JSON.stringify(driftHeight(F, x, z)),
  )
  check(
    'driftGradient is deterministic',
    JSON.stringify(driftGradient(F, x, z)) === JSON.stringify(driftGradient(F, x, z)),
  )
  check(
    'driftNormal is deterministic',
    JSON.stringify(driftNormal(F, x, z)) === JSON.stringify(driftNormal(F, x, z)),
  )
  check(
    'driftFrame is deterministic',
    JSON.stringify(driftFrame(F, x, z)) === JSON.stringify(driftFrame(F, x, z)),
  )
  const mesh1 = sampleDriftMesh(F, 9, 7)
  const mesh2 = sampleDriftMesh(F, 9, 7)
  check(
    'sampleDriftMesh is deterministic (positions)',
    JSON.stringify(Array.from(mesh1.positions)) === JSON.stringify(Array.from(mesh2.positions)),
  )
  check(
    'sampleDriftMesh is deterministic (indices)',
    JSON.stringify(Array.from(mesh1.indices)) === JSON.stringify(Array.from(mesh2.indices)),
  )
  check(
    'normalizeForm is deterministic',
    JSON.stringify(normalizeForm(DEFAULT_FORM)) === JSON.stringify(normalizeForm(DEFAULT_FORM)),
  )
}

// =============================================================================
// 8. normalizeForm: fills defaults, clamps out-of-range input, idempotent
// =============================================================================
{
  check('normalizeForm(undefined) returns the defaults', JSON.stringify(normalizeForm(undefined)) === JSON.stringify(DEFAULT_FORM))
  check('normalizeForm({}) returns the defaults', JSON.stringify(normalizeForm({})) === JSON.stringify(DEFAULT_FORM))
  check('normalizeForm(null) returns the defaults', JSON.stringify(normalizeForm(null)) === JSON.stringify(DEFAULT_FORM))

  const outOfRange = {
    amplitude: -50,
    crestX: 0.0,
    crestZ: 1.0,
    ridgeShear: -5,
    toeSharpX: 0.0,
    toeSharpZ: 5,
    footprint: { width: -10, depth: 0 },
  }
  const n = normalizeForm(outOfRange)
  check('normalizeForm clamps amplitude below range to 0', n.amplitude === 0, String(n.amplitude))
  check('normalizeForm clamps crestX below range to 0.15', n.crestX === 0.15, String(n.crestX))
  check('normalizeForm clamps crestZ above range to 0.85', n.crestZ === 0.85, String(n.crestZ))
  check('normalizeForm clamps ridgeShear below range to -0.4', n.ridgeShear === -0.4, String(n.ridgeShear))
  check('normalizeForm clamps toeSharpX below range to 0.45', n.toeSharpX === 0.45, String(n.toeSharpX))
  check('normalizeForm clamps toeSharpZ above range to 1.0', n.toeSharpZ === 1.0, String(n.toeSharpZ))
  check('normalizeForm clamps non-positive footprint.width up to a positive floor', n.footprint.width > 0, String(n.footprint.width))
  check('normalizeForm clamps non-positive footprint.depth up to a positive floor', n.footprint.depth > 0, String(n.footprint.depth))

  const aboveRange = {
    amplitude: 9999,
    crestX: 2,
    crestZ: -2,
    ridgeShear: 9,
    toeSharpX: 2,
    toeSharpZ: -2,
  }
  const n2 = normalizeForm(aboveRange)
  check('normalizeForm clamps amplitude above range to 250', n2.amplitude === 250, String(n2.amplitude))
  check('normalizeForm clamps crestX above range to 0.85', n2.crestX === 0.85, String(n2.crestX))
  check('normalizeForm clamps crestZ below range to 0.15', n2.crestZ === 0.15, String(n2.crestZ))
  check('normalizeForm clamps ridgeShear above range to 0.4', n2.ridgeShear === 0.4, String(n2.ridgeShear))
  check('normalizeForm clamps toeSharpX above range to 1.0', n2.toeSharpX === 1.0, String(n2.toeSharpX))
  check('normalizeForm clamps toeSharpZ below range to 0.45', n2.toeSharpZ === 0.45, String(n2.toeSharpZ))

  // Idempotence, including on already-out-of-range and on default input.
  for (const input of [outOfRange, aboveRange, DEFAULT_FORM, {}, undefined]) {
    const once = normalizeForm(input)
    const twice = normalizeForm(once)
    check(`normalizeForm is idempotent for ${JSON.stringify(input)}`, JSON.stringify(once) === JSON.stringify(twice))
  }

  // Partial input keeps unspecified fields at default.
  const partial = normalizeForm({ amplitude: 80 })
  check(
    'normalizeForm fills in unspecified fields from DEFAULT_FORM',
    partial.amplitude === 80 &&
      partial.crestX === DEFAULT_FORM.crestX &&
      partial.crestZ === DEFAULT_FORM.crestZ &&
      partial.ridgeShear === DEFAULT_FORM.ridgeShear &&
      partial.toeSharpX === DEFAULT_FORM.toeSharpX &&
      partial.toeSharpZ === DEFAULT_FORM.toeSharpZ &&
      partial.footprint.width === DEFAULT_FORM.footprint.width &&
      partial.footprint.depth === DEFAULT_FORM.footprint.depth,
    JSON.stringify(partial),
  )

  // Non-finite input (NaN, Infinity, wrong type) falls back to the default,
  // it does not propagate NaN.
  const junk = normalizeForm({ amplitude: NaN, crestX: Infinity, ridgeShear: 'left', footprint: { width: null } })
  check(
    'normalizeForm falls back to defaults for NaN/Infinity/wrong-typed fields rather than propagating them',
    Number.isFinite(junk.amplitude) &&
      Number.isFinite(junk.crestX) &&
      Number.isFinite(junk.ridgeShear) &&
      Number.isFinite(junk.footprint.width),
    JSON.stringify(junk),
  )
}

// =============================================================================
// 9. sampleDriftMesh: vertex/index counts, indices in range, no degenerate
//    triangles
// =============================================================================
{
  for (const [nx, nz] of [[2, 2], [5, 4], [17, 9], [3, 30]]) {
    const { positions, indices } = sampleDriftMesh(F, nx, nz)
    const expectedVerts = nx * nz
    const expectedTris = (nx - 1) * (nz - 1) * 2
    check(
      `mesh ${nx}×${nz}: positions.length = 3 · nx · nz`,
      positions.length === expectedVerts * 3,
      `${positions.length} vs ${expectedVerts * 3}`,
    )
    check(
      `mesh ${nx}×${nz}: indices.length = 3 · 2 · (nx-1)(nz-1)`,
      indices.length === expectedTris * 3,
      `${indices.length} vs ${expectedTris * 3}`,
    )
    check(`mesh ${nx}×${nz}: positions is a Float32Array`, positions instanceof Float32Array)
    check(`mesh ${nx}×${nz}: indices is a Uint32Array`, indices instanceof Uint32Array)

    let allInRange = true
    for (let i = 0; i < indices.length; i++) {
      if (indices[i] < 0 || indices[i] >= expectedVerts) allInRange = false
    }
    check(`mesh ${nx}×${nz}: every index is in range [0, vertexCount)`, allInRange)

    // No degenerate triangles: every triangle has non-zero area (checked via
    // the 3D cross product magnitude of two edges).
    let allNonDegenerate = true
    let minArea = Infinity
    for (let t = 0; t < indices.length; t += 3) {
      const ia = indices[t] * 3, ib = indices[t + 1] * 3, ic = indices[t + 2] * 3
      const ax = positions[ia], ay = positions[ia + 1], az = positions[ia + 2]
      const bx = positions[ib], by = positions[ib + 1], bz = positions[ib + 2]
      const cx = positions[ic], cy = positions[ic + 1], cz = positions[ic + 2]
      const e1x = bx - ax, e1y = by - ay, e1z = bz - az
      const e2x = cx - ax, e2y = cy - ay, e2z = cz - az
      const crossx = e1y * e2z - e1z * e2y
      const crossy = e1z * e2x - e1x * e2z
      const crossz = e1x * e2y - e1y * e2x
      const area = 0.5 * Math.sqrt(crossx * crossx + crossy * crossy + crossz * crossz)
      minArea = Math.min(minArea, area)
      if (!(area > 1e-9)) allNonDegenerate = false
    }
    check(`mesh ${nx}×${nz}: no degenerate triangles (min area ${minArea.toExponential(3)})`, allNonDegenerate)
  }

  // The mesh's boundary matches driftHeight directly at a handful of sample points.
  const { positions } = sampleDriftMesh(F, 5, 5)
  check('mesh: vertex (0,0) is the footprint origin at H = 0', positions[0] === 0 && positions[1] === 0 && positions[2] === 0)
  const lastX = F.footprint.width, lastZ = F.footprint.depth
  const lastIdx = (5 * 5 - 1) * 3
  check(
    'mesh: last vertex is the far corner (width, H(width,depth)=0, depth)',
    near(positions[lastIdx], lastX) && near(positions[lastIdx + 1], 0) && near(positions[lastIdx + 2], lastZ),
    `${positions[lastIdx]}, ${positions[lastIdx + 1]}, ${positions[lastIdx + 2]}`,
  )
}

// =============================================================================
// Summary
// =============================================================================
console.log('')
console.log(`test-form: ${passed} checks passed, ${failures.length} failed`)
if (failures.length > 0) {
  console.error('')
  console.error('Failures:')
  for (const f of failures) console.error(`  - ${f}`)
  process.exit(1)
}
