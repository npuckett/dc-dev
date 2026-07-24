/**
 * tests/test-placement.mjs — headless checks for core/placement.js.
 *
 * Plain node script, no test framework. Exits non-zero on any failure.
 *   node tests/test-placement.mjs
 *
 * The closed-form expectations below are computed INDEPENDENTLY inside this
 * file (small local chain/plane walkers, hard-coded fold sequences) rather than
 * by calling back into placement.js — that is the whole point: they are the
 * guard against sign-convention errors in the solver.
 */

import assert from 'node:assert/strict'
import { DEFAULT_CONFIG, cellTilts, normalizeConfig, validateConfig } from '../src/core/schema.js'
import { buildPreset } from '../src/core/presets.js'
import { panelWorldCorners, solveLayout } from '../src/core/placement.js'

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

function checkNoThrow(name, fn) {
  try {
    fn()
    passed++
  } catch (e) {
    failures.push(`${name} — ${e.message}`)
    console.error(`FAIL  ${name} — ${e.message}`)
  }
}

const near = (a, b, eps = 1e-9) => Math.abs(a - b) <= eps
const nearVec = (v, e, eps = 1e-9) => v.length === e.length && v.every((x, i) => near(x, e[i], eps))
const D = Math.PI / 180
const cfgOf = (over) => normalizeConfig({ ...DEFAULT_CONFIG, ...over })
const rowsOf = (zigzags) => zigzags.map((z) => ({ zigzagDeg: z, jointOverridesDeg: {} }))

// Independent 2D accordion walker: given the signed fold angle at every joint,
// walk `folds.length + 1` unit segments of length L with `gap` advanced along
// each joint bisector. Mirrors the documented algorithm, written from scratch.
function walkChain(folds, L, gap) {
  let px = 0
  let py = 0
  let a = 0
  const centers = []
  const tilts = []
  const verts = [[0, 0]]
  for (let c = 0; c <= folds.length; c++) {
    tilts.push(a)
    centers.push([px + (L / 2) * Math.cos(a * D), py + (L / 2) * Math.sin(a * D)])
    px += L * Math.cos(a * D)
    py += L * Math.sin(a * D)
    verts.push([px, py])
    if (c < folds.length) {
      const bis = (a + folds[c] / 2) * D
      px += gap * Math.cos(bis)
      py += gap * Math.sin(bis)
      verts.push([px, py])
      a += folds[c]
    }
  }
  const xs = verts.map((v) => v[0])
  return { centers, tilts, verts, minX: Math.min(...xs), maxX: Math.max(...xs) }
}

const centerShift = (chain, cols, size, gap) =>
  (cols * (size + gap) - gap) / 2 - (chain.minX + chain.maxX) / 2

// =============================================================================
// 1. Flat DEFAULT_CONFIG
// =============================================================================
{
  const cfg = DEFAULT_CONFIG
  const layout = solveLayout(cfg)

  check('flat: 30 panels', layout.panels.length === 30, `got ${layout.panels.length}`)
  check('flat: no warnings', layout.warnings.length === 0, JSON.stringify(layout.warnings))

  let lattice = true
  let quats = true
  let ys = true
  let meta = true
  let bad = ''
  for (const p of layout.panels) {
    const want = [30 + 62 * p.col, 3.7, 30 + 62 * p.row]
    if (!nearVec(p.position, want)) {
      lattice = false
      bad = `${p.id} at ${JSON.stringify(p.position)} want ${JSON.stringify(want)}`
    }
    if (!nearVec(p.quaternion, [0, 0, 0, 1])) quats = false
    if (p.position[1] !== 3.7) ys = false
    if (
      p.type !== '2x2' ||
      p.rectOrientation !== null ||
      p.cells.length !== 1 ||
      p.cells[0][0] !== p.row ||
      p.cells[0][1] !== p.col ||
      p.tiltDeg !== 0 ||
      p.rowPitchDeg !== 0 ||
      p.id !== `p${p.row}_${p.col}`
    ) {
      meta = false
    }
  }
  check('flat: centers on the 62-pitch lattice (30 + 62c, 3.7, 30 + 62r)', lattice, bad)
  check('flat: all quaternions are identity', quats)
  check('flat: y is exactly 3.7 (no float noise)', ys)
  check('flat: panel metadata (type/cells/tilt/pitch/id)', meta)

  check(
    'flat: rowPlanes trace z = 62r at y = 3.7 with pitch 0',
    layout.rowPlanes.length === 5 &&
      layout.rowPlanes.every((rp, r) => nearVec(rp.origin, [0, 3.7, 62 * r]) && rp.pitchDeg === 0),
    JSON.stringify(layout.rowPlanes),
  )

  // Determinism
  checkNoThrow('flat: solveLayout twice → JSON-identical', () => {
    assert.strictEqual(JSON.stringify(solveLayout(cfg)), JSON.stringify(solveLayout(cfg)))
  })
  checkNoThrow('flat: solveLayout twice → deep-equal', () => {
    assert.deepStrictEqual(solveLayout(cfg), solveLayout(cfg))
  })
  checkNoThrow('flat: raw partial config solves identically to normalized', () => {
    assert.deepStrictEqual(solveLayout(cfg), solveLayout(normalizeConfig(cfg)))
  })

  // panelWorldCorners order and extents on a known-flat panel.
  const corners = panelWorldCorners(layout.panels[0], cfg)
  check(
    'flat: panelWorldCorners of p0_0 = [-X,-Z] [+X,-Z] [+X,+Z] [-X,+Z]',
    nearVec(corners[0], [0, 3.7, 0]) &&
      nearVec(corners[1], [60, 3.7, 0]) &&
      nearVec(corners[2], [60, 3.7, 60]) &&
      nearVec(corners[3], [0, 3.7, 60]),
    JSON.stringify(corners),
  )
}

// =============================================================================
// 2. Single zig-zag row (row 2 θ = 30, everything else flat)
// =============================================================================
{
  const cfg = cfgOf({ rows: rowsOf([0, 0, 30, 0, 0]) })
  check('zigzag row 2: config validates', validateConfig(cfg).ok, JSON.stringify(validateConfig(cfg).errors))

  const layout = solveLayout(cfg)
  const flat = solveLayout(DEFAULT_CONFIG)

  // Independent closed form: θ = 30 alternating over 5 joints of row 2.
  const folds = [30, -30, 30, -30, 30]
  const chain = walkChain(folds, 60, 2)
  const shift = centerShift(chain, 6, 60, 2)

  check('zigzag row 2: 30 panels', layout.panels.length === 30, `got ${layout.panels.length}`)

  const row2 = layout.panels.filter((p) => p.row === 2)
  check('zigzag row 2: 6 panels in row 2', row2.length === 6)

  // Closed-form positions. Row planes are unfolded, so row 2's plane origin is
  // (0, 3.7, 124) and the row-local (x, y, z) maps straight through to world.
  let posOk = true
  let posBad = ''
  row2.forEach((p, c) => {
    const want = [chain.centers[c][0] + shift, 3.7 + chain.centers[c][1], 124 + 30]
    if (!nearVec(p.position, want, 1e-8)) {
      posOk = false
      posBad = `${p.id} ${JSON.stringify(p.position)} want ${JSON.stringify(want)}`
    }
  })
  check('zigzag row 2: panel centers match the closed-form chain', posOk, posBad)

  // Per-cell tilt must agree with schema.js.
  const tilts = cellTilts(cfg, 2)
  check(
    'zigzag row 2: tiltDeg === cellTilts(config, 2)',
    row2.every((p, c) => near(p.tiltDeg, tilts[c])),
    `${JSON.stringify(row2.map((p) => p.tiltDeg))} vs ${JSON.stringify(tilts)}`,
  )
  check(
    'zigzag row 2: tilt sequence is the 0/30 staircase',
    JSON.stringify(row2.map((p) => p.tiltDeg)) === JSON.stringify([0, 30, 0, 30, 0, 30]),
    JSON.stringify(row2.map((p) => p.tiltDeg)),
  )
  check(
    'zigzag row 2: tilts match the independent walker',
    row2.every((p, c) => near(p.tiltDeg, chain.tilts[c])),
  )

  // Placed footprint extents vs closed-form chain extents.
  const cs = row2.flatMap((p) => panelWorldCorners(p, cfg))
  const xs = cs.map((v) => v[0])
  const ysw = cs.map((v) => v[1])
  check(
    'zigzag row 2: placed X extent === closed-form chain X extent',
    near(Math.min(...xs), chain.minX + shift, 1e-8) &&
      near(Math.max(...xs), chain.maxX + shift, 1e-8),
    `placed [${Math.min(...xs)}, ${Math.max(...xs)}] vs closed-form [${chain.minX + shift}, ${chain.maxX + shift}]`,
  )
  check(
    'zigzag row 2: shortened row is centered on the nominal grid center (185)',
    near((Math.min(...xs) + Math.max(...xs)) / 2, 185, 1e-8),
    String((Math.min(...xs) + Math.max(...xs)) / 2),
  )

  // Ridge amplitude: three +30° segments each rise 60·sin30 = 30, plus five
  // bisector advances of 2·sin15 each.
  const expectedRidge = 3 * 60 * Math.sin(30 * D) + 5 * 2 * Math.sin(15 * D)
  check(
    'zigzag row 2: ridge Y = 3·60·sin30 + 5·2·sin15 accumulation',
    near(Math.max(...chain.verts.map((v) => v[1])), expectedRidge, 1e-9),
    `${Math.max(...chain.verts.map((v) => v[1]))} vs ${expectedRidge}`,
  )
  check(
    'zigzag row 2: highest placed corner = 3.7 + ridge amplitude',
    near(Math.max(...ysw), 3.7 + expectedRidge, 1e-8),
    `${Math.max(...ysw)} vs ${3.7 + expectedRidge}`,
  )

  // Other rows must be untouched.
  const sameRows = [0, 1, 3, 4]
  checkNoThrow('zigzag row 2: rows 0/1/3/4 identical to the flat layout', () => {
    for (const r of sameRows) {
      assert.deepStrictEqual(
        layout.panels.filter((p) => p.row === r),
        flat.panels.filter((p) => p.row === r),
        `row ${r} moved`,
      )
    }
  })
  checkNoThrow('zigzag row 2: rowPlanes identical to the flat layout', () => {
    assert.deepStrictEqual(layout.rowPlanes, flat.rowPlanes)
  })
  checkNoThrow('zigzag row 2: deterministic', () => {
    assert.deepStrictEqual(solveLayout(cfg), solveLayout(cfg))
  })
}

// =============================================================================
// 3. Row folds only ([30, 30, 30, 30], zig-zags 0)
// =============================================================================
{
  const cfg = cfgOf({ rowFoldsDeg: [30, 30, 30, 30] })
  check('row folds: config validates', validateConfig(cfg).ok, JSON.stringify(validateConfig(cfg).errors))
  const layout = solveLayout(cfg)

  // Independent Y–Z chain: O_{r+1} = O_r + 60·d_r + 2·bisector(d_r, d_{r+1}).
  const psi = [0, 30, 60, 90, 120]
  const d = (a) => [Math.sin(a * D), Math.cos(a * D)]
  const origins = [[0, 3.7, 0]]
  for (let r = 0; r < 4; r++) {
    const dr = d(psi[r])
    const dn = d(psi[r + 1])
    let bx = dr[0] + dn[0]
    let by = dr[1] + dn[1]
    const len = Math.hypot(bx, by)
    bx /= len
    by /= len
    const prev = origins[r]
    origins.push([0, prev[1] + 60 * dr[0] + 2 * bx, prev[2] + 60 * dr[1] + 2 * by])
  }

  check(
    'row folds: rowPlane origins trace the closed-form Y–Z chain',
    layout.rowPlanes.every((rp, r) => nearVec(rp.origin, origins[r], 1e-8)),
    `${JSON.stringify(layout.rowPlanes.map((rp) => rp.origin))} vs ${JSON.stringify(origins)}`,
  )
  check(
    'row folds: rowPlane pitch is cumulative (0/30/60/90/120)',
    layout.rowPlanes.every((rp, r) => near(rp.pitchDeg, psi[r])),
    JSON.stringify(layout.rowPlanes.map((rp) => rp.pitchDeg)),
  )
  check(
    'row folds: every panel in row r has rowPitchDeg = 30r',
    layout.panels.every((p) => near(p.rowPitchDeg, 30 * p.row)),
  )
  check('row folds: all tilts 0', layout.panels.every((p) => p.tiltDeg === 0))

  // Quaternions are pure rotations about world +X by −ψ (positive fold pitches
  // the next row up: d_r gains +Y).
  let qOk = true
  let qBad = ''
  for (const p of layout.panels) {
    const half = (-psi[p.row] * D) / 2
    const want = [Math.sin(half), 0, 0, Math.cos(half)]
    if (!nearVec(p.quaternion, want, 1e-9)) {
      qOk = false
      qBad = `${p.id} ${JSON.stringify(p.quaternion)} want ${JSON.stringify(want)}`
    }
  }
  check('row folds: quaternions are pure X-rotations of −ψ_r', qOk, qBad)
  check(
    'row folds: quaternion y/z components are exactly zero',
    layout.panels.every((p) => p.quaternion[1] === 0 && p.quaternion[2] === 0),
  )

  // Panel centers: O_r + (30 + 62c, 30·sinψ, 30·cosψ).
  let posOk = true
  let posBad = ''
  for (const p of layout.panels) {
    const o = origins[p.row]
    const want = [
      30 + 62 * p.col,
      o[1] + 30 * Math.sin(psi[p.row] * D),
      o[2] + 30 * Math.cos(psi[p.row] * D),
    ]
    if (!nearVec(p.position, want, 1e-8)) {
      posOk = false
      posBad = `${p.id} ${JSON.stringify(p.position)} want ${JSON.stringify(want)}`
    }
  }
  check('row folds: panel centers = O_r + 30·d_r offset, X on the lattice', posOk, posBad)

  // ψ = 90° must stand the row straight up: its depth direction is +Y, so row 3
  // rises vertically above row 2's far edge.
  check(
    'row folds: ψ = 90° row rises vertically (row 3 panels share one z)',
    layout.panels
      .filter((p) => p.row === 3)
      .every((p) => near(p.position[2], origins[3][2], 1e-8)),
  )

  checkNoThrow('row folds: deterministic', () => {
    assert.deepStrictEqual(solveLayout(cfg), solveLayout(cfg))
  })
}

// =============================================================================
// 4. Rects
// =============================================================================
// 4a. Horizontal rect
{
  const cfg = cfgOf({ rects: [{ row: 4, col: 2, orientation: 'horizontal' }] })
  const v = validateConfig(cfg)
  check('h-rect: config validates', v.ok, JSON.stringify(v.errors))
  check(
    'h-rect: W_RECT_LENGTH surfaced (121 ≠ 122)',
    v.warnings.some((w) => w.code === 'W_RECT_LENGTH'),
    JSON.stringify(v.warnings),
  )

  const layout = solveLayout(cfg)
  check('h-rect: 29 panels', layout.panels.length === 29, `got ${layout.panels.length}`)

  const rect = layout.panels.find((p) => p.id === 'p4_2')
  check('h-rect: panel p4_2 exists', !!rect)
  check('h-rect: type 2x4', rect.type === '2x4', rect.type)
  check('h-rect: rectOrientation horizontal', rect.rectOrientation === 'horizontal')
  check(
    'h-rect: owns 2 cells (4,2) and (4,3)',
    JSON.stringify(rect.cells) === JSON.stringify([[4, 2], [4, 3]]),
    JSON.stringify(rect.cells),
  )
  check('h-rect: no separate panel at (4,3)', !layout.panels.some((p) => p.id === 'p4_3'))

  // Yaw of +90° about local Y so the 121 length runs along the row.
  const s = Math.sin(Math.PI / 4)
  check(
    'h-rect: quaternion carries the +90° yaw about local Y',
    nearVec(rect.quaternion, [0, s, 0, s]),
    JSON.stringify(rect.quaternion),
  )

  const corners = panelWorldCorners(rect, cfg)
  const xs = corners.map((c) => c[0])
  const zs = corners.map((c) => c[2])
  check(
    'h-rect: world footprint spans 121 along the row (+X)',
    near(Math.max(...xs) - Math.min(...xs), 121, 1e-8),
    String(Math.max(...xs) - Math.min(...xs)),
  )
  check(
    'h-rect: world footprint spans 60 in depth (+Z)',
    near(Math.max(...zs) - Math.min(...zs), 60, 1e-8),
    String(Math.max(...zs) - Math.min(...zs)),
  )
  check(
    'h-rect: sits in row 4 depth band (z = 248..308)',
    near(Math.min(...zs), 248, 1e-8) && near(Math.max(...zs), 308, 1e-8),
    JSON.stringify([Math.min(...zs), Math.max(...zs)]),
  )

  // Row 4 is 121 long instead of 122 where the rect sits, so a centered row 4
  // shifts by +0.5 — the accepted hardware slack, placed honestly.
  const row4 = layout.panels.filter((p) => p.row === 4)
  check(
    'h-rect: row 4 chain re-centers (+0.5 shift from the 121 vs 122 slack)',
    near(row4[0].position[0], 30.5, 1e-8) && near(rect.position[0], 185, 1e-8),
    JSON.stringify(row4.map((p) => p.position[0])),
  )
  check(
    'h-rect: rows 0-3 identical to the flat layout',
    JSON.stringify(layout.panels.filter((p) => p.row < 4)) ===
      JSON.stringify(solveLayout(DEFAULT_CONFIG).panels.filter((p) => p.row < 4)),
  )
  checkNoThrow('h-rect: deterministic', () => {
    assert.deepStrictEqual(solveLayout(cfg), solveLayout(cfg))
  })
}

// 4b. Vertical rect at column 0 with differing row zig-zags
{
  const base = { rows: rowsOf([0, 10, 20, 0, 0]), rowFoldsDeg: [0, 0, 0, 0] }
  const without = cfgOf(base)
  const withRect = cfgOf({ ...base, rects: [{ row: 1, col: 0, orientation: 'vertical' }] })

  const v = validateConfig(withRect)
  check('v-rect: config validates (col 0 tilts are 0 in both rows)', v.ok, JSON.stringify(v.errors))

  const layout = solveLayout(withRect)
  const plain = solveLayout(without)

  check('v-rect: 29 panels', layout.panels.length === 29, `got ${layout.panels.length}`)

  const rect = layout.panels.find((p) => p.id === 'p1_0')
  check('v-rect: type 2x4', rect.type === '2x4', rect.type)
  check('v-rect: rectOrientation vertical', rect.rectOrientation === 'vertical')
  check(
    'v-rect: owns cells (1,0) and (2,0)',
    JSON.stringify(rect.cells) === JSON.stringify([[1, 0], [2, 0]]),
    JSON.stringify(rect.cells),
  )
  check('v-rect: row 2 emits no panel at column 0', !layout.panels.some((p) => p.id === 'p2_0'))
  check(
    'v-rect: no panel is anchored in row 2 column 0',
    !layout.panels.some((p) => p.row === 2 && p.col === 0),
  )

  // Placed in row 1's plane, near edge at Z_row = 0, extending 121 along d_1.
  check(
    'v-rect: placed in row 1 plane (rowPitchDeg 0, Z_row center = 60.5 → z = 122.5)',
    rect.rowPitchDeg === 0 && near(rect.position[2], 62 + 60.5, 1e-8),
    JSON.stringify(rect.position),
  )
  {
    const zs = panelWorldCorners(rect, withRect).map((c) => c[2])
    const xs = panelWorldCorners(rect, withRect).map((c) => c[0])
    check(
      'v-rect: 121 length runs along depth (z = 62..183), 60 across the row',
      near(Math.min(...zs), 62, 1e-8) &&
        near(Math.max(...zs), 183, 1e-8) &&
        near(Math.max(...xs) - Math.min(...xs), 60, 1e-8),
      JSON.stringify([Math.min(...zs), Math.max(...zs), Math.max(...xs) - Math.min(...xs)]),
    )
  }
  check('v-rect: no yaw (quaternion is identity here)', nearVec(rect.quaternion, [0, 0, 0, 1]))

  // The occupied cell advances row 2's chain exactly as a square would, so every
  // other panel in rows 1 and 2 is bit-identical to the rect-free layout.
  checkNoThrow('v-rect: row 2 columns 1..5 unmoved vs the same config without the rect', () => {
    assert.deepStrictEqual(
      layout.panels.filter((p) => p.row === 2 && p.col > 0),
      plain.panels.filter((p) => p.row === 2 && p.col > 0),
    )
  })
  checkNoThrow('v-rect: row 1 columns 1..5 unmoved vs the same config without the rect', () => {
    assert.deepStrictEqual(
      layout.panels.filter((p) => p.row === 1 && p.col > 0),
      plain.panels.filter((p) => p.row === 1 && p.col > 0),
    )
  })
  checkNoThrow('v-rect: rows 0/3/4 unmoved', () => {
    for (const r of [0, 3, 4]) {
      assert.deepStrictEqual(
        layout.panels.filter((p) => p.row === r),
        plain.panels.filter((p) => p.row === r),
      )
    }
  })
  checkNoThrow('v-rect: deterministic', () => {
    assert.deepStrictEqual(solveLayout(withRect), solveLayout(withRect))
  })

  // Every cell is owned exactly once.
  {
    const seen = new Set()
    let dup = false
    for (const p of layout.panels) for (const [r, c] of p.cells) {
      if (seen.has(`${r},${c}`)) dup = true
      seen.add(`${r},${c}`)
    }
    check('v-rect: all 30 cells owned exactly once', seen.size === 30 && !dup, `${seen.size} cells`)
  }
}

// =============================================================================
// 5. Presets
// =============================================================================
{
  const cfg = buildPreset('wave')
  let layout
  checkNoThrow('wave preset: solves without error', () => {
    layout = solveLayout(cfg)
  })
  check('wave preset: 6·5 − 2 rects = 28 panels', layout.panels.length === 28, `got ${layout.panels.length}`)
  check(
    'wave preset: one horizontal + one vertical rect panel',
    layout.panels.filter((p) => p.rectOrientation === 'horizontal').length === 1 &&
      layout.panels.filter((p) => p.rectOrientation === 'vertical').length === 1,
  )
  check(
    'wave preset: all 30 cells covered',
    layout.panels.reduce((n, p) => n + p.cells.length, 0) === 30,
  )
  checkNoThrow('wave preset: deterministic (JSON-identical)', () => {
    assert.strictEqual(JSON.stringify(solveLayout(cfg)), JSON.stringify(solveLayout(cfg)))
  })
  checkNoThrow('wave preset: deterministic (deep-equal)', () => {
    assert.deepStrictEqual(solveLayout(cfg), solveLayout(cfg))
  })

  for (const id of ['flat', 'calm', 'wave', 'crash']) {
    checkNoThrow(`${id} preset: solves and stays deterministic`, () => {
      const c = buildPreset(id)
      const a = solveLayout(c)
      assert.strictEqual(JSON.stringify(a), JSON.stringify(solveLayout(c)))
      assert.ok(a.panels.length > 0)
      for (const p of a.panels) {
        assert.ok(p.position.every(Number.isFinite), `${p.id} position`)
        assert.ok(p.quaternion.every(Number.isFinite), `${p.id} quaternion`)
      }
    })
  }
  for (const seed of [0, 1, 7, 42, 12345]) {
    checkNoThrow(`random(${seed}) preset: solves and stays deterministic`, () => {
      const c = buildPreset('random', seed)
      assert.strictEqual(JSON.stringify(solveLayout(c)), JSON.stringify(solveLayout(c)))
    })
  }
}

// =============================================================================
// 6. rowAnchor 'start'
// =============================================================================
{
  const cfg = cfgOf({ rowAnchor: 'start', rows: rowsOf([0, 0, 30, 0, 0]) })
  const layout = solveLayout(cfg)
  const chain = walkChain([30, -30, 30, -30, 30], 60, 2)
  check(
    "rowAnchor 'start': row 2 chain begins at X_row = 0",
    near(layout.panels.filter((p) => p.row === 2)[0].position[0], chain.centers[0][0], 1e-9),
    String(layout.panels.filter((p) => p.row === 2)[0].position[0]),
  )
  check(
    "rowAnchor 'start': flat rows still start at X_row = 0",
    near(layout.panels.filter((p) => p.row === 0)[0].position[0], 30, 1e-9),
  )
}

// =============================================================================
// Summary
// =============================================================================
console.log('')
console.log(`test-placement: ${passed} checks passed, ${failures.length} failed`)
if (failures.length > 0) {
  console.error('')
  console.error('Failures:')
  for (const f of failures) console.error(`  - ${f}`)
  process.exit(1)
}
