/**
 * tests/test-placement.mjs — headless checks for core/placement.js (v2).
 *
 * Plain node script, no test framework. Exits non-zero on any failure.
 *   node tests/test-placement.mjs
 *
 * The closed-form expectations below are computed INDEPENDENTLY inside this file
 * (a small local chain walker, hard-coded fold sequences) rather than by calling
 * back into placement.js / schema.js — that is the whole point: they are the
 * guard against sign-convention errors in the solver. The walker even advances
 * each gap along the bisector ANGLE (ψ + f/2) where the solver normalizes the
 * sum of the two direction vectors, so the two agree only if the geometry is
 * right.
 */

import assert from 'node:assert/strict'
import { DEFAULT_CONFIG, cellPitches, normalizeConfig, validateConfig } from '../src/core/schema.js'
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
const SHORE_Y = 3.7

const cfgOf = (over) => normalizeConfig({ ...DEFAULT_CONFIG, ...over })
const columnsOf = (perColumn) => perColumn.map((foldsDeg) => ({ foldsDeg: foldsDeg.slice() }))
const flatColumns = () => columnsOf([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

/**
 * Independent 2D fold-strip walker in the (Y, Z) plane.
 *
 * `segs` is a list of { length, foldAfter } — foldAfter is the signed hinge angle
 * at the joint reached after that segment (null for the last one). Returns each
 * segment's pitch, its start point and its center point.
 */
function walkColumn(segs, gap = 2) {
  let y = SHORE_Y
  let z = 0
  let psi = 0
  const out = []
  for (const seg of segs) {
    const dy = Math.sin(psi * D)
    const dz = Math.cos(psi * D)
    out.push({
      psi,
      origin: [y, z],
      center: [y + (seg.length / 2) * dy, z + (seg.length / 2) * dz],
    })
    y += seg.length * dy
    z += seg.length * dz
    if (seg.foldAfter !== null && seg.foldAfter !== undefined) {
      const bis = (psi + seg.foldAfter / 2) * D
      y += gap * Math.sin(bis)
      z += gap * Math.cos(bis)
      psi += seg.foldAfter
    }
  }
  return out
}

const squares = (folds) =>
  folds.map((f, i) => ({ length: 60, foldAfter: f })).concat([{ length: 60, foldAfter: null }])

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
      p.rowPitchDeg !== 0 ||
      p.id !== `p${p.row}_${p.col}`
    ) {
      meta = false
    }
  }
  check('flat: centers on the 62-pitch lattice (30 + 62c, 3.7, 30 + 62r)', lattice, bad)
  check('flat: all quaternions are identity', quats)
  check('flat: y is exactly 3.7 (no float noise)', ys)
  check('flat: panel metadata (type/cells/pitch/id)', meta)
  check(
    'flat: panels are emitted column-major',
    layout.panels.every((p, i) => p.col === Math.floor(i / 5) && p.row === i % 5),
    JSON.stringify(layout.panels.map((p) => p.id)),
  )
  check('flat: no tiltDeg field survives from v1', layout.panels.every((p) => p.tiltDeg === undefined))
  check('flat: no rowPlanes field survives from v1', layout.rowPlanes === undefined)

  check(
    'flat: columnChains — 6 entries, pitches all 0, 10 vertices each',
    layout.columnChains.length === 6 &&
      layout.columnChains.every(
        (ch, c) =>
          ch.col === c &&
          ch.pitchesDeg.length === 5 &&
          ch.pitchesDeg.every((p) => p === 0) &&
          ch.points.length === 10,
      ),
    JSON.stringify(layout.columnChains[0]),
  )
  check(
    'flat: columnChain points walk straight back (y 3.7, z 0/60/62/122/…)',
    nearVec(layout.columnChains[0].points[0], [3.7, 0]) &&
      nearVec(layout.columnChains[0].points[1], [3.7, 60]) &&
      nearVec(layout.columnChains[0].points[2], [3.7, 62]) &&
      nearVec(layout.columnChains[0].points[9], [3.7, 308]),
    JSON.stringify(layout.columnChains[0].points),
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
// 2. ONE folded column (column 2 = [30, -60, 60, -30]), everything else flat
// =============================================================================
{
  const columns = flatColumns()
  columns[2] = { foldsDeg: [30, -60, 60, -30] }
  const cfg = cfgOf({ columns })
  const v = validateConfig(cfg)
  check('one folded column: config validates', v.ok, JSON.stringify(v.errors))

  const layout = solveLayout(cfg)
  const flat = solveLayout(DEFAULT_CONFIG)

  check('one folded column: 30 panels', layout.panels.length === 30, `got ${layout.panels.length}`)
  check('one folded column: no warnings (chain never backtracks or dips)', layout.warnings.length === 0, JSON.stringify(layout.warnings))

  // Independent closed form for column 2's strip.
  const chain = walkColumn(squares([30, -60, 60, -30]))
  const wantPitch = [0, 30, -30, 30, 0]
  check(
    'one folded column: closed-form pitches are 0/30/-30/30/0',
    JSON.stringify(chain.map((s) => s.psi)) === JSON.stringify(wantPitch),
    JSON.stringify(chain.map((s) => s.psi)),
  )
  check(
    'one folded column: cellPitches agrees with the closed form',
    JSON.stringify(cellPitches(cfg, 2)) === JSON.stringify(wantPitch),
    JSON.stringify(cellPitches(cfg, 2)),
  )

  const col2 = layout.panels.filter((p) => p.col === 2)
  check('one folded column: 5 panels in column 2', col2.length === 5)

  let posOk = true
  let posBad = ''
  col2.forEach((p, r) => {
    const want = [154, chain[r].center[0], chain[r].center[1]]
    if (!nearVec(p.position, want, 1e-8)) {
      posOk = false
      posBad = `${p.id} ${JSON.stringify(p.position)} want ${JSON.stringify(want)}`
    }
  })
  check('one folded column: panel centers match the closed-form Y–Z chain', posOk, posBad)
  check(
    'one folded column: x stays exactly on the column lattice (154)',
    col2.every((p) => p.position[0] === 154),
    JSON.stringify(col2.map((p) => p.position[0])),
  )
  check(
    'one folded column: rowPitchDeg is the cumulative pitch',
    col2.every((p, r) => near(p.rowPitchDeg, wantPitch[r])),
    JSON.stringify(col2.map((p) => p.rowPitchDeg)),
  )

  // Quaternions are pure rotations about world +X by −ψ (a positive fold pitches
  // the next panel UP: its depth direction gains +Y).
  let qOk = true
  let qBad = ''
  col2.forEach((p, r) => {
    const half = (-wantPitch[r] * D) / 2
    const want = [Math.sin(half), 0, 0, Math.cos(half)]
    if (!nearVec(p.quaternion, want, 1e-9)) {
      qOk = false
      qBad = `${p.id} ${JSON.stringify(p.quaternion)} want ${JSON.stringify(want)}`
    }
  })
  check('one folded column: quaternions are pure X-rotations of −ψ', qOk, qBad)
  check(
    'one folded column: every quaternion y/z component is exactly zero',
    layout.panels.every((p) => p.quaternion[1] === 0 && p.quaternion[2] === 0),
  )

  // A positive fold really does go UP: row 1 is 30·sin30 = ~30cm above the shore.
  check(
    'one folded column: +30° fold lifts row 1 (y ≈ 3.7 + 0.52 + 30·sin30)',
    near(col2[1].position[1], SHORE_Y + 2 * Math.sin(15 * D) + 30 * Math.sin(30 * D), 1e-8),
    String(col2[1].position[1]),
  )

  // Chain vertices reported by the solver must match the walker's segment starts.
  check(
    'one folded column: columnChains[2] pitches match',
    JSON.stringify(layout.columnChains[2].pitchesDeg) === JSON.stringify(wantPitch),
    JSON.stringify(layout.columnChains[2].pitchesDeg),
  )
  check(
    'one folded column: columnChains[2] first vertex is the shore start (3.7, 0)',
    nearVec(layout.columnChains[2].points[0], [3.7, 0]),
    JSON.stringify(layout.columnChains[2].points[0]),
  )

  // Every OTHER column must be bit-identical to the flat layout — columns never
  // move in X and never influence each other.
  checkNoThrow('one folded column: columns 0/1/3/4/5 bit-identical to flat', () => {
    for (const c of [0, 1, 3, 4, 5]) {
      assert.deepStrictEqual(
        layout.panels.filter((p) => p.col === c),
        flat.panels.filter((p) => p.col === c),
        `column ${c} moved`,
      )
      assert.deepStrictEqual(layout.columnChains[c], flat.columnChains[c], `chain ${c} moved`)
    }
  })
  checkNoThrow('one folded column: deterministic', () => {
    assert.deepStrictEqual(solveLayout(cfg), solveLayout(cfg))
  })
}

// =============================================================================
// 3. Every column folded identically ([45,45,45,45]) — a cylinder
// =============================================================================
{
  const cfg = cfgOf({ columns: columnsOf(Array.from({ length: 6 }, () => [45, 45, 45, 45])) })
  check('cylinder: config validates', validateConfig(cfg).ok, JSON.stringify(validateConfig(cfg).errors))
  const layout = solveLayout(cfg)
  const chain = walkColumn(squares([45, 45, 45, 45]))
  const psi = [0, 45, 90, 135, 180]

  check(
    'cylinder: pitches are cumulative (0/45/90/135/180)',
    JSON.stringify(layout.columnChains[0].pitchesDeg) === JSON.stringify(psi),
    JSON.stringify(layout.columnChains[0].pitchesDeg),
  )
  let posOk = true
  let posBad = ''
  for (const p of layout.panels) {
    const want = [30 + 62 * p.col, chain[p.row].center[0], chain[p.row].center[1]]
    if (!nearVec(p.position, want, 1e-8)) {
      posOk = false
      posBad = `${p.id} ${JSON.stringify(p.position)} want ${JSON.stringify(want)}`
    }
  }
  check('cylinder: every column traces the same closed-form chain', posOk, posBad)
  check(
    'cylinder: all six chains are identical',
    layout.columnChains.every((ch) => JSON.stringify(ch.points) === JSON.stringify(layout.columnChains[0].points)),
  )
  check(
    'cylinder: ψ = 90° stands row 2 straight up (all of row 2 shares one z)',
    layout.panels.filter((p) => p.row === 2).every((p) => near(p.position[2], chain[2].center[1], 1e-8)),
  )
  check(
    'cylinder: rows past 90° warn W_CHAIN_BACKTRACK (12 panels at ψ = 135/180)',
    layout.warnings.filter((w) => w.code === 'W_CHAIN_BACKTRACK').length === 12,
    JSON.stringify(layout.warnings.map((w) => w.code)),
  )
  checkNoThrow('cylinder: deterministic', () => {
    assert.deepStrictEqual(solveLayout(cfg), solveLayout(cfg))
  })
}

// =============================================================================
// 4. Rects
// =============================================================================
// 4a. Vertical plate — 121 along the chain, joint removed, no yaw
{
  const columns = flatColumns()
  columns[2] = { foldsDeg: [0, 0, 40, 0] } // joint 1 (the removed one) stays flat
  const base = { columns }
  const without = cfgOf(base)
  const withRect = cfgOf({ ...base, rects: [{ row: 1, col: 2, orientation: 'vertical' }] })

  const v = validateConfig(withRect)
  check('v-plate: config validates (the removed joint is unfolded)', v.ok, JSON.stringify(v.errors))
  check(
    'v-plate: W_RECT_LENGTH surfaced (121 ≠ 122)',
    v.warnings.some((w) => w.code === 'W_RECT_LENGTH'),
    JSON.stringify(v.warnings),
  )

  const layout = solveLayout(withRect)
  const plain = solveLayout(without)
  check('v-plate: 29 panels', layout.panels.length === 29, `got ${layout.panels.length}`)

  const rect = layout.panels.find((p) => p.id === 'p1_2')
  check('v-plate: panel p1_2 exists', !!rect)
  check('v-plate: type 2x4', rect.type === '2x4', rect.type)
  check('v-plate: rectOrientation vertical', rect.rectOrientation === 'vertical')
  check(
    'v-plate: owns cells (1,2) and (2,2)',
    JSON.stringify(rect.cells) === JSON.stringify([[1, 2], [2, 2]]),
    JSON.stringify(rect.cells),
  )
  check('v-plate: no separate panel at (2,2)', !layout.panels.some((p) => p.id === 'p2_2'))
  check('v-plate: no yaw (quaternion is identity here)', nearVec(rect.quaternion, [0, 0, 0, 1]))

  // Independent closed form: the plate is ONE 121cm segment consuming rows 1+2,
  // so the chain advances 121 (not 60 + 2 + 60 = 122) and the joint after it is
  // joint 2 (the 40° fold).
  const chain = walkColumn([
    { length: 60, foldAfter: 0 },
    { length: 121, foldAfter: 40 },
    { length: 60, foldAfter: 0 },
    { length: 60, foldAfter: null },
  ])
  check(
    'v-plate: plate center sits 60.5 along the chain (z = 62 + 60.5)',
    nearVec(rect.position, [154, chain[1].center[0], chain[1].center[1]], 1e-8) &&
      near(rect.position[2], 122.5, 1e-8),
    JSON.stringify(rect.position),
  )
  {
    const corners = panelWorldCorners(rect, withRect)
    const zs = corners.map((c) => c[2])
    const xs = corners.map((c) => c[0])
    check(
      'v-plate: 121 length runs along the chain (z = 62..183), 60 across the column',
      near(Math.min(...zs), 62, 1e-8) &&
        near(Math.max(...zs), 183, 1e-8) &&
        near(Math.max(...xs) - Math.min(...xs), 60, 1e-8),
      JSON.stringify([Math.min(...zs), Math.max(...zs), Math.max(...xs) - Math.min(...xs)]),
    )
  }
  const behind = layout.panels.filter((p) => p.col === 2 && p.row > 2)
  check(
    'v-plate: rows behind it follow the shortened chain (1cm of plate slack)',
    behind.every((p, i) => nearVec(p.position, [154, chain[2 + i].center[0], chain[2 + i].center[1]], 1e-8)),
    JSON.stringify(behind.map((p) => p.position)),
  )
  check(
    'v-plate: rows behind it also pick up the 40° fold',
    behind.every((p) => near(p.rowPitchDeg, 40)),
    JSON.stringify(behind.map((p) => p.rowPitchDeg)),
  )
  checkNoThrow('v-plate: every other column untouched', () => {
    for (const c of [0, 1, 3, 4, 5]) {
      assert.deepStrictEqual(
        layout.panels.filter((p) => p.col === c),
        plain.panels.filter((p) => p.col === c),
      )
    }
  })
  checkNoThrow('v-plate: deterministic', () => {
    assert.deepStrictEqual(solveLayout(withRect), solveLayout(withRect))
  })

  // Every cell owned exactly once.
  {
    const seen = new Set()
    let dup = false
    for (const p of layout.panels) for (const [r, c] of p.cells) {
      if (seen.has(`${r},${c}`)) dup = true
      seen.add(`${r},${c}`)
    }
    check('v-plate: all 30 cells owned exactly once', seen.size === 30 && !dup, `${seen.size} cells`)
  }
}

// 4b. Horizontal plate — yawed, centered on the two-column slot, phantom neighbour
{
  const base = { columns: flatColumns() }
  const without = cfgOf(base)
  const withRect = cfgOf({ ...base, rects: [{ row: 2, col: 1, orientation: 'horizontal' }] })

  const v = validateConfig(withRect)
  check('h-plate: config validates', v.ok, JSON.stringify(v.errors))

  const layout = solveLayout(withRect)
  const plain = solveLayout(without)
  check('h-plate: 29 panels', layout.panels.length === 29, `got ${layout.panels.length}`)

  const rect = layout.panels.find((p) => p.id === 'p2_1')
  check('h-plate: panel p2_1 exists', !!rect)
  check('h-plate: type 2x4', rect.type === '2x4', rect.type)
  check('h-plate: rectOrientation horizontal', rect.rectOrientation === 'horizontal')
  check(
    'h-plate: owns cells (2,1) and (2,2)',
    JSON.stringify(rect.cells) === JSON.stringify([[2, 1], [2, 2]]),
    JSON.stringify(rect.cells),
  )
  check('h-plate: no separate panel at (2,2)', !layout.panels.some((p) => p.id === 'p2_2'))

  const s = Math.sin(Math.PI / 4)
  check(
    'h-plate: quaternion carries the +90° yaw about local Y',
    nearVec(rect.quaternion, [0, s, 0, s]),
    JSON.stringify(rect.quaternion),
  )
  check(
    'h-plate: x center is the two-column slot center (c·62 + 61 = 123)',
    rect.position[0] === 123,
    String(rect.position[0]),
  )
  check(
    'h-plate: (y, z) come from the owning column (flat here: 3.7, 154)',
    nearVec([rect.position[1], rect.position[2]], [3.7, 154], 1e-9),
    JSON.stringify(rect.position),
  )

  const corners = panelWorldCorners(rect, withRect)
  const xs = corners.map((c) => c[0])
  const zs = corners.map((c) => c[2])
  check(
    'h-plate: world footprint spans 121 across the columns (+X), 0.5 shy of each slot edge',
    near(Math.min(...xs), 62.5, 1e-8) && near(Math.max(...xs), 183.5, 1e-8),
    JSON.stringify([Math.min(...xs), Math.max(...xs)]),
  )
  check(
    'h-plate: world footprint spans 60 along the chain (+Z)',
    near(Math.max(...zs) - Math.min(...zs), 60, 1e-8) && near(Math.min(...zs), 124, 1e-8),
    JSON.stringify([Math.min(...zs), Math.max(...zs)]),
  )

  // The phantom cell advances column 2's chain exactly as a square would, so
  // every other panel in column 2 is bit-identical to the plate-free layout.
  checkNoThrow('h-plate: the phantom leaves column 2 bit-identical elsewhere', () => {
    assert.deepStrictEqual(
      layout.panels.filter((p) => p.col === 2),
      plain.panels.filter((p) => p.col === 2 && p.row !== 2),
    )
    assert.deepStrictEqual(layout.columnChains[2], plain.columnChains[2])
  })
  checkNoThrow('h-plate: columns 0/3/4/5 untouched', () => {
    for (const c of [0, 3, 4, 5]) {
      assert.deepStrictEqual(
        layout.panels.filter((p) => p.col === c),
        plain.panels.filter((p) => p.col === c),
      )
    }
  })
  checkNoThrow('h-plate: deterministic', () => {
    assert.deepStrictEqual(solveLayout(withRect), solveLayout(withRect))
  })
  {
    const seen = new Set()
    for (const p of layout.panels) for (const [r, c] of p.cells) seen.add(`${r},${c}`)
    check('h-plate: all 30 cells owned exactly once', seen.size === 30, `${seen.size} cells`)
  }
}

// =============================================================================
// 5. Floor support
// =============================================================================
{
  // A column that starts by folding DOWN drives its whole strip under the floor.
  const columns = flatColumns()
  columns[0] = { foldsDeg: [-30, 0, 0, 0] }
  const layout = solveLayout(cfgOf({ columns }))
  const below = layout.warnings.find((w) => w.code === 'W_BELOW_FLOOR')
  check('below floor: W_BELOW_FLOOR raised', !!below, JSON.stringify(layout.warnings))
  check(
    'below floor: the warning names the four sunken panels of column 0',
    below && ['p1_0', 'p2_0', 'p3_0', 'p4_0'].every((id) => below.message.includes(id)) && !below.message.includes('p0_0'),
    below && below.message,
  )
  check('flat layout raises no W_BELOW_FLOOR', !solveLayout(DEFAULT_CONFIG).warnings.some((w) => w.code === 'W_BELOW_FLOOR'))
}

// =============================================================================
// 6. Presets
// =============================================================================
{
  const cfg = buildPreset('wave')
  let layout
  checkNoThrow('wave preset: solves without error', () => {
    layout = solveLayout(cfg)
  })
  check('wave preset: 6·5 − 2 rects = 28 panels', layout.panels.length === 28, `got ${layout.panels.length}`)
  check(
    'wave preset: one horizontal + one vertical plate panel',
    layout.panels.filter((p) => p.rectOrientation === 'horizontal').length === 1 &&
      layout.panels.filter((p) => p.rectOrientation === 'vertical').length === 1,
  )
  check(
    'wave preset: all 30 cells covered',
    layout.panels.reduce((n, p) => n + p.cells.length, 0) === 30,
  )
  check(
    'wave preset: every column stays on its own x lattice slot',
    layout.panels.every(
      (p) => p.position[0] === (p.rectOrientation === 'horizontal' ? 62 * p.col + 61 : 62 * p.col + 30),
    ),
    JSON.stringify(layout.panels.map((p) => [p.id, p.position[0]])),
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
