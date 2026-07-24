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
import {
  columnEndGrounding,
  panelSolidCorners,
  panelSolidMinY,
  panelWorldCorners,
  solveLayout,
} from '../src/core/placement.js'
import {
  groundAllFolds,
  lastSurvivingJoint,
  solveGroundingFold,
  survivingJoints,
} from '../src/core/ground.js'

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
// 7. The grounded-end rule — every column's LAST panel must touch the floor
// =============================================================================
// 7a. The panel SOLID: 8 corners, the housing 3.7cm behind the lit face
{
  const layout = solveLayout(DEFAULT_CONFIG)
  const corners = panelSolidCorners(layout.panels[0], DEFAULT_CONFIG)
  check('solid: 8 corners', corners.length === 8, String(corners.length))
  check(
    'solid: the first 4 are the lit face (y = 3.7)',
    corners.slice(0, 4).every((p) => p[1] === 3.7),
    JSON.stringify(corners.slice(0, 4)),
  )
  check(
    'solid: the back 4 rest exactly on the floor (y = 0, x/z unchanged)',
    corners.slice(4).every((p, i) => p[1] === 0 && p[0] === corners[i][0] && p[2] === corners[i][2]),
    JSON.stringify(corners.slice(4)),
  )
  check(
    'solid: panelSolidMinY of a flat panel is exactly 0',
    panelSolidMinY(layout.panels[0], DEFAULT_CONFIG) === 0,
    String(panelSolidMinY(layout.panels[0], DEFAULT_CONFIG)),
  )
}

// 7b. Flat: every column grounded with zero clearance, no violations
{
  const layout = solveLayout(DEFAULT_CONFIG)
  check('grounded: flat raises no violations', layout.violations.length === 0, JSON.stringify(layout.violations))
  check(
    'grounded: flat marks all 6 chains grounded at clearance 0',
    layout.columnChains.every((ch) => ch.grounded === true && ch.endClearanceCm === 0),
    JSON.stringify(layout.columnChains.map((ch) => [ch.grounded, ch.endClearanceCm])),
  )
  check(
    'grounded: DEFAULT_CONFIG carries groundTolerance 0.5',
    DEFAULT_CONFIG.groundTolerance === 0.5,
    String(DEFAULT_CONFIG.groundTolerance),
  )
}

// 7c. A floating column — one 45° fold at the front lifts the whole strip
{
  const columns = flatColumns()
  columns[0] = { foldsDeg: [45, 0, 0, 0] }
  const cfg = cfgOf({ columns })
  check('floating: config validates', validateConfig(cfg).ok, JSON.stringify(validateConfig(cfg).errors))
  const layout = solveLayout(cfg)

  // Closed form: the chain start of row 4 is 3.7 + 2·sin22.5 + 3·(60+2)·sin45,
  // the panel is flat at ψ = 45°, so its lowest solid corner is that start point
  // minus the housing's own drop of 3.7·cos45.
  const y4 = SHORE_Y + 2 * Math.sin(22.5 * D) + 3 * 62 * Math.sin(45 * D)
  const wantClearance = y4 - 3.7 * Math.cos(45 * D)

  check(
    'floating: exactly one E_END_FLOATING, for column 0',
    layout.violations.length === 1 &&
      layout.violations[0].code === 'E_END_FLOATING' &&
      layout.violations[0].col === 0,
    JSON.stringify(layout.violations.map((v) => [v.code, v.col])),
  )
  check(
    'floating: clearance matches the closed form (≈ 133.4cm)',
    near(layout.violations[0].clearanceCm, wantClearance, 1e-6),
    `${layout.violations[0].clearanceCm} vs ${wantClearance}`,
  )
  check(
    'floating: the message names the panel and the rule',
    /p4_0/.test(layout.violations[0].message) && /returns to the water/.test(layout.violations[0].message),
    layout.violations[0].message,
  )
  check(
    'floating: only column 0 is marked, the flat five stay grounded',
    layout.columnChains[0].grounded === false &&
      layout.columnChains.slice(1).every((ch) => ch.grounded === true),
    JSON.stringify(layout.columnChains.map((ch) => ch.grounded)),
  )
  check(
    'floating: columnChains[0].endClearanceCm equals the violation clearance',
    layout.columnChains[0].endClearanceCm === layout.violations[0].clearanceCm,
    `${layout.columnChains[0].endClearanceCm} vs ${layout.violations[0].clearanceCm}`,
  )
  check(
    'floating: the chain never dips below the floor, so the two rules are independent',
    !layout.warnings.some((w) => w.code === 'W_BELOW_FLOOR') && layout.violations.length === 1,
    JSON.stringify(layout.warnings.map((w) => w.code)),
  )
}

// 7d. Edge contact — a TILTED last panel resting on one housing edge
{
  // Pitch profile [0, 10, 10, -10, -10]: the sines cancel, so the chain comes back
  // to shore height, and the last panel arrives pitched 10° DOWN. Its lit face is
  // still ~3.9cm up; what touches the floor is the low housing edge.
  const columns = flatColumns()
  columns[3] = { foldsDeg: [10, 0, -20, 0] }
  const cfg = cfgOf({ columns })
  const layout = solveLayout(cfg)
  const end = layout.panels.filter((p) => p.col === 3).at(-1)
  const litMin = panelWorldCorners(end, cfg).reduce((m, p) => Math.min(m, p[1]), Infinity)

  check(
    'edge contact: the last panel is genuinely tilted (ψ = −10°)',
    near(end.rowPitchDeg, -10),
    String(end.rowPitchDeg),
  )
  // The four 60cm rises and falls cancel exactly, and so do the four gap advances
  // along the bisectors ±5°/±10°/0° — except the very first, leaving 3.7 + 2·sin5°.
  check(
    'edge contact: the LIT FACE stays clear of the floor (3.7 + 2·sin5° ≈ 3.87cm)',
    near(litMin, SHORE_Y + 2 * Math.sin(5 * D), 1e-6),
    `${litMin} vs ${SHORE_Y + 2 * Math.sin(5 * D)}`,
  )
  check(
    'edge contact: the housing edge reaches the floor — grounded, no violation',
    layout.columnChains[3].grounded === true &&
      layout.columnChains[3].endClearanceCm > 0 &&
      layout.columnChains[3].endClearanceCm <= 0.5 &&
      layout.violations.length === 0,
    `clearance=${layout.columnChains[3].endClearanceCm} violations=${JSON.stringify(layout.violations)}`,
  )
  check(
    'edge contact: closed form — clearance = litMin − 3.7·cos10°',
    near(layout.columnChains[3].endClearanceCm, litMin - 3.7 * Math.cos(10 * D), 1e-8),
    `${layout.columnChains[3].endClearanceCm} vs ${litMin - 3.7 * Math.cos(10 * D)}`,
  )
}

// 7e. groundTolerance is honoured
{
  const columns = flatColumns()
  columns[0] = { foldsDeg: [10, 0, -20, 0] } // clearance ≈ 0.231cm
  const loose = solveLayout(cfgOf({ columns, groundTolerance: 0.5 }))
  const tight = solveLayout(cfgOf({ columns, groundTolerance: 0.1 }))
  check('groundTolerance: 0.231cm passes at 0.5', loose.violations.length === 0)
  check(
    'groundTolerance: the same design fails at 0.1',
    tight.violations.length === 1 && tight.violations[0].col === 0,
    JSON.stringify(tight.violations),
  )
}

// =============================================================================
// 8. The auto-ground solver (core/ground.js)
// =============================================================================
// 8a. Which hinge gets solved
{
  const cfg = cfgOf({ columns: flatColumns() })
  check('solver: a plain column\'s last hinge is joint 3', lastSurvivingJoint(cfg, 0) === 3, String(lastSurvivingJoint(cfg, 0)))
  check(
    'solver: surviving joints of a plain column are 0..3',
    JSON.stringify(survivingJoints(cfg, 0)) === JSON.stringify([0, 1, 2, 3]),
    JSON.stringify(survivingJoints(cfg, 0)),
  )
  const withEndPlate = cfgOf({
    columns: flatColumns(),
    rects: [{ row: 3, col: 1, orientation: 'vertical' }],
  })
  check(
    'solver: a vertical plate over rows 3+4 removes joint 3 — the last hinge is joint 2',
    lastSurvivingJoint(withEndPlate, 1) === 2 &&
      JSON.stringify(survivingJoints(withEndPlate, 1)) === JSON.stringify([0, 1, 2]),
    JSON.stringify(survivingJoints(withEndPlate, 1)),
  )
}

// 8b. A reachable floating column: solve it, apply it, re-solve, assert grounded
{
  const columns = flatColumns()
  columns[0] = { foldsDeg: [30, -30, 0, 0] } // the strip parks 31cm up and stays flat
  const cfg = cfgOf({ columns })
  const before = solveLayout(cfg)
  check(
    'solver: the test case really does float (≈ 31cm)',
    before.violations.length === 1 && before.violations[0].clearanceCm > 30,
    JSON.stringify(before.violations.map((v) => v.clearanceCm)),
  )

  const solved = solveGroundingFold(cfg, 0)
  check('solver: returns a solution', !!solved, JSON.stringify(solved))
  check('solver: it is the last hinge (joint 3)', solved?.jointIndex === 3, String(solved?.jointIndex))
  check(
    'solver: the solved fold is a sane downward angle inside the clamp',
    solved && solved.foldDeg < 0 && Math.abs(solved.foldDeg) <= 120,
    String(solved?.foldDeg),
  )
  check(
    'solver: it snapped to a readable angle (whole degrees here)',
    Number.isInteger(solved.foldDeg),
    String(solved.foldDeg),
  )

  const applied = cfgOf({
    columns: columns.map((col, i) =>
      i === 0 ? { foldsDeg: col.foldsDeg.map((f, k) => (k === solved.jointIndex ? solved.foldDeg : f)) } : col,
    ),
  })
  const after = solveLayout(applied)
  check('solver: applying it validates', validateConfig(applied).ok, JSON.stringify(validateConfig(applied).errors))
  check(
    'solver: applying it grounds the column (re-solve has no violations)',
    after.violations.length === 0 && after.columnChains[0].grounded === true,
    JSON.stringify(after.violations),
  )
  check(
    'solver: the reported clearance matches the re-solve',
    near(solved.clearanceCm, after.columnChains[0].endClearanceCm, 1e-9),
    `${solved.clearanceCm} vs ${after.columnChains[0].endClearanceCm}`,
  )
  check(
    'solver: the landing sits ON the floor, not through it',
    after.columnChains[0].endClearanceCm >= 0 &&
      after.columnChains[0].endClearanceCm <= 0.5 &&
      !after.warnings.some((w) => w.code === 'W_BELOW_FLOOR'),
    String(after.columnChains[0].endClearanceCm),
  )
  check(
    'solver: only the solved hinge changed',
    JSON.stringify(applied.columns[0].foldsDeg.slice(0, 3)) === JSON.stringify([30, -30, 0]),
    JSON.stringify(applied.columns[0].foldsDeg),
  )
  checkNoThrow('solver: pure — the input config is untouched', () => {
    assert.deepStrictEqual(cfg.columns[0].foldsDeg, [30, -30, 0, 0])
  })
  checkNoThrow('solver: deterministic', () => {
    assert.deepStrictEqual(solveGroundingFold(cfg, 0), solveGroundingFold(cfg, 0))
  })
}

// 8c. An UNREACHABLE end — the chain is a whole panel-length too high
{
  const columns = flatColumns()
  columns[0] = { foldsDeg: [45, 0, 0, 0] } // last vertex ≈ 136cm up; the panel is 60 long
  const cfg = cfgOf({ columns })
  check('unreachable: solveGroundingFold returns null', solveGroundingFold(cfg, 0) === null)

  // Prove it by brute force: neither extreme of the last hinge reaches the floor.
  let bestClearance = Infinity
  for (let f = -120; f <= 120; f += 1) {
    const trial = cfgOf({
      columns: columns.map((col, i) => (i === 0 ? { foldsDeg: [45, 0, 0, f] } : col)),
    })
    bestClearance = Math.min(bestClearance, columnEndGrounding(trial, 0).clearanceCm)
  }
  check(
    'unreachable: NO hinge angle in ±120° gets within tolerance (best ≈ 73cm)',
    bestClearance > 0.5,
    `best clearance ${bestClearance.toFixed(2)}cm`,
  )
  check(
    'unreachable: the column still floats at both extremes',
    columnEndGrounding(cfgOf({ columns: [{ foldsDeg: [45, 0, 0, -120] }, ...columns.slice(1)] }), 0).grounded === false &&
      columnEndGrounding(cfgOf({ columns: [{ foldsDeg: [45, 0, 0, 120] }, ...columns.slice(1)] }), 0).grounded === false,
  )
  check('unreachable: groundAllFolds gives up and returns null', groundAllFolds(cfg) === null)
}

// 8d. groundAllFolds on several floating columns at once
{
  const columns = columnsOf([
    [30, -30, 0, 0], // floats ≈ 31cm
    [0, 0, 0, 0], // already grounded
    [20, 0, -20, 0], // floats a little
    [10, 10, -20, 0], // floats
    [0, 0, 0, 0], // already grounded
    [30, -20, -10, 0], // floats high but still within reach
  ])
  const cfg = cfgOf({ columns })
  const before = solveLayout(cfg)
  check(
    'groundAll: 4 of the 6 columns start floating',
    before.violations.length === 4 &&
      JSON.stringify(before.violations.map((v) => v.col)) === JSON.stringify([0, 2, 3, 5]),
    JSON.stringify(before.violations.map((v) => [v.col, v.clearanceCm])),
  )

  const next = groundAllFolds(cfg)
  check('groundAll: returns a new config', !!next && next !== cfg)
  const after = solveLayout(next)
  check('groundAll: the result validates', validateConfig(next).ok, JSON.stringify(validateConfig(next).errors))
  check(
    'groundAll: every column is grounded afterwards',
    after.violations.length === 0 && after.columnChains.every((ch) => ch.grounded),
    JSON.stringify(after.violations),
  )
  check(
    'groundAll: the two already-grounded columns were left byte-identical',
    JSON.stringify(next.columns[1].foldsDeg) === JSON.stringify([0, 0, 0, 0]) &&
      JSON.stringify(next.columns[4].foldsDeg) === JSON.stringify([0, 0, 0, 0]),
    JSON.stringify(next.columns.map((c) => c.foldsDeg)),
  )
  check(
    'groundAll: only the last hinge of each floating column moved',
    [0, 2, 3, 5].every(
      (c) =>
        JSON.stringify(next.columns[c].foldsDeg.slice(0, 3)) ===
        JSON.stringify(columns[c].foldsDeg.slice(0, 3)),
    ),
    JSON.stringify(next.columns.map((c) => c.foldsDeg)),
  )
  check(
    'groundAll: nothing ends up under the floor',
    !after.warnings.some((w) => w.code === 'W_BELOW_FLOOR'),
    JSON.stringify(after.warnings.map((w) => w.code)),
  )
  check('groundAll: an already-grounded config returns null', groundAllFolds(DEFAULT_CONFIG) === null)
  checkNoThrow('groundAll: pure — the input is untouched', () => {
    assert.deepStrictEqual(cfg.columns[0].foldsDeg, [30, -30, 0, 0])
  })
  checkNoThrow('groundAll: deterministic', () => {
    assert.deepStrictEqual(groundAllFolds(cfg), groundAllFolds(cfg))
  })
}

// 8e. A vertical plate at the END: the plate itself is the panel to ground, and
//     the hinge that moves is joint 2 (joint 3 is interior to the plate)
{
  const columns = flatColumns()
  columns[2] = { foldsDeg: [40, -40, 20, 0] } // joint 3 must stay 0 under the plate
  const cfg = cfgOf({ columns, rects: [{ row: 3, col: 2, orientation: 'vertical' }] })
  check('v-plate end: config validates', validateConfig(cfg).ok, JSON.stringify(validateConfig(cfg).errors))

  const before = solveLayout(cfg)
  const endPanel = before.panels.filter((p) => p.col === 2).at(-1)
  check(
    'v-plate end: the last panel IS the 121cm plate spanning rows 3+4',
    endPanel.id === 'p3_2' && endPanel.type === '2x4' && endPanel.rectOrientation === 'vertical',
    JSON.stringify([endPanel.id, endPanel.type]),
  )
  check(
    'v-plate end: it floats',
    before.violations.length === 1 && before.violations[0].col === 2,
    JSON.stringify(before.violations),
  )

  const solved = solveGroundingFold(cfg, 2)
  check('v-plate end: solver returns a solution', !!solved, JSON.stringify(solved))
  check(
    'v-plate end: it moved joint 2, NOT the joint the plate removed',
    solved?.jointIndex === 2,
    String(solved?.jointIndex),
  )
  const next = groundAllFolds(cfg)
  const after = solveLayout(next)
  check(
    'v-plate end: grounding it keeps foldsDeg[3] at 0 (the plate is still rigid)',
    next.columns[2].foldsDeg[3] === 0 && validateConfig(next).ok,
    JSON.stringify(next.columns[2].foldsDeg),
  )
  check(
    'v-plate end: the plate now touches the floor',
    after.violations.length === 0 && after.columnChains[2].grounded === true,
    `clearance=${after.columnChains[2].endClearanceCm} violations=${JSON.stringify(after.violations)}`,
  )
  check(
    'v-plate end: it is still one 121cm plate (29 panels)',
    after.panels.length === 29 && after.panels.find((p) => p.id === 'p3_2').type === '2x4',
    String(after.panels.length),
  )
}

// 8f. Presets satisfy the rule
{
  for (const id of ['flat', 'calm', 'wave', 'crash']) {
    const cfg = buildPreset(id)
    const layout = solveLayout(cfg)
    check(
      `${id} preset: zero E_END_FLOATING violations`,
      layout.violations.length === 0,
      JSON.stringify(layout.violations.map((v) => [v.col, v.clearanceCm])),
    )
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
