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
import {
  DEFAULT_CONFIG,
  DEFAULT_GAP,
  WALL_COLUMN,
  cellPitches,
  chainStartY,
  columnChain,
  frontRestY,
  normalizeConfig,
  validateConfig,
} from '../src/core/schema.js'
import { buildPreset } from '../src/core/presets.js'
import { PANEL_PROFILE } from '../src/config.js'
import {
  columnEndGrounding,
  layoutBounds,
  panelSolidCorners,
  panelSolidMinY,
  panelWorldCorners,
  round9,
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
/**
 * Geometry the expectations below are built from — DERIVED from the config so a
 * future change to the default gap moves them all at once. At today's defaults:
 * gap 1, cell 60, so the cell pitch is 61 and 2·60 + gap = 121 = rectLength
 * exactly (a rect plate is a drop-in for two squares plus their joint).
 */
const GAP = DEFAULT_CONFIG.gap
const SIZE = DEFAULT_CONFIG.cell.size
const PITCH = SIZE + GAP
const RECT_LENGTH = DEFAULT_CONFIG.cell.rectLength
/**
 * How close "the front panel rests exactly on the floor" has to be, cm.
 *
 * The rule is EXACT in the closed form — `frontRestY` puts the front panel's solid
 * min y at 0 by construction — but the measured value comes back through two
 * deterministic roundings: the panel position and quaternion are snapped to 1e-9,
 * and the quaternion is re-normalized on read. Over a 121cm panel that leaves a few
 * times 1e-8 of a centimetre — i.e. a fraction of a nanometre of real geometry.
 */
const FRONT_REST_EPS = 1e-7

const cfgOf = (over) => normalizeConfig({ ...DEFAULT_CONFIG, ...over })
const columnsOf = (perColumn) => perColumn.map((foldsDeg) => ({ foldsDeg: foldsDeg.slice() }))
const flatColumns = () => columnsOf([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

/**
 * Independent 2D fold-strip walker in the (Y, Z) plane.
 *
 * `segs` is a list of { length, foldAfter } — foldAfter is the signed hinge angle
 * at the joint reached after that segment (null for the last one). Returns each
 * segment's pitch, its start point and its center point.
 *
 * `startPitch` is the FRONT panel's pitch (`columns[c].startPitchDeg`) and the
 * start height is derived from it here INDEPENDENTLY of schema.js's `frontRestY`:
 * 3.7·cos θ once the front pitches up (its front housing edge rests), lifted by the
 * panel's own dive L·sin|θ| once it pitches down (its rear edge rests). At θ = 0
 * that is exactly the historical 3.7.
 */
function walkColumn(segs, gap = GAP, startPitch = 0) {
  const front = segs.length > 0 ? segs[0].length : 60
  let y =
    Math.max(0, SHORE_Y * Math.cos(startPitch * D)) -
    Math.min(0, front * Math.sin(startPitch * D))
  let z = 0
  let psi = startPitch
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
  check(
    'flat: the default gap is 1cm, so the cell pitch is 61 and 2·60 + gap = 121 = rectLength',
    GAP === 1 && PITCH === 61 && 2 * SIZE + GAP === RECT_LENGTH,
    `gap=${GAP} pitch=${PITCH} 2·size+gap=${2 * SIZE + GAP} rectLength=${RECT_LENGTH}`,
  )

  let lattice = true
  let quats = true
  let ys = true
  let meta = true
  let bad = ''
  for (const p of layout.panels) {
    const want = [30 + 61 * p.col, 3.7, 30 + 61 * p.row]
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
  check('flat: centers on the 61-pitch lattice (30 + 61c, 3.7, 30 + 61r)', lattice, bad)
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
    'flat: columnChain points walk straight back (y 3.7, z 0/60/61/121/…/304)',
    nearVec(layout.columnChains[0].points[0], [3.7, 0]) &&
      nearVec(layout.columnChains[0].points[1], [3.7, 60]) &&
      nearVec(layout.columnChains[0].points[2], [3.7, 61]) &&
      nearVec(layout.columnChains[0].points[9], [3.7, 304]),
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
    const want = [152, chain[r].center[0], chain[r].center[1]]
    if (!nearVec(p.position, want, 1e-8)) {
      posOk = false
      posBad = `${p.id} ${JSON.stringify(p.position)} want ${JSON.stringify(want)}`
    }
  })
  check('one folded column: panel centers match the closed-form Y–Z chain', posOk, posBad)
  check(
    'one folded column: x stays exactly on the column lattice (2·61 + 30 = 152)',
    col2.every((p) => p.position[0] === 152),
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
    'one folded column: +30° fold lifts row 1 (y = 3.7 + gap·sin15 + 30·sin30)',
    near(col2[1].position[1], SHORE_Y + GAP * Math.sin(15 * D) + 30 * Math.sin(30 * D), 1e-8),
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
    const want = [30 + 61 * p.col, chain[p.row].center[0], chain[p.row].center[1]]
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
    'v-plate: NO W_RECT_LENGTH at the defaults (121 = 2·60 + gap 1, an exact fit)',
    !v.warnings.some((w) => w.code === 'W_RECT_LENGTH'),
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

  // Independent closed form: the plate is ONE 121cm segment consuming rows 1+2, so
  // the chain advances 121 — which at the default gap is exactly what two squares
  // and the joint between them would have advanced (60 + 1 + 60 = 121). The joint
  // after the plate is joint 2 (the 40° fold).
  check(
    'v-plate: the plate spans exactly two squares plus their joint',
    RECT_LENGTH === 2 * SIZE + GAP,
    `${RECT_LENGTH} vs ${2 * SIZE + GAP}`,
  )
  const chain = walkColumn([
    { length: 60, foldAfter: 0 },
    { length: RECT_LENGTH, foldAfter: 40 },
    { length: 60, foldAfter: 0 },
    { length: 60, foldAfter: null },
  ])
  check(
    'v-plate: plate center sits 60.5 along the chain (z = 61 + 60.5 = 121.5)',
    nearVec(rect.position, [152, chain[1].center[0], chain[1].center[1]], 1e-8) &&
      near(rect.position[2], 121.5, 1e-8),
    JSON.stringify(rect.position),
  )
  {
    const corners = panelWorldCorners(rect, withRect)
    const zs = corners.map((c) => c[2])
    const xs = corners.map((c) => c[0])
    check(
      'v-plate: 121 length runs along the chain (z = 61..182), 60 across the column',
      near(Math.min(...zs), 61, 1e-8) &&
        near(Math.max(...zs), 182, 1e-8) &&
        near(Math.max(...xs) - Math.min(...xs), 60, 1e-8),
      JSON.stringify([Math.min(...zs), Math.max(...zs), Math.max(...xs) - Math.min(...xs)]),
    )
  }
  const behind = layout.panels.filter((p) => p.col === 2 && p.row > 2)
  check(
    'v-plate: rows behind it follow the closed-form chain',
    behind.every((p, i) => nearVec(p.position, [152, chain[2 + i].center[0], chain[2 + i].center[1]], 1e-8)),
    JSON.stringify(behind.map((p) => p.position)),
  )
  // THE DROP-IN PROPERTY: 121 = 60 + gap + 60 exactly, so swapping two squares for
  // the plate leaves everything behind it bit-identical — the 1cm of slack the old
  // 2cm gap propagated down the rest of the column is gone.
  checkNoThrow('v-plate: rows behind it are BIT-IDENTICAL to the plate-free column', () => {
    assert.deepStrictEqual(
      layout.panels.filter((p) => p.col === 2 && p.row > 2),
      plain.panels.filter((p) => p.col === 2 && p.row > 2),
    )
  })
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
    'h-plate: x center is the two-column slot center (c·61 + 60.5 = 121.5)',
    rect.position[0] === 121.5,
    String(rect.position[0]),
  )
  check(
    'h-plate: (y, z) come from the owning column (flat here: 3.7, 2·61 + 30 = 152)',
    nearVec([rect.position[1], rect.position[2]], [3.7, 152], 1e-9),
    JSON.stringify(rect.position),
  )

  const corners = panelWorldCorners(rect, withRect)
  const xs = corners.map((c) => c[0])
  const zs = corners.map((c) => c[2])
  // ZERO SLACK: the 121cm plate fills its 2·60 + gap = 121cm slot exactly, so its
  // x extent runs from the LEFT column's outer edge to the RIGHT column's outer
  // edge — derived from the config so it cannot drift out of sync with the gap.
  const RECT_COL = 1
  const slotX0 = RECT_COL * PITCH // left column's outer (low-x) edge
  const slotX1 = (RECT_COL + 1) * PITCH + SIZE // right column's outer (high-x) edge
  check(
    'h-plate: x extent spans exactly the left column outer edge → right column outer edge (61..182)',
    near(Math.min(...xs), slotX0, 1e-8) &&
      near(Math.max(...xs), slotX1, 1e-8) &&
      near(slotX1 - slotX0, RECT_LENGTH, 1e-8) &&
      near(Math.min(...xs), 61, 1e-8) &&
      near(Math.max(...xs), 182, 1e-8),
    `plate x [${Math.min(...xs)}, ${Math.max(...xs)}] vs slot [${slotX0}, ${slotX1}] (${RECT_LENGTH}cm plate)`,
  )
  check(
    'h-plate: world footprint spans 60 along the chain (+Z)',
    near(Math.max(...zs) - Math.min(...zs), 60, 1e-8) && near(Math.min(...zs), 122, 1e-8),
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
  // 'crest plates': one vertical plate per column, so every plate fuses two cells
  // into one panel — 30 cells − 6 plates = 24 panels.
  check(
    'wave preset: 6·5 − 6 rects = 24 panels',
    layout.panels.length === 24 && layout.panels.length === 30 - cfg.rects.length,
    `got ${layout.panels.length}, ${cfg.rects.length} rects`,
  )
  check(
    'wave preset: six vertical plate panels, no horizontal one',
    layout.panels.filter((p) => p.rectOrientation === 'vertical').length === 6 &&
      layout.panels.filter((p) => p.rectOrientation === 'horizontal').length === 0,
    JSON.stringify(layout.panels.filter((p) => p.rectOrientation).map((p) => [p.id, p.rectOrientation])),
  )
  check(
    'wave preset: all 30 cells covered',
    layout.panels.reduce((n, p) => n + p.cells.length, 0) === 30,
  )
  check(
    'wave preset: every column stays on its own x lattice slot (pitch derived from the config)',
    layout.panels.every(
      (p) =>
        p.position[0] ===
        (p.rectOrientation === 'horizontal'
          ? PITCH * p.col + (2 * SIZE + GAP) / 2
          : PITCH * p.col + SIZE / 2),
    ),
    JSON.stringify(layout.panels.map((p) => [p.id, p.position[0]])),
  )
  checkNoThrow('wave preset: deterministic (JSON-identical)', () => {
    assert.strictEqual(JSON.stringify(solveLayout(cfg)), JSON.stringify(solveLayout(cfg)))
  })
  checkNoThrow('wave preset: deterministic (deep-equal)', () => {
    assert.deepStrictEqual(solveLayout(cfg), solveLayout(cfg))
  })

  // 'mirrored pairs' is the preset that exercises the HORIZONTAL plate placement —
  // four of them, each yawed across two columns and centered on the 121cm slot.
  const calm = buildPreset('calm')
  const calmLayout = solveLayout(calm)
  check(
    'calm preset: four horizontal plate panels, 26 panels in total',
    calmLayout.panels.filter((p) => p.rectOrientation === 'horizontal').length === 4 &&
      calmLayout.panels.filter((p) => p.rectOrientation === 'vertical').length === 0 &&
      calmLayout.panels.length === 26,
    `${calmLayout.panels.length} panels, plates=${JSON.stringify(calmLayout.panels.filter((p) => p.rectOrientation).map((p) => p.id))}`,
  )
  check(
    'calm preset: all 30 cells covered',
    calmLayout.panels.reduce((n, p) => n + p.cells.length, 0) === 30,
  )
  check(
    'calm preset: every horizontal plate is centered on its two-column slot',
    calmLayout.panels
      .filter((p) => p.rectOrientation === 'horizontal')
      .every((p) => p.position[0] === PITCH * p.col + (2 * SIZE + GAP) / 2),
    JSON.stringify(
      calmLayout.panels.filter((p) => p.rectOrientation === 'horizontal').map((p) => [p.id, p.position[0]]),
    ),
  )

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

  // Closed form: the chain start of row 4 is 3.7 + gap·sin22.5 + 3·(60+gap)·sin45,
  // the panel is flat at ψ = 45°, so its lowest solid corner is that start point
  // minus the housing's own drop of 3.7·cos45.
  const y4 = SHORE_Y + GAP * Math.sin(22.5 * D) + 3 * PITCH * Math.sin(45 * D)
  const wantClearance = y4 - 3.7 * Math.cos(45 * D)

  check(
    'floating: exactly one E_END_FLOATING, for column 0',
    layout.violations.length === 1 &&
      layout.violations[0].code === 'E_END_FLOATING' &&
      layout.violations[0].col === 0,
    JSON.stringify(layout.violations.map((v) => [v.code, v.col])),
  )
  check(
    'floating: clearance matches the closed form (≈ 130.9cm)',
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
  // along the bisectors ±5°/±10°/0° — except the very first, leaving 3.7 + gap·sin5°.
  check(
    'edge contact: the LIT FACE stays clear of the floor (3.7 + gap·sin5° ≈ 3.79cm)',
    near(litMin, SHORE_Y + GAP * Math.sin(5 * D), 1e-6),
    `${litMin} vs ${SHORE_Y + GAP * Math.sin(5 * D)}`,
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
  columns[0] = { foldsDeg: [10, 0, -20, 0] } // clearance ≈ 0.143cm at gap 1
  const loose = solveLayout(cfgOf({ columns, groundTolerance: 0.5 }))
  const tight = solveLayout(cfgOf({ columns, groundTolerance: 0.1 }))
  check('groundTolerance: a ~0.14cm clearance passes at 0.5', loose.violations.length === 0)
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
  columns[0] = { foldsDeg: [30, -30, 0, 0] } // the strip parks ~30.5cm up and stays flat
  const cfg = cfgOf({ columns })
  const before = solveLayout(cfg)
  check(
    'solver: the test case really does float (≈ 30.5cm)',
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
    // ground.js snaps to the coarsest of 1° / 0.5° / 0.25° / 0.1° / 0.01° that still
    // lands in the acceptance band — a half degree for this chain at gap 1.
    'solver: it snapped to a readable angle (one of the 1/0.5/0.25/0.1/0.01° grids)',
    [1, 0.5, 0.25, 0.1, 0.01].some(
      (step) => Math.abs(solved.foldDeg / step - Math.round(solved.foldDeg / step)) < 1e-9,
    ),
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
  columns[0] = { foldsDeg: [45, 0, 0, 0] } // last vertex ≈ 133cm up; the panel is 60 long
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
// 9. PITCHED FRONTS — columns[c].startPitchDeg
// =============================================================================
// 9a. BACKWARD COMPATIBILITY: startPitchDeg 0 lays out bit-identically to before.
//     The flat lattice values below are hard-coded, not derived — they are the
//     pre-pitched-fronts geometry, and this is the regression guard on it.
{
  const layout = solveLayout(DEFAULT_CONFIG)
  check(
    'front 0: the flat lattice is EXACTLY (30 + 61c, 3.7, 30 + 61r) — unchanged',
    layout.panels.every((p) => {
      const want = [30 + 61 * p.col, 3.7, 30 + 61 * p.row]
      return p.position.every((v, i) => v === want[i])
    }),
    JSON.stringify(layout.panels.slice(0, 3).map((p) => p.position)),
  )
  check(
    'front 0: every chain still starts at exactly (3.7, 0) and ends at (3.7, 304)',
    layout.columnChains.every(
      (ch) =>
        ch.points[0][0] === 3.7 &&
        ch.points[0][1] === 0 &&
        ch.points[9][0] === 3.7 &&
        ch.points[9][1] === 304,
    ),
    JSON.stringify(layout.columnChains[0].points),
  )
  check(
    'front 0: chainStartY is exactly SHORE_Y for every column',
    layout.columnChains.every((_, c) => chainStartY(DEFAULT_CONFIG, c) === SHORE_Y),
  )
  // A config that OMITS the new fields must solve identically to one that spells
  // out the defaults — the whole compatibility claim in one assertion.
  const legacy = {
    version: 2,
    units: 'cm',
    grid: { cols: 6, rows: 5 },
    cell: { size: 60, rectLength: 121 },
    gap: 1,
    gapTolerance: 1.5,
    groundTolerance: 0.5,
    columns: [
      { foldsDeg: [30, -60, 60, -30] },
      { foldsDeg: [20, -40, 40, -20] },
      { foldsDeg: [0, 0, 0, 0] },
      { foldsDeg: [0, 0, 0, 0] },
      { foldsDeg: [0, 0, 0, 0] },
      { foldsDeg: [0, 0, 0, 0] },
    ],
    rects: [],
    meta: {},
  }
  const explicit = structuredClone(legacy)
  explicit.columns = explicit.columns.map((col) => ({
    ...col,
    startPitchDeg: 0,
    endSupport: 'floor',
  }))
  checkNoThrow('front 0: a config without the new fields solves identically to one with them', () => {
    assert.deepStrictEqual(solveLayout(legacy), solveLayout(explicit))
  })
  check(
    'front 0: columnChains report endSupport "floor" by default',
    solveLayout(legacy).columnChains.every((ch) => ch.endSupport === 'floor'),
  )
}

// 9b. A PITCHED-UP front: y0 = 3.7·cos θ, front edge still on the window line,
//     and the front panel's solid resting on the floor by construction.
{
  const columns = flatColumns()
  columns[2] = { foldsDeg: [0, 0, 0, 0], startPitchDeg: 30 }
  const cfg = cfgOf({ columns })
  check('front 30: config validates', validateConfig(cfg).ok, JSON.stringify(validateConfig(cfg).errors))

  const wantY0 = 3.7 * Math.cos(30 * D) // ≈ 3.2043
  check(
    'front 30: y0 = 3.7·cos30 ≈ 3.204 (the front housing edge on the floor)',
    near(chainStartY(cfg, 2), wantY0, 1e-12) && near(columnChain(cfg, 2).points[0][0], wantY0, 1e-12),
    `${chainStartY(cfg, 2)} vs ${wantY0}`,
  )
  check(
    'front 30: the chain still starts at z = 0 — the front edge is on the window line',
    columnChain(cfg, 2).points[0][1] === 0,
    String(columnChain(cfg, 2).points[0][1]),
  )

  const layout = solveLayout(cfg)
  const col2 = layout.panels.filter((p) => p.col === 2)
  check(
    'front 30: the FRONT panel rests exactly on the floor (solid min y = 0)',
    near(panelSolidMinY(col2[0], cfg), 0, FRONT_REST_EPS),
    String(panelSolidMinY(col2[0], cfg)),
  )
  check(
    'front 30: nothing dips below the floor',
    !layout.warnings.some((w) => w.code === 'W_BELOW_FLOOR'),
    JSON.stringify(layout.warnings.map((w) => w.code)),
  )
  check(
    'front 30: every row of the column sits at 30° (no folds behind it)',
    JSON.stringify(cellPitches(cfg, 2)) === JSON.stringify([30, 30, 30, 30, 30]),
    JSON.stringify(cellPitches(cfg, 2)),
  )
  // Independent closed form for the whole strip, start height included.
  const chain = walkColumn(squares([0, 0, 0, 0]), GAP, 30)
  let posOk = true
  let posBad = ''
  col2.forEach((p, r) => {
    const want = [152, chain[r].center[0], chain[r].center[1]]
    if (!nearVec(p.position, want, 1e-8)) {
      posOk = false
      posBad = `${p.id} ${JSON.stringify(p.position)} want ${JSON.stringify(want)}`
    }
  })
  check('front 30: panel centers match the independent closed-form chain', posOk, posBad)
  // The other columns are untouched: a front pitch is per-column, like a fold.
  checkNoThrow('front 30: the other five columns stay bit-identical to flat', () => {
    const flat = solveLayout(DEFAULT_CONFIG)
    for (const c of [0, 1, 3, 4, 5]) {
      assert.deepStrictEqual(
        layout.panels.filter((p) => p.col === c),
        flat.panels.filter((p) => p.col === c),
      )
    }
  })
  check(
    'front 30: a climbing strip still floats at the end (the two rules are independent)',
    layout.violations.length === 1 && layout.violations[0].col === 2,
    JSON.stringify(layout.violations.map((v) => [v.col, v.clearanceCm])),
  )
}

// 9c. A PITCHED-DOWN front: the panel dives, so its REAR edge is the contact edge
//     and the whole chain lifts by L·sin|θ|. Nothing is clamped — the panels BEHIND
//     it keep diving and raise W_BELOW_FLOOR, which is the documented behaviour.
{
  const columns = flatColumns()
  columns[0] = { foldsDeg: [0, 0, 0, 0], startPitchDeg: -20 }
  const cfg = cfgOf({ columns })
  check('front -20: config validates', validateConfig(cfg).ok, JSON.stringify(validateConfig(cfg).errors))

  const wantY0 = 3.7 * Math.cos(20 * D) + 60 * Math.sin(20 * D) // ≈ 23.998
  check(
    'front -20: y0 = 3.7·cos20 + 60·sin20 ≈ 24.0 (the REAR edge on the floor)',
    near(chainStartY(cfg, 0), wantY0, 1e-12),
    `${chainStartY(cfg, 0)} vs ${wantY0}`,
  )
  check(
    'front -20: it agrees with frontRestY, and the chain still starts at z = 0',
    chainStartY(cfg, 0) === frontRestY(-20, 60) && columnChain(cfg, 0).points[0][1] === 0,
  )

  const layout = solveLayout(cfg)
  const col0 = layout.panels.filter((p) => p.col === 0)
  check(
    'front -20: the FRONT panel still rests exactly on the floor (solid min y = 0)',
    near(panelSolidMinY(col0[0], cfg), 0, FRONT_REST_EPS),
    String(panelSolidMinY(col0[0], cfg)),
  )
  const below = layout.warnings.find((w) => w.code === 'W_BELOW_FLOOR')
  check(
    'front -20: the panels BEHIND it dig in — W_BELOW_FLOOR, surfaced not prevented',
    !!below && ['p1_0', 'p2_0', 'p3_0', 'p4_0'].every((id) => below.message.includes(id)),
    below?.message,
  )
  check(
    'front -20: the front panel itself is NOT among them',
    !!below && !below.message.includes('p0_0'),
    below?.message,
  )
  // A 121cm front plate dives twice as far, so it lifts the chain twice as much.
  const plated = cfgOf({ columns, rects: [{ row: 0, col: 0, orientation: 'vertical' }] })
  check(
    'front -20 with a 121cm plate at row 0: y0 uses the PLATE length',
    near(chainStartY(plated, 0), frontRestY(-20, 121), 1e-12) &&
      chainStartY(plated, 0) > chainStartY(cfg, 0),
    `${chainStartY(plated, 0)} vs ${chainStartY(cfg, 0)}`,
  )
  check(
    'front -20 with a plate: the plate still rests exactly on the floor',
    near(panelSolidMinY(solveLayout(plated).panels.find((p) => p.id === 'p0_0'), plated), 0, FRONT_REST_EPS),
    String(panelSolidMinY(solveLayout(plated).panels.find((p) => p.id === 'p0_0'), plated)),
  )
}

// 9d. The front pitch really is grounded for ANY angle in range.
{
  let allRest = true
  let bad = ''
  for (let theta = -120; theta <= 120; theta += 5) {
    const columns = flatColumns()
    columns[3] = { foldsDeg: [0, 0, 0, 0], startPitchDeg: theta }
    const cfg = cfgOf({ columns })
    const front = solveLayout(cfg).panels.find((p) => p.id === 'p0_3')
    const minY = panelSolidMinY(front, cfg)
    if (!near(minY, 0, FRONT_REST_EPS)) {
      allRest = false
      bad = `θ=${theta}° → front panel min y ${minY}`
      break
    }
  }
  check('front: the front panel rests on the floor at every θ in ±120°', allRest, bad)
}

// =============================================================================
// 10. THE WALL — columns[WALL_COLUMN].endSupport = 'wall' exempts the column
//
// The wall closes the −X edge of the grid, so the only column that may lean on it is
// COLUMN 0. The A/B below puts the SAME climb-only profile on column 0 (wall) and on
// column 1 (floor): identical geometry, opposite verdicts.
// =============================================================================
{
  // A climb-only profile: the strip ends ~131cm in the air and cannot be landed.
  const climb = [45, 0, 0, 0]
  const floorCfg = cfgOf({
    columns: flatColumns().map((col, c) => (c === 1 ? { foldsDeg: climb.slice() } : col)),
  })
  const wallCfg = cfgOf({
    columns: flatColumns().map((col, c) =>
      c === WALL_COLUMN ? { foldsDeg: climb.slice(), endSupport: 'wall' } : col,
    ),
  })
  check('wall: both configs validate', validateConfig(floorCfg).ok && validateConfig(wallCfg).ok)

  const floorLayout = solveLayout(floorCfg)
  const wallLayout = solveLayout(wallCfg)

  check(
    'wall: the SAME profile on a FLOOR column raises E_END_FLOATING',
    floorLayout.violations.length === 1 &&
      floorLayout.violations[0].code === 'E_END_FLOATING' &&
      floorLayout.violations[0].col === 1,
    JSON.stringify(floorLayout.violations.map((v) => [v.code, v.col])),
  )
  check(
    'wall: on the WALL-SUPPORTED column 0 it raises NOTHING',
    wallLayout.violations.length === 0,
    JSON.stringify(wallLayout.violations),
  )
  check(
    'wall: its chain still reports the honest measurement (grounded false, high clearance)',
    wallLayout.columnChains[0].grounded === false &&
      wallLayout.columnChains[0].endClearanceCm > 100 &&
      near(wallLayout.columnChains[0].endClearanceCm, floorLayout.columnChains[1].endClearanceCm, 1e-9),
    `${wallLayout.columnChains[0].endClearanceCm} vs ${floorLayout.columnChains[1].endClearanceCm}`,
  )
  check(
    'wall: the chain entry carries endSupport, and only column 0 is "wall"',
    wallLayout.columnChains[0].endSupport === 'wall' &&
      wallLayout.columnChains.slice(1).every((ch) => ch.endSupport === 'floor'),
    JSON.stringify(wallLayout.columnChains.map((ch) => ch.endSupport)),
  )
  check(
    'wall: the (y, z) geometry is IDENTICAL — endSupport changes reporting, never placement',
    JSON.stringify(
      wallLayout.panels.filter((p) => p.col === WALL_COLUMN).map((p) => p.position.slice(1)),
    ) === JSON.stringify(floorLayout.panels.filter((p) => p.col === 1).map((p) => p.position.slice(1))),
  )
  // groundAllFolds must not "fix" a wall column: pulling its end down undoes the design.
  check(
    'wall: groundAllFolds leaves a wall-supported column alone (returns null here)',
    groundAllFolds(wallCfg) === null,
    JSON.stringify(groundAllFolds(wallCfg)?.columns?.[WALL_COLUMN]),
  )
  check(
    'wall: groundAllFolds still refuses the unreachable FLOOR column too',
    groundAllFolds(floorCfg) === null,
  )
  {
    // …and with one landable floor column beside the wall column, it lands exactly that one.
    const mixed = cfgOf({
      columns: flatColumns().map((col, c) =>
        c === WALL_COLUMN
          ? { foldsDeg: climb.slice(), endSupport: 'wall' }
          : c === 2
            ? { foldsDeg: [30, -30, 0, 0] }
            : col,
      ),
    })
    const next = groundAllFolds(mixed)
    check(
      'wall: groundAllFolds lands the floor column and leaves the wall column byte-identical',
      !!next &&
        JSON.stringify(next.columns[WALL_COLUMN].foldsDeg) === JSON.stringify(climb) &&
        solveLayout(next).violations.length === 0,
      JSON.stringify(next?.columns?.map((col) => col.foldsDeg)),
    )
  }
  // 'wall' is a validation error anywhere else, so the exemption cannot be smuggled
  // onto another column by hand — the solver never even sees such a config.
  check(
    "wall: the same flag on column 5 does not validate, so it cannot be relied on",
    !validateConfig(
      cfgOf({
        columns: flatColumns().map((col, c) =>
          c === 5 ? { foldsDeg: climb.slice(), endSupport: 'wall' } : col,
        ),
      }),
    ).ok,
  )
}

// =============================================================================
// 11. ROW RESOLUTION — the chain, the panels and the joints all follow grid.rows
// =============================================================================
{
  for (const rows of [5, 6, 7, 8]) {
    const cfg = cfgOf({
      grid: { cols: 6, rows },
      columns: Array.from({ length: 6 }, () => ({ foldsDeg: new Array(rows - 1).fill(0) })),
    })
    check(`rows ${rows}: config validates`, validateConfig(cfg).ok, JSON.stringify(validateConfig(cfg).errors))

    const layout = solveLayout(cfg)
    check(
      `rows ${rows}: 6·${rows} = ${6 * rows} panels`,
      layout.panels.length === 6 * rows,
      `got ${layout.panels.length}`,
    )
    check(
      `rows ${rows}: every chain reports ${rows} pitches and 2·${rows} = ${2 * rows} vertices`,
      layout.columnChains.every(
        (ch) => ch.pitchesDeg.length === rows && ch.points.length === 2 * rows,
      ),
      JSON.stringify(layout.columnChains[0].points.length),
    )
    // A flat strip of `rows` cells at the 61cm pitch reaches (rows-1)·61 + 60 in z.
    const wantZ = (rows - 1) * PITCH + SIZE
    check(
      `rows ${rows}: the lattice reaches z = ${wantZ} (${rows - 1}·61 + 60)`,
      near(layout.columnChains[0].points.at(-1)[1], wantZ, 1e-9) &&
        near(Math.max(...layout.panels.map((p) => p.position[2])), wantZ - SIZE / 2, 1e-9),
      `${layout.columnChains[0].points.at(-1)[1]} vs ${wantZ}`,
    )
    check(
      `rows ${rows}: the deepest panel is row ${rows - 1}, and every column has one`,
      layout.panels.filter((p) => p.row === rows - 1).length === 6,
    )
    check(
      `rows ${rows}: a flat grid is fully grounded with no warnings`,
      layout.violations.length === 0 && layout.warnings.length === 0,
      JSON.stringify([layout.violations.length, layout.warnings.map((w) => w.code)]),
    )
    // The surviving-joint count: one per row boundary, so rows-1 per column.
    check(
      `rows ${rows}: ${rows - 1} surviving joints per column, the last one ${rows - 2}`,
      survivingJoints(cfg, 0).length === rows - 1 && lastSurvivingJoint(cfg, 0) === rows - 2,
      JSON.stringify(survivingJoints(cfg, 0)),
    )
  }
  // A deep grid still folds and lands: 8 rows, an arc that comes back down.
  {
    // A SYMMETRIC arc over 8 rows: Σ sin ψ = 0, so it comes back to shore height
    // and the last panel lands on its housing edge with no solver help at all.
    const arc = [20, 10, -10, -20, -20, -10, 10]
    const cfg = cfgOf({
      grid: { cols: 6, rows: 8 },
      columns: Array.from({ length: 6 }, () => ({ foldsDeg: arc.slice() })),
    })
    check('rows 8: an arcing profile validates', validateConfig(cfg).ok)
    const layout = solveLayout(cfg)
    check(
      'rows 8: 48 panels, all six columns land, nothing under the floor',
      layout.panels.length === 48 &&
        layout.violations.length === 0 &&
        !layout.warnings.some((w) => w.code === 'W_BELOW_FLOOR'),
      `${layout.panels.length} panels, violations ${JSON.stringify(layout.violations.map((v) => v.col))}`,
    )
    check(
      'rows 8: the pitch profile is the running sum of all seven folds',
      JSON.stringify(cellPitches(cfg, 0)) === JSON.stringify([0, 20, 30, 20, 0, -20, -30, -20]),
      JSON.stringify(cellPitches(cfg, 0)),
    )
    checkNoThrow('rows 8: deterministic', () => {
      assert.deepStrictEqual(solveLayout(cfg), solveLayout(cfg))
    })
  }
}

// =============================================================================
// 12. The storm presets solve, land their floor columns, and use the wall
// =============================================================================
{
  for (const [id, rows] of [['swell', 7], ['surge', 7], ['wallcrash', 6]]) {
    const cfg = buildPreset(id)
    const layout = solveLayout(cfg)
    check(`${id}: rows ${rows}`, cfg.grid.rows === rows, String(cfg.grid.rows))
    check(
      `${id}: no E_END_FLOATING (wall-supported columns exempt)`,
      layout.violations.length === 0,
      JSON.stringify(layout.violations.map((v) => [v.col, v.clearanceCm])),
    )
    check(
      `${id}: no layout warnings at all`,
      layout.warnings.length === 0,
      JSON.stringify(layout.warnings.map((w) => w.code)),
    )
    check(
      `${id}: every FRONT panel rests exactly on the floor despite its pitch`,
      cfg.columns.every((_, c) => {
        const front = layout.panels.find((p) => p.col === c && p.row === 0)
        return front && near(panelSolidMinY(front, cfg), 0, FRONT_REST_EPS)
      }),
      JSON.stringify(
        cfg.columns.map((_, c) => {
          const front = layout.panels.find((p) => p.col === c && p.row === 0)
          return front ? round9(panelSolidMinY(front, cfg)) : null
        }),
      ),
    )
    checkNoThrow(`${id}: deterministic (deep-equal)`, () => {
      assert.deepStrictEqual(solveLayout(cfg), solveLayout(buildPreset(id)))
    })
  }
}

// =============================================================================
// 13. layoutBounds — the design's OVERALL BOX, housings included
//
// Two independent angles of attack:
//   FLAT     a closed form. The box must be exactly the lattice extent in X and Z
//            and exactly one panel thick in Y, all three derived from the config's
//            own constants rather than typed in.
//   ROTATED  a containment + tightness proof on a real folded preset: every one of
//            the 8 solid corners of every panel is inside the box, and each of the
//            box's six faces is TOUCHED by at least one corner (so it is the
//            minimal box, not merely a bounding one).
// =============================================================================
{
  const THICKNESS = PANEL_PROFILE.overallThickness

  // --- the flat default: exactly the lattice, exactly one panel thick ---------
  {
    const cfg = cfgOf({ columns: flatColumns() })
    const layout = solveLayout(cfg)
    const bounds = layoutBounds(layout, cfg)
    const cols = cfg.grid.cols
    const rows = cfg.grid.rows
    const wantW = cols * PITCH - GAP // 6·61 − 1 = 365
    const wantD = rows * PITCH - GAP // 5·61 − 1 = 304

    check(
      'bounds/flat: min is the origin corner — the housings rest ON the floor at y = 0',
      nearVec(bounds.min, [0, 0, 0]),
      JSON.stringify(bounds.min),
    )
    check(
      'bounds/flat: max is [cols·(size+gap) − gap, overallThickness, rows·(size+gap) − gap]',
      nearVec(bounds.max, [wantW, THICKNESS, wantD]),
      `${JSON.stringify(bounds.max)} want ${JSON.stringify([wantW, THICKNESS, wantD])}`,
    )
    check(
      'bounds/flat: the HEIGHT of a flat design is exactly the panel thickness',
      near(bounds.size[1], THICKNESS),
      `${bounds.size[1]} vs ${THICKNESS}`,
    )
    check(
      'bounds/flat: the DEPTH of a flat design is exactly rows·(size+gap) − gap',
      near(bounds.size[2], wantD),
      `${bounds.size[2]} vs ${wantD}`,
    )
    check(
      'bounds/flat: size = max − min and center = their midpoint, on all three axes',
      nearVec(bounds.size, [wantW, THICKNESS, wantD]) &&
        nearVec(bounds.center, [wantW / 2, THICKNESS / 2, wantD / 2]),
      `${JSON.stringify(bounds.size)} / ${JSON.stringify(bounds.center)}`,
    )
    // The whole point of measuring the SOLID: a lit-face-only box would float 3.7cm
    // up and be a zero-height plane.
    const faceMinY = Math.min(
      ...layout.panels.flatMap((p) => panelWorldCorners(p, cfg).map((corner) => corner[1])),
    )
    check(
      'bounds/flat: the box includes the HOUSINGS — its floor is 3.7cm below the lit faces',
      near(faceMinY, THICKNESS) && near(bounds.min[1], 0),
      `lit faces at ${faceMinY}, box floor at ${bounds.min[1]}`,
    )
  }

  // --- rotated / folded: containment and tightness ---------------------------
  for (const id of ['swell', 'surge', 'wallcrash', 'crash']) {
    const cfg = buildPreset(id)
    const layout = solveLayout(cfg)
    const bounds = layoutBounds(layout, cfg)
    const corners = layout.panels.flatMap((p) => panelSolidCorners(p, cfg))

    check(
      `bounds/${id}: contains all ${corners.length} solid corners of all ${layout.panels.length} panels`,
      corners.every((corner) =>
        corner.every((v, i) => v >= bounds.min[i] - 1e-9 && v <= bounds.max[i] + 1e-9),
      ),
      JSON.stringify([bounds.min, bounds.max]),
    )
    const touched = [0, 1, 2].map((i) => ({
      min: corners.some((corner) => near(corner[i], bounds.min[i], 1e-9)),
      max: corners.some((corner) => near(corner[i], bounds.max[i], 1e-9)),
    }))
    check(
      `bounds/${id}: TIGHT — every one of the six faces is touched by at least one corner`,
      touched.every((t) => t.min && t.max),
      JSON.stringify(touched),
    )
    check(
      `bounds/${id}: the box's peak equals the tallest solid corner, and its height its extent`,
      near(bounds.max[1], Math.max(...corners.map((corner) => corner[1]))) &&
        near(bounds.size[1], bounds.max[1] - bounds.min[1]),
      `${bounds.max[1]} / ${bounds.size[1]}`,
    )
    // A folded surface is taller than one panel and no wider than the lattice: the
    // columns never move in X (the model's first invariant).
    check(
      `bounds/${id}: still exactly as wide as the lattice — columns never move in X`,
      near(bounds.min[0], 0) && near(bounds.max[0], cfg.grid.cols * PITCH - GAP),
      JSON.stringify([bounds.min[0], bounds.max[0]]),
    )
    check(
      `bounds/${id}: and taller than a flat surface — the folds put real height in the box`,
      bounds.size[1] > THICKNESS * 5,
      String(bounds.size[1]),
    )
    checkNoThrow(`bounds/${id}: deterministic (deep-equal across two solves)`, () => {
      assert.deepStrictEqual(bounds, layoutBounds(solveLayout(buildPreset(id)), buildPreset(id)))
    })
  }

  // --- degenerate input ------------------------------------------------------
  check(
    'bounds: an empty layout gives an all-zero box rather than ±Infinity',
    JSON.stringify(layoutBounds({ panels: [] }, DEFAULT_CONFIG)) ===
      JSON.stringify({ min: [0, 0, 0], max: [0, 0, 0], size: [0, 0, 0], center: [0, 0, 0] }),
    JSON.stringify(layoutBounds({ panels: [] }, DEFAULT_CONFIG)),
  )
  check(
    'bounds: a missing layout is tolerated the same way',
    JSON.stringify(layoutBounds(undefined, DEFAULT_CONFIG)) ===
      JSON.stringify(layoutBounds({ panels: [] }, DEFAULT_CONFIG)),
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
