/**
 * tests/test-v3-tiling.mjs — headless checks for core/v3/tiling.js (v3 domino
 * tiling of the material cell grid).
 *
 * Plain node script, no test framework. Exits non-zero on any failure.
 *   node tests/test-v3-tiling.mjs
 *
 * The model under test (V3_SPEC.md §3): a deterministic, form-driven domino
 * tiling of `sheet.cols × sheet.rows` MATERIAL cells with 60×60 squares
 * ('2x2') and 60×121 plates ('2x4'). Everything here works in material (i, j)
 * / (u, v) space — never row/column, never world positions.
 */

import { driftHeight } from '../src/core/v3/form.js'
import { DEFAULT_CONFIG, normalizeConfig } from '../src/core/v3/schema.js'
import { buildTarget } from '../src/core/v3/target.js'
import {
  MIN_PLATES,
  OVERRIDE_MISFIT_CODE,
  SAGITTA_SAMPLE_COUNT,
  buildFewPlatesWarning,
  computeSagittaCm,
  materialToPlanApprox,
  solveTiling,
} from '../src/core/v3/tiling.js'

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

const STRATEGIES = ['flat-lie', 'ridge-aligned', 'toe-bands']
const GRID_SPREAD = [
  [4, 5], // minimum both axes
  [8, 10], // maximum both axes
  [6, 8], // the default
  [5, 7], // odd cols, odd cell count (35, odd)
  [7, 9], // odd cols, even cell count
]

const cfgFor = (cols, rows, strategy) => ({ ...structuredClone(DEFAULT_CONFIG), sheet: { cols, rows }, tiling: { strategy } })

// =============================================================================
// Helpers shared across sections
// =============================================================================

/** Every (i, j) cell in the grid, as "i,j" strings. */
function allCells(cols, rows) {
  const set = new Set()
  for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++) set.add(`${i},${j}`)
  return set
}

/**
 * Independent, brute-force neighbour derivation: scan every pair of
 * orthogonally-adjacent CELLS in the grid; whenever the two cells belong to
 * different tiles, record that tile pair. This is deliberately a totally
 * different method from tiling.js's own rectangle-overlap `axisAdjacency` —
 * it never looks at `uv`, only at which tile owns which cell — so agreement
 * between the two is real evidence, not the same bug checked against itself.
 */
function bruteForceAdjacentPairs(tiles, cols, rows) {
  const owner = new Map()
  for (const t of tiles) for (const [i, j] of t.cells) owner.set(`${i},${j}`, t.id)
  const pairs = new Set()
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) {
      const id = owner.get(`${i},${j}`)
      if (i + 1 < cols) {
        const idR = owner.get(`${i + 1},${j}`)
        if (idR !== id) pairs.add([id, idR].sort().join('|'))
      }
      if (j + 1 < rows) {
        const idU = owner.get(`${i},${j + 1}`)
        if (idU !== id) pairs.add([id, idU].sort().join('|'))
      }
    }
  }
  return pairs
}

function adjacencyPairSet(adjacency) {
  const set = new Set()
  for (const rec of adjacency) set.add([rec.a, rec.b].sort().join('|'))
  return set
}

function setsEqual(a, b) {
  if (a.size !== b.size) return false
  for (const v of a) if (!b.has(v)) return false
  return true
}

// =============================================================================
// 1. Genuine tiling: partition check, for every strategy × grid spread
// =============================================================================
for (const [cols, rows] of GRID_SPREAD) {
  for (const strategy of STRATEGIES) {
    const label = `${cols}x${rows} ${strategy}`
    const result = solveTiling(cfgFor(cols, rows, strategy))
    const expected = allCells(cols, rows)

    const seen = []
    for (const t of result.tiles) for (const [i, j] of t.cells) seen.push(`${i},${j}`)
    const seenSet = new Set(seen)

    check(`${label}: every cell appears exactly once (no duplicates)`, seen.length === seenSet.size, `${seen.length} entries, ${seenSet.size} unique`)
    check(`${label}: covers exactly cols×rows cells`, seen.length === cols * rows, `${seen.length} vs ${cols * rows}`)
    check(`${label}: the covered set equals the full grid (no gaps, nothing extra)`, setsEqual(seenSet, expected))
  }
}

// =============================================================================
// 2. Plates: two orthogonally-adjacent cells, never wrap, never overlap
// =============================================================================
for (const [cols, rows] of GRID_SPREAD) {
  for (const strategy of STRATEGIES) {
    const label = `${cols}x${rows} ${strategy}`
    const result = solveTiling(cfgFor(cols, rows, strategy))
    for (const t of result.tiles) {
      if (t.type === '2x2') {
        check(`${label} ${t.id}: square has exactly 1 cell`, t.cells.length === 1)
      } else if (t.type === '2x4') {
        check(`${label} ${t.id}: plate has exactly 2 cells`, t.cells.length === 2, JSON.stringify(t.cells))
        const [[ai, aj], [bi, bj]] = t.cells
        const dist = Math.abs(ai - bi) + Math.abs(aj - bj)
        check(`${label} ${t.id}: plate's two cells are orthogonally adjacent`, dist === 1, JSON.stringify(t.cells))
      } else {
        check(`${label} ${t.id}: has a recognized type`, false, t.type)
      }
      for (const [i, j] of t.cells) {
        check(`${label} ${t.id}: cell (${i},${j}) is in bounds (no wrap)`, i >= 0 && i < cols && j >= 0 && j < rows)
      }
    }
  }
}

// =============================================================================
// 3. Determinism — identical output across repeat calls, byte-identical
// =============================================================================
for (const [cols, rows] of GRID_SPREAD) {
  for (const strategy of STRATEGIES) {
    const cfg = cfgFor(cols, rows, strategy)
    const r1 = JSON.stringify(solveTiling(cfg))
    const r2 = JSON.stringify(solveTiling(cfg))
    const r3 = JSON.stringify(solveTiling(structuredClone(cfg)))
    check(`${cols}x${rows} ${strategy}: solveTiling is deterministic (repeat call)`, r1 === r2)
    check(`${cols}x${rows} ${strategy}: solveTiling is deterministic (fresh clone of the same config)`, r1 === r3)
  }
}

// =============================================================================
// 4. MIN_PLATES — reached where the grid can support it; W_FEW_PLATES logic
//
// P2b added a sagitta-based fit gate (tiling.js "PLATE FIT"): a candidate
// domino is only eligible to become a plate if a rigid 121cm plate stays
// within `tiling.plateFitToleranceCm` of the target surface over its span.
// That gate is orthogonal to the question this section originally asked —
// "does the underlying greedy highest-score-first placement still saturate
// past MIN_PLATES on every grid this schema allows" — so it is isolated here
// with a FLAT (amplitude 0) form: H ≡ 0 means every candidate's sagitta is
// exactly 0, so nothing is ever rejected by the gate, and this reduces to
// exactly the pre-P2b guarantee. The gate's own "fires for real, points at
// the knob, stays silent on a gentle form" behaviour is covered separately
// below (§13), including concrete curved configs where it now genuinely
// fires — see tiling.js's `buildFewPlatesWarning` doc comment.
// =============================================================================
check('MIN_PLATES is 4', MIN_PLATES === 4, String(MIN_PLATES))
const flatCfgFor = (cols, rows, strategy) => ({
  ...structuredClone(DEFAULT_CONFIG),
  sheet: { cols, rows },
  tiling: { strategy, plateFitToleranceCm: DEFAULT_CONFIG.tiling.plateFitToleranceCm },
  form: { ...DEFAULT_CONFIG.form, amplitude: 0 },
})
for (const [cols, rows] of GRID_SPREAD) {
  for (const strategy of STRATEGIES) {
    const result = solveTiling(flatCfgFor(cols, rows, strategy))
    const plateCount = result.tiles.filter((t) => t.type === '2x4').length
    check(
      `${cols}x${rows} ${strategy} (flat form, sagitta gate can't reject anything): MIN_PLATES (4) reached (got ${plateCount}), no W_FEW_PLATES`,
      plateCount >= MIN_PLATES && !result.warnings.some((w) => w.code === 'W_FEW_PLATES'),
      `plates=${plateCount} warnings=${JSON.stringify(result.warnings)}`,
    )
  }
}
{
  check('buildFewPlatesWarning(0, ...) raises W_FEW_PLATES', buildFewPlatesWarning(0, 'flat-lie', 6, 8)?.code === 'W_FEW_PLATES')
  check('buildFewPlatesWarning(3, ...) raises W_FEW_PLATES', buildFewPlatesWarning(3, 'flat-lie', 6, 8)?.code === 'W_FEW_PLATES')
  check('buildFewPlatesWarning(4, ...) raises nothing (right at MIN_PLATES)', buildFewPlatesWarning(4, 'flat-lie', 6, 8) === null)
  check('buildFewPlatesWarning(10, ...) raises nothing', buildFewPlatesWarning(10, 'flat-lie', 6, 8) === null)
  const w = buildFewPlatesWarning(2, 'toe-bands', 6, 8)
  check(
    'buildFewPlatesWarning message names the count, MIN_PLATES, the strategy and the grid',
    /2 plates/.test(w.message) && /MIN_PLATES \(4\)/.test(w.message) && /"toe-bands"/.test(w.message) && /6×8/.test(w.message),
    w.message,
  )
  check('buildFewPlatesWarning path is "tiling"', w.path === 'tiling')
}

// =============================================================================
// 5. Adjacency — symmetric and complete, verified by independent brute force
// =============================================================================
for (const [cols, rows] of GRID_SPREAD) {
  for (const strategy of STRATEGIES) {
    const label = `${cols}x${rows} ${strategy}`
    const result = solveTiling(cfgFor(cols, rows, strategy))

    const expectedPairs = bruteForceAdjacentPairs(result.tiles, cols, rows)
    const actualPairs = adjacencyPairSet(result.adjacency)
    check(
      `${label}: adjacency matches independent brute-force cell-edge scan exactly`,
      setsEqual(expectedPairs, actualPairs),
      `expected ${expectedPairs.size} pairs, got ${actualPairs.size}; ` +
        `missing=${[...expectedPairs].filter((p) => !actualPairs.has(p)).join(',')} ` +
        `extra=${[...actualPairs].filter((p) => !expectedPairs.has(p)).join(',')}`,
    )

    // No duplicate / reversed-duplicate records for the same pair (each
    // unordered pair appears in the adjacency array at most once).
    const seenKeys = new Set()
    let duplicate = false
    for (const rec of result.adjacency) {
      const key = [rec.a, rec.b].sort().join('|')
      if (seenKeys.has(key)) duplicate = true
      seenKeys.add(key)
    }
    check(`${label}: adjacency has no duplicate/reversed pair records`, !duplicate)

    // Every adjacency record's geometry is sane: positive length, gap-exact
    // separation between the two boundary coordinates, and axis ∈ {u, v}.
    const allSane = result.adjacency.every(
      (rec) =>
        (rec.axis === 'u' || rec.axis === 'v') &&
        rec.materialLength > 0 &&
        rec.edge.to - rec.edge.from === rec.materialLength &&
        Math.abs(Math.abs(rec.edge.b - rec.edge.a) - normalizeConfig(cfgFor(cols, rows, strategy)).gap) < 1e-6,
    )
    check(`${label}: every adjacency record's edge geometry is internally consistent`, allSane)
  }
}
// A plate really can be adjacent to two different tiles along one long side:
// find at least one grid/strategy combo where some tile has ≥ 3 neighbours
// (a plate flanked by two separate tiles along its 121cm edge plus at least
// one more elsewhere), confirming the enumeration isn't assuming "one
// neighbour per side".
{
  let sawThreePlusNeighbour = false
  for (const [cols, rows] of GRID_SPREAD) {
    for (const strategy of STRATEGIES) {
      const result = solveTiling(cfgFor(cols, rows, strategy))
      const degree = new Map()
      for (const rec of result.adjacency) {
        degree.set(rec.a, (degree.get(rec.a) || 0) + 1)
        degree.set(rec.b, (degree.get(rec.b) || 0) + 1)
      }
      if ([...degree.values()].some((d) => d >= 3)) sawThreePlusNeighbour = true
    }
  }
  check('at least one tile in the grid spread has 3+ neighbours (a plate flanked by two tiles on its long side)', sawThreePlusNeighbour)
}

// =============================================================================
// 6. materialToPlanApprox — the explicit, documented-approximate map
// =============================================================================
{
  const cfg = normalizeConfig(DEFAULT_CONFIG)
  const pitch = cfg.cell.size + cfg.gap
  const materialWidth = cfg.sheet.cols * pitch - cfg.gap
  const materialDepth = cfg.sheet.rows * pitch - cfg.gap

  const origin = materialToPlanApprox(DEFAULT_CONFIG, 0, 0)
  check('materialToPlanApprox(config, 0, 0) = (0, 0)', origin.x === 0 && origin.z === 0, JSON.stringify(origin))

  const far = materialToPlanApprox(DEFAULT_CONFIG, materialWidth, materialDepth)
  check(
    'materialToPlanApprox at (materialWidth, materialDepth) reaches the full footprint',
    Math.abs(far.x - cfg.form.footprint.width) < 1e-9 && Math.abs(far.z - cfg.form.footprint.depth) < 1e-9,
    JSON.stringify(far),
  )

  const half = materialToPlanApprox(DEFAULT_CONFIG, materialWidth / 2, materialDepth / 2)
  check(
    'materialToPlanApprox is linear: the midpoint maps to the midpoint',
    Math.abs(half.x - cfg.form.footprint.width / 2) < 1e-9 && Math.abs(half.z - cfg.form.footprint.depth / 2) < 1e-9,
    JSON.stringify(half),
  )

  check(
    'materialToPlanApprox is safe to call on raw (un-normalized) config',
    (() => {
      const p = materialToPlanApprox({ sheet: { cols: 6, rows: 8 } }, 0, 0)
      return p.x === 0 && p.z === 0
    })(),
  )
}

// =============================================================================
// 7. Odd cell count still tiles (leftover cells become squares)
// =============================================================================
{
  const [cols, rows] = [5, 7] // 35 cells, odd
  check('the odd-cell-count fixture really is odd', (cols * rows) % 2 === 1, String(cols * rows))
  for (const strategy of STRATEGIES) {
    const result = solveTiling(cfgFor(cols, rows, strategy))
    const covered = new Set()
    for (const t of result.tiles) for (const [i, j] of t.cells) covered.add(`${i},${j}`)
    check(`5x7 ${strategy}: odd grid still fully tiles`, covered.size === cols * rows, String(covered.size))
    const squareCount = result.tiles.filter((t) => t.type === '2x2').length
    check(`5x7 ${strategy}: an odd cell count forces at least one leftover square`, squareCount >= 1, String(squareCount))
  }
}

// =============================================================================
// 8. The three strategies genuinely diverge on a form with strong curvature
// =============================================================================
{
  // A form with a pronounced, off-centre crest and strong ridge shear, so
  // flat-lie, ridge-aligned and toe-bands each have a real, different signal
  // to respond to (a flat/uncurved form would make them agree trivially).
  //
  // Amplitude is tuned to 100, not the more extreme 220 a pre-P2b version of
  // this fixture used: with the "PLATE FIT" sagitta gate (tiling.js), 220 is
  // curved enough that only MIN_PLATES (4) candidates survive at all on this
  // 6×8 sheet at the default plateFitToleranceCm, and with nothing left to
  // choose among, all three strategies are forced onto the same 4 tiles —
  // agreeing trivially for the opposite reason (no candidates) rather than
  // demonstrating a real difference. 100 keeps strong, genuinely asymmetric
  // curvature (pronounced crest, steep ridge shear, sharp toes) while leaving
  // enough eligible candidates (7 plates' worth) for the strategies' scoring
  // to actually disagree about which ones to take.
  //
  // UPDATE: the strategies can only disagree while the FIT GATE IS BINDING —
  // i.e. while some candidates fit and others do not, so there is a real choice
  // to make. A FACETED target is locally planar by construction, so sagitta
  // collapses to near zero almost everywhere, nearly every candidate survives,
  // and greedy placement takes them all: all three strategies then agree
  // trivially because none of them had to choose. That is a genuine property of
  // the model, not a bug, and it is why the default settings no longer
  // differentiate the strategies — see HANDOFF's note on plate count.
  //
  // So this fixture makes the gate bind: angularity 0 (a smooth target, which
  // rigid plates genuinely struggle to lie on) plus a tight tolerance.
  const strongCfg = {
    ...structuredClone(DEFAULT_CONFIG),
    sheet: { cols: 6, rows: 8 },
    // 3.0, not tighter: at 1.2-2.0 only 3-6 candidates survive the gate, and
    // ridge-aligned and toe-bands then pick the SAME few — agreeing for want of
    // choice rather than because they behave alike. 3.0 leaves ~15 plates' worth
    // of eligible candidates, which is where all three genuinely diverge.
    tiling: { ...DEFAULT_CONFIG.tiling, plateFitToleranceCm: 3.0 },
    form: {
      ...DEFAULT_CONFIG.form, amplitude: 100, crestX: 0.75, crestZ: 0.25,
      ridgeShear: 0.4, toeSharpX: 0.45, toeSharpZ: 0.45, angularity: 0,
    },
  }
  const outputs = STRATEGIES.map((strategy) => JSON.stringify(
    solveTiling({ ...strongCfg, tiling: { ...strongCfg.tiling, strategy } }).tiles))
  check('flat-lie and ridge-aligned produce different tilings', outputs[0] !== outputs[1])
  check('flat-lie and toe-bands produce different tilings', outputs[0] !== outputs[2])
  check('ridge-aligned and toe-bands produce different tilings', outputs[1] !== outputs[2])

  // pattern field names the strategy that produced the tiling.
  for (const strategy of STRATEGIES) {
    const result = solveTiling({ ...strongCfg, tiling: { strategy } })
    check(`pattern field reports "${strategy}"`, result.pattern === strategy)
  }
}

// =============================================================================
// 9. Independent tie-break verification via a degenerate (flat) form
//
// With amplitude 0, H ≡ 0 everywhere, so flat-lie's discrete second
// difference is exactly 0 for EVERY candidate — every score ties, and the
// entire placement order collapses to the documented tie-break rule
// (i, j, axis) ascending. That makes the greedy "place highest-first, skip
// on conflict, fill with squares" mechanism independently re-derivable in
// this test file (a simple simulation, deliberately NOT calling any of
// tiling.js's internals) and checked against solveTiling's real output.
// =============================================================================
{
  const flatCfg = { ...structuredClone(DEFAULT_CONFIG), form: { ...DEFAULT_CONFIG.form, amplitude: 0 }, tiling: { strategy: 'flat-lie' } }
  const cols = 6
  const rows = 8

  // Independent simulation: candidates in the exact documented tie-break
  // order (i, j, axis, 'u' before 'v'), placed greedily.
  const candidates = []
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) {
      if (i + 1 < cols) candidates.push({ i, j, axis: 'u' })
      if (j + 1 < rows) candidates.push({ i, j, axis: 'v' })
    }
  }
  candidates.sort((a, b) => (a.i !== b.i ? a.i - b.i : a.j !== b.j ? a.j - b.j : a.axis < b.axis ? -1 : a.axis > b.axis ? 1 : 0))
  const covered = new Set()
  const expectedTileCells = []
  for (const c of candidates) {
    const cells = c.axis === 'u' ? [[c.i, c.j], [c.i + 1, c.j]] : [[c.i, c.j], [c.i, c.j + 1]]
    if (cells.some(([i, j]) => covered.has(`${i},${j}`))) continue
    for (const [i, j] of cells) covered.add(`${i},${j}`)
    expectedTileCells.push(cells)
  }
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) {
      if (!covered.has(`${i},${j}`)) expectedTileCells.push([[i, j]])
    }
  }

  const result = solveTiling(flatCfg)
  const actualTileCells = result.tiles.map((t) => t.cells)
  const expectedSet = new Set(expectedTileCells.map((c) => JSON.stringify(c)))
  const actualSet = new Set(actualTileCells.map((c) => JSON.stringify(c)))
  check(
    'with a degenerate (amplitude 0) form, the tie-break-only simulation matches solveTiling exactly',
    setsEqual(expectedSet, actualSet),
    `expected ${expectedSet.size} tiles, got ${actualSet.size}`,
  )
}

// =============================================================================
// 10. PLATE FIT — every placed plate's reported sagittaCm is verified by
// INDEPENDENTLY recomputing it here (not by trusting tiling.js's own value).
// Fresh implementation of the chord/perpendicular-distance math, built only
// from each tile's public `uv`/`axis` fields — the same "independent method,
// not the same bug checked against itself" spirit as §5's brute-force
// adjacency check.
//
// Measured against the TARGET (target.js), in 3D, because that is the surface
// the panels are actually seated on. This used to read form.js's driftHeight
// through materialToPlanApprox, matching an earlier tiling.js that was blind to
// faceting: it reported one sagitta for every angularity/facetCells setting
// while placement.js seated the panels on the faceted target. Both sides read
// the same surface now.
// =============================================================================
function independentSagittaCm(cfg, tile, target) {
  const { u0, v0, uLen, vLen } = tile.uv
  const along = tile.axis === 'u' ? { start: u0, len: uLen } : { start: v0, len: vLen }
  const crossPos = tile.axis === 'u' ? v0 + vLen / 2 : u0 + uLen / 2

  const pts = []
  for (let k = 0; k < SAGITTA_SAMPLE_COUNT; k++) {
    const t = k / (SAGITTA_SAMPLE_COUNT - 1)
    const pos = along.start + t * along.len
    const u = tile.axis === 'u' ? pos : crossPos
    const v = tile.axis === 'u' ? crossPos : pos
    pts.push(target.frameAtMaterial(u, v).point)
  }

  // Greatest perpendicular distance in 3D from the chord joining the two ends,
  // via |w x d| / |d| — a rigid plate is a straight chord in space.
  const a = pts[0]
  const b = pts[pts.length - 1]
  const d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]]
  const dLen = Math.hypot(d[0], d[1], d[2]) || 1e-9
  let maxSagitta = 0
  for (let k = 1; k < pts.length - 1; k++) {
    const w = [pts[k][0] - a[0], pts[k][1] - a[1], pts[k][2] - a[2]]
    maxSagitta = Math.max(maxSagitta, Math.hypot(
      w[1] * d[2] - w[2] * d[1],
      w[2] * d[0] - w[0] * d[2],
      w[0] * d[1] - w[1] * d[0],
    ) / dLen)
  }
  return maxSagitta
}

{
  // A spread of tolerances and one extra (more curved) form, so the sample of
  // placed plates isn't all trivially-flat-and-nowhere-near-the-limit cases.
  const fixtures = [
    { cols: 6, rows: 8, strategy: 'flat-lie', plateFitToleranceCm: 2.0, amplitude: 120 },
    { cols: 6, rows: 8, strategy: 'ridge-aligned', plateFitToleranceCm: 6.0, amplitude: 120 },
    { cols: 6, rows: 8, strategy: 'toe-bands', plateFitToleranceCm: 6.0, amplitude: 120 },
    { cols: 8, rows: 10, strategy: 'flat-lie', plateFitToleranceCm: 4.0, amplitude: 150 },
    { cols: 4, rows: 5, strategy: 'flat-lie', plateFitToleranceCm: 20, amplitude: 90 },
  ]
  let checkedAtLeastOnePlate = false
  for (const { cols, rows, strategy, plateFitToleranceCm, amplitude } of fixtures) {
    const cfg = normalizeConfig({
      ...structuredClone(DEFAULT_CONFIG),
      sheet: { cols, rows },
      tiling: { strategy, plateFitToleranceCm },
      form: { ...DEFAULT_CONFIG.form, amplitude },
    })
    const target = buildTarget(cfg)
    const result = solveTiling(cfg, target)
    const label = `${cols}x${rows} ${strategy} tol=${plateFitToleranceCm} amp=${amplitude}`
    for (const tile of result.tiles) {
      if (tile.type === '2x2') {
        check(`${label} ${tile.id}: square reports sagittaCm 0`, tile.sagittaCm === 0, String(tile.sagittaCm))
        continue
      }
      checkedAtLeastOnePlate = true
      const recomputed = independentSagittaCm(cfg, tile, target)
      check(
        `${label} ${tile.id}: independently recomputed sagitta matches the reported sagittaCm`,
        Math.abs(recomputed - tile.sagittaCm) < 1e-9,
        `recomputed=${recomputed} reported=${tile.sagittaCm}`,
      )
      check(
        `${label} ${tile.id}: independently recomputed sagitta is <= plateFitToleranceCm (${plateFitToleranceCm})`,
        recomputed <= plateFitToleranceCm + 1e-9,
        `recomputed=${recomputed}`,
      )
    }
  }
  check('at least one fixture above actually placed a plate (the sagitta checks aren\'t vacuous)', checkedAtLeastOnePlate)
}

// =============================================================================
// 11. Amplitude sweep — plate count falls monotonically as amplitude rises.
// A flat drift should tile almost entirely in plates; a steep one should not
// (V3_SPEC's physical rule: a rigid plate only fits where the surface is
// flat enough — tiling.js "PLATE FIT"). 6×8, flat-lie, default
// plateFitToleranceCm (2.0cm).
// =============================================================================
{
  const amplitudes = [0, 20, 40, 60, 80, 100, 120, 150, 180, 220, 250]
  const counts = amplitudes.map((amplitude) => {
    const cfg = { ...structuredClone(DEFAULT_CONFIG), form: { ...DEFAULT_CONFIG.form, amplitude }, tiling: { strategy: 'flat-lie' } }
    return solveTiling(cfg).tiles.filter((t) => t.type === '2x4').length
  })

  console.log('')
  console.log('amplitude -> plate count (6x8 sheet, flat-lie, plateFitToleranceCm=2.0cm):')
  for (let k = 0; k < amplitudes.length; k++) console.log(`  amplitude=${amplitudes[k]}: ${counts[k]} plates`)

  // Plate count falls STRONGLY but not monotonically with amplitude. Faceting is
  // why: `facetCells` blocks are fixed in material space, so as the form steepens
  // a candidate can move from straddling a crease (bad fit) to sitting inside one
  // facet (perfect fit), which can lift the count at a single step even while the
  // overall trend falls hard. Asserting strict monotonicity would be asserting
  // that faceting does not interact with amplitude, which is false — so assert
  // the trend, plus a bound on any single upward step.
  const first = counts[0]
  const last = counts[counts.length - 1]
  check('plate count falls substantially as the form steepens', last <= first / 2,
    `first=${first} last=${last} counts=${JSON.stringify(counts)}`)
  let worstRise = 0
  for (let k = 1; k < counts.length; k++) worstRise = Math.max(worstRise, counts[k] - counts[k - 1])
  check('no large upward jump in plate count across the sweep', worstRise <= 3,
    `worst single-step rise=${worstRise} counts=${JSON.stringify(counts)}`)
  check('the sweep actually varies — amplitude really drives plate count, not a flat line', new Set(counts).size > 1, JSON.stringify(counts))
  check('a flat form (amplitude 0) tiles almost entirely in plates (>=20 of 24 possible)', counts[0] >= 20, String(counts[0]))
  check(
    'a steep form (amplitude 250) tiles in at most MIN_PLATES worth of plates',
    counts[counts.length - 1] <= MIN_PLATES,
    String(counts[counts.length - 1]),
  )
}

// =============================================================================
// 12. Tolerance sweep — plate count rises monotonically as
// tiling.plateFitToleranceCm rises, and at the top of its range (20cm)
// substantially exceeds the default (2.0cm), approaching the old
// (pre-P2b) maximal-packing regime. 6×8, flat-lie, default (amplitude 120)
// form.
// =============================================================================
{
  const tolerances = [0.1, 0.5, 1, 2, 4, 8, 12, 16, 20]
  const counts = tolerances.map((plateFitToleranceCm) => {
    const cfg = { ...structuredClone(DEFAULT_CONFIG), tiling: { strategy: 'flat-lie', plateFitToleranceCm } }
    return solveTiling(cfg).tiles.filter((t) => t.type === '2x4').length
  })

  console.log('')
  console.log('plateFitToleranceCm -> plate count (6x8 sheet, flat-lie, amplitude=120):')
  for (let k = 0; k < tolerances.length; k++) console.log(`  tol=${tolerances[k]}: ${counts[k]} plates`)

  let nonDecreasing = true
  for (let k = 1; k < counts.length; k++) if (counts[k] < counts[k - 1]) nonDecreasing = false
  check('plate count is non-decreasing across the tolerance sweep', nonDecreasing, JSON.stringify(counts))
  check('the sweep actually varies — tolerance really drives plate count, not a flat line', new Set(counts).size > 1, JSON.stringify(counts))
  const defaultIdx = tolerances.indexOf(2)
  check(
    'at the top of the range (20cm), plate count is well above the default (2.0cm) — approaching maximal packing',
    counts[counts.length - 1] > counts[defaultIdx] * 2,
    JSON.stringify(counts),
  )
}

// =============================================================================
// 13. W_FEW_PLATES fires for REAL through solveTiling (P2b), not only via
// buildFewPlatesWarning called directly — when the form is too curved for
// plateFitToleranceCm to admit MIN_PLATES plates — and stays silent on a
// gentle one. The message points at tiling.plateFitToleranceCm so a user
// reads it as "raise this knob", not as a bug.
// =============================================================================
{
  // Curved enough: the amplitude sweep (§11) shows plate count hits 0 well
  // before amplitude 200 on this 6x8 sheet at the default tolerance.
  const curvedCfg = { ...structuredClone(DEFAULT_CONFIG), form: { ...DEFAULT_CONFIG.form, amplitude: 200 } }
  const curvedResult = solveTiling(curvedCfg)
  const curvedPlates = curvedResult.tiles.filter((t) => t.type === '2x4').length
  check('curved form (amplitude 200): plate count is below MIN_PLATES', curvedPlates < MIN_PLATES, String(curvedPlates))
  const curvedWarning = curvedResult.warnings.find((w) => w.code === 'W_FEW_PLATES')
  check('curved form: W_FEW_PLATES actually fires through solveTiling', !!curvedWarning, JSON.stringify(curvedResult.warnings))
  check(
    'curved form: W_FEW_PLATES message points at tiling.plateFitToleranceCm, explaining why',
    !!curvedWarning && /plateFitToleranceCm/.test(curvedWarning.message) && /too curved/.test(curvedWarning.message),
    curvedWarning?.message,
  )
  check(
    'curved form: the partition property still holds even when every cell falls through to a square',
    (() => {
      const covered = new Set()
      for (const t of curvedResult.tiles) for (const [i, j] of t.cells) covered.add(`${i},${j}`)
      return covered.size === curvedCfg.sheet.cols * curvedCfg.sheet.rows
    })(),
  )

  // Gentle: same sheet, low amplitude — no warning.
  const gentleCfg = { ...structuredClone(DEFAULT_CONFIG), form: { ...DEFAULT_CONFIG.form, amplitude: 20 } }
  const gentleResult = solveTiling(gentleCfg)
  check(
    'gentle form (amplitude 20): W_FEW_PLATES does not fire',
    !gentleResult.warnings.some((w) => w.code === 'W_FEW_PLATES'),
    JSON.stringify(gentleResult.warnings),
  )

  // A second, non-contrived trigger: a small (minimum, 4x5) sheet at the
  // ordinary DEFAULT form and DEFAULT tolerance also genuinely starves
  // plates — the same 365x430cm footprint divided into fewer, larger cells
  // means a single 121cm plate span now covers more of the curve.
  const smallGridResult = solveTiling({ ...structuredClone(DEFAULT_CONFIG), sheet: { cols: 4, rows: 5 } })
  check(
    '4x5 sheet at the plain default (amplitude 120, plateFitToleranceCm 2.0) config also fires W_FEW_PLATES for real',
    smallGridResult.warnings.some((w) => w.code === 'W_FEW_PLATES'),
    JSON.stringify(smallGridResult.warnings),
  )
}

// =============================================================================
// 14. MANUAL OVERRIDES (P8) — the partition property still holds with
// overrides present, across every strategy × grid size; an override is
// actually honoured (the requested cells come back as the requested
// type/axis, and `pinned: true`); overrides + the auto sweep/fill together
// still cover every cell exactly once. NON-NEGOTIABLE — everything
// downstream (adjacency, placement, the report) depends on the partition
// property, override or not.
// =============================================================================
{
  // A square, a 'u' plate, and a 'v' plate — anchored so they never overlap
  // each other and fit inside even the smallest grid in GRID_SPREAD (4x5).
  const overridesFixture = [
    { i: 0, j: 0, type: '2x2', axis: 'u' },
    { i: 1, j: 0, type: '2x4', axis: 'u' },
    { i: 0, j: 2, type: '2x4', axis: 'v' },
  ]
  const overridesCfgFor = (cols, rows, strategy) => ({
    ...structuredClone(DEFAULT_CONFIG),
    sheet: { cols, rows },
    tiling: { strategy, overrides: overridesFixture },
  })

  for (const [cols, rows] of GRID_SPREAD) {
    for (const strategy of STRATEGIES) {
      const label = `${cols}x${rows} ${strategy} +overrides`
      const result = solveTiling(overridesCfgFor(cols, rows, strategy))
      const expected = allCells(cols, rows)

      const seen = []
      for (const t of result.tiles) for (const [i, j] of t.cells) seen.push(`${i},${j}`)
      const seenSet = new Set(seen)
      check(`${label}: every cell appears exactly once (partition holds with overrides)`, seen.length === seenSet.size, `${seen.length} entries, ${seenSet.size} unique`)
      check(`${label}: covers exactly cols×rows cells`, seen.length === cols * rows, `${seen.length} vs ${cols * rows}`)
      check(`${label}: covered set equals the full grid (overrides + sweep + fill together)`, setsEqual(seenSet, expected))

      // The overrides are actually HONOURED: the requested cells come back as
      // the requested type/axis, tagged pinned — not merely "present somehow".
      const sqTile = result.tiles.find((t) => t.cells.length === 1 && t.cells[0][0] === 0 && t.cells[0][1] === 0)
      check(
        `${label}: override square at (0,0) is honoured (type 2x2, pinned)`,
        !!sqTile && sqTile.type === '2x2' && sqTile.pinned === true,
        JSON.stringify(sqTile),
      )
      const plateU = result.tiles.find((t) => t.type === '2x4' && t.axis === 'u' && t.cells[0][0] === 1 && t.cells[0][1] === 0)
      check(
        `${label}: override plate (axis u) at (1,0) is honoured (pinned)`,
        !!plateU && plateU.pinned === true,
        JSON.stringify(plateU),
      )
      const plateV = result.tiles.find((t) => t.type === '2x4' && t.axis === 'v' && t.cells[0][0] === 0 && t.cells[0][1] === 2)
      check(
        `${label}: override plate (axis v) at (0,2) is honoured (pinned)`,
        !!plateV && plateV.pinned === true,
        JSON.stringify(plateV),
      )

      // Every OTHER tile — whatever the sweep/fill produced — is NOT pinned.
      const overrideAnchors = new Set(overridesFixture.map((ov) => `${ov.i},${ov.j}`))
      const nonOverrideTiles = result.tiles.filter((t) => !overrideAnchors.has(`${t.cells[0][0]},${t.cells[0][1]}`))
      check(
        `${label}: every non-override tile is pinned:false (the algorithm chose it)`,
        nonOverrideTiles.every((t) => t.pinned === false),
      )
    }
  }
}

// =============================================================================
// 15. A deliberately bad override (a plate on a steeply curved region) IS
// PLACED, and DOES raise W_PLATE_OVERRIDE_MISFIT with a sagitta above
// tolerance. Both halves matter — this is the crux of the package: a manual
// override is never refused for not fitting, only reported (tiling.js's
// "MANUAL OVERRIDES", mirroring HANDOFF §3.6 / commit 565af13's v2 lesson).
// The worst-case candidate is found by independent brute force over the WHOLE
// grid, not guessed, so this is a real stress case rather than a lucky pick.
// =============================================================================
{
  const steepCfg = normalizeConfig({ ...structuredClone(DEFAULT_CONFIG), form: { ...DEFAULT_CONFIG.form, amplitude: 220 } })
  const target = buildTarget(steepCfg)

  let worst = null
  for (let j = 0; j < steepCfg.sheet.rows; j++) {
    for (let i = 0; i < steepCfg.sheet.cols; i++) {
      for (const axis of ['u', 'v']) {
        if (axis === 'u' && i + 1 >= steepCfg.sheet.cols) continue
        if (axis === 'v' && j + 1 >= steepCfg.sheet.rows) continue
        const sagittaCm = computeSagittaCm(steepCfg, { i, j, axis }, target)
        if (!worst || sagittaCm > worst.sagittaCm) worst = { i, j, axis, sagittaCm }
      }
    }
  }
  check('found a worst-case candidate domino on the steep (amplitude 220) form', !!worst)
  check(
    'the worst-case candidate genuinely exceeds the default plateFitToleranceCm (2.0cm) — a real stress case',
    !!worst && worst.sagittaCm > steepCfg.tiling.plateFitToleranceCm,
    JSON.stringify(worst),
  )

  const overriddenCfg = {
    ...structuredClone(steepCfg),
    tiling: { ...steepCfg.tiling, overrides: [{ i: worst.i, j: worst.j, type: '2x4', axis: worst.axis }] },
  }
  const result = solveTiling(overriddenCfg, target)
  const tileId = `pl_${worst.i}_${worst.j}_${worst.axis}`
  const placedTile = result.tiles.find((t) => t.id === tileId)

  check(
    'the deliberately-bad override plate IS PLACED (never refused for not fitting)',
    !!placedTile && placedTile.type === '2x4' && placedTile.pinned === true,
    JSON.stringify(placedTile),
  )
  check(
    "the placed tile's reported sagittaCm matches the independently brute-forced worst value",
    !!placedTile && Math.abs(placedTile.sagittaCm - worst.sagittaCm) < 1e-9,
    `placed=${placedTile?.sagittaCm} worst=${worst.sagittaCm}`,
  )

  const misfitWarning = result.warnings.find((w) => w.code === OVERRIDE_MISFIT_CODE && w.tile === tileId)
  check('W_PLATE_OVERRIDE_MISFIT DOES fire for the bad override', !!misfitWarning, JSON.stringify(result.warnings))
  check(
    'the misfit warning carries a sagittaCm strictly above its own reported toleranceCm',
    !!misfitWarning && misfitWarning.sagittaCm > misfitWarning.toleranceCm,
    JSON.stringify(misfitWarning),
  )
  check(
    'the misfit warning names the tile and points at tiling.plateFitToleranceCm',
    !!misfitWarning && misfitWarning.message.includes(tileId) && /plateFitToleranceCm/.test(misfitWarning.message),
    misfitWarning?.message,
  )

  // The partition property still holds even with a badly-fitting manual plate.
  const covered = new Set()
  for (const t of result.tiles) for (const [i, j] of t.cells) covered.add(`${i},${j}`)
  check(
    'partition property holds with a misfitting override present',
    covered.size === steepCfg.sheet.cols * steepCfg.sheet.rows,
    String(covered.size),
  )
}

// =============================================================================
// 16. Determinism with overrides present — mirrors section 3, but exercising
// the override code path specifically (place-overrides-first, then sweep).
// =============================================================================
{
  const cfg = {
    ...structuredClone(DEFAULT_CONFIG),
    tiling: {
      strategy: 'flat-lie',
      overrides: [
        { i: 0, j: 0, type: '2x4', axis: 'u' },
        { i: 2, j: 3, type: '2x2', axis: 'u' },
      ],
    },
  }
  const r1 = JSON.stringify(solveTiling(cfg))
  const r2 = JSON.stringify(solveTiling(cfg))
  const r3 = JSON.stringify(solveTiling(structuredClone(cfg)))
  check('solveTiling with overrides is deterministic (repeat call)', r1 === r2)
  check('solveTiling with overrides is deterministic (fresh clone of the same config)', r1 === r3)
}

// =============================================================================
// 17. An override set that pins EVERY cell leaves the strategy sweep nothing
// to do, and the result still tiles: every cell exactly once, entirely in
// pinned plates.
// =============================================================================
{
  const cols = 4
  const rows = 5 // the smallest grid GRID_SPREAD covers; cols is even, so a
  // row-major domino tiling along i covers it exactly with no leftover cell.
  const overrides = []
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i += 2) overrides.push({ i, j, type: '2x4', axis: 'u' })
  }
  check('the fixture actually pins every cell (4x5 = 20 cells, 10 dominoes)', overrides.length === (cols * rows) / 2, String(overrides.length))

  const cfg = { ...structuredClone(DEFAULT_CONFIG), sheet: { cols, rows }, tiling: { strategy: 'flat-lie', overrides } }
  const result = solveTiling(cfg)
  const covered = new Set()
  for (const t of result.tiles) for (const [i, j] of t.cells) covered.add(`${i},${j}`)
  check('fully-pinned grid still tiles: every cell covered exactly once', covered.size === cols * rows, String(covered.size))
  check('fully-pinned grid: every tile is a plate (the sweep had no cell left to give a square)', result.tiles.every((t) => t.type === '2x4'))
  check('fully-pinned grid: every tile is pinned', result.tiles.every((t) => t.pinned === true))
  check('fully-pinned grid: exactly 10 plates, none from the algorithm', result.tiles.length === 10, String(result.tiles.length))
}

// =============================================================================
// 18. Round-trip — a config with overrides survives normalizeConfig and a
// JSON round-trip unchanged, and solveTiling on the round-tripped config
// still honours the overrides exactly.
// =============================================================================
{
  const cfg = {
    ...structuredClone(DEFAULT_CONFIG),
    tiling: {
      strategy: 'flat-lie',
      plateFitToleranceCm: 3.5,
      overrides: [
        { i: 0, j: 0, type: '2x4', axis: 'u' },
        { i: 2, j: 3, type: '2x2', axis: 'u' },
      ],
    },
  }
  const once = normalizeConfig(cfg)
  const twice = normalizeConfig(once)
  check('normalizeConfig with overrides is idempotent', JSON.stringify(once) === JSON.stringify(twice))

  const roundTripped = JSON.parse(JSON.stringify(once))
  check('normalizeConfig output with overrides survives a JSON round-trip unchanged', JSON.stringify(once) === JSON.stringify(roundTripped))
  check(
    'the overrides array itself survives the round-trip with identical content',
    JSON.stringify(once.tiling.overrides) === JSON.stringify(roundTripped.tiling.overrides),
    JSON.stringify(once.tiling.overrides),
  )

  const result = solveTiling(roundTripped)
  const sqTile = result.tiles.find((t) => t.cells.length === 1 && t.cells[0][0] === 2 && t.cells[0][1] === 3)
  check('solveTiling on the round-tripped config still honours the square override at (2,3)', !!sqTile && sqTile.pinned === true, JSON.stringify(sqTile))
  const plateTile = result.tiles.find((t) => t.type === '2x4' && t.axis === 'u' && t.cells[0][0] === 0 && t.cells[0][1] === 0)
  check('solveTiling on the round-tripped config still honours the plate override at (0,0)', !!plateTile && plateTile.pinned === true, JSON.stringify(plateTile))
}

// =============================================================================
// 19. solveTiling is safe to call on RAW input carrying a garbage
// `tiling.overrides` array — normalizeConfig's sanitizing pass (schema.js's
// `sanitizeOverrides`) has already dropped anything malformed / out of bounds
// / conflicting before this file ever sees it, so a garbage array must not
// throw, and the one well-formed, non-conflicting entry must still survive.
// =============================================================================
{
  const garbageCfg = {
    ...structuredClone(DEFAULT_CONFIG),
    tiling: {
      strategy: 'flat-lie',
      overrides: [
        'not an object',
        { i: 'nope', j: 0, type: '2x4', axis: 'u' },
        { i: 0, j: 1, type: 'bogus' },
        { i: 99, j: 99, type: '2x2' },
        { i: 0, j: 0, type: '2x4', axis: 'u' }, // well-formed, first — should survive
        { i: 0, j: 0, type: '2x2' }, // conflicts with the entry above — dropped
      ],
    },
  }
  let threw = false
  let result
  try {
    result = solveTiling(garbageCfg)
  } catch (e) {
    threw = true
    console.error(e)
  }
  check('solveTiling tolerates a garbage tiling.overrides array without throwing', !threw)
  if (result) {
    const covered = new Set()
    for (const t of result.tiles) for (const [i, j] of t.cells) covered.add(`${i},${j}`)
    const { cols, rows } = normalizeConfig(DEFAULT_CONFIG).sheet
    check('garbage-overrides config still tiles the full grid exactly once', covered.size === cols * rows, String(covered.size))
    const tile00 = result.tiles.find((t) => t.cells[0][0] === 0 && t.cells[0][1] === 0)
    check(
      'the one well-formed, first-in-order override (plate at 0,0) survived sanitizing and is honoured',
      !!tile00 && tile00.type === '2x4' && tile00.pinned === true,
      JSON.stringify(tile00),
    )
  }
}

// =============================================================================
// Summary
// =============================================================================
console.log('')
console.log(`test-v3-tiling: ${passed} checks passed, ${failures.length} failed`)
if (failures.length > 0) {
  console.error('')
  console.error('Failures:')
  for (const f of failures) console.error(`  - ${f}`)
  process.exit(1)
}
