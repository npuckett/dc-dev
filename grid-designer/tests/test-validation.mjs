/**
 * tests/test-validation.mjs — headless checks for core/schema.js (v2).
 *
 * Plain node script, no test framework. Exits non-zero on any failure.
 *   node tests/test-validation.mjs
 *
 * The model under test: folding happens along COLUMNS only. Each column carries
 * grid.rows-1 signed hinge angles; row 0 is flat by construction.
 */

import {
  DEFAULT_CONFIG,
  cellPitches,
  columnChain,
  normalizeConfig,
  validateConfig,
} from '../src/core/schema.js'
import {
  presetFlat,
  presetCalm,
  presetWave,
  presetCrash,
  presetRandom,
} from '../src/core/presets.js'

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

const codes = (result) => result.errors.map((e) => e.code)
const warnCodes = (result) => result.warnings.map((w) => w.code)
const describe = (result) =>
  `errors=[${result.errors.map((e) => `${e.code}@${e.path}`).join(', ')}] ` +
  `warnings=[${result.warnings.map((w) => `${w.code}@${w.path}`).join(', ')}]`

function expectOk(name, config) {
  const result = validateConfig(config)
  check(name, result.ok, describe(result))
  return result
}

function expectError(name, config, code) {
  const result = validateConfig(config)
  check(`${name}: not ok`, !result.ok, describe(result))
  check(`${name}: ${code} fires`, codes(result).includes(code), describe(result))
  return result
}

function expectWarning(name, config, code) {
  const result = validateConfig(config)
  check(`${name}: ${code} warns`, warnCodes(result).includes(code), describe(result))
  return result
}

// A known-good starting point for crafting bad configs.
const good = () => structuredClone(DEFAULT_CONFIG)
const withFolds = (perColumn) => {
  const cfg = good()
  perColumn.forEach((foldsDeg, c) => {
    if (foldsDeg) cfg.columns[c] = { foldsDeg: foldsDeg.slice() }
  })
  return cfg
}

// =============================================================================
// 1. Valid configs
// =============================================================================
expectOk('DEFAULT_CONFIG validates', DEFAULT_CONFIG)
check('DEFAULT_CONFIG is version 2', DEFAULT_CONFIG.version === 2, String(DEFAULT_CONFIG.version))
check(
  'DEFAULT_CONFIG has 6 all-zero fold sequences',
  DEFAULT_CONFIG.columns.length === 6 &&
    DEFAULT_CONFIG.columns.every((c) => c.foldsDeg.length === 4 && c.foldsDeg.every((f) => f === 0)),
  JSON.stringify(DEFAULT_CONFIG.columns),
)
check('DEFAULT_CONFIG has no rowAnchor / rows / rowFoldsDeg', DEFAULT_CONFIG.rowAnchor === undefined && DEFAULT_CONFIG.rows === undefined && DEFAULT_CONFIG.rowFoldsDeg === undefined)
expectOk('presetFlat validates', presetFlat())
expectOk('presetCalm validates', presetCalm())
expectOk('presetWave validates', presetWave())
expectOk('presetCrash validates', presetCrash())
expectOk('presetRandom(1) validates', presetRandom(1))
expectOk('presetRandom(42) validates', presetRandom(42))

// =============================================================================
// 2. v1 configs are REJECTED outright
// =============================================================================
{
  const v1 = {
    version: 1,
    units: 'cm',
    grid: { cols: 6, rows: 5 },
    cell: { size: 60, rectLength: 121 },
    gap: 2,
    gapTolerance: 1.5,
    rowAnchor: 'center',
    rows: [
      { zigzagDeg: 0, jointOverridesDeg: {} },
      { zigzagDeg: 20, jointOverridesDeg: {} },
      { zigzagDeg: 30, jointOverridesDeg: {} },
      { zigzagDeg: 30, jointOverridesDeg: {} },
      { zigzagDeg: 10, jointOverridesDeg: {} },
    ],
    rowFoldsDeg: [15, 40, 55, -10],
    rects: [],
  }
  const result = expectError('v1 config rejected', v1, 'E_SHAPE')
  check(
    'v1 config: message says version must be 2 and names the row-accordion model',
    result.errors.some(
      (e) =>
        e.path === 'version' &&
        /version must be 2/.test(e.message) &&
        /row-accordion/.test(e.message),
    ),
    JSON.stringify(result.errors.map((e) => e.message)),
  )
  check(
    'v1 config: its missing `columns` array is reported too',
    result.errors.some((e) => e.code === 'E_SHAPE' && e.path === 'columns'),
    describe(result),
  )
}
expectError('E_SHAPE version 3', { ...good(), version: 3 }, 'E_SHAPE')
expectError('E_SHAPE bad units', { ...good(), units: 'in' }, 'E_SHAPE')

// =============================================================================
// 3. E_SHAPE — structural problems in `columns`
// =============================================================================
{
  const cfg = good()
  cfg.columns = cfg.columns.slice(0, 5) // 5 columns but grid.cols === 6
  const result = expectError('E_SHAPE columns length', cfg, 'E_SHAPE')
  check(
    'E_SHAPE columns length: path is columns',
    result.errors.some((e) => e.code === 'E_SHAPE' && e.path === 'columns'),
    describe(result),
  )
}
{
  const cfg = good()
  cfg.columns[3] = { foldsDeg: [0, 0, 0] } // needs grid.rows-1 = 4
  const result = expectError('E_SHAPE foldsDeg length', cfg, 'E_SHAPE')
  check(
    'E_SHAPE foldsDeg length: path names the column',
    result.errors.some((e) => e.path === 'columns[3].foldsDeg'),
    describe(result),
  )
}
{
  const cfg = good()
  delete cfg.columns
  expectError('E_SHAPE missing columns', cfg, 'E_SHAPE')
}
{
  const cfg = good()
  cfg.columns[1] = 42
  expectError('E_SHAPE column entry not an object', cfg, 'E_SHAPE')
}
{
  const cfg = good()
  cfg.columns[1] = { foldsDeg: 'nope' }
  expectError('E_SHAPE foldsDeg not an array', cfg, 'E_SHAPE')
}
{
  const cfg = good()
  cfg.columns[2].foldsDeg[1] = null
  expectError('E_SHAPE fold entry not a number', cfg, 'E_SHAPE')
}
{
  const cfg = good()
  cfg.grid.cols = 0
  expectError('E_SHAPE grid.cols 0', cfg, 'E_SHAPE')
}

// =============================================================================
// 4. E_RANGE — folds and gap
// =============================================================================
{
  const cfg = good()
  cfg.columns[1].foldsDeg[2] = 130
  const result = expectError('E_RANGE fold 130', cfg, 'E_RANGE')
  check(
    'E_RANGE fold: path names the joint',
    result.errors.some((e) => e.code === 'E_RANGE' && e.path === 'columns[1].foldsDeg[2]'),
    describe(result),
  )
}
{
  const cfg = good()
  cfg.columns[0].foldsDeg[0] = -121
  expectError('E_RANGE fold -121', cfg, 'E_RANGE')
}
{
  const cfg = good()
  cfg.columns[0].foldsDeg[0] = -120
  expectOk('fold exactly -120 is allowed', cfg)
}
expectError('E_RANGE gap 0', { ...good(), gap: 0 }, 'E_RANGE')
expectError('E_RANGE gap 11', { ...good(), gap: 11 }, 'E_RANGE')

// --- groundTolerance (how close the last panel must get to y = 0) ------------
{
  const result = expectError(
    'E_SHAPE groundTolerance 0',
    { ...good(), groundTolerance: 0 },
    'E_SHAPE',
  )
  check(
    'E_SHAPE groundTolerance: path and message name the field and the rule',
    result.errors.some(
      (e) =>
        e.code === 'E_SHAPE' &&
        e.path === 'groundTolerance' &&
        /positive number/.test(e.message) &&
        /touching the floor/.test(e.message),
    ),
    describe(result),
  )
}
expectError('E_SHAPE groundTolerance negative', { ...good(), groundTolerance: -1 }, 'E_SHAPE')
expectError('E_SHAPE groundTolerance not a number', { ...good(), groundTolerance: 'loose' }, 'E_SHAPE')
expectError('E_SHAPE groundTolerance null', { ...good(), groundTolerance: null }, 'E_SHAPE')
expectOk('groundTolerance 0.1 is allowed', { ...good(), groundTolerance: 0.1 })
expectOk('groundTolerance 5 is allowed', { ...good(), groundTolerance: 5 })

// =============================================================================
// 5. Rect bounds
// =============================================================================
{
  const cfg = good()
  cfg.rects = [{ row: 1, col: 5, orientation: 'horizontal' }] // needs col ≤ 4
  expectError('horizontal rect out of bounds', cfg, 'E_SHAPE')
}
{
  const cfg = good()
  cfg.rects = [{ row: 4, col: 2, orientation: 'vertical' }] // needs row ≤ 3
  expectError('vertical rect out of bounds', cfg, 'E_SHAPE')
}

// =============================================================================
// 6. E_RECT_OVERLAP
// =============================================================================
{
  const cfg = good()
  cfg.rects = [
    { row: 1, col: 1, orientation: 'horizontal' }, // cells (1,1),(1,2)
    { row: 1, col: 2, orientation: 'horizontal' }, // cells (1,2),(1,3) — shares (1,2)
  ]
  expectError('E_RECT_OVERLAP', cfg, 'E_RECT_OVERLAP')
}
{
  const cfg = good()
  cfg.rects = [
    { row: 1, col: 2, orientation: 'horizontal' }, // (1,2),(1,3)
    { row: 1, col: 3, orientation: 'vertical' }, // (1,3),(2,3) — shares (1,3)
  ]
  expectError('E_RECT_OVERLAP across orientations', cfg, 'E_RECT_OVERLAP')
}

// =============================================================================
// 7. E_FOLD_ON_REMOVED_JOINT — a vertical plate cannot bend
// =============================================================================
{
  const cfg = withFolds([null, null, [0, 25, 0, 0]])
  cfg.rects = [{ row: 1, col: 2, orientation: 'vertical' }] // removes joint 1 of column 2
  const result = expectError('E_FOLD_ON_REMOVED_JOINT', cfg, 'E_FOLD_ON_REMOVED_JOINT')
  check(
    'E_FOLD_ON_REMOVED_JOINT: message says set it to 0 or remove the rect',
    result.errors.some(
      (e) =>
        e.code === 'E_FOLD_ON_REMOVED_JOINT' &&
        /set it to 0 or remove the rect/.test(e.message) &&
        e.path === 'columns[2].foldsDeg[1]',
    ),
    describe(result),
  )
}
{
  // The same plate is fine when the joint it removes is unfolded.
  const cfg = withFolds([null, null, [0, 0, 25, 0]])
  cfg.rects = [{ row: 1, col: 2, orientation: 'vertical' }]
  expectOk('vertical plate on an unfolded joint is fine', cfg)
}
{
  // A fold on a DIFFERENT joint of the same column is fine.
  const cfg = withFolds([null, null, [30, 0, -30, 0]])
  cfg.rects = [{ row: 1, col: 2, orientation: 'vertical' }]
  expectOk('folds elsewhere in the column are fine', cfg)
}

// =============================================================================
// 8. E_CROSSCOL_ANGLE_MISMATCH — a horizontal plate needs matching pitches
// =============================================================================
{
  const cfg = withFolds([[30, 0, 0, 0]])
  cfg.rects = [{ row: 1, col: 0, orientation: 'horizontal' }] // pitch 30 vs 0 at row 1
  const result = expectError('E_CROSSCOL_ANGLE_MISMATCH', cfg, 'E_CROSSCOL_ANGLE_MISMATCH')
  check(
    'mismatch message names both pitches',
    result.errors.some(
      (e) =>
        e.code === 'E_CROSSCOL_ANGLE_MISMATCH' &&
        /30\.00/.test(e.message) &&
        /0\.00/.test(e.message),
    ),
    describe(result),
  )
}
{
  // Row 0 always sits at pitch 0 in every column, so a plate there always matches.
  const cfg = withFolds([[30, 0, 0, 0], [-40, 0, 0, 0]])
  cfg.rects = [{ row: 0, col: 0, orientation: 'horizontal' }]
  expectOk('horizontal plate on the shore row always matches', cfg)
}
{
  // Matching cumulative pitches reached by different fold sequences also pass.
  const cfg = withFolds([[20, 20, 0, 0], [40, 0, 0, 0]])
  cfg.rects = [{ row: 2, col: 0, orientation: 'horizontal' }] // both at 40° by row 2
  const result = expectOk('matching cumulative pitch passes (different sequences)', cfg)
  check(
    'matching pitch but diverged position warns W_CROSSCOL_POSITION',
    warnCodes(result).includes('W_CROSSCOL_POSITION'),
    describe(result),
  )
}

// =============================================================================
// 9. W_CROSSCOL_POSITION — same angle, different place
// =============================================================================
{
  // Column 0 goes up and comes back down: pitch 0 again at row 2, but 30cm higher
  // and 8cm shallower than column 1.
  const cfg = withFolds([[30, -30, 0, 0]])
  cfg.rects = [{ row: 2, col: 0, orientation: 'horizontal' }]
  const result = expectWarning('W_CROSSCOL_POSITION', cfg, 'W_CROSSCOL_POSITION')
  check('W_CROSSCOL_POSITION is only a warning', result.ok, describe(result))
  check(
    'W_CROSSCOL_POSITION message reports both origins',
    result.warnings.some((w) => w.code === 'W_CROSSCOL_POSITION' && /y 34\.74/.test(w.message)),
    JSON.stringify(result.warnings.map((w) => w.message)),
  )
}
{
  // Identical columns → identical chains → no position warning.
  const cfg = withFolds([[30, -30, 0, 0], [30, -30, 0, 0]])
  cfg.rects = [{ row: 2, col: 0, orientation: 'horizontal' }]
  const result = validateConfig(cfg)
  check(
    'no W_CROSSCOL_POSITION when the two chains are identical',
    result.ok && !warnCodes(result).includes('W_CROSSCOL_POSITION'),
    describe(result),
  )
}

// =============================================================================
// 10. W_RECT_LENGTH — 121 ≠ 2·60 + 2
// =============================================================================
{
  const cfg = good()
  cfg.rects = [{ row: 1, col: 2, orientation: 'horizontal' }]
  const result = expectWarning('W_RECT_LENGTH', cfg, 'W_RECT_LENGTH')
  check('W_RECT_LENGTH is only a warning', result.ok, describe(result))
}
{
  const result = validateConfig(good())
  check(
    'no W_RECT_LENGTH without rects',
    !warnCodes(result).includes('W_RECT_LENGTH'),
    describe(result),
  )
}
{
  const cfg = good()
  cfg.cell.rectLength = 122
  cfg.rects = [{ row: 1, col: 2, orientation: 'horizontal' }]
  const result = validateConfig(cfg)
  check(
    'rectLength 122 with gap 2 → no W_RECT_LENGTH',
    !warnCodes(result).includes('W_RECT_LENGTH'),
    describe(result),
  )
}

// =============================================================================
// 11. cellPitches / columnChain
// =============================================================================
{
  const cfg = withFolds([null, null, [30, -60, 60, -30]])
  const pitches = cellPitches(cfg, 2)
  check(
    'cellPitches: running sum of the column fold sequence',
    JSON.stringify(pitches) === JSON.stringify([0, 30, -30, 30, 0]),
    JSON.stringify(pitches),
  )
  check(
    'cellPitches: untouched columns stay all-zero',
    JSON.stringify(cellPitches(cfg, 0)) === JSON.stringify([0, 0, 0, 0, 0]),
    JSON.stringify(cellPitches(cfg, 0)),
  )
}
{
  const flat = cellPitches(DEFAULT_CONFIG, 0)
  check(
    'cellPitches: flat column is all zeros, one entry per row',
    flat.length === 5 && flat.every((p) => p === 0),
    JSON.stringify(flat),
  )
}
{
  // A vertical plate removes its joint: both of its rows share one pitch, and
  // the folds behind it still apply.
  const cfg = withFolds([null, null, [20, 0, 40, 0]])
  cfg.rects = [{ row: 1, col: 2, orientation: 'vertical' }]
  check('vertical-plate config validates', validateConfig(cfg).ok, describe(validateConfig(cfg)))
  const pitches = cellPitches(cfg, 2)
  check(
    'cellPitches: a vertical plate spans two rows at one pitch',
    JSON.stringify(pitches) === JSON.stringify([0, 20, 20, 60, 60]),
    JSON.stringify(pitches),
  )
}
{
  // The chain geometry itself: a flat column walks straight back in z.
  const chain = columnChain(DEFAULT_CONFIG, 3)
  check(
    'columnChain: flat column origins are (3.7, 62r)',
    chain.origins.every(([y, z], r) => Math.abs(y - 3.7) < 1e-12 && Math.abs(z - 62 * r) < 1e-12),
    JSON.stringify(chain.origins),
  )
  check(
    'columnChain: 10 vertices — the start, 5 segment ends, 4 gap ends',
    chain.points.length === 10,
    String(chain.points.length),
  )
  check(
    'columnChain: reports one segment per row for a rect-free column',
    chain.segments.length === 5 && chain.segments.every((s) => s.kind === 'square'),
    JSON.stringify(chain.segments.map((s) => s.kind)),
  )
}
{
  // A horizontal plate makes the neighbouring column's cell a phantom, which
  // still advances the chain exactly as a square would.
  const cfg = good()
  cfg.rects = [{ row: 2, col: 1, orientation: 'horizontal' }]
  const owner = columnChain(cfg, 1)
  const phantom = columnChain(cfg, 2)
  check(
    'columnChain: plate owner has an hrect segment, its neighbour a phantom',
    owner.segments[2].kind === 'hrect' && phantom.segments[2].kind === 'phantom',
    JSON.stringify([owner.segments.map((s) => s.kind), phantom.segments.map((s) => s.kind)]),
  )
  check(
    'columnChain: a phantom advances the chain like a square',
    JSON.stringify(phantom.origins) === JSON.stringify(columnChain(good(), 2).origins),
    JSON.stringify(phantom.origins),
  )
}

// =============================================================================
// 12. normalizeConfig fills defaults from minimal input
// =============================================================================
{
  const minimal = {
    version: 2,
    units: 'cm',
    columns: [
      { foldsDeg: [10, 0, 0, 0] },
      { foldsDeg: [10, 0, 0, 0] },
      { foldsDeg: [0, 10, 0, 0] },
      { foldsDeg: [0, 10, 0, 0] },
      { foldsDeg: [0, 0, 10, 0] },
      { foldsDeg: [0, 0, 10, 0] },
    ],
  }
  const before = structuredClone(minimal)
  const cfg = normalizeConfig(minimal)

  check('normalize: gap default', cfg.gap === 2.0, String(cfg.gap))
  check('normalize: gapTolerance default', cfg.gapTolerance === 1.5, String(cfg.gapTolerance))
  check(
    'normalize: groundTolerance default 0.5',
    cfg.groundTolerance === 0.5,
    String(cfg.groundTolerance),
  )
  check(
    'normalize: an explicit groundTolerance is passed through',
    normalizeConfig({ ...minimal, groundTolerance: 1.25 }).groundTolerance === 1.25,
  )
  check(
    'normalize: cell default',
    cfg.cell.size === 60 && cfg.cell.rectLength === 121,
    JSON.stringify(cfg.cell),
  )
  check(
    'normalize: grid default',
    cfg.grid.cols === 6 && cfg.grid.rows === 5,
    JSON.stringify(cfg.grid),
  )
  check('normalize: rects default []', Array.isArray(cfg.rects) && cfg.rects.length === 0)
  check('normalize: meta object', cfg.meta && typeof cfg.meta === 'object' && cfg.meta.notes === '')
  check(
    'normalize: columns passed through',
    cfg.columns.length === 6 && JSON.stringify(cfg.columns[2].foldsDeg) === JSON.stringify([0, 10, 0, 0]),
    JSON.stringify(cfg.columns),
  )
  check('normalize: no rowAnchor invented', cfg.rowAnchor === undefined)
  check(
    'normalize: input not mutated',
    JSON.stringify(minimal) === JSON.stringify(before),
    JSON.stringify(minimal),
  )
  check('normalize: minimal config validates', validateConfig(minimal).ok, describe(validateConfig(minimal)))
  check('normalize: no `name` invented', cfg.name === undefined, String(cfg.name))

  // Deep-copy check: mutating the output must not touch the input.
  cfg.columns[0].foldsDeg[0] = 99
  check('normalize: output foldsDeg arrays are copies', minimal.columns[0].foldsDeg[0] === 10)
}

// normalizeConfig tolerates junk without throwing.
{
  let threw = false
  try {
    normalizeConfig(undefined)
    normalizeConfig(null)
    normalizeConfig(42)
    validateConfig({})
    validateConfig(null)
    validateConfig({ version: 2, units: 'cm', columns: [{ foldsDeg: [1] }] })
    cellPitches({}, 0)
    columnChain({ grid: { cols: 'x', rows: null } }, 0)
  } catch (e) {
    threw = true
    console.error(e)
  }
  check('normalize/validate/chain tolerate junk input', !threw)
}

// =============================================================================
// Summary
// =============================================================================
console.log('')
console.log(`test-validation: ${passed} checks passed, ${failures.length} failed`)
if (failures.length > 0) {
  console.error('')
  console.error('Failures:')
  for (const f of failures) console.error(`  - ${f}`)
  process.exit(1)
}
