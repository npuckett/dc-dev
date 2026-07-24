/**
 * tests/test-validation.mjs — headless checks for core/schema.js.
 *
 * Plain node script, no test framework. Exits non-zero on any failure.
 *   node tests/test-validation.mjs
 */

import {
  DEFAULT_CONFIG,
  normalizeConfig,
  validateConfig,
  cellTilts,
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

// =============================================================================
// 1. Valid configs
// =============================================================================
expectOk('DEFAULT_CONFIG validates', DEFAULT_CONFIG)
expectOk('presetFlat validates', presetFlat())
expectOk('presetCalm validates', presetCalm())
expectOk('presetWave validates', presetWave())
expectOk('presetCrash validates', presetCrash())
expectOk('presetRandom(1) validates', presetRandom(1))
expectOk('presetRandom(42) validates', presetRandom(42))

// =============================================================================
// 2. E_SHAPE — wrong rows length
// =============================================================================
{
  const cfg = good()
  cfg.rows = cfg.rows.slice(0, 4) // 4 rows but grid.rows === 5
  const result = expectError('E_SHAPE rows length', cfg, 'E_SHAPE')
  check(
    'E_SHAPE rows length: path is rows',
    result.errors.some((e) => e.code === 'E_SHAPE' && e.path === 'rows'),
    describe(result),
  )
}

// E_SHAPE — wrong rowFoldsDeg length
{
  const cfg = good()
  cfg.rowFoldsDeg = [0, 0]
  expectError('E_SHAPE rowFoldsDeg length', cfg, 'E_SHAPE')
}

// E_SHAPE — bad version / units
expectError('E_SHAPE bad version', { ...good(), version: 2 }, 'E_SHAPE')
expectError('E_SHAPE bad units', { ...good(), units: 'in' }, 'E_SHAPE')

// =============================================================================
// 3. E_RANGE — zigzag 100
// =============================================================================
{
  const cfg = good()
  cfg.rows[2].zigzagDeg = 100
  expectError('E_RANGE zigzag 100', cfg, 'E_RANGE')
}

// E_RANGE — override outside ±80
{
  const cfg = good()
  cfg.rows[2].jointOverridesDeg = { 1: -95 }
  expectError('E_RANGE override -95', cfg, 'E_RANGE')
}

// E_RANGE — rowFold outside ±120
{
  const cfg = good()
  cfg.rowFoldsDeg[1] = 150
  expectError('E_RANGE rowFold 150', cfg, 'E_RANGE')
}

// E_RANGE — gap out of (0, 10]
expectError('E_RANGE gap 0', { ...good(), gap: 0 }, 'E_RANGE')
expectError('E_RANGE gap 11', { ...good(), gap: 11 }, 'E_RANGE')

// =============================================================================
// 4. E_SHORE_NOT_FLAT
// =============================================================================
{
  const cfg = good()
  cfg.rows[0].zigzagDeg = 10
  expectError('E_SHORE_NOT_FLAT zigzag', cfg, 'E_SHORE_NOT_FLAT')
}
{
  const cfg = good()
  cfg.rows[0].jointOverridesDeg = { 1: 20 }
  expectError('E_SHORE_NOT_FLAT overrides', cfg, 'E_SHORE_NOT_FLAT')
}

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

// =============================================================================
// 7. E_OVERRIDE_ON_REMOVED_JOINT
// =============================================================================
{
  const cfg = good()
  cfg.rows[1].zigzagDeg = 20
  cfg.rows[1].jointOverridesDeg = { 2: -35 }
  cfg.rects = [{ row: 1, col: 2, orientation: 'horizontal' }] // removes joint 2 of row 1
  expectError('E_OVERRIDE_ON_REMOVED_JOINT', cfg, 'E_OVERRIDE_ON_REMOVED_JOINT')
}

// A horizontal rect at a *different* joint is fine.
{
  const cfg = good()
  cfg.rows[1].zigzagDeg = 20
  cfg.rows[1].jointOverridesDeg = { 2: -35 }
  cfg.rects = [{ row: 1, col: 0, orientation: 'horizontal' }] // removes joint 0
  expectOk('override on a surviving joint is fine', cfg)
}

// =============================================================================
// 8. E_CROSSROW_ANGLE_MISMATCH
// =============================================================================
{
  const cfg = good()
  cfg.rows[1].zigzagDeg = 20
  cfg.rows[2].zigzagDeg = 40
  cfg.rects = [{ row: 1, col: 1, orientation: 'vertical' }] // tilt +20 vs +40 at col 1
  const result = expectError('E_CROSSROW_ANGLE_MISMATCH', cfg, 'E_CROSSROW_ANGLE_MISMATCH')
  check(
    'mismatch message names both tilts',
    result.errors.some((e) => e.code === 'E_CROSSROW_ANGLE_MISMATCH' && /20\.00/.test(e.message) && /40\.00/.test(e.message)),
    describe(result),
  )
}

// Column 0 always tilts 0° in every row, so a vertical rect there always matches.
{
  const cfg = good()
  cfg.rows[1].zigzagDeg = 20
  cfg.rows[2].zigzagDeg = 40
  cfg.rects = [{ row: 1, col: 0, orientation: 'vertical' }]
  expectOk('vertical rect at column 0 always matches', cfg)
}

// =============================================================================
// 9. W_CROSSROW_FOLD
// =============================================================================
{
  const cfg = good()
  cfg.rows[1].zigzagDeg = 20
  cfg.rows[2].zigzagDeg = 40
  cfg.rowFoldsDeg = [0, 25, 0, 0] // rows 1/2 boundary folds
  cfg.rects = [{ row: 1, col: 0, orientation: 'vertical' }] // tilts match, but the fold is non-zero
  const result = expectWarning('W_CROSSROW_FOLD', cfg, 'W_CROSSROW_FOLD')
  check('W_CROSSROW_FOLD is only a warning', result.ok, describe(result))
}

// ...and no warning when the spanned boundary is flat.
{
  const cfg = good()
  cfg.rects = [{ row: 1, col: 0, orientation: 'vertical' }]
  const result = validateConfig(cfg)
  check(
    'no W_CROSSROW_FOLD on a flat boundary',
    !warnCodes(result).includes('W_CROSSROW_FOLD'),
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

// No rects → no length warning.
{
  const result = validateConfig(good())
  check(
    'no W_RECT_LENGTH without rects',
    !warnCodes(result).includes('W_RECT_LENGTH'),
    describe(result),
  )
}

// Exact rectLength → no warning even with rects.
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
// 11. cellTilts
// =============================================================================
{
  const cfg = good()
  cfg.rows[1].zigzagDeg = 30
  const tilts = cellTilts(cfg, 1)
  // φ(1,j) = +30, -30, +30, -30, +30 → running sums 0, 30, 0, 30, 0, 30
  check(
    'cellTilts alternating chain',
    JSON.stringify(tilts) === JSON.stringify([0, 30, 0, 30, 0, 30]),
    JSON.stringify(tilts),
  )
}
{
  const cfg = good()
  cfg.rows[1].zigzagDeg = 30
  cfg.rects = [{ row: 1, col: 2, orientation: 'horizontal' }] // removes joint 2
  const tilts = cellTilts(cfg, 1)
  // joint 2 applies no fold → cells 2 and 3 share a tilt
  check(
    'cellTilts: horizontal rect removes its joint',
    JSON.stringify(tilts) === JSON.stringify([0, 30, 0, 0, -30, 0]),
    JSON.stringify(tilts),
  )
}
{
  const cfg = good()
  cfg.rows[1].zigzagDeg = 20
  cfg.rows[1].jointOverridesDeg = { 1: 45 } // signed absolute, replaces -20
  const tilts = cellTilts(cfg, 1)
  check(
    'cellTilts: override used as-is (sign not flipped)',
    JSON.stringify(tilts) === JSON.stringify([0, 20, 65, 85, 65, 85]),
    JSON.stringify(tilts),
  )
}
{
  const flatTilts = cellTilts(DEFAULT_CONFIG, 0)
  check(
    'cellTilts: flat row is all zeros',
    flatTilts.length === 6 && flatTilts.every((t) => t === 0),
    JSON.stringify(flatTilts),
  )
}

// =============================================================================
// 12. normalizeConfig fills defaults from minimal input
// =============================================================================
{
  const minimal = {
    version: 1,
    units: 'cm',
    rows: [
      { zigzagDeg: 0 },
      { zigzagDeg: 12 },
      { zigzagDeg: 18 },
      { zigzagDeg: 18 },
      { zigzagDeg: 6 },
    ],
    rowFoldsDeg: [0, 0, 0, 0],
  }
  const before = structuredClone(minimal)
  const cfg = normalizeConfig(minimal)

  check('normalize: gap default', cfg.gap === 2.0, String(cfg.gap))
  check('normalize: gapTolerance default', cfg.gapTolerance === 1.5, String(cfg.gapTolerance))
  check('normalize: rowAnchor default', cfg.rowAnchor === 'center', String(cfg.rowAnchor))
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
    'normalize: jointOverridesDeg filled per row',
    cfg.rows.every((r) => r.jointOverridesDeg && Object.keys(r.jointOverridesDeg).length === 0),
  )
  check(
    'normalize: rows/rowFolds passed through',
    cfg.rows.length === 5 && cfg.rows[1].zigzagDeg === 12 && cfg.rowFoldsDeg.length === 4,
  )
  check(
    'normalize: input not mutated',
    JSON.stringify(minimal) === JSON.stringify(before),
    JSON.stringify(minimal),
  )
  check('normalize: minimal config validates', validateConfig(minimal).ok, describe(validateConfig(minimal)))
  check('normalize: no `name` invented', cfg.name === undefined, String(cfg.name))

  // Deep-copy check: mutating the output must not touch the input.
  cfg.rows[1].zigzagDeg = 99
  cfg.rowFoldsDeg[0] = 99
  check('normalize: output rows are copies', minimal.rows[1].zigzagDeg === 12)
  check('normalize: output rowFoldsDeg is a copy', minimal.rowFoldsDeg[0] === 0)
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
  } catch (e) {
    threw = true
    console.error(e)
  }
  check('normalize/validate tolerate junk input', !threw)
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
