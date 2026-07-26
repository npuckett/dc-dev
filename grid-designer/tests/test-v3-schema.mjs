/**
 * tests/test-v3-schema.mjs — headless checks for core/v3/schema.js (v3 config).
 *
 * Plain node script, no test framework. Exits non-zero on any failure.
 *   node tests/test-v3-schema.mjs
 *
 * The model under test: ONE tiled drift surface (V3_SPEC.md §1, §6), replacing
 * v2's per-column fold strips entirely. `sheet.{cols,rows}` are MATERIAL cell
 * counts (i, j) — never "grid"/"columns"/"rows" in the v2 sense.
 */

import {
  AMPLITUDE_MAX,
  AMPLITUDE_MIN,
  CREST_MAX,
  CREST_MIN,
  DEFAULT_CONFIG,
  DEFAULT_GAP,
  DEFAULT_TILING,
  GAP_MAX,
  GAP_MIN,
  OVERRIDE_AXES,
  OVERRIDE_TYPES,
  PLACEMENT_TREES,
  PLATE_FIT_TOLERANCE_MAX,
  PLATE_FIT_TOLERANCE_MIN,
  RIDGE_SHEAR_MAX,
  RIDGE_SHEAR_MIN,
  SHEET_COLS_MAX,
  SHEET_COLS_MIN,
  SHEET_ROWS_MAX,
  SHEET_ROWS_MIN,
  TILING_STRATEGIES,
  TOE_SHARP_MAX,
  TOE_SHARP_MIN,
  normalizeConfig,
  validateConfig,
  DEFAULT_GROUND_TOLERANCE,
} from '../src/core/v3/schema.js'

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
  `valid=${result.valid} errors=[${result.errors.map((e) => `${e.code}@${e.path}`).join(', ')}] ` +
  `warnings=[${result.warnings.map((w) => `${w.code}@${w.path}`).join(', ')}]`

function expectValid(name, config) {
  const result = validateConfig(config)
  check(name, result.valid, describe(result))
  return result
}

function expectError(name, config, code) {
  const result = validateConfig(config)
  check(`${name}: invalid`, !result.valid, describe(result))
  check(`${name}: ${code} fires`, codes(result).includes(code), describe(result))
  return result
}

function expectWarning(name, config, code) {
  const result = validateConfig(config)
  check(`${name}: ${code} warns`, warnCodes(result).includes(code), describe(result))
  return result
}

const good = () => structuredClone(DEFAULT_CONFIG)

// =============================================================================
// 1. DEFAULT_CONFIG
// =============================================================================
expectValid('DEFAULT_CONFIG validates', DEFAULT_CONFIG)
check('DEFAULT_CONFIG is version 3', DEFAULT_CONFIG.version === 3, String(DEFAULT_CONFIG.version))
check('DEFAULT_CONFIG units is cm', DEFAULT_CONFIG.units === 'cm')
check(
  'DEFAULT_CONFIG has sheet.cols/rows, not grid.cols/rows',
  DEFAULT_CONFIG.sheet && DEFAULT_CONFIG.grid === undefined,
  JSON.stringify(DEFAULT_CONFIG.sheet),
)
check(
  'DEFAULT_CONFIG has no v2 fields (columns, rects, grid)',
  DEFAULT_CONFIG.columns === undefined && DEFAULT_CONFIG.rects === undefined && DEFAULT_CONFIG.grid === undefined,
)
check(
  'the defaults satisfy 2·cell.size + gap = cell.plateLength (60 + 1 + 60 = 121) — load-bearing, see HANDOFF §2.2',
  2 * DEFAULT_CONFIG.cell.size + DEFAULT_CONFIG.gap === DEFAULT_CONFIG.cell.plateLength,
  `${2 * DEFAULT_CONFIG.cell.size + DEFAULT_CONFIG.gap} vs ${DEFAULT_CONFIG.cell.plateLength}`,
)
check('DEFAULT_GAP is 1.0', DEFAULT_GAP === 1.0 && DEFAULT_CONFIG.gap === 1.0)

// =============================================================================
// 2. v1 and v2 configs are REJECTED outright, never migrated
// =============================================================================
{
  const v1 = {
    version: 1,
    units: 'cm',
    rowAnchor: 'center',
    rows: [{ zigzagDeg: 0 }, { zigzagDeg: 20 }],
    rowFoldsDeg: [15, 40],
  }
  const result = expectError('v1 config rejected', v1, 'E_SHAPE')
  check(
    'v1 config: message says version must be 3 and rejects v1/v2 by name',
    result.errors.some(
      (e) => e.path === 'version' && /version must be exactly 3/.test(e.message) && /v1 \(row-accordion\)/.test(e.message),
    ),
    describe(result),
  )
}
{
  // A genuine, richly-populated v2 config (per-column fold strips).
  const v2 = {
    version: 2,
    units: 'cm',
    grid: { cols: 6, rows: 5 },
    cell: { size: 60, rectLength: 121 },
    gap: 1.0,
    columns: [
      { foldsDeg: [0, 0, 0, 0] },
      { foldsDeg: [0, 0, 0, 0] },
      { foldsDeg: [0, 0, 0, 0] },
      { foldsDeg: [0, 0, 0, 0] },
      { foldsDeg: [0, 0, 0, 0] },
      { foldsDeg: [0, 0, 0, 0] },
    ],
    rects: [],
  }
  const result = expectError('v2 config rejected', v2, 'E_SHAPE')
  check(
    'v2 config: message names v2 (per-column fold strips) as unsupported',
    result.errors.some((e) => e.path === 'version' && /v2 \(per-column fold strips\)/.test(e.message)),
    describe(result),
  )
  check(
    'v2 config: NOT silently migrated — no v3-shaped tiling/placement/form keys are invented from it',
    !('tiling' in v2) && !('placement' in v2) && !('form' in v2),
  )
}
expectError('E_SHAPE version 4', { ...good(), version: 4 }, 'E_SHAPE')
expectError('E_SHAPE version 0', { ...good(), version: 0 }, 'E_SHAPE')
{
  // A MISSING version defaults to 3 (mirrors v2: absence defaults, only an
  // explicitly-wrong value errors — see schema.js's "TWO KINDS OF DEFAULTING").
  const cfg = good()
  delete cfg.version
  expectValid('a config with no version at all defaults to 3 and validates', cfg)
}
expectError('E_SHAPE bad units', { ...good(), units: 'in' }, 'E_SHAPE')

// =============================================================================
// 3. E_SHAPE / E_RANGE — sheet.cols / sheet.rows
// =============================================================================
check('SHEET_COLS range is 4..8', SHEET_COLS_MIN === 4 && SHEET_COLS_MAX === 8)
check('SHEET_ROWS range is 5..10', SHEET_ROWS_MIN === 5 && SHEET_ROWS_MAX === 10)
{
  const cfg = good()
  cfg.sheet = { cols: 'six', rows: 8 }
  expectError('E_SHAPE sheet.cols not a number', cfg, 'E_SHAPE')
}
{
  const cfg = good()
  cfg.sheet = { cols: 6.5, rows: 8 }
  expectError('E_SHAPE sheet.cols not an integer', cfg, 'E_SHAPE')
}
{
  const cfg = good()
  cfg.sheet = { cols: 3, rows: 8 }
  const result = expectError('E_RANGE sheet.cols below min', cfg, 'E_RANGE')
  check(
    'E_RANGE sheet.cols: message names the band',
    result.errors.some((e) => e.code === 'E_RANGE' && e.path === 'sheet.cols' && /4\.\.8/.test(e.message)),
    describe(result),
  )
}
{
  const cfg = good()
  cfg.sheet = { cols: 9, rows: 8 }
  expectError('E_RANGE sheet.cols above max', cfg, 'E_RANGE')
}
{
  const cfg = good()
  cfg.sheet = { cols: 6, rows: 4 }
  expectError('E_RANGE sheet.rows below min', cfg, 'E_RANGE')
}
{
  const cfg = good()
  cfg.sheet = { cols: 6, rows: 11 }
  expectError('E_RANGE sheet.rows above max', cfg, 'E_RANGE')
}
for (let cols = SHEET_COLS_MIN; cols <= SHEET_COLS_MAX; cols++) {
  expectValid(`sheet.cols ${cols} is accepted`, { ...good(), sheet: { cols, rows: 8 } })
}
for (let rows = SHEET_ROWS_MIN; rows <= SHEET_ROWS_MAX; rows++) {
  expectValid(`sheet.rows ${rows} is accepted`, { ...good(), sheet: { cols: 6, rows } })
}

// --- normalizeConfig clamps sheet.cols/rows -----------------------------
{
  check(
    'normalizeConfig clamps sheet.cols below range up to the min',
    normalizeConfig({ sheet: { cols: -50, rows: 8 } }).sheet.cols === SHEET_COLS_MIN,
  )
  check(
    'normalizeConfig clamps sheet.cols above range down to the max',
    normalizeConfig({ sheet: { cols: 999, rows: 8 } }).sheet.cols === SHEET_COLS_MAX,
  )
  check(
    'normalizeConfig clamps sheet.rows below range up to the min',
    normalizeConfig({ sheet: { cols: 6, rows: -50 } }).sheet.rows === SHEET_ROWS_MIN,
  )
  check(
    'normalizeConfig clamps sheet.rows above range down to the max',
    normalizeConfig({ sheet: { cols: 6, rows: 999 } }).sheet.rows === SHEET_ROWS_MAX,
  )
  check(
    'normalizeConfig rounds a fractional sheet.cols before clamping',
    normalizeConfig({ sheet: { cols: 6.4, rows: 8 } }).sheet.cols === 6,
  )
}

// =============================================================================
// 4. E_SHAPE — cell.size / cell.plateLength (positive number, no declared band)
// =============================================================================
expectError('E_SHAPE cell.size 0', { ...good(), cell: { size: 0, plateLength: 121 } }, 'E_SHAPE')
expectError('E_SHAPE cell.size negative', { ...good(), cell: { size: -60, plateLength: 121 } }, 'E_SHAPE')
expectError('E_SHAPE cell.size not a number', { ...good(), cell: { size: 'sixty', plateLength: 121 } }, 'E_SHAPE')
expectError('E_SHAPE cell.plateLength 0', { ...good(), cell: { size: 60, plateLength: 0 } }, 'E_SHAPE')
expectError('E_SHAPE cell.plateLength negative', { ...good(), cell: { size: 60, plateLength: -1 } }, 'E_SHAPE')
{
  check(
    'normalizeConfig falls back to the default cell size on garbage',
    normalizeConfig({ cell: { size: -5, plateLength: -5 } }).cell.size === 60,
  )
}

// =============================================================================
// 5. E_SHAPE / E_RANGE — gap (0..10)
// =============================================================================
check('GAP range is 0..10', GAP_MIN === 0 && GAP_MAX === 10)
expectError('E_SHAPE gap not a number', { ...good(), gap: 'loose' }, 'E_SHAPE')
expectError('E_SHAPE gap null', { ...good(), gap: null }, 'E_SHAPE')
expectError('E_RANGE gap -1', { ...good(), gap: -1 }, 'E_RANGE')
expectError('E_RANGE gap 11', { ...good(), gap: 11 }, 'E_RANGE')
expectValid('gap exactly 0 is allowed', { ...good(), gap: 0 })
expectValid('gap exactly 10 is allowed', { ...good(), gap: 10 })
{
  check('normalizeConfig clamps gap below range to 0', normalizeConfig({ gap: -50 }).gap === GAP_MIN)
  check('normalizeConfig clamps gap above range to 10', normalizeConfig({ gap: 500 }).gap === GAP_MAX)
}

// =============================================================================
// 6. form: every range clamp + every E_SHAPE/E_RANGE (mirrors form.js)
// =============================================================================
const formRanges = [
  ['amplitude', AMPLITUDE_MIN, AMPLITUDE_MAX],
  ['crestX', CREST_MIN, CREST_MAX],
  ['crestZ', CREST_MIN, CREST_MAX],
  ['ridgeShear', RIDGE_SHEAR_MIN, RIDGE_SHEAR_MAX],
  ['toeSharpX', TOE_SHARP_MIN, TOE_SHARP_MAX],
  ['toeSharpZ', TOE_SHARP_MIN, TOE_SHARP_MAX],
]
for (const [key, lo, hi] of formRanges) {
  {
    const cfg = good()
    cfg.form = { ...cfg.form, [key]: lo - 1000 }
    expectError(`E_RANGE form.${key} far below ${lo}`, cfg, 'E_RANGE')
    check(
      `normalizeConfig clamps form.${key} below range to ${lo}`,
      normalizeConfig(cfg).form[key] === lo,
      String(normalizeConfig(cfg).form[key]),
    )
  }
  {
    const cfg = good()
    cfg.form = { ...cfg.form, [key]: hi + 1000 }
    expectError(`E_RANGE form.${key} far above ${hi}`, cfg, 'E_RANGE')
    check(
      `normalizeConfig clamps form.${key} above range to ${hi}`,
      normalizeConfig(cfg).form[key] === hi,
      String(normalizeConfig(cfg).form[key]),
    )
  }
  {
    const cfg = good()
    cfg.form = { ...cfg.form, [key]: lo }
    expectValid(`form.${key} exactly at min (${lo}) is allowed`, cfg)
  }
  {
    const cfg = good()
    cfg.form = { ...cfg.form, [key]: hi }
    expectValid(`form.${key} exactly at max (${hi}) is allowed`, cfg)
  }
  {
    const cfg = good()
    cfg.form = { ...cfg.form, [key]: 'nope' }
    expectError(`E_SHAPE form.${key} not a number`, cfg, 'E_SHAPE')
  }
}
expectError(
  'E_SHAPE form.footprint.width non-positive',
  { ...good(), form: { ...good().form, footprint: { width: 0, depth: 430 } } },
  'E_SHAPE',
)
expectError(
  'E_SHAPE form.footprint.depth non-positive',
  { ...good(), form: { ...good().form, footprint: { width: 365, depth: -1 } } },
  'E_SHAPE',
)

// =============================================================================
// 7. tiling.strategy / placement.tree enums
// =============================================================================
check(
  'TILING_STRATEGIES are exactly flat-lie, ridge-aligned, toe-bands',
  JSON.stringify(TILING_STRATEGIES) === JSON.stringify(['flat-lie', 'ridge-aligned', 'toe-bands']),
  JSON.stringify(TILING_STRATEGIES),
)
check(
  'PLACEMENT_TREES are exactly bfs-corner, comb-v, comb-u',
  JSON.stringify(PLACEMENT_TREES) === JSON.stringify(['bfs-corner', 'comb-v', 'comb-u']),
  JSON.stringify(PLACEMENT_TREES),
)
for (const strategy of TILING_STRATEGIES) {
  expectValid(`tiling.strategy "${strategy}" is accepted`, { ...good(), tiling: { strategy } })
}
for (const tree of PLACEMENT_TREES) {
  expectValid(`placement.tree "${tree}" is accepted`, { ...good(), placement: { tree } })
}
{
  const result = expectError('E_SHAPE bad tiling.strategy', { ...good(), tiling: { strategy: 'random' } }, 'E_SHAPE')
  check(
    'E_SHAPE tiling.strategy: message lists the legal values',
    result.errors.some((e) => e.path === 'tiling.strategy' && /flat-lie/.test(e.message) && /ridge-aligned/.test(e.message)),
    describe(result),
  )
  check(
    'normalizeConfig falls back to the default strategy on an invalid one',
    normalizeConfig({ tiling: { strategy: 'random' } }).tiling.strategy === 'flat-lie',
  )
}
{
  expectError('E_SHAPE bad placement.tree', { ...good(), placement: { tree: 'spiral' } }, 'E_SHAPE')
  check(
    'normalizeConfig falls back to the default tree on an invalid one',
    normalizeConfig({ placement: { tree: 'spiral' } }).placement.tree === 'bfs-corner',
  )
}

// =============================================================================
// 7b. tiling.plateFitToleranceCm — 0.1..20cm (P2b, v3-tiling-specific — see
// tiling.js's "PLATE FIT" section for what the knob does)
// =============================================================================
check('PLATE_FIT_TOLERANCE range is 0.1..20', PLATE_FIT_TOLERANCE_MIN === 0.1 && PLATE_FIT_TOLERANCE_MAX === 20)
check('DEFAULT_TILING.plateFitToleranceCm is 2.0', DEFAULT_TILING.plateFitToleranceCm === 2.0)
check(
  'DEFAULT_CONFIG.tiling.plateFitToleranceCm is 2.0',
  DEFAULT_CONFIG.tiling.plateFitToleranceCm === 2.0,
  String(DEFAULT_CONFIG.tiling.plateFitToleranceCm),
)
{
  const cfg = good()
  cfg.tiling = { ...cfg.tiling, plateFitToleranceCm: PLATE_FIT_TOLERANCE_MIN - 1000 }
  const result = expectError('E_RANGE tiling.plateFitToleranceCm far below min', cfg, 'E_RANGE')
  check(
    'E_RANGE tiling.plateFitToleranceCm: message names the band',
    result.errors.some((e) => e.code === 'E_RANGE' && e.path === 'tiling.plateFitToleranceCm' && /0\.1\.\.20/.test(e.message)),
    describe(result),
  )
  check(
    'normalizeConfig clamps tiling.plateFitToleranceCm below range up to the min',
    normalizeConfig(cfg).tiling.plateFitToleranceCm === PLATE_FIT_TOLERANCE_MIN,
  )
}
{
  const cfg = good()
  cfg.tiling = { ...cfg.tiling, plateFitToleranceCm: PLATE_FIT_TOLERANCE_MAX + 1000 }
  expectError('E_RANGE tiling.plateFitToleranceCm far above max', cfg, 'E_RANGE')
  check(
    'normalizeConfig clamps tiling.plateFitToleranceCm above range down to the max',
    normalizeConfig(cfg).tiling.plateFitToleranceCm === PLATE_FIT_TOLERANCE_MAX,
  )
}
expectValid('tiling.plateFitToleranceCm exactly at min (0.1) is allowed', {
  ...good(),
  tiling: { ...good().tiling, plateFitToleranceCm: PLATE_FIT_TOLERANCE_MIN },
})
expectValid('tiling.plateFitToleranceCm exactly at max (20) is allowed', {
  ...good(),
  tiling: { ...good().tiling, plateFitToleranceCm: PLATE_FIT_TOLERANCE_MAX },
})
{
  const cfg = good()
  cfg.tiling = { ...cfg.tiling, plateFitToleranceCm: 'loose' }
  expectError('E_SHAPE tiling.plateFitToleranceCm not a number', cfg, 'E_SHAPE')
}
{
  const cfg = good()
  cfg.tiling = { ...cfg.tiling, plateFitToleranceCm: NaN }
  expectError('E_SHAPE tiling.plateFitToleranceCm NaN', cfg, 'E_SHAPE')
}
check(
  'a config with no tiling.plateFitToleranceCm at all defaults to 2.0 and validates',
  (() => {
    const cfg = good()
    cfg.tiling = { strategy: 'flat-lie' }
    return validateConfig(cfg).valid && normalizeConfig(cfg).tiling.plateFitToleranceCm === 2.0
  })(),
)
check(
  'normalizeConfig falls back to the default plateFitToleranceCm on garbage',
  normalizeConfig({ tiling: { plateFitToleranceCm: NaN } }).tiling.plateFitToleranceCm === 2.0,
)

// =============================================================================
// 8. gapTolerance / groundTolerance — positive number, no declared band
// =============================================================================
expectError('E_SHAPE gapTolerance 0', { ...good(), gapTolerance: 0 }, 'E_SHAPE')
expectError('E_SHAPE gapTolerance negative', { ...good(), gapTolerance: -1 }, 'E_SHAPE')
expectError('E_SHAPE gapTolerance not a number', { ...good(), gapTolerance: 'loose' }, 'E_SHAPE')
expectError('E_SHAPE groundTolerance 0', { ...good(), groundTolerance: 0 }, 'E_SHAPE')
{
  const result = expectError('E_SHAPE groundTolerance negative', { ...good(), groundTolerance: -1 }, 'E_SHAPE')
  check(
    'E_SHAPE groundTolerance: message names the floor-touch rule',
    result.errors.some((e) => e.path === 'groundTolerance' && /touching the floor/.test(e.message)),
    describe(result),
  )
}
expectValid('groundTolerance 0.1 is allowed', { ...good(), groundTolerance: 0.1 })
{
  check(
    'normalizeConfig falls back to the default gapTolerance on garbage',
    normalizeConfig({ gapTolerance: -5 }).gapTolerance === 1.5,
  )
  check(
    'normalizeConfig falls back to the default groundTolerance on garbage',
    normalizeConfig({ groundTolerance: NaN }).groundTolerance === DEFAULT_GROUND_TOLERANCE,
  )
}

// =============================================================================
// 9. W_PLATE_LENGTH — mirrors v2's W_RECT_LENGTH, silent at the defaults
// =============================================================================
{
  const result = validateConfig(good())
  check('no W_PLATE_LENGTH at the defaults (60+1+60=121 exact)', !warnCodes(result).includes('W_PLATE_LENGTH'), describe(result))
}
{
  const cfg = good()
  cfg.gap = 2 // ideal becomes 122, plateLength stays 121cm of hardware
  const result = expectWarning('W_PLATE_LENGTH (gap 2 vs plateLength 121)', cfg, 'W_PLATE_LENGTH')
  check('W_PLATE_LENGTH is only a warning', result.valid, describe(result))
  check(
    'W_PLATE_LENGTH names both numbers and the slack',
    result.warnings.some(
      (w) => w.code === 'W_PLATE_LENGTH' && /121cm/.test(w.message) && /122cm/.test(w.message) && /1\.00cm of slack/.test(w.message),
    ),
    JSON.stringify(result.warnings.map((w) => w.message)),
  )
}
{
  const cfg = good()
  cfg.cell = { size: 60, plateLength: 130 }
  expectWarning('W_PLATE_LENGTH (plateLength 130 vs ideal 121)', cfg, 'W_PLATE_LENGTH')
}
{
  // Fixing gap back up so it matches a mismatched plateLength silences the warning.
  const cfg = good()
  cfg.gap = 2
  cfg.cell = { size: 60, plateLength: 122 }
  const result = validateConfig(cfg)
  check('no W_PLATE_LENGTH once the numbers add up again', !warnCodes(result).includes('W_PLATE_LENGTH'), describe(result))
}

// =============================================================================
// 10. normalizeConfig: idempotent, deterministic, never mutates input
// =============================================================================
{
  const inputs = [
    DEFAULT_CONFIG,
    {},
    { sheet: { cols: 4, rows: 5 } },
    { version: 2, units: 'cm' }, // v2-shaped garbage from v3's point of view
    { gap: -50, form: { amplitude: 99999, crestX: -1, ridgeShear: 5 } },
    { tiling: { strategy: 'bogus' }, placement: { tree: 'bogus' } },
  ]
  for (const input of inputs) {
    const once = normalizeConfig(input)
    const twice = normalizeConfig(once)
    check(
      `normalizeConfig is idempotent for ${JSON.stringify(input).slice(0, 40)}`,
      JSON.stringify(once) === JSON.stringify(twice),
      `${JSON.stringify(once)} vs ${JSON.stringify(twice)}`,
    )
    const again = normalizeConfig(input)
    check(
      `normalizeConfig is deterministic (repeat call, same input) for ${JSON.stringify(input).slice(0, 40)}`,
      JSON.stringify(once) === JSON.stringify(again),
    )
  }
}
{
  const before = structuredClone(DEFAULT_CONFIG)
  const out = normalizeConfig(DEFAULT_CONFIG)
  check('normalizeConfig does not mutate its input', JSON.stringify(DEFAULT_CONFIG) === JSON.stringify(before))
  out.sheet.cols = 999
  check(
    'normalizeConfig output sheet is a fresh copy (mutating it does not touch DEFAULT_CONFIG)',
    DEFAULT_CONFIG.sheet.cols === 6,
  )
}

// =============================================================================
// 11. Missing / extra / garbage fields
// =============================================================================
{
  const minimal = { version: 3, units: 'cm' }
  expectValid('a minimal {version, units} config validates (everything else defaults)', minimal)
  const cfg = normalizeConfig(minimal)
  check('normalize: sheet defaults to 6x8', cfg.sheet.cols === 6 && cfg.sheet.rows === 8, JSON.stringify(cfg.sheet))
  check('normalize: cell defaults', cfg.cell.size === 60 && cfg.cell.plateLength === 121, JSON.stringify(cfg.cell))
  check('normalize: gap defaults to 1.0', cfg.gap === 1.0)
  check('normalize: tiling.strategy defaults to flat-lie', cfg.tiling.strategy === 'flat-lie')
  check('normalize: placement.tree defaults to bfs-corner', cfg.placement.tree === 'bfs-corner')
  check('normalize: no `name` invented for a config that omitted it', cfg.name === undefined)
}
{
  // Extra, unknown top-level and nested fields are tolerated and ignored.
  const withExtras = { ...good(), extraTopLevelField: 'ignored', sheet: { cols: 6, rows: 8, extraNested: true } }
  const result = validateConfig(withExtras)
  check('extra unknown fields do not break validation', result.valid, describe(result))
}
{
  let threw = false
  try {
    normalizeConfig(undefined)
    normalizeConfig(null)
    normalizeConfig(42)
    normalizeConfig('nope')
    normalizeConfig([])
    validateConfig(undefined)
    validateConfig(null)
    validateConfig(42)
    validateConfig('nope')
    validateConfig([])
    validateConfig({})
  } catch (e) {
    threw = true
    console.error(e)
  }
  check('normalize/validate tolerate garbage input without throwing', !threw)
}

// =============================================================================
// 12. NaN and Infinity inputs
// =============================================================================
{
  const cfg = {
    ...good(),
    gap: NaN,
    sheet: { cols: NaN, rows: Infinity },
    form: { ...good().form, amplitude: Infinity, ridgeShear: -Infinity, crestX: NaN },
    tiling: { ...good().tiling, plateFitToleranceCm: Infinity },
    gapTolerance: NaN,
    groundTolerance: Infinity,
  }
  const result = validateConfig(cfg)
  check('NaN/Infinity config is invalid, not a thrown exception', result.valid === false)
  check(
    'NaN/Infinity: every offending field gets E_SHAPE (not a silently-passed number)',
    [
      'gap',
      'sheet.cols',
      'sheet.rows',
      'form.amplitude',
      'form.ridgeShear',
      'form.crestX',
      'tiling.plateFitToleranceCm',
      'gapTolerance',
      'groundTolerance',
    ].every((path) => result.errors.some((e) => e.path === path)),
    describe(result),
  )
  const normalized = normalizeConfig(cfg)
  check(
    'NaN/Infinity: normalizeConfig falls back to sane finite defaults, never propagates NaN/Infinity',
    Number.isFinite(normalized.gap) &&
      Number.isFinite(normalized.sheet.cols) &&
      Number.isFinite(normalized.sheet.rows) &&
      Number.isFinite(normalized.form.amplitude) &&
      Number.isFinite(normalized.form.ridgeShear) &&
      Number.isFinite(normalized.form.crestX) &&
      Number.isFinite(normalized.tiling.plateFitToleranceCm) &&
      Number.isFinite(normalized.gapTolerance) &&
      Number.isFinite(normalized.groundTolerance),
    JSON.stringify(normalized),
  )
}

// =============================================================================
// 13. tiling.overrides (P8) — manual square/plate pins. See schema.js's
// "MANUAL TILE OVERRIDES": normalizeConfig SANITIZES (drops malformed /
// out-of-bounds / conflicting entries silently, first-in-array-order wins a
// contested cell); validateConfig REPORTS the same problems on the RAW array
// instead of dropping them.
// =============================================================================
check(
  'OVERRIDE_TYPES are exactly 2x2, 2x4',
  JSON.stringify(OVERRIDE_TYPES) === JSON.stringify(['2x2', '2x4']),
  JSON.stringify(OVERRIDE_TYPES),
)
check('OVERRIDE_AXES are exactly u, v', JSON.stringify(OVERRIDE_AXES) === JSON.stringify(['u', 'v']), JSON.stringify(OVERRIDE_AXES))
check(
  'DEFAULT_CONFIG.tiling.overrides is an empty array',
  Array.isArray(DEFAULT_CONFIG.tiling.overrides) && DEFAULT_CONFIG.tiling.overrides.length === 0,
)
check(
  'DEFAULT_TILING.overrides is an empty array',
  Array.isArray(DEFAULT_TILING.overrides) && DEFAULT_TILING.overrides.length === 0,
)
check(
  'normalizeConfig defaults tiling.overrides to [] when entirely absent',
  Array.isArray(normalizeConfig({}).tiling.overrides) && normalizeConfig({}).tiling.overrides.length === 0,
)

// --- a well-formed overrides array validates and survives normalizeConfig --
{
  const cfg = good()
  cfg.tiling = {
    ...cfg.tiling,
    overrides: [
      { i: 1, j: 2, type: '2x4', axis: 'u' },
      { i: 0, j: 0, type: '2x2' },
    ],
  }
  expectValid('a well-formed overrides array (one plate, one square) validates', cfg)
  const norm = normalizeConfig(cfg)
  check(
    'normalizeConfig keeps a well-formed plate override exactly',
    norm.tiling.overrides.some((o) => o.i === 1 && o.j === 2 && o.type === '2x4' && o.axis === 'u'),
    JSON.stringify(norm.tiling.overrides),
  )
  check(
    'normalizeConfig keeps a well-formed square override, defaulting its axis to "u"',
    norm.tiling.overrides.some((o) => o.i === 0 && o.j === 0 && o.type === '2x2' && o.axis === 'u'),
    JSON.stringify(norm.tiling.overrides),
  )
  check('normalizeConfig invents no extra overrides', norm.tiling.overrides.length === 2, JSON.stringify(norm.tiling.overrides))
}

// --- E_OVERRIDE_SHAPE: tiling.overrides itself not an array -----------------
{
  const cfg = good()
  cfg.tiling = { ...cfg.tiling, overrides: 'not an array' }
  expectError('E_OVERRIDE_SHAPE: tiling.overrides not an array', cfg, 'E_OVERRIDE_SHAPE')
  check(
    'normalizeConfig falls back to [] when tiling.overrides is not an array',
    Array.isArray(normalizeConfig(cfg).tiling.overrides) && normalizeConfig(cfg).tiling.overrides.length === 0,
  )
}

// --- E_OVERRIDE_SHAPE: malformed entries, individually and dropped together -
{
  // `i`/`non-integer` on a NUMBER (not NaN) is deliberately reported here but
  // ROUNDED by normalizeConfig, exactly like sheet.cols/rows elsewhere in this
  // file ("normalizeConfig rounds a fractional sheet.cols before clamping",
  // §3 above) — validateConfig still flags the raw non-integer input as
  // E_OVERRIDE_SHAPE, it just isn't one of the entries dropped entirely below.
  const cfg = good()
  cfg.tiling = { ...cfg.tiling, overrides: [{ i: 0, j: 1.5, type: '2x2' }] }
  expectError('E_OVERRIDE_SHAPE: non-integer j is reported on the raw input', cfg, 'E_OVERRIDE_SHAPE')
  check(
    'normalizeConfig ROUNDS a non-integer j rather than dropping it (consistent with sheet.cols/rows)',
    normalizeConfig(cfg).tiling.overrides.length === 1 && normalizeConfig(cfg).tiling.overrides[0].j === 2,
    JSON.stringify(normalizeConfig(cfg).tiling.overrides),
  )

  // These, by contrast, have no sane coercion and are DROPPED entirely.
  const cases = [
    ['not an object', 'a bare string entry'],
    [42, 'a bare number entry'],
    [null, 'a null entry'],
    [{ i: 'nope', j: 0, type: '2x2' }, 'non-numeric i'],
    [{ i: 0, j: 0, type: 'bogus' }, 'a bad type'],
    [{ i: 0, j: 0, type: '2x4' }, 'a 2x4 with no axis at all'],
    [{ i: 0, j: 0, type: '2x4', axis: 'z' }, 'a 2x4 with a bad axis'],
  ]
  for (const [entry, label] of cases) {
    const c = good()
    c.tiling = { ...c.tiling, overrides: [entry] }
    expectError(`E_OVERRIDE_SHAPE: ${label}`, c, 'E_OVERRIDE_SHAPE')
  }
  const c = good()
  c.tiling = { ...c.tiling, overrides: cases.map(([entry]) => entry) }
  check(
    'normalizeConfig drops every undroppable-malformed override, leaving none',
    normalizeConfig(c).tiling.overrides.length === 0,
    JSON.stringify(normalizeConfig(c).tiling.overrides),
  )
}

// --- E_OVERRIDE_BOUNDS: out of bounds, including a plate running off the edge
{
  // good() is the 6x8 default sheet: i in 0..5, j in 0..7.
  const cfg = good()
  cfg.tiling = { ...cfg.tiling, overrides: [{ i: 6, j: 0, type: '2x2' }] }
  const result = expectError('E_OVERRIDE_BOUNDS: i at/above sheet.cols', cfg, 'E_OVERRIDE_BOUNDS')
  check(
    'E_OVERRIDE_BOUNDS message names the sheet size',
    result.errors.some((e) => e.code === 'E_OVERRIDE_BOUNDS' && /6×8/.test(e.message)),
    describe(result),
  )
}
{
  const cfg = good()
  cfg.tiling = { ...cfg.tiling, overrides: [{ i: 0, j: 8, type: '2x2' }] }
  expectError('E_OVERRIDE_BOUNDS: j at/above sheet.rows', cfg, 'E_OVERRIDE_BOUNDS')
}
{
  const cfg = good()
  cfg.tiling = { ...cfg.tiling, overrides: [{ i: -1, j: 0, type: '2x2' }] }
  expectError('E_OVERRIDE_BOUNDS: negative i', cfg, 'E_OVERRIDE_BOUNDS')
}
{
  // The ANCHOR is in bounds, but the plate's second cell is not — exactly
  // "a plate override that would run off the sheet edge" from the spec.
  const cfg = good()
  cfg.tiling = { ...cfg.tiling, overrides: [{ i: 5, j: 0, type: '2x4', axis: 'u' }] } // i+1 = 6, off a 6-col sheet
  expectError('E_OVERRIDE_BOUNDS: a plate (axis u) running off the sheet edge', cfg, 'E_OVERRIDE_BOUNDS')
}
{
  const cfg = good()
  cfg.tiling = { ...cfg.tiling, overrides: [{ i: 0, j: 7, type: '2x4', axis: 'v' }] } // j+1 = 8, off an 8-row sheet
  expectError('E_OVERRIDE_BOUNDS: a plate (axis v) running off the sheet edge', cfg, 'E_OVERRIDE_BOUNDS')
}
{
  const cfg = good()
  cfg.tiling = { ...cfg.tiling, overrides: [{ i: 99, j: 99, type: '2x2' }] }
  check(
    'normalizeConfig drops an out-of-bounds override',
    normalizeConfig(cfg).tiling.overrides.length === 0,
    JSON.stringify(normalizeConfig(cfg).tiling.overrides),
  )
}

// --- E_OVERRIDE_CONFLICT: two overrides claiming the same cell --------------
{
  const cfg = good()
  cfg.tiling = {
    ...cfg.tiling,
    overrides: [
      { i: 2, j: 2, type: '2x2' },
      { i: 2, j: 2, type: '2x2' },
    ],
  }
  const result = expectError('E_OVERRIDE_CONFLICT: two identical square overrides', cfg, 'E_OVERRIDE_CONFLICT')
  check(
    'E_OVERRIDE_CONFLICT message names both entries and the contested cell',
    result.errors.some(
      (e) =>
        e.code === 'E_OVERRIDE_CONFLICT' &&
        /tiling\.overrides\[0\]/.test(e.message) &&
        /tiling\.overrides\[1\]/.test(e.message) &&
        /\(2, 2\)/.test(e.message),
    ),
    describe(result),
  )
}
{
  // A plate and a square disagreeing about one shared cell.
  const cfg = good()
  cfg.tiling = {
    ...cfg.tiling,
    overrides: [
      { i: 0, j: 0, type: '2x4', axis: 'u' }, // covers (0,0) and (1,0)
      { i: 1, j: 0, type: '2x2' }, // also claims (1,0)
    ],
  }
  expectError('E_OVERRIDE_CONFLICT: a plate and a square disagree about a shared cell', cfg, 'E_OVERRIDE_CONFLICT')
}
{
  // normalizeConfig resolves a conflict deterministically: first-in-array wins.
  const cfg = good()
  cfg.tiling = {
    ...cfg.tiling,
    overrides: [
      { i: 0, j: 0, type: '2x4', axis: 'u' },
      { i: 1, j: 0, type: '2x2' },
    ],
  }
  const norm = normalizeConfig(cfg)
  check(
    'normalizeConfig conflict resolution: the first override wins, the second is dropped',
    norm.tiling.overrides.length === 1 && norm.tiling.overrides[0].type === '2x4',
    JSON.stringify(norm.tiling.overrides),
  )
}
{
  // Non-adjacent, non-conflicting overrides are fine together.
  const cfg = good()
  cfg.tiling = {
    ...cfg.tiling,
    overrides: [
      { i: 0, j: 0, type: '2x2' },
      { i: 3, j: 3, type: '2x4', axis: 'v' },
    ],
  }
  expectValid('two non-conflicting overrides validate together', cfg)
}

// --- round-trip / idempotency with overrides present ------------------------
{
  const withOverrides = {
    ...good(),
    tiling: {
      ...good().tiling,
      overrides: [
        { i: 0, j: 0, type: '2x4', axis: 'u' },
        { i: 2, j: 3, type: '2x2', axis: 'u' },
      ],
    },
  }
  const once = normalizeConfig(withOverrides)
  const twice = normalizeConfig(once)
  check('normalizeConfig with overrides is idempotent', JSON.stringify(once) === JSON.stringify(twice))
  const roundTripped = JSON.parse(JSON.stringify(once))
  check(
    'normalizeConfig output with overrides survives a JSON round-trip unchanged',
    JSON.stringify(once) === JSON.stringify(roundTripped),
  )
}

// =============================================================================
// Summary
// =============================================================================
console.log('')
console.log(`test-v3-schema: ${passed} checks passed, ${failures.length} failed`)
if (failures.length > 0) {
  console.error('')
  console.error('Failures:')
  for (const f of failures) console.error(`  - ${f}`)
  process.exit(1)
}
