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
  DEFAULT_GAP,
  MAX_ROWS,
  MIN_RECTS,
  MIN_ROWS,
  SHORE_Y,
  WALL_COLUMN,
  WALL_X,
  cellPitches,
  chainStartY,
  columnChain,
  columnEndSupport,
  columnStartPitchDeg,
  frontRestY,
  normalizeConfig,
  validateConfig,
} from '../src/core/schema.js'
import {
  presetFlat,
  presetCalm,
  presetWave,
  presetCrash,
  presetRandom,
  presetSwell,
  presetSurge,
  presetWallcrash,
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
check(
  "DEFAULT_CONFIG spells out flat fronts and floor support in every column",
  DEFAULT_CONFIG.columns.every((c) => c.startPitchDeg === 0 && c.endSupport === 'floor'),
  JSON.stringify(DEFAULT_CONFIG.columns.map((c) => [c.startPitchDeg, c.endSupport])),
)
expectOk('presetFlat validates', presetFlat())
expectOk('presetCalm validates', presetCalm())
expectOk('presetWave validates', presetWave())
expectOk('presetCrash validates', presetCrash())
expectOk('presetRandom(1) validates', presetRandom(1))
expectOk('presetRandom(42) validates', presetRandom(42))
expectOk('presetSwell validates', presetSwell())
expectOk('presetSurge validates', presetSurge())
expectOk('presetWallcrash validates', presetWallcrash())

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
// 4b. PITCHED FRONTS — columns[c].startPitchDeg
//
// The pitch of ROW 0, the panel in the window. Same ±120 clamp as a hinge, same
// E_SHAPE / E_RANGE treatment. The front panel stays ON THE FLOOR at any value
// (see frontRestY below), so this is purely a shape knob.
// =============================================================================
{
  const cfg = good()
  cfg.columns[2].startPitchDeg = 35
  expectOk('startPitchDeg 35 is allowed', cfg)
}
{
  const cfg = good()
  cfg.columns[2].startPitchDeg = -35
  expectOk('startPitchDeg -35 (a diving front) is allowed', cfg)
}
{
  const cfg = good()
  cfg.columns[0].startPitchDeg = 120
  expectOk('startPitchDeg exactly 120 is allowed', cfg)
  cfg.columns[0].startPitchDeg = -120
  expectOk('startPitchDeg exactly -120 is allowed', cfg)
}
{
  const cfg = good()
  cfg.columns[3].startPitchDeg = 121
  const result = expectError('E_RANGE startPitchDeg 121', cfg, 'E_RANGE')
  check(
    'E_RANGE startPitchDeg: path names the column field',
    result.errors.some((e) => e.code === 'E_RANGE' && e.path === 'columns[3].startPitchDeg'),
    describe(result),
  )
}
expectError(
  'E_RANGE startPitchDeg -200',
  (() => {
    const cfg = good()
    cfg.columns[1].startPitchDeg = -200
    return cfg
  })(),
  'E_RANGE',
)
{
  const cfg = good()
  cfg.columns[1].startPitchDeg = 'steep'
  const result = expectError('E_SHAPE startPitchDeg not a number', cfg, 'E_SHAPE')
  check(
    'E_SHAPE startPitchDeg: message names the front / window panel',
    result.errors.some(
      (e) => e.path === 'columns[1].startPitchDeg' && /front \/ window panel/.test(e.message),
    ),
    describe(result),
  )
}
{
  const cfg = good()
  cfg.columns[1].startPitchDeg = null
  expectError('E_SHAPE startPitchDeg null', cfg, 'E_SHAPE')
}

// --- frontRestY / chainStartY: the front panel is grounded BY CONSTRUCTION ----
{
  check(
    'frontRestY(0, L) is exactly SHORE_Y for every panel length — the flat anchor',
    [60, 121, 1, 500].every((L) => frontRestY(0, L) === SHORE_Y && frontRestY(0, L) === 3.7),
    JSON.stringify([60, 121].map((L) => frontRestY(0, L))),
  )
  // Front pitched UP: the housing edge under the FRONT edge rests, so y0 = 3.7·cos θ.
  check(
    'frontRestY(30, 60) = 3.7·cos30 — front edge down at the window',
    Math.abs(frontRestY(30, 60) - 3.7 * Math.cos(Math.PI / 6)) < 1e-12,
    String(frontRestY(30, 60)),
  )
  check(
    'frontRestY is independent of L while the front pitches UP',
    frontRestY(30, 60) === frontRestY(30, 121),
    `${frontRestY(30, 60)} vs ${frontRestY(30, 121)}`,
  )
  // Front pitched DOWN: the REAR edge would go under the floor, so the whole chain
  // lifts by L·sin|θ| and the rear edge is what rests. This is the documented
  // negative-pitch behaviour: nothing clamps, the strip just starts higher.
  {
    const want = 3.7 * Math.cos((20 * Math.PI) / 180) + 60 * Math.sin((20 * Math.PI) / 180)
    check(
      'frontRestY(-20, 60) = 3.7·cos20 + 60·sin20 — rear edge down, chain lifted',
      Math.abs(frontRestY(-20, 60) - want) < 1e-12,
      `${frontRestY(-20, 60)} vs ${want}`,
    )
  }
  check(
    'frontRestY(-20, 121) lifts twice as far — a 121cm front plate dives twice as deep',
    frontRestY(-20, 121) > frontRestY(-20, 60),
    `${frontRestY(-20, 121)} vs ${frontRestY(-20, 60)}`,
  )
  // Past vertical the front edge itself is the lowest point of the solid.
  check(
    'frontRestY(120, 60) = 0 — past vertical, the front edge is the lowest point',
    frontRestY(120, 60) === 0,
    String(frontRestY(120, 60)),
  )
  check(
    'frontRestY tolerates junk (returns the flat anchor)',
    frontRestY('nope', 60) === SHORE_Y && frontRestY(0, 'nope') === SHORE_Y,
  )

  const cfg = good()
  cfg.columns[2].startPitchDeg = 30
  check(
    'chainStartY: a flat column starts at 3.7, the pitched one at 3.7·cos30',
    chainStartY(cfg, 0) === 3.7 &&
      Math.abs(chainStartY(cfg, 2) - 3.7 * Math.cos(Math.PI / 6)) < 1e-12,
    `${chainStartY(cfg, 0)} / ${chainStartY(cfg, 2)}`,
  )
  const plated = good()
  plated.columns[2].startPitchDeg = -20
  plated.rects = [{ row: 0, col: 2, orientation: 'vertical' }]
  check(
    'chainStartY: a 121cm plate at row 0 is the front panel, so its length is used',
    Math.abs(chainStartY(plated, 2) - frontRestY(-20, 121)) < 1e-12,
    `${chainStartY(plated, 2)} vs ${frontRestY(-20, 121)}`,
  )
}

// --- the pitch profile now starts at startPitchDeg ---------------------------
{
  const cfg = withFolds([null, null, [10, -20, 20, -10]])
  cfg.columns[2].startPitchDeg = 25
  check(
    'cellPitches: row 0 IS startPitchDeg and every fold accumulates on top of it',
    JSON.stringify(cellPitches(cfg, 2)) === JSON.stringify([25, 35, 15, 35, 25]),
    JSON.stringify(cellPitches(cfg, 2)),
  )
  check(
    'columnStartPitchDeg reads it back, clamped',
    columnStartPitchDeg(cfg, 2) === 25 && columnStartPitchDeg(cfg, 0) === 0,
    String(columnStartPitchDeg(cfg, 2)),
  )
  check(
    'columnStartPitchDeg clamps an out-of-range value rather than returning NaN',
    columnStartPitchDeg({ columns: [{ startPitchDeg: 400 }] }, 0) === 120 &&
      columnStartPitchDeg({ columns: [{ startPitchDeg: 'x' }] }, 0) === 0,
  )
}

// =============================================================================
// 4c. THE WALL — columns[c].endSupport
//
// 'wall' is legal ONLY on the wall-adjacent column, which is COLUMN 0: the wall
// closes the −X edge of the grid at x = WALL_X = 0, column 0's outer edge. Every
// other column has grid between it and the wall, so there is nothing to bracket to.
// =============================================================================
{
  check("WALL_X is 0 — the −X edge, column 0's outer edge", WALL_X === 0, String(WALL_X))
  check('WALL_COLUMN is 0 — the column against it', WALL_COLUMN === 0, String(WALL_COLUMN))
  check(
    "WALL_X really is column 0's outer edge on the lattice (c·(size+gap) at c = 0)",
    WALL_X === WALL_COLUMN * (DEFAULT_CONFIG.cell.size + DEFAULT_CONFIG.gap),
  )
}
{
  const cfg = good()
  cfg.columns[0].endSupport = 'wall'
  expectOk("endSupport 'wall' on column 0 is allowed", cfg)
  check('columnEndSupport reads it back', columnEndSupport(cfg, 0) === 'wall')
  check('columnEndSupport defaults the rest to floor', columnEndSupport(cfg, 5) === 'floor')
}
{
  const cfg = good()
  cfg.columns[3].endSupport = 'wall'
  const result = expectError("E_SHAPE endSupport 'wall' on column 3", cfg, 'E_SHAPE')
  check(
    'E_SHAPE wall-only: message explains that only the wall-adjacent column qualifies',
    result.errors.some(
      (e) =>
        e.code === 'E_SHAPE' &&
        e.path === 'columns[3].endSupport' &&
        /only valid for the WALL-ADJACENT column/.test(e.message) &&
        /columns\[0\]/.test(e.message) &&
        /the first one/.test(e.message),
    ),
    describe(result),
  )
}
{
  // The LAST column used to be the wall-adjacent one; it is now an error like any
  // other non-zero column. This is the regression guard on the flip.
  const cfg = good()
  cfg.columns[5].endSupport = 'wall'
  const result = expectError("E_SHAPE endSupport 'wall' on column 5 (the far jamb)", cfg, 'E_SHAPE')
  check(
    'E_SHAPE wall-only: the message points at columns[0], not the clicked column',
    result.errors.some(
      (e) => e.path === 'columns[5].endSupport' && /columns\[0\]/.test(e.message),
    ),
    describe(result),
  )
}
{
  const cfg = good()
  cfg.columns[1].endSupport = 'wall'
  expectError("E_SHAPE endSupport 'wall' on column 1 (one in from the wall)", cfg, 'E_SHAPE')
}
{
  const cfg = good()
  cfg.columns[0].endSupport = 'ceiling'
  const result = expectError('E_SHAPE endSupport enum', cfg, 'E_SHAPE')
  check(
    'E_SHAPE endSupport enum: message lists the two legal values',
    result.errors.some(
      (e) =>
        e.path === 'columns[0].endSupport' && /"floor" or "wall"/.test(e.message),
    ),
    describe(result),
  )
}
{
  const cfg = good()
  cfg.columns[0].endSupport = null
  expectError('E_SHAPE endSupport null', cfg, 'E_SHAPE')
}

// =============================================================================
// 4d. ROW RESOLUTION — grid.rows must be MIN_ROWS..MAX_ROWS (5..8)
//
// Below 5 the design stops reading as a shore with a wave breaking behind it;
// above 8 a 61cm-pitch strip runs out of store. foldsDeg length is checked against
// grid.rows, never against a literal.
// =============================================================================
/**
 * A config at `rows` rows, with matching (all-zero) fold sequences and no rects.
 * `rows` may be junk (6.5, 0) — the fold count then just rounds, since the point of
 * those cases is that `grid.rows` itself is rejected.
 */
const withRows = (rows) => {
  const cfg = good()
  cfg.grid = { cols: 6, rows }
  const folds = Math.max(0, Math.round(rows) - 1)
  cfg.columns = cfg.columns.map(() => ({ foldsDeg: new Array(folds).fill(0) }))
  cfg.rects = []
  return cfg
}
{
  check('MIN_ROWS is 5, MAX_ROWS is 8', MIN_ROWS === 5 && MAX_ROWS === 8, `${MIN_ROWS}..${MAX_ROWS}`)
  for (let rows = MIN_ROWS; rows <= MAX_ROWS; rows++) {
    const cfg = withRows(rows)
    expectOk(`grid.rows ${rows} is accepted with ${rows - 1} folds per column`, cfg)
    check(
      `grid.rows ${rows}: every pitch profile has ${rows} entries`,
      cellPitches(cfg, 0).length === rows,
      String(cellPitches(cfg, 0).length),
    )
  }
}
{
  const result = expectError('E_SHAPE grid.rows 4', withRows(4), 'E_SHAPE')
  check(
    'E_SHAPE grid.rows: path and message name the band and the reason',
    result.errors.some(
      (e) =>
        e.code === 'E_SHAPE' &&
        e.path === 'grid.rows' &&
        /integer in 5\.\.8/.test(e.message) &&
        /wave breaking/.test(e.message),
    ),
    describe(result),
  )
}
expectError('E_SHAPE grid.rows 9', withRows(9), 'E_SHAPE')
expectError('E_SHAPE grid.rows 1', withRows(1), 'E_SHAPE')
expectError('E_SHAPE grid.rows 0', withRows(0), 'E_SHAPE')
expectError('E_SHAPE grid.rows 6.5', withRows(6.5), 'E_SHAPE')
{
  // foldsDeg length is measured against grid.rows, not against 5.
  const cfg = withRows(7)
  cfg.columns[2] = { foldsDeg: [0, 0, 0, 0] } // 4 folds where 6 are needed
  const result = expectError('E_SHAPE foldsDeg length at rows 7', cfg, 'E_SHAPE')
  check(
    'E_SHAPE foldsDeg length: the message quotes grid.rows-1 (6), not 4',
    result.errors.some(
      (e) => e.path === 'columns[2].foldsDeg' && /exactly grid\.rows-1 \(6\)/.test(e.message),
    ),
    describe(result),
  )
}
{
  // Rect bounds follow grid.rows too.
  const cfg = withRows(8)
  cfg.rects = [{ row: 6, col: 1, orientation: 'vertical' }] // needs row ≤ 6 at rows 8
  expectOk('vertical rect at row 6 is in bounds at rows 8', cfg)
  cfg.rects = [{ row: 7, col: 1, orientation: 'vertical' }]
  expectError('vertical rect at row 7 is out of bounds even at rows 8', cfg, 'E_SHAPE')
}

// =============================================================================
// 4e. BACKWARD COMPATIBILITY — a config written before the new fields existed
// =============================================================================
{
  const legacy = {
    version: 2,
    units: 'cm',
    grid: { cols: 6, rows: 5 },
    columns: [
      { foldsDeg: [10, -10, 0, 0] },
      { foldsDeg: [10, -10, 0, 0] },
      { foldsDeg: [0, 0, 0, 0] },
      { foldsDeg: [0, 0, 0, 0] },
      { foldsDeg: [0, 0, 0, 0] },
      { foldsDeg: [0, 0, 0, 0] },
    ],
  }
  const cfg = normalizeConfig(legacy)
  check(
    'normalize: a column without startPitchDeg gets 0 (a flat front)',
    cfg.columns.every((col) => col.startPitchDeg === 0),
    JSON.stringify(cfg.columns.map((col) => col.startPitchDeg)),
  )
  check(
    "normalize: a column without endSupport gets 'floor'",
    cfg.columns.every((col) => col.endSupport === 'floor'),
    JSON.stringify(cfg.columns.map((col) => col.endSupport)),
  )
  expectOk('a pre-pitched-fronts config still validates', legacy)
  check(
    'normalize: an EXPLICIT startPitchDeg / endSupport is passed through',
    (() => {
      const src = structuredClone(legacy)
      src.columns[5] = { foldsDeg: [0, 0, 0, 0], startPitchDeg: -12, endSupport: 'wall' }
      const out = normalizeConfig(src)
      return out.columns[5].startPitchDeg === -12 && out.columns[5].endSupport === 'wall'
    })(),
  )
  check(
    'the flat chain is untouched by the new fields (still starts at 3.7, z 0)',
    JSON.stringify(columnChain(legacy, 3).points[0]) === JSON.stringify([3.7, 0]),
    JSON.stringify(columnChain(legacy, 3).points[0]),
  )
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
  // Column 0 goes up and comes back down: pitch 0 again at row 2, but ~30cm higher
  // and ~8cm shallower than column 1.
  const cfg = withFolds([[30, -30, 0, 0]])
  cfg.rects = [{ row: 2, col: 0, orientation: 'horizontal' }]
  const result = expectWarning('W_CROSSCOL_POSITION', cfg, 'W_CROSSCOL_POSITION')
  check('W_CROSSCOL_POSITION is only a warning', result.ok, describe(result))
  check(
    'W_CROSSCOL_POSITION message reports both origins',
    result.warnings.some((w) => w.code === 'W_CROSSCOL_POSITION' && /y 34\.22/.test(w.message)),
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
// 10. W_RECT_LENGTH — fires only when rectLength ≠ 2·size + gap
//
// At the DEFAULTS it must stay silent: 2·60 + 1 = 121 = cell.rectLength exactly,
// which is the whole point of the 1cm gap — a rect plate is a drop-in for two
// squares plus the joint between them.
// =============================================================================
{
  check(
    'the default gap is 1.0cm',
    DEFAULT_GAP === 1.0 && DEFAULT_CONFIG.gap === 1.0,
    `DEFAULT_GAP=${DEFAULT_GAP} DEFAULT_CONFIG.gap=${DEFAULT_CONFIG.gap}`,
  )
  check(
    'the defaults satisfy 2·cell.size + gap = cell.rectLength (60 + 1 + 60 = 121)',
    2 * DEFAULT_CONFIG.cell.size + DEFAULT_CONFIG.gap === DEFAULT_CONFIG.cell.rectLength,
    `${2 * DEFAULT_CONFIG.cell.size + DEFAULT_CONFIG.gap} vs ${DEFAULT_CONFIG.cell.rectLength}`,
  )
}
{
  // Defaults + a horizontal rect: an exact fit, so NO warning at all.
  const cfg = good()
  cfg.rects = [{ row: 1, col: 2, orientation: 'horizontal' }]
  const result = validateConfig(cfg)
  check(
    'defaults + a rect → no W_RECT_LENGTH (exact fit)',
    result.ok && !warnCodes(result).includes('W_RECT_LENGTH'),
    describe(result),
  )
}
{
  // …and with a vertical rect, which is the orientation the slack used to
  // propagate along.
  const cfg = good()
  cfg.rects = [{ row: 1, col: 2, orientation: 'vertical' }]
  const result = validateConfig(cfg)
  check(
    'defaults + a vertical rect → no W_RECT_LENGTH either',
    result.ok && !warnCodes(result).includes('W_RECT_LENGTH'),
    describe(result),
  )
}
{
  // FORCED MISMATCH: the old default gap against today's 121cm hardware. The rule
  // still fires, names both numbers and reports the 1cm of slack.
  const cfg = good()
  cfg.gap = 2
  cfg.rects = [{ row: 1, col: 2, orientation: 'horizontal' }]
  const result = expectWarning('W_RECT_LENGTH (gap 2 vs rectLength 121)', cfg, 'W_RECT_LENGTH')
  check('W_RECT_LENGTH is only a warning', result.ok, describe(result))
  check(
    'W_RECT_LENGTH names rectLength, the ideal 122 and the 1.00cm of slack',
    result.warnings.some(
      (w) =>
        w.code === 'W_RECT_LENGTH' &&
        w.path === 'cell.rectLength' &&
        /121cm/.test(w.message) &&
        /122cm/.test(w.message) &&
        /1\.00cm of slack/.test(w.message),
    ),
    JSON.stringify(result.warnings.map((w) => w.message)),
  )
}
{
  // The other direction: a plate LONGER than the slot also mismatches.
  const cfg = good()
  cfg.cell.rectLength = 130
  cfg.rects = [{ row: 1, col: 2, orientation: 'horizontal' }]
  expectWarning('W_RECT_LENGTH (rectLength 130 vs ideal 121)', cfg, 'W_RECT_LENGTH')
}
{
  // A mismatched combination with NO rects stays silent — nothing to be slack in.
  const cfg = good()
  cfg.gap = 2
  const result = validateConfig(cfg)
  check(
    'no W_RECT_LENGTH without rects, even on a mismatched combination',
    !warnCodes(result).includes('W_RECT_LENGTH'),
    describe(result),
  )
}
{
  // gap 2 becomes exact again once the plate matches it.
  const cfg = good()
  cfg.gap = 2
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
// 10b. W_FEW_RECTS — the plate-pattern rule
//
// Every design must use at least MIN_RECTS = 4 of the 60×121 plates, placed by one
// nameable rule (recorded in meta.rectPattern). Fewer is a WARNING, not an error:
// the grid map removes plates one click at a time, so a design mid-repattern has to
// stay committable.
// =============================================================================
{
  check('MIN_RECTS is 4', MIN_RECTS === 4, String(MIN_RECTS))
}
/** Four legal vertical plates on a flat grid — the 'mirrored quad' pattern. */
const fourPlates = () => [
  { row: 1, col: 1, orientation: 'vertical' },
  { row: 3, col: 1, orientation: 'vertical' },
  { row: 1, col: 4, orientation: 'vertical' },
  { row: 3, col: 4, orientation: 'vertical' },
]
{
  // 0 plates — the bare DEFAULT_CONFIG geometry.
  const result = expectWarning('W_FEW_RECTS at 0 rects', good(), 'W_FEW_RECTS')
  check('W_FEW_RECTS is only a warning', result.ok, describe(result))
  check(
    'W_FEW_RECTS names the threshold, the count and the pattern requirement',
    result.warnings.some(
      (w) =>
        w.code === 'W_FEW_RECTS' &&
        w.path === 'rects' &&
        /≥ 4 rect plates/.test(w.message) &&
        /got 0/.test(w.message) &&
        /should read in the pattern/.test(w.message) &&
        /meta\.rectPattern/.test(w.message),
    ),
    JSON.stringify(result.warnings.map((w) => w.message)),
  )
}
{
  // 3 plates — still short of the rule.
  const cfg = good()
  cfg.rects = fourPlates().slice(0, 3)
  const result = expectWarning('W_FEW_RECTS at 3 rects', cfg, 'W_FEW_RECTS')
  check(
    'W_FEW_RECTS reports the actual count (3)',
    result.warnings.some((w) => w.code === 'W_FEW_RECTS' && /got 3/.test(w.message)),
    JSON.stringify(result.warnings.map((w) => w.message)),
  )
}
{
  // 4 plates — silent.
  const cfg = good()
  cfg.rects = fourPlates()
  const result = validateConfig(cfg)
  check(
    'no W_FEW_RECTS at exactly 4 rects',
    result.ok && !warnCodes(result).includes('W_FEW_RECTS'),
    describe(result),
  )
}
{
  // More than 4 — also silent.
  const cfg = good()
  cfg.rects = [...fourPlates(), { row: 1, col: 2, orientation: 'vertical' }]
  const result = validateConfig(cfg)
  check(
    'no W_FEW_RECTS at 5 rects',
    result.ok && !warnCodes(result).includes('W_FEW_RECTS'),
    describe(result),
  )
}
{
  // Warning ORDER follows the doc order in core/schema.js: W_CROSSCOL_POSITION,
  // then W_RECT_LENGTH, then W_FEW_RECTS.
  const cfg = withFolds([[30, -30, 0, 0]])
  cfg.gap = 2 // forces W_RECT_LENGTH against the 121cm plate
  cfg.rects = [{ row: 2, col: 0, orientation: 'horizontal' }] // and W_CROSSCOL_POSITION
  const result = validateConfig(cfg)
  check(
    'warnings come back in doc order (W_CROSSCOL_POSITION, W_RECT_LENGTH, W_FEW_RECTS)',
    JSON.stringify(warnCodes(result)) ===
      JSON.stringify(['W_CROSSCOL_POSITION', 'W_RECT_LENGTH', 'W_FEW_RECTS']),
    describe(result),
  )
}
{
  // Every preset satisfies the rule and names its pattern.
  const named = [
    ['presetFlat', presetFlat()],
    ['presetCalm', presetCalm()],
    ['presetWave', presetWave()],
    ['presetCrash', presetCrash()],
    ['presetRandom(1)', presetRandom(1)],
    ['presetRandom(42)', presetRandom(42)],
    ['presetSwell', presetSwell()],
    ['presetSurge', presetSurge()],
    ['presetWallcrash', presetWallcrash()],
  ]
  for (const [label, cfg] of named) {
    const result = validateConfig(cfg)
    check(
      `${label}: ≥ ${MIN_RECTS} plates, no W_FEW_RECTS, meta.rectPattern set`,
      cfg.rects.length >= MIN_RECTS &&
        !warnCodes(result).includes('W_FEW_RECTS') &&
        typeof cfg.meta.rectPattern === 'string' &&
        cfg.meta.rectPattern.length > 0,
      `rects=${cfg.rects.length} pattern=${JSON.stringify(cfg.meta.rectPattern)} ${describe(result)}`,
    )
  }
}
{
  // DEFAULT_CONFIG is the bare reference geometry, not a design — it is the one
  // config that raises W_FEW_RECTS on purpose.
  check(
    'DEFAULT_CONFIG raises W_FEW_RECTS (it is reference geometry, not a design)',
    warnCodes(validateConfig(DEFAULT_CONFIG)).includes('W_FEW_RECTS'),
    describe(validateConfig(DEFAULT_CONFIG)),
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
    'columnChain: flat column origins are (3.7, 61r) — the 60cm cell + 1cm gap pitch',
    chain.origins.every(([y, z], r) => Math.abs(y - 3.7) < 1e-12 && Math.abs(z - 61 * r) < 1e-12),
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

  check('normalize: gap default is 1.0', cfg.gap === 1.0 && cfg.gap === DEFAULT_GAP, String(cfg.gap))
  check(
    'normalize: an explicit gap is passed through',
    normalizeConfig({ ...minimal, gap: 2.5 }).gap === 2.5,
  )
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
