/**
 * tests/test-v4-schema.mjs — headless checks for core/v4/schema.js.
 *
 * Plain node script, no test framework. Exits non-zero on any failure.
 *   node tests/test-v4-schema.mjs
 *
 * The two things worth testing about a schema are the two halves of its
 * contract, and both are easy to fake:
 *
 *   normalizeConfig  is IDEMPOTENT — asserted by deep-equal on the second pass,
 *        not by eyeballing one field.
 *   validateConfig   RANGE-CHECKS EVERY RANGED KNOB — asserted one knob at a
 *        time, each with a NON-VACUOUS NEGATIVE: the out-of-range value must
 *        raise E_RANGE *at that knob's path*, and the in-range value must be
 *        accepted. A test that only checks `valid === false` passes just as
 *        happily when a different knob is broken.
 *
 * That second discipline exists because v3's schema clamped two faceting knobs
 * without ever checking them (HANDOFF §5.2, §6), and the tests as written did
 * not notice.
 *
 * §8 is new and is the one that protects a user's saved work: a v4 config of the
 * earlier RIBBON shape must be REFUSED, not normalized into a default lattice.
 * `version` cannot catch it — V4_SPEC §9 generalised the model without moving
 * the number — so `E_LEGACY_SHAPE` is the only thing standing between a stale
 * localStorage design and a silent reset.
 */

import { deepStrictEqual } from 'node:assert'
import {
  ANGLE_MAX,
  ANGLE_MIN,
  CONNECTOR_BIN_ANGLE_MAX,
  CONNECTOR_BIN_ANGLE_MIN,
  CONNECTOR_BIN_SPAN_MAX,
  CONNECTOR_BIN_SPAN_MIN,
  CONNECTOR_LENGTH_MAX,
  CONNECTOR_LENGTH_MIN,
  CONNECTOR_MIN_PER_JOINT_MAX,
  CONNECTOR_MIN_PER_JOINT_MIN,
  CONNECTOR_POWER_EDGES,
  CONNECTOR_SPACING_MAX,
  CONNECTOR_SPACING_MIN,
  CONNECTOR_SUPPLY_MODES,
  DEFAULT_CONFIG,
  DEFAULT_CONNECTORS,
  EDGE_AXES,
  GAP_MAX,
  GAP_MIN,
  Y_OFFSET_MAX,
  Y_OFFSET_MIN,
  DEFAULT_PLACEMENT,
  LATTICE_COLS_MAX,
  LATTICE_COLS_MIN,
  LATTICE_ROWS_MAX,
  LATTICE_ROWS_MIN,
  PANEL_TYPES,
  PATTERN_KINDS,
  PHASE_MAX,
  PHASE_MIN,
  WALL_ANCHORS,
  WALL_OFFSET_MAX,
  WALL_OFFSET_MIN,
  WINDOW_OFFSET_MAX,
  WINDOW_OFFSET_MIN,
  clamp,
  edgeGridSize,
  normalizeConfig,
  validateConfig,
} from '../src/core/v4/schema.js'
import { CONNECTOR_LIMITS } from '../src/core/v3/connectors.js'
import {
  CONNECTOR_BIN_ANGLE_MAX as V3_BIN_ANGLE_MAX,
  CONNECTOR_BIN_ANGLE_MIN as V3_BIN_ANGLE_MIN,
  CONNECTOR_BIN_SPAN_MAX as V3_BIN_SPAN_MAX,
  CONNECTOR_BIN_SPAN_MIN as V3_BIN_SPAN_MIN,
  CONNECTOR_LENGTH_MAX as V3_LENGTH_MAX,
  CONNECTOR_LENGTH_MIN as V3_LENGTH_MIN,
  CONNECTOR_MIN_PER_JOINT_MAX as V3_MIN_PER_JOINT_MAX,
  CONNECTOR_MIN_PER_JOINT_MIN as V3_MIN_PER_JOINT_MIN,
  CONNECTOR_SPACING_MAX as V3_SPACING_MAX,
  CONNECTOR_SPACING_MIN as V3_SPACING_MIN,
  DEFAULT_CONNECTORS as V3_DEFAULT_CONNECTORS,
} from '../src/core/v3/schema.js'

let passed = 0
let failed = 0
const ok = (c, m) => { if (c) passed++; else { failed++; console.log(`  FAIL: ${m}`) } }
const deepEq = (a, b, m) => {
  try { deepStrictEqual(a, b); passed++ } catch { failed++; console.log(`  FAIL: ${m}`) }
}

const good = () => JSON.parse(JSON.stringify(DEFAULT_CONFIG))

/** Does validation reject `config` with `code` at `path`? */
const rejects = (config, code, path) => {
  const v = validateConfig(config)
  return !v.valid && v.errors.some((e) => e.code === code && e.path === path)
}
/** Is `path` free of errors? (The other half of a non-vacuous negative.) */
const accepts = (config, path) => {
  const v = validateConfig(config)
  return !v.errors.some((e) => e.path === path)
}

console.log('=== test-v4-schema ===')

// -----------------------------------------------------------------------------
// 1. The default config is the fixed point of both halves.
// -----------------------------------------------------------------------------
console.log('1. the default config')
{
  const v = validateConfig(DEFAULT_CONFIG)
  ok(v.valid, `DEFAULT_CONFIG validates (errors: ${JSON.stringify(v.errors)})`)
  ok(v.warnings.length === 0, `and raises no warnings (got ${JSON.stringify(v.warnings)})`)
  deepEq(normalizeConfig(DEFAULT_CONFIG), good(), 'normalizeConfig(DEFAULT_CONFIG) is DEFAULT_CONFIG')
  ok(DEFAULT_CONFIG.version === 4, 'version is 4')
  ok(DEFAULT_CONFIG.lattice.cols === 3 && DEFAULT_CONFIG.lattice.rows === 5,
    'the default lattice is 3 × 5')
  ok(DEFAULT_CONFIG.angleDeg === 30 && DEFAULT_CONFIG.gap === 2, 'at 30° and a 2cm gap')
  ok(DEFAULT_CONFIG.pattern.phase === 0, 'phase 0')
  ok(DEFAULT_CONFIG.placement.wallAnchor === 'free', 'and free of the wall')
  ok(DEFAULT_CONFIG.strip === undefined, 'and it carries no `strip` at all')
  ok(Object.isFrozen(DEFAULT_CONFIG), 'DEFAULT_CONFIG is frozen')
  // The freeze is shallow, so the arrays must not be shared with anything.
  const a = normalizeConfig({})
  const b = normalizeConfig({})
  ok(a.overrides.cells !== b.overrides.cells && a.overrides.edges !== b.overrides.edges &&
     a.overrides.cells !== DEFAULT_CONFIG.overrides.cells,
    'every normalized config gets its own override arrays')
}

// -----------------------------------------------------------------------------
// 2. The ranges this file deliberately RESTATES rather than imports.
//
// schema.js keeps the gap band and the connector bands as literals so it has no
// dependencies at all. A deliberate duplication is only safe while something
// checks it, and this is that something.
// -----------------------------------------------------------------------------
console.log('2. restated ranges still agree with their sources')
{
  ok(GAP_MIN === CONNECTOR_LIMITS.minSpanCm,
    `GAP_MIN (${GAP_MIN}) is CONNECTOR_LIMITS.minSpanCm (${CONNECTOR_LIMITS.minSpanCm})`)
  ok(GAP_MAX === CONNECTOR_LIMITS.maxSpanCm,
    `GAP_MAX (${GAP_MAX}) is CONNECTOR_LIMITS.maxSpanCm (${CONNECTOR_LIMITS.maxSpanCm})`)
  ok(CONNECTOR_LENGTH_MIN === V3_LENGTH_MIN && CONNECTOR_LENGTH_MAX === V3_LENGTH_MAX,
    'connector length band matches v3')
  ok(CONNECTOR_SPACING_MIN === V3_SPACING_MIN && CONNECTOR_SPACING_MAX === V3_SPACING_MAX,
    'connector spacing band matches v3')
  ok(CONNECTOR_MIN_PER_JOINT_MIN === V3_MIN_PER_JOINT_MIN &&
     CONNECTOR_MIN_PER_JOINT_MAX === V3_MIN_PER_JOINT_MAX, 'minPerJoint band matches v3')
  ok(CONNECTOR_BIN_SPAN_MIN === V3_BIN_SPAN_MIN && CONNECTOR_BIN_SPAN_MAX === V3_BIN_SPAN_MAX,
    'bin span band matches v3')
  ok(CONNECTOR_BIN_ANGLE_MIN === V3_BIN_ANGLE_MIN && CONNECTOR_BIN_ANGLE_MAX === V3_BIN_ANGLE_MAX,
    'bin angle band matches v3')
  deepEq(DEFAULT_CONNECTORS, V3_DEFAULT_CONNECTORS, 'the connector defaults are v3\'s, unchanged')

  // The level field has PERIOD 2, so the phase band is 0..1 rather than the
  // ribbon's 0..3 — otherwise `phase: 3` would silently mean `phase: 1`.
  ok(PHASE_MIN === 0 && PHASE_MAX === 1, 'the phase band is 0..1, the period of the level field')
  // '2x4' is gone: the lattice pitch is one number in both axes, and a 60 × 121
  // plate does not have one plan size (V4_SPEC §9.8).
  deepEq(PANEL_TYPES, ['2x2'], 'only 2x2 — a rectangular panel needs a rectangular lattice')
  deepEq(EDGE_AXES, ['x', 'z'], 'an edge crosses x or z')
  deepEq(WALL_ANCHORS, ['free', 'braced'], 'and the wall is either used or not')
}

// -----------------------------------------------------------------------------
// 3. edgeGridSize — the bounds rule three places have to agree on.
// -----------------------------------------------------------------------------
console.log('3. how many edges a lattice has')
{
  deepEq(edgeGridSize('x', 3, 5), { iCount: 2, jCount: 5 }, '3 × 5 has 2 × 5 = 10 x-edges')
  deepEq(edgeGridSize('z', 3, 5), { iCount: 3, jCount: 4 }, 'and 3 × 4 = 12 z-edges')
  // A one-column lattice is the ribbon: no x-edges at all.
  ok(edgeGridSize('x', 1, 5).iCount === 0, 'a 1-column lattice has no x-edges — that is the ribbon')
  ok(edgeGridSize('z', 1, 5).jCount === 4, 'and four z-edges, its four angled units')
  ok(edgeGridSize('z', 3, 1).jCount === 0, 'a 1-row lattice has no z-edges')
  ok(edgeGridSize('x', 1, 1).iCount === 0 && edgeGridSize('z', 1, 1).jCount === 0,
    'a single cell has no edges of either kind')
}

// -----------------------------------------------------------------------------
// 4. clamp, and normalizeConfig's filling + clamping.
// -----------------------------------------------------------------------------
console.log('4. normalizeConfig fills and clamps')
{
  ok(clamp(5, 0, 10) === 5 && clamp(-1, 0, 10) === 0 && clamp(11, 0, 10) === 10, 'clamp')

  // `name` is the one field that is passed through only when it is PRESENT —
  // v3's contract, kept — so an empty config normalizes to the default without
  // acquiring a name it was never given.
  deepEq(normalizeConfig({}), (() => { const g = good(); delete g.name; return g })(),
    'normalizeConfig({}) is the default config, minus the name it was not given')
  ok(normalizeConfig({}).name === undefined, 'and an unnamed config stays unnamed')
  ok(normalizeConfig({ name: 'study' }).name === 'study', 'a name given is a name kept')

  const wild = normalizeConfig({
    lattice: { cols: 99, rows: -4, panelType: 'nope' },
    gap: 1e6,
    angleDeg: -20,
    pattern: { kind: 'zigzag', phase: 17 },
    placement: { wallOffsetCm: -5, windowOffsetCm: 1e9, groundToFloor: 'yes', yOffsetCm: 1e6,
      wallAnchor: 'welded' },
    connectors: { lengthCm: 0, spacingCm: 1e4, minPerJoint: 99, binSpanCm: 0, binAngleDeg: 1e3,
      powerEdge: 'sideways', supplyMode: 'melt' },
  })
  ok(wild.lattice.cols === LATTICE_COLS_MAX, 'lattice.cols clamps up')
  ok(wild.lattice.rows === LATTICE_ROWS_MIN, 'lattice.rows clamps down')
  ok(wild.lattice.panelType === '2x2', 'an unknown panelType falls back')
  ok(wild.gap === GAP_MAX, 'gap clamps')
  ok(wild.angleDeg === ANGLE_MIN, 'angleDeg clamps')
  ok(wild.pattern.kind === 'trapezoid', 'an unknown pattern kind falls back')
  ok(wild.pattern.phase === PHASE_MAX, 'phase clamps')
  ok(wild.placement.wallOffsetCm === WALL_OFFSET_MIN, 'wallOffsetCm clamps')
  ok(wild.placement.windowOffsetCm === WINDOW_OFFSET_MAX, 'windowOffsetCm clamps')
  ok(wild.placement.groundToFloor === true, 'groundToFloor coerces to a boolean')
  ok(wild.placement.yOffsetCm === Y_OFFSET_MAX, 'yOffsetCm clamps up')
  ok(normalizeConfig({ placement: { yOffsetCm: -1e6 } }).placement.yOffsetCm === Y_OFFSET_MIN,
    'and down — the band is negative at one end, so both ends are checked')
  ok(wild.placement.wallAnchor === 'free', 'an unknown wallAnchor falls back')
  ok(wild.connectors.lengthCm === CONNECTOR_LENGTH_MIN, 'connectors.lengthCm clamps')
  ok(wild.connectors.spacingCm === CONNECTOR_SPACING_MAX, 'connectors.spacingCm clamps')
  ok(wild.connectors.minPerJoint === CONNECTOR_MIN_PER_JOINT_MAX, 'connectors.minPerJoint clamps')
  ok(wild.connectors.binSpanCm === CONNECTOR_BIN_SPAN_MIN, 'connectors.binSpanCm clamps')
  ok(wild.connectors.binAngleDeg === CONNECTOR_BIN_ANGLE_MAX, 'connectors.binAngleDeg clamps')
  ok(wild.connectors.powerEdge === 'low' && wild.connectors.supplyMode === 'relief',
    'unknown connector enums fall back')

  // Integer knobs really are integers after normalization.
  const frac = normalizeConfig({ lattice: { cols: 2.6, rows: 8.4 }, pattern: { phase: 0.7 },
    connectors: { minPerJoint: 2.5 } })
  ok(Number.isInteger(frac.lattice.cols) && frac.lattice.cols === 3, 'lattice.cols rounds')
  ok(Number.isInteger(frac.lattice.rows) && frac.lattice.rows === 8, 'lattice.rows rounds')
  ok(Number.isInteger(frac.pattern.phase) && frac.pattern.phase === 1, 'phase rounds')
  ok(Number.isInteger(frac.connectors.minPerJoint), 'minPerJoint rounds')

  // A non-numeric value is not coerced to NaN — it falls back to the default.
  const junk = normalizeConfig({ gap: 'wide', angleDeg: null, lattice: { rows: 'five' } })
  ok(junk.gap === DEFAULT_CONFIG.gap, 'a non-numeric gap falls back to the default')
  ok(junk.angleDeg === DEFAULT_CONFIG.angleDeg, 'a null angleDeg falls back')
  ok(junk.lattice.rows === DEFAULT_CONFIG.lattice.rows, 'a non-numeric rows falls back')

  ok(normalizeConfig(null).version === 4, 'normalizeConfig(null) is the default config')
  ok(normalizeConfig(undefined).gap === DEFAULT_CONFIG.gap, 'normalizeConfig(undefined) too')

  // A malformed `overrides` is replaced by empty tables rather than exploding.
  deepEq(normalizeConfig({ overrides: 'nope' }).overrides, { cells: [], edges: [] },
    'a non-object overrides normalizes to empty tables')
  deepEq(normalizeConfig({ overrides: { cells: 3, edges: null } }).overrides, { cells: [], edges: [] },
    'and so do non-array tables')

  // Never mutates its input.
  const input = { gap: 99, overrides: { cells: [{ i: 0, j: 0, present: false }], edges: [] } }
  const before = JSON.stringify(input)
  normalizeConfig(input)
  ok(JSON.stringify(input) === before, 'normalizeConfig never mutates its argument')
}

// -----------------------------------------------------------------------------
// 5. IDEMPOTENCE — by deep-equal, over a spread of hostile inputs.
// -----------------------------------------------------------------------------
console.log('5. normalizeConfig is idempotent')
{
  const cases = [
    {},
    null,
    DEFAULT_CONFIG,
    { gap: 1e6, angleDeg: -20, lattice: { cols: 99, rows: 2.4 } },
    { pattern: { phase: 9 }, placement: { groundToFloor: 0, wallAnchor: 'braced' } },
    { overrides: {
      cells: [
        { i: 0, j: 2, present: false },
        { i: 1, j: 1, flipped: true },
        { i: 1, j: 1, present: false },   // duplicate — dropped
        { i: 0, j: 99 },                  // out of grid — dropped
        { i: 2, j: 2 },                   // entirely default — dropped
        'nonsense',
      ],
      edges: [
        { i: 0, j: 0, axis: 'x', present: false },
        { i: 0, j: 0, axis: 'z', present: true },   // entirely default — dropped
        { i: 9, j: 0, axis: 'x', present: false },  // out of grid — dropped
        { i: 0, j: 0, axis: 'y', present: false },  // unknown axis — dropped
      ],
    } },
    { overrides: 'nope' },
    { connectors: { lengthCm: 4.000001, minPerJoint: 1 } },
    { name: 'x', meta: { notes: 'hello', extra: 1 } },
  ]
  for (const c of cases) {
    const once = normalizeConfig(c)
    const twice = normalizeConfig(once)
    deepEq(twice, once, `idempotent on ${JSON.stringify(c)?.slice(0, 50)}`)
    // And byte-stable through JSON, which is how it reaches localStorage.
    deepEq(normalizeConfig(JSON.parse(JSON.stringify(once))), once,
      `survives a JSON round-trip: ${JSON.stringify(c)?.slice(0, 40)}`)
  }
}

// -----------------------------------------------------------------------------
// 6. EVERY ranged knob, one at a time, with a non-vacuous negative.
//
// Each row asserts three things: below the band is E_RANGE at that exact path,
// above the band is E_RANGE at that exact path, and a value INSIDE the band
// raises nothing at that path. The third is what makes the first two mean
// something.
// -----------------------------------------------------------------------------
console.log('6. every ranged knob is range-checked')
{
  const set = (path, value) => {
    const cfg = good()
    const parts = path.split('.')
    let node = cfg
    for (let i = 0; i < parts.length - 1; i++) node = node[parts[i]]
    node[parts[parts.length - 1]] = value
    return cfg
  }

  const ranged = [
    ['lattice.cols', LATTICE_COLS_MIN, LATTICE_COLS_MAX, 1],
    ['lattice.rows', LATTICE_ROWS_MIN, LATTICE_ROWS_MAX, 1],
    ['gap', GAP_MIN, GAP_MAX, 0.1],
    ['angleDeg', ANGLE_MIN, ANGLE_MAX, 1],
    ['pattern.phase', PHASE_MIN, PHASE_MAX, 1],
    ['placement.wallOffsetCm', WALL_OFFSET_MIN, WALL_OFFSET_MAX, 1],
    ['placement.windowOffsetCm', WINDOW_OFFSET_MIN, WINDOW_OFFSET_MAX, 1],
    ['placement.yOffsetCm', Y_OFFSET_MIN, Y_OFFSET_MAX, 1],
    ['connectors.lengthCm', CONNECTOR_LENGTH_MIN, CONNECTOR_LENGTH_MAX, 0.5],
    ['connectors.spacingCm', CONNECTOR_SPACING_MIN, CONNECTOR_SPACING_MAX, 1],
    ['connectors.minPerJoint', CONNECTOR_MIN_PER_JOINT_MIN, CONNECTOR_MIN_PER_JOINT_MAX, 1],
    ['connectors.binSpanCm', CONNECTOR_BIN_SPAN_MIN, CONNECTOR_BIN_SPAN_MAX, 0.01],
    ['connectors.binAngleDeg', CONNECTOR_BIN_ANGLE_MIN, CONNECTOR_BIN_ANGLE_MAX, 0.1],
  ]

  for (const [path, lo, hi, step] of ranged) {
    ok(rejects(set(path, lo - step), 'E_RANGE', path), `${path} below ${lo} → E_RANGE at ${path}`)
    ok(rejects(set(path, hi + step), 'E_RANGE', path), `${path} above ${hi} → E_RANGE at ${path}`)
    // The non-vacuous half: both endpoints and the middle are ACCEPTED.
    ok(accepts(set(path, lo), path), `${path} = ${lo} (the bottom of the band) is accepted`)
    ok(accepts(set(path, hi), path), `${path} = ${hi} (the top of the band) is accepted`)
    ok(rejects(set(path, 'x'), 'E_SHAPE', path), `${path} non-numeric → E_SHAPE at ${path}`)
    ok(rejects(set(path, NaN), 'E_SHAPE', path), `${path} NaN → E_SHAPE at ${path}`)
  }

  // ...and clamping is not a substitute for reporting: normalizeConfig would
  // have quietly fixed every one of those. That is the whole "two kinds of
  // defaulting" contract, and it is the thing v3 got wrong twice.
  ok(normalizeConfig(set('angleDeg', 999)).angleDeg === ANGLE_MAX &&
     rejects(set('angleDeg', 999), 'E_RANGE', 'angleDeg'),
    'an out-of-range angle is CLAMPED by normalize and REPORTED by validate')
  // Including the phase band that NARROWED for §9: 2 was legal on the ribbon.
  ok(rejects(set('pattern.phase', 2), 'E_RANGE', 'pattern.phase'),
    'phase 2 was legal on the ribbon and is out of band for a period-2 level field')

  // The integer knobs additionally reject a fractional value.
  for (const path of ['lattice.cols', 'lattice.rows', 'pattern.phase', 'connectors.minPerJoint']) {
    ok(rejects(set(path, 2.5), 'E_SHAPE', path), `${path} = 2.5 → E_SHAPE (it is a count)`)
  }
}

// -----------------------------------------------------------------------------
// 7. Enums, version, and the boolean.
// -----------------------------------------------------------------------------
console.log('7. enums, version, groundToFloor')
{
  ok(rejects({ ...good(), version: 3 }, 'E_SHAPE', 'version'), 'a v3 config is REJECTED, not migrated')
  ok(rejects({ ...good(), version: 2 }, 'E_SHAPE', 'version'), 'a v2 config too')
  ok(rejects({ ...good(), version: '4' }, 'E_SHAPE', 'version'), 'and the string "4"')
  ok(validateConfig(good()).valid, 'version 4 is accepted')
  ok(!validateConfig(42).valid && validateConfig(42).errors[0].code === 'E_SHAPE',
    'a non-object config is E_SHAPE')

  const withLattice = (panelType) => ({ ...good(), lattice: { ...good().lattice, panelType } })
  ok(rejects(withLattice('3x3'), 'E_SHAPE', 'lattice.panelType'), 'an unknown panelType is E_SHAPE')
  ok(rejects(withLattice('2x4'), 'E_SHAPE', 'lattice.panelType'),
    'and so is 2x4 — a plate needs a rectangular lattice, which is §9.8')
  for (const t of PANEL_TYPES) ok(accepts(withLattice(t), 'lattice.panelType'), `panelType "${t}" is accepted`)

  const withKind = (kind) => ({ ...good(), pattern: { ...good().pattern, kind } })
  ok(rejects(withKind('sawtooth'), 'E_SHAPE', 'pattern.kind'), 'an unknown pattern kind is E_SHAPE')
  for (const k of PATTERN_KINDS) ok(accepts(withKind(k), 'pattern.kind'), `kind "${k}" is accepted`)

  const withPlace = (over) => ({ ...good(), placement: { ...good().placement, ...over } })
  ok(rejects(withPlace({ wallAnchor: 'bolted' }), 'E_SHAPE', 'placement.wallAnchor'),
    'an unknown wallAnchor is E_SHAPE')
  for (const a of WALL_ANCHORS) {
    ok(accepts(withPlace({ wallAnchor: a }), 'placement.wallAnchor'), `wallAnchor "${a}" is accepted`)
  }

  const withConn = (over) => ({ ...good(), connectors: { ...good().connectors, ...over } })
  ok(rejects(withConn({ powerEdge: 'middle' }), 'E_SHAPE', 'connectors.powerEdge'),
    'an unknown powerEdge is E_SHAPE')
  for (const e of CONNECTOR_POWER_EDGES) {
    ok(accepts(withConn({ powerEdge: e }), 'connectors.powerEdge'), `powerEdge "${e}" is accepted`)
  }
  ok(rejects(withConn({ supplyMode: 'ignore' }), 'E_SHAPE', 'connectors.supplyMode'),
    'an unknown supplyMode is E_SHAPE')
  for (const m of CONNECTOR_SUPPLY_MODES) {
    ok(accepts(withConn({ supplyMode: m }), 'connectors.supplyMode'), `supplyMode "${m}" is accepted`)
  }

  ok(rejects(withPlace({ groundToFloor: 'yes' }), 'E_SHAPE', 'placement.groundToFloor'),
    'groundToFloor: "yes" is E_SHAPE, not silently true')
  ok(rejects(withPlace({ groundToFloor: 1 }), 'E_SHAPE', 'placement.groundToFloor'), 'and 1 is too')
  ok(accepts(withPlace({ groundToFloor: false }), 'placement.groundToFloor'), 'false is accepted')
  ok(accepts(withPlace({ groundToFloor: true }), 'placement.groundToFloor'), 'true is accepted')

  // Warnings do not make a config invalid.
  const single = validateConfig(withConn({ minPerJoint: 1 }))
  ok(single.valid && single.warnings.some((w) => w.code === 'W_SINGLE_CONNECTOR_JOINTS'),
    'minPerJoint 1 warns (a joint held by one part is a hinge) but is still valid')
  const nopower = validateConfig(withConn({ powerEdge: 'none' }))
  ok(nopower.valid && nopower.warnings.some((w) => w.code === 'W_POWER_SUPPLY_IGNORED'),
    'powerEdge "none" warns but is still valid')
}

// -----------------------------------------------------------------------------
// 8. THE STALE RIBBON CONFIG — refused, never quietly replaced.
// -----------------------------------------------------------------------------
console.log('8. a v4 config of the ribbon shape is refused')
{
  const ribbon = {
    version: 4,
    name: 'fold study 1',
    strip: { count: 1, units: 9, panelType: '2x2' },
    gap: 2,
    angleDeg: 30,
    pattern: { kind: 'trapezoid', phase: 0 },
    placement: { wallOffsetCm: 0, windowOffsetCm: 0, groundToFloor: true },
    overrides: [{ strip: 0, unit: 3, present: false, flipped: false, role: 'auto' }],
    connectors: { ...DEFAULT_CONNECTORS },
    meta: { notes: '' },
  }
  const v = validateConfig(ribbon)
  ok(!v.valid, 'a stale ribbon config does not validate')
  ok(v.errors.length === 1 && v.errors[0].code === 'E_LEGACY_SHAPE' && v.errors[0].path === 'strip',
    `and the single error is E_LEGACY_SHAPE at "strip" (got ${JSON.stringify(v.errors.map((e) => e.code))})`)
  ok(/lattice/.test(v.errors[0].message) && /cells/.test(v.errors[0].message),
    'whose message names what to rebuild it as')
  // Reported alone: every other check would be answering questions about a
  // default lattice the user never asked for.
  ok(v.warnings.length === 0, 'and nothing else is reported alongside it')

  // The version check CANNOT catch this — that is why the code exists.
  ok(ribbon.version === 4, 'the stale config is version 4, exactly like a current one')

  // Both halves of the guard: it takes `strip` present AND `lattice` absent.
  ok(validateConfig({ ...good(), strip: { count: 1 } }).valid,
    'a config carrying BOTH is accepted — the lattice wins and the stray key is ignored')
  ok(rejects({ ...ribbon, lattice: undefined }, 'E_LEGACY_SHAPE', 'strip'),
    'an explicitly-undefined lattice is still absent')
  ok(validateConfig(good()).valid, 'and a current config is untouched by the check')

  // normalizeConfig, by contrast, still hands back something usable — the split
  // is deliberate (normalize never reports, validate never fixes), and the store
  // is what runs validate before trusting anything.
  const n = normalizeConfig(ribbon)
  ok(n.lattice.cols === 3 && n.lattice.rows === 5 && n.strip === undefined,
    'normalizeConfig turns it into the default lattice and drops `strip` — which is exactly why ' +
    'validateConfig has to refuse it first')
  deepEq(n.overrides, { cells: [], edges: [] }, 'and the ribbon\'s flat override array vanishes')
}

// -----------------------------------------------------------------------------
// 9. Overrides — reported by validate, sanitized by normalize.
// -----------------------------------------------------------------------------
console.log('9. overrides')
{
  const withCells = (cells) => ({ ...good(), overrides: { cells, edges: [] } })
  const withEdges = (edges) => ({ ...good(), overrides: { cells: [], edges } })

  ok(rejects({ ...good(), overrides: 'nope' }, 'E_OVERRIDE_SHAPE', 'overrides'),
    'a non-object overrides is E_OVERRIDE_SHAPE')
  ok(rejects({ ...good(), overrides: { cells: 'x', edges: [] } }, 'E_OVERRIDE_SHAPE', 'overrides.cells'),
    'a non-array cells table is reported')
  ok(rejects({ ...good(), overrides: { cells: [], edges: 7 } }, 'E_OVERRIDE_SHAPE', 'overrides.edges'),
    'and a non-array edges table')

  // --- cells ---
  ok(rejects(withCells([5]), 'E_OVERRIDE_SHAPE', 'overrides.cells[0]'), 'a non-object cell entry')
  ok(rejects(withCells([{ j: 3 }]), 'E_OVERRIDE_SHAPE', 'overrides.cells[0].i'), 'a missing i')
  ok(rejects(withCells([{ i: 0 }]), 'E_OVERRIDE_SHAPE', 'overrides.cells[0].j'), 'a missing j')
  ok(rejects(withCells([{ i: 0, j: 1.5 }]), 'E_OVERRIDE_SHAPE', 'overrides.cells[0].j'), 'a fractional j')
  ok(rejects(withCells([{ i: 0, j: 1, present: 'no' }]), 'E_OVERRIDE_SHAPE', 'overrides.cells[0].present'),
    'a non-boolean present')
  ok(rejects(withCells([{ i: 0, j: 1, flipped: 1 }]), 'E_OVERRIDE_SHAPE', 'overrides.cells[0].flipped'),
    'a non-boolean flipped')

  // Bounds: BOTH indices are 0-BASED, unlike the ribbon's 1-based `unit`.
  ok(accepts(withCells([{ i: 0, j: 0, present: false }]), 'overrides.cells[0]'), 'cell (0,0) is in bounds')
  ok(accepts(withCells([{ i: 2, j: 4, present: false }]), 'overrides.cells[0]'),
    'and (2,4), the far corner of a 3 × 5')
  ok(rejects(withCells([{ i: 3, j: 0, present: false }]), 'E_OVERRIDE_BOUNDS', 'overrides.cells[0]'),
    'i = 3 with 3 columns is out of bounds')
  ok(rejects(withCells([{ i: 0, j: 5, present: false }]), 'E_OVERRIDE_BOUNDS', 'overrides.cells[0]'),
    'j = 5 with 5 rows is out of bounds')
  ok(rejects(withCells([{ i: -1, j: 0, present: false }]), 'E_OVERRIDE_BOUNDS', 'overrides.cells[0]'),
    'and i = −1 — the wall anchor is not addressable as a cell')

  ok(rejects(withCells([
    { i: 1, j: 1, present: false },
    { i: 1, j: 1, flipped: true },
  ]), 'E_OVERRIDE_CONFLICT', 'overrides.cells[1]'), 'two overrides on one cell conflict')

  const noop = validateConfig(withCells([{ i: 1, j: 1, present: true, flipped: false }]))
  ok(noop.valid && noop.warnings.some((w) => w.code === 'W_OVERRIDE_NO_OP'),
    'an entirely-default cell override warns that it will be dropped')

  // --- edges ---
  ok(rejects(withEdges([{ i: 0, j: 0, present: false }]), 'E_OVERRIDE_SHAPE', 'overrides.edges[0].axis'),
    'an edge with no axis')
  ok(rejects(withEdges([{ i: 0, j: 0, axis: 'y', present: false }]),
    'E_OVERRIDE_SHAPE', 'overrides.edges[0].axis'), 'and an unknown axis')
  for (const axis of EDGE_AXES) {
    ok(accepts(withEdges([{ i: 0, j: 0, axis, present: false }]), 'overrides.edges[0].axis'),
      `axis "${axis}" is accepted`)
  }
  ok(rejects(withEdges([{ i: 0, j: 0, axis: 'x', present: 3 }]),
    'E_OVERRIDE_SHAPE', 'overrides.edges[0].present'), 'a non-boolean present')

  // Bounds depend on the AXIS: a 3 × 5 lattice has 2 × 5 x-edges and 3 × 4 z ones.
  ok(accepts(withEdges([{ i: 1, j: 4, axis: 'x', present: false }]), 'overrides.edges[0]'),
    'x-edge (1,4) is the last one')
  ok(rejects(withEdges([{ i: 2, j: 0, axis: 'x', present: false }]),
    'E_OVERRIDE_BOUNDS', 'overrides.edges[0]'), 'x-edge (2,0) is past the last column')
  ok(accepts(withEdges([{ i: 2, j: 3, axis: 'z', present: false }]), 'overrides.edges[0]'),
    'z-edge (2,3) is the last one')
  ok(rejects(withEdges([{ i: 0, j: 4, axis: 'z', present: false }]),
    'E_OVERRIDE_BOUNDS', 'overrides.edges[0]'), 'z-edge (0,4) is past the last row')
  // ...and a 1-column lattice has NO x-edges at all, so any is out of bounds.
  ok(rejects({ ...good(), lattice: { ...good().lattice, cols: 1 },
    overrides: { cells: [], edges: [{ i: 0, j: 0, axis: 'x', present: false }] } },
  'E_OVERRIDE_BOUNDS', 'overrides.edges[0]'),
  'a 1-column lattice has no x-edges, so even (0,0,"x") is out of bounds')

  ok(rejects(withEdges([
    { i: 0, j: 0, axis: 'x', present: false },
    { i: 0, j: 0, axis: 'x', present: false },
  ]), 'E_OVERRIDE_CONFLICT', 'overrides.edges[1]'), 'two overrides on one edge conflict')
  ok(accepts(withEdges([
    { i: 0, j: 0, axis: 'x', present: false },
    { i: 0, j: 0, axis: 'z', present: false },
  ]), 'overrides.edges[1]'), 'but the two axes at one (i,j) are different edges')

  const edgeNoop = validateConfig(withEdges([{ i: 0, j: 0, axis: 'x', present: true }]))
  ok(edgeNoop.valid && edgeNoop.warnings.some((w) => w.code === 'W_OVERRIDE_NO_OP'),
    'an entirely-default edge override warns too')

  // --- normalizeConfig's half: drop, dedupe, and SORT ---
  const n = normalizeConfig({
    ...good(),
    overrides: {
      cells: [
        { i: 2, j: 4, flipped: true },
        { i: 0, j: 1, present: false },
        { i: 0, j: 1, flipped: true },     // duplicate, first wins
        { i: 1, j: 1 },                    // entirely default
        { i: 40, j: 0, present: false },   // out of grid
        null,
      ],
      edges: [
        { i: 1, j: 2, axis: 'z', present: false },
        { i: 0, j: 0, axis: 'x', present: false },
        { i: 0, j: 0, axis: 'x', present: false },  // duplicate
        { i: 0, j: 0, axis: 'z' },                  // entirely default
      ],
    },
  })
  deepEq(n.overrides.cells, [
    { i: 0, j: 1, present: false, flipped: false },
    { i: 2, j: 4, present: true, flipped: true },
  ], 'normalize drops no-ops / out-of-grid / duplicates and sorts cells by (i, j)')
  deepEq(n.overrides.edges, [
    { i: 0, j: 0, axis: 'x', present: false },
    { i: 1, j: 2, axis: 'z', present: false },
  ], 'and sorts edges by (axis, i, j)')

  // Order in, order out: two configs that describe the same design normalize to
  // the same bytes regardless of how the tables were typed.
  const a = normalizeConfig(withCells([{ i: 2, j: 0, flipped: true }, { i: 0, j: 3, present: false }]))
  const b = normalizeConfig(withCells([{ i: 0, j: 3, present: false }, { i: 2, j: 0, flipped: true }]))
  ok(JSON.stringify(a) === JSON.stringify(b), 'override table order carries no information')

  // Bounds are checked against the CLAMPED grid, not the raw one.
  const shrunk = normalizeConfig({ lattice: { cols: 2, rows: 2 },
    overrides: { cells: [{ i: 2, j: 0, present: false }],
      edges: [{ i: 1, j: 0, axis: 'x', present: false }] } })
  ok(shrunk.overrides.cells.length === 0 && shrunk.overrides.edges.length === 0,
    'an override outside the clamped grid is dropped')

  // There is no `role` override any more — a cell's level is the checkerboard's
  // to decide, and a forced role would ask for a geometry that cannot close.
  const withRole = normalizeConfig(withCells([{ i: 0, j: 0, present: false, role: 'high' }]))
  deepEq(withRole.overrides.cells, [{ i: 0, j: 0, present: false, flipped: false }],
    'a stray `role` on a cell override is dropped rather than honoured')
}

// -----------------------------------------------------------------------------
// L. THE LEGACY PLACEMENT KEY
//
// `placement.groundClearanceCm` was renamed to `yOffsetCm` when the ground
// spacers were dropped (V4_SPEC §9.12), and `version` was deliberately NOT
// bumped — the shape did not change, only a name and a band. That makes
// normalizeConfig the only thing standing between the user's saved slots and a
// silent reset to the default, so the read is asserted rather than assumed.
// -----------------------------------------------------------------------------
console.log('L. the legacy groundClearanceCm key')
{
  const old = normalizeConfig({ placement: { groundClearanceCm: 22 } })
  ok(old.placement.yOffsetCm === 22, `a config carrying only groundClearanceCm: 22 normalizes to yOffsetCm 22 (got ${old.placement.yOffsetCm})`)
  ok(!('groundClearanceCm' in old.placement), 'and the old key does not survive into the output')
  ok(DEFAULT_PLACEMENT.yOffsetCm !== 22, 'which is not the default, so the check is not vacuous')

  // Both present: the NEW key wins, so a config written by this version means
  // what it says rather than being overruled by a stale sibling.
  const both = normalizeConfig({ placement: { groundClearanceCm: 22, yOffsetCm: 7 } })
  ok(both.placement.yOffsetCm === 7, `both keys present prefers the new one (got ${both.placement.yOffsetCm})`)

  // Neither: the default, not a hole.
  ok(normalizeConfig({}).placement.yOffsetCm === DEFAULT_PLACEMENT.yOffsetCm,
    'neither key present falls back to the default')

  // The legacy value is still CLAMPED — it arrives from a file, so it is not
  // trusted any more than a fresh one.
  ok(normalizeConfig({ placement: { groundClearanceCm: 1e9 } }).placement.yOffsetCm === Y_OFFSET_MAX,
    'and a wild legacy value clamps like any other')
}

console.log(`\ntest-v4-schema: ${passed} checks passed, ${failed} failed`)
process.exit(failed ? 1 : 0)
