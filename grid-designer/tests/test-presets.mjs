/**
 * tests/test-presets.mjs — headless checks for core/presets.js (v2).
 *
 * Plain node script, no test framework. Exits non-zero on any failure.
 *   node tests/test-presets.mjs
 *
 * Beyond determinism and validity, every preset must be BUILDABLE ON THE FLOOR:
 * its solved layout may not raise W_BELOW_FLOOR, and — the grounded-end rule from
 * core/schema.js — may not raise E_END_FLOATING either: EVERY column's last panel
 * has to come back down and touch the ground.
 */

import assert from 'node:assert/strict'
import { cellPitches, validateConfig } from '../src/core/schema.js'
import {
  PRESETS,
  buildPreset,
  mulberry32,
  presetFlat,
  presetCalm,
  presetWave,
  presetCrash,
  presetRandom,
} from '../src/core/presets.js'
import { solveLayout } from '../src/core/placement.js'

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

const describe = (result) =>
  `errors=[${result.errors.map((e) => `${e.code}@${e.path}: ${e.message}`).join(' | ')}]`
const maxFold = (cfg) => Math.max(...cfg.columns.flatMap((col) => col.foldsDeg.map(Math.abs)))
const profiles = (cfg) => cfg.columns.map((_, c) => cellPitches(cfg, c))

// =============================================================================
// 1. Every preset validates
// =============================================================================
const named = [
  ['flat', presetFlat()],
  ['calm', presetCalm()],
  ['wave', presetWave()],
  ['crash', presetCrash()],
  ['random(1)', presetRandom(1)],
  ['random(2)', presetRandom(2)],
  ['random(42)', presetRandom(42)],
  ['random(0)', presetRandom(0)],
  ['random(123456789)', presetRandom(123456789)],
]

for (const [label, cfg] of named) {
  const result = validateConfig(cfg)
  check(`${label} validates`, result.ok, describe(result))
}

// A sweep of seeds — all must be valid.
{
  let allOk = true
  let firstBad = ''
  for (let seed = 0; seed < 60; seed++) {
    const result = validateConfig(presetRandom(seed))
    if (!result.ok) {
      allOk = false
      firstBad = `seed ${seed}: ${describe(result)}`
      break
    }
  }
  check('presetRandom validates for seeds 0..59', allOk, firstBad)
}

// =============================================================================
// 2. Ground support — no preset may leave a panel under the floor
// =============================================================================
for (const [label, cfg] of named) {
  const layout = solveLayout(cfg)
  const below = layout.warnings.filter((w) => w.code === 'W_BELOW_FLOOR')
  check(
    `${label}: no W_BELOW_FLOOR`,
    below.length === 0,
    below.map((w) => w.message).join(' | '),
  )
}
{
  let allSupported = true
  let firstBad = ''
  for (let seed = 0; seed < 60; seed++) {
    const layout = solveLayout(presetRandom(seed))
    const below = layout.warnings.find((w) => w.code === 'W_BELOW_FLOOR')
    if (below) {
      allSupported = false
      firstBad = `seed ${seed}: ${below.message}`
      break
    }
  }
  check('presetRandom stays above the floor for seeds 0..59', allSupported, firstBad)
}

// =============================================================================
// 2b. The grounded-end rule — every column's last panel must touch the floor
// =============================================================================
for (const [label, cfg] of named) {
  const layout = solveLayout(cfg)
  check(
    `${label}: no E_END_FLOATING — every column lands`,
    layout.violations.length === 0,
    layout.violations.map((v) => `col ${v.col} @ ${v.clearanceCm.toFixed(2)}cm`).join(' | '),
  )
  check(
    `${label}: every chain reports grounded with a clearance on the floor`,
    layout.columnChains.every(
      (ch) => ch.grounded === true && ch.endClearanceCm >= -0.25 && ch.endClearanceCm <= 0.5,
    ),
    JSON.stringify(layout.columnChains.map((ch) => [ch.grounded, ch.endClearanceCm])),
  )
}
{
  let allGrounded = true
  let firstBad = ''
  for (let seed = 0; seed <= 30; seed++) {
    const layout = solveLayout(presetRandom(seed))
    if (layout.violations.length > 0) {
      allGrounded = false
      firstBad = `seed ${seed}: ${layout.violations.map((v) => `col ${v.col} @ ${v.clearanceCm.toFixed(1)}cm`).join(', ')}`
      break
    }
  }
  check('presetRandom lands every column for seeds 0..30', allGrounded, firstBad)
}
// The landing is the LAST row's business: every preset's last pitch is ≤ 0, i.e.
// the strip is coming down (or level) as it reaches the back of the store.
for (const [label, cfg] of named) {
  const last = profiles(cfg).map((p) => p[p.length - 1])
  check(
    `${label}: every column's final pitch is level or descending`,
    last.every((psi) => psi <= 1e-9),
    JSON.stringify(last),
  )
}

// =============================================================================
// 3. Determinism
// =============================================================================
checkNoThrow('presetRandom(42) is deterministic (deep-equal)', () => {
  assert.deepStrictEqual(presetRandom(42), presetRandom(42))
})
checkNoThrow('presetRandom(42) is deterministic (JSON-identical)', () => {
  assert.strictEqual(JSON.stringify(presetRandom(42)), JSON.stringify(presetRandom(42)))
})
checkNoThrow('presetRandom(1) is deterministic', () => {
  assert.deepStrictEqual(presetRandom(1), presetRandom(1))
})

{
  const a = presetRandom(1)
  const b = presetRandom(2)
  check(
    'presetRandom(1) !== presetRandom(2)',
    JSON.stringify(a) !== JSON.stringify(b),
    'seeds 1 and 2 produced identical configs',
  )
}

// Distinct seeds should mostly produce distinct configs.
{
  const seen = new Set()
  for (let seed = 0; seed < 25; seed++) seen.add(JSON.stringify(presetRandom(seed)))
  check('25 seeds produce ≥ 20 distinct configs', seen.size >= 20, `got ${seen.size}`)
}

// The named presets must be distinct from each other.
{
  const shaped = ['flat', 'calm', 'wave', 'crash'].map((id) =>
    JSON.stringify(buildPreset(id).columns),
  )
  check('flat/calm/wave/crash all differ', new Set(shaped).size === 4)
}

// The PRNG itself.
{
  const a = mulberry32(7)
  const b = mulberry32(7)
  const seqA = [a(), a(), a(), a()]
  const seqB = [b(), b(), b(), b()]
  check('mulberry32 is deterministic', JSON.stringify(seqA) === JSON.stringify(seqB))
  check(
    'mulberry32 stays in [0,1)',
    seqA.every((v) => v >= 0 && v < 1),
    JSON.stringify(seqA),
  )
}

// =============================================================================
// 4. Preset content / metadata
// =============================================================================
{
  const ids = PRESETS.map((p) => p.id)
  check(
    'PRESETS lists flat/calm/wave/crash/random',
    ['flat', 'calm', 'wave', 'crash', 'random'].every((id) => ids.includes(id)),
    JSON.stringify(ids),
  )
  check(
    'PRESETS entries have label + description',
    PRESETS.every((p) => typeof p.label === 'string' && p.label && typeof p.description === 'string' && p.description),
  )
  check('only the random preset is seeded', PRESETS.filter((p) => p.seeded).map((p) => p.id).join() === 'random')
}

for (const id of ['flat', 'calm', 'wave', 'crash']) {
  const cfg = buildPreset(id)
  check(`buildPreset('${id}') sets meta.preset`, cfg.meta.preset === id, JSON.stringify(cfg.meta))
  check(`buildPreset('${id}') validates`, validateConfig(cfg).ok, describe(validateConfig(cfg)))
}
{
  const cfg = buildPreset('random', 99)
  check('buildPreset random sets meta.preset/seed', cfg.meta.preset === 'random' && cfg.meta.seed === 99, JSON.stringify(cfg.meta))
  check('buildPreset random matches presetRandom', JSON.stringify(cfg) === JSON.stringify(presetRandom(99)))
}
checkNoThrow('buildPreset throws on unknown id', () => {
  assert.throws(() => buildPreset('nope'))
})

// Shape: 6 columns × 4 joints, version 2.
for (const [label, cfg] of named) {
  check(
    `${label}: version 2, 6×5, 6 fold sequences of 4`,
    cfg.version === 2 &&
      cfg.grid.cols === 6 &&
      cfg.grid.rows === 5 &&
      cfg.columns.length === 6 &&
      cfg.columns.every((col) => Array.isArray(col.foldsDeg) && col.foldsDeg.length === 4),
    JSON.stringify({ version: cfg.version, grid: cfg.grid, columns: cfg.columns.length }),
  )
  check(
    `${label}: every cumulative pitch profile starts at 0 (row 0 is the shore)`,
    profiles(cfg).every((p) => p[0] === 0),
    JSON.stringify(profiles(cfg)),
  )
  check(
    `${label}: no v1 leftovers (rows / rowFoldsDeg / rowAnchor)`,
    cfg.rows === undefined && cfg.rowFoldsDeg === undefined && cfg.rowAnchor === undefined,
  )
}

// Flat really is flat; the shaped presets really are not.
{
  const flat = presetFlat()
  check(
    'presetFlat has no angles at all',
    flat.columns.every((col) => col.foldsDeg.every((f) => f === 0)) && flat.rects.length === 0,
  )
}
{
  const calm = presetCalm()
  const crash = presetCrash()
  check('presetCalm is gentler than presetCrash', maxFold(calm) < maxFold(crash), `${maxFold(calm)} vs ${maxFold(crash)}`)
  check('presetCalm folds stay small', maxFold(calm) <= 20, String(maxFold(calm)))
  check(
    'presetCrash reaches a near-vertical crest (some cumulative pitch ≥ 85°)',
    profiles(crash).some((p) => p.some((psi) => psi >= 85)),
    JSON.stringify(profiles(crash)),
  )
}
{
  const wave = presetWave()
  check('presetWave includes both plate types', wave.rects.length === 2 &&
    wave.rects.some((r) => r.orientation === 'vertical') &&
    wave.rects.some((r) => r.orientation === 'horizontal'), JSON.stringify(wave.rects))

  // The vertical plate must sit on an unfolded joint (else E_FOLD_ON_REMOVED_JOINT).
  const v = wave.rects.find((r) => r.orientation === 'vertical')
  check(
    'presetWave: the vertical plate sits on an unfolded joint',
    wave.columns[v.col].foldsDeg[v.row] === 0,
    JSON.stringify(wave.columns[v.col].foldsDeg),
  )
  // The horizontal plate must span two columns at the same pitch.
  const h = wave.rects.find((r) => r.orientation === 'horizontal')
  check(
    'presetWave: the horizontal plate spans two columns at matching pitch',
    cellPitches(wave, h.col)[h.row] === cellPitches(wave, h.col + 1)[h.row],
    JSON.stringify([cellPitches(wave, h.col), cellPitches(wave, h.col + 1)]),
  )

  // "A wave moving across the grid": neighbouring columns are phase-shifted, so
  // no two adjacent fold sequences are equal and the crest row drifts back.
  const seqs = wave.columns.map((col) => JSON.stringify(col.foldsDeg))
  check(
    'presetWave: every column has a distinct fold sequence',
    new Set(seqs).size === 6,
    JSON.stringify(seqs),
  )
  const crestRow = profiles(wave).map((p) => p.indexOf(Math.max(...p)))
  check(
    'presetWave: the crest drifts monotonically back across the columns',
    crestRow.every((r, c) => c === 0 || r >= crestRow[c - 1]) && crestRow[5] > crestRow[0],
    JSON.stringify(crestRow),
  )
}
{
  // Random plates are only kept when they are geometrically legal.
  for (const seed of [0, 1, 5, 42]) {
    const cfg = presetRandom(seed)
    const ok = cfg.rects.every((rect) =>
      rect.orientation === 'vertical'
        ? cfg.columns[rect.col].foldsDeg[rect.row] === 0
        : cellPitches(cfg, rect.col)[rect.row] === cellPitches(cfg, rect.col + 1)[rect.row],
    )
    check(`random(${seed}): kept plates are geometrically legal`, ok, JSON.stringify(cfg.rects))
  }
}

// =============================================================================
// Summary
// =============================================================================
console.log('')
console.log(`test-presets: ${passed} checks passed, ${failures.length} failed`)
if (failures.length > 0) {
  console.error('')
  console.error('Failures:')
  for (const f of failures) console.error(`  - ${f}`)
  process.exit(1)
}
