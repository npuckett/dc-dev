/**
 * tests/test-presets.mjs — headless checks for core/presets.js.
 *
 * Plain node script, no test framework. Exits non-zero on any failure.
 *   node tests/test-presets.mjs
 */

import assert from 'node:assert/strict'
import { validateConfig } from '../src/core/schema.js'
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
// 2. Determinism
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
// 3. Preset content / metadata
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

// Shore row flat in every preset.
for (const [label, cfg] of named) {
  check(
    `${label}: shore row 0 is flat`,
    cfg.rows[0].zigzagDeg === 0 && Object.keys(cfg.rows[0].jointOverridesDeg).length === 0,
    JSON.stringify(cfg.rows[0]),
  )
}

// Shape: 6×5 with matching array lengths.
for (const [label, cfg] of named) {
  check(
    `${label}: 6×5 with matching array lengths`,
    cfg.grid.cols === 6 && cfg.grid.rows === 5 && cfg.rows.length === 5 && cfg.rowFoldsDeg.length === 4,
    JSON.stringify({ grid: cfg.grid, rows: cfg.rows.length, folds: cfg.rowFoldsDeg.length }),
  )
}

// Flat really is flat; the shaped presets really are not.
{
  const flat = presetFlat()
  check(
    'presetFlat has no angles at all',
    flat.rows.every((r) => r.zigzagDeg === 0) && flat.rowFoldsDeg.every((f) => f === 0) && flat.rects.length === 0,
  )
}
{
  const calm = presetCalm()
  const crash = presetCrash()
  const maxZig = (cfg) => Math.max(...cfg.rows.map((r) => Math.abs(r.zigzagDeg)))
  check('presetCalm is gentler than presetCrash', maxZig(calm) < maxZig(crash), `${maxZig(calm)} vs ${maxZig(crash)}`)
  check('presetCalm angles stay low', maxZig(calm) <= 20, String(maxZig(calm)))
}
{
  const wave = presetWave()
  check('presetWave includes rects', wave.rects.length >= 2, JSON.stringify(wave.rects))
  check(
    'presetWave has a rising-then-falling pitch profile',
    wave.rowFoldsDeg.some((f) => f > 0) && wave.rowFoldsDeg.some((f) => f < 0),
    JSON.stringify(wave.rowFoldsDeg),
  )
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
