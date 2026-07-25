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
 *
 * It must also satisfy THE PLATE-PATTERN RULE (core/schema.js): at least
 * MIN_RECTS = 4 of the 60×121 plates, placed by ONE nameable rule whose name is
 * recorded in `meta.rectPattern` — so no preset may raise W_FEW_RECTS, and the
 * 'random' preset draws a NAMED template rather than scattering candidates.
 */

import assert from 'node:assert/strict'
import { MIN_RECTS, cellPitches, validateConfig } from '../src/core/schema.js'
import {
  PRESETS,
  RANDOM_TEMPLATES,
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
const warnCodes = (cfg) => validateConfig(cfg).warnings.map((w) => w.code)
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
// 2c. THE PLATE-PATTERN RULE — ≥ MIN_RECTS plates, by a rule that has a name
// =============================================================================
for (const [label, cfg] of named) {
  check(
    `${label}: places at least ${MIN_RECTS} plates`,
    cfg.rects.length >= MIN_RECTS,
    `${cfg.rects.length} rects: ${JSON.stringify(cfg.rects)}`,
  )
  check(
    `${label}: no W_FEW_RECTS`,
    !warnCodes(cfg).includes('W_FEW_RECTS'),
    JSON.stringify(warnCodes(cfg)),
  )
  check(
    `${label}: meta.rectPattern names the rule`,
    typeof cfg.meta.rectPattern === 'string' && cfg.meta.rectPattern.trim().length > 0,
    JSON.stringify(cfg.meta),
  )
  check(
    `${label}: meta.notes repeats the pattern name`,
    typeof cfg.meta.notes === 'string' && cfg.meta.notes.includes(cfg.meta.rectPattern),
    JSON.stringify(cfg.meta.notes),
  )
  // Every plate the preset placed must actually be legal in the surface it designed:
  // a vertical plate needs its joint unfolded, a horizontal one needs both columns
  // at the same pitch. (validateConfig covers this too; asserted here per plate so a
  // failure names the plate.)
  const illegal = cfg.rects.filter((rect) =>
    rect.orientation === 'vertical'
      ? cfg.columns[rect.col].foldsDeg[rect.row] !== 0
      : cellPitches(cfg, rect.col)[rect.row] !== cellPitches(cfg, rect.col + 1)[rect.row],
  )
  check(
    `${label}: every placed plate is geometrically legal`,
    illegal.length === 0,
    JSON.stringify(illegal),
  )
}
{
  // The whole random seed sweep, not just the sampled seeds above.
  let allOk = true
  let firstBad = ''
  const patterns = new Set()
  for (let seed = 0; seed < 60; seed++) {
    const cfg = presetRandom(seed)
    patterns.add(cfg.meta.rectPattern)
    const problems = []
    if (cfg.rects.length < MIN_RECTS) problems.push(`only ${cfg.rects.length} rects`)
    if (warnCodes(cfg).includes('W_FEW_RECTS')) problems.push('W_FEW_RECTS')
    if (!cfg.meta.rectPattern) problems.push('no meta.rectPattern')
    if (problems.length > 0 && allOk) {
      allOk = false
      firstBad = `seed ${seed}: ${problems.join(', ')}`
    }
  }
  check(`presetRandom places ≥ ${MIN_RECTS} plates for seeds 0..59`, allOk, firstBad)
  check(
    'presetRandom draws at least 2 different named templates over seeds 0..59',
    patterns.size >= 2,
    JSON.stringify([...patterns]),
  )
  check(
    'every drawn pattern name is one of RANDOM_TEMPLATES',
    [...patterns].every((p) => RANDOM_TEMPLATES.some((t) => t.id === p)),
    `${JSON.stringify([...patterns])} vs ${JSON.stringify(RANDOM_TEMPLATES.map((t) => t.id))}`,
  )
}
{
  check(
    'RANDOM_TEMPLATES: 4 templates, each with ≥ MIN_RECTS plates and a name',
    RANDOM_TEMPLATES.length === 4 &&
      RANDOM_TEMPLATES.every(
        (t) => typeof t.id === 'string' && t.id.length > 0 && t.rects.length >= MIN_RECTS,
      ),
    JSON.stringify(RANDOM_TEMPLATES.map((t) => [t.id, t.rects.length])),
  )
  check(
    'RANDOM_TEMPLATES ids are unique',
    new Set(RANDOM_TEMPLATES.map((t) => t.id)).size === RANDOM_TEMPLATES.length,
  )
  check(
    "RANDOM_TEMPLATES[0] is the can't-fail fallback 'mirrored-pairs'",
    RANDOM_TEMPLATES[0].id === 'mirrored-pairs',
    RANDOM_TEMPLATES[0].id,
  )
}
{
  // The four hand-authored presets each use a DIFFERENT pattern rule.
  const names = ['flat', 'calm', 'wave', 'crash'].map((id) => buildPreset(id).meta.rectPattern)
  check(
    'flat/calm/wave/crash each name a different plate pattern',
    new Set(names).size === 4,
    JSON.stringify(names),
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
    flat.columns.every((col) => col.foldsDeg.every((f) => f === 0)),
    JSON.stringify(flat.columns),
  )
  // 'mirrored quad': two stacked vertical plates in each of the mirrored columns 1
  // and 4 — the pattern reads as one shape used twice, about the grid centre.
  check(
    "presetFlat's plates are the mirrored quad (1,1) (3,1) (1,4) (3,4)",
    JSON.stringify(flat.rects) ===
      JSON.stringify([
        { row: 1, col: 1, orientation: 'vertical' },
        { row: 3, col: 1, orientation: 'vertical' },
        { row: 1, col: 4, orientation: 'vertical' },
        { row: 3, col: 4, orientation: 'vertical' },
      ]),
    JSON.stringify(flat.rects),
  )
  check(
    'presetFlat: the quad is mirror-symmetric about the grid centre (c ↔ 5−c)',
    flat.rects.every((r) =>
      flat.rects.some(
        (m) => m.row === r.row && m.col === flat.grid.cols - 1 - r.col && m.orientation === r.orientation,
      ),
    ),
    JSON.stringify(flat.rects),
  )
}
{
  // 'mirrored pairs': four HORIZONTAL plates bridging the outer column pairs (0,1)
  // and (4,5) at rows 1 and 3. They only drop in because those four columns share
  // one profile, so the two chains a plate spans coincide exactly.
  const calm = presetCalm()
  check(
    'presetCalm uses four horizontal plates (mirrored pairs)',
    calm.rects.length === 4 && calm.rects.every((r) => r.orientation === 'horizontal'),
    JSON.stringify(calm.rects),
  )
  check(
    'presetCalm: the pairs are mirrored — (·,0) bridging 0|1 and (·,4) bridging 4|5',
    JSON.stringify(calm.rects.map((r) => `${r.row},${r.col}`)) ===
      JSON.stringify(['1,0', '3,0', '1,4', '3,4']),
    JSON.stringify(calm.rects),
  )
  check(
    'presetCalm is left–right symmetric (columns c and 5−c share a profile)',
    calm.columns.every(
      (col, c) => JSON.stringify(col.foldsDeg) === JSON.stringify(calm.columns[5 - c].foldsDeg),
    ),
    JSON.stringify(calm.columns.map((col) => col.foldsDeg)),
  )
  check(
    'presetCalm: no W_CROSSCOL_POSITION — its bridged chains coincide',
    !warnCodes(calm).includes('W_CROSSCOL_POSITION'),
    JSON.stringify(warnCodes(calm)),
  )
  check(
    'presetCalm: every plate clears the landing row',
    calm.rects.every((r) => r.row < calm.grid.rows - 1),
    JSON.stringify(calm.rects),
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
  // 'landing plates': every column's LAST two rows fused into one 121cm plate. That
  // plate is why the crest above can be this steep — it reaches twice as far down as
  // a square would, so a chain standing 85° up at row 2 can still find the floor.
  check(
    'presetCrash places a landing plate on every column',
    crash.rects.length === 6 &&
      crash.rects.every(
        (r) => r.orientation === 'vertical' && r.row === crash.grid.rows - 2,
      ) &&
      new Set(crash.rects.map((r) => r.col)).size === 6,
    JSON.stringify(crash.rects),
  )
  check(
    "presetCrash: each landing plate's two rows share one pitch (the plate is rigid)",
    crash.columns.every((col) => col.foldsDeg[crash.grid.rows - 2] === 0),
    JSON.stringify(crash.columns.map((col) => col.foldsDeg)),
  )
  check(
    'presetCrash: every landing plate is tilted down onto the floor, none level',
    profiles(crash).every((p) => p[p.length - 1] < -5),
    JSON.stringify(profiles(crash).map((p) => p[p.length - 1])),
  )
}
{
  const wave = presetWave()
  // 'crest plates': ONE vertical plate across every column's crest plateau. The
  // plateau exists because the plate demands it — a rigid 121cm plate removes joint
  // r, so rows r and r+1 must sit at one pitch.
  check(
    'presetWave places one vertical plate per column (crest plates)',
    wave.rects.length === 6 &&
      wave.rects.every((r) => r.orientation === 'vertical') &&
      new Set(wave.rects.map((r) => r.col)).size === 6,
    JSON.stringify(wave.rects),
  )
  check(
    "presetWave: the plates sit on each column's crest plateau, stepping back at mid-window",
    JSON.stringify(wave.rects.map((r) => `${r.row},${r.col}`)) ===
      JSON.stringify(['1,0', '1,1', '1,2', '2,3', '2,4', '2,5']),
    JSON.stringify(wave.rects),
  )
  // Every plate must sit on an unfolded joint (else E_FOLD_ON_REMOVED_JOINT) — and
  // that joint must be the CREST, i.e. the plateau carries the profile's max pitch.
  check(
    'presetWave: every plate sits on an unfolded joint',
    wave.rects.every((r) => wave.columns[r.col].foldsDeg[r.row] === 0),
    JSON.stringify(wave.columns.map((col) => col.foldsDeg)),
  )
  check(
    "presetWave: each plate's two rows are the column's crest (both at its max pitch)",
    wave.rects.every((r) => {
      const p = cellPitches(wave, r.col)
      const crest = Math.max(...p)
      return p[r.row] === crest && p[r.row + 1] === crest
    }),
    JSON.stringify(profiles(wave)),
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
  // A random config's plates come from the drawn TEMPLATE, and the fold profiles are
  // authored around them — so the placed set must be exactly that template's rects,
  // and every one of them geometrically legal.
  for (const seed of [0, 1, 5, 42]) {
    const cfg = presetRandom(seed)
    const template = RANDOM_TEMPLATES.find((t) => t.id === cfg.meta.rectPattern)
    check(
      `random(${seed}): its plates are exactly the drawn template's ('${cfg.meta.rectPattern}')`,
      !!template && JSON.stringify(cfg.rects) === JSON.stringify(template.rects),
      JSON.stringify(cfg.rects),
    )
    const ok = cfg.rects.every((rect) =>
      rect.orientation === 'vertical'
        ? cfg.columns[rect.col].foldsDeg[rect.row] === 0
        : cellPitches(cfg, rect.col)[rect.row] === cellPitches(cfg, rect.col + 1)[rect.row],
    )
    check(`random(${seed}): kept plates are geometrically legal`, ok, JSON.stringify(cfg.rects))
  }
  // …across the whole sweep, not just four seeds.
  let allLegal = true
  let firstBad = ''
  for (let seed = 0; seed < 60; seed++) {
    const cfg = presetRandom(seed)
    const bad = cfg.rects.filter((rect) =>
      rect.orientation === 'vertical'
        ? cfg.columns[rect.col].foldsDeg[rect.row] !== 0
        : cellPitches(cfg, rect.col)[rect.row] !== cellPitches(cfg, rect.col + 1)[rect.row],
    )
    if (bad.length > 0) {
      allLegal = false
      firstBad = `seed ${seed} [${cfg.meta.rectPattern}]: ${JSON.stringify(bad)}`
      break
    }
  }
  check('presetRandom: every plate is legal for seeds 0..59', allLegal, firstBad)
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
