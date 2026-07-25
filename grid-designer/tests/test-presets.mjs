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
import {
  MIN_RECTS,
  WALL_COLUMN,
  cellPitches,
  columnEndSupport,
  validateConfig,
} from '../src/core/schema.js'
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
  presetSwell,
  presetSurge,
  presetWallcrash,
} from '../src/core/presets.js'
import { panelSolidMinY, solveLayout } from '../src/core/placement.js'

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
/** THE CALM FAMILY: 5 rows, flat fronts, every column on the floor. */
const calmNamed = [
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
/** THE STORM FAMILY: deeper grids, a pitched front in EVERY column, the wall. */
const stormNamed = [
  ['swell', presetSwell()],
  ['surge', presetSurge()],
  ['wallcrash', presetWallcrash()],
]
const named = [...calmNamed, ...stormNamed]

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
    `${label}: no E_END_FLOATING — every floor column lands`,
    layout.violations.length === 0,
    layout.violations.map((v) => `col ${v.col} @ ${v.clearanceCm.toFixed(2)}cm`).join(' | '),
  )
  check(
    `${label}: every FLOOR chain reports grounded with a clearance on the floor`,
    layout.columnChains
      .filter((ch) => ch.endSupport !== 'wall')
      .every((ch) => ch.grounded === true && ch.endClearanceCm >= -0.25 && ch.endClearanceCm <= 0.5),
    JSON.stringify(layout.columnChains.map((ch) => [ch.endSupport, ch.grounded, ch.endClearanceCm])),
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
// The landing is the LAST row's business: every FLOOR column's last pitch is ≤ 0,
// i.e. the strip is coming down (or level) as it reaches the back of the store. A
// WALL column is the exception on purpose — 'wallcrash' column 5 is still climbing
// when it meets the wall, which is the whole point of it.
for (const [label, cfg] of named) {
  const last = profiles(cfg).map((p) => p[p.length - 1])
  check(
    `${label}: every FLOOR column's final pitch is level or descending`,
    last.every((psi, c) => columnEndSupport(cfg, c) === 'wall' || psi <= 1e-9),
    JSON.stringify(last.map((psi, c) => [columnEndSupport(cfg, c), psi])),
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

// Shape: 6 columns × (grid.rows − 1) joints, version 2, rows inside 5..8.
for (const [label, cfg] of named) {
  check(
    `${label}: version 2, 6 columns × ${cfg.grid.rows} rows, ${cfg.grid.rows - 1} folds each`,
    cfg.version === 2 &&
      cfg.grid.cols === 6 &&
      cfg.grid.rows >= 5 &&
      cfg.grid.rows <= 8 &&
      cfg.columns.length === 6 &&
      cfg.columns.every(
        (col) => Array.isArray(col.foldsDeg) && col.foldsDeg.length === cfg.grid.rows - 1,
      ),
    JSON.stringify({ version: cfg.version, grid: cfg.grid, columns: cfg.columns.length }),
  )
  check(
    `${label}: every profile's first entry IS the column's startPitchDeg`,
    profiles(cfg).every((p, c) => p[0] === cfg.columns[c].startPitchDeg),
    JSON.stringify(profiles(cfg).map((p) => p[0])),
  )
  check(
    `${label}: every endSupport is 'floor', or 'wall' on the WALL-ADJACENT column 0 only`,
    cfg.columns.every(
      (col, c) => col.endSupport === 'floor' || (col.endSupport === 'wall' && c === WALL_COLUMN),
    ),
    JSON.stringify(cfg.columns.map((col) => col.endSupport)),
  )
  check(
    `${label}: no v1 leftovers (rows / rowFoldsDeg / rowAnchor)`,
    cfg.rows === undefined && cfg.rowFoldsDeg === undefined && cfg.rowAnchor === undefined,
  )
}
// The CALM family's defining rule: 5 rows and a flat front in every column.
for (const [label, cfg] of calmNamed) {
  check(
    `${label} (calm family): 5 rows, every front FLAT on the shore (startPitchDeg 0)`,
    cfg.grid.rows === 5 && cfg.columns.every((col) => col.startPitchDeg === 0),
    JSON.stringify(cfg.columns.map((col) => col.startPitchDeg)),
  )
  check(
    `${label} (calm family): every column stands on the floor`,
    cfg.columns.every((col) => col.endSupport === 'floor'),
    JSON.stringify(cfg.columns.map((col) => col.endSupport)),
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
// 5. THE STORM FAMILY — swell / surge / wallcrash
//
// Their signature rule, asserted first: NO FLAT FRONT ANYWHERE. Every column of
// every storm preset has startPitchDeg ≠ 0 — that is what separates the family from
// the calm one, and it is the thing a future edit is most likely to break.
// =============================================================================
{
  check(
    'PRESETS lists the storm family as swell / surge / wallcrash',
    JSON.stringify(PRESETS.filter((p) => p.family === 'storm').map((p) => p.id)) ===
      JSON.stringify(['swell', 'surge', 'wallcrash']),
    JSON.stringify(PRESETS.map((p) => [p.id, p.family])),
  )
  check(
    'PRESETS: every entry declares a family, and the calm ones come first',
    PRESETS.every((p) => p.family === 'calm' || p.family === 'storm') &&
      PRESETS.findIndex((p) => p.family === 'storm') ===
        PRESETS.filter((p) => p.family === 'calm').length,
    JSON.stringify(PRESETS.map((p) => p.family)),
  )
  check('no storm preset is seeded', PRESETS.filter((p) => p.family === 'storm').every((p) => !p.seeded))
}

for (const [label, cfg] of stormNamed) {
  // --- the signature rule ---------------------------------------------------
  check(
    `${label}: EVERY column has a pitched front (startPitchDeg ≠ 0)`,
    cfg.columns.every((col) => col.startPitchDeg !== 0),
    JSON.stringify(cfg.columns.map((col) => col.startPitchDeg)),
  )
  check(
    `${label}: every front pitch is inside ±120 and a whole degree (readable JSON)`,
    cfg.columns.every((col) => Number.isInteger(col.startPitchDeg) && Math.abs(col.startPitchDeg) <= 120),
    JSON.stringify(cfg.columns.map((col) => col.startPitchDeg)),
  )
  // The pitched front costs nothing in support — the panel still rests on the floor.
  {
    const layout = solveLayout(cfg)
    check(
      `${label}: every pitched FRONT panel still rests on the floor`,
      cfg.columns.every((_, c) => {
        const front = layout.panels.find((p) => p.col === c && p.row === 0)
        return front && Math.abs(panelSolidMinY(front, cfg)) < 1e-7
      }),
      JSON.stringify(
        cfg.columns.map((_, c) => {
          const front = layout.panels.find((p) => p.col === c && p.row === 0)
          return front ? Number(panelSolidMinY(front, cfg).toFixed(9)) : null
        }),
      ),
    )
  }
  // --- plates crossing ROWS, at least four, by a named rule ------------------
  check(
    `${label}: at least ${MIN_RECTS} plates and every one of them VERTICAL (crossing rows)`,
    cfg.rects.length >= MIN_RECTS && cfg.rects.every((r) => r.orientation === 'vertical'),
    JSON.stringify(cfg.rects),
  )
  check(
    `${label}: deterministic (JSON-identical across two builds)`,
    JSON.stringify(buildPreset(label.split('(')[0])) === JSON.stringify(cfg),
  )
}

// --- swell: 7 rows, spine plates, everything ramping to a peak AT THE WALL -------
//
// The design intent, encoded so it cannot silently regress:
//   1. NO ROW NEAR THE WINDOW LIES FLAT. Every front is at least
//      SWELL_MIN_FRONT_DEG steep, and row 1 is steeper still, so the two rows a
//      person at the window actually sees are visibly climbing.
//   2. EVERYTHING RAMPS TOWARD THE WALL. The wall is beside column 0
//      (schema.js WALL_COLUMN), so both the front pitch and the crest height fall
//      monotonically with the column index and column 0 is the strict maximum of
//      each.
//   3. ALL SIX COLUMNS STILL LAND ON THE FLOOR — that is what separates swell from
//      'wallcrash'; nothing here may take endSupport 'wall'.
{
  const swell = presetSwell()
  const layout = solveLayout(swell)
  /** Steepest a swell front may be allowed to get — anything flatter reads as lying flat. */
  const SWELL_MIN_FRONT_DEG = 28
  const fronts = swell.columns.map((col) => col.startPitchDeg)
  check('swell: 7 rows — the descent capacity a 150cm crest needs', swell.grid.rows === 7, String(swell.grid.rows))
  check(
    `swell: EVERY front is steeply pitched — at least ${SWELL_MIN_FRONT_DEG}°, never flat-looking`,
    fronts.every((psi) => psi >= SWELL_MIN_FRONT_DEG),
    JSON.stringify(fronts),
  )
  check(
    'swell: the front pitch RAMPS monotonically toward the wall — steepest at column 0',
    fronts.every((psi, c) => c === 0 || psi < fronts[c - 1]) &&
      fronts[WALL_COLUMN] === Math.max(...fronts),
    JSON.stringify(fronts),
  )
  check(
    'swell: ROW 1 keeps climbing — steeper than the front in every column, so the first two rows rise',
    profiles(swell).every((p) => p[1] > p[0] && p[1] >= SWELL_MIN_FRONT_DEG),
    JSON.stringify(profiles(swell).map((p) => [p[0], p[1]])),
  )
  check(
    "swell: 'spine plates' — one vertical plate mid-strip (rows 2–3) in every column",
    swell.meta.rectPattern === 'spine plates' &&
      swell.rects.length === 6 &&
      swell.rects.every((r) => r.row === 2 && r.orientation === 'vertical') &&
      new Set(swell.rects.map((r) => r.col)).size === 6,
    JSON.stringify([swell.meta.rectPattern, swell.rects]),
  )
  check(
    "swell: each spine plate's two rows share one pitch (the plate is rigid)",
    swell.columns.every((col) => col.foldsDeg[2] === 0),
    JSON.stringify(swell.columns.map((col) => col.foldsDeg)),
  )
  check(
    'swell: EVERY column stands on the floor — no wall support anywhere',
    swell.columns.every((col) => col.endSupport === 'floor') &&
      layout.columnChains.every((ch) => ch.grounded === true) &&
      layout.violations.length === 0,
    JSON.stringify(layout.columnChains.map((ch) => [ch.endSupport, ch.grounded])),
  )
  // THE RAMP: the crest heights are the design input, and they climb monotonically
  // toward the wall — water piling up against it. Measured from the SOLVED chains,
  // never from a hard-coded table, so a change in the walker shows up here.
  const peaks = layout.columnChains.map((ch) => Math.max(...ch.points.map((p) => p[0])))
  check(
    'swell: the crest RAMPS monotonically up toward column 0, which is the strict maximum',
    peaks.every((y, c) => c === 0 || y < peaks[c - 1]) &&
      peaks[WALL_COLUMN] === Math.max(...peaks) &&
      peaks.filter((y) => y === peaks[WALL_COLUMN]).length === 1,
    JSON.stringify(peaks.map((y) => Number(y.toFixed(1)))),
  )
  check(
    'swell: the ramp is DRAMATIC — the wall column is at least 60cm taller than column 5',
    peaks[WALL_COLUMN] - peaks[5] >= 60,
    `col0 ${peaks[WALL_COLUMN].toFixed(1)} … col5 ${peaks[5].toFixed(1)}`,
  )
  check(
    'swell: it peaks near 150cm at the wall and stays a ripple (< 90cm) at the far jamb',
    peaks[WALL_COLUMN] > 145 && peaks[5] < 90,
    JSON.stringify(peaks.map((y) => Number(y.toFixed(1)))),
  )
  check(
    'swell: every step of the ramp is a real one — no two neighbours within 8cm',
    peaks.every((y, c) => c === 0 || peaks[c - 1] - y > 8),
    JSON.stringify(peaks.map((y) => Number(y.toFixed(1)))),
  )
  check(
    'swell: the crest is a real rise — over 25× the flat lit-face height',
    Math.max(...peaks) > 25 * 3.7,
    String(Math.max(...peaks)),
  )
  check(
    'swell: no layout warnings at all',
    layout.warnings.length === 0,
    JSON.stringify(layout.warnings.map((w) => w.code)),
  )
}

// --- surge: 7 rows, two plates per column, peaks well past 150cm ----------------
{
  const surge = presetSurge()
  const layout = solveLayout(surge)
  check('surge: 7 rows', surge.grid.rows === 7, String(surge.grid.rows))
  check(
    'surge: STEEP fronts, 25° … 45°, rising with column index',
    JSON.stringify(surge.columns.map((col) => col.startPitchDeg)) ===
      JSON.stringify([25, 29, 33, 37, 41, 45]),
    JSON.stringify(surge.columns.map((col) => col.startPitchDeg)),
  )
  check(
    "surge: 'double plates' — TWO vertical plates in every column (12 in all)",
    surge.meta.rectPattern === 'double plates' && surge.rects.length === 12,
    JSON.stringify([surge.meta.rectPattern, surge.rects.length]),
  )
  check(
    'surge: a crest plate at rows 2–3 and a landing plate at rows 5–6 in each column',
    [0, 1, 2, 3, 4, 5].every(
      (c) =>
        surge.rects.some((r) => r.col === c && r.row === 2 && r.orientation === 'vertical') &&
        surge.rects.some((r) => r.col === c && r.row === 5 && r.orientation === 'vertical'),
    ),
    JSON.stringify(surge.rects.map((r) => `${r.row},${r.col}`)),
  )
  check(
    'surge: two plates per column is more than half the columns (the rule asked for ≥ 3)',
    [0, 1, 2, 3, 4, 5].filter((c) => surge.rects.filter((r) => r.col === c).length >= 2).length === 6,
  )
  check(
    "surge: both plates' joints are unfolded in every column",
    surge.columns.every((col) => col.foldsDeg[2] === 0 && col.foldsDeg[5] === 0),
    JSON.stringify(surge.columns.map((col) => col.foldsDeg)),
  )
  const peaks = layout.columnChains.map((ch) => Math.max(...ch.points.map((p) => p[0])))
  check(
    'surge: every column peaks WELL above 150cm',
    peaks.every((y) => y > 150),
    JSON.stringify(peaks.map((y) => Number(y.toFixed(1)))),
  )
  check(
    'surge: and is the tallest floor-grounded preset — taller than swell everywhere',
    Math.min(...peaks) >
      Math.max(
        ...solveLayout(presetSwell()).columnChains.map((ch) => Math.max(...ch.points.map((p) => p[0]))),
      ),
    `surge min ${Math.min(...peaks).toFixed(1)}`,
  )
  check(
    'surge: ALL SIX columns still land on the floor — nothing leans on the wall',
    surge.columns.every((col) => col.endSupport === 'floor') && layout.violations.length === 0,
    JSON.stringify(layout.columnChains.map((ch) => [ch.endSupport, ch.endClearanceCm])),
  )
  check(
    'surge: no layout warnings at all',
    layout.warnings.length === 0,
    JSON.stringify(layout.warnings.map((w) => w.code)),
  )
}

// --- wallcrash: 6 rows, a ramp toward the −X wall, column 0 splashing up it -----
{
  const wc = presetWallcrash()
  const layout = solveLayout(wc)
  check('wallcrash: 6 rows', wc.grid.rows === 6, String(wc.grid.rows))
  check(
    'wallcrash: front pitch RAMPS toward the wall — steepest at column 0, 43° … 8°',
    JSON.stringify(wc.columns.map((col) => col.startPitchDeg)) ===
      JSON.stringify([43, 36, 29, 22, 15, 8]),
    JSON.stringify(wc.columns.map((col) => col.startPitchDeg)),
  )
  check(
    'wallcrash: COLUMN 0 is WALL-SUPPORTED and nothing else is',
    wc.columns[WALL_COLUMN].endSupport === 'wall' &&
      wc.columns.slice(1).every((col) => col.endSupport === 'floor'),
    JSON.stringify(wc.columns.map((col) => col.endSupport)),
  )
  check(
    'wallcrash: columns 1–5 all LAND on the floor',
    layout.columnChains.slice(1).every((ch) => ch.grounded === true) &&
      layout.violations.length === 0,
    JSON.stringify(layout.columnChains.map((ch) => [ch.grounded, ch.endClearanceCm])),
  )
  // THE SPLASH: the wall column's end panel is high in the air, and legally so.
  check(
    "wallcrash: column 0's end is ELEVATED — well over 30cm off the floor",
    layout.columnChains[WALL_COLUMN].endClearanceCm > 30,
    `${layout.columnChains[WALL_COLUMN].endClearanceCm.toFixed(1)}cm`,
  )
  check(
    'wallcrash: that elevated end raises NO violation (the wall holds it)',
    layout.columnChains[WALL_COLUMN].grounded === false &&
      layout.columnChains[WALL_COLUMN].endSupport === 'wall' &&
      !layout.violations.some((v) => v.col === WALL_COLUMN),
    JSON.stringify(layout.violations),
  )
  // The ramp: amplitude climbs monotonically TOWARD the wall, i.e. DOWN the column
  // index — column 0 is against the wall, so it is the violent end.
  const peaks = layout.columnChains.map((ch) => Math.max(...ch.points.map((p) => p[0])))
  check(
    'wallcrash: amplitude RAMPS monotonically from column 5 up to column 0 (the wall)',
    peaks.every((y, c) => c === 0 || y < peaks[c - 1]),
    JSON.stringify(peaks.map((y) => Number(y.toFixed(1)))),
  )
  check(
    'wallcrash: column 5 is gentle (< 80cm) and column 0 at the wall is violent (> 150cm)',
    peaks[5] < 80 && peaks[WALL_COLUMN] > 150,
    `col5 ${peaks[5].toFixed(1)} … col0 ${peaks[WALL_COLUMN].toFixed(1)}`,
  )
  check(
    "wallcrash: 'wall splash' — the plates march diagonally back and toward the wall",
    wc.meta.rectPattern === 'wall splash' &&
      JSON.stringify(wc.rects.map((r) => `${r.row},${r.col}`)) ===
        JSON.stringify(['4,0', '4,1', '3,2', '2,3', '1,4', '0,5']),
    JSON.stringify([wc.meta.rectPattern, wc.rects.map((r) => `${r.row},${r.col}`)]),
  )
  check(
    'wallcrash: the plate row DEEPENS as the columns approach the wall',
    wc.rects.every((r, i) => i === 0 || r.row <= wc.rects[i - 1].row) &&
      wc.rects[0].row === 4 &&
      wc.rects[5].row === 0,
    JSON.stringify(wc.rects.map((r) => r.row)),
  )
  check(
    "wallcrash: every plate's joint is unfolded, so each plate really is rigid",
    wc.rects.every((r) => wc.columns[r.col].foldsDeg[r.row] === 0),
    JSON.stringify(wc.columns.map((col) => col.foldsDeg)),
  )
  check(
    'wallcrash: no layout warnings at all',
    layout.warnings.length === 0,
    JSON.stringify(layout.warnings.map((w) => w.code)),
  )
}

// --- the three storm presets differ from each other and from the calm family ----
{
  const shapes = named.map(([, cfg]) => JSON.stringify([cfg.grid, cfg.columns]))
  check('all twelve sampled presets have distinct shapes', new Set(shapes).size === shapes.length)
  const patterns = stormNamed.map(([, cfg]) => cfg.meta.rectPattern)
  check(
    'the storm presets name three different plate patterns',
    new Set(patterns).size === 3,
    JSON.stringify(patterns),
  )
  const calmPatterns = calmNamed.slice(0, 4).map(([, cfg]) => cfg.meta.rectPattern)
  check(
    'no storm pattern name collides with a calm one',
    patterns.every((p) => !calmPatterns.includes(p)),
    JSON.stringify([patterns, calmPatterns]),
  )
}

// =============================================================================
// 6. REGRESSION GUARD — the calm family's outputs must not have moved
//
// Hard-coded from the pre-WP10 build. Pitched fronts, the wall and the row range
// were all added as OPT-IN fields, so every one of these has to be byte-identical:
// if a change to the chain walker or the grounding solver shifts them, this fires.
// =============================================================================
{
  const EXPECTED = {
    flat: {
      folds: [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
      rects: ['1,1,v', '3,1,v', '1,4,v', '3,4,v'],
      pattern: 'mirrored quad',
    },
    calm: {
      folds: [
        [10, -10, -10, 10],
        [10, -10, -10, 10],
        [5, 5, -15, -5],
        [5, 5, -15, -5],
        [10, -10, -10, 10],
        [10, -10, -10, 10],
      ],
      rects: ['1,0,h', '3,0,h', '1,4,h', '3,4,h'],
      pattern: 'mirrored pairs',
    },
    wave: {
      folds: [
        [40, 0, -80, -1],
        [35, 0, -70, -1],
        [30, 0, -60, -0.5],
        [0, 25, 0, -86],
        [0, 22, 0, -73],
        [0, 18, 0, -57],
      ],
      rects: ['1,0,v', '1,1,v', '1,2,v', '2,3,v', '2,4,v', '2,5,v'],
      pattern: 'crest plates',
    },
    crash: {
      folds: [
        [0, 85, -115.5, 0],
        [10, 65, -110.25, 0],
        [20, 40, -97.75, 0],
        [25, 20, -79.9, 0],
        [15, 15, -52.5, 0],
        [5, 10, -25, 0],
      ],
      rects: ['3,0,v', '3,1,v', '3,2,v', '3,3,v', '3,4,v', '3,5,v'],
      pattern: 'landing plates',
    },
  }
  for (const [id, want] of Object.entries(EXPECTED)) {
    const cfg = buildPreset(id)
    check(
      `regression: ${id}'s fold sequences are unchanged`,
      JSON.stringify(cfg.columns.map((col) => col.foldsDeg)) === JSON.stringify(want.folds),
      JSON.stringify(cfg.columns.map((col) => col.foldsDeg)),
    )
    check(
      `regression: ${id}'s plates are unchanged`,
      JSON.stringify(cfg.rects.map((r) => `${r.row},${r.col},${r.orientation[0]}`)) ===
        JSON.stringify(want.rects),
      JSON.stringify(cfg.rects),
    )
    check(
      `regression: ${id} still names the pattern "${want.pattern}"`,
      cfg.meta.rectPattern === want.pattern,
      String(cfg.meta.rectPattern),
    )
    check(
      `regression: ${id} is still 6×5 with flat fronts on the floor`,
      cfg.grid.rows === 5 &&
        cfg.columns.every((col) => col.startPitchDeg === 0 && col.endSupport === 'floor'),
      JSON.stringify(cfg.grid),
    )
  }
  // The seeded generator too — the random stream and its templates must not shift.
  const RANDOM_EXPECTED = {
    1: {
      pattern: 'alternating-bands',
      folds: [
        [22, 0, -43, -2.5],
        [45, -25, -52, 0],
        [22, 0, -46, 4],
        [45, -25, -52, 0],
        [6, 0, -18.067091464598626, 12.067091464598626],
        [7, 15, -36.4, 0],
      ],
    },
    42: {
      pattern: 'mirrored-pairs',
      folds: [
        [27, -7, -32, -25],
        [22, 0, -44, 0],
        [12, 16, -37, -23],
        [21, -10, -15, -25],
        [22, 0, -44, 0],
        [23, 8, -55, -6.5],
      ],
    },
  }
  for (const [seed, want] of Object.entries(RANDOM_EXPECTED)) {
    const cfg = presetRandom(Number(seed))
    check(
      `regression: random(${seed}) still draws '${want.pattern}' with the same folds`,
      cfg.meta.rectPattern === want.pattern &&
        JSON.stringify(cfg.columns.map((col) => col.foldsDeg)) === JSON.stringify(want.folds),
      `${cfg.meta.rectPattern} ${JSON.stringify(cfg.columns.map((col) => col.foldsDeg))}`,
    )
  }
  // …and the solved geometry, not just the config: the flat lattice is exact.
  {
    const flatLayout = solveLayout(buildPreset('flat'))
    check(
      'regression: presetFlat still lays out on the exact 61cm lattice at y = 3.7',
      flatLayout.panels.every((p) => {
        const wantY = 3.7
        return p.position[1] === wantY && p.position[0] === 30 + 61 * p.col
      }),
      JSON.stringify(flatLayout.panels.slice(0, 3).map((p) => p.position)),
    )
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
