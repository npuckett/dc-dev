/**
 * grid-designer — procedural configuration presets.
 *
 * HEADLESS ZONE (src/core/): pure functions, importable from plain node.
 *   - explicit `.js` extensions on ALL relative imports
 *   - may import `three` math classes only; never components / store / DOM
 *
 * Every function here returns a complete config that passes `validateConfig`
 * with `ok: true` (warnings such as W_RECT_LENGTH / W_CROSSROW_FOLD are
 * expected and intentional — they surface real physical slack).
 *
 * The narrative these presets serve: row 0 is the shore at the storefront
 * window, always flat. "calm" is barely rippled, "wave" swells toward the
 * shore, "crash" is steep and dramatic.
 */

import { DEFAULT_CONFIG, normalizeConfig, validateConfig } from './schema.js'

// -----------------------------------------------------------------------------
// PRNG — mulberry32. Small, fast, deterministic per 32-bit seed.
// -----------------------------------------------------------------------------
/**
 * @param {number} seed
 * @returns {() => number} generator of floats in [0, 1)
 */
export function mulberry32(seed) {
  let a = seed >>> 0
  return function next() {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// -----------------------------------------------------------------------------
// Preset metadata (for the UI's preset bar)
// -----------------------------------------------------------------------------
export const PRESETS = [
  {
    id: 'flat',
    label: 'Flat',
    description: 'Every panel level on the floor — the reference state.',
    seeded: false,
  },
  {
    id: 'calm',
    label: 'Calm',
    description: 'Gentle low-angle ripples with a slow rise away from the shore.',
    seeded: false,
  },
  {
    id: 'wave',
    label: 'Wave',
    description: 'Pronounced accordion zig-zags swelling and breaking toward the shore.',
    seeded: false,
  },
  {
    id: 'crash',
    label: 'Crash',
    description: 'Steep folds and a near-vertical crest — water coming down hard.',
    seeded: false,
  },
  {
    id: 'random',
    label: 'Random',
    description: 'Deterministic per seed; angles kept inside safe ranges.',
    seeded: true,
  },
]

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------
const GRID = { cols: 6, rows: 5 }

function baseConfig({ name, zigzags, rowFoldsDeg, rects = [], overrides = {}, meta }) {
  const rows = zigzags.map((zigzagDeg, r) => ({
    zigzagDeg,
    jointOverridesDeg: { ...(overrides[r] ?? {}) },
  }))
  return normalizeConfig({
    ...DEFAULT_CONFIG,
    name,
    grid: { ...GRID },
    cell: { ...DEFAULT_CONFIG.cell },
    rows,
    rowFoldsDeg: rowFoldsDeg.slice(),
    rects: rects.map((rect) => ({ ...rect })),
    meta,
  })
}

// -----------------------------------------------------------------------------
// Presets
// -----------------------------------------------------------------------------
/** Flat reference surface: no zig-zag, no row folds. */
export function presetFlat() {
  return baseConfig({
    name: 'flat',
    zigzags: [0, 0, 0, 0, 0],
    rowFoldsDeg: [0, 0, 0, 0],
    meta: { preset: 'flat', notes: 'Reference state — all panels level.' },
  })
}

/** Barely-rippled water; everything low-angle. */
export function presetCalm() {
  return baseConfig({
    name: 'calm',
    zigzags: [0, 8, 14, 12, 6],
    rowFoldsDeg: [8, 14, 18, 10],
    meta: { preset: 'calm', notes: 'Gentle ripples, slow steady rise away from the shore.' },
  })
}

/**
 * A swell breaking toward the shore: zig-zag builds through the middle rows and
 * eases at the back, row pitch rises steeply then rolls back over the crest.
 *
 * Includes two rects for interest:
 *   - a vertical plate at column 0 (both cells tilt 0°, so the rigid-plate
 *     constraint is satisfied at any zig-zag amplitude) — it spans a folding row
 *     boundary on purpose, which raises the informational W_CROSSROW_FOLD flag.
 *   - a horizontal plate in the back row, removing joint 2 of that row.
 */
export function presetWave() {
  return baseConfig({
    name: 'wave',
    zigzags: [0, 15, 30, 38, 22],
    rowFoldsDeg: [12, 28, 40, -8],
    overrides: { 1: { 3: -45 } },
    rects: [
      { row: 2, col: 0, orientation: 'vertical' },
      { row: 4, col: 2, orientation: 'horizontal' },
    ],
    meta: { preset: 'wave', notes: 'Swell building and breaking toward the shore.' },
  })
}

/** Steep, dramatic: deep accordion folds and a near-vertical crest. */
export function presetCrash() {
  return baseConfig({
    name: 'crash',
    zigzags: [0, 40, 62, 70, 50],
    rowFoldsDeg: [30, 60, 75, -30],
    overrides: { 3: { 2: 75 } },
    meta: { preset: 'crash', notes: 'Steep folds, crest coming down hard.' },
  })
}

/**
 * Deterministic random configuration.
 *
 * Same seed → byte-identical config. All draws happen in a fixed order, so
 * dropping an invalid candidate rect never shifts the random stream.
 *
 * @param {number} [seed=1]
 * @returns {object} config, guaranteed `validateConfig(...).ok === true`
 */
export function presetRandom(seed = 1) {
  const rnd = mulberry32(seed)
  const { cols, rows: rowCount } = GRID

  // Row zig-zag amplitudes — row 0 is the shore and stays flat.
  const zigzags = [0]
  const overrides = {}
  for (let r = 1; r < rowCount; r++) {
    zigzags.push(5 + Math.floor(rnd() * 46)) // 5..50
    // Occasionally break the alternation with a signed override.
    const wantsOverride = rnd() < 0.3
    const j = Math.floor(rnd() * (cols - 1)) // 0..cols-2
    const sign = rnd() < 0.5 ? -1 : 1
    const mag = 10 + Math.floor(rnd() * 40) // 10..49
    if (wantsOverride) overrides[r] = { [j]: sign * mag }
  }

  // Row-to-row dihedrals: mostly rising, occasionally rolling back over.
  const rowFoldsDeg = []
  for (let r = 0; r < rowCount - 1; r++) {
    rowFoldsDeg.push(-10 + Math.floor(rnd() * 61)) // -10..50
  }

  // Candidate horizontal rects — only kept if the whole config still validates.
  const candidates = []
  for (let k = 0; k < 3; k++) {
    candidates.push({
      row: 1 + Math.floor(rnd() * (rowCount - 1)), // 1..rowCount-1
      col: Math.floor(rnd() * (cols - 1)), // 0..cols-2
      orientation: 'horizontal',
    })
  }

  const meta = { preset: 'random', seed, notes: `mulberry32 seed ${seed}` }

  const rects = []
  for (const candidate of candidates) {
    const trial = baseConfig({
      name: `random ${seed}`,
      zigzags,
      rowFoldsDeg,
      overrides,
      rects: [...rects, candidate],
      meta,
    })
    if (validateConfig(trial).ok) rects.push(candidate)
  }

  return baseConfig({
    name: `random ${seed}`,
    zigzags,
    rowFoldsDeg,
    overrides,
    rects,
    meta,
  })
}

/**
 * Build a preset by id.
 *
 * @param {string} id one of PRESETS[].id
 * @param {number} [seed] used by the 'random' preset
 * @returns {object} config
 */
export function buildPreset(id, seed = 1) {
  switch (id) {
    case 'flat': return presetFlat()
    case 'calm': return presetCalm()
    case 'wave': return presetWave()
    case 'crash': return presetCrash()
    case 'random': return presetRandom(seed)
    default: throw new Error(`unknown preset id: ${id}`)
  }
}
