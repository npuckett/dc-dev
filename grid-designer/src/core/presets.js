/**
 * grid-designer — procedural configuration presets (schema v2).
 *
 * HEADLESS ZONE (src/core/): pure functions, importable from plain node.
 *   - explicit `.js` extensions on ALL relative imports
 *   - may import `three` math classes only; never components / store / DOM
 *
 * Every function here returns a complete config that passes `validateConfig`
 * with `ok: true` (warnings such as W_RECT_LENGTH are expected and intentional —
 * they surface real physical slack), and whose solved layout is free of
 * W_BELOW_FLOOR: every fold chain stays supported on the floor.
 *
 * =============================================================================
 * HOW THESE ARE BUILT
 * =============================================================================
 * The model folds along COLUMNS only (see core/schema.js). A preset is therefore
 * six fold sequences — one per column, four signed hinge angles each — and the
 * interesting knob is the PHASE relationship between them:
 *
 *   - identical sequences  → a cylindrical fold, perfect in-row joints
 *   - phase-shifted        → a wave travelling across the window (each column's
 *                            crest one step further back than its left neighbour)
 *
 * Each sequence is authored as a CUMULATIVE PITCH profile ψ_0..ψ_4 (ψ_0 = 0 by
 * construction — row 0 is the shore, flat on the ground) and differenced into
 * hinge angles, because the pitch profile is what governs ground support: the
 * chain's height only ever falls while ψ < 0, so as long as a descent comes
 * after the rise that paid for it, nothing ends up under the floor.
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
    description: 'Low-amplitude ripples, each column’s crest a step behind the last.',
    seeded: false,
  },
  {
    id: 'wave',
    label: 'Wave',
    description: 'A clear swell travelling across the window, plus two rigid plates.',
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
    description: 'Deterministic per seed; rising profiles keep every strip supported.',
    seeded: true,
  },
]

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------
const GRID = { cols: 6, rows: 5 }

/**
 * Difference a cumulative pitch profile into per-joint hinge angles.
 * `pitches[0]` must be 0 (row 0 is the shore).
 *
 * @param {number[]} pitches length grid.rows
 * @returns {number[]} length grid.rows-1
 */
function foldsFromPitches(pitches) {
  const folds = []
  for (let k = 0; k + 1 < pitches.length; k++) folds.push(pitches[k + 1] - pitches[k])
  return folds
}

/**
 * @param {{name:string, folds:number[][], rects?:Array, meta:object}} spec
 * @returns {object} normalized config
 */
function baseConfig({ name, folds, rects = [], meta }) {
  return normalizeConfig({
    ...DEFAULT_CONFIG,
    name,
    grid: { ...GRID },
    cell: { ...DEFAULT_CONFIG.cell },
    columns: folds.map((foldsDeg) => ({ foldsDeg: foldsDeg.slice() })),
    rects: rects.map((rect) => ({ ...rect })),
    meta,
  })
}

// -----------------------------------------------------------------------------
// Presets
// -----------------------------------------------------------------------------
/** Flat reference surface: every joint unfolded. */
export function presetFlat() {
  return baseConfig({
    name: 'flat',
    folds: [
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
    ],
    meta: { preset: 'flat', notes: 'Reference state — all panels level.' },
  })
}

/**
 * Barely-rippled water: a ±10° ridge whose crest drifts one row further back
 * every couple of columns, so the in-row connections stay nearly exact.
 *
 * Pitch profiles (ψ_0..ψ_4), column 0 → 5:
 *   [0,10,-10,0,0] [0,10,0,-10,0] [0,5,10,-10,-5]
 *   [0,0,10,0,-10] [0,0,5,10,-5]  [0,0,0,10,0]
 */
export function presetCalm() {
  return baseConfig({
    name: 'calm',
    folds: [
      foldsFromPitches([0, 10, -10, 0, 0]),
      foldsFromPitches([0, 10, 0, -10, 0]),
      foldsFromPitches([0, 5, 10, -10, -5]),
      foldsFromPitches([0, 0, 10, 0, -10]),
      foldsFromPitches([0, 0, 5, 10, -5]),
      foldsFromPitches([0, 0, 0, 10, 0]),
    ],
    meta: { preset: 'calm', notes: 'Gentle ripple drifting across the window.' },
  })
}

/**
 * A swell travelling across the grid: a ±35° crest that starts just behind the
 * shore at column 0 and moves one row deeper every two columns.
 *
 * Pitch profiles (ψ_0..ψ_4), column 0 → 5:
 *   [0,35,-35,0,0]  [0,35,0,-35,0]  [0,35,35,-35,-35]
 *   [0,20,35,0,-35] [0,0,35,35,-35] [0,0,20,35,0]
 *
 * Includes both plate types, each placed where the geometry allows it:
 *   - a VERTICAL plate at (1,2): column 2's joint 1 is unfolded (0°), so the
 *     rigid 121cm plate can span rows 1 and 2 of that strip.
 *   - a HORIZONTAL plate at (1,0): columns 0 and 1 have both reached pitch 35°
 *     at row 1 with identical chains in front of it, so one plate lies flat
 *     across both — the only in-row connection the surface makes rigid.
 */
export function presetWave() {
  return baseConfig({
    name: 'wave',
    folds: [
      foldsFromPitches([0, 35, -35, 0, 0]),
      foldsFromPitches([0, 35, 0, -35, 0]),
      foldsFromPitches([0, 35, 35, -35, -35]),
      foldsFromPitches([0, 20, 35, 0, -35]),
      foldsFromPitches([0, 0, 35, 35, -35]),
      foldsFromPitches([0, 0, 20, 35, 0]),
    ],
    rects: [
      { row: 1, col: 2, orientation: 'vertical' },
      { row: 1, col: 0, orientation: 'horizontal' },
    ],
    meta: { preset: 'wave', notes: 'Swell travelling across the window, two rigid plates.' },
  })
}

/**
 * Steep and dramatic: near-vertical crests (cumulative pitch up to 88°) sweeping
 * back across the grid, with hinge angles as large as ±120°.
 *
 * Pitch profiles (ψ_0..ψ_4), column 0 → 5:
 *   [0,88,0,-88,0]  [0,60,60,-60,-60] [0,45,88,0,-88]
 *   [0,20,70,88,0]  [0,0,45,88,45]    [0,0,20,70,88]
 */
export function presetCrash() {
  return baseConfig({
    name: 'crash',
    folds: [
      foldsFromPitches([0, 88, 0, -88, 0]),
      foldsFromPitches([0, 60, 60, -60, -60]),
      foldsFromPitches([0, 45, 88, 0, -88]),
      foldsFromPitches([0, 20, 70, 88, 0]),
      foldsFromPitches([0, 0, 45, 88, 45]),
      foldsFromPitches([0, 0, 20, 70, 88]),
    ],
    meta: { preset: 'crash', notes: 'Steep folds, crest coming down hard.' },
  })
}

/**
 * Deterministic random configuration.
 *
 * Same seed → byte-identical config. ALL draws happen up front in a fixed order,
 * so dropping a candidate rect never shifts the random stream.
 *
 * Cumulative pitches are drawn NON-NEGATIVE (0..45°): a monotonically rising
 * strip can never dip below the floor, which is what keeps every seed free of
 * W_BELOW_FLOOR without rejection sampling.
 *
 * The two candidate plates are made geometrically legal before the folds are
 * differenced out — a vertical plate flattens the joint it removes, a horizontal
 * plate copies its left column's pitch into its right one — and then kept only if
 * the whole config still validates.
 *
 * @param {number} [seed=1]
 * @returns {object} config, guaranteed `validateConfig(...).ok === true`
 */
export function presetRandom(seed = 1) {
  const rnd = mulberry32(seed)
  const { cols, rows: rowCount } = GRID

  // --- fixed draw order --------------------------------------------------
  const pitches = []
  for (let c = 0; c < cols; c++) {
    const profile = [0]
    for (let r = 1; r < rowCount; r++) profile.push(Math.floor(rnd() * 46)) // 0..45
    pitches.push(profile)
  }
  const vCand = {
    row: Math.floor(rnd() * (rowCount - 1)), // 0..rows-2
    col: Math.floor(rnd() * cols),
    orientation: 'vertical',
  }
  const hCand = {
    row: Math.floor(rnd() * rowCount),
    col: Math.floor(rnd() * (cols - 1)), // 0..cols-2
    orientation: 'horizontal',
  }

  // --- make the candidates legal ------------------------------------------
  pitches[vCand.col][vCand.row + 1] = pitches[vCand.col][vCand.row]
  pitches[hCand.col + 1][hCand.row] = pitches[hCand.col][hCand.row]

  const folds = pitches.map(foldsFromPitches)
  const meta = { preset: 'random', seed, notes: `mulberry32 seed ${seed}` }
  const name = `random ${seed}`

  const rects = []
  for (const candidate of [vCand, hCand]) {
    const trial = baseConfig({ name, folds, rects: [...rects, candidate], meta })
    if (validateConfig(trial).ok) rects.push(candidate)
  }

  return baseConfig({ name, folds, rects, meta })
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
