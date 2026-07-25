/**
 * grid-designer — procedural configuration presets (schema v2).
 *
 * HEADLESS ZONE (src/core/): pure functions, importable from plain node.
 *   - explicit `.js` extensions on ALL relative imports
 *   - may import `three` math classes only; never components / store / DOM
 *
 * Every function here returns a complete config that
 *   - passes `validateConfig` with `ok: true` (informational warnings such as
 *     W_CROSSCOL_POSITION may still appear; W_RECT_LENGTH cannot, because every
 *     preset inherits the default geometry where 2·size + gap = 60 + 1 + 60 = 121
 *     is exactly cell.rectLength, so a plate is an exact drop-in),
 *   - is free of W_BELOW_FLOOR: no panel dips under the floor, and
 *   - is free of E_END_FLOATING: EVERY column's last panel touches the ground.
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
 * Each sequence is authored as a CUMULATIVE PITCH profile and differenced into
 * hinge angles, because the pitch profile is what governs both ground rules:
 *
 *   ROW 0 is the shore, always ψ = 0 (flat on the ground at the window).
 *   ROWS 1..3 are the AUTHORED HEAD — the wave's shape. The chain climbs by
 *     60·sin ψ per row, so a head whose sines cancel arcs back down to the shore
 *     height; a head that only rises leaves the strip stranded in mid-air.
 *   ROW 4 is the LANDING. Its pitch is not authored: the profile is closed with
 *     ψ_4 = 0 (last panel parallel to the floor) and then, if that leaves the end
 *     panel floating, `core/ground.js`'s `solveGroundingFold` solves the last
 *     hinge for floor contact — one panel edge-down on the ground, the wave
 *     washing out at the back of the store.
 *
 * That two-step (author the head, solve the landing) is why the fold sequences
 * below contain odd angles like −47°: those are solved values, not design intent.
 */

import { DEFAULT_CONFIG, normalizeConfig, validateConfig } from './schema.js'
import { solveLayout } from './placement.js'
import { groundAllFolds } from './ground.js'

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
    description: 'Deterministic per seed; every strip arcs up and lands back on the floor.',
    seeded: true,
  },
]

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------
const GRID = { cols: 6, rows: 5 }
const DEG = Math.PI / 180
const asinDeg = (v) => Math.asin(v < -1 ? -1 : v > 1 ? 1 : v) / DEG
const sinDeg = (d) => Math.sin(d * DEG)
const clampNum = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v)
/**
 * Largest Σ sin ψ a random column's head may accumulate. 60cm·0.75 = 45cm of
 * height, which the one 60cm landing panel behind it can still give back.
 */
const RANDOM_MAX_RISE = 0.75

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
 * Close an authored HEAD (ψ_1..ψ_{rows-2}) into a full pitch profile by adding
 * the landing row at ψ = 0 — the last panel lying parallel to the floor. When the
 * chain has not come all the way back down that leaves the end in the air, which
 * `land()` then fixes by solving the last hinge.
 *
 * @param {number[]} head cumulative pitches for rows 1..rows-2
 * @returns {number[]} hinge angles, length grid.rows-1
 */
function foldsFromHead(head) {
  return foldsFromPitches([0, ...head, 0])
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

/**
 * Land every column that is not already resting on the floor: solve its last
 * surviving hinge for floor contact. `includeSunken` is on because a preset wants
 * its last panel ON the ground, not buried a hand's width under it (which would
 * satisfy the grounded-end rule while raising W_BELOW_FLOOR).
 *
 * @param {object} config
 * @returns {object} config (the same object when nothing needed landing)
 */
function land(config) {
  return groundAllFolds(config, { includeSunken: true }) ?? config
}

/** Columns whose last panel still floats. */
const floatingCols = (config) => solveLayout(config).violations.map((v) => v.col)

/** Panels dipping below y = 0 — nothing to stand on. */
const sunkenPanels = (config) =>
  solveLayout(config).warnings.filter((w) => w.code === 'W_BELOW_FLOOR')

// -----------------------------------------------------------------------------
// Presets
// -----------------------------------------------------------------------------
/** Flat reference surface: every joint unfolded. Trivially grounded everywhere. */
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
 * across the window, so the in-row connections stay nearly exact.
 *
 * Authored heads ψ_1..ψ_3, column 0 → 5:
 *   [10,0,-10] [10,10,-10] [5,10,-5] [0,10,-10] [0,0,10] [0,0,5]
 * Landed profiles (ψ_0..ψ_4):
 *   [0,10,0,-10,0]  [0,10,10,-10,-10] [0,5,10,-5,-10]
 *   [0,0,10,-10,0]  [0,0,0,10,-10]    [0,0,0,5,-5]
 * The first and fourth arc exactly back to shore height and land FLAT; the others
 * finish edge-down on the floor at a shallow angle.
 */
export function presetCalm() {
  return land(
    baseConfig({
      name: 'calm',
      folds: [
        foldsFromHead([10, 0, -10]),
        foldsFromHead([10, 10, -10]),
        foldsFromHead([5, 10, -5]),
        foldsFromHead([0, 10, -10]),
        foldsFromHead([0, 0, 10]),
        foldsFromHead([0, 0, 5]),
      ],
      meta: { preset: 'calm', notes: 'Gentle ripple drifting across the window.' },
    }),
  )
}

/**
 * A swell travelling across the grid: a ±35° crest that starts just behind the
 * shore at column 0 and moves one row deeper across the window.
 *
 * Authored heads ψ_1..ψ_3, column 0 → 5:
 *   [35,-35,0] [35,0,-35] [35,35,-35] [20,35,-20] [0,35,0] [0,20,20]
 * The first three arc back to shore height by themselves and land flat; the last
 * three are still up in the air at row 4 and get a solved landing hinge, which is
 * the wave breaking and washing out at the back of the store.
 *
 * Includes both plate types, each placed where the geometry allows it:
 *   - a VERTICAL plate at (1,2): column 2's joint 1 is unfolded (its head holds
 *     35° across rows 1 and 2), so the rigid 121cm plate can span them.
 *   - a HORIZONTAL plate at (1,0): columns 0 and 1 have both reached pitch 35° at
 *     row 1 with identical chains in front of it, so one plate lies flat across
 *     both — the only in-row connection the surface makes rigid.
 * Both sit at row 1, well in front of the landing hinge, so grounding the columns
 * never fights the plates.
 */
export function presetWave() {
  return land(
    baseConfig({
      name: 'wave',
      folds: [
        foldsFromHead([35, -35, 0]),
        foldsFromHead([35, 0, -35]),
        foldsFromHead([35, 35, -35]),
        foldsFromHead([20, 35, -20]),
        foldsFromHead([0, 35, 0]),
        foldsFromHead([0, 20, 20]),
      ],
      rects: [
        { row: 1, col: 2, orientation: 'vertical' },
        { row: 1, col: 0, orientation: 'horizontal' },
      ],
      meta: { preset: 'wave', notes: 'Swell travelling across the window, two rigid plates.' },
    }),
  )
}

/**
 * Steep and dramatic: near-vertical crests (cumulative pitch up to 88°) sweeping
 * back across the grid, with hinge angles as large as ±120°.
 *
 * Authored heads ψ_1..ψ_3, column 0 → 5:
 *   [88,0,-88] [60,60,-60] [45,60,-60] [20,45,-45] [0,0,45] [0,0,50]
 * A crest that steep has to be reached EARLY: with hinges capped at ±120° a chain
 * standing 88° up at row 2 can no longer get back down within the two rows it has
 * left, so the later columns trade crest height for crest depth and plunge to the
 * floor with a near-vertical last panel.
 */
export function presetCrash() {
  return land(
    baseConfig({
      name: 'crash',
      folds: [
        foldsFromHead([88, 0, -88]),
        foldsFromHead([60, 60, -60]),
        foldsFromHead([45, 60, -60]),
        foldsFromHead([20, 45, -45]),
        foldsFromHead([0, 0, 45]),
        foldsFromHead([0, 0, 50]),
      ],
      meta: { preset: 'crash', notes: 'Steep folds, crest coming down hard.' },
    }),
  )
}

/**
 * Deterministic random configuration.
 *
 * Same seed → byte-identical config. ALL draws happen up front in a fixed order,
 * so dropping a candidate rect never shifts the random stream.
 *
 * The head of each column is drawn as a RISE (two non-negative pitches) followed
 * by a DESCENT, and the descent is clamped into the band where both ground rules
 * can still be met:
 *   - it may not undo more than the rise (Σ sin ψ ≥ 0), or the strip would sink
 *     through the floor → W_BELOW_FLOOR;
 *   - it must leave the chain low enough that one 60cm panel can still reach the
 *     ground (Σ sin ψ ≤ RANDOM_MAX_RISE), or the landing is unreachable →
 *     E_END_FLOATING.
 * The landing hinge is then solved, exactly as for the hand-authored presets.
 *
 * The two candidate plates are made geometrically legal before the folds are
 * differenced out — a vertical plate flattens the joint it removes, a horizontal
 * plate copies its left column's pitch into its right one — and then kept only if
 * the whole config still validates, still lands every column and still keeps every
 * panel above the floor. That last test matters because overwriting a pitch can
 * turn a column's descent into a deeper drop than the rise that paid for it.
 *
 * @param {number} [seed=1]
 * @returns {object} config, valid, supported and fully grounded
 */
export function presetRandom(seed = 1) {
  const rnd = mulberry32(seed)
  const { cols, rows: rowCount } = GRID

  // --- fixed draw order --------------------------------------------------
  const draws = []
  for (let c = 0; c < cols; c++) {
    draws.push({
      rise1: Math.floor(rnd() * 46), // 0..45
      rise2: Math.floor(rnd() * 46), // 0..45
      tail: Math.floor(rnd() * 31) - 15, // -15..+15 asymmetry of the descent
    })
  }
  const vCand = {
    row: Math.floor(rnd() * (rowCount - 1)), // 0..rows-2
    col: Math.floor(rnd() * cols),
    orientation: 'vertical',
  }
  const hCand = {
    // rows-2 at most: a horizontal plate on the LAST row would pin two columns'
    // landing pitch to each other and block grounding them independently.
    row: Math.floor(rnd() * (rowCount - 1)),
    col: Math.floor(rnd() * (cols - 1)), // 0..cols-2
    orientation: 'horizontal',
  }

  // --- heads: rise, then a descent clamped into the landable band ----------
  const heads = draws.map(({ rise1, rise2, tail }) => {
    const rise = sinDeg(rise1) + sinDeg(rise2)
    const deepest = -asinDeg(rise) // undoes the whole rise (Σ sin ψ = 0)
    const highest = asinDeg(RANDOM_MAX_RISE - rise) // leaves the landing reachable
    const nominal = Math.round(-asinDeg(rise / 2) + tail)
    return [rise1, rise2, clampNum(nominal, Math.min(deepest, highest), highest)]
  })
  const folds = heads.map(foldsFromHead)

  const meta = { preset: 'random', seed, notes: `mulberry32 seed ${seed}` }
  const name = `random ${seed}`

  // --- candidate plates, made legal then kept only if they survive ---------
  let pitches = heads.map((head) => [0, ...head, 0])
  const rects = []
  for (const candidate of [vCand, hCand]) {
    const legal = pitches.map((p) => p.slice())
    if (candidate.orientation === 'vertical') {
      legal[candidate.col][candidate.row + 1] = legal[candidate.col][candidate.row]
    } else {
      legal[candidate.col + 1][candidate.row] = legal[candidate.col][candidate.row]
    }
    const trial = land(
      baseConfig({ name, folds: legal.map(foldsFromPitches), rects: [...rects, candidate], meta }),
    )
    if (
      validateConfig(trial).ok &&
      floatingCols(trial).length === 0 &&
      sunkenPanels(trial).length === 0
    ) {
      rects.push(candidate)
      pitches = legal
    }
  }

  return land(baseConfig({ name, folds: pitches.map(foldsFromPitches), rects, meta }))
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
