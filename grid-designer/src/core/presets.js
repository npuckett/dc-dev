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
 *   - is free of W_BELOW_FLOOR: no panel dips under the floor,
 *   - is free of E_END_FLOATING: EVERY column's last panel touches the ground, and
 *   - is free of W_FEW_RECTS: every preset places AT LEAST `MIN_RECTS` = 4 plates.
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
 *   THE LAST ROW is the LANDING. Its pitch is generally not authored: the profile
 *     is closed and then, if that leaves the end panel floating, `core/ground.js`'s
 *     `solveGroundingFold` solves the column's LAST SURVIVING hinge for floor
 *     contact — one panel edge-down on the ground, the wave washing out at the back
 *     of the store.
 *
 * That two-step (author the head, solve the landing) is why the fold sequences
 * below contain odd angles like −47°: those are solved values, not design intent.
 *
 * =============================================================================
 * THE PLATE PATTERNS — WHY EVERY PRESET CARRIES FOUR OR MORE
 * =============================================================================
 * The 60×121 plate is half the panel kit, and a design that uses one or two reads
 * as a grid of squares with a couple of mistakes in it. So every preset places at
 * least `MIN_RECTS` = 4 plates BY ONE NAMEABLE RULE (schema.js, THE PLATE-PATTERN
 * RULE), records that rule in `meta.rectPattern`, and repeats it in `meta.notes`:
 *
 *   flat   'mirrored quad'    two stacked plates in each of the mirrored columns
 *                             1 and 4 — rows 1–2 and 3–4. 4 vertical plates.
 *   calm   'mirrored pairs'   horizontal plates bridging the outer column pairs
 *                             (0,1) and (4,5) at rows 1 and 3. 4 horizontal plates.
 *   wave   'crest plates'     one vertical plate across EVERY column's crest
 *                             plateau; the plateau steps a row deeper at
 *                             mid-window, so the plates trace the swell. 6 plates.
 *   crash  'landing plates'   every column's last two rows fused into one rigid
 *                             121cm plate that skids down into the floor. 6 plates.
 *   random one of four named templates drawn from the seed (see RANDOM_TEMPLATES).
 *
 * THE FOLDS ARE DESIGNED AROUND THE PLATES, NOT THE OTHER WAY ROUND. A vertical
 * plate at (r, c) removes joint r, so rows r and r+1 must sit at ONE pitch — a
 * PLATEAU in the pitch profile. Every preset above therefore authors its plateaus
 * first (a crest plateau for `wave`, a landing plateau for `crash`) and differences
 * the hinge angles out of them; hunting for legal plate positions in a profile
 * authored without them mostly finds none.
 *
 * Two knock-on effects of a plate on the grounded-end rule:
 *   - a plate spanning the LAST two rows removes the column's last hinge, so the
 *     landing is solved one joint further forward and the panel being landed is the
 *     121cm plate itself — twice the length, so nearly twice the reach;
 *   - the landing must therefore be solved AFTER the plates are in the config.
 *     `land()` is always the outermost call here for exactly that reason.
 */

import { DEFAULT_CONFIG, MIN_RECTS, normalizeConfig, validateConfig } from './schema.js'
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
    description: 'Every panel level on the floor, with a mirrored quad of plates.',
    seeded: false,
  },
  {
    id: 'calm',
    label: 'Calm',
    description: 'A symmetric ripple bridged by mirrored pairs of horizontal plates.',
    seeded: false,
  },
  {
    id: 'wave',
    label: 'Wave',
    description: 'A swell travelling across the window, a rigid plate on every crest.',
    seeded: false,
  },
  {
    id: 'crash',
    label: 'Crash',
    description: 'Near-vertical crests plunging onto one long landing plate per column.',
    seeded: false,
  },
  {
    id: 'random',
    label: 'Random',
    description: 'Deterministic per seed: one of four named plate patterns, always grounded.',
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
 * Largest Σ sin ψ a random column's head may accumulate when the LANDING PANEL IS
 * A SQUARE. 60cm·0.75 = 45cm of height, which the one 60cm panel behind it can
 * still give back.
 */
const RANDOM_MAX_RISE = 0.75
/**
 * The same budget when the landing panel is a 121cm PLATE (a vertical rect on the
 * last two rows). The plate is twice as long, so it reaches twice as far down;
 * 1.05 · 60cm = 63cm of climb still lands inside the ±120° hinge limit.
 */
const RANDOM_MAX_RISE_PLATED = 1.05
/**
 * Steepest pitch a two-row PLATEAU may sit at in a random column, degrees. Two
 * rows at the same pitch spend 2·sin ψ of the rise budget, so 22° (2·0.375 = 0.75)
 * is exactly what a square landing panel can still undo.
 */
const PLATEAU_MAX_DEG = 22

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
 * Close an authored HEAD (ψ_1..ψ_{rows-2}) into a full pitch profile by adding the
 * landing row at ψ = 0 — the last panel lying parallel to the floor. When the chain
 * has not come all the way back down that leaves the end in the air, which `land()`
 * then fixes by solving the column's last surviving hinge.
 *
 * @param {number[]} head cumulative pitches for rows 1..rows-2
 * @returns {number[]} pitch profile, length grid.rows
 */
function pitchesFromHead(head) {
  return [0, ...head, 0]
}

/**
 * Close an authored head by HOLDING its last pitch instead of levelling out — the
 * profile a vertical plate on the last two rows requires, since that plate removes
 * the final joint and both its rows must share one pitch (foldsDeg[rows-2] = 0).
 * `land()` then solves the hinge in FRONT of the plate, which tilts the whole
 * 121cm plate down onto the floor.
 *
 * @param {number[]} head cumulative pitches for rows 1..rows-2
 * @returns {number[]} pitch profile, length grid.rows
 */
function pitchesHeldToEnd(head) {
  return [0, ...head, head[head.length - 1]]
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

/** `baseConfig` from cumulative pitch PROFILES rather than hinge angles. */
function fromProfiles({ name, profiles, rects, meta }) {
  return baseConfig({ name, folds: profiles.map(foldsFromPitches), rects, meta })
}

/**
 * Land every column that is not already resting on the floor: solve its last
 * surviving hinge for floor contact. `includeSunken` is on because a preset wants
 * its last panel ON the ground, not buried a hand's width under it (which would
 * satisfy the grounded-end rule while raising W_BELOW_FLOOR).
 *
 * ALWAYS THE OUTERMOST CALL: a plate removes hinges and pins pitches, so which
 * hinge lands a column — and how far it has to reach — is only known once the
 * plates are in the config.
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

/**
 * Is this config a finished design? Valid, every column landed, nothing under the
 * floor, and enough plates to read as a pattern.
 *
 * @param {object} config
 * @returns {boolean}
 */
function isBuildable(config) {
  return (
    validateConfig(config).ok &&
    floatingCols(config).length === 0 &&
    sunkenPanels(config).length === 0 &&
    config.rects.length >= MIN_RECTS
  )
}

// -----------------------------------------------------------------------------
// Presets
// -----------------------------------------------------------------------------
/**
 * Flat reference surface: every joint unfolded, trivially grounded everywhere.
 *
 * PLATE PATTERN — 'mirrored quad': columns 1 and 4 are mirror images of each other
 * about the grid centre (c ↔ 5−c), and each carries TWO stacked vertical plates,
 * rows 1–2 and 3–4. Four plates, one shape, read twice. A flat grid satisfies every
 * rect constraint (all hinges are 0, so no plate is asked to bend), which makes
 * this the one preset whose pattern needs no fold design at all.
 */
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
    rects: [
      { row: 1, col: 1, orientation: 'vertical' },
      { row: 3, col: 1, orientation: 'vertical' },
      { row: 1, col: 4, orientation: 'vertical' },
      { row: 3, col: 4, orientation: 'vertical' },
    ],
    meta: {
      preset: 'flat',
      rectPattern: 'mirrored quad',
      notes:
        'Reference state — all panels level. Plate pattern "mirrored quad": two stacked ' +
        'vertical plates (rows 1–2 and 3–4) in each of the mirrored columns 1 and 4.',
    },
  })
}

/**
 * Barely-rippled water, LEFT–RIGHT SYMMETRIC about the grid centre: columns c and
 * 5−c carry the same fold sequence, so the surface reads as one shallow swell
 * standing in the window rather than travelling across it.
 *
 *   cols 0, 1, 4, 5   ψ = [0, 10, 0, −10, 0]     an arc that returns to shore
 *                                                height and lands FLAT
 *   cols 2, 3         ψ = [0, 5, 10, −5, −10]    a shallower double ripple that
 *                                                finishes edge-down on the floor
 * Both profiles cancel their own sines, so nothing here needs the grounding solver
 * and every hinge stays inside ±15°.
 *
 * PLATE PATTERN — 'mirrored pairs': four HORIZONTAL plates bridging the two outer
 * column pairs, (0,1) and (4,5), at rows 1 and 3. A horizontal plate is one rigid
 * 121cm piece lying across two columns, so it demands the two chains agree at that
 * row in pitch AND in position — which is exactly why those four columns are given
 * the identical profile above: their chains coincide, the plates drop in with zero
 * slack, and no W_CROSSCOL_POSITION appears. Rows 1 and 3 (not 4) keep the plates
 * clear of the landing row, so grounding a column can never fight a plate.
 */
export function presetCalm() {
  const arc = [0, 10, 0, -10, 0]
  const ripple = [0, 5, 10, -5, -10]
  return land(
    fromProfiles({
      name: 'calm',
      profiles: [arc, arc, ripple, ripple, arc, arc],
      rects: [
        { row: 1, col: 0, orientation: 'horizontal' },
        { row: 3, col: 0, orientation: 'horizontal' },
        { row: 1, col: 4, orientation: 'horizontal' },
        { row: 3, col: 4, orientation: 'horizontal' },
      ],
      meta: {
        preset: 'calm',
        rectPattern: 'mirrored pairs',
        notes:
          'Symmetric shallow ripple (columns c and 5−c share a profile). Plate pattern ' +
          '"mirrored pairs": horizontal plates bridging the outer column pairs (0,1) and ' +
          '(4,5) at rows 1 and 3.',
      },
    }),
  )
}

/**
 * A swell travelling across the grid, built as a CREST PLATEAU that steps one row
 * deeper at mid-window:
 *
 *   cols 0, 1, 2   ψ = [0, a, a, −a, ⟨solved⟩]   a = 40, 35, 30 — crest at rows 1–2
 *   cols 3, 4, 5   ψ = [0, 0, a, a, ⟨solved⟩]    a = 25, 22, 18 — crest at rows 2–3
 *
 * The first three arc up over the shore and come straight back down; the last three
 * run flat out of the window before rising, so their crest sits a row further back
 * and their landing hinge is steep — the wave breaking and washing out at the back
 * of the store. Amplitude decays across the window, so the surface reads as one
 * swell rolling in from the left and dying out to the right.
 *
 * PLATE PATTERN — 'crest plates': ONE vertical plate across every column's crest
 * plateau — (1,0), (1,1), (1,2), (2,3), (2,4), (2,5). Six plates, one rule, and
 * because the plateau steps back at mid-window they trace the swell diagonally in
 * plan. The plateaus exist BECAUSE of the plates: a vertical plate at (r, c) is one
 * rigid 121cm piece, so it removes joint r and pins rows r and r+1 to a single
 * pitch. Every crest here is two rows wide for exactly that reason.
 */
export function presetWave() {
  /** Crest at rows 1–2: up to the plateau, then straight back down. */
  const frontCrest = (a) => pitchesFromHead([a, a, -a])
  /** Crest at rows 2–3: flat out of the window, then up onto the plateau. */
  const backCrest = (a) => pitchesFromHead([0, a, a])
  return land(
    fromProfiles({
      name: 'wave',
      profiles: [
        frontCrest(40),
        frontCrest(35),
        frontCrest(30),
        backCrest(25),
        backCrest(22),
        backCrest(18),
      ],
      rects: [
        { row: 1, col: 0, orientation: 'vertical' },
        { row: 1, col: 1, orientation: 'vertical' },
        { row: 1, col: 2, orientation: 'vertical' },
        { row: 2, col: 3, orientation: 'vertical' },
        { row: 2, col: 4, orientation: 'vertical' },
        { row: 2, col: 5, orientation: 'vertical' },
      ],
      meta: {
        preset: 'wave',
        rectPattern: 'crest plates',
        notes:
          'Swell travelling across the window. Plate pattern "crest plates": one vertical ' +
          'plate across every column\'s crest plateau — rows 1–2 for columns 0–2, rows 2–3 ' +
          'for columns 3–5, so the plates trace the swell.',
      },
    }),
  )
}

/**
 * Steep and dramatic: a near-vertical crest (cumulative pitch 85° in column 0)
 * sweeping back across the grid and plunging onto the floor.
 *
 *   ψ = [0, ψ₁, ψ₂, d, d]  with (ψ₁, ψ₂) = (0,85) (10,75) (20,60) (25,45) (15,30) (5,15)
 *
 * and d — shared by the last two rows — solved by `core/ground.js`. The crest can
 * be this steep ONLY because of the plate pattern below: a 121cm plate gives back
 * up to 121·sin|d| of height where a 60cm panel gives back 60·sin|d|, so a column
 * standing 85° up at row 2 can still reach the floor within the ±120° hinge limit.
 * The plate lands on its far housing edge — the water hitting the ground hard.
 *
 * PLATE PATTERN — 'landing plates': every column's last two rows fused into ONE
 * rigid 121cm plate — (3,0) … (3,5). Six plates in a single band across the back of
 * the store, all skidding into the floor at different angles. Because the plate
 * removes each column's final joint, the landing is solved at joint 2 instead, and
 * the pitch profiles hold their last authored pitch to the end (`pitchesHeldToEnd`)
 * so foldsDeg[3] stays 0 as the plate requires.
 */
export function presetCrash() {
  /** Climb to ψ₂, then hold whatever the landing solver picks across the plate. */
  const plunge = (psi1, psi2) => pitchesHeldToEnd([psi1, psi2, 0])
  return land(
    fromProfiles({
      name: 'crash',
      profiles: [
        plunge(0, 85),
        plunge(10, 75),
        plunge(20, 60),
        plunge(25, 45),
        plunge(15, 30),
        plunge(5, 15),
      ],
      rects: [
        { row: 3, col: 0, orientation: 'vertical' },
        { row: 3, col: 1, orientation: 'vertical' },
        { row: 3, col: 2, orientation: 'vertical' },
        { row: 3, col: 3, orientation: 'vertical' },
        { row: 3, col: 4, orientation: 'vertical' },
        { row: 3, col: 5, orientation: 'vertical' },
      ],
      meta: {
        preset: 'crash',
        rectPattern: 'landing plates',
        notes:
          'Steep crest coming down hard. Plate pattern "landing plates": every column\'s ' +
          'last two rows fused into one rigid 121cm plate that skids down into the floor.',
      },
    }),
  )
}

// -----------------------------------------------------------------------------
// Random — named plate templates
// -----------------------------------------------------------------------------
/**
 * The plate patterns `presetRandom` draws from. A template is a RULE, not a list of
 * lucky positions: it says which cells get a plate AND what that implies for the
 * fold profiles, because a plate is rigid and pins the surface it covers.
 *
 * Each entry carries
 *   id        the rule's name, written into `meta.rectPattern`
 *   notes     one sentence of it, for `meta.notes`
 *   rects     the plates themselves (≥ MIN_RECTS by construction)
 *   plateaus  per column, the rows where a VERTICAL plate fuses rows r and r+1 —
 *             i.e. where the pitch profile must hold a two-row plateau
 *   pinTo     per column, the column whose row-1 pitch it must copy (−1 = free),
 *             which is what lets a HORIZONTAL plate span the two of them
 *
 * 'mirrored-pairs' comes first and is the FALLBACK: two stacked plates in two
 * mirrored columns is the one pattern that cannot fail — the plateaus are shallow
 * by construction and each plated column lands on a 121cm plate, which has reach to
 * spare.
 */
export const RANDOM_TEMPLATES = [
  {
    id: 'mirrored-pairs',
    notes:
      'two stacked vertical plates (rows 1–2 and 3–4) in each of the mirrored columns 1 and 4',
    rects: [
      { row: 1, col: 1, orientation: 'vertical' },
      { row: 3, col: 1, orientation: 'vertical' },
      { row: 1, col: 4, orientation: 'vertical' },
      { row: 3, col: 4, orientation: 'vertical' },
    ],
    plateaus: [[], [1, 3], [], [], [1, 3], []],
    pinTo: [-1, -1, -1, -1, -1, -1],
  },
  {
    id: 'alternating-bands',
    notes:
      'crest plates (rows 1–2) on the even columns, landing plates (rows 3–4) on the odd ones',
    rects: [
      { row: 1, col: 0, orientation: 'vertical' },
      { row: 3, col: 1, orientation: 'vertical' },
      { row: 1, col: 2, orientation: 'vertical' },
      { row: 3, col: 3, orientation: 'vertical' },
      { row: 1, col: 4, orientation: 'vertical' },
      { row: 3, col: 5, orientation: 'vertical' },
    ],
    plateaus: [[1], [3], [1], [3], [1], [3]],
    pinTo: [-1, -1, -1, -1, -1, -1],
  },
  {
    id: 'diagonal-cascade',
    notes: 'one vertical plate per column stepping a row deeper: (0,0), (1,1), (2,2), (3,3)',
    rects: [
      { row: 0, col: 0, orientation: 'vertical' },
      { row: 1, col: 1, orientation: 'vertical' },
      { row: 2, col: 2, orientation: 'vertical' },
      { row: 3, col: 3, orientation: 'vertical' },
    ],
    plateaus: [[0], [1], [2], [3], [], []],
    pinTo: [-1, -1, -1, -1, -1, -1],
  },
  {
    id: 'shore-rafts',
    notes:
      'the two front rows fused into horizontal plates across each column pair — (0,1), (2,3), (4,5)',
    rects: [
      { row: 0, col: 0, orientation: 'horizontal' },
      { row: 1, col: 0, orientation: 'horizontal' },
      { row: 0, col: 2, orientation: 'horizontal' },
      { row: 1, col: 2, orientation: 'horizontal' },
      { row: 0, col: 4, orientation: 'horizontal' },
      { row: 1, col: 4, orientation: 'horizontal' },
    ],
    plateaus: [[], [], [], [], [], []],
    // Row 0 sits at pitch 0 in every column, so the front rafts are always legal;
    // the row-1 rafts need each pair to reach the SAME pitch there, hence the pin.
    pinTo: [-1, 0, -1, 2, -1, 4],
  },
]

/**
 * The descent pitch that closes a head: it undoes about half the rise, nudged by
 * the draw's `tail`, then clamped into the band where both ground rules still hold.
 *   - it may not undo more than the rise (Σ sin ψ ≥ 0), or the strip sinks through
 *     the floor → W_BELOW_FLOOR;
 *   - it must leave the chain low enough for the landing panel to reach the ground
 *     (Σ sin ψ ≤ `budget`), or the landing is unreachable → E_END_FLOATING.
 *
 * @param {number} rise Σ sin ψ accumulated by the rows in front of it
 * @param {number} tail the draw's asymmetry, degrees
 * @param {number} budget RANDOM_MAX_RISE, or the plated variant
 * @returns {number} degrees
 */
function closingDescent(rise, tail, budget) {
  const deepest = -asinDeg(rise) // undoes the whole rise (Σ sin ψ = 0)
  const highest = asinDeg(budget - rise) // leaves the landing reachable
  const nominal = Math.round(-asinDeg(rise / 2) + tail)
  return clampNum(nominal, Math.min(deepest, highest), highest)
}

/**
 * One random column's cumulative pitch profile, authored AROUND its plateaus.
 *
 * The plateau set comes from the template and is what makes its plates legal:
 * `[1]` means rows 1 and 2 share a pitch, `[3]` means the last two rows do (so the
 * landing is solved one joint further forward, on a 121cm plate), `[1, 3]` means
 * both. A profile is built to fit those plateaus AND to keep its rise inside the
 * budget its landing panel can give back — never patched afterwards, because
 * overwriting a pitch is what turns a legal descent into a plunge through the floor.
 *
 * @param {{rise1:number, rise2:number, tail:number}} draw
 * @param {number[]} plateaus rows where a vertical plate fuses rows r, r+1
 * @param {number|null} psi1Pin a row-1 pitch this column must match, or null
 * @returns {number[]} pitch profile, length grid.rows
 */
function randomColumnPitches(draw, plateaus, psi1Pin) {
  const { rise1, rise2, tail } = draw
  const key = plateaus.join(',')

  // A plate on the last two rows lands the column on 121cm of plate, not 60cm of
  // square — twice the reach, so twice the rise budget.
  const plated = plateaus.includes(GRID.rows - 2)
  const budget = plated ? RANDOM_MAX_RISE_PLATED : RANDOM_MAX_RISE

  switch (key) {
    // Plate on rows 0–1: the shore row is flat, so row 1 is pinned flat with it.
    case '0': {
      const head = [0, rise2]
      return pitchesFromHead([...head, closingDescent(sinDeg(rise2), tail, budget)])
    }
    // Plate on rows 1–2: one plateau, then a free descent.
    case '1': {
      const a = Math.min(rise1, PLATEAU_MAX_DEG)
      return pitchesFromHead([a, a, closingDescent(2 * sinDeg(a), tail, budget)])
    }
    // Plate on rows 2–3: the plateau IS the crest; the landing row closes it.
    case '2': {
      const a = Math.min(rise2, PLATEAU_MAX_DEG)
      const psi1 = Math.min(rise1, Math.max(0, asinDeg(RANDOM_MAX_RISE - 2 * sinDeg(a))))
      return pitchesFromHead([Math.round(psi1), a, a])
    }
    // Plate on rows 3–4: climb, then let the solver tilt the whole plate down.
    case '3': {
      const psi1 = rise1
      const psi2 = Math.min(rise2, Math.round(asinDeg(RANDOM_MAX_RISE_PLATED - sinDeg(psi1))))
      return pitchesHeldToEnd([psi1, Math.max(0, psi2), 0])
    }
    // Plates on rows 1–2 AND 3–4: a plateau, then the plate the solver tilts down.
    case '1,3': {
      const a = Math.min(rise1, PLATEAU_MAX_DEG)
      return pitchesHeldToEnd([a, a, 0])
    }
    // No plate in this column: the plain rise / rise / descend head.
    default: {
      const psi1 = psi1Pin === null ? rise1 : psi1Pin
      const rise = sinDeg(psi1) + sinDeg(rise2)
      return pitchesFromHead([psi1, rise2, closingDescent(rise, tail, budget)])
    }
  }
}

/**
 * Deterministic random configuration.
 *
 * Same seed → byte-identical config. ALL draws happen up front in a fixed order
 * (six column draws, then the template), so nothing downstream can shift the
 * random stream.
 *
 * The seed picks one of `RANDOM_TEMPLATES` — a NAMED plate pattern — and the six
 * fold profiles are then authored to fit it: the template's plateaus become two-row
 * plateaus in the pitch profiles, its horizontal plates pin their column pair to a
 * common row-1 pitch, and each profile's rise is clamped to what its landing panel
 * (60cm square, or 121cm plate) can give back. `land()` runs last, once the plates
 * are in place.
 *
 * If a drawn template somehow does not produce a buildable design, the remaining
 * templates are tried in a fixed rotation and 'mirrored-pairs' — whose shallow
 * plateaus and 121cm landing plates cannot fail — is the final fallback. A random
 * config therefore NEVER comes back with fewer than MIN_RECTS plates.
 *
 * @param {number} [seed=1]
 * @returns {object} config, valid, supported, fully grounded, ≥ 4 plates
 */
export function presetRandom(seed = 1) {
  const rnd = mulberry32(seed)
  const { cols } = GRID

  // --- fixed draw order --------------------------------------------------
  const draws = []
  for (let c = 0; c < cols; c++) {
    draws.push({
      rise1: Math.floor(rnd() * 46), // 0..45
      rise2: Math.floor(rnd() * 46), // 0..45
      tail: Math.floor(rnd() * 31) - 15, // -15..+15 asymmetry of the descent
    })
  }
  const drawn = Math.floor(rnd() * RANDOM_TEMPLATES.length)

  const build = (template) => {
    const profiles = draws.map((draw, c) => {
      const pin = template.pinTo[c]
      const psi1Pin =
        pin >= 0 ? randomColumnPitches(draws[pin], template.plateaus[pin], null)[1] : null
      return randomColumnPitches(draw, template.plateaus[c], psi1Pin)
    })
    return land(
      fromProfiles({
        name: `random ${seed}`,
        profiles,
        rects: template.rects,
        meta: {
          preset: 'random',
          seed,
          rectPattern: template.id,
          notes: `mulberry32 seed ${seed} · plate pattern "${template.id}": ${template.notes}`,
        },
      }),
    )
  }

  // The drawn template first, then the rest in a fixed rotation — deterministic,
  // and 'mirrored-pairs' (index 0) is reached last on the way round.
  for (let i = 0; i < RANDOM_TEMPLATES.length; i++) {
    const template = RANDOM_TEMPLATES[(drawn + i) % RANDOM_TEMPLATES.length]
    const candidate = build(template)
    if (isBuildable(candidate)) return candidate
  }
  return build(RANDOM_TEMPLATES[0])
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
