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
 *   - is free of E_END_FLOATING: every FLOOR-supported column's last panel touches
 *     the ground (a WALL-supported column is exempt by design — see the storm
 *     family below), and
 *   - is free of W_FEW_RECTS: every preset places AT LEAST `MIN_RECTS` = 4 plates.
 *
 * =============================================================================
 * TWO FAMILIES
 * =============================================================================
 * THE CALM FAMILY — flat / calm / wave / crash / random. Six columns, 5 rows,
 *   every front panel FLAT on the shore (`startPitchDeg` 0) and every column
 *   standing on the floor. `random` is this family's generator.
 *
 * THE STORM FAMILY — swell / surge / wallcrash. Different rules:
 *   1. EVERY column has a PITCHED FRONT (`startPitchDeg` ≠ 0). That is the
 *      family's signature: the water is already moving as it comes through the
 *      window, instead of lying flat on the shore first.
 *   2. Higher row resolution (8, 7, 6) — the shape needs the depth. `swell` is the
 *      deepest at the grid's 8-row ceiling; it and `surge` differ in shape (a low
 *      asymmetric drift ramping to ~117cm at the wall versus one uniform 168cm crest)
 *      and in plate pattern (one spine plate per column, doubled at the wall, versus
 *      two in every column). `swell` spends its rows on a DEEP CREST and a
 *      three-panel LEE rather than on height — see `presetSwell`, WHY 8 ROWS.
 *   3. Plates cross ROWS (vertical 121cm spines) rather than bridging columns, so
 *      each strip reads as a long rigid run inside a moving surface.
 *   4. 'wallcrash' additionally engages THE WALL: its column 0 is
 *      `endSupport: 'wall'` and ends deliberately HIGH, water splashing up the −X
 *      wall while the other five land on the floor.
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
 * hinge angles, because the pitch profile is what governs both ground rules. A
 * profile is the whole column: `profiles[c][0]` becomes that column's
 * `startPitchDeg` (the front panel's pitch) and the successive differences become
 * its `foldsDeg` — see `fromProfiles`.
 *
 *   ROW 0 is the front / window panel. In the CALM family its pitch is 0 (flat on
 *     the shore); in the STORM family it is deliberately not. Either way the front
 *     panel rests on the floor by construction (schema.js `frontRestY`), so a
 *     pitched front costs nothing in support — it just starts the chain at a
 *     different height.
 *   ROWS 1..rows-2 are the AUTHORED HEAD — the wave's shape. The chain climbs by
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
 *   swell  'doubled spine plates'
 *                             one vertical plate at the CREST (rows 3–4 of 8) in EVERY
 *                             column — a long rigid spine carrying the windward rise
 *                             of a drift banking against the wall — DOUBLED in the two
 *                             wall-most (tallest) columns, which carry a second plate
 *                             right behind it (rows 5–6) as their slipface. 8 plates.
 *   surge  'double plates'    a crest plate AND a landing plate in every column:
 *                             twelve 121cm plates, the steepest floor-grounded
 *                             design the kit can stand up.
 *   wallcrash 'wall splash'   the plates MARCH diagonally back and toward the −X
 *                             wall — (0,5) (1,4) (2,3) (3,2) (4,1) (4,0) — with the
 *                             last two rearing up the wall itself. 6 plates.
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

import {
  DEFAULT_CONFIG,
  MIN_RECTS,
  frontRestY,
  normalizeConfig,
  validateConfig,
} from './schema.js'
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
//
// `family` groups the bar: 'calm' (flat fronts on the shore, 5 rows) then 'storm'
// (pitched fronts everywhere, deeper grids, wall engagement). The bar draws a
// divider between the two.
// -----------------------------------------------------------------------------
export const PRESETS = [
  {
    id: 'flat',
    label: 'Flat',
    family: 'calm',
    description: 'Every panel level on the floor, with a mirrored quad of plates.',
    seeded: false,
  },
  {
    id: 'calm',
    label: 'Calm',
    family: 'calm',
    description: 'A symmetric ripple bridged by mirrored pairs of horizontal plates.',
    seeded: false,
  },
  {
    id: 'wave',
    label: 'Wave',
    family: 'calm',
    description: 'A swell travelling across the window, a rigid plate on every crest.',
    seeded: false,
  },
  {
    id: 'crash',
    label: 'Crash',
    family: 'calm',
    description: 'Near-vertical crests plunging onto one long landing plate per column.',
    seeded: false,
  },
  {
    id: 'random',
    label: 'Random',
    family: 'calm',
    description: 'Deterministic per seed: one of four named plate patterns, always grounded.',
    seeded: true,
  },
  {
    id: 'swell',
    label: 'Swell',
    family: 'storm',
    description:
      'STORM · 8 rows. Half water swell, half snow drift: a broken, chopping shoreline, then a long gradual windward rise — three rows plus a spine plate — cresting DEEP in the grid so the approach from the window stays open, and a steeper lee. Ramps from a 42cm start at the far jamb to a ~117cm crest against the wall, the two wall-most columns carrying a doubled spine. All grounded.',
    seeded: false,
  },
  {
    id: 'surge',
    label: 'Surge',
    family: 'storm',
    description:
      'STORM · 7 rows. Steep 25–45° fronts, a crest plate AND a landing plate in every column, peaks well past 150cm — still all on the floor.',
    seeded: false,
  },
  {
    id: 'wallcrash',
    label: 'Wallcrash',
    family: 'storm',
    description:
      'STORM · 6 rows. Amplitude ramps toward the −X wall beside column 0; that column is wall-supported and ends high, splashing up it.',
    seeded: false,
  },
]

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------
/** Every preset is six columns wide — the installation's real width. */
const COLS = 6
/** The calm family's row resolution. The storm family overrides it per preset. */
const CALM_ROWS = 5
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
 * Height a chain has reached after a run of `[length, pitch]` segments, starting
 * from `startY`. The 1cm gap advances are deliberately IGNORED: across a whole
 * column they add well under 1cm in y, which is far below the resolution the storm
 * family's angles are rounded to, and leaving them out keeps this a one-line closed
 * form the preset code can reason with. The authoritative geometry is still
 * `columnChain`; this is only used to CHOOSE angles, never to place anything.
 *
 * @param {number} startY cm
 * @param {Array<[number, number]>} steps [length cm, cumulative pitch deg]
 * @returns {number} cm
 */
const climbTo = (startY, steps) => steps.reduce((y, [L, psi]) => y + L * sinDeg(psi), startY)

/**
 * The (negative, whole-degree) pitch a `length`-cm panel needs to take a chain
 * standing `fromCm` up down to `toCm`. Clamped into a panel's actual reach, so an
 * over-ambitious ask becomes the steepest available descent rather than NaN.
 *
 * @param {number} fromCm height at the start of the panel
 * @param {number} toCm height wanted at its far end
 * @param {number} length the panel's length along the chain, cm
 * @returns {number} degrees, ≤ 0
 */
function descentPitch(fromCm, toCm, length) {
  return -Math.round(asinDeg(clampNum((fromCm - toCm) / length, 0, 1)))
}

/**
 * The crest pitch a 121cm plate needs so a chain that has climbed to `fromCm`
 * tops out at `crestCm`. This is what lets the storm family give every column the
 * SAME crest height while their front pitches differ: a steeper front has already
 * spent more of the budget, so its plate sits flatter.
 *
 * @param {number} fromCm height at the start of the plate
 * @param {number} crestCm height wanted at its far end
 * @param {number} length the plate's length along the chain, cm
 * @returns {number} degrees, ≥ 0
 */
function crestPitch(fromCm, crestCm, length) {
  return Math.round(asinDeg(clampNum((crestCm - fromCm) / length, 0, 1)))
}

/**
 * Difference a cumulative pitch profile into per-joint hinge angles.
 *
 * `pitches[0]` is the column's `startPitchDeg` (the front panel's pitch) — 0 for
 * the calm family, non-zero throughout the storm family — and is NOT part of the
 * result: only the rows-1 differences are hinges.
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
 * @param {{name:string, rows?:number, folds:number[][], startPitches?:number[],
 *          endSupports?:string[], rects?:Array, meta:object}} spec
 * @returns {object} normalized config
 */
function baseConfig({
  name,
  rows = CALM_ROWS,
  folds,
  startPitches,
  endSupports,
  rects = [],
  meta,
}) {
  return normalizeConfig({
    ...DEFAULT_CONFIG,
    name,
    grid: { cols: COLS, rows },
    cell: { ...DEFAULT_CONFIG.cell },
    columns: folds.map((foldsDeg, c) => ({
      foldsDeg: foldsDeg.slice(),
      startPitchDeg: startPitches ? startPitches[c] : 0,
      endSupport: endSupports ? endSupports[c] : 'floor',
    })),
    rects: rects.map((rect) => ({ ...rect })),
    meta,
  })
}

/**
 * `baseConfig` from cumulative pitch PROFILES rather than hinge angles: each
 * profile's first entry becomes its column's `startPitchDeg` and the successive
 * differences become its hinges, so `rows` follows the profile length. That is why
 * the storm family can pitch its fronts without a second code path — a pitched
 * front is simply a profile that does not start at 0.
 */
function fromProfiles({ name, profiles, rects, meta, endSupports }) {
  return baseConfig({
    name,
    rows: profiles[0].length,
    folds: profiles.map(foldsFromPitches),
    startPitches: profiles.map((p) => p[0]),
    endSupports,
    rects,
    meta,
  })
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
// THE STORM FAMILY — pitched fronts, deeper grids, the wall
//
// Shared rules (see TWO FAMILIES at the top):
//   - EVERY column's front panel is pitched (`startPitchDeg` ≠ 0) — the signature,
//     asserted in tests/test-presets.mjs;
//   - the plates run ALONG the columns (vertical 121cm spines crossing rows), never
//     across them;
//   - the crest height is authored as a TARGET and each column's plate pitch is
//     solved back from it (`crestPitch`), so the AMPLITUDE PROFILE is the design
//     input and the plate angles are consequences. 'surge' gives all six columns the
//     same target (one uniform 168cm crest); 'swell' gives them a ramp that peaks at
//     the wall. Either way a column whose front has already spent more of the climb
//     budget automatically gets a flatter plate.
// -----------------------------------------------------------------------------
/**
 * How high each 'swell' column tops out, cm — the swell's AMPLITUDE PROFILE, and
 * the whole shape of the design: a drift piling up AGAINST THE WALL. The wall is
 * the plane x = `WALL_X` = 0 beside COLUMN 0 (schema.js), so the ramp runs DOWN the
 * column index: column 0 is the tall end and column 5, the far jamb — the LEFT of
 * the 3D view, since the camera looks in from the window with +X to the left — is
 * where the surface starts low.
 *
 * A WIDE DYNAMIC RANGE UNDER A LOW CEILING is the point of these numbers: the
 * previous version ramped 150 → 80cm, which read as too tall and too uniform (a
 * high wall of water rather than a drift). 116 → 42 keeps the same 15cm-per-column
 * step, so the ramp still reads unmistakably from the floor, but it nearly doubles
 * the ratio between the two ends (2.8× instead of 1.9×) while taking 35cm off the
 * peak.
 */
const SWELL_CREST_CM = [116, 101, 86, 71, 56, 42]
/**
 * 'swell' is the one preset at the grid's 8-ROW CEILING (schema.js `MAX_ROWS`), and
 * the reason is the CREST ROW below: reaching the crest a row deeper than the old
 * build did, without turning the lee into a cliff, needs eight rows. See
 * `presetSwell`, WHY 8 ROWS.
 */
const SWELL_ROWS = 8
/**
 * The row the SPINE PLATE starts on — and therefore the crest row, since the chain
 * tops out at the far end of the plate. Rows 0–2 climb (a window row, the shore
 * shoulder and one more square), so the crest lands at the end of row 4: ONE UNIT
 * DEEPER than the old rows 2–3 spine, which is what keeps the tall part of the
 * surface from occluding everything behind it when read from the window.
 */
const SWELL_SPINE_ROW = 3
/**
 * The row the SECOND plate starts on in the doubled columns — the slipface, right
 * behind the spine, leaving row 7 as their landing square.
 */
const SWELL_SLIPFACE_ROW = 5
/**
 * The columns that carry a SECOND plate: the two wall-most, i.e. the two TALLEST
 * strips (`SWELL_CREST_CM` peaks at column 0). Eight plates from one clause — "one
 * spine plate per column, doubled on the two wall-most" — rather than two extras
 * scattered where they happen to fit.
 */
const SWELL_DOUBLED_COLS = [0, 1]
/**
 * How much of the rise left after the shore the CLIMB ROW takes, relative to an even
 * spread over itself and the spine plate behind it (1.0 = even). Over 1 on purpose:
 * the climb row is steeper than the plate, so the windward DECELERATES into the
 * crest — a rounded top rather than a corner — and the plate, being the last thing
 * before the crest, is the flattest part of the rise.
 */
const SWELL_CLIMB_LEAD = 1.6
/**
 * Fraction of its own crest height each column hands to its 60cm landing panel.
 * A FRACTION rather than one shared height, so every column's lee is the SAME
 * SHAPE scaled — the low jamb columns get a gently sloping lee instead of two flat
 * rows and one steep drop at the end, which is what a single 46cm handover did once
 * the crest at the far end came down to 42. 0.35 of the crest is always well inside
 * what one 60cm panel can give back, so no column ever returns E_UNGROUNDABLE.
 */
const SWELL_HANDOVER_FRACTION = 0.35
/**
 * How much of the lee drop the FIRST descent row takes (the rest goes to the
 * second). Under half on purpose: it makes the lee a smooth S — gradual off the
 * crest, steepest at the slipface in the middle, easing into the landing — instead
 * of one hard break behind the plate. It is also what keeps the hinge behind the
 * spine down near −50° rather than the −89° an evenly split lee needed.
 *
 * It applies to the columns whose lee is TWO SQUARES, i.e. everything outside
 * `SWELL_DOUBLED_COLS`; a doubled column's lee is one rigid 121cm plate, so there is
 * no split to make — it takes the whole drop at a single pitch.
 */
const SWELL_LEE_SPLIT = 0.4
/**
 * 'swell' front-panel pitch, degrees, as a function of the column index: 42° hard
 * up at the wall down to 20° at the far jamb. The ramp is STEEPER THAN THE CREST
 * RAMP in relative terms (2.1× end to end) because the fronts are what "starts much
 * lower on the left" is read from — the far jamb comes out of the window almost
 * lying down while the wall column rears up. It still ramps the same way the crest
 * does, so the whole surface leans toward the wall.
 */
const swellFrontDeg = (c) => Math.round(42 - 4.4 * c)
/**
 * The SHORE BEND: the signed hinge 'swell' puts at joint 0, between the window row
 * and the row behind it, per column.
 *
 * Every value is NEGATIVE and at least 12° — a real crease, not a ramp. The front
 * rears up out of the window and the row behind it drops back to a SHOULDER, so the
 * two rows a person at the window actually reads bend in OPPOSITE directions and
 * the shoreline breaks instead of inclining. The magnitudes are deliberately
 * uneven (13–22°) so the six columns break at different sharpnesses and the shore
 * reads as CHOP rather than as one uniform fold.
 *
 * It is authored, never solved: the spine plate behind it absorbs whatever climb is
 * left, which is why the shore detail can be spent freely without stealing from the
 * main windward rise.
 */
const SWELL_SHORE_BEND_DEG = [-15, -20, -14, -22, -16, -13]
/** How high every 'surge' column tops out, cm — the drama, and well past 150. */
const SURGE_CREST_CM = 168
/**
 * The height a 'surge' column hands to its 121cm LANDING PLATE. Twice the panel,
 * so twice the reach: 118cm of drop is comfortably inside it, which is the whole
 * reason surge can crest at 168 and still stand on the floor.
 */
const SURGE_HANDOVER_CM = 118

/**
 * SWELL — 8 rows. HALF WATER SWELL, HALF SNOW DRIFT: a mass that starts low at the
 * far jamb, accumulates against the wall beside column 0 and peaks there, under a
 * ceiling low enough to walk past.
 *
 * -----------------------------------------------------------------------------
 * THE HYBRID — what is taken from each
 * -----------------------------------------------------------------------------
 * A water swell is SMOOTH and near-symmetric; a snow drift is ASYMMETRIC — a long
 * gradual windward slope up to a crest, then a short steep lee slipface — and it
 * ACCUMULATES AGAINST OBSTACLES. This preset takes:
 *
 *   from the swell   continuous curvature. Every hinge except the shore bend and the
 *                    slipface break is small, so each column reads as one bending
 *                    surface rather than a folded zig-zag.
 *   from the drift   1. front-to-back ASYMMETRY: 301cm of panel (rows 0–4, the two
 *                       window rows, one more square and the 121cm spine) is spent on
 *                       the rise and only ~180cm (rows 5–7) on the drop, so the lee is
 *                       roughly 1.7× as steep as the windward slope in every column;
 *                    2. ACCUMULATION AGAINST THE WALL: the crest ramps
 *                       monotonically to its maximum at column 0, exactly where the
 *                       wall closes the −X edge (schema.js `WALL_X`) — a drift
 *                       banking against a fence.
 *
 * -----------------------------------------------------------------------------
 * PER COLUMN
 * -----------------------------------------------------------------------------
 *   row 0    the window panel, pitched `swellFrontDeg(c)` — 42° at the wall down to
 *            20° at the far jamb, rearing out of the window.
 *   row 1    the SHOULDER: `SWELL_SHORE_BEND_DEG[c]` (−13° … −22°) BACK from the
 *            front. The shore therefore BREAKS rather than inclining — the two rows
 *            at the window bend in opposite directions, at a different sharpness in
 *            every column, which is what makes the shoreline read as chop.
 *   row 2    the CLIMB row: the rise resumes off the shoulder, steeper than the plate
 *            behind it (`SWELL_CLIMB_LEAD`) so the windward DECELERATES into the crest.
 *   rows 3–4 the 121cm SPINE PLATE, its pitch solved so the chain tops out at that
 *            column's `SWELL_CREST_CM`. THE CREST IS AT THE FAR END OF THIS PLATE —
 *            row 4 of 8, one unit deeper than the old rows 2–3 spine put it.
 *   rows 5–6 the LEE, dropping from the crest to `SWELL_HANDOVER_FRACTION` of it. Two
 *            squares split `SWELL_LEE_SPLIT` / rest (a smooth S) in columns 2–5; ONE
 *            121cm SLIPFACE PLATE at a single pitch in the two doubled columns.
 *   row 7    the landing, solved by `land()` onto the floor.
 *
 * All six columns stand on the FLOOR — that is swell's signature and what separates
 * it from `wallcrash`, which engages the wall structurally (`endSupport: 'wall'`).
 * Swell only LEANS toward the wall; nothing here is bracketed to it.
 *
 * -----------------------------------------------------------------------------
 * WHY 8 ROWS — AND WHAT THE 8th COSTS
 * -----------------------------------------------------------------------------
 * The crest had to move ONE ROW DEEPER, because the room is read from the WINDOW side:
 * with the crest at row 3 the tall part stood close to the glass and hid the rows
 * behind it. Deeper crest ⇒ longer, gentler approach ⇒ more panels turning their lit
 * face toward the viewer. Measured from a window-side eye point, the count of panels
 * presenting their lit face went 23 of 36 (crest at row 3, 7 rows) to 28 of 40 — five
 * more panels facing the viewer, and 70% of the surface instead of 64%.
 *
 * 7 ROWS CANNOT DO IT, and the arithmetic is short enough to state. The crest sits at
 * the far end of the spine plate, so pushing it to row 4 leaves only rows 5–6 — 120cm
 * of panel — to bring column 0 down from ~117cm. That is a mean sin of 0.97: two
 * essentially vertical panels, or one 121cm plate at −72° hinged −82° off the spine.
 * The 6-row build was rejected two revisions ago for exactly that reason ("a CLIFF, not
 * a slipface"), and it is the same cliff here. It was BUILT AND SOLVED rather than
 * argued about: the 7-row / row-4-crest variant does not even land — column 0 comes
 * back E_END_FLOATING and the in-row gap report triples, because a plated lee beside a
 * squared one diverges hard.
 *
 * 8 rows give 180cm behind the crest again (60 · 60 · 60), so the same ~117cm comes
 * down as roughly −38° / −38° / −40° in column 0: sane panels, the steepest hinge
 * around −50°, and the drift's ASYMMETRY now comes out of the row split itself — 301cm
 * of rise against 180cm of fall, a 1.7 ratio where the 7-row build managed 1.3.
 *
 * THE PRICE IS FLOOR: 8 rows push the footprint from 417cm to ~478cm of depth, +61cm.
 * The cost is unavoidable rather than sloppy — the FAR JAMB column sets the depth, and
 * a column that peaks at 42cm over 481cm of chain is a nearly flat ramp whose depth is
 * its chain length; the only ways to shorten it are to raise the low end of the ramp
 * (which the "start much lower on the left" brief forbids) or to corrugate it (which
 * the swell half of the hybrid forbids). Stated plainly: 61cm of extra depth buys the
 * open window-side approach. Taken, with the number on the table.
 *
 * -----------------------------------------------------------------------------
 * THE RAMP
 * -----------------------------------------------------------------------------
 * Two gradients, both pointing the same way, are what make it read as accumulation
 * against the wall rather than as a travelling wave: the crest ramps 116 → 42cm and
 * the front pitch ramps 42° → 20°, monotonically, toward column 0. Because each
 * column's spine pitch is solved BACK from its crest target, the tall columns get the
 * steep plate and the short ones a nearly flat one — the amplitude profile is the
 * design input, not an accident of six hand-picked angles. The range is WIDER than
 * the old 150 → 80 version (2.8× end to end instead of 1.9×) under a 35cm lower
 * ceiling, which is what "start much lower on the left and don't build to such a high
 * point" asks for.
 *
 * A ramp this strong necessarily makes neighbouring strips DIVERGE, so the in-row
 * joint report flags joints a symmetric design would not. That is real information
 * about how much flex the printed connectors have to absorb. The 8-row build adds a
 * row of joints (74 in-row joints, 40 flagged, against the 7-row build's 65 and 35) but
 * NOT a worse worst case: 16.5cm of gap deviation against 16.9cm, because
 * `SWELL_CLIMB_LEAD` keeps neighbouring windward runs closer together than the old
 * profile did. The doubled columns are the one place the lee shape differs between
 * neighbours, which is why the doubling is kept to the two columns at the end of the
 * ramp rather than scattered through it.
 *
 * PLATE PATTERN — 'doubled spine plates': ONE vertical plate at the crest of every
 * column, rows 3–4, at (3,0) … (3,5) — DOUBLED in the two wall-most columns, which
 * carry a second plate right behind the first, at (5,0) and (5,1), as their slipface.
 * EIGHT plates from one clause: one spine per column, doubled on the two biggest
 * strips. It reads as a band across the crest with the two tallest columns running one
 * rigid 242cm run from crest to landing — the strips that lift the most mass are the
 * ones stiffened twice.
 *
 * Every plateau in the pitch profiles exists BECAUSE of a plate: a vertical plate at
 * (r, c) removes joint r, so rows r and r+1 must share one pitch. Hence
 * `spine`/`spine` at rows 3–4 everywhere and `slipface`/`slipface` at rows 5–6 in the
 * doubled columns — and hence the doubled columns' landing is solved at joint 6, which
 * tilts the whole 121cm slipface plate down (see `land()`).
 */
export function presetSwell() {
  const profile = (c) => {
    const front = swellFrontDeg(c)
    // The shore break: row 1 falls BACK from the front, by a different amount in
    // every column. Authored, not solved — see SWELL_SHORE_BEND_DEG.
    const shoulder = front + SWELL_SHORE_BEND_DEG[c]
    const y0 = frontRestY(front, 60)
    const afterShore = climbTo(y0, [[60, front], [60, shoulder]])
    // What is left of the climb after the shore, spread over the CLIMB ROW (60cm)
    // and the SPINE PLATE (121cm) behind it — 181cm of windward. The climb row takes
    // more than its share (SWELL_CLIMB_LEAD) so the rise decelerates into the crest.
    const remaining = SWELL_CREST_CM[c] - afterShore
    const climb = crestPitch(afterShore, afterShore + (60 * SWELL_CLIMB_LEAD * remaining) / 181, 60)
    const beforeSpine = climbTo(afterShore, [[60, climb]])
    // The spine plate absorbs whatever climb the shore detail and the climb row left,
    // which is why the shore detail is free — and it tops out at this column's
    // crest target, one row deeper than the old rows 2–3 spine did.
    const spine = crestPitch(beforeSpine, SWELL_CREST_CM[c], 121)
    const crest = climbTo(beforeSpine, [[121, spine]])
    // The lee, as a fraction of this column's own crest so every column's slipface
    // is the same shape scaled.
    const handover = crest * SWELL_HANDOVER_FRACTION
    if (SWELL_DOUBLED_COLS.includes(c)) {
      // The two tallest strips take their slipface on ONE rigid 121cm plate (rows
      // 5–6) instead of two squares: same reach, one joint instead of two, and it is
      // where this preset's two extra plates live.
      // rows: 0 front · 1 shoulder · 2 climb · 3–4 spine plate · 5–6 slipface plate · 7 landing
      const slipface = descentPitch(crest, handover, 121)
      return [front, shoulder, climb, spine, spine, slipface, slipface, 0]
    }
    // The uneven split makes the lee a smooth S rather than one hard break behind
    // the plate.
    const descent1 = descentPitch(crest, crest - SWELL_LEE_SPLIT * (crest - handover), 60)
    const descent2 = descentPitch(climbTo(crest, [[60, descent1]]), handover, 60)
    // rows: 0 front · 1 shoulder · 2 climb · 3–4 spine plate · 5–6 the lee · 7 landing
    return [front, shoulder, climb, spine, spine, descent1, descent2, 0]
  }

  return land(
    fromProfiles({
      name: 'swell',
      profiles: Array.from({ length: COLS }, (_, c) => profile(c)),
      rects: [
        ...Array.from({ length: COLS }, (_, c) => ({
          row: SWELL_SPINE_ROW,
          col: c,
          orientation: 'vertical',
        })),
        ...SWELL_DOUBLED_COLS.map((c) => ({
          row: SWELL_SLIPFACE_ROW,
          col: c,
          orientation: 'vertical',
        })),
      ],
      meta: {
        preset: 'swell',
        rectPattern: 'doubled spine plates',
        notes:
          'STORM family: halfway between a water swell and a snow drift, accumulating against ' +
          'the −X wall beside column 0. FROM THE SWELL — smooth continuous curvature, every ' +
          'hinge small apart from the two deliberate ones. FROM THE DRIFT — (a) each column is ' +
          'asymmetric front to back: 301cm of panel rises over rows 0–4 and only 180cm falls ' +
          'over rows 5–7, so the lee slipface is about 1.7× as steep as the windward slope; ' +
          '(b) it banks against an obstacle, the crest ramping monotonically to its maximum AT ' +
          'the wall column (116 → 42cm, and the fronts 42° → 20° with it), so the far jamb — ' +
          'the LEFT of the 3D view — starts low and the mass piles up on the right. The CREST ' +
          `SITS DEEP — at the far end of the spine plate, row ${SWELL_SPINE_ROW + 1} of ` +
          `${SWELL_ROWS} — because the surface is read from the WINDOW side: a deeper crest ` +
          'makes the approach a longer gradual rise and turns more panel faces toward the ' +
          'viewer instead of hiding them behind the tall part. The shore still BREAKS instead ' +
          'of inclining: row 1 falls 13–22° back from the pitched front, a different amount in ' +
          'every column, giving a visible crease at the window that reads as chop. All six ' +
          'columns still land on the floor — nothing is wall-supported. Plate pattern "doubled ' +
          `spine plates": one vertical 121cm plate at the crest (rows ${SWELL_SPINE_ROW}–` +
          `${SWELL_SPINE_ROW + 1} of ${SWELL_ROWS}) in every column, doubled in the two ` +
          'wall-most — the tallest — columns, which carry a second plate right behind it (rows ' +
          `${SWELL_SLIPFACE_ROW}–${SWELL_SLIPFACE_ROW + 1}) as their slipface. Eight plates.`,
      },
    }),
  )
}

/**
 * SURGE — 7 rows, the most violent design that still stands entirely on the floor.
 *
 * Fronts pitched 25 + 4c (25° … 45°) — steep enough that the water is already
 * rearing as it leaves the window — climbing hard to a CREST PLATE across rows 2–3
 * whose pitch is solved for `SURGE_CREST_CM` ≈ 168cm, then one descent row handing
 * ~118cm to a 121cm LANDING PLATE across rows 5–6 that `land()` tilts down onto the
 * floor. Peaks past 150cm in every column.
 *
 * The two plates are what make it possible. A 60cm landing square could only give
 * back ~60cm, which would cap the crest near 110; the 121cm plate gives back nearly
 * twice that, and the crest plate spends 121cm of panel on a single climb instead of
 * two 60cm steps with a hinge budget between them.
 *
 * PLATE PATTERN — 'double plates': TWO vertical plates in every column — a crest
 * plate at rows 2–3 and a landing plate at rows 5–6, (2,c) and (5,c) for all six c.
 * Twelve 121cm plates, half the surface, in two clean bands.
 */
export function presetSurge() {
  const CREST_ROW = 2
  const LANDING_ROW = 5 // = rows - 2
  const profile = (c) => {
    const front = 25 + 4 * c
    const climb = front + 18
    const y0 = frontRestY(front, 60)
    const beforeCrest = climbTo(y0, [[60, front], [60, climb]])
    const crestDeg = crestPitch(beforeCrest, SURGE_CREST_CM, 121)
    const crest = climbTo(beforeCrest, [[121, crestDeg]])
    const descent = descentPitch(crest, SURGE_HANDOVER_CM, 60)
    // rows: 0 front · 1 climb · 2–3 crest plate · 4 descent · 5–6 landing plate
    return [front, climb, crestDeg, crestDeg, descent, descent, descent]
  }

  return land(
    fromProfiles({
      name: 'surge',
      profiles: Array.from({ length: COLS }, (_, c) => profile(c)),
      rects: Array.from({ length: COLS }, (_, c) => [
        { row: CREST_ROW, col: c, orientation: 'vertical' },
        { row: LANDING_ROW, col: c, orientation: 'vertical' },
      ]).flat(),
      meta: {
        preset: 'surge',
        rectPattern: 'double plates',
        notes:
          'STORM family: steep pitched fronts (25° → 45°), crests past 150cm, every ' +
          'column still landing on the floor. Plate pattern "double plates": two vertical ' +
          '121cm plates per column — a crest plate at rows 2–3 and a landing plate at rows ' +
          '5–6 — twelve in two bands, and the reason the crest can be this high.',
      },
    }),
  )
}

/**
 * WALLCRASH — 6 rows. The wave hits the WALL, which is at the −X edge of the grid,
 * beside COLUMN 0 (schema.js `WALL_X` / `WALL_COLUMN`).
 *
 * Everything RAMPS toward that wall, so everything ramps DOWN in column index: the
 * front pitch (43° at the wall → 8° at the far jamb), the amplitude (a ~200cm
 * rear-up in column 0 → a 62cm ripple in column 5), and the plate positions.
 * Columns 1–5 are ordinary floor columns and land; COLUMN 0 IS `endSupport: 'wall'`
 * — side-bracketed to the plane x = `WALL_X` = 0 — so it is exempt from the
 * grounded-end rule and climbs monotonically to end HIGH against the wall, the water
 * splashing up it. `land()` skips it automatically (core/ground.js).
 *
 * The six profiles are authored by hand rather than from a shared budget, because
 * the plate diagonal gives every column a DIFFERENT segment structure — column 5's
 * front panel IS a 121cm plate, column 1's landing panel is one — and therefore a
 * different descent capacity to design against.
 *
 * PLATE PATTERN — 'wall splash': the plates MARCH diagonally back and toward the
 * wall — (0,5) (1,4) (2,3) (3,2) (4,1) (4,0) — one row deeper per column as they
 * approach it, until the final two rear up side by side against the wall itself.
 * Every plateau in the profiles below is there to hold one of them.
 */
export function presetWallcrash() {
  // Listed in column order; the row DEEPENS as the column index falls toward the
  // wall, and the last two share row 4 because there is nowhere deeper to go.
  const rects = [
    { row: 4, col: 0, orientation: 'vertical' },
    { row: 4, col: 1, orientation: 'vertical' },
    { row: 3, col: 2, orientation: 'vertical' },
    { row: 2, col: 3, orientation: 'vertical' },
    { row: 1, col: 4, orientation: 'vertical' },
    { row: 0, col: 5, orientation: 'vertical' },
  ]

  // Cumulative pitch per row (0..5). The plateau in each is the plate's two rows;
  // the last authored pitch is a first guess that `land()` refines (except column 0,
  // which is wall-supported and left exactly as authored).
  const profiles = [
    [43, 58, 52, 30, 12, 12], //  c0 · plate rows 4–5 · WALL: climbs to ~200cm and stays
    [36, 60, 30, -35, -45, -45], //  c1 · plate rows 4–5 (the landing plate) · ~122cm
    [29, 42, 45, -30, -30, -30], //  c2 · plate rows 3–4 · ~116cm
    [22, 30, 26, 26, -60, -60], //  c3 · plate rows 2–3 · ~109cm
    [15, 30, 30, 5, -30, -30], //  c4 · plate rows 1–2 · ~85cm
    [8, 8, 22, 18, -20, -20], //  c5 · plate rows 0–1 · a 62cm ripple, gentle
  ]

  return land(
    fromProfiles({
      name: 'wallcrash',
      profiles,
      endSupports: ['wall', 'floor', 'floor', 'floor', 'floor', 'floor'],
      rects,
      meta: {
        preset: 'wallcrash',
        rectPattern: 'wall splash',
        notes:
          'STORM family: front pitch and amplitude both ramp toward the −X wall beside ' +
          'column 0 (43° → 8°, ~200cm → 62cm). Columns 1–5 land on the floor; column 0 is ' +
          'wall-supported and deliberately ends high, splashing up the wall. Plate pattern ' +
          '"wall splash": the plates march diagonally back and toward the wall — (0,5) ' +
          '(1,4) (2,3) (3,2) (4,1) (4,0).',
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
  const plated = plateaus.includes(CALM_ROWS - 2)
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
  const cols = COLS

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
    case 'swell': return presetSwell()
    case 'surge': return presetSurge()
    case 'wallcrash': return presetWallcrash()
    default: throw new Error(`unknown preset id: ${id}`)
  }
}
