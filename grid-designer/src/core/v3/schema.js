/**
 * grid-designer v3 — configuration schema, normalization and validation.
 *
 * HEADLESS ZONE (src/core/): pure functions, importable from plain node.
 *   - explicit `.js` extensions on ALL relative imports
 *   - may import `three` math classes only; never components / store / DOM
 *   - same input → same output, no hidden state
 *
 * =============================================================================
 * WHAT THIS DESCRIBES  (schema v3 — "one surface, tiled by rigid panels")
 * =============================================================================
 * v2 modelled the installation as 6 independent 2D column fold-chains. v3 throws
 * that model out: the installation is now ONE 3D surface — a snow drift, `H(x,z)`
 * — tiled by rigid 60×60 squares and 60×121 plates, decided by a **tiling
 * algorithm** (`tiling.js`), not authored by hand. "Row" and "column" are RETIRED
 * words. What survives are three coordinate systems (V3_SPEC.md §1), and this
 * schema's job is to describe the config an LLM or a human uses to drive them:
 *
 *   material  (u, v) cm, or (i, j) in CELLS  — position on the unrolled flat
 *             sheet. `sheet.cols` × `sheet.rows` cells form an exact lattice:
 *             cell (i, j) owns u ∈ [i·pitch, i·pitch+size], v ∈ [j·pitch,
 *             j·pitch+size], pitch = cell.size + gap. `i` runs across the
 *             window (+X, wall side first), `j` runs window → back (+Z). THE
 *             CODE MUST CALL THESE `i`/`j` OR `u`/`v`, NEVER row/col — a v3
 *             material index must never be confused with a v2 grid position.
 *   plan      (x, z) cm — the floor. `config.form` authors `H(x, z)` here
 *             (src/core/v3/form.js consumes exactly these six numbers plus
 *             `footprint`; this file's only job re: `form` is to fill/clamp it
 *             the same way and hand it through unchanged).
 *   world     (x, y, z) cm, Y up — where panels actually end up, produced by
 *             the (not-yet-built) placement walk. This file never touches it.
 *
 * THIS FILE (schema.js) OWNS `sheet`, `cell`, `gap`, `tiling`, `placement`,
 * `gapTolerance`, `groundTolerance`, `meta` — the structural/numeric envelope.
 * `form` is delegated wholesale to `./form.js`'s `normalizeForm` /
 * `DEFAULT_FORM`, which is the single source of truth for those six knobs'
 * ranges (this file does NOT re-implement form's clamping logic, only its own
 * range checks for reporting — see "TWO KINDS OF DEFAULTING" below).
 *
 * WHAT DOWNSTREAM PACKAGES DO WITH EACH FIELD (so a cold reader can place this
 * file in the pipeline):
 *   - `tiling.js` (P2, next in this package) reads `sheet.{cols,rows}`,
 *     `cell.{size,plateLength}`, `gap`, `form`, `tiling.strategy` to solve a
 *     domino tiling of the material grid — which cells get a 2×2 square and
 *     which get a 2×4 plate.
 *   - `placement.js` (P3, not yet built) reads `placement.tree` to choose the
 *     spanning-tree strategy that walks tiles from material → world.
 *   - `report.js` (P4, not yet built) reads `gapTolerance` / `groundTolerance`
 *     to flag joints and grounding.
 *
 * =============================================================================
 * THE PLATE-LENGTH EXACTNESS (carried over from v2 — HANDOFF.md §2.2)
 * =============================================================================
 * `cell.plateLength` should equal `2·cell.size + gap`. At the defaults,
 * 60 + 1 + 60 = 121 EXACTLY: a 60×121 plate is a perfect drop-in for two 60×60
 * squares plus the 1cm joint between them, so swapping squares for a plate
 * moves nothing else on the sheet. This is load-bearing for the whole panel
 * kit staying modular. `validateConfig` warns (`W_PLATE_LENGTH`, non-blocking —
 * see v2's identical `W_RECT_LENGTH` reasoning) when the numbers don't add up;
 * it never silently corrects them.
 *
 * =============================================================================
 * VERSION — v1 AND v2 CONFIGS ARE REJECTED, NEVER MIGRATED
 * =============================================================================
 * `version` must be exactly 3. A v1 (row-accordion) or v2 (per-column fold
 * strip) config is REJECTED OUTRIGHT by `validateConfig`, exactly as v1 was
 * rejected by v2. This is not an oversight to "fix" by writing a migrator: the
 * three models describe physically different objects (independent column
 * chains vs. one tiled doubly-curved surface) and do not map onto one another.
 * A silent reinterpretation would produce a config that LOOKS like it loaded
 * successfully but describes something the author never intended — worse than
 * a loud rejection. See V3_SPEC.md §6 and HANDOFF.md §3.1.
 *
 * =============================================================================
 * TWO KINDS OF DEFAULTING — normalizeConfig vs. validateConfig
 * =============================================================================
 * Unlike v2's `normalizeConfig` (which fills MISSING fields but passes invalid
 * PRESENT values through untouched, so `validateConfig` can still see and
 * report them), v3's `normalizeConfig` fills defaults **and clamps every
 * numeric field into its V3_SPEC §6 range**. That is a deliberate difference:
 * v3's tiling/placement math (this package's `tiling.js`, and later
 * `placement.js`) must be able to call `normalizeConfig(anything)` and get
 * back numbers it can safely feed into a solver, the same defensive contract
 * `form.js`'s `normalizeForm` already gives its six knobs. `normalizeConfig`
 * is therefore NOT the function that decides whether a config is fit to show a
 * user — that is `validateConfig`'s job, and it does its own independent
 * inspection of the RAW input (via the private `withDefaults` below, which
 * mirrors v2's non-clamping fill-only behaviour) so an out-of-range value is
 * still reported even though `normalizeConfig` would have silently clamped it.
 * Concretely: `validateConfig({ ...good, gap: 999 })` still raises `E_RANGE`,
 * even though `normalizeConfig({ ...good, gap: 999 }).gap` comes back as 10.
 *
 * =============================================================================
 * JSON CONFIG SCHEMA (v3) — V3_SPEC.md §6, this is the shape an LLM generator
 * should emit
 * =============================================================================
 *
 *   {
 *     "version": 3,                            // must be exactly 3
 *     "units": "cm",                           // must be "cm"
 *     "name": "drift study 1",                 // optional, free text
 *     "sheet": { "cols": 6, "rows": 8 },        // MATERIAL cell counts (i, j).
 *                                              // cols 4..8, rows 5..10.
 *     "cell": { "size": 60, "plateLength": 121 }, // square cell 60cm; 2-cell
 *                                              // plate 121cm (see PLATE-LENGTH
 *                                              // EXACTNESS above)
 *     "gap": 1.0,                              // physical joint gap, 0..10cm
 *     "form": {                                // authored drift heightfield —
 *       "amplitude": 120, "crestX": 0.62, "crestZ": 0.55, "ridgeShear": 0.18,
 *       "toeSharpX": 0.8, "toeSharpZ": 0.9,     // see src/core/v3/form.js for
 *       "footprint": { "width": 365, "depth": 487 }  // drift's plan extent;
 *                                                   // OMIT IT to derive from
 *                                                   // the sheet (recommended)
 *     },
 *     "tiling": {
 *       "strategy": "flat-lie",                 // 'flat-lie' | 'ridge-aligned'
 *                                              // | 'toe-bands' — V3_SPEC §3.1
 *       "plateFitToleranceCm": 2.0               // max sagitta (cm) a plate's
 *                                              // 121cm span may bow away from
 *                                              // the target surface before
 *                                              // the tiler rejects it in
 *                                              // favour of two squares — see
 *                                              // tiling.js's "PLATE FIT".
 *                                              // 0.1..20; the top of the
 *                                              // range restores the old
 *                                              // maximal-packing behaviour.
 *     },
 *     "placement": { "mode": "surface-fit",     // 'surface-fit' | 'chain'
 *                    "tree": "bfs-corner" },    // 'bfs-corner' | 'comb-v'
 *                                              // | 'comb-u' — V3_SPEC §4.3
 *     "gapTolerance": 1.5,                     // report flags joints deviating
 *                                              // this far from `gap`
 *     "groundTolerance": 0.5,                  // how close a grounded tile's
 *                                              // solid must get to y = 0
 *     "meta": {                                // provenance, free-form
 *       "preset": "drift", "tilePattern": "flat-lie", "notes": ""
 *     }
 *   }
 *
 * =============================================================================
 * RANGES — V3_SPEC.md §6
 * =============================================================================
 *   sheet.cols        4 .. 8            (integer)
 *   sheet.rows        5 .. 10           (integer)
 *   gap               0 .. 10           cm
 *   form.amplitude    0 .. 250          cm         (mirrors form.js)
 *   form.crestX/Z     0.15 .. 0.85                 (mirrors form.js)
 *   form.ridgeShear   -0.4 .. 0.4                  (mirrors form.js)
 *   form.toeSharpX/Z  0.45 .. 1.0                  (mirrors form.js)
 *   tiling.plateFitToleranceCm  0.1 .. 20         cm  (v3-tiling-specific, P2b)
 * `cell.size` / `cell.plateLength` / `gapTolerance` / `groundTolerance` have no
 * declared band — only "must be a positive number", exactly v2's treatment of
 * the analogous fields.
 *
 * =============================================================================
 * VALIDATION CODES
 * =============================================================================
 * Errors (valid === false):
 *   E_SHAPE    structural problems: config not an object, wrong version (see
 *              VERSION above), wrong units, non-integer/out-of-declared-type
 *              sheet.cols/rows, non-positive cell.size/plateLength, an invalid
 *              tiling.strategy / placement.tree enum value, non-positive
 *              gapTolerance/groundTolerance, a non-finite gap or form knob
 *   E_RANGE    a finite but out-of-band value: sheet.cols/rows outside their
 *              integer band, gap outside 0..10, any form knob outside its
 *              V3_SPEC §6 range, or tiling.plateFitToleranceCm outside 0.1..20
 * Warnings (do not affect `valid`):
 *   W_PLATE_LENGTH   cell.plateLength ≠ 2·cell.size + gap — see PLATE-LENGTH
 *                    EXACTNESS above. Silent at the defaults.
 */

import { DEFAULT_FORM, normalizeForm } from './form.js'

// -----------------------------------------------------------------------------
// Ranges — V3_SPEC.md §6
// -----------------------------------------------------------------------------
export const SHEET_COLS_MIN = 4
export const SHEET_COLS_MAX = 8
export const SHEET_ROWS_MIN = 5
export const SHEET_ROWS_MAX = 10
export const GAP_MIN = 0
export const GAP_MAX = 10
/**
 * Mirrors of form.js's own (module-private) clamp constants, kept in sync by
 * hand against V3_SPEC.md §6 — this file needs them to raise `E_RANGE` on the
 * RAW input (see "TWO KINDS OF DEFAULTING" above); `normalizeForm` already
 * clamps silently for its own callers, which is a different job.
 */
export const AMPLITUDE_MIN = 0
export const AMPLITUDE_MAX = 250
export const CREST_MIN = 0.15
export const CREST_MAX = 0.85
export const RIDGE_SHEAR_MIN = -0.4
export const RIDGE_SHEAR_MAX = 0.4
export const TOE_SHARP_MIN = 0.45
export const TOE_SHARP_MAX = 1.0
/**
 * `tiling.plateFitToleranceCm` — see tiling.js's "PLATE FIT" section (added
 * P2b). Not mirrored from form.js; this range is v3-tiling-specific.
 */
export const PLATE_FIT_TOLERANCE_MIN = 0.1
export const PLATE_FIT_TOLERANCE_MAX = 20

export const TILING_STRATEGIES = ['flat-lie', 'ridge-aligned', 'toe-bands']
export const PLACEMENT_TREES = ['bfs-corner', 'comb-v', 'comb-u']

/**
 * How tiles are positioned.
 *   'surface-fit' (default) — every tile is fitted to the target surface
 *      independently, so the incompatibility is shared out across ALL joints.
 *      This is what the physical connectors actually do, and it is the only
 *      mode whose output still reads as the drift that was authored.
 *   'chain' — tiles are hinged off one another along a spanning tree, making
 *      every tree joint exactly `gap` with zero skew and dumping the entire
 *      error onto the cycle-closing edges. Keep it for comparison: it shows
 *      what insisting on exact joints costs, and a chain is what v2 did.
 */
export const PLACEMENT_MODES = ['surface-fit', 'chain']

// -----------------------------------------------------------------------------
// Defaults
// -----------------------------------------------------------------------------
export const DEFAULT_SHEET = { cols: 6, rows: 8 }
export const DEFAULT_CELL = { size: 60, plateLength: 121 }
/**
 * See "THE PLATE-LENGTH EXACTNESS" above: 2·60 + 1 = 121 = plateLength exactly.
 */
export const DEFAULT_GAP = 1.0
export const DEFAULT_GAP_TOLERANCE = 1.5
/**
 * How close a tile's solid must come to y = 0 to count as touching the floor.
 *
 * v2 used 0.5cm, which it could afford: a 1D column chain was solved so its end
 * panel landed EXACTLY. v3 cannot. A rigid 60cm panel laid on a doubly-curved
 * drift deviates from it by roughly (30**2 / 2) x curvature no matter where you
 * put it, so the tiles along the grounded edges come to within a few cm of the
 * floor and no closer. That residual is real, is what shims and connectors take
 * up, and is reported per-edge rather than hidden — see `support.edges` in
 * placement.js. 2cm is the honest threshold for a tiled shell.
 */
export const DEFAULT_GROUND_TOLERANCE = 2.0
/**
 * `plateFitToleranceCm` — see tiling.js's "PLATE FIT" section: the maximum
 * sagitta (bow of the target surface away from a rigid plate's straight
 * chord, measured along the plate's 121cm long axis) a candidate plate may
 * have before the tiler refuses to place it and falls through to two squares
 * instead. Default 2.0cm is a middle ground. Turning it DOWN toward
 * PLATE_FIT_TOLERANCE_MIN (0.1cm) makes the tiler pickier — fewer plates,
 * concentrated on the flattest parts of the form. Turning it UP toward
 * PLATE_FIT_TOLERANCE_MAX (20cm) relaxes the check until it stops rejecting
 * anything a legal domino could occupy — that is exactly this package's
 * pre-P2b behaviour (maximal domino packing, plate count decoupled from the
 * form), kept reachable on purpose as the permissive end of the knob rather
 * than removed.
 */
export const DEFAULT_TILING = { strategy: 'flat-lie', plateFitToleranceCm: 2.0 }
export const DEFAULT_PLACEMENT = { tree: 'bfs-corner', mode: 'surface-fit' }

export const DEFAULT_CONFIG = Object.freeze({
  version: 3,
  units: 'cm',
  name: 'drift study 1',
  sheet: { cols: DEFAULT_SHEET.cols, rows: DEFAULT_SHEET.rows },
  cell: { size: DEFAULT_CELL.size, plateLength: DEFAULT_CELL.plateLength },
  gap: DEFAULT_GAP,
  form: {
    amplitude: DEFAULT_FORM.amplitude,
    crestX: DEFAULT_FORM.crestX,
    crestZ: DEFAULT_FORM.crestZ,
    ridgeShear: DEFAULT_FORM.ridgeShear,
    toeSharpX: DEFAULT_FORM.toeSharpX,
    toeSharpZ: DEFAULT_FORM.toeSharpZ,
    angularity: DEFAULT_FORM.angularity,
    facetCells: DEFAULT_FORM.facetCells,
    // Matches the default 6x8 sheet exactly (6*61-1 = 365, 8*61-1 = 487); an
    // omitted footprint derives the same way for any sheet — see withDefaults.
    footprint: { width: 365, depth: 487 },
  },
  tiling: { strategy: DEFAULT_TILING.strategy, plateFitToleranceCm: DEFAULT_TILING.plateFitToleranceCm },
  placement: { tree: DEFAULT_PLACEMENT.tree, mode: DEFAULT_PLACEMENT.mode },
  gapTolerance: DEFAULT_GAP_TOLERANCE,
  groundTolerance: DEFAULT_GROUND_TOLERANCE,
  meta: { preset: 'drift', tilePattern: 'flat-lie', notes: '' },
})

// -----------------------------------------------------------------------------
// Small helpers
// -----------------------------------------------------------------------------
const isPlainObject = (v) => typeof v === 'object' && v !== null && !Array.isArray(v)
const isFiniteNumber = (v) => typeof v === 'number' && Number.isFinite(v)
const isInt = (v) => isFiniteNumber(v) && Number.isInteger(v)

export function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v
}

function numberOr(v, fallback) {
  return isFiniteNumber(v) ? v : fallback
}

function positiveOr(v, fallback) {
  return isFiniteNumber(v) && v > 0 ? v : fallback
}

function clampInt(v, fallback, lo, hi) {
  const n = isFiniteNumber(v) ? Math.round(v) : fallback
  return clamp(n, lo, hi)
}

// -----------------------------------------------------------------------------
// withDefaults — PRIVATE. Fills MISSING pieces only; a present-but-invalid
// value is passed through verbatim. This is v2-normalizeConfig-shaped on
// purpose: `validateConfig` uses it so an explicitly-bad value is still there
// to report, while entirely-absent structure still gets a safe default so a
// minimal hand-written config can validate. Never exported — the public
// `normalizeConfig` (below) is the clamping one; this is its non-clamping
// cousin, kept private so nobody downstream is tempted to treat it as safe to
// feed a solver.
// -----------------------------------------------------------------------------
function withDefaults(raw) {
  const src = isPlainObject(raw) ? raw : {}
  const sheetSrc = isPlainObject(src.sheet) ? src.sheet : {}
  const cellSrc = isPlainObject(src.cell) ? src.cell : {}
  const formSrc = isPlainObject(src.form) ? src.form : {}
  const footprintSrc = isPlainObject(formSrc.footprint) ? formSrc.footprint : {}
  const tilingSrc = isPlainObject(src.tiling) ? src.tiling : {}
  const placementSrc = isPlainObject(src.placement) ? src.placement : {}

  // An OMITTED footprint is derived from the sheet, not taken from a constant.
  //
  // The drift is zero outside its footprint, so a footprint SHORTER than the
  // sheet leaves the trailing cells on dead-flat ground with a slope
  // discontinuity at the boundary — and the tiles straddling it cannot follow
  // the kink. Measured: a 430cm-deep drift under a 487cm sheet drove the worst
  // joint deviation to 9.8cm, where a matched footprint gives 2.67cm on the
  // same form. Deriving the default keeps it self-consistent at every sheet
  // size. An explicit footprint is still honoured — a drift that deliberately
  // feathers out before the sheet ends is a legitimate design.
  const dSize = cellSrc.size !== undefined ? cellSrc.size : DEFAULT_CELL.size
  const dGap = src.gap !== undefined ? src.gap : DEFAULT_GAP
  const dCols = sheetSrc.cols !== undefined ? sheetSrc.cols : DEFAULT_SHEET.cols
  const dRows = sheetSrc.rows !== undefined ? sheetSrc.rows : DEFAULT_SHEET.rows
  const dPitch = (Number(dSize) || DEFAULT_CELL.size) + (Number(dGap) || 0)
  const derivedFootprint = {
    width: (Number(dCols) || DEFAULT_SHEET.cols) * dPitch - (Number(dGap) || 0),
    depth: (Number(dRows) || DEFAULT_SHEET.rows) * dPitch - (Number(dGap) || 0),
  }

  const out = {
    version: src.version !== undefined ? src.version : 3,
    units: src.units !== undefined ? src.units : 'cm',
    sheet: {
      cols: sheetSrc.cols !== undefined ? sheetSrc.cols : DEFAULT_SHEET.cols,
      rows: sheetSrc.rows !== undefined ? sheetSrc.rows : DEFAULT_SHEET.rows,
    },
    cell: {
      size: cellSrc.size !== undefined ? cellSrc.size : DEFAULT_CELL.size,
      plateLength: cellSrc.plateLength !== undefined ? cellSrc.plateLength : DEFAULT_CELL.plateLength,
    },
    gap: src.gap !== undefined ? src.gap : DEFAULT_GAP,
    form: {
      amplitude: formSrc.amplitude !== undefined ? formSrc.amplitude : DEFAULT_FORM.amplitude,
      crestX: formSrc.crestX !== undefined ? formSrc.crestX : DEFAULT_FORM.crestX,
      crestZ: formSrc.crestZ !== undefined ? formSrc.crestZ : DEFAULT_FORM.crestZ,
      ridgeShear: formSrc.ridgeShear !== undefined ? formSrc.ridgeShear : DEFAULT_FORM.ridgeShear,
      toeSharpX: formSrc.toeSharpX !== undefined ? formSrc.toeSharpX : DEFAULT_FORM.toeSharpX,
      toeSharpZ: formSrc.toeSharpZ !== undefined ? formSrc.toeSharpZ : DEFAULT_FORM.toeSharpZ,
      // Faceting: quantizes the smooth drift onto the panel lattice so the
      // panels can BE the surface rather than approximate it. Consumed by
      // target.js — see form.js for what the two knobs mean.
      angularity: formSrc.angularity !== undefined ? formSrc.angularity : DEFAULT_FORM.angularity,
      facetCells: formSrc.facetCells !== undefined ? formSrc.facetCells : DEFAULT_FORM.facetCells,
      footprint: {
        width: footprintSrc.width !== undefined ? footprintSrc.width : derivedFootprint.width,
        depth: footprintSrc.depth !== undefined ? footprintSrc.depth : derivedFootprint.depth,
      },
    },
    tiling: {
      strategy: tilingSrc.strategy !== undefined ? tilingSrc.strategy : DEFAULT_TILING.strategy,
      plateFitToleranceCm:
        tilingSrc.plateFitToleranceCm !== undefined ? tilingSrc.plateFitToleranceCm : DEFAULT_TILING.plateFitToleranceCm,
    },
    placement: {
      tree: placementSrc.tree !== undefined ? placementSrc.tree : DEFAULT_PLACEMENT.tree,
      mode: placementSrc.mode !== undefined ? placementSrc.mode : DEFAULT_PLACEMENT.mode,
    },
    gapTolerance: src.gapTolerance !== undefined ? src.gapTolerance : DEFAULT_GAP_TOLERANCE,
    groundTolerance: src.groundTolerance !== undefined ? src.groundTolerance : DEFAULT_GROUND_TOLERANCE,
    meta: { notes: '', ...(isPlainObject(src.meta) ? src.meta : {}) },
  }
  if (src.name !== undefined) out.name = src.name
  return out
}

// -----------------------------------------------------------------------------
// normalizeConfig — PUBLIC. Fill defaults AND clamp every numeric field into
// its V3_SPEC §6 range. See "TWO KINDS OF DEFAULTING" above for why this
// differs from v2's normalizeConfig and from this file's own `withDefaults`.
// Idempotent and deterministic: normalizeConfig(normalizeConfig(x)) deep-equals
// normalizeConfig(x) for any x, because every field is independently
// defaulted/clamped with no cross-field state. Never mutates `raw`.
// -----------------------------------------------------------------------------
export function normalizeConfig(raw) {
  const cfg = withDefaults(raw)

  const out = {
    version: cfg.version !== undefined ? cfg.version : 3,
    units: cfg.units !== undefined ? cfg.units : 'cm',
    sheet: {
      cols: clampInt(cfg.sheet.cols, DEFAULT_SHEET.cols, SHEET_COLS_MIN, SHEET_COLS_MAX),
      rows: clampInt(cfg.sheet.rows, DEFAULT_SHEET.rows, SHEET_ROWS_MIN, SHEET_ROWS_MAX),
    },
    cell: {
      size: positiveOr(cfg.cell.size, DEFAULT_CELL.size),
      plateLength: positiveOr(cfg.cell.plateLength, DEFAULT_CELL.plateLength),
    },
    gap: clamp(numberOr(cfg.gap, DEFAULT_GAP), GAP_MIN, GAP_MAX),
    // Delegate entirely to form.js — the single source of truth for these six
    // knobs' fill+clamp behaviour (P1, already verified). `cfg.form` is always
    // a plain object here (withDefaults guarantees it), so this just clamps.
    form: normalizeForm(cfg.form),
    tiling: {
      strategy: TILING_STRATEGIES.includes(cfg.tiling.strategy) ? cfg.tiling.strategy : DEFAULT_TILING.strategy,
      plateFitToleranceCm: clamp(
        numberOr(cfg.tiling.plateFitToleranceCm, DEFAULT_TILING.plateFitToleranceCm),
        PLATE_FIT_TOLERANCE_MIN,
        PLATE_FIT_TOLERANCE_MAX,
      ),
    },
    placement: {
      tree: PLACEMENT_TREES.includes(cfg.placement.tree) ? cfg.placement.tree : DEFAULT_PLACEMENT.tree,
      mode: PLACEMENT_MODES.includes(cfg.placement.mode) ? cfg.placement.mode : DEFAULT_PLACEMENT.mode,
    },
    gapTolerance: positiveOr(cfg.gapTolerance, DEFAULT_GAP_TOLERANCE),
    groundTolerance: positiveOr(cfg.groundTolerance, DEFAULT_GROUND_TOLERANCE),
    meta: { ...cfg.meta },
  }
  if (cfg.name !== undefined) out.name = cfg.name
  return out
}

// -----------------------------------------------------------------------------
// validateConfig
// -----------------------------------------------------------------------------
/**
 * Validate a config. Safe to call on raw (un-normalized) input — inspects the
 * RAW value of every field (via the private `withDefaults`, which only fills
 * in entirely-missing structure) so an explicitly out-of-range value is always
 * reported, even though `normalizeConfig` would have silently clamped it.
 *
 * @param {object} config
 * @returns {{ valid: boolean,
 *             errors: Array<{code:string,message:string,path:string}>,
 *             warnings: Array<{code:string,message:string,path:string}> }}
 */
export function validateConfig(config) {
  const errors = []
  const warnings = []
  const err = (code, message, path) => errors.push({ code, message, path })
  const warn = (code, message, path) => warnings.push({ code, message, path })

  if (!isPlainObject(config)) {
    err('E_SHAPE', 'config must be an object', '')
    return finish(errors, warnings)
  }

  const cfg = withDefaults(config)

  // --- version — v1/v2 are REJECTED, never migrated (see file header) -------
  if (cfg.version !== 3) {
    err(
      'E_SHAPE',
      `version must be exactly 3 (got ${JSON.stringify(cfg.version)}) — v1 (row-accordion) and v2 ` +
        `(per-column fold strips) configs are not supported: those models do not map onto v3's one ` +
        `tiled drift surface, so a v1/v2 config is REJECTED here, never silently reinterpreted`,
      'version',
    )
  }
  if (cfg.units !== 'cm') {
    err('E_SHAPE', `units must be "cm" (got ${JSON.stringify(cfg.units)})`, 'units')
  }

  // --- sheet: MATERIAL cell counts (i, j) — cols 4..8, rows 5..10 -----------
  if (!isFiniteNumber(cfg.sheet.cols) || !Number.isInteger(cfg.sheet.cols)) {
    err('E_SHAPE', `sheet.cols must be an integer (got ${JSON.stringify(cfg.sheet.cols)})`, 'sheet.cols')
  } else if (cfg.sheet.cols < SHEET_COLS_MIN || cfg.sheet.cols > SHEET_COLS_MAX) {
    err(
      'E_RANGE',
      `sheet.cols must be in ${SHEET_COLS_MIN}..${SHEET_COLS_MAX} (got ${cfg.sheet.cols})`,
      'sheet.cols',
    )
  }
  if (!isFiniteNumber(cfg.sheet.rows) || !Number.isInteger(cfg.sheet.rows)) {
    err('E_SHAPE', `sheet.rows must be an integer (got ${JSON.stringify(cfg.sheet.rows)})`, 'sheet.rows')
  } else if (cfg.sheet.rows < SHEET_ROWS_MIN || cfg.sheet.rows > SHEET_ROWS_MAX) {
    err(
      'E_RANGE',
      `sheet.rows must be in ${SHEET_ROWS_MIN}..${SHEET_ROWS_MAX} (got ${cfg.sheet.rows})`,
      'sheet.rows',
    )
  }

  // --- cell: no declared band, just "positive number" (v2's treatment) -----
  if (!isFiniteNumber(cfg.cell.size) || cfg.cell.size <= 0) {
    err('E_SHAPE', `cell.size must be a positive number (got ${JSON.stringify(cfg.cell.size)})`, 'cell.size')
  }
  if (!isFiniteNumber(cfg.cell.plateLength) || cfg.cell.plateLength <= 0) {
    err(
      'E_SHAPE',
      `cell.plateLength must be a positive number (got ${JSON.stringify(cfg.cell.plateLength)})`,
      'cell.plateLength',
    )
  }

  // --- gap: 0..10cm -----------------------------------------------------
  if (!isFiniteNumber(cfg.gap)) {
    err('E_SHAPE', `gap must be a number (got ${JSON.stringify(cfg.gap)})`, 'gap')
  } else if (cfg.gap < GAP_MIN || cfg.gap > GAP_MAX) {
    err('E_RANGE', `gap must be in ${GAP_MIN}..${GAP_MAX} cm (got ${cfg.gap})`, 'gap')
  }

  // --- form: mirrors form.js's own ranges, checked against the RAW input --
  const form = cfg.form
  const checkFormRange = (key, lo, hi, label) => {
    const v = form[key]
    if (!isFiniteNumber(v)) {
      err('E_SHAPE', `form.${key} must be a number (got ${JSON.stringify(v)})`, `form.${key}`)
    } else if (v < lo || v > hi) {
      err('E_RANGE', `form.${key} (${label}) must be in ${lo}..${hi} (got ${v})`, `form.${key}`)
    }
  }
  checkFormRange('amplitude', AMPLITUDE_MIN, AMPLITUDE_MAX, 'drift amplitude, cm')
  checkFormRange('crestX', CREST_MIN, CREST_MAX, 'crest position along the window axis')
  checkFormRange('crestZ', CREST_MIN, CREST_MAX, 'crest position along the wall→back axis')
  checkFormRange('ridgeShear', RIDGE_SHEAR_MIN, RIDGE_SHEAR_MAX, 'ridge diagonal walk rate')
  checkFormRange('toeSharpX', TOE_SHARP_MIN, TOE_SHARP_MAX, 'window-edge toe sharpness')
  checkFormRange('toeSharpZ', TOE_SHARP_MIN, TOE_SHARP_MAX, 'wall-edge toe sharpness')
  if (!isFiniteNumber(form.footprint.width) || form.footprint.width <= 0) {
    err(
      'E_SHAPE',
      `form.footprint.width must be a positive number (got ${JSON.stringify(form.footprint.width)})`,
      'form.footprint.width',
    )
  }
  if (!isFiniteNumber(form.footprint.depth) || form.footprint.depth <= 0) {
    err(
      'E_SHAPE',
      `form.footprint.depth must be a positive number (got ${JSON.stringify(form.footprint.depth)})`,
      'form.footprint.depth',
    )
  }

  // --- tiling.strategy / placement.tree enums --------------------------
  if (!TILING_STRATEGIES.includes(cfg.tiling.strategy)) {
    err(
      'E_SHAPE',
      `tiling.strategy must be one of ${TILING_STRATEGIES.map((s) => `"${s}"`).join(', ')} ` +
        `(got ${JSON.stringify(cfg.tiling.strategy)})`,
      'tiling.strategy',
    )
  }
  if (!PLACEMENT_TREES.includes(cfg.placement.tree)) {
    err(
      'E_SHAPE',
      `placement.tree must be one of ${PLACEMENT_TREES.map((s) => `"${s}"`).join(', ')} ` +
        `(got ${JSON.stringify(cfg.placement.tree)})`,
      'placement.tree',
    )
  }
  if (!PLACEMENT_MODES.includes(cfg.placement.mode)) {
    err(
      'E_SHAPE',
      `placement.mode must be one of ${PLACEMENT_MODES.map((s) => `"${s}"`).join(', ')} ` +
        `(got ${JSON.stringify(cfg.placement.mode)})`,
      'placement.mode',
    )
  }

  // --- tiling.plateFitToleranceCm: 0.1..20cm — see tiling.js's "PLATE FIT" -
  if (!isFiniteNumber(cfg.tiling.plateFitToleranceCm)) {
    err(
      'E_SHAPE',
      `tiling.plateFitToleranceCm must be a number (got ${JSON.stringify(cfg.tiling.plateFitToleranceCm)})`,
      'tiling.plateFitToleranceCm',
    )
  } else if (
    cfg.tiling.plateFitToleranceCm < PLATE_FIT_TOLERANCE_MIN ||
    cfg.tiling.plateFitToleranceCm > PLATE_FIT_TOLERANCE_MAX
  ) {
    err(
      'E_RANGE',
      `tiling.plateFitToleranceCm must be in ${PLATE_FIT_TOLERANCE_MIN}..${PLATE_FIT_TOLERANCE_MAX} cm ` +
        `(got ${cfg.tiling.plateFitToleranceCm})`,
      'tiling.plateFitToleranceCm',
    )
  }

  // --- tolerances: no declared band, just "positive number" ------------
  if (!isFiniteNumber(cfg.gapTolerance) || cfg.gapTolerance <= 0) {
    err(
      'E_SHAPE',
      `gapTolerance must be a positive number (got ${JSON.stringify(cfg.gapTolerance)})`,
      'gapTolerance',
    )
  }
  if (!isFiniteNumber(cfg.groundTolerance) || cfg.groundTolerance <= 0) {
    err(
      'E_SHAPE',
      `groundTolerance must be a positive number — how close a grounded tile's solid must get to ` +
        `y = 0 to count as touching the floor (got ${JSON.stringify(cfg.groundTolerance)})`,
      'groundTolerance',
    )
  }

  // --- W_PLATE_LENGTH — surfaced, never corrected (see file header) ----
  // Silent at the defaults: 2·60 + 1 = 121 = plateLength exactly.
  if (isFiniteNumber(cfg.cell.size) && isFiniteNumber(cfg.gap) && isFiniteNumber(cfg.cell.plateLength)) {
    const ideal = 2 * cfg.cell.size + cfg.gap
    if (cfg.cell.plateLength !== ideal) {
      warn(
        'W_PLATE_LENGTH',
        `cell.plateLength ${cfg.cell.plateLength}cm ≠ 2·cell.size + gap (${ideal}cm) — ` +
          `${(ideal - cfg.cell.plateLength).toFixed(2)}cm of slack per plate is accepted and surfaced, ` +
          `not corrected`,
        'cell.plateLength',
      )
    }
  }

  return finish(errors, warnings)
}

function finish(errors, warnings) {
  return { valid: errors.length === 0, errors, warnings }
}
