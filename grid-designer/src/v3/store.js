/**
 * grid-designer v3 — application state (zustand), the drift-surface store.
 *
 * =============================================================================
 * SAME CONTRACT AS v2's src/store.js — read that file's header first
 * =============================================================================
 * This is a NEW store for the v3 pivot (V3_SPEC.md), living beside the v2 one
 * rather than replacing it (see the P5 work order: v2's `src/store.js` and
 * every `src/components/*.jsx` are UNTOUCHED). It keeps v2's contract exactly:
 *
 *   - the store owns exactly one piece of DESIGN truth, `config` (the v3
 *     schema — src/core/v3/schema.js);
 *   - every mutation goes through `commit()`, which runs `validateConfig` and
 *     commits ONLY if `valid` (v3's validateConfig returns `{valid, errors,
 *     warnings}` — note the field is `valid`, not v2's `ok`); otherwise the
 *     previous config is kept and the errors land in `lastErrors`;
 *   - derived data (`layout` from `solveLayout`, `report` from `buildReport`)
 *     is memoized in a WeakMap keyed on config OBJECT IDENTITY via
 *     `getDerived(config)`, so solve+report run once per committed change,
 *     never per frame — `solveLayout` walks a spanning tree and measures every
 *     joint/collision, which is not cheap;
 *   - UI-only state (hover, selection, colour mode, ghost/bounds toggles) lives
 *     OUTSIDE `config`, never goes through `commit()`, and never reaches the
 *     exporters;
 *   - undo/redo is a capped stack of previously committed configs, restored
 *     WITHOUT re-validating (they were valid when stored);
 *   - the working config is seeded from `loadWorkingConfig()` when it validates
 *     (a stale v2 save is discarded by persistence.js's bumped
 *     `EXPECTED_CONFIG_VERSION = 3` before it ever reaches `validateConfig`
 *     here), and autosaved after every successful commit / undo / redo.
 *
 * WHY A SEPARATE STORE MODULE RATHER THAN REUSING v2's: v2's store imports
 * v2-only core modules (core/schema.js, core/merge.js, core/ground.js,
 * core/presets.js) that do not exist in a form v3 can use — the two schemas
 * describe physically different objects (V3_SPEC.md §6). A parallel module is
 * what "build the v3 UI alongside v2" means at the store layer.
 *
 * =============================================================================
 * ACTIONS
 * =============================================================================
 * Every setter CLAMPS its input to the range schema.js declares (or, for
 * `form.angularity` / `form.facetCells`, the range documented in form.js — see
 * the note by `ANGULARITY_RANGE` below) before committing, so a slider/stepper
 * bound to that same range can never produce a rejected commit. This mirrors
 * v2's `setColumnFold` / `setRows`, which clamp for the same reason. Enum
 * setters (`tiling.strategy`, `placement.tree`, `placement.mode`) fall back to
 * the current value on an unrecognised string rather than committing garbage.
 *
 * `combineCells` / `splitTileAt` / `clearOverrides` (P8) manage
 * `config.tiling.overrides` — manual square/plate pins, reproducing v2's
 * pair-click merge interaction (GridMap.jsx / core/merge.js) on top of v3's
 * simpler model. They go through `commit()` like everything else, and also
 * set the UI-only `lastActionNotice` (a plain-language line describing what
 * just happened) and `armedCell` (the plan view's held-square state).
 */

import { create } from 'zustand'
import { produce } from 'immer'
import {
  normalizeConfig,
  validateConfig,
  clamp,
  DEFAULT_CONFIG,
  SHEET_COLS_MIN,
  SHEET_COLS_MAX,
  SHEET_ROWS_MIN,
  SHEET_ROWS_MAX,
  GAP_MIN,
  GAP_MAX,
  AMPLITUDE_MIN,
  AMPLITUDE_MAX,
  CREST_MIN,
  CREST_MAX,
  RIDGE_SHEAR_MIN,
  RIDGE_SHEAR_MAX,
  TOE_SHARP_MIN,
  TOE_SHARP_MAX,
  PLATE_FIT_TOLERANCE_MIN,
  PLATE_FIT_TOLERANCE_MAX,
  TILING_STRATEGIES,
  PLACEMENT_TREES,
  PLACEMENT_MODES,
} from '../core/v3/schema.js'
import { buildPreset } from '../core/v3/presets.js'
import { solveLayout } from '../core/v3/placement.js'
import { buildReport } from '../core/v3/report.js'
import { OVERRIDE_MISFIT_CODE } from '../core/v3/tiling.js'
import { loadWorkingConfig, saveWorkingConfig } from '../persistence.js'

/**
 * `form.angularity` (0..1) and `form.facetCells` (1..4) — see src/core/v3/form.js's
 * `ANGULARITY_MIN/MAX` and `FACET_CELLS_MIN/MAX`. THESE ARE NOT RE-EXPORTED BY
 * schema.js (unlike every other form knob's range), and schema.js's own
 * `validateConfig` never range-checks them either — a genuine gap in the
 * frozen v3 core relative to its own "TWO KINDS OF DEFAULTING" contract
 * (schema.js's header says an out-of-range value should still raise `E_RANGE`
 * even though `normalizeConfig` clamps it; these two fields silently clamp
 * with no raw-value check at all). Reported rather than silently worked
 * around: src/core/v3/ is frozen for this package, so the values below are
 * hand-mirrored from form.js's private constants instead of imported.
 */
const ANGULARITY_MIN = 0
const ANGULARITY_MAX = 1
const FACET_CELLS_MIN = 1
const FACET_CELLS_MAX = 4

// -----------------------------------------------------------------------------
// Derived-data cache — see v2 store.js's identical pattern.
// -----------------------------------------------------------------------------
const derivedCache = new WeakMap()

/**
 * Solve + report a config, memoized on the config OBJECT IDENTITY.
 *
 * @param {object} config a validated, normalized v3 config
 * @returns {{ layout: object, report: object }} reference-stable per config
 */
export function getDerived(config) {
  let entry = derivedCache.get(config)
  if (!entry) {
    const layout = solveLayout(config)
    const report = buildReport(config, layout)
    entry = { layout, report }
    derivedCache.set(config, entry)
  }
  return entry
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/** Coerce a UI value to a finite number, or `fallback` when blank/unparseable. */
function numOr(v, fallback) {
  if (v === null || v === undefined || v === '') return fallback
  const n = Number(v)
  return Number.isFinite(n) ? n : fallback
}

/**
 * The material cells a `tiling.overrides` entry covers, as `[i, j]` pairs —
 * mirrors tiling.js's own candidate-to-cells logic exactly (see that file's
 * `makePlateTile`/`makeSquareTile`). Used by `combineCells`/`splitTileAt` to
 * find and drop any override that already touches a cell they are about to
 * repin.
 */
function overrideCells(ov) {
  if (ov.type === '2x4') {
    return ov.axis === 'u' ? [[ov.i, ov.j], [ov.i + 1, ov.j]] : [[ov.i, ov.j], [ov.i, ov.j + 1]]
  }
  return [[ov.i, ov.j]]
}

/** How many previous configs the undo stack keeps (mirrors v2's HISTORY_LIMIT). */
export const HISTORY_LIMIT = 50

// -----------------------------------------------------------------------------
// Store
// -----------------------------------------------------------------------------
const useStoreV3 = create((set, get) => {
  /**
   * Apply an immer recipe to `config` and commit only if the result validates.
   * @param {(draft: object) => void} recipe
   * @returns {boolean} whether the change was committed
   */
  function commit(recipe) {
    const next = produce(get().config, recipe)
    if (next === get().config) return true // no-op recipe
    return commitConfig(next)
  }

  /**
   * Validate and commit a whole config object. On success the OUTGOING config
   * is pushed onto the undo stack (redo stack cleared) and autosaved
   * (debounced — see persistence.js). Also lets go of any ARMED cell (P8):
   * any successful config change can reshape the tiling, so a cell armed
   * against the previous layout is no longer safe to assume valid.
   */
  function commitConfig(candidate) {
    const result = validateConfig(candidate)
    if (!result.valid) {
      set({ lastErrors: result.errors, lastWarnings: result.warnings })
      return false
    }
    const state = get()
    const past = [...state.past, { config: state.config, warnings: state.lastWarnings }]
    while (past.length > HISTORY_LIMIT) past.shift()
    set({
      config: candidate,
      lastErrors: [],
      lastWarnings: result.warnings,
      past,
      future: [],
      canUndo: past.length > 0,
      canRedo: false,
      armedCell: null,
    })
    saveWorkingConfig(candidate)
    return true
  }

  /**
   * Seed from the autosaved working config when there is one AND it validates
   * (persistence.js's `EXPECTED_CONFIG_VERSION = 3` already discards a stale
   * v2 save before this ever sees it); otherwise fall back to
   * `DEFAULT_CONFIG`. Never throws.
   */
  function loadInitialConfig() {
    const fallback = () => {
      const config = normalizeConfig(DEFAULT_CONFIG)
      return { config, validation: validateConfig(config) }
    }
    let saved
    try {
      saved = loadWorkingConfig()
    } catch {
      saved = null
    }
    if (!saved) return fallback()
    try {
      const config = normalizeConfig(saved)
      const validation = validateConfig(config)
      if (validation.valid) return { config, validation }
    } catch {
      // fall through to the default below
    }
    return fallback()
  }

  const { config: initialConfig, validation: initialValidation } = loadInitialConfig()

  return {
    // --- state --------------------------------------------------------------
    config: initialConfig,
    /** Errors from the most recent REJECTED action ([] after a success). */
    lastErrors: initialValidation.errors,
    /** Warnings from the most recent accepted config (informational). */
    lastWarnings: initialValidation.warnings,
    /** Undo stack: `{ config, warnings }` entries, oldest first, capped. */
    past: [],
    /** Redo stack: entries popped by `undo()`, most recently undone first. */
    future: [],
    canUndo: false,
    canRedo: false,

    // --- UI-only state (never part of `config`, never reaches exporters) ----
    /** Draw the overall measuring box in the 3D view? */
    showBounds: true,
    /** Draw the translucent ghost of the target drift surface? */
    showGhost: true,
    /** Colour mode for the 3D viewport: 'type' | 'gap' | 'clearance' | 'facet'. */
    colorMode: 'type',
    /** Tile id under the pointer in the viewport or the tiling map, or null. */
    hoveredTileId: null,
    /**
     * The material cell `{ i, j }` currently ARMED in the plan view (P8, manual
     * overrides) — a square held, waiting for a click on a highlighted
     * neighbour to combine with, or Escape/elsewhere to let go. Mirrors v2's
     * identically-named GridMap concept. UI-only: never touches `config`.
     */
    armedCell: null,
    /**
     * Plain-language description of what the last `combineCells` /
     * `splitTileAt` / `clearOverrides` call actually did (or why it was
     * rejected) — mirrors v2's `lastNotice`. UI-only, shown under the plan
     * view; `null` until the first such action.
     */
    lastActionNotice: null,

    // --- actions: form knobs --------------------------------------------------
    /**
     * Set one `config.form` field, clamped to its V3_SPEC §6 range (or, for
     * angularity/facetCells, form.js's private range — see the note above).
     * @param {'amplitude'|'crestX'|'crestZ'|'ridgeShear'|'toeSharpX'|'toeSharpZ'|'angularity'|'facetCells'} key
     * @param {number} value
     */
    setForm: (key, value) =>
      commit((draft) => {
        if (!draft.form || typeof draft.form !== 'object') return
        const current = draft.form[key]
        const n = numOr(value, current)
        const ranges = {
          amplitude: [AMPLITUDE_MIN, AMPLITUDE_MAX],
          crestX: [CREST_MIN, CREST_MAX],
          crestZ: [CREST_MIN, CREST_MAX],
          ridgeShear: [RIDGE_SHEAR_MIN, RIDGE_SHEAR_MAX],
          toeSharpX: [TOE_SHARP_MIN, TOE_SHARP_MAX],
          toeSharpZ: [TOE_SHARP_MIN, TOE_SHARP_MAX],
          angularity: [ANGULARITY_MIN, ANGULARITY_MAX],
          facetCells: [FACET_CELLS_MIN, FACET_CELLS_MAX],
        }
        const range = ranges[key]
        if (!range) return
        const clamped = clamp(n, range[0], range[1])
        draft.form[key] = key === 'facetCells' ? Math.round(clamped) : clamped
      }),

    // --- actions: sheet -------------------------------------------------------
    setSheetCols: (n) =>
      commit((draft) => {
        if (!draft.sheet) return
        draft.sheet.cols = clamp(Math.round(numOr(n, draft.sheet.cols)), SHEET_COLS_MIN, SHEET_COLS_MAX)
      }),
    setSheetRows: (n) =>
      commit((draft) => {
        if (!draft.sheet) return
        draft.sheet.rows = clamp(Math.round(numOr(n, draft.sheet.rows)), SHEET_ROWS_MIN, SHEET_ROWS_MAX)
      }),

    // --- actions: tiling --------------------------------------------------------
    setTilingStrategy: (strategy) =>
      commit((draft) => {
        if (!draft.tiling) return
        if (!TILING_STRATEGIES.includes(strategy)) return
        draft.tiling.strategy = strategy
      }),
    /**
     * The plate budget. `null` means unlimited. Pinned plates spend it too — a
     * plate placed by hand is still a plate you have to buy — so a budget below
     * the current pin count is legal, places every pin anyway, and reports
     * W_PLATE_BUDGET_EXCEEDED rather than refusing.
     */
    setMaxPlates: (n) =>
      commit((draft) => {
        if (!draft.tiling) return
        draft.tiling.maxPlates = n === null || n === undefined
          ? null
          : Math.max(0, Math.round(numOr(n, 0)))
      }),

    setPlateFitTolerance: (n) =>
      commit((draft) => {
        if (!draft.tiling) return
        draft.tiling.plateFitToleranceCm = clamp(
          numOr(n, draft.tiling.plateFitToleranceCm),
          PLATE_FIT_TOLERANCE_MIN,
          PLATE_FIT_TOLERANCE_MAX,
        )
      }),

    // --- actions: manual tile overrides (P8) + plan-view arming --------------
    // Reproduces v2's pair-click merge interaction (GridMap.jsx / core/merge.js,
    // commit 565af13) on top of v3's simpler model: an override is just a
    // pinned `{ i, j, type, axis }` entry (schema.js's "MANUAL TILE
    // OVERRIDES"), so there is no geometric coercion to compute here — the
    // consequence is entirely captured by `tiling.js`'s sagitta gate and its
    // `W_PLATE_OVERRIDE_MISFIT` warning. Every action below goes through
    // `commit()`, so validation, undo/redo and autosave all come for free.

    /** Arm a square cell `{ i, j }` in the plan view — the first click of a
     *  combine. UI-only: never touches `config`. */
    armCell: (cell) => set({ armedCell: cell ?? null }),
    /** Let go of the armed cell — Escape, a click elsewhere, or after an action. */
    clearArmed: () => set({ armedCell: null }),

    /**
     * Combine two ADJACENT material cells `{ i, j }` into one 60×121 plate by
     * adding a `'2x4'` override anchored at their lower-index corner (axis
     * implied by whether they differ in i or j — mirrors tiling.js's own
     * candidate convention, so there is no H/V mode to choose). Any existing
     * override touching either cell (a prior square or plate pin) is dropped
     * first, so the new combination always wins.
     *
     * A manually-placed plate is placed even where it bows past
     * `tiling.plateFitToleranceCm` (tiling.js's "MANUAL OVERRIDES") — this
     * action does not fail for "doesn't fit"; `lastActionNotice` reports the
     * cost instead, reading the `W_PLATE_OVERRIDE_MISFIT` warning straight off
     * the freshly-committed config's own derived report.
     *
     * @param {{i:number,j:number}} a
     * @param {{i:number,j:number}} b
     * @returns {boolean} whether the combination committed
     */
    combineCells: (a, b) => {
      if (!a || !b || !Number.isInteger(a.i) || !Number.isInteger(a.j) || !Number.isInteger(b.i) || !Number.isInteger(b.j)) {
        set({ lastActionNotice: 'combine needs two cells, each { i, j }' })
        return false
      }
      const dr = Math.abs(a.i - b.i)
      const dc = Math.abs(a.j - b.j)
      if (dr + dc !== 1) {
        set({
          lastActionNotice: `(${a.i}, ${a.j}) and (${b.i}, ${b.j}) are not adjacent — a plate needs two cells sharing an edge`,
        })
        return false
      }
      const axis = a.j === b.j ? 'u' : 'v'
      const i = Math.min(a.i, b.i)
      const j = Math.min(a.j, b.j)
      const touched = new Set([`${a.i},${a.j}`, `${b.i},${b.j}`])

      const committed = commit((draft) => {
        if (!draft.tiling) return
        const existing = Array.isArray(draft.tiling.overrides) ? draft.tiling.overrides : []
        draft.tiling.overrides = existing.filter((ov) => !overrideCells(ov).some(([ci, cj]) => touched.has(`${ci},${cj}`)))
        draft.tiling.overrides.push({ i, j, type: '2x4', axis })
      })

      if (committed) {
        const { report } = getDerived(get().config)
        const tileId = `pl_${i}_${j}_${axis}`
        const misfit = report.warnings.find((w) => w.code === OVERRIDE_MISFIT_CODE && w.tile === tileId)
        set({
          lastActionNotice: misfit
            ? `combined (${a.i}, ${a.j}) + (${b.i}, ${b.j}) into a plate — ${misfit.message}`
            : `combined (${a.i}, ${a.j}) + (${b.i}, ${b.j}) into a ${axis === 'u' ? 'horizontal' : 'vertical'} 121cm plate`,
        })
      } else {
        const msg = get().lastErrors.map((e) => e.message).join('; ') || 'the change was rejected'
        set({ lastActionNotice: `could not combine (${a.i}, ${a.j}) and (${b.i}, ${b.j}): ${msg}` })
      }
      return committed
    },

    /**
     * Split the plate covering material cell `(i, j)` back into two squares,
     * by PINNING both of its cells as `'2x2'` overrides (not merely removing
     * whatever override made it a plate). That distinction matters: if
     * splitting only erased an override, an algorithm-chosen plate (no
     * override at all) would just get re-chosen on the very next solve, and a
     * merely-unpinned pair could too — pinning both cells as squares is what
     * makes the split a durable decision instead of a no-op.
     *
     * DELIBERATE ASYMMETRY, carried over from v2's core/merge.js (commit
     * 565af13) with the same reasoning even though v3's placement has no
     * geometric coercion to give back: splitting does not restore anything
     * else that placing the plate changed — it only pins "two panels here".
     * `undo()` is the way back to the pre-plate state, not another click.
     *
     * @param {number} i
     * @param {number} j
     * @returns {boolean} whether the split committed
     */
    splitTileAt: (i, j) => {
      const { layout } = getDerived(get().config)
      const tile = layout.tiles.find((t) => t.cells.some(([ci, cj]) => ci === i && cj === j))
      if (!tile || tile.type !== '2x4') {
        set({ lastActionNotice: `(${i}, ${j}) is not a plate — nothing to split` })
        return false
      }
      const cellsToSquare = tile.cells
      const touched = new Set(cellsToSquare.map(([ci, cj]) => `${ci},${cj}`))

      const committed = commit((draft) => {
        if (!draft.tiling) return
        const existing = Array.isArray(draft.tiling.overrides) ? draft.tiling.overrides : []
        draft.tiling.overrides = existing.filter((ov) => !overrideCells(ov).some(([ci, cj]) => touched.has(`${ci},${cj}`)))
        for (const [ci, cj] of cellsToSquare) draft.tiling.overrides.push({ i: ci, j: cj, type: '2x2', axis: 'u' })
      })

      if (committed) {
        set({
          lastActionNotice: `split ${tile.id} back into two squares at ${cellsToSquare
            .map(([ci, cj]) => `(${ci}, ${cj})`)
            .join(' + ')}`,
        })
      } else {
        const msg = get().lastErrors.map((e) => e.message).join('; ') || 'the change was rejected'
        set({ lastActionNotice: `could not split (${i}, ${j}): ${msg}` })
      }
      return committed
    },

    /** Remove every manual override (P8) and let the algorithm decide every cell again. */
    clearOverrides: () => {
      const before = (get().config.tiling.overrides ?? []).length
      if (before === 0) {
        set({ lastActionNotice: 'no manual overrides to clear' })
        return true
      }
      const committed = commit((draft) => {
        if (!draft.tiling) return
        draft.tiling.overrides = []
      })
      if (committed) {
        set({
          lastActionNotice: `cleared ${before} manual override${before === 1 ? '' : 's'} — the algorithm now decides every cell`,
          armedCell: null,
        })
      }
      return committed
    },

    // --- actions: placement -----------------------------------------------------
    setPlacementMode: (mode) =>
      commit((draft) => {
        if (!draft.placement) return
        if (!PLACEMENT_MODES.includes(mode)) return
        draft.placement.mode = mode
      }),
    setPlacementTree: (tree) =>
      commit((draft) => {
        if (!draft.placement) return
        if (!PLACEMENT_TREES.includes(tree)) return
        draft.placement.tree = tree
      }),

    // --- actions: gap / tolerances -----------------------------------------------
    setGap: (n) =>
      commit((draft) => {
        draft.gap = clamp(numOr(n, draft.gap), GAP_MIN, GAP_MAX)
      }),
    setGapTolerance: (n) =>
      commit((draft) => {
        const v = numOr(n, draft.gapTolerance)
        draft.gapTolerance = v > 0 ? v : draft.gapTolerance
      }),
    setGroundTolerance: (n) =>
      commit((draft) => {
        const v = numOr(n, draft.groundTolerance)
        draft.groundTolerance = v > 0 ? v : draft.groundTolerance
      }),

    // --- actions: whole-config -----------------------------------------------
    /**
     * Replace the config wholesale (the JSON panel's paste/Apply path).
     * Normalized, then validated; invalid input is ignored (errors → lastErrors).
     */
    setConfig: (config) => {
      let candidate
      try {
        candidate = normalizeConfig(config)
      } catch (err) {
        set({ lastErrors: [{ code: 'E_SHAPE', message: String(err.message ?? err), path: '' }] })
        return false
      }
      return commitConfig(candidate)
    },

    /** Reset to the shipped default drift config. Always commits. */
    resetConfig: () => commitConfig(normalizeConfig(DEFAULT_CONFIG)),

    /**
     * Load a named drift preset. Each preset pins a different answer to "how
     * much joint deviation is this installation willing to build?" — see
     * core/v3/presets.js for what each one trades away.
     */
     applyPreset: (id) => {
       const cfg = buildPreset(id)
       if (!cfg) return
       commitConfig(cfg)
     },

    /** Step back to the previously committed config. NOT re-validated. */
    undo: () => {
      const state = get()
      if (state.past.length === 0) return false
      const entry = state.past[state.past.length - 1]
      set({
        config: entry.config,
        lastWarnings: entry.warnings,
        lastErrors: [],
        past: state.past.slice(0, -1),
        future: [{ config: state.config, warnings: state.lastWarnings }, ...state.future],
        canUndo: state.past.length - 1 > 0,
        canRedo: true,
        armedCell: null,
      })
      saveWorkingConfig(entry.config)
      return true
    },

    /** Re-apply the change the last `undo()` took back. Also not re-validated. */
    redo: () => {
      const state = get()
      if (state.future.length === 0) return false
      const entry = state.future[0]
      const past = [...state.past, { config: state.config, warnings: state.lastWarnings }]
      while (past.length > HISTORY_LIMIT) past.shift()
      set({
        config: entry.config,
        lastWarnings: entry.warnings,
        lastErrors: [],
        past,
        future: state.future.slice(1),
        canUndo: past.length > 0,
        canRedo: state.future.length - 1 > 0,
        armedCell: null,
      })
      saveWorkingConfig(entry.config)
      return true
    },

    // --- actions: UI-only (never touch `config`, never fail validation) -----
    toggleBounds: (on) => set((s) => ({ showBounds: on === undefined ? !s.showBounds : Boolean(on) })),
    toggleGhost: (on) => set((s) => ({ showGhost: on === undefined ? !s.showGhost : Boolean(on) })),
    setColorMode: (mode) => set({ colorMode: mode }),
    setHoveredTile: (id) => set({ hoveredTileId: id ?? null }),
  }
})

export default useStoreV3
