/**
 * grid-designer — application state (zustand + immer).
 *
 * =============================================================================
 * WHAT LIVES HERE
 * =============================================================================
 * The store owns exactly one piece of truth: `config` (the JSON schema object
 * documented in core/schema.js). Everything geometric — panel placement and the
 * joint-consistency report — is DERIVED from it by the pure headless functions
 * in core/, never stored.
 *
 * =============================================================================
 * THE COMMIT RULE
 * =============================================================================
 * Every mutating action goes through `commit()`, which
 *   1. produces the candidate config with immer,
 *   2. runs `validateConfig` on it,
 *   3. commits ONLY if `ok`; otherwise the previous config is kept untouched
 *      and the errors are stashed in `lastErrors` for the UI to display.
 * So the store can never hold an invalid config, and the UI never has to guard
 * against one. (Example: `setRowZigzag(0, 20)` is rejected by
 * E_SHORE_NOT_FLAT — row 0 is the shore and stays flat.)
 *
 * =============================================================================
 * DERIVED DATA MEMOIZATION
 * =============================================================================
 * `getDerived(config)` caches `{ layout, report }` in a module-level WeakMap
 * keyed on the config OBJECT IDENTITY. Because immer returns a fresh (frozen)
 * object on every accepted mutation and the same object otherwise, this means:
 *   - solve + report run at most once per config change, never per frame;
 *   - the returned object is reference-stable, so components can select it
 *     directly (`useStore(s => getDerived(s.config))`) without re-render loops;
 *   - dropped configs are garbage-collected with their cache entry.
 *
 * Chosen over zustand middleware because it also works outside React (the
 * Playwright harness and node scripts can call `getDerived` directly).
 */

import { create } from 'zustand'
import { produce } from 'immer'
import {
  MAX_ROW_FOLD_DEG,
  MAX_ZIGZAG_DEG,
  clamp,
  normalizeConfig,
  validateConfig,
} from './core/schema.js'
import { buildPreset } from './core/presets.js'
import { solveLayout } from './core/placement.js'
import { jointReport } from './core/report.js'

// -----------------------------------------------------------------------------
// Derived-data cache
// -----------------------------------------------------------------------------
const derivedCache = new WeakMap()

/**
 * Solve + measure a config, memoized on the config's object identity.
 *
 * @param {object} config a validated, normalized config
 * @returns {{ layout: object, report: object }} reference-stable per config
 */
export function getDerived(config) {
  let entry = derivedCache.get(config)
  if (!entry) {
    const layout = solveLayout(config)
    const report = jointReport(layout, config)
    entry = { layout, report }
    derivedCache.set(config, entry)
  }
  return entry
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------
const DEFAULT_SEED = 1

/** Coerce a UI value to a finite number, or null when blank/unparseable. */
function numOrNull(v) {
  if (v === null || v === undefined || v === '') return null
  const n = Number(v)
  return Number.isFinite(n) ? n : null
}

/**
 * Does a rect entry occupy cell (r, c)?
 *
 * Mirrors the cell arithmetic in schema.js's overlap check and placement.js's
 * rect index: horizontal spans (row, col)+(row, col+1), vertical spans
 * (row, col)+(row+1, col). Kept local to the store — `rects` is config data, so
 * the grid map's "which rect did I click?" lookup needs no core additions.
 *
 * @param {object} rect a config rect entry
 * @param {number} r row
 * @param {number} c column
 * @returns {boolean}
 */
function rectCoversCell(rect, r, c) {
  if (!rect || typeof rect !== 'object') return false
  if (rect.orientation === 'horizontal') {
    return rect.row === r && (rect.col === c || rect.col + 1 === c)
  }
  if (rect.orientation === 'vertical') {
    return rect.col === c && (rect.row === r || rect.row + 1 === r)
  }
  return false
}

// -----------------------------------------------------------------------------
// Store
// -----------------------------------------------------------------------------
const useStore = create((set, get) => {
  /**
   * Apply an immer recipe to `config` and commit only if the result validates.
   *
   * @param {(draft: object) => void} recipe
   * @returns {boolean} whether the change was committed
   */
  function commit(recipe) {
    const next = produce(get().config, recipe)
    if (next === get().config) return true // no-op recipe
    return commitConfig(next)
  }

  /** Validate and commit a whole config object. */
  function commitConfig(candidate) {
    const result = validateConfig(candidate)
    if (!result.ok) {
      set({ lastErrors: result.errors, lastWarnings: result.warnings })
      return false
    }
    set({ config: candidate, lastErrors: [], lastWarnings: result.warnings })
    return true
  }

  const initialConfig = normalizeConfig(buildPreset('flat'))
  const initialValidation = validateConfig(initialConfig)

  return {
    // --- state ------------------------------------------------------------
    config: initialConfig,
    seed: DEFAULT_SEED,
    /** Errors from the most recent REJECTED action ([] after a success). */
    lastErrors: initialValidation.errors,
    /** Warnings from the most recent accepted config (informational). */
    lastWarnings: initialValidation.warnings,

    // --- actions ----------------------------------------------------------
    /**
     * Set a row's base zig-zag amplitude. Clamped to ±80°.
     * Row 0 is the shore: any non-zero value is rejected by validation.
     */
    setRowZigzag: (r, deg) =>
      commit((draft) => {
        const row = draft.rows?.[r]
        if (!row) return
        const n = numOrNull(deg)
        row.zigzagDeg = n === null ? 0 : clamp(n, -MAX_ZIGZAG_DEG, MAX_ZIGZAG_DEG)
      }),

    /**
     * Set the row-to-row dihedral between rows i and i+1. Clamped to ±120°.
     * Positive pitches row i+1 up.
     */
    setRowFold: (i, deg) =>
      commit((draft) => {
        if (!Array.isArray(draft.rowFoldsDeg) || i < 0 || i >= draft.rowFoldsDeg.length) return
        const n = numOrNull(deg)
        draft.rowFoldsDeg[i] = n === null ? 0 : clamp(n, -MAX_ROW_FOLD_DEG, MAX_ROW_FOLD_DEG)
      }),

    /**
     * Set (or clear) the signed per-joint override at joint j of row r.
     * `null` removes the override so the alternating ±zigzag rule applies again.
     */
    setJointOverride: (r, j, degOrNull) =>
      commit((draft) => {
        const row = draft.rows?.[r]
        if (!row) return
        if (!row.jointOverridesDeg) row.jointOverridesDeg = {}
        const key = String(j)
        const n = numOrNull(degOrNull)
        if (n === null) delete row.jointOverridesDeg[key]
        else row.jointOverridesDeg[key] = clamp(n, -MAX_ZIGZAG_DEG, MAX_ZIGZAG_DEG)
      }),

    /**
     * Replace the config with a preset. `seed` only matters for 'random';
     * when given it is also stored so the seed input stays in sync.
     */
    applyPreset: (id, seed) => {
      const s = seed === undefined ? get().seed : seed
      let candidate
      try {
        candidate = normalizeConfig(buildPreset(id, s))
      } catch (err) {
        set({ lastErrors: [{ code: 'E_PRESET', message: String(err.message ?? err), path: '' }] })
        return false
      }
      if (seed !== undefined) set({ seed: s })
      return commitConfig(candidate)
    },

    /** Store the random-preset seed. Does NOT rebuild the config on its own. */
    setSeed: (n) => {
      const v = numOrNull(n)
      set({ seed: v === null ? 0 : Math.trunc(v) })
    },

    /**
     * Add a 60×121 two-cell plate.
     *
     * Goes through the same commit rule as everything else, so an illegal
     * placement (overlapping an existing rect, out of bounds, or — for a
     * vertical plate — landing on two cells whose in-row tilts disagree) leaves
     * the config untouched and lands in `lastErrors` for the grid map to show.
     *
     * @param {{row:number, col:number, orientation:'horizontal'|'vertical'}} rect
     * @returns {boolean} whether the placement was committed
     */
    addRect: (rect) =>
      commit((draft) => {
        if (!rect || typeof rect !== 'object') return
        if (!Array.isArray(draft.rects)) draft.rects = []
        draft.rects.push({ row: rect.row, col: rect.col, orientation: rect.orientation })
      }),

    /**
     * Remove the rect occupying cell (r, c), if any.
     *
     * Still validated: dropping a horizontal plate restores the in-row joint it
     * removed, which can retroactively break a vertical plate elsewhere in the
     * same rows — that rejection is reported rather than silently applied.
     *
     * @param {number} r row
     * @param {number} c column
     * @returns {boolean} whether the removal was committed
     */
    removeRectAt: (r, c) =>
      commit((draft) => {
        const rects = Array.isArray(draft.rects) ? draft.rects : []
        const i = rects.findIndex((rect) => rectCoversCell(rect, r, c))
        if (i >= 0) rects.splice(i, 1)
      }),

    /**
     * Replace the config wholesale (WP4's JSON import path).
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
  }
})

export default useStore
