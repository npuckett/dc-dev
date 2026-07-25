/**
 * grid-designer — application state (zustand + immer).
 *
 * =============================================================================
 * WHAT LIVES HERE
 * =============================================================================
 * The store owns exactly one piece of DESIGN truth: `config` (the JSON schema
 * object documented in core/schema.js). Everything geometric — panel placement,
 * the joint-consistency report, the overall bounding box — is DERIVED from it by
 * the pure headless functions in core/, never stored.
 *
 * Alongside it sit a couple of pieces of plain VIEW state (`seed`, `showBounds`,
 * the last errors / warnings). They are not part of the design, never go through
 * the commit rule, and never reach the exporters.
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
 * against one. (Example: `setColumnFold(c, k, 20)` on a joint a vertical plate
 * removed is rejected by E_FOLD_ON_REMOVED_JOINT — that plate cannot bend.)
 *
 * The GROUNDED-END rule is deliberately OUTSIDE that gate: a config whose last
 * panels float is perfectly valid and committable (dragging a fold slider passes
 * through such states constantly) — it just carries `violations` in its derived
 * layout. `groundColumn(c)` / `groundAllColumns()` solve them away on demand, and
 * report E_UNGROUNDABLE through `lastErrors` when the last hinge cannot reach. A
 * WALL-SUPPORTED column (`endSupport: 'wall'`, legal only on the last column) is
 * exempt from the rule and skipped by both actions.
 *
 * `setRows(n)` is the one action that changes the SHAPE of the config rather than a
 * value in it — it resizes every `foldsDeg` and drops plates that no longer fit,
 * all through the same single commit, so the store never holds a half-resized grid.
 *
 * =============================================================================
 * DERIVED DATA MEMOIZATION
 * =============================================================================
 * `getDerived(config)` caches `{ layout, report, violations, bounds }` in a WeakMap
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
  MAX_FOLD_DEG,
  MAX_ROWS,
  MIN_ROWS,
  clamp,
  columnEndSupport,
  normalizeConfig,
  validateConfig,
} from './core/schema.js'
import { buildPreset } from './core/presets.js'
import { layoutBounds, solveLayout } from './core/placement.js'
import { groundAllFolds, solveGroundingFold } from './core/ground.js'
import { jointReport } from './core/report.js'

// -----------------------------------------------------------------------------
// Derived-data cache
// -----------------------------------------------------------------------------
const derivedCache = new WeakMap()

/**
 * Solve + measure a config, memoized on the config's object identity.
 *
 * `violations` is `layout.violations` hoisted to the top level — the grounded-end
 * rule (E_END_FLOATING, see core/placement.js) is a property of the whole design
 * rather than of any one panel, and four components read it.
 *
 * `bounds` is the design's OVERALL BOX (`layoutBounds`): the axis-aligned extent
 * of every panel's solid, housings included. It is memoized here rather than
 * computed in the viewport because it walks all 8 corners of every panel, and the
 * measuring box that draws it re-renders on camera moves.
 *
 * @param {object} config a validated, normalized config
 * @returns {{ layout: object, report: object, violations: Array, bounds: object }}
 *          reference-stable per config
 */
export function getDerived(config) {
  let entry = derivedCache.get(config)
  if (!entry) {
    const layout = solveLayout(config)
    const report = jointReport(layout, config)
    entry = {
      layout,
      report,
      violations: layout.violations,
      bounds: layoutBounds(layout, config),
    }
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
    /**
     * Draw the overall measuring box (and its cm dimension labels) in the 3D view?
     *
     * PLAIN UI STATE, deliberately NOT part of `config`: it is a ruler held up
     * against the design, not a property of it, so it never lands in the exported
     * JSON / OBJ and never goes through the commit rule. On by default — the peak
     * height is the number a design gets tuned against — and dismissible from the
     * toolbar when it gets in the way of judging the surface.
     */
    showBounds: true,

    // --- actions ----------------------------------------------------------
    /**
     * Set the signed dihedral at joint k of column c (the hinge between rows k
     * and k+1 of that fold strip). Clamped to ±120°; positive pitches the next
     * panel UP. Rejected by E_FOLD_ON_REMOVED_JOINT when a vertical plate spans
     * that joint.
     */
    setColumnFold: (c, k, deg) =>
      commit((draft) => {
        const folds = draft.columns?.[c]?.foldsDeg
        if (!Array.isArray(folds) || k < 0 || k >= folds.length) return
        const n = numOrNull(deg)
        folds[k] = n === null ? 0 : clamp(n, -MAX_FOLD_DEG, MAX_FOLD_DEG)
      }),

    /**
     * Set the pitch of column `c`'s FRONT (window) panel — row 0. Clamped to
     * ±120°; positive tilts the front UP out of the window. The chain's front edge
     * stays on the window line z = 0 and the front panel stays ON THE FLOOR at any
     * pitch (core/schema.js `frontRestY` picks the start height), so this only ever
     * changes the shape of the strip, never whether its front is supported.
     *
     * A NEGATIVE pitch dives the strip at the window: the front panel's rear edge
     * becomes its contact edge, the whole chain starts that much higher, and a
     * profile that keeps diving pushes later panels under the floor (W_BELOW_FLOOR).
     * That is surfaced, not prevented.
     */
    setColumnStartPitch: (c, deg) =>
      commit((draft) => {
        const column = draft.columns?.[c]
        if (!column || typeof column !== 'object') return
        const n = numOrNull(deg)
        column.startPitchDeg = n === null ? 0 : clamp(n, -MAX_FOLD_DEG, MAX_FOLD_DEG)
      }),

    /**
     * Set how column `c`'s deep end is held up: 'floor' (it must land — the
     * grounded-end rule) or 'wall' (side-bracketed to the −X wall, so it may end
     * high). 'wall' is only valid on the wall-adjacent column, so the store's
     * commit rule rejects it anywhere else with E_SHAPE.
     */
    setColumnEndSupport: (c, endSupport) =>
      commit((draft) => {
        const column = draft.columns?.[c]
        if (!column || typeof column !== 'object') return
        column.endSupport = endSupport
      }),

    /**
     * Change the ROW RESOLUTION of the whole grid (how many panels deep each
     * column strip is). Range MIN_ROWS..MAX_ROWS = 5..8; anything else is clamped
     * into it rather than rejected, so a stepper can never wedge the UI.
     *
     * The resize is deliberately SIMPLE AND PREDICTABLE, all in one commit:
     *   - every column's `foldsDeg` is truncated or zero-padded AT THE END (the
     *     back of the strip), so the shape you already designed at the window is
     *     never touched — growing adds level rows behind the last one, shrinking
     *     drops the deepest hinges;
     *   - `startPitchDeg` / `endSupport` are untouched;
     *   - rects that no longer fit are DROPPED (a plate whose anchor row is past
     *     the new last row, or a vertical plate that no longer has a second row).
     * Every kept plate stays legal, because the folds in front of the truncation
     * point do not move.
     *
     * Grounding is NOT re-solved: shrinking can leave a column's new last panel in
     * mid-air, which shows up as an E_END_FLOATING violation for "Ground all" to
     * fix — the same as any other fold edit.
     *
     * @param {number} n desired row count
     * @returns {boolean} whether the change was committed
     */
    setRows: (n) => {
      const v = numOrNull(n)
      const rows = v === null ? MIN_ROWS : clamp(Math.trunc(v), MIN_ROWS, MAX_ROWS)
      return commit((draft) => {
        if (draft.grid?.rows === rows) return
        draft.grid.rows = rows
        const want = rows - 1
        for (const column of Array.isArray(draft.columns) ? draft.columns : []) {
          if (!column || !Array.isArray(column.foldsDeg)) continue
          while (column.foldsDeg.length > want) column.foldsDeg.pop()
          while (column.foldsDeg.length < want) column.foldsDeg.push(0)
        }
        if (Array.isArray(draft.rects)) {
          draft.rects = draft.rects.filter((rect) => {
            if (!rect || typeof rect !== 'object') return false
            const lastRow = rect.orientation === 'vertical' ? rows - 2 : rows - 1
            return Number.isInteger(rect.row) && rect.row >= 0 && rect.row <= lastRow
          })
        }
      })
    },

    /**
     * Give every column the fold sequence of column `c` — a cylindrical fold,
     * whose in-row joints are all exact by construction (identical profiles).
     *
     * Rejected (leaving the config untouched) when a rect forbids the result:
     * a vertical plate whose joint would end up folded → E_FOLD_ON_REMOVED_JOINT,
     * a horizontal plate whose two columns would stop agreeing in pitch →
     * E_CROSSCOL_ANGLE_MISMATCH. In practice copying makes every column identical,
     * so only the vertical-plate case can fire.
     *
     * @param {number} c source column
     * @returns {boolean} whether the change was committed
     */
    copyColumnToAll: (c) =>
      commit((draft) => {
        const source = draft.columns?.[c]?.foldsDeg
        if (!Array.isArray(source)) return
        const folds = source.slice()
        draft.columns.forEach((column, i) => {
          if (i === c) return
          if (!column || !Array.isArray(column.foldsDeg)) return
          column.foldsDeg = folds.slice()
        })
      }),

    /**
     * Rotate every column's fold sequence one step up in index (column c's folds
     * move to column c+1; the last column's wrap around to column 0).
     *
     * This is the one-click phase shift that turns a static fold into a TRAVELLING
     * wave: repeated clicks walk the crest across the window. Note the 3D view
     * looks in from the shore, so rising column index runs right-to-left on screen.
     *
     * @returns {boolean} whether the change was committed
     */
    shiftColumnsRight: () =>
      commit((draft) => {
        const columns = draft.columns
        if (!Array.isArray(columns) || columns.length < 2) return
        const before = columns.map((column) =>
          Array.isArray(column?.foldsDeg) ? column.foldsDeg.slice() : null,
        )
        columns.forEach((column, i) => {
          const source = before[(i - 1 + columns.length) % columns.length]
          if (!column || !Array.isArray(column.foldsDeg) || !source) return
          column.foldsDeg = source.slice()
        })
      }),

    /**
     * Bring column `c`'s last panel back down to the floor (the grounded-end rule
     * in core/schema.js: "the wave returns to the water") by solving its LAST
     * SURVIVING hinge — the deepest joint no vertical plate has removed.
     *
     * Not every floating column can be rescued this way: once the chain's last
     * vertex is much more than a panel-length above the floor, no angle in ±120°
     * reaches down, and the fold profile in FRONT of the end has to arc back down
     * instead. That case reports E_UNGROUNDABLE and changes nothing.
     *
     * A WALL-SUPPORTED column is a NO-OP: it is bracketed to the −X wall and is
     * meant to end high, so there is nothing to solve. (The UI hides its button
     * too — this is the belt to that braces.)
     *
     * @param {number} c column index
     * @returns {boolean} whether the change was committed
     */
    groundColumn: (c) => {
      const config = get().config
      if (columnEndSupport(config, c) === 'wall') {
        set({ lastErrors: [] })
        return true
      }
      const solved = solveGroundingFold(config, c)
      if (!solved) {
        set({
          lastErrors: [
            {
              code: 'E_UNGROUNDABLE',
              message:
                `column ${c}'s last panel cannot reach the floor from where its chain ends — ` +
                `no angle within ±${MAX_FOLD_DEG}° at its last hinge gets there. Fold the rows in ` +
                `FRONT of it back down: the pitch profile has to arc over and descend, not just climb.`,
              path: `columns[${c}].foldsDeg`,
            },
          ],
        })
        return false
      }
      return commit((draft) => {
        const folds = draft.columns?.[c]?.foldsDeg
        if (!Array.isArray(folds)) return
        folds[solved.jointIndex] = solved.foldDeg
      })
    },

    /**
     * Ground every floating column at once. Columns already touching the floor are
     * left exactly as they are, a WALL-SUPPORTED column is skipped entirely (its
     * end is held by the wall, and pulling it down would undo the design), and a
     * column the solver cannot rescue is skipped (it stays in `violations`) rather
     * than blocking the others.
     *
     * @returns {boolean} whether the change was committed
     */
    groundAllColumns: () => {
      const config = get().config
      if (getDerived(config).violations.length === 0) {
        set({ lastErrors: [] })
        return true // already grounded everywhere — nothing to do
      }
      const next = groundAllFolds(config)
      if (!next) {
        set({
          lastErrors: [
            {
              code: 'E_UNGROUNDABLE',
              message:
                'no floating column could be brought down by its last hinge alone — those fold ' +
                'profiles end too high above the floor. Fold the rows in front of the ends back ' +
                'down so each column arcs over and descends.',
              path: 'columns',
            },
          ],
        })
        return false
      }
      return commitConfig(next)
    },

    /**
     * Set every hinge of every column to 0° — the flat reference surface.
     * Rects are KEPT (a flat surface satisfies every rect constraint, so this
     * always commits), which makes it a safe escape hatch from a rejected state.
     *
     * @returns {boolean} whether the change was committed
     */
    flattenFolds: () =>
      commit((draft) => {
        for (const column of Array.isArray(draft.columns) ? draft.columns : []) {
          if (!column || !Array.isArray(column.foldsDeg)) continue
          column.foldsDeg = column.foldsDeg.map(() => 0)
        }
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

    /**
     * Show / hide the overall measuring box in the 3D view. UI-only: it does not
     * touch `config`, so it cannot fail validation and changes nothing derived.
     *
     * @param {boolean} [on] force a state; omitted toggles
     */
    toggleBounds: (on) =>
      set((s) => ({ showBounds: on === undefined ? !s.showBounds : Boolean(on) })),

    /** Store the random-preset seed. Does NOT rebuild the config on its own. */
    setSeed: (n) => {
      const v = numOrNull(n)
      set({ seed: v === null ? 0 : Math.trunc(v) })
    },

    /**
     * Add a 60×121 two-cell plate.
     *
     * Goes through the same commit rule as everything else, so an illegal
     * placement (overlapping an existing rect, out of bounds, a vertical plate on
     * a folded joint, or a horizontal plate across two columns at different
     * pitches) leaves the config untouched and lands in `lastErrors` for the grid
     * map to show.
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
     * Still validated, so any rejection is reported rather than silently applied.
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
