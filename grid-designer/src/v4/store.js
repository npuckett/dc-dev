/**
 * grid-designer v4 — application state (zustand), the folded-ribbon store.
 *
 * =============================================================================
 * SAME CONTRACT AS src/v3/store.js — read that file's header first
 * =============================================================================
 * A NEW store for the v4 pivot (V4_SPEC.md §7), living beside the v3 one rather
 * than replacing it: src/v3/ stays on disk, unmounted, as the record of the
 * retired approach (HANDOFF §0). The contract is carried over unchanged:
 *
 *   - the store owns exactly one piece of DESIGN truth, `config` (schema v4 —
 *     src/core/v4/schema.js);
 *   - every mutation goes through `commit()`, which runs `validateConfig` and
 *     commits ONLY if `valid`; otherwise the previous config is kept and the
 *     errors land in `lastErrors`;
 *   - derived data is memoized in a WeakMap keyed on config OBJECT IDENTITY via
 *     `getDerived(config)`. The three solvers are CHAINED — `solveChain` feeds
 *     `solveConnectorsV4` feeds `buildReportV4` — because each of them will
 *     otherwise re-solve its predecessor internally, and the envelope bisection
 *     inside `buildReportV4` is already 60 full solves of the design. Handing
 *     the chain and the stations down is the difference between one solve per
 *     committed change and four;
 *   - UI-only state (hover, selection, colour mode, toggles) lives OUTSIDE
 *     `config`, never goes through `commit()`, and never reaches the exporters;
 *   - undo/redo is a capped stack of previously committed configs, restored
 *     WITHOUT re-validating (they were valid when stored);
 *   - the working config is seeded from `loadWorkingConfig()` when it validates
 *     (persistence.js's `EXPECTED_CONFIG_VERSION = 4` discards a stale v3 save
 *     before it ever reaches `validateConfig` here — V4_SPEC §6), and autosaved
 *     after every successful commit / undo / redo.
 *
 * =============================================================================
 * WHY commitConfig NORMALIZES, WHERE v3's DID NOT
 * =============================================================================
 * v4's `overrides` array is the one field a UI edit can leave in a shape that is
 * VALID but not CANONICAL: toggling a unit off and on again produces an entry
 * that says `{ present: true, flipped: false, role: 'auto' }`, which
 * `validateConfig` accepts (with a `W_OVERRIDE_NO_OP` warning) and
 * `normalizeConfig` would drop. Leaving it in `config` would mean two sessions
 * that arrived at the same ribbon serialize differently — exactly the property
 * schema.js's override contract exists to guarantee. So every candidate is run
 * through `normalizeConfig` (idempotent, never mutating) before validation, and
 * the store's `config` is always the canonical form of the design.
 *
 * That costs the cheap `next === get().config` no-op check v3 relied on, since
 * normalize returns a fresh object every time. It is replaced by a value
 * comparison, which is what the check was always really asking.
 *
 * =============================================================================
 * ACTIONS
 * =============================================================================
 * Every setter CLAMPS its input to the range src/core/v4/schema.js declares
 * before committing, so a slider or stepper bound to that same range can never
 * produce a rejected commit. Enum setters fall back to the current value on an
 * unrecognised string rather than committing garbage.
 *
 * The per-unit actions (`toggleUnitPresent` / `toggleUnitFlipped` /
 * `toggleEdge`) all write through `patchCell` / `patchEdge`, which find-or-create the
 * `(i, j)` or `(i, j, axis)` entry and let `normalizeConfig` decide whether it survives.
 * Nothing here re-implements the override contract; it just states the change.
 */

import { create } from 'zustand'
import { produce } from 'immer'
import {
  normalizeConfig,
  validateConfig,
  clamp,
  DEFAULT_CONFIG,
  GAP_MIN,
  GAP_MAX,
  ANGLE_MIN,
  ANGLE_MAX,
  PHASE_MIN,
  PHASE_MAX,
  WALL_OFFSET_MIN,
  WALL_OFFSET_MAX,
  WINDOW_OFFSET_MIN,
  WINDOW_OFFSET_MAX,
  Y_OFFSET_MIN,
  Y_OFFSET_MAX,
  WALL_THICKNESS_MIN,
  WALL_THICKNESS_MAX,
  CONNECTOR_LENGTH_MIN,
  CONNECTOR_LENGTH_MAX,
  CONNECTOR_SPACING_MIN,
  CONNECTOR_SPACING_MAX,
  CONNECTOR_MIN_PER_JOINT_MIN,
  CONNECTOR_MIN_PER_JOINT_MAX,
  CONNECTOR_BIN_SPAN_MIN,
  CONNECTOR_BIN_SPAN_MAX,
  CONNECTOR_BIN_ANGLE_MIN,
  CONNECTOR_BIN_ANGLE_MAX,
  CONNECTOR_POWER_EDGES,
  CONNECTOR_SUPPLY_MODES,
  LATTICE_COLS_MIN,
  LATTICE_COLS_MAX,
  LATTICE_ROWS_MIN,
  LATTICE_ROWS_MAX,
  WALL_ANCHORS,
  EDGE_AXES,
  DEFAULT_CELL_OVERRIDE,
  DEFAULT_EDGE_OVERRIDE,
  OBSTACLE_ANCHORS,
  OBSTACLE_POS_MIN,
  OBSTACLE_POS_MAX,
  OBSTACLE_SIZE_MIN,
  OBSTACLE_SIZE_MAX,
  PATTERN_KINDS,
  DEFAULT_WAVE,
  SCRUNCH_MIN,
  SCRUNCH_MAX,
  ATTRACTOR_MIN,
  ATTRACTOR_MAX,
} from '../core/v4/schema.js'
import { solveLattice } from '../core/v4/lattice.js'
import { solveConnectorsV4 } from '../core/v4/connectors.js'
import { buildReportV4 } from '../core/v4/report.js'
import { deleteSlot, loadSlot, loadWorkingConfig, saveSlot, saveWorkingConfig } from '../persistence.js'

// -----------------------------------------------------------------------------
// Derived-data cache — see the file header for why the three solves are chained.
// -----------------------------------------------------------------------------
const derivedCache = new WeakMap()

/**
 * Solve + report a config, memoized on the config OBJECT IDENTITY.
 *
 * @param {object} config a validated, normalized v4 config
 * @returns {{ chain: object, connectors: object, report: object }}
 *   reference-stable per config
 */
export function getDerived(config) {
  let entry = derivedCache.get(config)
  if (!entry) {
    const chain = solveLattice(config)
    const connectors = solveConnectorsV4(config, chain)
    const report = buildReportV4(config, chain, connectors)
    entry = { chain, connectors, report }
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

const oneOf = (list, v, fallback) => (list.includes(v) ? v : fallback)

/**
 * Find-or-create the `(i, j)` CELL override and merge `patch` into it.
 *
 * Deliberately does NOT decide whether the result is worth keeping — an entry
 * that ends up entirely default is dropped by `normalizeConfig` in
 * `commitConfig`, which is the one place that rule lives.
 */
function patchCell(draft, i, j, patch) {
  if (!draft.overrides || typeof draft.overrides !== 'object') draft.overrides = { cells: [], edges: [] }
  if (!Array.isArray(draft.overrides.cells)) draft.overrides.cells = []
  const list = draft.overrides.cells
  const idx = list.findIndex((ov) => ov.i === i && ov.j === j)
  if (idx >= 0) list[idx] = { ...list[idx], ...patch }
  else list.push({ i, j, ...DEFAULT_CELL_OVERRIDE, ...patch })
}

/** Find-or-create the `(i, j, axis)` EDGE override and merge `patch` into it. */
function patchEdge(draft, i, j, axis, patch) {
  if (!draft.overrides || typeof draft.overrides !== 'object') draft.overrides = { cells: [], edges: [] }
  if (!Array.isArray(draft.overrides.edges)) draft.overrides.edges = []
  const list = draft.overrides.edges
  const idx = list.findIndex((ov) => ov.i === i && ov.j === j && ov.axis === axis)
  if (idx >= 0) list[idx] = { ...list[idx], ...patch }
  else list.push({ i, j, axis, ...DEFAULT_EDGE_OVERRIDE, ...patch })
}

/**
 * Switching a cell ON re-connects it to whatever is already there.
 *
 * Drops any `present: false` edge override on an edge that touches `(i, j)` and
 * whose OTHER cell is present. Without this, "add a panel" quietly meant "add a
 * panel, unconnected": the growth in `addCellAt` suppresses the new edges it did
 * not ask for, and those suppressions would then outlive the empty slots they
 * were protecting, so filling one in left it floating beside its neighbour with
 * no ramp between them.
 *
 * Edges to an ABSENT neighbour are left alone. A ramp needs only one cell, so
 * turning those on too would make a single click sprout cantilevers into empty
 * space — which is the thing `addCellAt` suppresses them for in the first place.
 */
function connectCell(draft, i, j) {
  const cells = draft.overrides?.cells ?? []
  const present = (a, b) => {
    const ov = cells.find((c) => c.i === a && c.j === b)
    return ov ? ov.present : true
  }
  const touching = [
    { i: i - 1, j, axis: 'x', far: [i - 1, j] },
    { i, j, axis: 'x', far: [i + 1, j] },
    { i, j: j - 1, axis: 'z', far: [i, j - 1] },
    { i, j, axis: 'z', far: [i, j + 1] },
  ]
  draft.overrides.edges = (draft.overrides.edges ?? []).filter((e) => {
    const m = touching.find((t) => t.i === e.i && t.j === e.j && t.axis === e.axis)
    if (!m || e.present) return true
    return !present(m.far[0], m.far[1])
  })
}

/** The override in force for a CELL, or the default when there is none. */
export function overrideFor(config, i, j) {
  const found = (config.overrides?.cells ?? []).find((ov) => ov.i === i && ov.j === j)
  return found ?? { i, j, ...DEFAULT_CELL_OVERRIDE }
}

/** The override in force for an EDGE, or the default when there is none. */
export function edgeOverrideFor(config, i, j, axis) {
  const found = (config.overrides?.edges ?? []).find((ov) => ov.i === i && ov.j === j && ov.axis === axis)
  return found ?? { i, j, axis, ...DEFAULT_EDGE_OVERRIDE }
}

/** Total override count across both lists — what "clear" has to clear. */
function overrideCount(config) {
  return (config.overrides?.cells ?? []).length + (config.overrides?.edges ?? []).length
}

/** How many previous configs the undo stack keeps (mirrors v3's HISTORY_LIMIT). */
export const HISTORY_LIMIT = 50

// -----------------------------------------------------------------------------
// Store
// -----------------------------------------------------------------------------
const useStoreV4 = create((set, get) => {
  /**
   * Apply an immer recipe to `config` and commit only if the result validates.
   * @param {(draft: object) => void} recipe
   * @returns {boolean} whether the change was committed
   */
  function commit(recipe) {
    const next = produce(get().config, recipe)
    if (next === get().config) return true // no-op recipe — immer handed back the original
    return commitConfig(next)
  }

  /**
   * Canonicalize, validate and commit a whole config. On success the OUTGOING
   * config is pushed onto the undo stack (redo stack cleared) and autosaved
   * (debounced — see persistence.js).
   */
  function commitConfig(raw) {
    let candidate
    try {
      candidate = normalizeConfig(raw)
    } catch (err) {
      set({ lastErrors: [{ code: 'E_SHAPE', message: String(err?.message ?? err), path: '' }] })
      return false
    }
    // The value comparison the identity check used to be — see the file header.
    // A slider re-emitting the value it already has must not push a history entry.
    if (JSON.stringify(candidate) === JSON.stringify(get().config)) return true

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
    })
    saveWorkingConfig(candidate)
    return true
  }

  /**
   * Seed from the autosaved working config when there is one AND it validates
   * (persistence.js's `EXPECTED_CONFIG_VERSION = 4` already discards a stale v3
   * save before this ever sees it); otherwise fall back to `DEFAULT_CONFIG`.
   * Never throws.
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
    /**
     * Draw the 3D-printed connectors? Default ON — they are the structure (there
     * is no substructure), and in v4 they are also the thing the whole model is
     * built around, so a view without them is a view of half the argument.
     */
    showConnectors: true,
    /** The world origin triad. On by default: the convention it draws is the one
     *  every on-site dimension is quoted from. */
    showOrigin: true,
    /** Colour mode for the 3D viewport: 'role' | 'fold' | 'flip' | 'flags'. */
    colorMode: 'role',
    /** Unit id under the pointer in the viewport or the units table, or null. */
    hoveredUnitId: null,
    /** Unit id clicked in the viewport or the units table, or null. */
    selectedUnitId: null,
    /** Plain-language description of what the last override action did. */
    lastActionNotice: null,

    // --- actions: the knobs the design is actually driven by ----------------
    /**
     * The lattice size, in FLAT CELLS. The ramps are derived, so these two
     * numbers and θ are the whole shape (V4_SPEC §9). An m × n lattice is
     * mn cells + (m−1)n + m(n−1) ramps: 3 × 5 is 15 + 22 = 37 panels.
     */
    setCols: (n) =>
      commit((draft) => {
        draft.lattice.cols = clamp(Math.round(numOr(n, draft.lattice.cols)), LATTICE_COLS_MIN, LATTICE_COLS_MAX)
      }),

    setRows: (n) =>
      commit((draft) => {
        draft.lattice.rows = clamp(Math.round(numOr(n, draft.lattice.rows)), LATTICE_ROWS_MIN, LATTICE_ROWS_MAX)
      }),

    /**
     * θ — the single shape parameter (V4_SPEC §1). Clamped to the SCHEMA range,
     * not to `envelope.maxAngleDeg`: the envelope is a report about the design,
     * and refusing to let the user past it would turn a permission back into the
     * refusal v4 exists to stop giving (HANDOFF §0).
     */
    setAngleDeg: (v) =>
      commit((draft) => {
        draft.angleDeg = clamp(numOr(v, draft.angleDeg), ANGLE_MIN, ANGLE_MAX)
      }),

    /** The joint width — in v4 an INPUT the geometry honours exactly. */
    setGap: (v) =>
      commit((draft) => {
        draft.gap = clamp(numOr(v, draft.gap), GAP_MIN, GAP_MAX)
      }),

    /**
     * Brace the low-x boundary against the wall line, or leave it free
     * (V4_SPEC §9.4). Braced adds one descending ramp per present HIGH cell in
     * column 0; ground cells there need nothing, being already on the floor.
     * What the toe actually fixes to is an open hardware question — the geometry
     * only says where it lands.
     */
    setWallAnchor: (v) =>
      commit((draft) => {
        draft.placement.wallAnchor = oneOf(WALL_ANCHORS, v, draft.placement.wallAnchor)
      }),

    /**
     * Switch between the two fold patterns (`core/v4/wave.js`).
     *
     * This is a change of MODEL, not of a parameter: the checkerboard has two
     * levels and one angle, the wave has a separable height field and one angle
     * per edge, and the checkerboard provably cannot carry the second. So the
     * notice says what actually changed rather than letting the viewport be the
     * first place you find out the network is a different shape.
     *
     * Switching to 'wave' writes the neutral knobs if the config has none;
     * switching away drops the block (normalizeConfig does that, not this).
     */
    setPatternKind: (kind) => {
      const next = oneOf(PATTERN_KINDS, kind, get().config.pattern.kind)
      if (next === get().config.pattern.kind) return true
      const ok = commit((draft) => {
        draft.pattern.kind = next
        if (next === 'wave' && !draft.pattern.wave) draft.pattern.wave = { ...DEFAULT_WAVE }
      })
      if (ok) {
        set({
          lastActionNotice:
            next === 'wave'
              ? 'wave — the height field is now h(i,j) = f(i) + g(j), so the levels are no longer a ' +
                'checkerboard: at zero scrunch this is a uniform egg-crate with three storeys, not two. ' +
                'That is the only family that can carry a varying angle'
              : 'trapezoid — back to the two-level checkerboard and one angle everywhere',
        })
      }
      return ok
    },

    /**
     * One of the wave's four knobs. Clamped on the way in, like every other
     * setter here, so a slider drag can never produce a rejected commit.
     *
     * Ignored outright when the pattern is not a wave: `normalizeConfig` would
     * drop the block anyway, and committing it would push a no-op onto the undo
     * stack for a control that is not even on screen.
     *
     * @param {'scrunchX'|'scrunchZ'|'attractorX'|'attractorZ'} key
     */
    setWaveKnob: (key, value) =>
      commit((draft) => {
        if (draft.pattern.kind !== 'wave') return
        if (!draft.pattern.wave) draft.pattern.wave = { ...DEFAULT_WAVE }
        const w = draft.pattern.wave
        const [lo, hi] = key === 'scrunchX' || key === 'scrunchZ'
          ? [SCRUNCH_MIN, SCRUNCH_MAX]
          : [ATTRACTOR_MIN, ATTRACTOR_MAX]
        if (!Object.prototype.hasOwnProperty.call(DEFAULT_WAVE, key)) return
        w[key] = clamp(numOr(value, w[key]), lo, hi)
      }),

    /** Which level cell (0,0) starts on. The field has period 2, so this is a
     *  straight swap of ground and high across the whole lattice. Does nothing
     *  under the wave, which has no checkerboard to shift. */
    setPhase: (n) =>
      commit((draft) => {
        draft.pattern.phase = clamp(Math.round(numOr(n, draft.pattern.phase)), PHASE_MIN, PHASE_MAX)
      }),

    // --- actions: placement --------------------------------------------------
    setWallOffset: (v) =>
      commit((draft) => {
        draft.placement.wallOffsetCm = clamp(
          numOr(v, draft.placement.wallOffsetCm),
          WALL_OFFSET_MIN,
          WALL_OFFSET_MAX,
        )
      }),
    setWindowOffset: (v) =>
      commit((draft) => {
        draft.placement.windowOffsetCm = clamp(
          numOr(v, draft.placement.windowOffsetCm),
          WINDOW_OFFSET_MIN,
          WINDOW_OFFSET_MAX,
        )
      }),
    setGroundToFloor: (on) =>
      commit((draft) => {
        draft.placement.groundToFloor = on === undefined ? !draft.placement.groundToFloor : Boolean(on)
      }),
    /**
     * Where the whole network sits in y. A rigid translation of the finished
     * design — nothing inside it changes shape. Only bites when `groundToFloor`
     * is on: with grounding off there is nothing anchoring the network in y for
     * the offset to be measured from.
     */
    setYOffset: (v) =>
      commit((draft) => {
        draft.placement.yOffsetCm = clamp(
          numOr(v, draft.placement.yOffsetCm),
          Y_OFFSET_MIN,
          Y_OFFSET_MAX,
        )
      }),

    /** The ROOM, not the design. See schema.js's DEFAULT_ROOM: the wall builds up
     *  away from the installation, so this moves no panel. */
    setWallThickness: (v) =>
      commit((draft) => {
        draft.room.wallThicknessCm = clamp(
          numOr(v, draft.room.wallThicknessCm),
          WALL_THICKNESS_MIN,
          WALL_THICKNESS_MAX,
        )
      }),

    // --- actions: connectors -------------------------------------------------
    /**
     * One setter for the whole block rather than seven: the knobs are read
     * together (part count, part types and forced fit are functions of all of
     * them) and nothing here needs a per-knob commit. Clamped on the way in, so
     * a slider drag never commits an invalid config.
     *
     * @param {string} key one of the `config.connectors` fields
     * @param {number|string} value
     */
    setConnectorKnob: (key, value) =>
      commit((draft) => {
        const c = draft.connectors
        switch (key) {
          case 'lengthCm':
            c.lengthCm = clamp(numOr(value, c.lengthCm), CONNECTOR_LENGTH_MIN, CONNECTOR_LENGTH_MAX)
            break
          case 'spacingCm':
            c.spacingCm = clamp(numOr(value, c.spacingCm), CONNECTOR_SPACING_MIN, CONNECTOR_SPACING_MAX)
            break
          case 'minPerJoint':
            c.minPerJoint = clamp(
              Math.round(numOr(value, c.minPerJoint)),
              CONNECTOR_MIN_PER_JOINT_MIN,
              CONNECTOR_MIN_PER_JOINT_MAX,
            )
            break
          case 'binSpanCm':
            c.binSpanCm = clamp(numOr(value, c.binSpanCm), CONNECTOR_BIN_SPAN_MIN, CONNECTOR_BIN_SPAN_MAX)
            break
          case 'binAngleDeg':
            c.binAngleDeg = clamp(numOr(value, c.binAngleDeg), CONNECTOR_BIN_ANGLE_MIN, CONNECTOR_BIN_ANGLE_MAX)
            break
          case 'powerEdge':
            c.powerEdge = oneOf(CONNECTOR_POWER_EDGES, value, c.powerEdge)
            break
          case 'supplyMode':
            c.supplyMode = oneOf(CONNECTOR_SUPPLY_MODES, value, c.supplyMode)
            break
          default:
            break
        }
      }),

    // --- actions: editing the network ---------------------------------------
    /**
     * Remove or restore one flat CELL, taking its ramps with it. This is what
     * makes a ragged edge (V4_SPEC §9.7), and it is the same operation as
     * growing the network: switching a cell ON at a free face lands it on the
     * lattice by construction, so there is no separate "add panel" mechanism.
     *
     * Nothing else moves. The lattice is GENERATED, not chained, so every other
     * panel keeps a bit-identical position — the one exception being grounding,
     * which legitimately shifts the whole network in y if the lowest panel was
     * the one removed.
     */
    toggleCell: (i, j) => {
      const { cols, rows } = get().config.lattice
      // Outside the current rectangle this is an ADD, and the rectangle has to
      // grow to hold it — see `addCellAt`. Routed here so the plan editor has a
      // single click handler and the caller never has to know which it is.
      if (i < 0 || j < 0 || i >= cols || j >= rows) return get().addCellAt(i, j)

      const was = overrideFor(get().config, i, j).present
      const ok = commit((draft) => {
        patchCell(draft, i, j, { present: !was })
        if (was) return
        connectCell(draft, i, j)
      })
      if (ok) {
        set({
          lastActionNotice: was
            ? `cell (${i}, ${j}) removed — its ramps stay, hanging off their far ends`
            : `cell (${i}, ${j}) added, and joined to the neighbours it has — it lands on the ` +
              'lattice, so those joints close exactly',
        })
      }
      return ok
    },

    /**
     * Add a cell OUTSIDE the current rectangle, growing the lattice to hold it.
     *
     * `lattice.cols/rows` is a bounding rectangle, not the design — the design
     * is which cells inside it are present. So growing at an edge must add
     * exactly ONE panel and leave the rest of the new row or column empty,
     * otherwise clicking one slot would silently add five.
     *
     * GROWING AT THE WALL OR WINDOW SIDE RE-ORIGINS THE LATTICE. Cell (0,0) is
     * defined as the corner nearest the wall and window, so adding at i = −1
     * shifts every existing index by one. Two things have to move with it:
     *
     *   - every cell and edge override, or the design would appear to slide one
     *     pitch across the lattice while the panels stayed put;
     *   - `pattern.phase`, because `level = (i + j + phase) mod 2` — shifting an
     *     index by one inverts the whole checkerboard unless the phase absorbs
     *     it. (Phase is 0 or 1, so −phase ≡ +phase mod 2 and a plain flip is
     *     exactly right.)
     *
     * The network is anchored by its own minimum x/z to `wallOffsetCm` /
     * `windowOffsetCm`, so growing at those edges holds the near edge where you
     * put it and moves the rest outward. That is what the offsets MEAN, and the
     * notice says so rather than leaving it to be discovered.
     */
    addCellAt: (i, j) => {
      const { cols, rows } = get().config.lattice
      const shiftI = i < 0 ? -i : 0
      const shiftJ = j < 0 ? -j : 0
      const newCols = Math.max(cols + shiftI, i + shiftI + 1)
      const newRows = Math.max(rows + shiftJ, j + shiftJ + 1)
      if (newCols > LATTICE_COLS_MAX || newRows > LATTICE_ROWS_MAX) {
        set({ lastActionNotice: `the lattice cannot grow past ${LATTICE_COLS_MAX} × ${LATTICE_ROWS_MAX} cells` })
        return false
      }

      const ti = i + shiftI
      const tj = j + shiftJ
      const ok = commit((draft) => {
        draft.lattice.cols = newCols
        draft.lattice.rows = newRows
        if (!draft.overrides || typeof draft.overrides !== 'object') draft.overrides = { cells: [], edges: [] }
        if (!Array.isArray(draft.overrides.cells)) draft.overrides.cells = []
        if (!Array.isArray(draft.overrides.edges)) draft.overrides.edges = []
        for (const c of draft.overrides.cells) { c.i += shiftI; c.j += shiftJ }
        for (const e of draft.overrides.edges) { e.i += shiftI; e.j += shiftJ }
        if ((shiftI + shiftJ) % 2 === 1) draft.pattern.phase = draft.pattern.phase ? 0 : 1

        const wasInside = (a, b) =>
          a - shiftI >= 0 && a - shiftI < cols && b - shiftJ >= 0 && b - shiftJ < rows

        // Every CELL slot the growth created is empty except the one clicked.
        for (let a = 0; a < newCols; a++) {
          for (let b = 0; b < newRows; b++) {
            if (wasInside(a, b) || (a === ti && b === tj)) continue
            draft.overrides.cells.push({ i: a, j: b, present: false, flipped: false })
          }
        }

        // ...AND every RAMP slot it created, unless it touches the clicked cell.
        //
        // This is not tidiness, it is the promise the click makes. A ramp needs
        // only ONE cell (V4_SPEC §9.3), so widening the rectangle by a column
        // would otherwise hang a ramp off EVERY cell of the column next to it —
        // clicking one slot at the edge of a 3 × 5 added seven panels, not one.
        // The new edges that do not touch the target start switched off, and
        // clicking them turns them on like any other panel.
        for (const axis of ['x', 'z']) {
          const iMax = axis === 'x' ? newCols - 1 : newCols
          const jMax = axis === 'z' ? newRows - 1 : newRows
          for (let a = 0; a < iMax; a++) {
            for (let b = 0; b < jMax; b++) {
              const hiA = axis === 'x' ? a + 1 : a
              const hiB = axis === 'z' ? b + 1 : b
              // An edge is OLD only if both its cells were already in the grid.
              if (wasInside(a, b) && wasInside(hiA, hiB)) continue
              const touchesTarget = (a === ti && b === tj) || (hiA === ti && hiB === tj)
              if (touchesTarget) continue
              draft.overrides.edges.push({ i: a, j: b, axis, present: false })
            }
          }
        }
        // If the target already had an "absent" override (it cannot here, but a
        // re-entrant call could), make sure the click still means "add".
        const t = draft.overrides.cells.find((c) => c.i === ti && c.j === tj)
        if (t) t.present = true
        connectCell(draft, ti, tj)
      })

      if (ok) {
        const reOrigin = shiftI || shiftJ
        set({
          lastActionNotice:
            `cell added at the ${i < 0 ? 'wall' : j < 0 ? 'window' : 'far'} edge — the lattice grew to ` +
            `${newCols} × ${newRows}` +
            (reOrigin
              ? ', and re-origined: the near edge stays on its offset, so everything else moved out one pitch'
              : ''),
        })
      }
      return ok
    },

    /**
     * Shrink the rectangle to the cells actually in use.
     *
     * Purely bookkeeping — no panel moves in the world, because the lattice is
     * re-origined and the offsets re-anchor the same near edge. It exists so
     * that trimming an edge away does not leave the plan grid permanently
     * padded with empty slots you cannot get rid of.
     */
    trimLattice: () => {
      const cfg = get().config
      const { cols, rows } = cfg.lattice
      let iMin = Infinity
      let iMax = -Infinity
      let jMin = Infinity
      let jMax = -Infinity
      for (let i = 0; i < cols; i++) {
        for (let j = 0; j < rows; j++) {
          if (!overrideFor(cfg, i, j).present) continue
          if (i < iMin) iMin = i
          if (i > iMax) iMax = i
          if (j < jMin) jMin = j
          if (j > jMax) jMax = j
        }
      }
      if (!Number.isFinite(iMin)) {
        set({ lastActionNotice: 'nothing to trim to — every cell is switched off' })
        return false
      }
      if (iMin === 0 && jMin === 0 && iMax === cols - 1 && jMax === rows - 1) {
        set({ lastActionNotice: 'already trimmed — the rectangle is exactly the cells in use' })
        return true
      }

      const ok = commit((draft) => {
        draft.lattice.cols = iMax - iMin + 1
        draft.lattice.rows = jMax - jMin + 1
        // Same re-origin rule as addCellAt, in the other direction.
        if ((iMin + jMin) % 2 === 1) draft.pattern.phase = draft.pattern.phase ? 0 : 1
        const shift = (list) => list
          .map((o) => ({ ...o, i: o.i - iMin, j: o.j - jMin }))
          .filter((o) => o.i >= 0 && o.j >= 0)
        draft.overrides.cells = shift(draft.overrides.cells)
        draft.overrides.edges = shift(draft.overrides.edges)
      })
      if (ok) {
        set({
          lastActionNotice:
            `trimmed to ${iMax - iMin + 1} × ${jMax - jMin + 1} — empty slots dropped, no panel moved`,
        })
      }
      return ok
    },

    /**
     * Turn one flat cell over. A flip is a PHYSICAL statement about the joint,
     * not a display option: the connector grips the back flange, so a joint
     * whose two panels face opposite ways has its flanges on opposite sides and
     * no part in this family can span it (V4_SPEC §2).
     *
     * On the network that bites harder than it did on the ribbon — a cell has up
     * to four joints and a ramp cannot be flipped, so flipping a cell mismatches
     * EVERY joint it has. The notice says so rather than letting the report be
     * the first place you find out.
     */
    toggleCellFlipped: (i, j) => {
      const was = overrideFor(get().config, i, j).flipped
      const ok = commit((draft) => patchCell(draft, i, j, { flipped: !was }))
      if (ok) {
        set({
          lastActionNotice: was
            ? `cell (${i}, ${j}) turned back the right way up`
            : `cell (${i}, ${j}) flipped — every joint it has now carries no connector, ` +
              'because a ramp cannot be flipped to match it',
        })
      }
      return ok
    },

    /**
     * Remove or restore one RAMP without touching either cell it joins — the way
     * to open the network up rather than cut material out of its edge.
     */
    toggleEdge: (i, j, axis) => {
      if (!EDGE_AXES.includes(axis)) return false
      const was = edgeOverrideFor(get().config, i, j, axis).present
      const ok = commit((draft) => patchEdge(draft, i, j, axis, { present: !was }))
      if (ok) {
        set({
          lastActionNotice: was
            ? `ramp (${i}, ${j}) ${axis} removed — the two cells it joined are now unconnected`
            : `ramp (${i}, ${j}) ${axis} restored`,
        })
      }
      return ok
    },

    /**
     * Edit one obstacle — the room's column and anything like it.
     *
     * An obstacle is a ROOM FACT, never a design choice (obstacles.js), so
     * these setters change what the tool is planning AROUND rather than what it
     * is planning. Nothing here removes a panel: a panel running through the
     * column is reported and outlined, and switching it off is your call.
     */
    setObstacleField: (id, field, value) =>
      commit((draft) => {
        const o = (draft.obstacles ?? []).find((x) => x.id === id)
        if (!o) return
        if (field === 'anchor') {
          o.anchor = OBSTACLE_ANCHORS.includes(value) ? value : o.anchor
          return
        }
        const [lo, hi] =
          field === 'xCm' || field === 'zCm'
            ? [OBSTACLE_POS_MIN, OBSTACLE_POS_MAX]
            : [OBSTACLE_SIZE_MIN, OBSTACLE_SIZE_MAX]
        o[field] = clamp(numOr(value, o[field]), lo, hi)
      }),

    /** Drop every cell and edge override, restoring the full lattice. */
    clearOverrides: () => {
      const before = overrideCount(get().config)
      if (before === 0) {
        set({ lastActionNotice: 'nothing is edited — the lattice is already full' })
        return true
      }
      const ok = commit((draft) => {
        draft.overrides = { cells: [], edges: [] }
      })
      if (ok) {
        set({
          lastActionNotice: `cleared ${before} edit${before === 1 ? '' : 's'} — the lattice is full again`,
        })
      }
      return ok
    },

    // --- actions: whole-config -----------------------------------------------
    /**
     * Replace the config wholesale (the JSON panel's paste/Apply path, and the
     * slot loader's). Normalized, then validated; invalid input is ignored and
     * the errors land in `lastErrors`. A v1/v2/v3 config is REJECTED outright —
     * schema.js never migrates (V4_SPEC §6).
     */
    loadJson: (config) => commitConfig(config),

    /** Reset to the shipped default ribbon. Always commits. */
    resetConfig: () => commitConfig(DEFAULT_CONFIG),

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
      })
      saveWorkingConfig(entry.config)
      return true
    },

    // --- actions: named slots ------------------------------------------------
    // persistence.js is schema-agnostic and checks the embedded `config.version`
    // against `EXPECTED_CONFIG_VERSION = 4`, so a v3 slot is reported unreadable
    // rather than resurrected. Listing slots is a pure read and stays in the
    // panel; anything that WRITES design truth goes through the store.
    /** @returns {boolean} whether the slot was written */
    saveToSlot: (name) => {
      const trimmed = String(name ?? '').trim()
      if (!trimmed) return false
      return saveSlot(trimmed, get().config)
    },
    /** @returns {'ok'|'unreadable'|'invalid'} */
    loadFromSlot: (name) => {
      const loaded = loadSlot(name)
      if (!loaded) return 'unreadable'
      return commitConfig(loaded) ? 'ok' : 'invalid'
    },
    removeSlot: (name) => deleteSlot(name),

    // --- actions: UI-only (never touch `config`, never fail validation) -----
    toggleBounds: (on) => set((s) => ({ showBounds: on === undefined ? !s.showBounds : Boolean(on) })),
    toggleConnectors: (on) =>
      set((s) => ({ showConnectors: on === undefined ? !s.showConnectors : Boolean(on) })),
    toggleOrigin: (on) => set((s) => ({ showOrigin: on === undefined ? !s.showOrigin : Boolean(on) })),

    setColorMode: (mode) => set({ colorMode: mode }),
    setHoveredUnit: (id) => set({ hoveredUnitId: id ?? null }),
    setSelectedUnit: (id) => set((s) => ({ selectedUnitId: s.selectedUnitId === id ? null : (id ?? null) })),
  }
})

export default useStoreV4
