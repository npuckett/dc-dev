/**
 * grid-designer v4 — configuration schema, normalization and validation.
 *
 * HEADLESS ZONE (src/core/): pure functions, importable from plain node.
 *   - explicit `.js` extensions on ALL relative imports
 *   - may import `three` math classes only; never components / store / DOM
 *   - same input → same output, no hidden state
 *
 * =============================================================================
 * WHAT THIS DESCRIBES  (schema v4 — "the folded network")
 * =============================================================================
 * v3 modelled the installation as ONE authored drift surface tiled by rigid
 * panels, and then measured how badly the panels failed to be that surface.
 * V4_SPEC §0 retires that: v4 chooses **folds the connectors can already build**
 * and lets the form be whatever those compose into. So the config no longer
 * describes a surface at all. There is no `form`, no `sheet`, no `tiling`, no
 * tolerances — nothing to reconcile means nothing to set a tolerance on.
 *
 * V4_SPEC §9 then generalises the one-dimensional ribbon to a two-dimensional
 * NETWORK, and it changes what you EDIT without changing any of the geometry:
 * the unit of design stops being a panel in a chain and becomes a **flat cell on
 * a lattice**, with the angled panels derived. So `strip: { count, units }`
 * becomes `lattice: { cols, rows }`, and the flat per-unit override array
 * becomes `{ cells, edges }`.
 *
 * What is left is a very short list, and every entry is a physical decision:
 *
 *   lattice.cols / rows    how many flat cells across x and along z
 *   gap                    the joint width — and in v4 this is an INPUT the
 *                          geometry honours exactly, not an outcome that gets
 *                          measured (V4_SPEC §2's bisector construction, which
 *                          §9.1 collapses to a half-angle step)
 *   angleDeg               θ, the single shape parameter
 *   pattern.phase          which corner of the checkerboard is on the ground
 *   placement.*            where the network sits, and whether it is propped
 *                          against the wall
 *   overrides              per-cell removal / flip, per-edge removal
 *   connectors.*           unchanged from v3 — same knobs, same defaults
 *
 * `version` must be exactly 4. A v1/v2/v3 config is REJECTED, never migrated,
 * because the models describe physically different objects and do not map onto
 * one another.
 *
 * AND SO IS A v4 CONFIG OF THE OLDER, RIBBON SHAPE. `version` did not move for
 * §9 — the model is the same one, generalised — so the version check cannot
 * catch a stale ribbon design sitting in localStorage. `E_LEGACY_SHAPE` does: a
 * config carrying `strip` and no `lattice` is refused outright rather than
 * quietly normalizing into the default 3×5 network, which would look like the
 * tool had silently thrown the user's design away. `persistence.js`'s
 * `EXPECTED_CONFIG_VERSION` stays at 4 and the store's `validateConfig` call is
 * what discards it.
 *
 * =============================================================================
 * TWO KINDS OF DEFAULTING — normalizeConfig vs. validateConfig
 * =============================================================================
 * Carried over verbatim from v3, because it is the contract every consumer in
 * this repository already assumes:
 *
 *   normalizeConfig  fills missing fields AND clamps every numeric knob into its
 *        declared range, so `lattice.js` / `connectors.js` / `report.js` can call
 *        it on anything and get numbers safe to feed a solver. It is IDEMPOTENT
 *        (`normalize(normalize(x))` deep-equals `normalize(x)`) and never
 *        mutates its argument. It never reports.
 *   validateConfig   inspects the RAW input — via the private `withDefaults`,
 *        which only fills entirely-absent structure — and REPORTS what
 *        normalizeConfig would have silently corrected. It never fixes.
 *        Concretely: `validateConfig({ ...good, angleDeg: 999 })` raises
 *        `E_RANGE` even though `normalizeConfig` would hand back 75.
 *
 * EVERY KNOB WITH A DECLARED RANGE IS RANGE-CHECKED HERE. v3's schema clamped
 * `form.angularity` and `form.facetCells` in normalize but forgot to check them
 * in validate, so an out-of-range value was silently corrected instead of
 * reported — HANDOFF §5.2 records that as a real gap and §6 records that a
 * subagent had to find it. The check below is exhaustive against the RANGES
 * table, and `tests/test-v4-schema.mjs` asserts one non-vacuous negative per
 * knob so the exhaustiveness cannot rot.
 *
 * =============================================================================
 * THE OVERRIDES — two sparse tables, cells and edges
 * =============================================================================
 * V4_SPEC §9.7:
 *
 *   overrides: {
 *     cells: [{ i, j, present, flipped }],
 *     edges: [{ i, j, axis, present }],      // axis: 'x' | 'z'
 *   }
 *
 * `i` runs along x FROM THE WALL and `j` along z FROM THE WINDOW, both 0-BASED —
 * unlike the ribbon's 1-based `unit`, because a cell is a lattice coordinate
 * rather than a position in a sequence. An edge is named by its LOW-INDEX cell
 * and its axis: `(i, j, 'x')` joins `(i,j)`–`(i+1,j)`, `(i, j, 'z')` joins
 * `(i,j)`–`(i,j+1)`. Note that is NOT the same as its GROUND cell, which is
 * whichever of the two the checkerboard puts at level 0; lattice.js keeps the
 * two straight and the distinction is deliberate — the name has to be stable
 * under a change of `phase`, and the ground cell is not.
 *
 * There is no `role` override any more. Roles are no longer authored: a cell's
 * level is `(i + j + phase) mod 2` and a ramp's direction follows from it, so a
 * forced role would be a request for a geometry the lattice cannot close
 * (V4_SPEC §9.1). Switching a cell off is what makes a ragged edge; switching an
 * edge off is what opens the network without deleting a flat.
 *
 * normalizeConfig's half of the contract, all three parts of which exist to make
 * its output a deterministic FUNCTION of the design rather than of how the
 * override tables happened to be typed:
 *   - malformed / out-of-grid / duplicate entries are DROPPED (first in array
 *     order wins a contested cell or edge);
 *   - an entry that is ENTIRELY DEFAULT says nothing and is dropped, because
 *     keeping it would let two configs describing the same object fail to
 *     compare equal;
 *   - the survivors are SORTED, so array order carries no information.
 * validateConfig does the opposite on all three and reports
 * `E_OVERRIDE_SHAPE` / `E_OVERRIDE_BOUNDS` / `E_OVERRIDE_CONFLICT`.
 *
 * =============================================================================
 * RANGES — V4_SPEC §6, §9
 * =============================================================================
 *   lattice.cols             1 .. 10    (integer)
 *   lattice.rows             1 .. 10    (integer)
 *   gap                      0.4 .. 8   cm
 *   angleDeg                 0 .. 75    degrees
 *   pattern.phase            0 .. 1     (integer)
 *   placement.wallOffsetCm   0 .. 200   cm
 *   placement.windowOffsetCm 0 .. 200   cm
 *   connectors.lengthCm      4 .. 30    cm
 *   connectors.spacingCm     10 .. 200  cm
 *   connectors.minPerJoint   1 .. 6     (integer)
 *   connectors.binSpanCm     0.05 .. 5  cm
 *   connectors.binAngleDeg   0.5 .. 30  degrees
 *
 * The gap band is not a taste judgement — it is `CONNECTOR_LIMITS` read as a
 * permission. Below 0.4cm there is no room to work a part onto the rims at all;
 * above 8cm the part's spine is a beam being asked to act like a strap. The
 * angle band's ceiling is where the bisector starts to lose meaning (V4_SPEC §2)
 * and where a 60cm panel standing at 75° is already a wall rather than a fold.
 *
 * `pattern.phase` was 0..3 for the ribbon's period-4 wave and is 0..1 here: the
 * level field has PERIOD 2, so 2 and 3 would be indistinguishable from 0 and 1
 * and a config saying `phase: 3` would silently mean something else. Narrowing
 * the band is the honest description of the field that now exists.
 *
 * =============================================================================
 * VALIDATION CODES
 * =============================================================================
 * Errors (valid === false):
 *   E_LEGACY_SHAPE       a v4 config of the RIBBON shape — `strip` present and
 *                        `lattice` absent. Refused, not migrated: see the header.
 *   E_SHAPE              config not an object, `version` not exactly 4, a
 *                        non-integer where an integer is required, a non-finite
 *                        number, a non-boolean `groundToFloor`, or an unknown
 *                        enum value (panelType, pattern.kind, placement.
 *                        wallAnchor, connectors.powerEdge / supplyMode)
 *   E_RANGE              a finite but out-of-band value — see RANGES above
 *   E_OVERRIDE_SHAPE     `overrides` not an object, `overrides.cells` /
 *                        `.edges` not arrays, or an entry that is not an object
 *                        / has a non-integer i or j / a non-boolean present or
 *                        flipped / an unknown axis
 *   E_OVERRIDE_BOUNDS    an override points outside the cell or edge grid
 *   E_OVERRIDE_CONFLICT  two overrides claim the same cell or edge
 * Warnings (do not affect `valid`):
 *   W_SINGLE_CONNECTOR_JOINTS  minPerJoint is 1 — a joint held by one part is a
 *                              hinge, not a fixture
 *   W_POWER_SUPPLY_IGNORED     powerEdge is 'none' — not a physical option, only
 *                              a way to measure what the constraint costs
 *   W_OVERRIDE_NO_OP           an override that is entirely default and will be
 *                              dropped by normalizeConfig
 */

// -----------------------------------------------------------------------------
// Ranges — V4_SPEC §6, §9
// -----------------------------------------------------------------------------
/**
 * How many flat cells across x and along z.
 *
 * The ceiling is a cost decision rather than a physical one: `report.js` finds
 * the connector envelope by bisecting a whole solve 60 times in each direction,
 * so the work grows with the lattice. A 10 × 10 network is 100 cells and 180
 * ramps, which is already a bigger installation than the brief describes.
 */
export const LATTICE_COLS_MIN = 1
export const LATTICE_COLS_MAX = 10
export const LATTICE_ROWS_MIN = 1
export const LATTICE_ROWS_MAX = 10

/**
 * The gap band IS `CONNECTOR_LIMITS.minSpanCm` / `maxSpanCm` (core/v3/
 * connectors.js), restated here as a config range rather than re-derived —
 * v4's whole premise is that a design is built inside the connector envelope,
 * so a gap the part family cannot span is not a design to be flagged later, it
 * is a number the schema should never have accepted. Kept as literals rather
 * than imported so this file has no dependency at all: the two are asserted
 * equal in tests/test-v4-schema.mjs, which is the honest way to keep a
 * deliberate restatement from drifting.
 */
export const GAP_MIN = 0.4
export const GAP_MAX = 8

export const ANGLE_MIN = 0
export const ANGLE_MAX = 75

/** Which corner of the checkerboard is on the ground. Period 2 — see the header. */
export const PHASE_MIN = 0
export const PHASE_MAX = 1

export const WALL_OFFSET_MIN = 0
export const WALL_OFFSET_MAX = 200
export const WINDOW_OFFSET_MIN = 0
export const WINDOW_OFFSET_MAX = 200

// -----------------------------------------------------------------------------
// Connector ranges — carried over from v3 unchanged, and deliberately so: the
// part family did not change, only what is asked of it. Restated rather than
// imported for the same reason as the gap band; test-v4-schema.mjs asserts the
// two files agree.
// -----------------------------------------------------------------------------
/** How far one part runs ALONG the joint. Under ~4cm there is not enough rim to
 *  grip; over ~30cm the part re-acquires the variation it exists to avoid. */
export const CONNECTOR_LENGTH_MIN = 4
export const CONNECTOR_LENGTH_MAX = 30
/** The most unsupported joint allowed between two consecutive parts. */
export const CONNECTOR_SPACING_MIN = 10
export const CONNECTOR_SPACING_MAX = 200
/** The floor on parts per joint. ONE part on a joint is a hinge, not a fixture:
 *  with no substructure, a singly-connected joint is free to rotate about it. */
export const CONNECTOR_MIN_PER_JOINT_MIN = 1
export const CONNECTOR_MIN_PER_JOINT_MAX = 6
/** How coarsely distinct parts are merged into one printable type. */
export const CONNECTOR_BIN_SPAN_MIN = 0.05
export const CONNECTOR_BIN_SPAN_MAX = 5
export const CONNECTOR_BIN_ANGLE_MIN = 0.5
export const CONNECTOR_BIN_ANGLE_MAX = 30

// -----------------------------------------------------------------------------
// Enums
// -----------------------------------------------------------------------------
/**
 * Panel footprints.
 *
 * ONLY '2x2'. The ribbon's schema listed '2x4' as a value it would accept and
 * never build; the lattice cannot even accept it, because §9.1's pitch is the
 * same number in x and z and a 60 × 121 plate does not have one plan size.
 * Plates are §9.8 and will need a rectangular lattice to arrive with them, so
 * listing the value here would be a promise this model cannot keep.
 */
export const PANEL_TYPES = ['2x2']

/**
 * The only pattern kind. The name is the ribbon's, and it is still the right
 * one: a profile cut through the network along either axis reads `_ / - \`,
 * which is what the word describes. It is an enum rather than a boolean because
 * the shape of that profile is exactly the axis a second kind would move along.
 */
export const PATTERN_KINDS = ['trapezoid']

/** Which axis a lattice edge runs along. */
export const EDGE_AXES = ['x', 'z']

/**
 * What holds the network up on the wall side (V4_SPEC §9.4).
 *
 *   'free'    nothing. The wall-side column cantilevers off its own edge.
 *   'braced'  one extra ramp per present HIGH cell at i = 0, running down to
 *             ground level toward the wall. What it attaches to THERE is not
 *             modelled and is an open hardware question — the geometry says
 *             where the toe lands, nothing more.
 */
export const WALL_ANCHORS = ['free', 'braced']

// -----------------------------------------------------------------------------
// OBSTACLES — things in the room, not parts of the design
// -----------------------------------------------------------------------------
/** How `(xCm, zCm)` locates the box — see obstacles.js's header on why this has
 *  to be stated rather than guessed. */
export const OBSTACLE_ANCHORS = ['corner', 'centre']

export const OBSTACLE_POS_MIN = -500
export const OBSTACLE_POS_MAX = 2000
export const OBSTACLE_SIZE_MIN = 1
export const OBSTACLE_SIZE_MAX = 500

/**
 * The room's structural column, measured on site: 380cm in x, 285cm in z from
 * the window/wall corner, 50cm square, floor to ceiling.
 *
 * It ships as a DEFAULT because it is a fact about the space rather than a
 * choice about the design — the same category as the wall plane at x = 0. A
 * design made without it on screen is a design made against the wrong room.
 */
export const DEFAULT_OBSTACLES = [
  {
    id: 'column',
    label: 'column',
    xCm: 380,
    zCm: 285,
    widthCm: 50,
    depthCm: 50,
    heightCm: 300,
    anchor: 'corner',
  },
]

/** Which edge of every panel carries its power supply — a GLOBAL convention.
 *  'none' is not a physical option; it exists to measure what the constraint
 *  costs, the same way v3's `chain` placement showed what exact joints cost. */
export const CONNECTOR_POWER_EDGES = ['low', 'high', 'none']

/** 'relief' — the supply is flush with the flange to within 1mm, so a relief in
 *  the lip clears it and the joint carries a part (which then bears on the
 *  supply housing, reported per station). 'block' — treat it as solid. */
export const CONNECTOR_SUPPLY_MODES = ['relief', 'block']

// -----------------------------------------------------------------------------
// Defaults
// -----------------------------------------------------------------------------
/** Three cells across and five deep: enough to carry every ramp direction, both
 *  corner conditions and a braced column, and small enough to read. */
export const DEFAULT_LATTICE = { cols: 3, rows: 5, panelType: '2x2' }
export const DEFAULT_GAP = 2.0
/** 30° at a 2cm gap sits just inside the envelope — see report.js's
 *  `envelope`, which reports the headroom rather than asserting it here. */
export const DEFAULT_ANGLE_DEG = 30
export const DEFAULT_PATTERN = { kind: 'trapezoid', phase: 0 }
export const DEFAULT_PLACEMENT = {
  wallOffsetCm: 0,
  windowOffsetCm: 0,
  groundToFloor: true,
  wallAnchor: 'free',
}
export const DEFAULT_CONNECTORS = {
  lengthCm: 10,
  spacingCm: 50,
  minPerJoint: 2,
  binSpanCm: 0.5,
  binAngleDeg: 5,
  powerEdge: 'low',
  supplyMode: 'relief',
}

/** The defaults an override is compared against when deciding it says nothing. */
export const DEFAULT_CELL_OVERRIDE = { present: true, flipped: false }
export const DEFAULT_EDGE_OVERRIDE = { present: true }

export const DEFAULT_CONFIG = Object.freeze({
  version: 4,
  name: 'fold study 1',
  lattice: { ...DEFAULT_LATTICE },
  gap: DEFAULT_GAP,
  angleDeg: DEFAULT_ANGLE_DEG,
  pattern: { ...DEFAULT_PATTERN },
  placement: { ...DEFAULT_PLACEMENT },
  // Fresh arrays, not shared references: Object.freeze is shallow and would not
  // stop one config's overrides being mutated out from under every other config
  // that defaulted from this one.
  overrides: { cells: [], edges: [] },
  obstacles: DEFAULT_OBSTACLES.map((o) => ({ ...o })),
  connectors: { ...DEFAULT_CONNECTORS },
  meta: { notes: '' },
})

// -----------------------------------------------------------------------------
// Small helpers
// -----------------------------------------------------------------------------
const isPlainObject = (v) => typeof v === 'object' && v !== null && !Array.isArray(v)
const isFiniteNumber = (v) => typeof v === 'number' && Number.isFinite(v)

export function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v
}

function numberOr(v, fallback) {
  return isFiniteNumber(v) ? v : fallback
}

function clampInt(v, fallback, lo, hi) {
  const n = isFiniteNumber(v) ? Math.round(v) : fallback
  return clamp(n, lo, hi)
}

const oneOf = (list, v, fallback) => (list.includes(v) ? v : fallback)

/**
 * How many edges of each axis a `cols` × `rows` lattice has.
 *
 * Exported because it is the bounds rule three places need to agree on — the
 * override sanitizer, the override validator, and the tests — and writing it
 * three times is how the three drift apart. A 1-column lattice has NO x edges at
 * all, which is the ribbon, and a 1-row lattice has no z edges.
 */
export function edgeGridSize(axis, cols, rows) {
  return axis === 'x' ? { iCount: cols - 1, jCount: rows } : { iCount: cols, jCount: rows - 1 }
}

// -----------------------------------------------------------------------------
// withDefaults — PRIVATE. Fills MISSING structure only; a present-but-invalid
// value passes through verbatim so `validateConfig` can still see and report
// it. Never exported: the public `normalizeConfig` below is the clamping one,
// and nothing downstream should be tempted to treat this output as safe to feed
// a solver.
// -----------------------------------------------------------------------------
function withDefaults(raw) {
  const src = isPlainObject(raw) ? raw : {}
  const latticeSrc = isPlainObject(src.lattice) ? src.lattice : {}
  const patternSrc = isPlainObject(src.pattern) ? src.pattern : {}
  const placementSrc = isPlainObject(src.placement) ? src.placement : {}
  const connectorsSrc = isPlainObject(src.connectors) ? src.connectors : {}
  const overridesSrc = isPlainObject(src.overrides) ? src.overrides : {}

  const pick = (obj, key, fallback) => (obj[key] !== undefined ? obj[key] : fallback)

  const out = {
    version: pick(src, 'version', 4),
    lattice: {
      cols: pick(latticeSrc, 'cols', DEFAULT_LATTICE.cols),
      rows: pick(latticeSrc, 'rows', DEFAULT_LATTICE.rows),
      panelType: pick(latticeSrc, 'panelType', DEFAULT_LATTICE.panelType),
    },
    gap: pick(src, 'gap', DEFAULT_GAP),
    angleDeg: pick(src, 'angleDeg', DEFAULT_ANGLE_DEG),
    pattern: {
      kind: pick(patternSrc, 'kind', DEFAULT_PATTERN.kind),
      phase: pick(patternSrc, 'phase', DEFAULT_PATTERN.phase),
    },
    placement: {
      wallOffsetCm: pick(placementSrc, 'wallOffsetCm', DEFAULT_PLACEMENT.wallOffsetCm),
      windowOffsetCm: pick(placementSrc, 'windowOffsetCm', DEFAULT_PLACEMENT.windowOffsetCm),
      groundToFloor: pick(placementSrc, 'groundToFloor', DEFAULT_PLACEMENT.groundToFloor),
      wallAnchor: pick(placementSrc, 'wallAnchor', DEFAULT_PLACEMENT.wallAnchor),
    },
    // Passed through RAW, even if the tables are not arrays — same "fill missing
    // only" contract as every other field. validateConfig inspects these
    // directly so a malformed entry is reported rather than silently vanishing.
    overrides: {
      cells: pick(overridesSrc, 'cells', []),
      edges: pick(overridesSrc, 'edges', []),
    },
    // Same raw pass-through as the overrides. An ABSENT key defaults to the
    // room's known column; an explicit empty array means "no obstacles", which
    // is a different statement and has to survive normalization.
    obstacles: src.obstacles === undefined
      ? DEFAULT_OBSTACLES.map((o) => ({ ...o }))
      : src.obstacles,
    connectors: {
      lengthCm: pick(connectorsSrc, 'lengthCm', DEFAULT_CONNECTORS.lengthCm),
      spacingCm: pick(connectorsSrc, 'spacingCm', DEFAULT_CONNECTORS.spacingCm),
      minPerJoint: pick(connectorsSrc, 'minPerJoint', DEFAULT_CONNECTORS.minPerJoint),
      binSpanCm: pick(connectorsSrc, 'binSpanCm', DEFAULT_CONNECTORS.binSpanCm),
      binAngleDeg: pick(connectorsSrc, 'binAngleDeg', DEFAULT_CONNECTORS.binAngleDeg),
      powerEdge: pick(connectorsSrc, 'powerEdge', DEFAULT_CONNECTORS.powerEdge),
      supplyMode: pick(connectorsSrc, 'supplyMode', DEFAULT_CONNECTORS.supplyMode),
    },
    meta: { notes: '', ...(isPlainObject(src.meta) ? src.meta : {}) },
  }
  if (src.name !== undefined) out.name = src.name
  return out
}

/**
 * PRIVATE. normalizeConfig's half of the cell-override contract (see the file
 * header): sanitize into something safe to hand `lattice.js` — well-formed, in
 * bounds against the ALREADY-CLAMPED grid, disjoint, no-ops removed, and sorted
 * so array order carries no information.
 */
function sanitizeCells(raw, cols, rows) {
  if (!Array.isArray(raw)) return []
  const claimed = new Set()
  const out = []
  for (const ov of raw) {
    if (!isPlainObject(ov)) continue
    const i = isFiniteNumber(ov.i) ? Math.round(ov.i) : NaN
    const j = isFiniteNumber(ov.j) ? Math.round(ov.j) : NaN
    if (!Number.isInteger(i) || !Number.isInteger(j)) continue
    if (i < 0 || i >= cols || j < 0 || j >= rows) continue
    const key = `${i},${j}`
    if (claimed.has(key)) continue

    const present = ov.present === undefined ? DEFAULT_CELL_OVERRIDE.present : Boolean(ov.present)
    const flipped = ov.flipped === undefined ? DEFAULT_CELL_OVERRIDE.flipped : Boolean(ov.flipped)
    claimed.add(key)
    // An entirely-default override says nothing about the design. Dropping it is
    // what makes normalizeConfig's output a function of the OBJECT rather than
    // of the editing history that produced it — two sessions that arrive at the
    // same network must serialize identically.
    if (present === DEFAULT_CELL_OVERRIDE.present && flipped === DEFAULT_CELL_OVERRIDE.flipped) {
      continue
    }
    out.push({ i, j, present, flipped })
  }
  out.sort((a, b) => a.i - b.i || a.j - b.j)
  return out
}

/** The same, for edges. Bounds depend on the axis — see `edgeGridSize`. */
/**
 * PRIVATE. Obstacles are ROOM FACTS, so unlike the overrides there is nothing to
 * check them against — no grid to be in bounds of, no cell that must exist. A
 * column may perfectly well sit outside the lattice; that is a common and
 * correct state, not an error. So this only clamps to sane magnitudes, fills the
 * defaults, and drops entries that are not objects.
 *
 * Ids are made unique by suffixing rather than by rejecting a duplicate: an id
 * collision is a naming problem, and losing a physical column to one would be a
 * much worse outcome than an awkward label.
 */
function sanitizeObstacles(raw) {
  if (!Array.isArray(raw)) return []
  const seen = new Set()
  const out = []
  raw.forEach((o, k) => {
    if (!isPlainObject(o)) return
    let id = typeof o.id === 'string' && o.id.trim() ? o.id.trim() : `obstacle-${k + 1}`
    let n = 2
    while (seen.has(id)) id = `${id}-${n++}`
    seen.add(id)
    out.push({
      id,
      label: typeof o.label === 'string' && o.label.trim() ? o.label.trim() : id,
      xCm: clamp(numberOr(o.xCm, 0), OBSTACLE_POS_MIN, OBSTACLE_POS_MAX),
      zCm: clamp(numberOr(o.zCm, 0), OBSTACLE_POS_MIN, OBSTACLE_POS_MAX),
      widthCm: clamp(numberOr(o.widthCm, 50), OBSTACLE_SIZE_MIN, OBSTACLE_SIZE_MAX),
      depthCm: clamp(numberOr(o.depthCm, 50), OBSTACLE_SIZE_MIN, OBSTACLE_SIZE_MAX),
      heightCm: clamp(numberOr(o.heightCm, 300), OBSTACLE_SIZE_MIN, 1000),
      anchor: oneOf(OBSTACLE_ANCHORS, o.anchor, 'corner'),
    })
  })
  return out
}

function sanitizeEdges(raw, cols, rows) {
  if (!Array.isArray(raw)) return []
  const claimed = new Set()
  const out = []
  for (const ov of raw) {
    if (!isPlainObject(ov)) continue
    if (!EDGE_AXES.includes(ov.axis)) continue
    const i = isFiniteNumber(ov.i) ? Math.round(ov.i) : NaN
    const j = isFiniteNumber(ov.j) ? Math.round(ov.j) : NaN
    if (!Number.isInteger(i) || !Number.isInteger(j)) continue
    const { iCount, jCount } = edgeGridSize(ov.axis, cols, rows)
    if (i < 0 || i >= iCount || j < 0 || j >= jCount) continue
    const key = `${i},${j},${ov.axis}`
    if (claimed.has(key)) continue

    const present = ov.present === undefined ? DEFAULT_EDGE_OVERRIDE.present : Boolean(ov.present)
    claimed.add(key)
    if (present === DEFAULT_EDGE_OVERRIDE.present) continue
    out.push({ i, j, axis: ov.axis, present })
  }
  out.sort((a, b) => (a.axis < b.axis ? -1 : a.axis > b.axis ? 1 : 0) || a.i - b.i || a.j - b.j)
  return out
}

// -----------------------------------------------------------------------------
// normalizeConfig — PUBLIC. Fill defaults AND clamp. Idempotent, deterministic,
// never mutates `raw`.
// -----------------------------------------------------------------------------
export function normalizeConfig(raw) {
  const cfg = withDefaults(raw)

  // Computed up front because the override sanitizers need the FINAL grid to
  // decide what "in bounds" means.
  const cols = clampInt(cfg.lattice.cols, DEFAULT_LATTICE.cols, LATTICE_COLS_MIN, LATTICE_COLS_MAX)
  const rows = clampInt(cfg.lattice.rows, DEFAULT_LATTICE.rows, LATTICE_ROWS_MIN, LATTICE_ROWS_MAX)

  const out = {
    version: cfg.version !== undefined ? cfg.version : 4,
    lattice: {
      cols,
      rows,
      panelType: oneOf(PANEL_TYPES, cfg.lattice.panelType, DEFAULT_LATTICE.panelType),
    },
    gap: clamp(numberOr(cfg.gap, DEFAULT_GAP), GAP_MIN, GAP_MAX),
    angleDeg: clamp(numberOr(cfg.angleDeg, DEFAULT_ANGLE_DEG), ANGLE_MIN, ANGLE_MAX),
    pattern: {
      kind: oneOf(PATTERN_KINDS, cfg.pattern.kind, DEFAULT_PATTERN.kind),
      // Clamped rather than wrapped, as every other integer knob here clamps.
      phase: clampInt(cfg.pattern.phase, DEFAULT_PATTERN.phase, PHASE_MIN, PHASE_MAX),
    },
    placement: {
      wallOffsetCm: clamp(
        numberOr(cfg.placement.wallOffsetCm, DEFAULT_PLACEMENT.wallOffsetCm),
        WALL_OFFSET_MIN,
        WALL_OFFSET_MAX,
      ),
      windowOffsetCm: clamp(
        numberOr(cfg.placement.windowOffsetCm, DEFAULT_PLACEMENT.windowOffsetCm),
        WINDOW_OFFSET_MIN,
        WINDOW_OFFSET_MAX,
      ),
      groundToFloor: Boolean(cfg.placement.groundToFloor),
      wallAnchor: oneOf(WALL_ANCHORS, cfg.placement.wallAnchor, DEFAULT_PLACEMENT.wallAnchor),
    },
    overrides: {
      cells: sanitizeCells(cfg.overrides.cells, cols, rows),
      edges: sanitizeEdges(cfg.overrides.edges, cols, rows),
    },
    obstacles: sanitizeObstacles(cfg.obstacles),
    connectors: {
      lengthCm: clamp(
        numberOr(cfg.connectors.lengthCm, DEFAULT_CONNECTORS.lengthCm),
        CONNECTOR_LENGTH_MIN,
        CONNECTOR_LENGTH_MAX,
      ),
      spacingCm: clamp(
        numberOr(cfg.connectors.spacingCm, DEFAULT_CONNECTORS.spacingCm),
        CONNECTOR_SPACING_MIN,
        CONNECTOR_SPACING_MAX,
      ),
      // A count of physical parts, so an integer.
      minPerJoint: clampInt(
        cfg.connectors.minPerJoint,
        DEFAULT_CONNECTORS.minPerJoint,
        CONNECTOR_MIN_PER_JOINT_MIN,
        CONNECTOR_MIN_PER_JOINT_MAX,
      ),
      binSpanCm: clamp(
        numberOr(cfg.connectors.binSpanCm, DEFAULT_CONNECTORS.binSpanCm),
        CONNECTOR_BIN_SPAN_MIN,
        CONNECTOR_BIN_SPAN_MAX,
      ),
      binAngleDeg: clamp(
        numberOr(cfg.connectors.binAngleDeg, DEFAULT_CONNECTORS.binAngleDeg),
        CONNECTOR_BIN_ANGLE_MIN,
        CONNECTOR_BIN_ANGLE_MAX,
      ),
      powerEdge: oneOf(CONNECTOR_POWER_EDGES, cfg.connectors.powerEdge, DEFAULT_CONNECTORS.powerEdge),
      supplyMode: oneOf(CONNECTOR_SUPPLY_MODES, cfg.connectors.supplyMode, DEFAULT_CONNECTORS.supplyMode),
    },
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
 * RAW value of every field, so an explicitly out-of-range value is reported even
 * though `normalizeConfig` would have clamped it silently.
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

  // --- the stale RIBBON config ---------------------------------------------
  // Checked on the raw object, before anything is defaulted, and reported alone:
  // every other check below would be answering questions about a 3×5 lattice the
  // user never asked for. `version` cannot catch this — §9 did not move it,
  // because the model is the same model generalised — so this is the only thing
  // standing between a stale localStorage design and a silent reset.
  if (config.strip !== undefined && config.lattice === undefined) {
    err(
      'E_LEGACY_SHAPE',
      'this is a v4 config of the earlier RIBBON shape (it has `strip` and no `lattice`). The model ' +
        'generalised to a 2-D lattice in V4_SPEC §9 without changing `version`, so it is refused here ' +
        'rather than normalized into a default network — a design silently replaced is worse than one ' +
        'reported as unreadable. Rebuild it as `lattice: { cols, rows }` with `overrides: { cells, edges }`',
      'strip',
    )
    return finish(errors, warnings)
  }

  const cfg = withDefaults(config)

  // --- version — v1/v2/v3 are REJECTED, never migrated (see file header) ----
  if (cfg.version !== 4) {
    err(
      'E_SHAPE',
      `version must be exactly 4 (got ${JSON.stringify(cfg.version)}) — v3 (one tiled drift surface) ` +
        `and earlier configs are not supported: those models do not map onto v4's folded network, so a ` +
        `stale config is REJECTED here, never silently reinterpreted`,
      'version',
    )
  }

  // --- a range check that reports rather than clamps ------------------------
  // One helper for every ranged knob, so "did we check them all" is answerable
  // by reading the call list rather than by auditing branches. HANDOFF §5.2 is
  // the record of what happens when that list is incomplete.
  const checkRange = (path, v, lo, hi, label, { integer = false } = {}) => {
    if (!isFiniteNumber(v)) {
      err('E_SHAPE', `${path} must be a number (got ${JSON.stringify(v)})`, path)
      return false
    }
    if (integer && !Number.isInteger(v)) {
      err('E_SHAPE', `${path} must be an integer — ${label} (got ${v})`, path)
      return false
    }
    if (v < lo || v > hi) {
      err('E_RANGE', `${path} (${label}) must be in ${lo}..${hi} (got ${v})`, path)
      return false
    }
    return true
  }

  const colsOk = checkRange('lattice.cols', cfg.lattice.cols, LATTICE_COLS_MIN, LATTICE_COLS_MAX,
    'flat cells across x, from the wall', { integer: true })
  const rowsOk = checkRange('lattice.rows', cfg.lattice.rows, LATTICE_ROWS_MIN, LATTICE_ROWS_MAX,
    'flat cells along z, from the window', { integer: true })
  checkRange('gap', cfg.gap, GAP_MIN, GAP_MAX, 'joint width, cm')
  checkRange('angleDeg', cfg.angleDeg, ANGLE_MIN, ANGLE_MAX, 'θ, the single shape parameter, degrees')
  checkRange('pattern.phase', cfg.pattern.phase, PHASE_MIN, PHASE_MAX,
    'which corner of the checkerboard is on the ground', { integer: true })
  checkRange('placement.wallOffsetCm', cfg.placement.wallOffsetCm, WALL_OFFSET_MIN, WALL_OFFSET_MAX,
    'nearest material to the wall plane x = 0, cm')
  checkRange('placement.windowOffsetCm', cfg.placement.windowOffsetCm, WINDOW_OFFSET_MIN, WINDOW_OFFSET_MAX,
    'nearest material to the window line z = 0, cm')

  // --- enums ----------------------------------------------------------------
  const checkEnum = (path, v, list, label) => {
    if (!list.includes(v)) {
      err(
        'E_SHAPE',
        `${path} must be one of ${list.map((s) => `"${s}"`).join(', ')} — ${label} ` +
          `(got ${JSON.stringify(v)})`,
        path,
      )
    }
  }
  checkEnum('lattice.panelType', cfg.lattice.panelType, PANEL_TYPES, 'panel footprint')
  checkEnum('pattern.kind', cfg.pattern.kind, PATTERN_KINDS, 'the fold pattern')
  checkEnum('placement.wallAnchor', cfg.placement.wallAnchor, WALL_ANCHORS,
    'what holds the network up on the wall side')

  // --- groundToFloor: a switch, not a truthiness test -----------------------
  // normalizeConfig coerces with Boolean(); this reports, because
  // `groundToFloor: 'no'` is almost certainly a mistake and coercing it to true
  // silently is exactly the failure mode the two-kinds-of-defaulting split
  // exists to prevent.
  if (typeof cfg.placement.groundToFloor !== 'boolean') {
    err(
      'E_SHAPE',
      `placement.groundToFloor must be a boolean (got ${JSON.stringify(cfg.placement.groundToFloor)})`,
      'placement.groundToFloor',
    )
  }

  // --- overrides — reported on the RAW tables, never silently dropped -------
  {
    if (config.overrides !== undefined && !isPlainObject(config.overrides)) {
      err(
        'E_OVERRIDE_SHAPE',
        `overrides must be an object with "cells" and "edges" arrays (got ` +
          `${JSON.stringify(config.overrides)}) — the ribbon's flat array is not this shape`,
        'overrides',
      )
    }
    // Bounds only mean something once the grid itself is sane; an out-of-range
    // lattice.cols/rows already raised its own error above.
    const boundsKnown = colsOk && rowsOk

    // --- cells ---
    const cells = cfg.overrides.cells
    if (!Array.isArray(cells)) {
      err('E_OVERRIDE_SHAPE', `overrides.cells must be an array (got ${JSON.stringify(cells)})`,
        'overrides.cells')
    } else {
      const claimedBy = new Map()
      cells.forEach((ov, idx) => {
        const path = `overrides.cells[${idx}]`
        if (!isPlainObject(ov)) {
          err('E_OVERRIDE_SHAPE', `${path} must be an object (got ${JSON.stringify(ov)})`, path)
          return
        }
        const iOk = isFiniteNumber(ov.i) && Number.isInteger(ov.i)
        const jOk = isFiniteNumber(ov.j) && Number.isInteger(ov.j)
        if (!iOk) {
          err('E_OVERRIDE_SHAPE',
            `${path}.i must be a 0-based integer along x (got ${JSON.stringify(ov.i)})`, `${path}.i`)
        }
        if (!jOk) {
          err('E_OVERRIDE_SHAPE',
            `${path}.j must be a 0-based integer along z (got ${JSON.stringify(ov.j)})`, `${path}.j`)
        }
        if (ov.present !== undefined && typeof ov.present !== 'boolean') {
          err('E_OVERRIDE_SHAPE',
            `${path}.present must be a boolean (got ${JSON.stringify(ov.present)})`, `${path}.present`)
        }
        if (ov.flipped !== undefined && typeof ov.flipped !== 'boolean') {
          err('E_OVERRIDE_SHAPE',
            `${path}.flipped must be a boolean (got ${JSON.stringify(ov.flipped)})`, `${path}.flipped`)
        }
        if (!iOk || !jOk) return // a malformed key has no cell to reason about further

        if (boundsKnown) {
          if (ov.i < 0 || ov.i >= cfg.lattice.cols || ov.j < 0 || ov.j >= cfg.lattice.rows) {
            err(
              'E_OVERRIDE_BOUNDS',
              `${path} — cell (${ov.i}, ${ov.j}) is outside the ${cfg.lattice.cols} × ` +
                `${cfg.lattice.rows} lattice (i 0..${cfg.lattice.cols - 1}, j 0..${cfg.lattice.rows - 1})`,
              path,
            )
            return // a cell that does not exist cannot also be reported as contested
          }
        }

        const key = `${ov.i},${ov.j}`
        if (claimedBy.has(key)) {
          err('E_OVERRIDE_CONFLICT',
            `${path} and overrides.cells[${claimedBy.get(key)}] both claim cell (${ov.i}, ${ov.j})`, path)
        } else {
          claimedBy.set(key, idx)
        }

        const present = ov.present === undefined ? DEFAULT_CELL_OVERRIDE.present : ov.present
        const flipped = ov.flipped === undefined ? DEFAULT_CELL_OVERRIDE.flipped : ov.flipped
        if (present === true && flipped === false) {
          warn(
            'W_OVERRIDE_NO_OP',
            `${path} is entirely default (present, not flipped) — it says nothing about the design ` +
              'and normalizeConfig will drop it',
            path,
          )
        }
      })
    }

    // --- edges ---
    const edges = cfg.overrides.edges
    if (!Array.isArray(edges)) {
      err('E_OVERRIDE_SHAPE', `overrides.edges must be an array (got ${JSON.stringify(edges)})`,
        'overrides.edges')
    } else {
      const claimedBy = new Map()
      edges.forEach((ov, idx) => {
        const path = `overrides.edges[${idx}]`
        if (!isPlainObject(ov)) {
          err('E_OVERRIDE_SHAPE', `${path} must be an object (got ${JSON.stringify(ov)})`, path)
          return
        }
        const iOk = isFiniteNumber(ov.i) && Number.isInteger(ov.i)
        const jOk = isFiniteNumber(ov.j) && Number.isInteger(ov.j)
        const axisOk = EDGE_AXES.includes(ov.axis)
        if (!iOk) {
          err('E_OVERRIDE_SHAPE',
            `${path}.i must be a 0-based integer (got ${JSON.stringify(ov.i)})`, `${path}.i`)
        }
        if (!jOk) {
          err('E_OVERRIDE_SHAPE',
            `${path}.j must be a 0-based integer (got ${JSON.stringify(ov.j)})`, `${path}.j`)
        }
        if (!axisOk) {
          err(
            'E_OVERRIDE_SHAPE',
            `${path}.axis must be one of ${EDGE_AXES.map((s) => `"${s}"`).join(', ')} — an edge is ` +
              `named by its low-index cell and the axis it crosses (got ${JSON.stringify(ov.axis)})`,
            `${path}.axis`,
          )
        }
        if (ov.present !== undefined && typeof ov.present !== 'boolean') {
          err('E_OVERRIDE_SHAPE',
            `${path}.present must be a boolean (got ${JSON.stringify(ov.present)})`, `${path}.present`)
        }
        if (!iOk || !jOk || !axisOk) return

        if (boundsKnown) {
          const { iCount, jCount } = edgeGridSize(ov.axis, cfg.lattice.cols, cfg.lattice.rows)
          if (ov.i < 0 || ov.i >= iCount || ov.j < 0 || ov.j >= jCount) {
            err(
              'E_OVERRIDE_BOUNDS',
              `${path} — edge (${ov.i}, ${ov.j}, "${ov.axis}") is outside the ${cfg.lattice.cols} × ` +
                `${cfg.lattice.rows} lattice, which has ${Math.max(0, iCount)} × ${Math.max(0, jCount)} ` +
                `"${ov.axis}" edges`,
              path,
            )
            return
          }
        }

        const key = `${ov.i},${ov.j},${ov.axis}`
        if (claimedBy.has(key)) {
          err('E_OVERRIDE_CONFLICT',
            `${path} and overrides.edges[${claimedBy.get(key)}] both claim edge (${ov.i}, ${ov.j}, ` +
              `"${ov.axis}")`, path)
        } else {
          claimedBy.set(key, idx)
        }

        const present = ov.present === undefined ? DEFAULT_EDGE_OVERRIDE.present : ov.present
        if (present === true) {
          warn(
            'W_OVERRIDE_NO_OP',
            `${path} is entirely default (present) — it says nothing about the design and ` +
              'normalizeConfig will drop it',
            path,
          )
        }
      })
    }
  }

  // --- connectors — same ranges as v3, checked on the RAW input -------------
  {
    const conn = cfg.connectors
    checkRange('connectors.lengthCm', conn.lengthCm, CONNECTOR_LENGTH_MIN, CONNECTOR_LENGTH_MAX,
      'part length along the joint, cm')
    checkRange('connectors.spacingCm', conn.spacingCm, CONNECTOR_SPACING_MIN, CONNECTOR_SPACING_MAX,
      'max unsupported joint between parts, cm')
    checkRange('connectors.minPerJoint', conn.minPerJoint,
      CONNECTOR_MIN_PER_JOINT_MIN, CONNECTOR_MIN_PER_JOINT_MAX,
      'parts per joint floor — a count of physical parts', { integer: true })
    checkRange('connectors.binSpanCm', conn.binSpanCm, CONNECTOR_BIN_SPAN_MIN, CONNECTOR_BIN_SPAN_MAX,
      'part-type span bin, cm')
    checkRange('connectors.binAngleDeg', conn.binAngleDeg, CONNECTOR_BIN_ANGLE_MIN, CONNECTOR_BIN_ANGLE_MAX,
      'part-type angle bin, degrees')
    checkEnum('connectors.powerEdge', conn.powerEdge, CONNECTOR_POWER_EDGES, 'which edge carries the supply')
    checkEnum('connectors.supplyMode', conn.supplyMode, CONNECTOR_SUPPLY_MODES, 'how the supply is treated')

    if (conn.powerEdge === 'none') {
      warn(
        'W_POWER_SUPPLY_IGNORED',
        'connectors.powerEdge is "none" — the panels\' power supplies are being ignored, so connector ' +
          'placement will not reflect what can physically be fitted',
        'connectors.powerEdge',
      )
    }
    if (conn.minPerJoint === 1) {
      warn(
        'W_SINGLE_CONNECTOR_JOINTS',
        'connectors.minPerJoint is 1 — a joint held by one part is free to rotate about it. ' +
          'With no substructure, two per joint is what makes a joint rigid',
        'connectors.minPerJoint',
      )
    }
  }

  return finish(errors, warnings)
}

function finish(errors, warnings) {
  return { valid: errors.length === 0, errors, warnings }
}
