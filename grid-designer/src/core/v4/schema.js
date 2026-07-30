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
 *   pattern.wave.scrunchX/Z  0 .. 0.9   (present only when kind is 'wave')
 *   pattern.wave.attractorX/Z 0 .. 1    (ditto)
 *   placement.wallOffsetCm   0 .. 200   cm
 *   placement.windowOffsetCm 0 .. 200   cm
 *   placement.yOffsetCm      -100 .. 400  cm
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
 *   W_WAVE_SETTINGS_IGNORED    `pattern.wave` is set while the kind is not
 *                              'wave' — the block does nothing and will be
 *                              dropped
 *   W_PHASE_IGNORED            `pattern.phase` is non-zero under 'wave', where
 *                              there is no checkerboard to shift
 *
 * =============================================================================
 * THE WAVE BLOCK IS CONDITIONAL, AND THAT IS DELIBERATE
 * =============================================================================
 * `pattern.wave` appears in `normalizeConfig`'s output ONLY when
 * `pattern.kind === 'wave'`. It is the same rule the overrides follow — a field
 * that says nothing about the design is not written down — and here it has a
 * second, harder job: every checkerboard design that already exists must
 * serialize, and SOLVE, byte for byte as it did before the wave was built.
 * Emitting four dead numbers into every trapezoid config would break that on the
 * first save. `tests/test-v4-wave.mjs` §8 is the check.
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

/**
 * Where the whole network sits in y. With `groundToFloor` on it is measured from
 * the network's own lowest present material, which lands at this height; with
 * grounding off it is inert, because there is nothing anchoring the design in y
 * for an offset to be measured from.
 *
 * The band is the ROOM, not a part. It was 0..50 when this number was the height
 * of a spacer standing under a flat cell, and that part is gone (§9.12). What
 * bounds it now is the wall the design is hung against: the mullions run to
 * 375cm, so a design has to be placeable anywhere on them, and 400 clears the top
 * of the tallest one with the deepest network still under it. The floor is
 * negative on purpose — a design can be sunk below y = 0 to sit in a well or to
 * be read from underneath, and −100 is about as far as that stays a building.
 */
export const Y_OFFSET_MIN = -100
export const Y_OFFSET_MAX = 400

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
 * The fold patterns.
 *
 *   'trapezoid'  the two-level checkerboard of §9.1. `level = (i+j+phase) mod 2`,
 *        one angle everywhere. THE DEFAULT, and frozen: designs exist in it and
 *        `tests/test-v4-wave.mjs` §8 asserts its solve is byte-identical to the
 *        commit that introduced the wave.
 *   'wave'       the separable field `h(i,j) = f(i) + g(j)`, with a per-edge
 *        angle that steepens toward an attractor (`wave.js`). It is a DIFFERENT
 *        SHAPE, not a parameterisation of the trapezoid — at zero scrunch it is
 *        a uniform egg-crate with three levels, not a checkerboard with two —
 *        because the checkerboard provably cannot carry a varying angle at all.
 *        wave.js's header has the proof and its measured residuals.
 *
 * An enum rather than a boolean because it selects a MODEL: `pattern.phase`
 * means nothing under 'wave' (there is no checkerboard to shift) and
 * `pattern.wave` means nothing under 'trapezoid'.
 */
export const PATTERN_KINDS = ['trapezoid', 'wave']

// -----------------------------------------------------------------------------
// The wave's four knobs — V4_SPEC §9.14
// -----------------------------------------------------------------------------
/**
 * How much of the base plan advance the far end of an axis gives up.
 *
 * The ceiling is 0.9 rather than 1 because 1 would ask for a zero-length plan
 * advance, which is a 90° ramp — outside `ANGLE_MAX` and outside the connector
 * envelope by a wide margin. 0.9 is already well past what the angle band can
 * deliver at any realistic gap (at 30°/2cm the steepest edge `ANGLE_MAX` allows
 * is a 66.5% scrunch), so the band's top is a place the solver REPORTS from —
 * `W_SCRUNCH_UNREACHABLE`, naming the edge — rather than a promise it keeps.
 */
export const SCRUNCH_MIN = 0
export const SCRUNCH_MAX = 0.9

/**
 * Where along the axis full scrunch is reached, as a fraction of the edge run.
 *
 * 1 spreads the compression over the whole axis, which is the plain reading of
 * "compress in a simple linear fashion" and the default. 0.5 finishes it half way
 * and leaves the far half uniformly tight. 0 is the degenerate end — everything
 * is already past the attractor, so the whole axis compresses uniformly — and it
 * is kept in the band because it is a real thing to ask for, with the caveat that
 * it is the one setting where edge 0 does NOT sit at the base angle.
 */
export const ATTRACTOR_MIN = 0
export const ATTRACTOR_MAX = 1

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
// -----------------------------------------------------------------------------
// THE ROOM — measured fabric of the space, not part of the design
// -----------------------------------------------------------------------------
/**
 * Dimensions of the space the installation goes into, measured on site.
 *
 * Same category as `obstacles` and for the same reason: these are FACTS about
 * the room, so they ship as defaults rather than as choices, and nothing here
 * ever changes the design — they exist to place it against something real.
 *
 * The wall plane stays at **x = 0**, which is the datum every other dimension in
 * the tool is quoted from. `wallThicknessCm` is the wall's build-up running
 * AWAY from the installation, so the wall occupies `x ∈ [−wallThicknessCm, 0]`
 * and the network still starts at x = 0. Thickening the wall therefore never
 * moves a panel — if it ever does, something has confused the room with the
 * design.
 */
export const WALL_THICKNESS_MIN = 0
export const WALL_THICKNESS_MAX = 100

export const DEFAULT_ROOM = {
  /** Measured on site, 2026-07-28. */
  wallThicknessCm: 8.9,
}

export const OBSTACLE_ANCHORS = ['corner', 'centre']

/**
 * What an obstacle IS, which decides how it draws — not how it is tested.
 *
 *   'solid'  material standing in the room: a column, a duct. Drawn opaque.
 *   'zone'   RESERVED EMPTY SPACE the installation must keep out of: a heating
 *            gap, a service run, a maintenance swing. Drawn as a translucent
 *            volume, because a keep-out that looks like a wall reads as
 *            something you could bolt to.
 *   'glass'  material, but see-through. Separate from 'solid' ONLY so a 5.9m
 *            sheet standing between the camera and the design does not black
 *            the whole thing out.
 *
 * Both are tested identically — the design must not enter either — so this
 * never touches `solveObstacles`.
 */
export const OBSTACLE_KINDS = ['solid', 'zone', 'glass']

export const OBSTACLE_POS_MIN = -500
export const OBSTACLE_POS_MAX = 2000
/** How far an obstacle's base may sit above or BELOW the floor. Negative is
 *  real and routine: a mullion runs down past floor level into the build-up. */
export const OBSTACLE_BASE_Y_MIN = -500
export const OBSTACLE_BASE_Y_MAX = 2000

/**
 * Thin enough for glazing. This was 1, sized for columns, and it silently
 * clamped the 0.5cm glass to 1.0 — the second time this pair of bounds has
 * quietly rewritten a measurement (OBSTACLE_SIZE_MAX did it to the 594cm
 * heating run). Both failures produce a plausible-looking number, which is why
 * the extents are checked against hand arithmetic rather than read back.
 */
export const OBSTACLE_SIZE_MIN = 0.1
/**
 * Room-scale, not column-scale. This was 500, which silently CLAMPED the 594cm
 * heating run to 500 and put its far end 94cm short — a wrong number that looks
 * entirely plausible, which is the worst kind. An obstacle can legitimately span
 * a whole elevation, so the ceiling is now the room's own scale.
 */
export const OBSTACLE_SIZE_MAX = 2000

/**
 * The room's structural column, measured on site: 380cm in x, 285cm in z from
 * the window/wall corner, 50cm square, floor to ceiling.
 *
 * It ships as a DEFAULT because it is a fact about the space rather than a
 * choice about the design — the same category as the wall plane at x = 0. A
 * design made without it on screen is a design made against the wrong room.
 */
/**
 * THE WINDOW MULLIONS, measured on site 2026-07-28.
 *
 * All the same section and the same height; what differs is where they sit
 * along x. The measurement taken on site was CENTRE TO CENTRE, so that is what
 * is written here and the min corners are DERIVED — transcribing five
 * hand-computed corners would put the arithmetic in a place no test can see,
 * and a 3.175cm half-width slip would look entirely plausible in the result.
 *
 * Mullion 1 is the anchor: its x,z min corner sits ON the heating gap's own
 * (−81.3, −59.7) datum, so the two share that edge exactly. Everything else
 * steps from its CENTRE.
 *
 * A useful consistency check that fell out of this, not designed in: mullion 5
 * ends at x = 511.75 and the heating run ends at 512.7. Two independent site
 * measurements landing 0.95cm apart is the kind of agreement that says both
 * were read correctly.
 */
const CORNER_X_CM = -81.3
const CORNER_Z_CM = -59.7
const MULLION_SECTION = { acrossCm: 6.35, depthCm: 19.05, heightCm: 400, baseYCm: -25 }
/** Centre-to-centre, walking away FROM THE CORNER. Used on both facades. */
const MULLION_SPACINGS_CM = [129.5, 152.4, 152.4, 152.4]

const SILL_THICKNESS_CM = 5
const SILL_TOP_Y_CM = MULLION_SECTION.baseYCm
const SIDEWALK_DROP_CM = 40
const GLASS_THICKNESS_CM = 0.5
const CAP_THICKNESS_CM = 1

const SIDEWALK_Y_CM = SILL_TOP_Y_CM - SIDEWALK_DROP_CM
/** How far a facade runs from the corner: last mullion's far face. Derived, so
 *  a change to the spacings moves the heating run with the glazing. */
const FACADE_RUN_CM = MULLION_SPACINGS_CM.reduce((a, b) => a + b, 0) + MULLION_SECTION.acrossCm
/** The return elevation's inner face — where the floor of that bay begins. */
const RETURN_INNER_X_CM = CORNER_X_CM + MULLION_SECTION.depthCm
const MULLION_TOP_Y_CM = MULLION_SECTION.baseYCm + MULLION_SECTION.heightCm

/**
 * THE WINDOW, AND IT TURNS A CORNER.
 *
 * This is the corner of the building. The glazing runs along x, reaches the
 * corner at (−81.3, −59.7), and turns 90° to run along z up the returning
 * elevation — same section, same glass detail, same height off the sidewalk,
 * and the same spacing pattern measured out from the corner.
 *
 * So the two facades are ONE description with an axis swapped, not two lists.
 * Writing them out twice would mean every later correction had to be made in
 * two places and would be made in one.
 *
 *   'x'  the main elevation. Outer face at z = −59.7, depth running INWARD
 *        (+z, toward the room). Mullions step along x from the corner.
 *   'z'  the return. Outer face at x = −81.3, depth running inward (+x).
 *        Mullions step along z from the corner.
 *
 * MULLION_SECTION is stated as `acrossCm` (6.35, the face width) and `depthCm`
 * (19.05, how far it reaches back) rather than width/depth, because which world
 * axis each maps to is exactly what the turn changes. Naming them x and z here
 * is what would make the return facade come out 6.35 deep and 19.05 wide — a
 * mullion lying on its side, which reads as plausible in a plan view and is
 * completely wrong in section.
 *
 * y is identical on both: sill −30 → −25, glass and mullions −25 → 375, caps
 * and sidewalk down to −65. The corner does not change any height.
 *
 * y = 0 IS THE FLOOR THE PANELS STAND ON — the surface visible in the site
 * photo. The deep ledge it looks like from the street is the room floor, not a
 * sill; reading it as a sill would put a solid exactly where the installation
 * sits.
 *
 * THE CORNER POST IS SHARED, and appears in both facades' lists. Their first
 * mullions overlap in a 6.35 × 6.35 column at the corner — that overlap IS the
 * post, described twice. Harmless here (obstacles are context, and the design
 * is tested against each independently) but do not read the count as a part
 * count.
 *
 * STILL NOT MEASURED: the sill's depth, set to the mullion footprint — the
 * minimal claim, enough to carry the glass and the mullions and nothing more.
 * If it oversails toward the room it can reach the network.
 */
function facade(axis) {
  const along = axis === 'x' ? 'x' : 'z'
  const corner = along === 'x' ? CORNER_X_CM : CORNER_Z_CM
  const outer = along === 'x' ? CORNER_Z_CM : CORNER_X_CM
  const { acrossCm, depthCm } = MULLION_SECTION

  // Lay the box out in (along, outward) terms, then map onto world x/z once.
  // Every element goes through this, so the turn cannot be got right for the
  // mullions and wrong for the glass.
  const place = (id, label, kind, alongStart, alongLen, outStart, outLen, baseY, height, labelled = true) => ({
    id, label, kind, labelled,
    xCm: along === 'x' ? alongStart : outStart,
    zCm: along === 'x' ? outStart : alongStart,
    widthCm: along === 'x' ? alongLen : outLen,
    depthCm: along === 'x' ? outLen : alongLen,
    heightCm: height,
    baseYCm: baseY,
    anchor: 'corner',
  })

  const centres = []
  let c = corner + acrossCm / 2
  centres.push(c)
  for (const step of MULLION_SPACINGS_CM) {
    c += step
    centres.push(c)
  }
  const round = (v) => Math.round(v * 1e9) / 1e9
  const starts = centres.map((v) => round(v - acrossCm / 2))
  const runLen = round(starts[starts.length - 1] + acrossCm - corner)
  const tag = along === 'x' ? '' : '-return'
  // Appended, so a label reads "sill (sidewalk) — return" rather than growing a
  // second bracket. The main facade's labels stay exactly as they were.
  const suffix = along === 'x' ? '' : ' — return'

  const out = []
  starts.forEach((a, k) => {
    out.push(place(`mullion-${k + 1}${tag}`, `mullion ${k + 1}${suffix}`, 'solid',
      a, acrossCm, outer, depthCm, MULLION_SECTION.baseYCm, MULLION_SECTION.heightCm))
  })
  out.push(place(`sill${tag}`, `sill${suffix}`, 'solid',
    corner, runLen, outer, depthCm, SILL_TOP_Y_CM - SILL_THICKNESS_CM, SILL_THICKNESS_CM))
  out.push(place(`sill-sidewalk${tag}`, `sill (sidewalk)${suffix}`, 'solid',
    corner, runLen, outer, depthCm, SIDEWALK_Y_CM - SILL_THICKNESS_CM, SILL_THICKNESS_CM))
  // Flush to the STREET face, so the mullion depth reads from inside.
  out.push(place(`glass${tag}`, `glass${suffix}`, 'glass',
    corner, runLen, outer, GLASS_THICKNESS_CM, SILL_TOP_Y_CM, MULLION_TOP_Y_CM - SILL_TOP_Y_CM))
  // One cap per mullion, just OUTSIDE the glazing plane, down to the sidewalk.
  starts.forEach((a, k) => {
    out.push(place(`mullion-${k + 1}${tag}-cap`, `mullion ${k + 1}${suffix} cap`, 'solid',
      a, acrossCm, round(outer - CAP_THICKNESS_CM), CAP_THICKNESS_CM,
      SIDEWALK_Y_CM, MULLION_TOP_Y_CM - SIDEWALK_Y_CM, false))
  })
  return out
}

/**
 * THE STAIRCASE.
 *
 * A switchback in the lobby: up in −Z, across the landing in −X, up again in
 * +Z, with flight 2 beside flight 1 rather than above it. White solid fascia,
 * glass balustrade — see the two site photos.
 *
 * WHAT IS ANCHORED (stated on site, not inferred):
 *   · the bottom of flight 1 is at SIDEWALK LEVEL, y = −65. There is a lower
 *     interior floor at the same height as the pavement outside.
 *   · the landing runs parallel to the front window, along x, and reaches −X as
 *     far as mullion 1.
 *   · flight 1's close edge is at mullion 2's x.
 *   · both flights are the same width.
 *
 * WHAT THAT FORCES. Those last two only reconcile one way: if the flights sit
 * side by side, flight 1 starting at M2 and flight 2 ending at M1, each flight
 * is exactly the M1→M2 spacing wide — 129.5 — and the landing spans both, 259
 * long. The width is therefore DERIVED from two measured alignments rather than
 * guessed, which is the strongest thing about this model.
 *
 * WHAT IS ASSUMED, and it is a lot:
 *   · FLOOR TO FLOOR = 400, chosen to match the mullion height exactly. The
 *     user did not know it. Every level below hangs off this one number: change
 *     it and the landing and upper floor both move.
 *   · 12 risers of 16.67 and 11 treads of 28 per flight — a 200 rise and a 308
 *     run, comfortably inside commercial range (riser 15–18, tread ≥ 28).
 *   · the balustrade is 110 above the nosing, 1.2 thick.
 *
 * WHAT IS ESTIMATED FROM THE PHOTOS, and is the weakest number here: the
 * landing's Z BAND, 285 → 435. The landing sits well behind the facade plane
 * the photos were calibrated on, so anything scaled off the mullions
 * understates it. Treat as ±40.
 *
 * THE BALUSTRADES FOLLOW THE STEPS, one segment per tread, rather than being a
 * single slab over the whole rake. The first cut used the bounding slab, on the
 * theory that claiming too much space is the safe direction for a keep-out. It
 * is not: that slab spanned y −48 → 245 over the flight's whole z run and
 * REPORTED A CLASH WITH THE NETWORK THAT DOES NOT EXIST. Where the network
 * actually reaches (z 523) the real rail is at y 66 → 176, and the network tops
 * out at 35 — 31cm of clearance, called a collision.
 *
 * A placement tool that cries wolf gets its warnings ignored, which costs more
 * than the space the approximation saved. Stepping the glass makes it accurate
 * to one riser, and the false positive goes away.
 *
 * The space UNDER each flight is left open, because it is: image 2 has a
 * bicycle parked under the lower run.
 */
const STAIR_LOWER_Y_CM = SIDEWALK_Y_CM
/**
 * THE LANDING'S HEIGHT IS DERIVED FROM THE STAIR, not measured.
 *
 * It was given as 215, but as an ESTIMATE. The riser is the exact quantity: a
 * standard 17.5, and FIFTEEN of them off the sidewalk floor at −65 puts the
 * landing at 197.5 — 17.5 under the estimate, i.e. exactly one riser, which is
 * the size of discrepancy an eyeballed height produces.
 *
 * Deriving it this way means the landing IS flight 1's top tread rather than a
 * surface above it: the flight climbs 15 risers, the fifteenth arrives at the
 * landing, and there is no further step. Sixteen risers put a redundant tread
 * at the landing level and then had to be given somewhere to go.
 */
const STAIR_RISER_CM = 17.5
const STAIR_RISERS_TO_LANDING = 15
const STAIR_LANDING_Y_CM = SIDEWALK_Y_CM + STAIR_RISERS_TO_LANDING * STAIR_RISER_CM
/** MEASURED: the landing front to back. */
const STAIR_LANDING_DEPTH_CM = 160
/**
 * The flight geometry is DERIVED from the two measured levels, and it lands on
 * standard stair proportions exactly — the best evidence yet that both are
 * right.
 *
 *   riser  17.5, standard
 *   2R + G = 63 (the tread/riser rule) gives G = 28, exactly
 *   15 risers off the floor at -65 -> the landing at 197.5
 *   15 treads x 28 = 420 of run
 *
 * Nothing was chosen here but the riser count, and only one count makes both
 * numbers come out whole.
 */
const STAIR_RISERS_PER_FLIGHT = STAIR_RISERS_TO_LANDING
/** The landing's underside sits on the LAST RISER's underside — so its depth is
 *  one riser, not a guessed fascia. 40 was an assumption and it put the landing
 *  slab 22.5cm below the stair it belongs to. */
const STAIR_FASCIA_CM = STAIR_RISER_CM
const STAIR_TREAD_CM = 28
const STAIR_BALUSTRADE_H_CM = 110
const STAIR_GLASS_T_CM = 1.2
/**
 * The white structural band — the raking edge beam that carries the glass
 * balustrade and wraps both flights and the landing. From the site photos it is
 * the visible solid fascia along the outer edge, with the glass rail sitting on
 * top of it.
 *
 * ESTIMATED, not measured. Against the model's own dimensions — 17.5cm risers,
 * 28cm goings — the band in image 2 reads about 2.5 risers deep, so 45cm. It
 * hangs BELOW the walking line: its top is the glass base (the tread nosing, or
 * the landing surface) and it runs down from there, which is where the sloping
 * soffit beam actually is. `STAIR_BAND_X_CM` is its thickness across the rail —
 * thicker than the 1.2cm glass, so it reads as the "strong band" it is.
 */
const STAIR_BAND_H_CM = 45
const STAIR_BAND_X_CM = 6
/** MEASURED: the landing's near edge. */
const STAIR_LANDING_Z0_CM = 96
const STAIR_LANDING_Z1_CM = STAIR_LANDING_Z0_CM + STAIR_LANDING_DEPTH_CM
/** The column passes THROUGH the landing and does not touch it. */
const STAIR_COLUMN_CLEAR_CM = 2
/** ASSUMED: the well between the two flights. 100 clears the 50cm column by
 *  25 each side, which is what "a sizeable X gap" reads as. */
const STAIR_WELL_CM = 100

/**
 * THE COLUMN IS DERIVED FROM THE STAIR, not measured.
 *
 * It was placed first, from an estimate flagged on site as possibly off. The
 * stair then pinned it three ways — the flights span M2→M5, they are equal, and
 * the column is CENTRED IN THE WELL BETWEEN THEM — and those three put its
 * centre at the midpoint of M2→M5, x = 279.975. That is 125cm from the
 * estimate. Measured alignments beat an estimate, so the column now follows the
 * stair and cannot drift away from the well it stands in.
 *
 * Its z is measured: 271.
 */
const COLUMN_SIZE_CM = 50
const STAIR_WELL_CENTRE_CM = CORNER_X_CM + MULLION_SECTION.acrossCm / 2
  + MULLION_SPACINGS_CM[0] + (MULLION_SPACINGS_CM[1] + MULLION_SPACINGS_CM[2] + MULLION_SPACINGS_CM[3]) / 2
const COLUMN_X0_CM = Math.round((STAIR_WELL_CENTRE_CM - COLUMN_SIZE_CM / 2) * 1e9) / 1e9
/** MEASURED. */
const COLUMN_Z0_CM = 271

/**
 * THE STAIRCASE — a switchback, flight 2 running back ABOVE flight 1.
 *
 * ANCHORED, all stated on site:
 *   · flight 1's far +X edge on mullion 5, its close edge on mullion 2 — so it
 *     is 457.2 wide, the M2→M5 span. This is a broad feature stair, not a
 *     circulation run, and the first model was 3.5x too narrow.
 *   · the landing is LONGER than the two flights' combined width, and that
 *     difference is where it takes the column — which is the well: 457.2 of
 *     landing against 357.2 of flight, the 100cm gap being the well itself.
 *   · the landing's near Z edge is at z = 96.
 *   · the bottom of flight 1 is at sidewalk level, y = −65.
 *
 * THE FLIGHTS WRAP THE COLUMN, not the landing. The column stands in the WELL
 * between the two flights — centred in it in x, and at z 271, past the
 * landing's back edge at 256. So the landing is one plain box; it stops short
 * of the column rather than being penetrated by it.
 *
 * WHY FLIGHT 2 IS ABOVE, NOT BESIDE. Both flights are 457.2 wide and offset by
 * only 129.5 in x, so they overlap laterally and cannot sit side by side. They
 * share a Z band and are separated VERTICALLY instead: at the landing they meet
 * at the landing level, and diverge thereafter. That is the standard switchback
 * and it is what image 2 shows.
 *
 * STILL ASSUMED: what flight 2 climbs TO. Its rise is mirrored from flight 1
 * for want of an upper-floor level, which puts its head at y 495 — above the
 * mullion head at 375. Flight 1 is fully anchored; flight 2's TOP is not, and
 * that is the number to correct next.
 *
 * The space under each flight is left open, because it is — image 2 has a
 * bicycle parked under the lower run.
 */
function staircase() {
  const round = (v) => Math.round(v * 1e9) / 1e9
  const m1 = CORNER_X_CM + MULLION_SECTION.acrossCm / 2
  const m2 = m1 + MULLION_SPACINGS_CM[0]
  const m5 = m1 + MULLION_SPACINGS_CM.reduce((a, b) => a + b, 0)

  // Two EQUAL flights either side of the well. Flight 1 (coming up) is the +X
  // one with its far edge on mullion 5; flight 2 is the −X one with its near
  // edge on mullion 2. The well between them is centred on the M2→M5 midpoint,
  // which is what puts the column there.
  const flightW = round((m5 - m2 - STAIR_WELL_CM) / 2)
  const f2x0 = m2
  const f1x0 = round(m5 - flightW)
  const f1x1 = m5
  const width = flightW
  // The landing spans the flights and the well between them — nothing more.
  // It is a staircase, so its −X edge is flight 2's −X edge, not some overhang
  // beyond it. That also settles "the landing is longer than the two widths of
  // the stairs, and that is where it wraps the column": 457.2 against 357.2 of
  // actual flight, and the 100cm difference IS the well the column stands in.
  const landX0 = f2x0
  const landX1 = m5

  const landingY = STAIR_LANDING_Y_CM
  const rise = landingY - STAIR_LOWER_Y_CM
  const riser = STAIR_RISER_CM
  // ONE TREAD PER RISER, because the loop below emits a tread for every riser
  // including the last (whose surface is the landing level). With the classic
  // (risers − 1) the flight was one tread short and its top step landed INSIDE
  // the landing's z band — flush in height, overlapping in plan, which reads
  // as correct in a section and is wrong in the model.
  const run = STAIR_RISERS_PER_FLIGHT * STAIR_TREAD_CM
  const zTop = STAIR_LANDING_Z1_CM
  const zBot = round(zTop + run)

  const out = []
  const box = (id, label, kind, x, w, y, h, z, d, labelled = true) => out.push({
    id, label, kind,
    xCm: round(x), zCm: round(z), widthCm: round(w), depthCm: round(d),
    heightCm: round(h), baseYCm: round(y), labelled, anchor: 'corner',
  })

  // `k` runs to the riser count INCLUSIVE: step 1's top is one standard rise
  // off the lower floor, and step 15's top IS the landing. Flight 2 then starts
  // FROM the landing — its first step rises off it, so the two flights share
  // that level rather than stacking a redundant tread on it.
  for (let k = 1; k <= STAIR_RISERS_PER_FLIGHT; k++) {
    box(`stair-f1-step-${k}`, k === 1 ? 'stair flight 1' : `stair flight 1 step ${k}`, 'solid',
      f1x0, width, STAIR_LOWER_Y_CM + (k - 1) * riser, riser,
      zBot - k * STAIR_TREAD_CM, STAIR_TREAD_CM, k === 1)
    box(`stair-f2-step-${k}`, k === 1 ? 'stair flight 2' : `stair flight 2 step ${k}`, 'solid',
      f2x0, width, landingY + (k - 1) * riser, riser,
      zTop + (k - 1) * STAIR_TREAD_CM, STAIR_TREAD_CM, k === 1)
    box(`stair-f1-glass-${k}`, `stair flight 1 balustrade ${k}`, 'glass',
      f1x1, STAIR_GLASS_T_CM, STAIR_LOWER_Y_CM + k * riser, STAIR_BALUSTRADE_H_CM,
      zBot - k * STAIR_TREAD_CM, STAIR_TREAD_CM, false)
    box(`stair-f2-glass-${k}`, `stair flight 2 balustrade ${k}`, 'glass',
      f2x0 - STAIR_GLASS_T_CM, STAIR_GLASS_T_CM, landingY + k * riser, STAIR_BALUSTRADE_H_CM,
      zTop + (k - 1) * STAIR_TREAD_CM, STAIR_TREAD_CM, false)
    // The white band, directly BELOW each glass run: its top is the glass base
    // (the tread nosing), and it runs STAIR_BAND_H_CM down as the outer edge
    // beam. Flight 1's beam is inboard of its +X edge; flight 2's inboard of its
    // −X edge — each on the same side its glass is.
    //
    // The bottom is CLAMPED to the sidewalk level: a full 45cm beam hung under
    // the lowest treads would sink 27.5cm below the floor the stair stands on.
    // So the beam shortens as it meets the ground rather than diving through it,
    // which is what a raking stringer actually does at its foot.
    const f1top = STAIR_LOWER_Y_CM + k * riser
    const f1bot = Math.max(f1top - STAIR_BAND_H_CM, STAIR_LOWER_Y_CM)
    box(`stair-f1-band-${k}`, `stair flight 1 band ${k}`, 'solid',
      f1x1 - STAIR_BAND_X_CM, STAIR_BAND_X_CM,
      f1bot, f1top - f1bot,
      zBot - k * STAIR_TREAD_CM, STAIR_TREAD_CM, false)
    box(`stair-f2-band-${k}`, `stair flight 2 band ${k}`, 'solid',
      f2x0, STAIR_BAND_X_CM,
      landingY + k * riser - STAIR_BAND_H_CM, STAIR_BAND_H_CM,
      zTop + (k - 1) * STAIR_TREAD_CM, STAIR_TREAD_CM, false)
  }

  // --- the landing ---------------------------------------------------------
  // ONE box, not four. It was split around a column penetration back when the
  // landing ran to z 380; at 160 deep it stops at 256 and the column starts at
  // 271, so nothing passes through it and the hole was cutting a
  // negative-depth piece.
  //
  // The column is still wrapped — by the FLIGHTS, via the well between them,
  // which is where it always sat in x. "The stairs wrap around the column" is
  // about the stairs, and the landing simply stops short of it.
  const ly = landingY - STAIR_FASCIA_CM
  box('stair-landing', 'stair landing', 'solid',
    landX0, landX1 - landX0, ly, STAIR_FASCIA_CM,
    STAIR_LANDING_Z0_CM, STAIR_LANDING_Z1_CM - STAIR_LANDING_Z0_CM)
  box('stair-landing-glass', 'stair landing balustrade', 'glass',
    landX0, landX1 - landX0, landingY, STAIR_BALUSTRADE_H_CM,
    STAIR_LANDING_Z0_CM, STAIR_GLASS_T_CM, false)
  // The band under the landing's front balustrade, running the full x span so it
  // ties flight 2's beam to flight 1's — this is where the band "wraps" the
  // landing. Top at the landing surface, hanging STAIR_BAND_H_CM below it.
  box('stair-landing-band', 'stair landing band', 'solid',
    landX0, landX1 - landX0, landingY - STAIR_BAND_H_CM, STAIR_BAND_H_CM,
    STAIR_LANDING_Z0_CM, STAIR_BAND_X_CM, false)

  return out
}

/**
 * Obstacles ALWAYS come from the room as measured. A saved list cannot override
 * them.
 *
 * This started as "saved entries override by id, unknown ids appended", so a
 * tuned element would survive. There is no UI to tune one, and the rule cost
 * far more than it bought: every element here is DERIVED — the facades from the
 * mullion spacings, the stair from its riser, the column from the stair's well
 * — so a browser that saved before a correction kept serving the old geometry,
 * and a deleted element (`stair-f1-step-16`) came back as an "unknown id".
 *
 * The user's report was "I'm refreshing and literally nothing changed", twice,
 * for two different elements. Both times my own checks passed because they
 * called `resetConfig()` first and so never took the path that mattered.
 *
 * An explicit empty list still means "no room" — that is a deliberate statement
 * for studying the design alone, and it is not what a stale save carries.
 */
function mergeObstacles(saved) {
  if (Array.isArray(saved) && saved.length === 0) return []
  return DEFAULT_OBSTACLES.map((o) => ({ ...o }))
}

export const DEFAULT_OBSTACLES = [
  {
    // The raised platform the display panels stand on. Fills from the wall/window
    // corner out to mullion 5 (the stair's +X edge), and back in +z to the return
    // elevation's far end at 533.35. Rises 65cm from the sidewalk to the floor
    // at y = 0 — the surface `y = 0` is DEFINED to be, and what the panels sit
    // on. Derived from the room's own datums, so it cannot drift from them.
    id: 'floor',
    label: 'display floor',
    kind: 'solid',
    xCm: 0,
    zCm: 0,
    widthCm: CORNER_X_CM + FACADE_RUN_CM,
    depthCm: CORNER_Z_CM + FACADE_RUN_CM,
    heightCm: -SIDEWALK_Y_CM,
    baseYCm: SIDEWALK_Y_CM,
    labelled: true,
    anchor: 'corner',
  },
  {
    id: 'column',
    label: 'column',
    kind: 'solid',
    xCm: COLUMN_X0_CM,
    zCm: COLUMN_Z0_CM,
    widthCm: COLUMN_SIZE_CM,
    depthCm: COLUMN_SIZE_CM,
    // Its top is the top of the WINDOWS — derived from the mullion head, so
    // the two cannot drift apart.
    heightCm: MULLION_TOP_Y_CM,
    baseYCm: 0,
    labelled: true,
    anchor: 'corner',
  },
  {
    // The heating run along the window side, measured on site 2026-07-28.
    // Occupies x −81.3 → 512.7, z −59.7 → 0: it lies on the WINDOW SIDE of the
    // z = 0 reference plane, and reaches past the wall face at x = 0.
    //
    // Its min corner (−81.3, −59.7) is a stated DATUM — the user places
    // subsequent window elements from it, so it must not be quietly re-anchored
    // to a centre or re-derived from the network.
    //
    // heightCm is NOT measured. 20cm is a placeholder chosen to sit below the
    // network's own 15cm standoff so it cannot silently pass a clash test it
    // should fail; it is flagged in the UI as unmeasured rather than presented
    // as a dimension.
    id: 'heating',
    label: 'heating gap',
    kind: 'zone',
    // STARTS AT THE RETURN MULLIONS' INNER FACE, not at the corner. The return
    // elevation occupies x −81.3 → −62.25, so the run along the main elevation
    // cannot begin until past it. Derived from the section rather than typed,
    // so it follows if the mullion depth is ever corrected.
    xCm: RETURN_INNER_X_CM,
    zCm: CORNER_Z_CM,
    widthCm: 512.7 - RETURN_INNER_X_CM,
    depthCm: -CORNER_Z_CM,
    // A TRENCH: bottom on the mullions' bottom at y = −25, top at the FLOOR,
    // y = 0. The 20cm top was my placeholder and is gone.
    //
    // Stopping at the floor is what lets this run pass UNDER the wall slab
    // rather than through it — the wall is drawn from y = 0 up, so a gap below
    // the floor and a wall above it never meet. That is exactly why the second
    // run can reach the wall's room-side face at x = 0.
    heightCm: -SILL_TOP_Y_CM,
    baseYCm: SILL_TOP_Y_CM,
    labelled: true,
    anchor: 'corner',
  },
  {
    // The run along the RETURN elevation, in the bay beyond the wall.
    //
    // Spans the floor from the return glazing's inner face to the wall's
    // room-side face (x −62.25 → 0, 62.25 wide — slightly wider than the main
    // run's 59.7 depth), and runs in +z from where the main run ends to the far
    // end of the return glazing. Starting at z = 0 makes the two tile exactly:
    // the main run already covers the corner square across this whole x range,
    // so there is neither an overlap nor a missed strip.
    //
    // It reaches x = 0, the wall's ROOM-SIDE face, and that is only coherent
    // because the trench stops at the floor — see the note on `heating`.
    id: 'heating-return',
    label: 'heating gap — return',
    kind: 'zone',
    xCm: RETURN_INNER_X_CM,
    zCm: 0,
    widthCm: -RETURN_INNER_X_CM,
    depthCm: CORNER_Z_CM + FACADE_RUN_CM,
    heightCm: -SILL_TOP_Y_CM,
    baseYCm: SILL_TOP_Y_CM,
    labelled: true,
    anchor: 'corner',
  },
  ...facade('x'),
  ...facade('z'),
  ...staircase(),
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
/**
 * The wave's neutral setting: no scrunch, the attractor at the far end.
 *
 * Note what this is NOT. `{ kind: 'wave', ...DEFAULT_WAVE }` is not the same
 * network as `{ kind: 'trapezoid' }` — it is the uniform-angle EGG-CRATE, three
 * levels rather than two (wave.js's header). The neutral wave is the wave with
 * its scrunch off, not the trapezoid by another name.
 */
export const DEFAULT_WAVE = { scrunchX: 0, scrunchZ: 0, attractorX: 1, attractorZ: 1 }
/**
 * `yOffsetCm` defaults to 15, not 0. The number came in as a ground clearance —
 * everything laying flat on the ground stood off it by 15cm — and the part that
 * held that gap open has since been dropped (§9.12), but the default is left
 * where it is: it is the height the existing designs were drawn at, and moving
 * it would silently redraw every one of them.
 */
export const DEFAULT_PLACEMENT = {
  wallOffsetCm: 0,
  windowOffsetCm: 0,
  groundToFloor: true,
  // THE PANELS SIT ON THE FLOOR, AND THE FLOOR IS y = 0 — the surface visible
  // in the site photo. 15 was the ground-spacer height and outlived the part it
  // was named after; leaving it there floated the whole network 15cm off the
  // floor it is supposed to stand on.
  yOffsetCm: 0,
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
  room: { ...DEFAULT_ROOM },
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
      // The wave block is RAW and CONDITIONAL. Raw, like the overrides, so a
      // malformed knob is reported rather than silently defaulted. Conditional,
      // because it means nothing under 'trapezoid': filling it in there would
      // make `validateConfig` report on a block the user never wrote, and would
      // put a key in `normalizeConfig`'s output that changes the serialization of
      // every checkerboard design that already exists.
      ...(patternSrc.wave !== undefined
        ? { wave: patternSrc.wave }
        : patternSrc.kind === 'wave'
          ? { wave: { ...DEFAULT_WAVE } }
          : {}),
    },
    placement: {
      wallOffsetCm: pick(placementSrc, 'wallOffsetCm', DEFAULT_PLACEMENT.wallOffsetCm),
      windowOffsetCm: pick(placementSrc, 'windowOffsetCm', DEFAULT_PLACEMENT.windowOffsetCm),
      groundToFloor: pick(placementSrc, 'groundToFloor', DEFAULT_PLACEMENT.groundToFloor),
      // `groundClearanceCm` is the LEGACY name of this key, read here and folded
      // into the new one. It is not a compatibility shim in the abstract: every
      // design the user has saved to disk, and the working config in their
      // localStorage, carries the old key, and `version` is still 4 — the shape
      // did not change, only the name and the band — so nothing else would
      // migrate them. Dropping the read would silently reset all of that work to
      // the default. The NEW key wins when both are present, so a config written
      // by this version means what it says.
      yOffsetCm: pick(
        placementSrc,
        'yOffsetCm',
        pick(placementSrc, 'groundClearanceCm', DEFAULT_PLACEMENT.yOffsetCm),
      ),
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
    room: {
      wallThicknessCm: pick(
        isPlainObject(src.room) ? src.room : {},
        'wallThicknessCm',
        DEFAULT_ROOM.wallThicknessCm,
      ),
    },
    obstacles: mergeObstacles(src.obstacles),
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
      baseYCm: clamp(numberOr(o.baseYCm, 0), OBSTACLE_BASE_Y_MIN, OBSTACLE_BASE_Y_MAX),
      // Whether the viewport draws its name. Repeated sub-elements — steps,
      // balustrade segments, mullion caps — set this false: 46 stair parts each
      // shouting their id buried the model they were meant to help place.
      labelled: o.labelled === undefined ? true : Boolean(o.labelled),
      anchor: oneOf(OBSTACLE_ANCHORS, o.anchor, 'corner'),
      kind: oneOf(OBSTACLE_KINDS, o.kind, 'solid'),
    })
  })
  return out
}

/**
 * PRIVATE. The wave's four knobs, filled and clamped.
 *
 * There is no "drop the no-op" rule here, unlike the overrides: the block is
 * emitted whole or not at all (by KIND, in normalizeConfig), so a wave config
 * always carries all four numbers and two wave designs still compare equal iff
 * they are the same design.
 */
function sanitizeWave(raw) {
  const src = isPlainObject(raw) ? raw : {}
  const num = (key, lo, hi) => clamp(numberOr(src[key], DEFAULT_WAVE[key]), lo, hi)
  return {
    scrunchX: num('scrunchX', SCRUNCH_MIN, SCRUNCH_MAX),
    scrunchZ: num('scrunchZ', SCRUNCH_MIN, SCRUNCH_MAX),
    attractorX: num('attractorX', ATTRACTOR_MIN, ATTRACTOR_MAX),
    attractorZ: num('attractorZ', ATTRACTOR_MIN, ATTRACTOR_MAX),
  }
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

  // The kind decides whether the wave block exists at all — see `withDefaults`.
  const patternKind = oneOf(PATTERN_KINDS, cfg.pattern.kind, DEFAULT_PATTERN.kind)

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
      kind: patternKind,
      // Clamped rather than wrapped, as every other integer knob here clamps.
      // Kept under 'wave' even though it does nothing there — dropping it would
      // lose the setting the moment you looked at a wave and switched back.
      phase: clampInt(cfg.pattern.phase, DEFAULT_PATTERN.phase, PHASE_MIN, PHASE_MAX),
      ...(patternKind === 'wave' ? { wave: sanitizeWave(cfg.pattern.wave) } : {}),
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
      yOffsetCm: clamp(
        numberOr(cfg.placement.yOffsetCm, DEFAULT_PLACEMENT.yOffsetCm),
        Y_OFFSET_MIN,
        Y_OFFSET_MAX,
      ),
      wallAnchor: oneOf(WALL_ANCHORS, cfg.placement.wallAnchor, DEFAULT_PLACEMENT.wallAnchor),
    },
    overrides: {
      cells: sanitizeCells(cfg.overrides.cells, cols, rows),
      edges: sanitizeEdges(cfg.overrides.edges, cols, rows),
    },
    room: {
      wallThicknessCm: clamp(
        numberOr(cfg.room.wallThicknessCm, DEFAULT_ROOM.wallThicknessCm),
        WALL_THICKNESS_MIN,
        WALL_THICKNESS_MAX,
      ),
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
  checkRange('placement.yOffsetCm', cfg.placement.yOffsetCm,
    Y_OFFSET_MIN, Y_OFFSET_MAX,
    'where the whole network sits in y — from its own lowest material when grounded, cm')
  // The room, not the design: the wall builds up AWAY from the installation, so
  // this never moves a panel. See DEFAULT_ROOM.
  checkRange('room.wallThicknessCm', cfg.room.wallThicknessCm, WALL_THICKNESS_MIN, WALL_THICKNESS_MAX,
    'wall build-up running away from the installation, cm')

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

  // --- the wave's four knobs ------------------------------------------------
  // Checked whenever the block is PRESENT, not only when the kind is 'wave': a
  // config carrying `scrunchX: 5` is wrong whether or not it is currently being
  // used, and reporting it only after the kind is switched would make the error
  // appear to come from the switch.
  if (cfg.pattern.wave !== undefined) {
    if (!isPlainObject(cfg.pattern.wave)) {
      err(
        'E_SHAPE',
        `pattern.wave must be an object with scrunchX / scrunchZ / attractorX / attractorZ ` +
          `(got ${JSON.stringify(cfg.pattern.wave)})`,
        'pattern.wave',
      )
    } else {
      const w = cfg.pattern.wave
      checkRange('pattern.wave.scrunchX', w.scrunchX, SCRUNCH_MIN, SCRUNCH_MAX,
        'fraction of the base plan advance given up across x')
      checkRange('pattern.wave.scrunchZ', w.scrunchZ, SCRUNCH_MIN, SCRUNCH_MAX,
        'fraction of the base plan advance given up across z')
      checkRange('pattern.wave.attractorX', w.attractorX, ATTRACTOR_MIN, ATTRACTOR_MAX,
        'where along x full scrunch is reached, as a fraction of the edge run')
      checkRange('pattern.wave.attractorZ', w.attractorZ, ATTRACTOR_MIN, ATTRACTOR_MAX,
        'where along z full scrunch is reached, as a fraction of the edge run')
    }
    if (cfg.pattern.kind !== 'wave') {
      warn(
        'W_WAVE_SETTINGS_IGNORED',
        `pattern.wave is set but pattern.kind is ${JSON.stringify(cfg.pattern.kind)} — the wave knobs ` +
          'do nothing on the checkerboard, and normalizeConfig will drop the block',
        'pattern.wave',
      )
    }
  }
  if (cfg.pattern.kind === 'wave' && cfg.pattern.phase !== DEFAULT_PATTERN.phase) {
    warn(
      'W_PHASE_IGNORED',
      'pattern.phase does nothing under the wave — there is no checkerboard to shift. The wave\'s ' +
        'height field is `h(i,j) = f(i) + g(j)` and both runs start at 0 by construction',
      'pattern.phase',
    )
  }
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
