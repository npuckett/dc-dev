/**
 * grid-designer — controls for one COLUMN's fold strip, plus the cross-column
 * toolbar that operates on all six of them at once.
 *
 * =============================================================================
 * THE MODEL THIS DRIVES
 * =============================================================================
 * Folding happens only along COLUMNS (schema v2). Column c is an independent
 * strip: a chain of `grid.rows` panels hinged at the `grid.rows - 1` joints
 * between consecutive rows, starting ON THE GROUND at the window (row 0, the
 * front panel, at whatever `startPitchDeg` it is given) and folding back into the
 * store. So one block per column, a FRONT slider plus one slider per hinge, and
 * the interesting design knob is the PHASE relationship between the six sequences
 * — which is what the toolbar's "Shift →" exists for.
 *
 * VIEWPORT HANDEDNESS — the 3D camera sits on the shore side (−Z) looking into
 * the room, so world +X runs to the LEFT on screen: column 0 appears on the
 * RIGHT, column cols-1 on the LEFT. Blocks are listed in index order (0 first)
 * and every header says which side of the viewport it is, because guessing wrong
 * while chasing a wave across the window is the easiest mistake to make here.
 *
 * =============================================================================
 * WHAT EACH BLOCK SHOWS
 * =============================================================================
 *   - the FRONT slider (±120°, 1° steps) — `columns[c].startPitchDeg`, the pitch
 *     of row 0, the panel in the window. It sits ABOVE the hinge sliders because
 *     it is the head of the pitch profile, not a joint: every later row's pitch is
 *     this plus the folds in front of it. The front panel stays on the floor at
 *     any value (core/schema.js `frontRestY`), so this slider never breaks
 *     support at the window; a NEGATIVE value dives the strip and can push later
 *     panels under the floor, which shows up as W_BELOW_FLOOR.
 *   - the `grid.rows - 1` hinge sliders (±120°, 1° steps, live value readout),
 *     each labelled by the rows it joins ("0→1" … ). A slider is DISABLED when a
 *     vertical plate spans that joint: the plate is one rigid 121cm panel, so the
 *     joint does not exist any more (`removedJoints`) and the store would reject
 *     the change with E_FOLD_ON_REMOVED_JOINT anyway.
 *   - the cumulative pitch ψ per row, read from the solved
 *     `layout.columnChains[c].pitchesDeg` (never recomputed here, so the number
 *     shown is the number the solver used) — its first entry IS the front pitch.
 *   - a profile sparkline: the column's own Y–Z chain
 *     (`layout.columnChains[c].points`) drawn shore-at-left, on a scale SHARED
 *     by all six columns so their heights are directly comparable. Segments with
 *     an endpoint below y = 0 are drawn red — those panels have nothing to stand
 *     on (the same condition placement.js reports as W_BELOW_FLOOR). The vertical
 *     axis is EXAGGERATED (a 308cm-deep chain in a 90px box would otherwise flatten
 *     every fold into the baseline); the pitch row above is the honest angle
 *     readout, the sparkline is the shape.
 *   - the END-SUPPORT verdict for that column. A FLOOR column must land (schema.js:
 *     the last panel comes back and touches the floor): floating gets a red
 *     FLOATING badge with the clearance in cm, a red dot at the end of its
 *     sparkline, and a "Ground end" button that hands the last hinge to the solver
 *     in core/ground.js; grounded gets a small green endpoint dot. A WALL-SUPPORTED
 *     column — only the wall-adjacent one may be — gets a teal WALL-SUPPORTED badge
 *     with the height its end reaches instead, a teal endpoint dot, and NO "Ground
 *     end" button: it is bracketed to the −X wall and is meant to end high.
 *   - on the WALL-ADJACENT column only (c = `WALL_COLUMN` = 0), a "bracket to wall" /
 *     "stand on floor" toggle. No other column gets one, because `endSupport:
 *     'wall'` is an E_SHAPE error anywhere else — there is no wall to bracket to.
 *     Note the sides: the wall is at LOW x, so it is beside the strip the 3D camera
 *     shows RIGHTMOST, which is column 0.
 *
 * Sliders commit on every `input` event: the solve is memoized per config and a
 * whole 6×8 solve + joint report is well under a millisecond.
 */

import useStore, { getDerived } from '../store.js'
import {
  MAX_FOLD_DEG,
  MAX_ROWS,
  MIN_ROWS,
  WALL_COLUMN,
  removedJoints,
} from '../core/schema.js'

// -----------------------------------------------------------------------------
// Sparkline metrics (SVG user units = px at the rendered size)
// -----------------------------------------------------------------------------
const SPARK_W = 90
const SPARK_H = 36
const SPARK_PAD = 3.5

const PROFILE_LINE = '#9cc2ec'
const PROFILE_FILL = '#4a7fc4'
const PROFILE_BAD = '#ff5f6d'
const PROFILE_FLOOR = '#4a4a68'
const PROFILE_SHORE = '#8fcaff'
/** End-of-chain marker: the column's last panel touches the floor / floats. */
const PROFILE_GROUNDED = '#63d19b'
const PROFILE_FLOATING = '#ff5f6d'
/** …or is bracketed to the −X wall, in which case ending high is the design. */
const PROFILE_WALL = '#3ad0c0'
/**
 * Smallest height the vertical axis ever spans, cm. Keeps a FLAT column visibly
 * above the floor line instead of merging with it (a 3.7cm-high chain in a box
 * scaled to a 200cm crest would be a sub-pixel sliver).
 */
const PROFILE_MIN_HEIGHT_CM = 30

/**
 * Shared (y, z) bounds over every column chain, so all six sparklines are drawn
 * on one scale and their heights compare directly. y = 0 (the floor) and z = 0
 * (the window) are always included, so a flat column reads as a low flat line
 * rather than filling the box.
 *
 * @param {{columnChains: Array}} layout output of solveLayout
 * @returns {{zMin:number, zMax:number, yLo:number, yHi:number}}
 */
export function chainBounds(layout) {
  let zMin = 0
  let zMax = 1
  let yLo = 0
  let yHi = PROFILE_MIN_HEIGHT_CM
  for (const chain of layout.columnChains ?? []) {
    for (const [y, z] of chain.points ?? []) {
      if (Number.isFinite(z)) {
        if (z < zMin) zMin = z
        if (z > zMax) zMax = z
      }
      if (Number.isFinite(y)) {
        if (y < yLo) yLo = y
        if (y > yHi) yHi = y
      }
    }
  }
  return { zMin, zMax, yLo, yHi }
}

/**
 * One column's fold profile in its own Y–Z plane: window at the LEFT (z = 0),
 * depth to the right, height up. Depth and height are scaled INDEPENDENTLY (both
 * shared across the six columns) — at 90×36px an equal-aspect 308cm chain would
 * squash every fold into the baseline, so the sparkline shows the shape and the
 * pitch row above it carries the true angles.
 */
function ProfileSparkline({ c, points, bounds, grounded, clearanceCm, wallSupported }) {
  const availW = SPARK_W - 2 * SPARK_PAD
  const availH = SPARK_H - 2 * SPARK_PAD
  const zSpan = Math.max(bounds.zMax - bounds.zMin, 1e-6)
  const ySpan = Math.max(bounds.yHi - bounds.yLo, 1e-6)
  const sx = availW / zSpan
  const sy = availH / ySpan
  const px = (z) => SPARK_PAD + (z - bounds.zMin) * sx
  const py = (y) => SPARK_H - SPARK_PAD - (y - bounds.yLo) * sy

  const poly = points.map(([y, z]) => `${px(z).toFixed(2)},${py(y).toFixed(2)}`).join(' ')

  // Segments with an endpoint under the floor — unsupported panels.
  const below = []
  for (let i = 0; i + 1 < points.length; i++) {
    const [y0, z0] = points[i]
    const [y1, z1] = points[i + 1]
    if (y0 < 0 || y1 < 0) below.push([px(z0), py(y0), px(z1), py(y1)])
  }

  const floorY = py(0)
  const start = points[0] ?? [0, 0]
  const end = points[points.length - 1] ?? start
  // Filled silhouette down to the floor — what makes a nearly-flat profile read
  // as "low and flat" instead of as an empty box.
  const area =
    points.length > 1
      ? `${px(start[1]).toFixed(2)},${floorY.toFixed(2)} ${poly} ${px(end[1]).toFixed(2)},${floorY.toFixed(2)}`
      : null

  return (
    <svg
      className="col-profile"
      data-testid={`profile-${c}`}
      viewBox={`0 0 ${SPARK_W} ${SPARK_H}`}
      width={SPARK_W}
      height={SPARK_H}
      role="img"
      aria-label={`column ${c} fold profile, window at the left${below.length > 0 ? ', dips below the floor' : ''}${wallSupported ? ', end supported by the wall' : grounded ? ', end grounded' : ', end floating'}`}
    >
      <title>
        {`column ${c} fold profile — window/shore at the left, depth to the right, height up` +
          (below.length > 0 ? ' · red = below the floor (nothing to stand on)' : '') +
          (wallSupported
            ? ` · teal endpoint = bracketed to the −X wall, ending ${clearanceCm.toFixed(1)}cm up`
            : grounded
              ? ' · green endpoint = the last panel touches the floor'
              : ` · red endpoint = the last panel floats ${clearanceCm.toFixed(1)}cm up`)}
      </title>
      <line
        x1={SPARK_PAD}
        y1={floorY}
        x2={SPARK_W - SPARK_PAD}
        y2={floorY}
        stroke={PROFILE_FLOOR}
        strokeWidth={0.7}
        strokeDasharray="2 2"
      />
      <line
        x1={px(bounds.zMin)}
        y1={SPARK_PAD}
        x2={px(bounds.zMin)}
        y2={SPARK_H - SPARK_PAD}
        stroke={PROFILE_SHORE}
        strokeWidth={0.7}
        strokeOpacity={0.5}
      />
      {area && <polygon points={area} fill={PROFILE_FILL} fillOpacity={0.22} stroke="none" />}
      <polyline points={poly} fill="none" stroke={PROFILE_LINE} strokeWidth={1.3} />
      {below.map(([x1, y1, x2, y2], i) => (
        <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke={PROFILE_BAD} strokeWidth={1.6} />
      ))}
      <circle cx={px(start[1])} cy={py(start[0])} r={1.5} fill={PROFILE_SHORE} />
      {/* the end of the chain: does the last panel come back to the water? */}
      <circle
        data-testid={`profile-end-${c}`}
        data-grounded={grounded ? 'true' : 'false'}
        data-end-support={wallSupported ? 'wall' : 'floor'}
        cx={px(end[1])}
        cy={py(end[0])}
        r={wallSupported ? 2.2 : grounded ? 1.6 : 2.4}
        fill={wallSupported ? PROFILE_WALL : grounded ? PROFILE_GROUNDED : PROFILE_FLOATING}
        fillOpacity={grounded && !wallSupported ? 0.75 : 1}
      />
    </svg>
  )
}

// -----------------------------------------------------------------------------
// Cross-column toolbar
// -----------------------------------------------------------------------------
/**
 * The whole-grid fold operations. All go through the store's commit rule, so a
 * move the rects forbid (a vertical plate's joint would end up folded, or a
 * horizontal plate's two columns would stop agreeing in pitch) changes nothing
 * and lands in the error box instead — remove the plate, or shift the other way.
 *
 * "Ground all" is the one that enforces the grounded-end rule across the grid: it
 * solves the last hinge of every FLOATING column and leaves the rest alone
 * (wall-supported columns are skipped — they are meant to end high).
 *
 * The ROWS STEPPER lives here too, because changing the row resolution is a
 * whole-grid operation like the others. It commits through `setRows`, which
 * truncates or zero-pads every column's hinges AT THE BACK of the strip and drops
 * plates that no longer fit — see the store for the exact semantics, repeated in
 * the control's tooltip.
 *
 * The one button here that is NOT a design operation is "bounds": it shows / hides
 * the 3D view's overall measuring box (`showBounds`, plain UI state), so it goes
 * nowhere near `commit()` and cannot be rejected. It sits in this row because that
 * is where the other whole-grid controls are.
 *
 * @param {{selected: number, floating: number, rows: number}} props the column
 *        "Copy → all" reads from, how many columns currently float, and the grid's
 *        row resolution
 */
export function ColumnToolbar({ selected, floating = 0, rows = MIN_ROWS }) {
  const copyColumnToAll = useStore((s) => s.copyColumnToAll)
  const shiftColumnsRight = useStore((s) => s.shiftColumnsRight)
  const flattenFolds = useStore((s) => s.flattenFolds)
  const groundAllColumns = useStore((s) => s.groundAllColumns)
  const setRows = useStore((s) => s.setRows)
  const showBounds = useStore((s) => s.showBounds)
  const toggleBounds = useStore((s) => s.toggleBounds)
  const undo = useStore((s) => s.undo)
  const redo = useStore((s) => s.redo)
  const canUndo = useStore((s) => s.canUndo)
  const canRedo = useStore((s) => s.canRedo)

  const rowsTitle =
    `row resolution — how many panels deep each column strip is (${MIN_ROWS}–${MAX_ROWS}). ` +
    'Growing adds level rows at the BACK of every strip; shrinking drops the deepest hinges ' +
    'and any plate that no longer fits. The shape you designed at the window never moves.'

  return (
    <div className="col-tools" data-testid="column-tools">
      <span className="rows-stepper" data-testid="rows-stepper" title={rowsTitle}>
        <button
          type="button"
          className="tool-btn rows-btn"
          data-testid="rows-dec"
          disabled={rows <= MIN_ROWS}
          title={`one row shallower (min ${MIN_ROWS})`}
          onClick={() => setRows(rows - 1)}
        >
          −
        </button>
        <output className="rows-value" data-testid="rows-value">
          {rows} rows
        </output>
        <button
          type="button"
          className="tool-btn rows-btn"
          data-testid="rows-inc"
          disabled={rows >= MAX_ROWS}
          title={`one row deeper (max ${MAX_ROWS})`}
          onClick={() => setRows(rows + 1)}
        >
          +
        </button>
      </span>
      <button
        type="button"
        className={`tool-btn${floating > 0 ? ' tool-btn-urgent' : ''}`}
        data-testid="tool-ground-all"
        title={
          floating > 0
            ? `solve the last hinge of the ${floating} column(s) whose end panel floats, so every strip comes back down and touches the floor`
            : 'every column already touches the floor — nothing to ground'
        }
        onClick={() => groundAllColumns()}
      >
        Ground all
      </button>
      <button
        type="button"
        className="tool-btn"
        data-testid="tool-copy-all"
        title={`give every column column ${selected}'s fold sequence — a cylindrical fold, whose in-row joints are all exact`}
        onClick={() => copyColumnToAll(selected)}
      >
        Copy col {selected} → all
      </button>
      <button
        type="button"
        className="tool-btn"
        data-testid="tool-shift-right"
        title="move every fold sequence one column up in index (col c → col c+1, wrapping) — one step of a travelling wave. Note: rising index runs RIGHT-to-LEFT in the 3D view."
        onClick={() => shiftColumnsRight()}
      >
        Shift →
      </button>
      <button
        type="button"
        className={`tool-btn${showBounds ? ' tool-btn-on' : ''}`}
        data-testid="tool-bounds"
        aria-pressed={showBounds}
        title={
          showBounds
            ? "hide the overall measuring box and its cm dimensions in the 3D view (a ruler, not part of the design — it never reaches the exported JSON or OBJ)"
            : 'show the overall measuring box: a wireframe around the whole surface with its width, PEAK HEIGHT and depth in centimetres'
        }
        onClick={() => toggleBounds()}
      >
        bounds
      </button>
      <button
        type="button"
        className="tool-btn"
        data-testid="tool-undo"
        disabled={!canUndo}
        title={
          canUndo
            ? 'undo the last change (Cmd/Ctrl+Z) — the way back from a merge, which flattens the joint it spans and is deliberately not undone by re-splitting the plate'
            : 'nothing to undo yet (Cmd/Ctrl+Z)'
        }
        onClick={() => undo()}
      >
        Undo
      </button>
      <button
        type="button"
        className="tool-btn"
        data-testid="tool-redo"
        disabled={!canRedo}
        title={canRedo ? 'redo the change you just undid (Shift+Cmd/Ctrl+Z)' : 'nothing to redo (Shift+Cmd/Ctrl+Z)'}
        onClick={() => redo()}
      >
        Redo
      </button>
      <button
        type="button"
        className="tool-btn"
        data-testid="tool-flatten"
        title="set every hinge to 0° — back to the flat reference surface (rects are kept)"
        onClick={() => flattenFolds()}
      >
        Flatten
      </button>
    </div>
  )
}

// -----------------------------------------------------------------------------
// One column block
// -----------------------------------------------------------------------------
/** 1 → "1st", 2 → "2nd", … (only ever called with small counts). */
function ordinal(n) {
  const suffix = n % 10 === 1 && n !== 11 ? 'st' : n % 10 === 2 && n !== 12 ? 'nd' : n % 10 === 3 && n !== 13 ? 'rd' : 'th'
  return `${n}${suffix}`
}

/**
 * @param {{c: number, bounds: object, selected: boolean, onSelect: (c:number)=>void}} props
 */
export default function ColumnControls({ c, bounds, selected, onSelect }) {
  const config = useStore((s) => s.config)
  const setColumnFold = useStore((s) => s.setColumnFold)
  const setColumnStartPitch = useStore((s) => s.setColumnStartPitch)
  const setColumnEndSupport = useStore((s) => s.setColumnEndSupport)
  const groundColumn = useStore((s) => s.groundColumn)
  const { layout } = getDerived(config)

  const column = config.columns?.[c]
  const folds = column?.foldsDeg ?? []
  const startPitchDeg = Number(column?.startPitchDeg) || 0
  const chain = layout.columnChains?.[c]
  const pitches = chain?.pitchesDeg ?? []
  const points = chain?.points ?? []
  const grounded = chain?.grounded !== false
  const wallSupported = chain?.endSupport === 'wall'
  const clearanceCm = Number(chain?.endClearanceCm) || 0
  const removed = removedJoints(config, c)
  const lastCol = (config.grid?.cols ?? 1) - 1
  // The camera looks in from the shore, so +X (rising column index) runs LEFT.
  // Column 0 is also the one against the −X wall, which is worth saying in place.
  const side =
    c === WALL_COLUMN
      ? 'rightmost · at the wall'
      : c === lastCol
        ? 'leftmost'
        : `${ordinal(c + 1)} from right`
  // A floor column in mid-air is a broken rule; a wall column in mid-air is the design.
  const floatingRule = !grounded && !wallSupported

  return (
    <section
      className={`col-block${selected ? ' col-selected' : ''}${floatingRule ? ' col-floating' : ''}${wallSupported ? ' col-wall' : ''}`}
      data-testid={`column-${c}`}
    >
      <header className="col-head">
        <button
          type="button"
          className="col-title"
          data-testid={`column-select-${c}`}
          title={`select column ${c} as the source for "Copy col → all" — in the 3D view it is the ${side} strip (the camera looks in from the window, so column index rises right-to-left)`}
          aria-pressed={selected}
          onClick={() => onSelect?.(c)}
        >
          Col {c} <span className="col-side">({side})</span>
        </button>
        <ProfileSparkline
          c={c}
          points={points}
          bounds={bounds}
          grounded={grounded}
          clearanceCm={clearanceCm}
          wallSupported={wallSupported}
        />
      </header>

      {/* END SUPPORT — only the wall-adjacent column (column 0, against the −X
          wall) has a wall to bracket to, so only it gets the toggle; the store
          rejects 'wall' anywhere else anyway. */}
      {c === WALL_COLUMN && (
        <p className="col-wall-row">
          {wallSupported && (
            <span
              className="col-wall-badge"
              data-testid={`column-wall-${c}`}
              title={`column ${c} is side-bracketed to the wall at the −X edge of the grid, so it is exempt from the grounded-end rule: its last panel may end in mid-air (here ${clearanceCm.toFixed(2)}cm up) — the water splashing up the wall`}
            >
              WALL-SUPPORTED · ends {clearanceCm.toFixed(0)}cm up
            </span>
          )}
          <button
            type="button"
            className={`tool-btn col-wall-btn${wallSupported ? ' col-wall-btn-on' : ''}`}
            data-testid={`column-endsupport-${c}`}
            aria-pressed={wallSupported}
            title={
              wallSupported
                ? `stand column ${c} on the floor again — it will have to bring its last panel back down like every other column`
                : `side-bracket column ${c} to the wall at the −X edge of the grid: it becomes exempt from the grounded-end rule and may end high, splashing up the wall`
            }
            onClick={() => setColumnEndSupport(c, wallSupported ? 'floor' : 'wall')}
          >
            {wallSupported ? 'stand on floor' : 'bracket to wall'}
          </button>
        </p>
      )}

      {floatingRule && (
        <p className="col-float-row" data-testid={`column-floating-${c}`}>
          <span
            className="col-float-badge"
            title={`the last panel of column ${c} floats ${clearanceCm.toFixed(2)}cm above the floor — every floor-supported column must bring its last panel back down to touch the ground (the wave returns to the water)`}
          >
            FLOATING {clearanceCm.toFixed(1)}cm
          </span>
          <button
            type="button"
            className="tool-btn tool-btn-urgent col-ground-btn"
            data-testid={`column-ground-${c}`}
            title={`solve column ${c}'s last hinge so its end panel comes down and touches the floor`}
            onClick={() => groundColumn(c)}
          >
            Ground end
          </button>
        </p>
      )}

      <p
        className="col-pitch"
        title="cumulative pitch ψ per row, window → back. Row 0 IS the front panel's own pitch (columns[c].startPitchDeg) — the FRONT slider below sets it."
      >
        {pitches.map((p, r) => (
          <span
            key={r}
            className={`${p < 0 ? 'pitch-neg' : ''}${r === 0 ? ' pitch-front' : ''}`.trim() || undefined}
          >
            {Math.round(p)}°
          </span>
        ))}
      </p>

      {/* the FRONT panel's pitch — the head of the profile, not a joint */}
      <div className="slider-row slider-row-front">
        <label className="slider-label" htmlFor={`front-${c}`}>
          front
        </label>
        <input
          id={`front-${c}`}
          data-testid={`front-${c}`}
          type="range"
          min={-MAX_FOLD_DEG}
          max={MAX_FOLD_DEG}
          step={1}
          value={startPitchDeg}
          title={`pitch of column ${c}'s FRONT panel (row 0, in the window) — positive tilts it UP out of the window. Its front edge stays on the window line and the panel stays on the floor at any angle; a negative value dives the strip, which can push later panels under the floor (W_BELOW_FLOOR).`}
          onChange={(e) => setColumnStartPitch(c, Number(e.target.value))}
        />
        <output className="slider-value" htmlFor={`front-${c}`}>
          {startPitchDeg}°
        </output>
      </div>

      {folds.map((deg, k) => (
        <div className="slider-row" key={k}>
          <label className="slider-label" htmlFor={`fold-${c}-${k}`}>
            {k}→{k + 1}
          </label>
          <input
            id={`fold-${c}-${k}`}
            data-testid={`fold-${c}-${k}`}
            type="range"
            min={-MAX_FOLD_DEG}
            max={MAX_FOLD_DEG}
            step={1}
            value={Number(deg) || 0}
            disabled={removed.has(k)}
            title={
              removed.has(k)
                ? `joint ${k} is removed by the vertical plate at (${k}, ${c}) — that plate is one rigid 121cm panel and cannot bend, so this hinge does not exist (remove the plate to fold here)`
                : `signed hinge angle between rows ${k} and ${k + 1} of column ${c} — positive pitches row ${k + 1} UP`
            }
            onChange={(e) => setColumnFold(c, k, Number(e.target.value))}
          />
          <output className="slider-value" htmlFor={`fold-${c}-${k}`}>
            {Number(deg) || 0}°
          </output>
        </div>
      ))}
    </section>
  )
}
