/**
 * grid-designer — controls for one COLUMN's fold strip, plus the cross-column
 * toolbar that operates on all six of them at once.
 *
 * =============================================================================
 * THE MODEL THIS DRIVES
 * =============================================================================
 * Folding happens only along COLUMNS (schema v2). Column c is an independent
 * strip: a chain of `grid.rows` panels hinged at the `grid.rows - 1` joints
 * between consecutive rows, starting flat on the ground at the window (row 0,
 * the shore) and folding back into the store. So one block per column, one
 * slider per hinge, and the interesting design knob is the PHASE relationship
 * between the six sequences — which is what the toolbar's "Shift →" exists for.
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
 *   - the 4 hinge sliders (±120°, 1° steps, live value readout), each labelled
 *     by the rows it joins ("0→1" … "3→4"). A slider is DISABLED when a vertical
 *     plate spans that joint: the plate is one rigid 121cm panel, so the joint
 *     does not exist any more (`removedJoints`) and the store would reject the
 *     change with E_FOLD_ON_REMOVED_JOINT anyway.
 *   - the cumulative pitch ψ per row, read from the solved
 *     `layout.columnChains[c].pitchesDeg` (never recomputed here, so the number
 *     shown is the number the solver used).
 *   - a profile sparkline: the column's own Y–Z chain
 *     (`layout.columnChains[c].points`) drawn shore-at-left, on a scale SHARED
 *     by all six columns so their heights are directly comparable. Segments with
 *     an endpoint below y = 0 are drawn red — those panels have nothing to stand
 *     on (the same condition placement.js reports as W_BELOW_FLOOR). The vertical
 *     axis is EXAGGERATED (a 308cm-deep chain in a 90px box would otherwise flatten
 *     every fold into the baseline); the pitch row above is the honest angle
 *     readout, the sparkline is the shape.
 *
 * Sliders commit on every `input` event: the solve is memoized per config and a
 * whole 6×5 solve + joint report is well under a millisecond.
 */

import useStore, { getDerived } from '../store.js'
import { MAX_FOLD_DEG, removedJoints } from '../core/schema.js'

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
function ProfileSparkline({ c, points, bounds }) {
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
      aria-label={`column ${c} fold profile, window at the left${below.length > 0 ? ', dips below the floor' : ''}`}
    >
      <title>
        {`column ${c} fold profile — window/shore at the left, depth to the right, height up` +
          (below.length > 0 ? ' · red = below the floor (nothing to stand on)' : '')}
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
    </svg>
  )
}

// -----------------------------------------------------------------------------
// Cross-column toolbar
// -----------------------------------------------------------------------------
/**
 * The three whole-grid fold operations. All go through the store's commit rule,
 * so a move the rects forbid (a vertical plate's joint would end up folded, or a
 * horizontal plate's two columns would stop agreeing in pitch) changes nothing
 * and lands in the error box instead — remove the plate, or shift the other way.
 *
 * @param {{selected: number}} props the column "Copy → all" reads from
 */
export function ColumnToolbar({ selected }) {
  const copyColumnToAll = useStore((s) => s.copyColumnToAll)
  const shiftColumnsRight = useStore((s) => s.shiftColumnsRight)
  const flattenFolds = useStore((s) => s.flattenFolds)

  return (
    <div className="col-tools" data-testid="column-tools">
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
  const { layout } = getDerived(config)

  const folds = config.columns?.[c]?.foldsDeg ?? []
  const chain = layout.columnChains?.[c]
  const pitches = chain?.pitchesDeg ?? []
  const points = chain?.points ?? []
  const removed = removedJoints(config, c)
  const lastCol = (config.grid?.cols ?? 1) - 1
  // The camera looks in from the shore, so +X (rising column index) runs LEFT.
  const side = c === 0 ? 'rightmost' : c === lastCol ? 'leftmost' : `${ordinal(c + 1)} from right`

  return (
    <section
      className={`col-block${selected ? ' col-selected' : ''}`}
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
        <ProfileSparkline c={c} points={points} bounds={bounds} />
      </header>

      <p className="col-pitch" title="cumulative pitch ψ per row, window → back (row 0 is the shore, always 0°)">
        {pitches.map((p, r) => (
          <span key={r} className={p < 0 ? 'pitch-neg' : undefined}>
            {Math.round(p)}°
          </span>
        ))}
      </p>

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
