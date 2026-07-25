/**
 * grid-designer — 2D plan view of the cols × rows grid, and the place/remove UI
 * for the 60×121 two-cell plates ("rects"). `rows` follows `grid.rows` (5..8), so
 * the map grows and shrinks with the row stepper.
 *
 * ORIENTATION: row 0 is drawn at the BOTTOM, matching the mental model of
 * standing outside the storefront looking in — the shore/window edge is nearest
 * the viewer, rows recede upward (away, +Z). So screen row = rows-1-r.
 * COLUMN 0 IS AT THE LEFT here (x rises rightward), which is the OPPOSITE of the
 * 3D view — the camera there looks in from the window, so +X runs leftward on
 * screen. The WALL closes the −X edge of the grid, beside column 0, so it is the
 * LEFT edge of this map and the RIGHT edge of the 3D view.
 * The ROW LABELS therefore sit on the RIGHT, out of the wall's way.
 *
 * WHAT IS DRAWN
 *   - one shape per PLACED PANEL, read off the derived layout rather than the
 *     config's rect list, so the map always shows what the solver actually
 *     emitted: squares as single cells, horizontal plates spanning two columns
 *     of a row, vertical plates spanning two rows of a column (tinted with the
 *     same cool blue the 3D rect material uses).
 *   - a per-cell PITCH WASH: each cell is tinted by its column chain's cumulative
 *     pitch ψ at that row (`layout.columnChains[c].pitchesDeg[r]`), opacity rising
 *     with |ψ| — cool blue where the strip has pitched UP, warm amber where it has
 *     pitched back DOWN. A wave travelling across the window therefore reads as a
 *     diagonal band in plan, which is exactly the thing the fold sliders are being
 *     dragged to produce.
 *   - a blue rule plus a "window / shore" caption along row 0's outer edge.
 *   - a thick teal rule plus a rotated "wall" caption along the LEFT edge — the
 *     −X side of the grid, where the real room's wall is. Column 0 is the only one
 *     that may set `endSupport: 'wall'` and be bracketed to it.
 *   - `meta.rectPattern` as a caption under the title, when the config names one.
 *     Every design is supposed to place ≥ 4 plates by ONE nameable rule (see THE
 *     PLATE-PATTERN RULE in core/schema.js); the caption is where that rule's name
 *     lives in the UI, so the map you place plates on also tells you what pattern
 *     you are placing them into. A config below four plates raises W_FEW_RECTS,
 *     which surfaces in the control panel's warnings box.
 *   - a dashed red outline + corner marker on the BACK cell (row rows-1) of every
 *     column whose last panel does not touch the floor — the grounded-end rule in
 *     core/schema.js, reported as `layout.violations`. Its tooltip carries the
 *     clearance ("end floating — X cm off the floor").
 *
 * INTERACTION — PAIR-CLICK MERGE / CLICK-TO-SPLIT
 *   A transparent 1-cell hit grid sits on top of the artwork (which is
 *   pointer-events: none), so every cell is a uniform click target regardless of
 *   what shape covers it:
 *     - click a SQUARE          → it is ARMED (a warm outline), and every neighbour
 *                                 it could be merged with lights up
 *     - click a HIGHLIGHTED     → the two squares become one 60×121 plate
 *       neighbour
 *     - click the armed cell
 *       again / a non-candidate → disarm.  Escape does the same.
 *     - click a PLATE           → split it back into two squares
 *   The orientation is IMPLIED by which pair you pick — H when the two cells sit
 *   side by side in a row, V when one is behind the other in a column — so there is
 *   no mode to set any more.
 *
 *   THE TWO CANDIDATE STYLES matter, because a merge is not always free: a plate is
 *   RIGID, so it may have to change the surface to exist (see core/merge.js).
 *     FREE   (green, solid outline, "+")  nothing changes but the plate list
 *     COERCE (amber, dashed outline, "~") the merge also flattens the joint it
 *            spans, or matches the neighbouring column's profile in front of it
 *   Each candidate's tooltip spells the consequence out BEFORE the click, and
 *   `lastNotice` (the calm line under the map) repeats what actually happened after
 *   it. Undo in the toolbar (or Cmd/Ctrl+Z) is the way back — a merge is
 *   deliberately not undone by re-splitting the plate.
 *
 *   Everything still goes through the store's commit rule, so nothing illegal can
 *   land. Because `mergeCandidates` only offers merges core/merge.js has already
 *   proven, a highlighted cell never bounces; the rejection box below is for the
 *   remaining paths (a merge whose coercion would break another plate, and plates
 *   placed from the JSON panel or the console):
 *     E_FOLD_ON_REMOVED_JOINT    a V plate over a joint that is folded
 *     E_CROSSCOL_ANGLE_MISMATCH  an H plate over two columns at different pitches
 *     E_RECT_OVERLAP / E_MERGE_* overlapping, or an impossible pair
 *   Those are reported INLINE under the map — the cell you just clicked is what
 *   they are about — and the control panel's general error box skips exactly them,
 *   so a rejection is explained once, not twice.
 */

import { useEffect } from 'react'
import useStore, { getDerived } from '../store.js'
import { mergeCandidates } from '../core/merge.js'

// -----------------------------------------------------------------------------
// SVG metrics (user units; the <svg> scales to the panel width via viewBox)
// -----------------------------------------------------------------------------
const CELL = 40
const GAP = 4
const PITCH = CELL + GAP
/** Room on the LEFT for the wall rule and its rotated caption. */
const PAD_L = 16
/** Room on the RIGHT for the row labels, which the wall displaced from the left. */
const PAD_R = 12
const PAD_T = 12
const PAD_B = 17

const SQUARE_FILL = '#272736'
const SQUARE_STROKE = '#3c3c56'
const RECT_FILL = '#1f3350'
const RECT_STROKE = '#4a7fc4'
const SHORE = '#8fcaff'
/** The −X wall — and the badge colour of a column bracketed to it. */
const WALL = '#3ad0c0'
/** A column whose last panel does not reach the floor. */
const FLOATING = '#ff5f6d'

/** The square being held, waiting for a partner to merge with. */
const ARMED = '#ffd479'
/** A merge that changes nothing but the plate list. */
const FREE = '#5fd08a'
/** A merge that must also flatten a joint / match a column. */
const COERCE = '#ffa94d'

/** Pitch wash: cool where the chain has pitched up, warm where it came back down. */
const PITCH_UP = '#8fc4ff'
const PITCH_DOWN = '#ffb27f'
/** |ψ| that saturates the wash (a panel standing straight up). */
const PITCH_FULL_DEG = 90
const PITCH_MAX_OPACITY = 0.4

/**
 * The rect-specific validation codes, reported INLINE under the map (which is
 * where the click that caused them happened). ControlPanel excludes exactly this
 * set from its general error box, so a rejected placement is explained once.
 *
 * E_SHAPE is deliberately NOT here: it also covers pasted-config problems, which
 * belong in the general box, and an out-of-bounds click is already impossible
 * from the map's own hit grid.
 */
export const PLACEMENT_CODES = new Set([
  'E_FOLD_ON_REMOVED_JOINT',
  'E_CROSSCOL_ANGLE_MISMATCH',
  'E_RECT_OVERLAP',
  'E_MERGE_SHAPE',
  'E_MERGE_BOUNDS',
  'E_MERGE_SAME_CELL',
  'E_MERGE_NOT_ADJACENT',
  'E_MERGE_OCCUPIED',
  'E_MERGE_REJECTED',
])

export default function GridMap() {
  const config = useStore((s) => s.config)
  const removeRectAt = useStore((s) => s.removeRectAt)
  const lastErrors = useStore((s) => s.lastErrors)
  const lastNotice = useStore((s) => s.lastNotice)
  const armedCell = useStore((s) => s.armedCell)
  const armCell = useStore((s) => s.armCell)
  const clearArmed = useStore((s) => s.clearArmed)
  const mergeCells = useStore((s) => s.mergeCells)
  const { layout, violations } = getDerived(config)

  // Escape disarms — the standard way out of a held selection.
  useEffect(() => {
    if (!armedCell) return undefined
    const onKey = (e) => {
      if (e.key === 'Escape') clearArmed()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [armedCell, clearArmed])

  const cols = config.grid.cols
  const rows = config.grid.rows

  const width = PAD_L + cols * CELL + (cols - 1) * GAP + PAD_R
  const height = PAD_T + rows * CELL + (rows - 1) * GAP + PAD_B
  const x = (c) => PAD_L + c * PITCH
  const y = (r) => PAD_T + (rows - 1 - r) * PITCH

  // cell → the panel that owns it (every cell is owned: a cell covered by a
  // vertical plate anchored in the row in front of it is listed in that plate's
  // `cells`, so the lookup is total).
  const ownerOf = new Map()
  for (const panel of layout.panels) {
    for (const [r, c] of panel.cells) ownerOf.set(`${r},${c}`, panel)
  }

  // The armed square's mergeable neighbours, keyed "r,c" — every one of them is a
  // merge core/merge.js has already proven, so clicking it cannot bounce.
  const candidates = new Map()
  if (armedCell) {
    for (const cand of mergeCandidates(config, armedCell)) {
      candidates.set(`${cand.row},${cand.col}`, cand)
    }
  }
  const isArmed = (r, c) => Boolean(armedCell) && armedCell.row === r && armedCell.col === c

  // A cell owned by a '2x2' square counts as EMPTY here: squares are the default
  // state of a cell, so a click either arms it or completes a merge with it.
  const onCellClick = (r, c) => {
    if (ownerOf.get(`${r},${c}`)?.type === '2x4') {
      if (armedCell) clearArmed()
      removeRectAt(r, c) // split the plate back into two squares
      return
    }
    if (isArmed(r, c)) {
      clearArmed()
      return
    }
    if (candidates.has(`${r},${c}`)) {
      mergeCells(armedCell, { row: r, col: c })
      return
    }
    armCell({ row: r, col: c })
  }

  /** Cumulative pitch of cell (r, c) — straight from the solved chain. */
  const pitchOf = (r, c) => {
    const p = layout.columnChains?.[c]?.pitchesDeg?.[r]
    return Number.isFinite(p) ? p : 0
  }

  const placementErrors = lastErrors.filter((e) => PLACEMENT_CODES.has(e.code))

  // The name of the rule this design's plates follow, when it has one.
  const rectPattern =
    typeof config.meta?.rectPattern === 'string' && config.meta.rectPattern.trim() !== ''
      ? config.meta.rectPattern.trim()
      : null

  // Columns whose last panel does not touch the floor (schema.js's grounded-end
  // rule). Marked on the BACK row — the deep end of the strip, drawn at the top.
  const floatingClearance = new Map(violations.map((v) => [v.col, v.clearanceCm]))

  return (
    <section className="grid-map" data-testid="grid-map">
      <header className="grid-map-head">
        <span className="grid-map-title">plan view</span>
        {armedCell && (
          <span className="grid-map-armed-label" data-testid="gridmap-armed-label">
            armed ({armedCell.row}, {armedCell.col}) · {candidates.size} to merge with · Esc
          </span>
        )}
      </header>

      {rectPattern && (
        <p
          className="grid-map-pattern"
          data-testid="rect-pattern"
          title={`plate pattern: the rule this design's ${config.rects.length} plates follow (meta.rectPattern)`}
        >
          plates: <b>{rectPattern}</b> · {config.rects.length}
        </p>
      )}

      <svg
        className="grid-map-svg"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`${cols} by ${rows} panel grid, row 0 (the shore) at the bottom`}
      >
        {/* ---- artwork (never a click target) ---- */}
        <g style={{ pointerEvents: 'none' }}>
          {/* row labels on the RIGHT — the wall owns the left margin now */}
          {Array.from({ length: rows }, (_, r) => (
            <text
              key={r}
              className="grid-map-rowlabel"
              x={x(cols - 1) + CELL + 4}
              y={y(r) + CELL / 2 + 3}
            >
              {r}
            </text>
          ))}

          {/* column indices — in PLAN view column 0 is at the left; the 3D camera
              looks in from the window, so there it is the rightmost strip. */}
          {Array.from({ length: cols }, (_, c) => (
            <text key={`cl-${c}`} className="grid-map-collabel" x={x(c) + CELL / 2} y={PAD_T - 4}>
              {c}
            </text>
          ))}

          {layout.panels.map((panel) => {
            const isRect = panel.type === '2x4'
            const horiz = panel.rectOrientation === 'horizontal'
            const vert = panel.rectOrientation === 'vertical'
            return (
              <rect
                key={panel.id}
                x={x(panel.col)}
                y={vert ? y(panel.row + 1) : y(panel.row)}
                width={horiz ? 2 * CELL + GAP : CELL}
                height={vert ? 2 * CELL + GAP : CELL}
                rx={3}
                fill={isRect ? RECT_FILL : SQUARE_FILL}
                stroke={isRect ? RECT_STROKE : SQUARE_STROKE}
                strokeWidth={isRect ? 1.4 : 1}
              />
            )
          })}

          {/* pitch wash — one tint per cell, so the wave reads in plan */}
          {Array.from({ length: rows }, (_, r) =>
            Array.from({ length: cols }, (_, c) => {
              const psi = pitchOf(r, c)
              const strength = Math.min(Math.abs(psi) / PITCH_FULL_DEG, 1)
              if (strength < 0.02) return null
              return (
                <rect
                  key={`pitch-${r}-${c}`}
                  x={x(c)}
                  y={y(r)}
                  width={CELL}
                  height={CELL}
                  rx={3}
                  fill={psi >= 0 ? PITCH_UP : PITCH_DOWN}
                  fillOpacity={strength * PITCH_MAX_OPACITY}
                />
              )
            }),
          )}

          {/* grounded-end rule: outline the back cell of every floating column */}
          {Array.from(floatingClearance.keys()).map((c) => (
            <g key={`float-${c}`}>
              <rect
                x={x(c) + 1}
                y={y(rows - 1) + 1}
                width={CELL - 2}
                height={CELL - 2}
                rx={2.5}
                fill="none"
                stroke={FLOATING}
                strokeWidth={1.6}
                strokeDasharray="3 2"
              />
              <path
                d={`M ${x(c) + CELL - 12} ${y(rows - 1) + 1} L ${x(c) + CELL - 1} ${y(rows - 1) + 1} L ${x(c) + CELL - 1} ${y(rows - 1) + 12} Z`}
                fill={FLOATING}
              />
            </g>
          ))}

          {/* the armed square, and the neighbours it can be merged with. FREE =
              solid green + "+", COERCE = dashed amber + "~": the difference is
              whether the merge also has to change the surface. */}
          {armedCell && (
            <g data-testid="gridmap-armed">
              <rect
                x={x(armedCell.col) + 1}
                y={y(armedCell.row) + 1}
                width={CELL - 2}
                height={CELL - 2}
                rx={2.5}
                fill={ARMED}
                fillOpacity={0.16}
                stroke={ARMED}
                strokeWidth={2.4}
              />
              <circle cx={x(armedCell.col) + CELL / 2} cy={y(armedCell.row) + CELL / 2} r={3.4} fill={ARMED} />
            </g>
          )}

          {[...candidates.values()].map((cand) => {
            const free = cand.kind === 'free'
            const color = free ? FREE : COERCE
            return (
              <g
                key={`cand-${cand.row}-${cand.col}`}
                data-testid={`candidate-${cand.row}-${cand.col}`}
                data-kind={cand.kind}
              >
                <rect
                  x={x(cand.col) + 1}
                  y={y(cand.row) + 1}
                  width={CELL - 2}
                  height={CELL - 2}
                  rx={2.5}
                  fill={color}
                  fillOpacity={free ? 0.2 : 0.13}
                  stroke={color}
                  strokeWidth={2}
                  strokeDasharray={free ? undefined : '4 2.5'}
                />
                <text
                  className="grid-map-candmark"
                  x={x(cand.col) + CELL / 2}
                  y={y(cand.row) + CELL / 2 + 5}
                  fill={color}
                >
                  {free ? '+' : '~'}
                </text>
              </g>
            )
          })}

          <line
            x1={PAD_L}
            y1={y(0) + CELL + 2.5}
            x2={PAD_L + cols * CELL + (cols - 1) * GAP}
            y2={y(0) + CELL + 2.5}
            stroke={SHORE}
            strokeWidth={1.4}
          />
          <text
            className="grid-map-shore"
            x={PAD_L + (cols * CELL + (cols - 1) * GAP) / 2}
            y={height - 4}
            textAnchor="middle"
          >
            window / shore
          </text>

          {/* the −X wall: the LEFT edge here (beside column 0), the RIGHT edge in
              the 3D view. Grouped so the marker has a non-degenerate bounding box —
              a bare vertical <line> is zero-width and reads as invisible to
              Playwright. */}
          <g data-testid="gridmap-wall">
            <line
              x1={PAD_L - 3}
              y1={PAD_T}
              x2={PAD_L - 3}
              y2={y(0) + CELL}
              stroke={WALL}
              strokeWidth={2.6}
            />
            <text
              className="grid-map-wall"
              transform={`translate(${PAD_L - 10} ${(PAD_T + y(0) + CELL) / 2}) rotate(-90)`}
              textAnchor="middle"
            >
              wall
            </text>
          </g>
        </g>

        {/* ---- uniform 1-cell hit grid on top ---- */}
        <g>
          {Array.from({ length: rows }, (_, r) =>
            Array.from({ length: cols }, (_, c) => {
              const owner = ownerOf.get(`${r},${c}`)
              const occupied = owner?.type === '2x4'
              const floats = r === rows - 1 && floatingClearance.has(c)
              const cand = candidates.get(`${r},${c}`)
              const armed = isArmed(r, c)
              const action = occupied
                ? `click to split the ${owner.rectOrientation} plate back into two squares`
                : armed
                  ? 'armed — click a highlighted neighbour to merge, or click here again to let go'
                  : cand
                    ? `click to merge: ${cand.description}`
                    : armedCell
                      ? 'not mergeable with the armed square — click to arm this one instead'
                      : 'click to arm this square, then click a neighbour to merge them into a 121cm plate'
              return (
                <rect
                  key={`hit-${r}-${c}`}
                  data-testid={`cell-${r}-${c}`}
                  data-merge={cand ? cand.kind : armed ? 'armed' : undefined}
                  className="grid-map-hit"
                  x={x(c)}
                  y={y(r)}
                  width={CELL}
                  height={CELL}
                  fill="#ffffff"
                  fillOpacity={0}
                  onClick={() => onCellClick(r, c)}
                >
                  <title>
                    {(floats
                      ? `end floating — ${floatingClearance.get(c).toFixed(1)}cm off the floor · `
                      : '') + `(${r}, ${c}) · pitch ${Math.round(pitchOf(r, c))}° — ${action}`}
                  </title>
                </rect>
              )
            }),
          )}
        </g>
      </svg>

      <p className="grid-map-hint">
        click a square, then a highlighted neighbour, to <b>combine</b> them into one 60×121 plate —{' '}
        <span className="cand-free">green +</span> is free, <span className="cand-coerce">amber ~</span>{' '}
        also changes geometry (Esc lets go) · click a plate to <b>split</b> it · tint = pitch{' '}
        <span className="wash-up">up</span> / <span className="wash-down">down</span> · col 0: left
        here, <b>right</b> in 3D · the <span className="wash-wall">wall</span> closes the left edge
        (col 0)
      </p>

      {lastNotice && (
        <p className="grid-map-notice" data-testid="gridmap-notice" title="what the last change did">
          {lastNotice}
        </p>
      )}

      {placementErrors.length > 0 && (
        <div className="grid-map-error" data-testid="gridmap-error">
          <strong>plate rejected</strong>
          {placementErrors.map((e, i) => (
            <p key={i}>
              <code>{e.code}</code> {e.message}
            </p>
          ))}
        </div>
      )}
    </section>
  )
}
