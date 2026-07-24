/**
 * grid-designer — 2D plan view of the 6×5 grid, and the place/remove UI for the
 * 60×121 two-cell plates ("rects").
 *
 * ORIENTATION: row 0 is drawn at the BOTTOM, matching the mental model of
 * standing outside the storefront looking in — the shore/window edge is nearest
 * the viewer, rows recede upward (away, +Z). So screen row = rows-1-r.
 *
 * WHAT IS DRAWN
 *   - one shape per PLACED PANEL, read off the derived layout rather than the
 *     config's rect list, so the map always shows what the solver actually
 *     emitted: squares as single cells, horizontal plates spanning two columns
 *     of a row, vertical plates spanning two rows of a column (tinted with the
 *     same cool blue the 3D rect material uses).
 *   - small amber dots on the in-row joints carrying a signed override, so the
 *     hand-tuned joints are visible next to the plates that depend on them.
 *   - a blue rule plus a "window / shore" caption along row 0's outer edge.
 *
 * INTERACTION
 *   A transparent 1-cell hit grid sits on top of the artwork (which is
 *   pointer-events: none), so every one of the 30 cells is a uniform click
 *   target regardless of what shape covers it:
 *     - empty cell → place a plate of the toolbar's current orientation,
 *       anchored there: H spans (r,c)+(r,c+1), V spans (r,c)+(r+1,c)
 *     - any cell of an existing plate → remove that plate
 *   Both go through the store's commit rule, so an illegal placement changes
 *   nothing and its validation errors appear in the control panel's error box
 *   immediately below this map.
 */

import { useState } from 'react'
import useStore, { getDerived } from '../store.js'

// -----------------------------------------------------------------------------
// SVG metrics (user units; the <svg> scales to the panel width via viewBox)
// -----------------------------------------------------------------------------
const CELL = 40
const GAP = 4
const PITCH = CELL + GAP
const PAD_L = 13
const PAD_R = 3
const PAD_T = 3
const PAD_B = 17

const SQUARE_FILL = '#272736'
const SQUARE_STROKE = '#3c3c56'
const RECT_FILL = '#1f3350'
const RECT_STROKE = '#4a7fc4'
const SHORE = '#8fcaff'
const OVERRIDE = '#e8b45a'

const MODES = [
  { id: 'horizontal', label: 'H', hint: 'place a horizontal plate: (r,c) + (r,c+1)' },
  { id: 'vertical', label: 'V', hint: 'place a vertical plate: (r,c) + (r+1,c)' },
]

export default function GridMap() {
  const config = useStore((s) => s.config)
  const addRect = useStore((s) => s.addRect)
  const removeRectAt = useStore((s) => s.removeRectAt)
  const { layout } = getDerived(config)
  const [mode, setMode] = useState('horizontal')

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

  // A cell owned by a '2x2' square counts as EMPTY here: squares are the default
  // state of a cell, so the only thing a click can do there is place a plate.
  const onCellClick = (r, c) => {
    if (ownerOf.get(`${r},${c}`)?.type === '2x4') removeRectAt(r, c)
    else addRect({ row: r, col: c, orientation: mode })
  }

  return (
    <section className="grid-map" data-testid="grid-map">
      <header className="grid-map-head">
        <span className="grid-map-title">plan view</span>
        <span className="grid-map-modes">
          {MODES.map((m) => (
            <button
              key={m.id}
              type="button"
              data-testid={`gridmap-mode-${m.id}`}
              className={`mode-btn${mode === m.id ? ' mode-active' : ''}`}
              title={m.hint}
              aria-pressed={mode === m.id}
              onClick={() => setMode(m.id)}
            >
              {m.label}
            </button>
          ))}
        </span>
      </header>

      <svg
        className="grid-map-svg"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`${cols} by ${rows} panel grid, row 0 (the shore) at the bottom`}
      >
        {/* ---- artwork (never a click target) ---- */}
        <g style={{ pointerEvents: 'none' }}>
          {Array.from({ length: rows }, (_, r) => (
            <text key={r} className="grid-map-rowlabel" x={PAD_L - 4} y={y(r) + CELL / 2 + 3}>
              {r}
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

          {config.rows.flatMap((row, r) =>
            Object.keys(row.jointOverridesDeg ?? {}).map((key) => {
              const j = Number(key)
              if (!Number.isInteger(j) || j < 0 || j > cols - 2) return null
              return (
                <circle
                  key={`ov-${r}-${key}`}
                  cx={x(j) + CELL + GAP / 2}
                  cy={y(r) + CELL / 2}
                  r={2.6}
                  fill={OVERRIDE}
                />
              )
            }),
          )}

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
        </g>

        {/* ---- uniform 1-cell hit grid on top ---- */}
        <g>
          {Array.from({ length: rows }, (_, r) =>
            Array.from({ length: cols }, (_, c) => {
              const owner = ownerOf.get(`${r},${c}`)
              const occupied = owner?.type === '2x4'
              return (
                <rect
                  key={`hit-${r}-${c}`}
                  data-testid={`cell-${r}-${c}`}
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
                    {`(${r}, ${c}) — ${occupied ? `click to remove the ${owner.rectOrientation} plate` : `click to place a ${mode} plate`}`}
                  </title>
                </rect>
              )
            }),
          )}
        </g>
      </svg>

      <p className="grid-map-hint">
        click an empty cell to place a <b>{mode === 'horizontal' ? 'horizontal' : 'vertical'}</b>{' '}
        60×121 plate · click a plate to remove it
      </p>
    </section>
  )
}
