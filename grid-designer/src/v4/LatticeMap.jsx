/**
 * grid-designer v4 — the plan-view lattice editor.
 *
 * THIS IS THE EDITOR. Everything else in the left column is a knob or a
 * readout; this is where the network is actually shaped.
 *
 * =============================================================================
 * WHY A GRID AND NOT A LIST
 * =============================================================================
 * The ribbon's UnitsPanel was a list, because a chain of nine panels IS a list.
 * A network is not — it is a lattice with two kinds of thing on it, and the
 * question you ask of it ("what happens if I take that corner off?") is
 * spatial. So this is a plan view, and clicking it edits the design directly.
 *
 * Growing the network panel by panel and editing a generated one are THE SAME
 * OPERATION here, which is the whole reason the lattice model was chosen
 * (V4_SPEC §9.7): switching a cell on at a free face lands it on the lattice by
 * construction, so it closes exactly. There is no "add panel" mode, no hinge to
 * drag, and nothing to reconcile afterwards.
 *
 * =============================================================================
 * WHAT IS DRAWN, AND THE MIRROR
 * =============================================================================
 * The lattice is interleaved: flat CELLS on even indices, RAMPS on the odd ones
 * between them, and the odd/odd positions left empty — those are the corner
 * holes, which are real (V4_SPEC §9.1) rather than a rendering gap. Drawing
 * them as blanks is the honest picture: four ramps meet at a lattice corner
 * without touching.
 *
 *   WALL (x = 0) is the LEFT edge, i increasing rightward.
 *   WINDOW / SHORE (z = 0) is the BOTTOM edge, j increasing upward.
 *
 * That is v3's TilingMap convention and the OPPOSITE of the 3D viewport, which
 * looks in from the window and therefore shows the wall on the right. Both
 * files' headers say so because it trips everyone; do not "fix" either one.
 *
 * =============================================================================
 * THE THREE EDITS
 * =============================================================================
 * EVERY TILE IS ONE PANEL, and clicking it removes or restores exactly that
 * panel. That is the whole interaction, and it took a rule change in the core
 * to be true: a ramp used to need BOTH its cells, so removing a flat silently
 * took up to four angled panels with it and the grid was not really editing
 * panels at all. A ramp now needs ONE cell (V4_SPEC §9.3) and cantilevers off
 * its single joint when the other end goes — which is exactly the shape of a
 * wall anchor, so the model already had the precedent.
 *
 * ⇧-click flips a flat cell. A flip is a physical statement about every joint
 * the cell has, since a ramp cannot be flipped to match; the store's notice
 * says so at the moment of the click.
 *
 * Nothing you do here moves anything else — the lattice is generated, not
 * chained, so every other panel keeps a bit-identical position.
 */

import useStoreV4, { getDerived, overrideFor, edgeOverrideFor } from './store.js'

/** Is this ramp held at BOTH ends, or cantilevered off one? */
function bothCells(config, chain, i, j, axis) {
  const hi = axis === 'x' ? { i: i + 1, j } : { i, j: j + 1 }
  return overrideFor(config, i, j).present && overrideFor(config, hi.i, hi.j).present
}

/** Cells sit on even indices, ramps on odd ones, so a `cols` × `rows` lattice
 *  needs `2·cols − 1` columns of tiles. */
const span = (n) => Math.max(1, 2 * n - 1)

/**
 * The grid is drawn with a RING of empty slots one cell wide all the way round,
 * so the rectangle can be grown from the same click that shrinks it. Without it
 * you could only add panels back inside the bounding box you already had, and
 * the only way to reach further was the cols/rows steppers — which add a whole
 * row at a time and cannot reach the wall or window sides at all.
 *
 * Tile index −2 is the ring cell at i = −1; index 2·cols is the one at i = cols.
 */
const RING = 2

export default function LatticeMap() {
  const config = useStoreV4((s) => s.config)
  const toggleCell = useStoreV4((s) => s.toggleCell)
  const toggleCellFlipped = useStoreV4((s) => s.toggleCellFlipped)
  const toggleEdge = useStoreV4((s) => s.toggleEdge)
  const clearOverrides = useStoreV4((s) => s.clearOverrides)
  const trimLattice = useStoreV4((s) => s.trimLattice)
  const setHoveredUnit = useStoreV4((s) => s.setHoveredUnit)
  const hoveredUnitId = useStoreV4((s) => s.hoveredUnitId)
  const notice = useStoreV4((s) => s.lastActionNotice)

  const { chain, report } = getDerived(config)
  const { cols, rows } = config.lattice
  const levels = chain.lattice.levels
  const counts = report.metrics.counts

  // Which panels actually exist, so a tile can show "switched off" apart from
  // "impossible" — a ramp whose cell is gone is neither present nor editable.
  const live = new Set(chain.panels.filter((p) => p.present).map((p) => p.id))

  // Panels running through a column or the like. Marked here because this is
  // the panel-by-panel view, and "switch that one off" is the fix.
  const fouling = new Set(report.obstacles.flatMap((o) => o.hits.map((h) => h.id)))

  const editCount =
    (config.overrides?.cells ?? []).length + (config.overrides?.edges ?? []).length

  const tiles = []
  for (let jj = span(rows) - 1 + RING; jj >= -RING; jj--) {
    for (let ii = -RING; ii < span(cols) + RING; ii++) {
      // `>> 1` floors toward zero, which is wrong for the negative ring, so the
      // ring's cell index is derived rather than shifted.
      const i = ii >= 0 ? ii >> 1 : -1
      const j = jj >= 0 ? jj >> 1 : -1
      const cellCol = ((ii % 2) + 2) % 2 === 0
      const cellRow = ((jj % 2) + 2) % 2 === 0
      const key = `${ii}-${jj}`

      // Outside the rectangle: only the CELL slots are offerable. A ramp cannot
      // exist without a cell, so the ring's ramp positions are simply blank.
      const outside = i < 0 || j < 0 || i >= cols || j >= rows
      if (outside) {
        if (cellCol && cellRow) {
          tiles.push(
            <button
              key={key}
              type="button"
              className="lm-cell lm-off lm-ring"
              data-testid={`lm-add-${i}-${j}`}
              onClick={() => toggleCell(i, j)}
              title={`grow the network here — adds one panel at (${i}, ${j}) and nothing else`}
            >
              +
            </button>,
          )
        } else {
          tiles.push(<div key={key} className="lm-hole" />)
        }
        continue
      }

      // odd/odd — the corner hole. Not a gap in the drawing: nothing is there.
      if (!cellCol && !cellRow) {
        tiles.push(<div key={key} className="lm-hole" title="corner hole — four ramps meet here without touching" />)
        continue
      }

      if (cellCol && cellRow) {
        const id = `Ci${i}j${j}`
        const ov = overrideFor(config, i, j)
        const high = levels[i]?.[j] === 1
        const on = ov.present
        const cls = [
          'lm-cell',
          high ? 'lm-high' : 'lm-ground',
          on ? '' : 'lm-off',
          ov.flipped ? 'lm-flipped' : '',
          fouling.has(id) ? 'lm-obstacle' : '',
          hoveredUnitId === id ? 'lm-hot' : '',
        ].filter(Boolean).join(' ')
        tiles.push(
          <button
            key={key}
            type="button"
            className={cls}
            data-testid={`lm-cell-${i}-${j}`}
            onMouseEnter={() => setHoveredUnit(id)}
            onMouseLeave={() => setHoveredUnit(null)}
            onClick={(e) => (e.shiftKey ? toggleCellFlipped(i, j) : toggleCell(i, j))}
            title={
              `cell (${i}, ${j}) — ${high ? 'high' : 'ground'}${ov.flipped ? ', flipped' : ''}\n` +
              `${on ? 'click to remove' : 'click to add'} · shift-click to flip`
            }
          >
            {on ? (ov.flipped ? '⊘' : high ? 'H' : 'G') : '+'}
          </button>,
        )
        continue
      }

      // A ramp: odd column → an x-edge, odd row → a z-edge.
      const axis = cellRow ? 'x' : 'z'
      const id = `Ei${i}j${j}${axis}`
      const ov = edgeOverrideFor(config, i, j, axis)
      const exists = live.has(id)
      const orphaned = !exists && ov.present // both-cells rule killed it, not the user
      // `hanging` is a ramp held by only one cell — a real panel, cantilevered,
      // and worth showing apart from a fully supported one because it is the
      // state a ragged edge is made of.
      const hanging = exists && !bothCells(config, chain, i, j, axis)
      const cls = ['lm-ramp', `lm-ramp-${axis}`, ov.present ? '' : 'lm-off', orphaned ? 'lm-orphan' : '',
        hanging ? 'lm-hanging' : '',
        fouling.has(id) ? 'lm-obstacle' : '',
        hoveredUnitId === id ? 'lm-hot' : ''].filter(Boolean).join(' ')
      tiles.push(
        <button
          key={key}
          type="button"
          className={cls}
          data-testid={`lm-edge-${i}-${j}-${axis}`}
          disabled={orphaned}
          onMouseEnter={() => exists && setHoveredUnit(id)}
          onMouseLeave={() => setHoveredUnit(null)}
          onClick={() => toggleEdge(i, j, axis)}
          title={
            orphaned
              ? 'no ramp here — both the cells it could hang from are gone'
              : `angled panel (${i}, ${j}) ${axis}${hanging ? ', cantilevered off one end' : ''}` +
                ` — click to ${ov.present ? 'remove' : 'restore'}`
          }
        >
          {orphaned ? '' : ov.present ? (axis === 'x' ? '╱' : '╲') : '+'}
        </button>,
      )
    }
  }

  return (
    <section className="grid-map lattice-map" data-testid="lattice-map">
      <header className="grid-map-head">
        <span className="grid-map-title">plan · click to edit</span>
        <div className="form-panel-history">
          <button
            type="button"
            className="tool-btn"
            data-testid="lattice-trim"
            onClick={() => trimLattice()}
            title="shrink the rectangle to the cells actually in use — no panel moves"
          >
            trim
          </button>
          <button
            type="button"
            className="tool-btn"
            data-testid="lattice-clear"
            disabled={editCount === 0}
            onClick={() => clearOverrides()}
            title="restore every cell and ramp inside the rectangle"
          >
            reset grid
          </button>
        </div>
      </header>

      <div
        className="lm-grid"
        style={{
          // The RING adds a slot either side, so the column count must include
          // it — emitting more tiles than the template has columns silently
          // wraps them and scrambles the whole plan.
          gridTemplateColumns: `repeat(${span(cols) + 2 * RING}, ${cols > 5 ? '1.4rem' : '1.75rem'})`,
        }}
      >
        {tiles}
      </div>

      <p className="grid-map-hint">
        <b>every tile is one panel — click it to take it out or put it back.</b> The big squares are
        the flat cells (<code>G</code> ground, <code>H</code> high); the ones between them are the
        angled panels (<code>╱</code> <code>╲</code>). <code>+</code> is an empty slot: click it to
        grow the network there. <b>Shift-click</b> a flat to flip it.
      </p>
      <p className="grid-map-hint">
        An angled panel needs <b>one</b> flat, not two — take a flat out and its four ramps stay,
        hanging off their far ends (shown <span className="lm-legend-hanging">dimmer</span>). Only a
        ramp with nothing left at either end disappears. To build <i>outside</i> the rectangle, add a
        row or column with <b>cells x / cells z</b> above and switch off what you do not want.
      </p>
      <p className="grid-map-hint">
        <b>wall</b> is the left edge, <b>window</b> the bottom — the mirror of the 3D view, which
        looks in from the window. Nothing you do here moves anything else.
      </p>

      {report.obstacles.map((o) => (
        <p
          key={o.id}
          className={o.hitCount ? 'grid-map-error' : 'grid-map-hint'}
          data-testid={`lm-obstacle-${o.id}`}
        >
          <b>{o.label}</b> — x {o.extents.min[0]}–{o.extents.max[0]}, z {o.extents.min[2]}–
          {o.extents.max[2]} cm.{' '}
          {o.hitCount ? (
            <>
              <b>{o.hitCount} panel{o.hitCount === 1 ? '' : 's'} run through it</b> (outlined red) —
              click them off, or move the network with the offsets.
            </>
          ) : (
            <>Clear — nearest panel {o.nearestClearanceCm === null ? '—' : `${o.nearestClearanceCm.toFixed(0)}cm`} away.</>
          )}
        </p>
      ))}

      <p className="grid-map-hint">
        {counts.cells} cells ({counts.groundCells} ground, {counts.highCells} high) ·{' '}
        {counts.ramps} ramps{counts.anchorRamps ? ` + ${counts.anchorRamps} wall anchors` : ''} ·{' '}
        <b>{counts.panels} panels</b>, {counts.joints} joints
        {counts.absent ? ` · ${counts.absent} slot${counts.absent === 1 ? '' : 's'} empty` : ''}
      </p>

      {notice && (
        <div className="grid-map-notice" data-testid="lattice-notice">
          {notice}
        </div>
      )}
    </section>
  )
}
