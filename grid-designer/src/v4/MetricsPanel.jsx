/**
 * grid-designer v4 — the measuring box, and the per-axis breakdown it hides.
 *
 * V4_SPEC §5. The overall W × H × D is the number everyone asks for first and
 * the least informative one in the file: it is the box the ribbon fits in, and
 * a ribbon is mostly not in its box. So it is stated once, at the top, and then
 * taken apart three ways — because "how long is this thing" has three different
 * true answers and the interesting one is usually not the box.
 *
 *   per strip (column)   the plan run, the height, and — the number the folding
 *                        is FOR — the developed length beside the compression
 *                        ratio. `developedLengthCm` is `n·L + (n−1)·gap`: the
 *                        flat material you cut, whether or not every unit ends
 *                        up present. The ratio is how much of that run the
 *                        folding bought back; 1.0 is a flat strip lying down,
 *                        and the smaller it gets the more sheet is standing up.
 *   per row (unit index) the x extents across strips. With one strip this is
 *                        just the panel width, and it stays visible anyway —
 *                        it is the axis the sideways build will populate
 *                        (V4_SPEC §8), and a column of 60s now is the honest
 *                        baseline for that.
 *   per unit             z start/end, plan run, y start/end, rise. Taken from
 *                        the REFERENCE segment rather than the solid, so the
 *                        rise agrees with the tilt it illustrates instead of
 *                        carrying half a housing thickness.
 *
 * Absent units are dimmed rather than dropped, matching UnitsPanel: they still
 * have a position (removal does not re-solve the chain) and they still cost
 * material, and both of those are things the table is for.
 */

import useStoreV4, { getDerived } from './store.js'

const n1 = (v) => (Number.isFinite(v) ? v.toFixed(1) : '—')

export default function MetricsPanel() {
  const config = useStoreV4((s) => s.config)
  const hoveredUnitId = useStoreV4((s) => s.hoveredUnitId)
  const setHoveredUnit = useStoreV4((s) => s.setHoveredUnit)
  const { report } = getDerived(config)
  const { overall, columns, rows, panels, counts, material, lattice } = report.metrics
  const spacers = report.spacers
  const [w, h, d] = overall.size

  return (
    <section className="metrics-panel" data-testid="metrics-panel">
      <header className="form-panel-head">
        <span className="grid-map-title">metrics</span>
      </header>

      {/* --- the box ------------------------------------------------------- */}
      <div className="report-metric-grid" data-testid="metrics-overall">
        <div className="report-metric">
          <span className="report-metric-label">width (x)</span>
          <span className="report-metric-value">{n1(w)} cm</span>
        </div>
        <div className="report-metric">
          <span className="report-metric-label">height (y)</span>
          <span className="report-metric-value">{n1(h)} cm</span>
        </div>
        <div className="report-metric">
          <span className="report-metric-label">depth (z)</span>
          <span className="report-metric-value">{n1(d)} cm</span>
        </div>
        <div className="report-metric">
          <span className="report-metric-label">from window</span>
          <span className="report-metric-value">
            {n1(overall.min[2])}–{n1(overall.max[2])}
          </span>
        </div>
      </div>

      {/* --- what the lattice is made of ------------------------------------- */}
      <div className="report-section">
        <h4 className="report-section-title">the lattice</h4>
        <div className="report-metric-grid" data-testid="metrics-counts">
          <div className="report-metric">
            <span className="report-metric-label">panels</span>
            <span className="report-metric-value">{counts.panels}</span>
          </div>
          <div className="report-metric">
            <span className="report-metric-label">flat cells</span>
            <span className="report-metric-value">
              {counts.groundCells} + {counts.highCells}
            </span>
          </div>
          <div className="report-metric">
            <span className="report-metric-label">ramps</span>
            <span className="report-metric-value">
              {counts.ramps}
              {counts.anchorRamps ? ` + ${counts.anchorRamps}` : ''}
            </span>
          </div>
          <div className="report-metric">
            <span className="report-metric-label">joints</span>
            <span className="report-metric-value">{counts.joints}</span>
          </div>
          {/* The posts under the cells that rest on the floor. Counted here
              rather than in the report panel because it is a quantity of parts,
              which is what this table is — and a count that goes to zero is how
              you notice grounding is off. */}
          <div className="report-metric" data-testid="metrics-spacers">
            <span className="report-metric-label">spacers</span>
            <span className="report-metric-value">
              {spacers.count}
              {spacers.count > 0 ? ` @ ${n1(spacers.clearanceCm)}cm` : ''}
            </span>
          </div>
          <div className="report-metric">
            <span className="report-metric-label">cell pitch</span>
            <span className="report-metric-value">{n1(lattice.pitchCm)} cm</span>
          </div>
          <div className="report-metric">
            <span className="report-metric-label">level rise</span>
            <span className="report-metric-value">{n1(lattice.riseCm)} cm</span>
          </div>
          <div className="report-metric">
            <span className="report-metric-label">sheet used</span>
            <span className="report-metric-value">{n1(material.materialAreaCm2 / 10000)} m²</span>
          </div>
          <div className="report-metric">
            <span className="report-metric-label">plan covered</span>
            <span className="report-metric-value">{n1(material.planAreaCm2 / 10000)} m²</span>
          </div>
        </div>
        <p className="form-hint">
          the <b>cell pitch</b> is <code>60 + 2·gap·cos(θ/2) + 60·cos θ</code> and it is the same in
          x and z — which is why every cycle in the network closes exactly, with nothing left over.
        </p>
      </div>

      {/* --- per column (i, running away from the window) -------------------- */}
      <div className="report-section">
        <h4 className="report-section-title">per column — along z</h4>
        <div className="metrics-scroll">
          <table className="metrics-table" data-testid="metrics-columns">
            <thead>
              <tr>
                <th>i</th>
                <th>x from</th>
                <th>x to</th>
                <th>plan run</th>
                <th>height</th>
                <th>panels</th>
                <th>cells</th>
              </tr>
            </thead>
            <tbody>
              {columns.map((c) => (
                <tr key={c.i} className={c.presentCells === 0 ? 'metrics-row-absent' : undefined}>
                  <td>{c.i}</td>
                  <td>{n1(c.xMinCm)}</td>
                  <td>{n1(c.xMaxCm)}</td>
                  <td>{n1(c.planRunCm)}</td>
                  <td>{n1(c.heightCm)}</td>
                  <td>{c.panelCount}</td>
                  <td>
                    {c.presentCells} / {c.cellSlots}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="form-hint">
          a column is one strip of the lattice running from the window back — the ribbon this tool
          started as. All lengths cm.
        </p>
      </div>

      {/* --- per row (j, running away from the wall) -------------------------- */}
      <div className="report-section">
        <h4 className="report-section-title">per row — along x</h4>
        <div className="metrics-scroll">
          <table className="metrics-table" data-testid="metrics-rows">
            <thead>
              <tr>
                <th>j</th>
                <th>z from</th>
                <th>z to</th>
                <th>width</th>
                <th>height</th>
                <th>panels</th>
                <th>cells</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.j} className={r.presentCells === 0 ? 'metrics-row-absent' : undefined}>
                  <td>{r.j}</td>
                  <td>{n1(r.zMinCm)}</td>
                  <td>{n1(r.zMaxCm)}</td>
                  <td>{n1(r.widthCm)}</td>
                  <td>{n1(r.heightCm)}</td>
                  <td>{r.panelCount}</td>
                  <td>
                    {r.presentCells} / {r.cellSlots}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* --- per panel -------------------------------------------------------- */}
      <div className="report-section">
        <h4 className="report-section-title">per panel</h4>
        <div className="metrics-scroll">
          <table className="metrics-table metrics-table-units" data-testid="metrics-panels">
            <thead>
              <tr>
                <th>id</th>
                <th />
                <th>x</th>
                <th>z</th>
                <th>run</th>
                <th>y from</th>
                <th>y to</th>
                <th>rise</th>
              </tr>
            </thead>
            <tbody>
              {panels.map((u) => (
                <tr
                  key={u.id}
                  className={[
                    u.present ? '' : 'metrics-row-absent',
                    hoveredUnitId === u.id ? 'metrics-row-hover' : '',
                  ]
                    .filter(Boolean)
                    .join(' ')}
                  onMouseEnter={() => setHoveredUnit(u.id)}
                  onMouseLeave={() => setHoveredUnit(null)}
                >
                  <td>
                    {u.kind === 'cell' ? `${u.i},${u.j}` : `${u.i},${u.j}${u.axis}`}
                    {u.anchor && <span className="unit-anchor">⌐</span>}
                  </td>
                  <td className="unit-glyph">{u.glyph}</td>
                  <td>{n1(u.xStartCm)}</td>
                  <td>{n1(u.zStartCm)}</td>
                  <td>{n1(u.planRunCm)}</td>
                  <td>{n1(u.yStartCm)}</td>
                  <td>{n1(u.yEndCm)}</td>
                  <td className={u.riseCm < 0 ? 'metrics-fall' : undefined}>
                    {u.riseCm > 0 ? '+' : ''}
                    {n1(u.riseCm)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="form-hint">
          x and z are read off the reference plane — the panels' lit face — from the wall and the
          window respectively. <b>rise</b> is the tilt acting over the panel's own length, so it is
          0 on every flat cell and ±{n1(lattice.riseCm)} on every ramp.
        </p>
      </div>

    </section>
  )
}
