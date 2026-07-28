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
          the <b>cell pitch</b> is <code>60 + 2·gap·cos(θ/2) + 60·cos θ</code>
          {lattice.wave
            ? <> at the <b>base</b> angle — under the wave every edge has its own, and the tables
              below are the truth. The cycles still close exactly, because the height field is
              separable rather than because the pitch is uniform.</>
            : <> and it is the same in x and z — which is why every cycle in the network closes
              exactly, with nothing left over.</>}
        </p>
      </div>

      {/* --- the wave's per-axis edge tables ---------------------------------
          One row per lattice EDGE, which is one ramp. This is the whole content
          of the mode: the angle is no longer a single number, and a panel that
          reported only `lattice.pitchCm` would be quoting edge 0 as though it
          were the design. The total run is stated against the run the same
          lattice would have had at a uniform base angle, because "how much
          shorter did the scrunch make it" is the question the knob is asked. */}
      {lattice.wave && ['x', 'z'].map((ax) => {
        const w = lattice.wave[ax]
        return (
          <div className="report-section" key={ax}>
            <h4 className="report-section-title">wave · {ax} edges</h4>
            {w.edgeCount === 0 ? (
              <p className="form-hint">no {ax} edges — a single line of cells has nothing to scrunch.</p>
            ) : (
              <>
                <div className="metrics-scroll">
                  <table className="metrics-table" data-testid={`metrics-wave-${ax}`}>
                    <thead>
                      <tr>
                        <th>k</th>
                        <th>θ</th>
                        <th>scrunch</th>
                        <th>advance</th>
                        <th>pitch</th>
                        <th>rise</th>
                        <th>{ax} line</th>
                      </tr>
                    </thead>
                    <tbody>
                      {w.angleDeg.map((deg, k) => (
                        <tr key={k}>
                          <td>{k}</td>
                          <td>{deg.toFixed(2)}°</td>
                          <td>{(w.scrunchFactor[k] * 100).toFixed(0)}%</td>
                          <td>{n1(w.advanceCm[k])}</td>
                          <td>{n1(w.pitchCm[k])}</td>
                          <td className={w.riseCm[k] < 0 ? 'metrics-fall' : undefined}>
                            {w.riseCm[k] > 0 ? '+' : ''}
                            {n1(w.riseCm[k])}
                          </td>
                          <td>{n1(w.lineCm[k + 1])}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="form-hint" data-testid={`metrics-wave-${ax}-run`}>
                  total plan run <b>{n1(w.planRunCm)} cm</b> against {n1(w.unscrunchedRunCm)} cm
                  unscrunched — <b>{((1 - w.compression) * 100).toFixed(1)}% shorter</b>. θ runs{' '}
                  {Math.min(...w.angleDeg).toFixed(2)}° to {Math.max(...w.angleDeg).toFixed(2)}°;
                  the base angle is the one at edge 0, at the front.
                </p>
              </>
            )}
          </div>
        )
      })}

      {lattice.wave && (
        <div className="report-section">
          <h4 className="report-section-title">wave · height field</h4>
          <div className="metrics-scroll">
            <table className="metrics-table" data-testid="metrics-wave-heights">
              <thead>
                <tr>
                  <th>i \ j</th>
                  {lattice.wave.heights[0].map((_, j) => <th key={j}>{j}</th>)}
                </tr>
              </thead>
              <tbody>
                {lattice.wave.heights.map((col, i) => (
                  <tr key={i}>
                    <td>{i}</td>
                    {col.map((h, j) => <td key={j}>{n1(h)}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="form-hint">
            <code>h(i,j) = f(i) + g(j)</code>, in cm off the floor, over{' '}
            <b>{lattice.wave.storeyCount} distinct storeys</b>. Separability is not a simplification
            — it is the exact condition for every 4-cycle to close, and the only reason the angles
            are allowed to differ at all.
          </p>
        </div>
      )}

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
          0 on every flat cell and{' '}
          {lattice.wave
            ? 'whatever that ramp\'s own angle gives on every ramp — see the edge tables above'
            : <>±{n1(lattice.riseCm)} on every ramp</>}.
        </p>
      </div>

    </section>
  )
}
