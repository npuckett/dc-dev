/**
 * grid-designer — left control column.
 *
 * Top: the joint-report summary badge (total / ok / flagged) plus the worst
 * measured deviations, so the cost of a fold pattern is visible while dragging.
 * Then the 2D grid map (where the 60×121 plates are placed and removed), the
 * message boxes, one `RowControls` block per row with the row-to-row fold slider
 * between consecutive rows, and the config-JSON disclosure at the bottom.
 *
 * `lastErrors` shows the errors from the most recent REJECTED action — the store
 * keeps the previous config in that case, so the UI stays consistent while
 * explaining why nothing moved. It sits directly under the grid map on purpose:
 * a rejected plate placement is the most common way to see it.
 */

import useStore, { getDerived } from '../store.js'
import GridMap from './GridMap.jsx'
import JsonPanel from './JsonPanel.jsx'
import RowControls from './RowControls.jsx'

export default function ControlPanel() {
  const config = useStore((s) => s.config)
  const lastErrors = useStore((s) => s.lastErrors)
  const lastWarnings = useStore((s) => s.lastWarnings)
  const { layout, report } = getDerived(config)
  const { summary } = report
  const rowCount = config.grid.rows

  return (
    <aside className="control-panel">
      <div className="report-badge" data-testid="report-summary">
        <span className="badge-item">
          <b>{summary.total}</b> joints
        </span>
        <span className="badge-item badge-ok">
          <b>{summary.ok}</b> ok
        </span>
        <span className={`badge-item${summary.flagged > 0 ? ' badge-flagged' : ''}`}>
          <b>{summary.flagged}</b> flagged
        </span>
      </div>
      <p className="report-detail">
        {layout.panels.length} panels · worst gap dev {summary.worstGapDeviation.toFixed(2)}cm ·
        worst skew {summary.worstSkew.toFixed(1)}°
      </p>

      <GridMap />

      {lastErrors.length > 0 && (
        <div className="msg-list msg-errors" data-testid="last-errors">
          <strong>change rejected</strong>
          {lastErrors.map((e, i) => (
            <p key={i}>
              <code>{e.code}</code> {e.message}
            </p>
          ))}
        </div>
      )}

      {(lastWarnings.length > 0 || layout.warnings.length > 0) && (
        <div className="msg-list msg-warnings">
          {[...lastWarnings, ...layout.warnings].map((w, i) => (
            <p key={i}>
              <code>{w.code}</code> {w.message}
            </p>
          ))}
        </div>
      )}

      <div className="rows-scroll">
        {Array.from({ length: rowCount }, (_, r) => (
          <RowControls key={r} r={r} />
        ))}
      </div>

      <JsonPanel />
    </aside>
  )
}
