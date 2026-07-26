/**
 * grid-designer — left control column.
 *
 * Top: the joint-report summary badge (total / ok / flagged) plus the worst
 * measured deviations, so the cost of a fold pattern is visible while dragging.
 * The badge row also carries the GROUNDED-END verdict for the whole design — a red
 * "N floating" badge when any column's last panel ends in mid-air, a quiet "all
 * grounded ✓" tick otherwise (see the rule in core/schema.js). The E_END_FLOATING
 * messages themselves get their own box below the map, headed with the rule.
 * Under that sits the design's OVERALL SIZE — `width × height × depth` in whole
 * centimetres, the same box the 3D view draws (core/placement.js `layoutBounds`) —
 * so the numbers a design is tuned against are readable without hunting in 3D.
 * Then the 2D grid map (where the 60×121 plates are placed and removed), the
 * message boxes, the cross-column toolbar, one control block per COLUMN (each
 * column is a fold strip: one slider per hinge), and the config-JSON disclosure.
 *
 * `lastErrors` holds the errors from the most recent REJECTED action — the store
 * keeps the previous config in that case, so the UI stays consistent while
 * explaining why nothing moved. The three RECT-specific codes are left to the grid
 * map, which reports them inline under its own artwork (see PLACEMENT_CODES); this
 * box takes everything else (bad pastes, preset failures), so no rejection is
 * ever explained twice in a 320px column.
 *
 * The selected column is UI-only state and lives here rather than in the store:
 * it is the source column for the toolbar's "Copy col → all" and changes nothing
 * about the design.
 */

import { useMemo, useState } from 'react'
import useStore, { getDerived } from '../store.js'
import { isStorageAvailable } from '../persistence.js'
import GridMap, { PLACEMENT_CODES } from './GridMap.jsx'
import JsonPanel from './JsonPanel.jsx'
import SlotsPanel from './SlotsPanel.jsx'
import ColumnControls, { ColumnToolbar, chainBounds } from './ColumnControls.jsx'

export default function ControlPanel() {
  const config = useStore((s) => s.config)
  const lastErrors = useStore((s) => s.lastErrors)
  const lastWarnings = useStore((s) => s.lastWarnings)
  // `overall` is the whole design's box; `bounds` further down is the SPARKLINE scale
  const { layout, report, violations, bounds: overall } = getDerived(config)
  const { summary } = report
  const colCount = config.grid.cols
  const floating = violations.length

  // The rect-specific rejections are shown by the map itself, right where the
  // click happened; showing them again here would just eat the panel's height.
  const panelErrors = lastErrors.filter((e) => !PLACEMENT_CODES.has(e.code))

  const [selected, setSelected] = useState(0)
  // One scale for all six sparklines, so their heights compare directly.
  const bounds = useMemo(() => chainBounds(layout), [layout])
  const selectedCol = Math.min(selected, colCount - 1)
  // A real probe (set + remove a throwaway key), not just "does the API exist" —
  // private browsing can expose localStorage but throw on first write. Checked
  // once per mount rather than per config change; storage availability doesn't
  // flip mid-session in practice.
  const autosaveOk = useMemo(() => isStorageAvailable(), [])

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
        {floating > 0 ? (
          <span
            className="badge-item badge-flagged"
            data-testid="badge-floating"
            title={`${floating} column(s) end in mid-air — every column's last panel must come back down and touch the floor. Use "Ground all", or fold the last hinges down by hand.`}
          >
            <b>{floating}</b> floating
          </span>
        ) : (
          <span className="badge-item badge-grounded" data-testid="badge-grounded" title="every column's last panel touches the floor">
            all grounded ✓
          </span>
        )}
        {autosaveOk ? (
          <span
            className="badge-item badge-grounded"
            data-testid="badge-autosave"
            title="the working design is saved to this browser automatically — a reload (including a dev hot reload) restores it"
          >
            autosave ✓
          </span>
        ) : (
          <span
            className="badge-item badge-flagged"
            data-testid="badge-autosave"
            title="localStorage is unavailable (private browsing, disabled storage, or a full quota) — nothing is being autosaved. Copy or download the JSON before reloading."
          >
            autosave off
          </span>
        )}
      </div>
      <p className="report-detail">
        {layout.panels.length} panels · worst gap dev {summary.worstGapDeviation.toFixed(2)}cm ·
        worst skew {summary.worstSkew.toFixed(1)}°
      </p>
      <p
        className="report-detail report-bounds"
        data-testid="bounds-summary"
        title="overall extent of the built surface (width × height × depth), housings included — the same box the 3D view draws. Peak height is the second number."
      >
        {Math.round(overall.size[0])} × {Math.round(overall.size[1])} ×{' '}
        {Math.round(overall.size[2])} cm
      </p>

      <GridMap />

      {panelErrors.length > 0 && (
        <div className="msg-list msg-errors" data-testid="last-errors">
          <strong>change rejected</strong>
          {panelErrors.map((e, i) => (
            <p key={i}>
              <code>{e.code}</code> {e.message}
            </p>
          ))}
        </div>
      )}

      {violations.length > 0 && (
        <div className="msg-list msg-errors" data-testid="layout-violations">
          <strong>the wave must return to the water</strong>
          {violations.map((v, i) => (
            <p key={i}>
              <code>{v.code}</code> {v.message}
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

      <ColumnToolbar selected={selectedCol} floating={floating} rows={config.grid.rows} />

      <div className="cols-scroll">
        {Array.from({ length: colCount }, (_, c) => (
          <ColumnControls
            key={c}
            c={c}
            bounds={bounds}
            selected={c === selectedCol}
            onSelect={setSelected}
          />
        ))}
      </div>

      <SlotsPanel />
      <JsonPanel />
    </aside>
  )
}
