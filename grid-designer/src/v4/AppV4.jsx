/**
 * grid-designer v4 — root shell for the folded-ribbon tool.
 *
 * The same layout as v3's AppV3.jsx (top bar / fixed-width left control column /
 * 3D viewport), rebuilt against the v4 store and the frozen v4 core. src/v3/ is
 * UNTOUCHED and stays on disk, unmounted — HANDOFF §0 retires the approach, not
 * the record of it.
 *
 *   top bar : title + export buttons
 *   left    : the shape controls, the per-unit table, the metrics, the report,
 *             config JSON, saved-design slots (fixed 340px)
 *   main    : the 3D viewport (RibbonViewport.jsx)
 *
 * PANEL ORDER IS AN ARGUMENT, not a layout accident. StripPanel first because
 * units and angle are the two knobs the design is actually driven by; UnitsPanel
 * next because per-unit edits are the second-order moves made against what the
 * first panel produced; then the numbers, then the report — which leads with the
 * envelope, the one thing v3 could never say (report.js's header).
 *
 * The store is exposed as `window.__gridDesignerStoreV4` with a
 * `window.__gridDesignerDerivedV4()` helper, mirroring v3's convention, so the
 * whole model can be driven and read from the browser console.
 *
 * UNDO / REDO shortcuts: Cmd/Ctrl+Z / +Shift+Z, inert while focus is in a text
 * field or number input so the browser's own text-undo wins there.
 */

import { useEffect } from 'react'
import useStoreV4, { getDerived } from './store.js'
import ExportButtons from './ExportButtons.jsx'
import StripPanel from './StripPanel.jsx'
import LatticeMap from './LatticeMap.jsx'
import MetricsPanel from './MetricsPanel.jsx'
import ReportPanel from './ReportPanel.jsx'
import JsonPanel from './JsonPanel.jsx'
import SlotsPanel from './SlotsPanel.jsx'
import RibbonViewport from './RibbonViewport.jsx'

/** Is focus somewhere the browser's own text undo is what Cmd+Z should mean? */
function inTextField(target) {
  if (!target || typeof target !== 'object') return false
  const tag = target.tagName
  if (tag === 'TEXTAREA') return true
  if (tag === 'INPUT') {
    const type = (target.type ?? '').toLowerCase()
    return type === 'number' || type === 'text' || type === 'search'
  }
  return Boolean(target.isContentEditable)
}

export default function AppV4() {
  useEffect(() => {
    window.__gridDesignerStoreV4 = useStoreV4
    window.__gridDesignerDerivedV4 = () => getDerived(useStoreV4.getState().config)
  }, [])

  useEffect(() => {
    const onKey = (e) => {
      if (!(e.metaKey || e.ctrlKey) || e.altKey) return
      if (e.key.toLowerCase() !== 'z') return
      if (inTextField(e.target)) return
      e.preventDefault()
      const { undo, redo } = useStoreV4.getState()
      if (e.shiftKey) redo()
      else undo()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const lastErrors = useStoreV4((s) => s.lastErrors)

  return (
    <div className="app">
      <header className="top-bar">
        <h1 className="app-title">grid-designer</h1>
        <span className="app-subtitle">v4 — the folded ribbon</span>
        <ExportButtons />
      </header>
      <div className="main-layout">
        <aside className="control-panel control-panel-v3">
          <StripPanel />

          {lastErrors.length > 0 && (
            <div className="msg-list msg-errors" data-testid="v4-last-errors">
              <strong>change rejected</strong>
              {lastErrors.map((e, i) => (
                <p key={i}>
                  <code>{e.code}</code> {e.message}
                </p>
              ))}
            </div>
          )}

          <LatticeMap />
          <MetricsPanel />
          <ReportPanel />
          <JsonPanel />
          <SlotsPanel />
        </aside>
        <main className="viewport-wrap">
          <RibbonViewport />
        </main>
      </div>
    </div>
  )
}
