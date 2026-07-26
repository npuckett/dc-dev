/**
 * grid-designer v3 — root shell for the drift-surface tool.
 *
 * Same overall layout as v2's src/App.jsx (top bar / left control column / 3D
 * viewport), rebuilt against the v3 store and v3 core (V3_SPEC.md). v2's
 * App.jsx is UNTOUCHED — this is the parallel v3 entry point src/main.jsx
 * mounts once everything here works.
 *
 *   top bar : title + export buttons
 *   left    : drift-form controls, tiling plan view, joint/fit report,
 *             config JSON, saved-design slots (fixed 340px — a hair wider
 *             than v2's 320px, because the report panel's metric grid needs
 *             the room)
 *   main    : the 3D viewport (DriftViewport.jsx)
 *
 * The store is exposed as `window.__gridDesignerStoreV3` for console-driven
 * inspection, mirroring v2's `window.__gridDesignerStore` convention.
 *
 * UNDO / REDO shortcuts: same behaviour as v2's App.jsx — Cmd/Ctrl+Z / +Shift+Z,
 * inert while focus is in a text field or number input so the browser's own
 * text-undo wins there.
 */

import { useEffect } from 'react'
import useStoreV3, { getDerived } from './store.js'
import ExportButtons from './ExportButtons.jsx'
import FormPanel from './FormPanel.jsx'
import PresetBar from './PresetBar.jsx'
import TilingMap from './TilingMap.jsx'
import ReportPanel from './ReportPanel.jsx'
import JsonPanel from './JsonPanel.jsx'
import SlotsPanel from './SlotsPanel.jsx'
import DriftViewport from './DriftViewport.jsx'

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

export default function AppV3() {
  useEffect(() => {
    window.__gridDesignerStoreV3 = useStoreV3
    window.__gridDesignerDerivedV3 = () => getDerived(useStoreV3.getState().config)
  }, [])

  useEffect(() => {
    const onKey = (e) => {
      if (!(e.metaKey || e.ctrlKey) || e.altKey) return
      if (e.key.toLowerCase() !== 'z') return
      if (inTextField(e.target)) return
      e.preventDefault()
      const { undo, redo } = useStoreV3.getState()
      if (e.shiftKey) redo()
      else undo()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const lastErrors = useStoreV3((s) => s.lastErrors)

  return (
    <div className="app">
      <header className="top-bar">
        <h1 className="app-title">grid-designer</h1>
        <span className="app-subtitle">v3 — snow drift, tiled by rigid panels</span>
        <ExportButtons />
      </header>
      <div className="main-layout">
        <aside className="control-panel control-panel-v3">
          <PresetBar />
          <FormPanel />

          {lastErrors.length > 0 && (
            <div className="msg-list msg-errors" data-testid="v3-last-errors">
              <strong>change rejected</strong>
              {lastErrors.map((e, i) => (
                <p key={i}>
                  <code>{e.code}</code> {e.message}
                </p>
              ))}
            </div>
          )}

          <TilingMap />
          <ReportPanel />
          <JsonPanel />
          <SlotsPanel />
        </aside>
        <main className="viewport-wrap">
          <DriftViewport />
        </main>
      </div>
    </div>
  )
}
