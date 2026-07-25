/**
 * grid-designer — root shell.
 *
 *   top bar : title + preset buttons + export buttons
 *   left    : grid map, messages, column fold controls, config JSON (fixed 320px)
 *   main    : 3D viewport
 *
 * The store is exposed as `window.__gridDesignerStore` for Playwright and
 * console-driven automation (same pattern as panel-designer's App.jsx).
 *
 * UNDO / REDO shortcuts are wired HERE rather than in a component, because they act
 * on the whole design rather than on any one panel of the UI: Cmd/Ctrl+Z steps back,
 * Shift+Cmd/Ctrl+Z re-applies. They are deliberately INERT while focus is in the
 * config-JSON textarea or a number input, where the browser's own undo — of the text
 * being typed — is what the user means.
 */

import { useEffect } from 'react'
import useStore, { getDerived } from './store.js'
import PresetBar from './components/PresetBar.jsx'
import ExportButtons from './components/ExportButtons.jsx'
import ControlPanel from './components/ControlPanel.jsx'
import Viewport from './components/Viewport.jsx'

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

export default function App() {
  useEffect(() => {
    window.__gridDesignerStore = useStore
    window.__gridDesignerDerived = () => getDerived(useStore.getState().config)
  }, [])

  useEffect(() => {
    const onKey = (e) => {
      if (!(e.metaKey || e.ctrlKey) || e.altKey) return
      if (e.key.toLowerCase() !== 'z') return
      if (inTextField(e.target)) return
      e.preventDefault()
      const { undo, redo } = useStore.getState()
      if (e.shiftKey) redo()
      else undo()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  return (
    <div className="app">
      <header className="top-bar">
        <h1 className="app-title">grid-designer</h1>
        <span className="app-subtitle">Drop Ceiling V2 — folded surface</span>
        <PresetBar />
        <ExportButtons />
      </header>
      <div className="main-layout">
        <ControlPanel />
        <main className="viewport-wrap">
          <Viewport />
        </main>
      </div>
    </div>
  )
}
