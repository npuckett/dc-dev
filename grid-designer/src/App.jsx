/**
 * grid-designer — root shell.
 *
 *   top bar : title + preset buttons + export buttons
 *   left    : grid map, messages, column fold controls, config JSON (fixed 320px)
 *   main    : 3D viewport
 *
 * The store is exposed as `window.__gridDesignerStore` for Playwright and
 * console-driven automation (same pattern as panel-designer's App.jsx).
 */

import { useEffect } from 'react'
import useStore, { getDerived } from './store.js'
import PresetBar from './components/PresetBar.jsx'
import ExportButtons from './components/ExportButtons.jsx'
import ControlPanel from './components/ControlPanel.jsx'
import Viewport from './components/Viewport.jsx'

export default function App() {
  useEffect(() => {
    window.__gridDesignerStore = useStore
    window.__gridDesignerDerived = () => getDerived(useStore.getState().config)
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
