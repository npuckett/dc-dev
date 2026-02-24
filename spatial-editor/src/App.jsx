/**
 * App — Main layout
 *
 * Three-column layout:
 *  Left:   JSON panel (collapsible)
 *  Center: 3D viewport + toolbar overlay
 *  Right:  Inspector panel
 */

import React, { useCallback, useEffect } from 'react'
import Viewport from './components/Viewport'
import Toolbar from './components/Toolbar'
import Inspector from './components/Inspector'
import JsonPanel from './components/JsonPanel'
import TopBar from './components/TopBar'
import useStore from './store'
import './App.css'

export default function App() {
  const undo = useStore(s => s.undo)
  const redo = useStore(s => s.redo)
  const removeObject = useStore(s => s.removeObject)
  const selectedId = useStore(s => s.selectedId)
  const deselect = useStore(s => s.deselect)
  const setTransformMode = useStore(s => s.setTransformMode)

  // Keyboard shortcuts
  useEffect(() => {
    function handleKey(e) {
      // Don't capture when typing in inputs
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return

      if ((e.metaKey || e.ctrlKey) && e.key === 'z') {
        e.preventDefault()
        if (e.shiftKey) redo()
        else undo()
      }
      if (e.key === 'Delete' || e.key === 'Backspace') {
        if (selectedId) {
          e.preventDefault()
          removeObject(selectedId)
        }
      }
      if (e.key === 'Escape') {
        deselect()
      }
      // Transform mode shortcuts
      if (e.key === 'g' || e.key === 'w') setTransformMode('translate')
      if (e.key === 'r') setTransformMode('rotate')
      if (e.key === 's' && !e.metaKey && !e.ctrlKey) setTransformMode('scale')
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [undo, redo, removeObject, selectedId, deselect, setTransformMode])

  return (
    <div className="app">
      <TopBar />
      <div className="app-body">
        <JsonPanel />
        <div className="viewport-container">
          <Toolbar />
          <Viewport />
        </div>
        <Inspector />
      </div>
    </div>
  )
}
