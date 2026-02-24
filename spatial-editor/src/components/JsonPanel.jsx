/**
 * JsonPanel — Live JSON editor showing the full scene
 *
 * Edits to the JSON update the 3D scene and vice versa.
 */

import React, { useState, useEffect, useRef, useCallback } from 'react'
import useStore from '../store'

export default function JsonPanel() {
  const scene = useStore(s => s.scene)
  const replaceScene = useStore(s => s.replaceScene)
  const [collapsed, setCollapsed] = useState(false)
  const [text, setText] = useState('')
  const [error, setError] = useState(null)
  const [isEditing, setIsEditing] = useState(false)
  const textRef = useRef(null)

  // Sync scene → text when scene changes (and not actively editing)
  useEffect(() => {
    if (!isEditing) {
      setText(JSON.stringify(scene, null, 2))
      setError(null)
    }
  }, [scene, isEditing])

  const handleChange = useCallback((e) => {
    const val = e.target.value
    setText(val)
    try {
      JSON.parse(val)
      setError(null)
    } catch (err) {
      setError(err.message)
    }
  }, [])

  const handleApply = useCallback(() => {
    try {
      const parsed = JSON.parse(text)
      if (!parsed.objects || !parsed.name) {
        setError('Missing required fields: name, objects')
        return
      }
      replaceScene(parsed)
      setError(null)
      setIsEditing(false)
    } catch (err) {
      setError(err.message)
    }
  }, [text, replaceScene])

  const handleFocus = useCallback(() => setIsEditing(true), [])
  const handleBlur = useCallback(() => {
    // Delay to allow Apply button click
    setTimeout(() => setIsEditing(false), 200)
  }, [])

  if (collapsed) {
    return (
      <div style={{
        width: 28,
        background: '#12122a',
        borderRight: '1px solid #2a2a4a',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        paddingTop: 8,
        cursor: 'pointer',
        flexShrink: 0,
      }} onClick={() => setCollapsed(false)}>
        <span style={{ fontSize: 11, writingMode: 'vertical-rl', color: '#606080' }}>
          JSON
        </span>
      </div>
    )
  }

  return (
    <div className="json-panel">
      <div className="json-panel-header">
        <span>Scene JSON</span>
        <button onClick={() => setCollapsed(true)} style={{ padding: '2px 6px' }}>◀</button>
      </div>
      <textarea
        ref={textRef}
        className="json-editor"
        value={text}
        onChange={handleChange}
        onFocus={handleFocus}
        onBlur={handleBlur}
        spellCheck={false}
      />
      {error && <div className="json-error">⚠ {error}</div>}
      <div className="json-actions">
        <button onClick={handleApply} disabled={!!error}>
          Apply Changes
        </button>
        <button onClick={() => {
          setText(JSON.stringify(scene, null, 2))
          setError(null)
          setIsEditing(false)
        }}>
          Reset
        </button>
        <button onClick={() => {
          navigator.clipboard.writeText(text)
        }}>
          Copy
        </button>
      </div>
    </div>
  )
}
