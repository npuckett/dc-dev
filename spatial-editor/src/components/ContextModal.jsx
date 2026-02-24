/**
 * ContextModal — Shows the agent-readable scene context export
 */

import React, { useRef } from 'react'

export default function ContextModal({ text, onClose }) {
  const textareaRef = useRef()

  function handleCopy() {
    navigator.clipboard.writeText(text)
    // Brief visual feedback
    if (textareaRef.current) {
      textareaRef.current.select()
    }
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <h3>Scene Context for Agents</h3>
          <button onClick={onClose} style={{ background: 'none', border: 'none', color: '#808080', fontSize: 18, cursor: 'pointer' }}>✕</button>
        </div>
        <div className="modal-body">
          <p style={{ fontSize: 12, color: '#808090', marginBottom: 8 }}>
            Paste this into an agent conversation to give it full spatial context for your scene.
          </p>
          <textarea ref={textareaRef} readOnly value={text} />
        </div>
        <div className="modal-footer">
          <button onClick={handleCopy}>Copy to Clipboard</button>
          <button onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  )
}
