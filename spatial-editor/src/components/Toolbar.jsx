/**
 * Toolbar — Object creation + transform mode selection
 */

import React from 'react'
import useStore from '../store'
import { ROLE_COLORS } from '../schema'

const OBJECT_PRESETS = [
  { type: 'box', label: 'Box', icon: '◻', defaultRole: 'structural' },
  { type: 'plane', label: 'Plane', icon: '▭', defaultRole: 'actuator' },
  { type: 'point', label: 'Point', icon: '●', defaultRole: 'reference' },
  { type: 'zone', label: 'Zone', icon: '⬡', defaultRole: 'zone' },
  { type: 'group', label: 'Group', icon: '⊞', defaultRole: 'structural' },
  { type: 'sensor', label: 'Sensor', icon: '◎', defaultRole: 'sensor' },
  { type: 'camera', label: 'Camera', icon: '📷', defaultRole: 'sensor' },
]

const TRANSFORM_MODES = [
  { mode: 'translate', label: 'W', title: 'Move (W)' },
  { mode: 'rotate', label: 'R', title: 'Rotate (R)' },
  { mode: 'scale', label: 'S', title: 'Scale (S)' },
]

export default function Toolbar() {
  const addObject = useStore(s => s.addObject)
  const transformMode = useStore(s => s.transformMode)
  const setTransformMode = useStore(s => s.setTransformMode)

  return (
    <div className="toolbar">
      <div className="toolbar-section">
        <div className="toolbar-section-label">Transform</div>
        {TRANSFORM_MODES.map(({ mode, label, title }) => (
          <button
            key={mode}
            className={transformMode === mode ? 'active' : ''}
            onClick={() => setTransformMode(mode)}
            title={title}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="toolbar-section">
        <div className="toolbar-section-label">Add Object</div>
        {OBJECT_PRESETS.map(({ type, label, icon, defaultRole }) => (
          <button
            key={type}
            className="add-btn"
            onClick={() => addObject(type, { tags: { role: defaultRole } })}
            title={`Add ${label}`}
          >
            <span
              className="type-icon"
              style={{ background: ROLE_COLORS[defaultRole] }}
            />
            {label}
          </button>
        ))}
      </div>
    </div>
  )
}
