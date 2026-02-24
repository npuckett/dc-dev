/**
 * Inspector — Properties panel for the selected object
 *
 * Shows: name, type, position, rotation, dimensions/bounds,
 *        role, tags, references, and a delete button.
 */

import React, { useState, useCallback } from 'react'
import useStore from '../store'
import { ROLES, ROLE_COLORS, RELATION_TYPES } from '../schema'

export default function Inspector() {
  const selectedId = useStore(s => s.selectedId)
  const scene = useStore(s => s.scene)
  const obj = scene.objects.find(o => o.id === selectedId)

  return (
    <div className="inspector">
      <div className="inspector-header">Inspector</div>
      {obj ? <ObjectInspector obj={obj} scene={scene} /> : <ObjectList />}
    </div>
  )
}

/**
 * When nothing is selected, show the object list
 */
function ObjectList() {
  const objects = useStore(s => s.scene.objects)
  const select = useStore(s => s.select)
  const selectedId = useStore(s => s.selectedId)

  if (objects.length === 0) {
    return <div className="inspector-empty">No objects in scene.<br />Use the toolbar to add objects.</div>
  }

  return (
    <div className="object-list">
      {objects.map(obj => (
        <div
          key={obj.id}
          className={`object-list-item ${selectedId === obj.id ? 'selected' : ''}`}
          onClick={() => select(obj.id)}
        >
          <span
            className="obj-icon"
            style={{ background: ROLE_COLORS[obj.tags?.role] || ROLE_COLORS.default }}
          />
          <span className="obj-name">{obj.name}</span>
          <span className="obj-type">{obj.type}</span>
        </div>
      ))}
    </div>
  )
}

/**
 * Full property editor for a selected object
 */
function ObjectInspector({ obj, scene }) {
  const updateObject = useStore(s => s.updateObject)
  const removeObject = useStore(s => s.removeObject)
  const addReference = useStore(s => s.addReference)
  const removeReference = useStore(s => s.removeReference)
  const deselect = useStore(s => s.deselect)

  const refs = scene.references.filter(r => r.from === obj.id || r.to === obj.id)
  const otherObjects = scene.objects.filter(o => o.id !== obj.id)

  function update(updates) {
    updateObject(obj.id, updates)
  }

  return (
    <>
      {/* Identity */}
      <div className="inspector-section">
        <div className="inspector-section-title">Identity</div>
        <div className="inspector-field">
          <label>Name</label>
          <input
            value={obj.name}
            onChange={e => update({ name: e.target.value })}
          />
        </div>
        <div className="inspector-field">
          <label>Type</label>
          <span style={{ fontSize: 11, color: '#a0a0b0' }}>{obj.type}</span>
        </div>
        <div className="inspector-field">
          <label>Role</label>
          <select
            value={obj.tags?.role || 'structural'}
            onChange={e => update({ tags: { role: e.target.value } })}
          >
            {ROLES.map(r => (
              <option key={r} value={r}>{r}</option>
            ))}
          </select>
          <span
            className="role-badge"
            style={{
              background: ROLE_COLORS[obj.tags?.role] || ROLE_COLORS.default,
              color: '#000',
            }}
          >
            {obj.tags?.role || 'structural'}
          </span>
        </div>
      </div>

      {/* Transform */}
      {obj.type !== 'zone' && (
        <div className="inspector-section">
          <div className="inspector-section-title">Transform</div>
          <VectorField
            label="Position"
            value={obj.position || [0, 0, 0]}
            onChange={v => update({ position: v })}
          />
          <VectorField
            label="Rotation"
            value={obj.rotation || [0, 0, 0]}
            onChange={v => update({ rotation: v })}
            step={1}
          />
        </div>
      )}

      {/* Geometry */}
      {(obj.type === 'box' || obj.type === 'plane') && (
        <div className="inspector-section">
          <div className="inspector-section-title">Dimensions</div>
          <VectorField
            label="Size"
            value={obj.dimensions || (obj.type === 'box' ? [100, 100, 100] : [100, 100])}
            onChange={v => update({ dimensions: v })}
            labels={obj.type === 'box' ? ['W', 'H', 'D'] : ['W', 'H']}
          />
        </div>
      )}

      {obj.type === 'point' && (
        <div className="inspector-section">
          <div className="inspector-section-title">Geometry</div>
          <div className="inspector-field">
            <label>Radius</label>
            <input
              type="number"
              value={obj.radius || 5}
              step={1}
              onChange={e => update({ radius: parseFloat(e.target.value) || 1 })}
            />
          </div>
        </div>
      )}

      {obj.type === 'sensor' && (
        <div className="inspector-section">
          <div className="inspector-section-title">Sensor</div>
          <div className="inspector-field">
            <label>Range</label>
            <input
              type="number"
              value={obj.range || 200}
              step={10}
              onChange={e => update({ range: parseFloat(e.target.value) || 10 })}
            />
          </div>
          <div className="inspector-field">
            <label>Angle °</label>
            <input
              type="number"
              value={obj.angle || 60}
              step={5}
              min={1}
              max={180}
              onChange={e => update({ angle: parseFloat(e.target.value) || 30 })}
            />
          </div>
        </div>
      )}

      {obj.type === 'camera' && (
        <div className="inspector-section">
          <div className="inspector-section-title">Camera</div>
          <div className="inspector-field">
            <label>FOV °</label>
            <input
              type="number"
              value={obj.fov || 70}
              step={5}
              min={1}
              max={180}
              onChange={e => update({ fov: parseFloat(e.target.value) || 30 })}
            />
          </div>
          <div className="inspector-field">
            <label>Range</label>
            <input
              type="number"
              value={obj.range || 500}
              step={10}
              onChange={e => update({ range: parseFloat(e.target.value) || 10 })}
            />
          </div>
          <div className="inspector-field">
            <label>Aspect</label>
            <input
              type="number"
              value={obj.aspect || 1.78}
              step={0.01}
              min={0.1}
              onChange={e => update({ aspect: parseFloat(e.target.value) || 1 })}
            />
          </div>
        </div>
      )}

      {obj.type === 'zone' && (
        <div className="inspector-section">
          <div className="inspector-section-title">Bounds</div>
          <BoundsEditor
            bounds={obj.bounds || { x: [-50, 50], y: [0, 100], z: [-50, 50] }}
            onChange={b => update({ bounds: b })}
          />
        </div>
      )}

      {obj.type === 'mesh' && (
        <div className="inspector-section">
          <div className="inspector-section-title">Mesh</div>
          <div className="inspector-field">
            <label>Vertices</label>
            <span style={{ fontSize: 11, color: '#a0a0b0' }}>{obj.geometry?.vertexCount?.toLocaleString() ?? '—'}</span>
          </div>
          <div className="inspector-field">
            <label>Faces</label>
            <span style={{ fontSize: 11, color: '#a0a0b0' }}>{obj.geometry?.faceCount?.toLocaleString() ?? '—'}</span>
          </div>
          <div className="inspector-field">
            <label>Raw Size</label>
            <span style={{ fontSize: 11, color: '#a0a0b0' }}>
              {obj.geometry?.size ? obj.geometry.size.map(v => Math.round(v * 10) / 10).join(' × ') : '—'}
            </span>
          </div>
          {obj.tags?.source_file && (
            <div className="inspector-field">
              <label>Source</label>
              <span style={{ fontSize: 11, color: '#a0a0b0' }}>{obj.tags.source_file}</span>
            </div>
          )}
          <VectorField
            label="Scale"
            value={obj.scale || [1, 1, 1]}
            onChange={v => update({ scale: v })}
            step={0.1}
          />
        </div>
      )}

      {/* Tags */}
      <div className="inspector-section">
        <div className="inspector-section-title">Tags</div>
        <TagsEditor
          tags={obj.tags || {}}
          onChange={tags => updateObject(obj.id, { tags })}
          reservedKeys={['role']}
        />
      </div>

      {/* References */}
      <div className="inspector-section">
        <div className="inspector-section-title">References</div>
        {refs.map(r => {
          const otherObj = scene.objects.find(
            o => o.id === (r.from === obj.id ? r.to : r.from)
          )
          const direction = r.from === obj.id ? '→' : '←'
          return (
            <div key={r.id} className="ref-item">
              <span>{direction}</span>
              <span className="ref-relation">{r.relation}</span>
              <span>{otherObj?.name || '?'}</span>
              <button className="tag-remove" onClick={() => removeReference(r.id)}>×</button>
            </div>
          )
        })}
        <AddReferenceRow
          currentId={obj.id}
          otherObjects={otherObjects}
          onAdd={(relation, targetId) => addReference(obj.id, relation, targetId)}
        />
      </div>

      {/* Actions */}
      <div className="inspector-section">
        <button
          onClick={() => { removeObject(obj.id); deselect() }}
          style={{ background: '#401818', borderColor: '#602828', width: '100%' }}
        >
          Delete Object
        </button>
      </div>
    </>
  )
}

/**
 * Editable vector field (position, rotation, dimensions)
 */
function VectorField({ label, value, onChange, labels, step = 0.1 }) {
  const defaultLabels = ['X', 'Y', 'Z']
  const ls = labels || defaultLabels.slice(0, value.length)

  return (
    <>
      <div className="vector-labels">
        {ls.map(l => <span key={l} className="vector-label">{l}</span>)}
      </div>
      <div className="inspector-field">
        <label>{label}</label>
        <div className="vector-inputs">
          {value.map((v, i) => (
            <input
              key={i}
              type="number"
              value={Math.round(v * 100) / 100}
              step={step}
              onChange={e => {
                const newVal = [...value]
                newVal[i] = parseFloat(e.target.value) || 0
                onChange(newVal)
              }}
            />
          ))}
        </div>
      </div>
    </>
  )
}

/**
 * Zone bounds editor (min/max for each axis)
 */
function BoundsEditor({ bounds, onChange }) {
  const axes = ['x', 'y', 'z']
  return (
    <>
      {axes.map(axis => (
        <div key={axis} className="bounds-row">
          <label>{axis.toUpperCase()}</label>
          <input
            type="number"
            value={bounds[axis]?.[0] || 0}
            step={1}
            onChange={e => {
              onChange({
                ...bounds,
                [axis]: [parseFloat(e.target.value) || 0, bounds[axis]?.[1] || 0],
              })
            }}
          />
          <span style={{ color: '#505060', fontSize: 10 }}>to</span>
          <input
            type="number"
            value={bounds[axis]?.[1] || 0}
            step={1}
            onChange={e => {
              onChange({
                ...bounds,
                [axis]: [bounds[axis]?.[0] || 0, parseFloat(e.target.value) || 0],
              })
            }}
          />
        </div>
      ))}
    </>
  )
}

/**
 * Tags key-value editor
 */
function TagsEditor({ tags, onChange, reservedKeys = [] }) {
  const editableTags = Object.entries(tags).filter(([k]) => !reservedKeys.includes(k))
  const [newKey, setNewKey] = useState('')
  const [newVal, setNewVal] = useState('')

  function addTag() {
    if (!newKey.trim()) return
    onChange({ ...tags, [newKey.trim()]: newVal })
    setNewKey('')
    setNewVal('')
  }

  function removeTag(key) {
    const next = { ...tags }
    delete next[key]
    onChange(next)
  }

  return (
    <>
      {editableTags.map(([key, val]) => (
        <div key={key} className="tag-row">
          <input value={key} readOnly style={{ width: 60, opacity: 0.7 }} />
          <input
            value={val}
            onChange={e => onChange({ ...tags, [key]: e.target.value })}
          />
          <button className="tag-remove" onClick={() => removeTag(key)}>×</button>
        </div>
      ))}
      <div className="tag-row">
        <input
          placeholder="key"
          value={newKey}
          onChange={e => setNewKey(e.target.value)}
          style={{ width: 60 }}
          onKeyDown={e => e.key === 'Enter' && addTag()}
        />
        <input
          placeholder="value"
          value={newVal}
          onChange={e => setNewVal(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && addTag()}
        />
        <button onClick={addTag} style={{ padding: '2px 6px', fontSize: 12 }}>+</button>
      </div>
    </>
  )
}

/**
 * Add reference dropdown row
 */
function AddReferenceRow({ currentId, otherObjects, onAdd }) {
  const [relation, setRelation] = useState(RELATION_TYPES[0])
  const [targetId, setTargetId] = useState('')

  function handleAdd() {
    if (!targetId) return
    onAdd(relation, targetId)
    setTargetId('')
  }

  return (
    <div className="add-ref-row">
      <select value={relation} onChange={e => setRelation(e.target.value)}>
        {RELATION_TYPES.map(r => <option key={r} value={r}>{r}</option>)}
      </select>
      <select value={targetId} onChange={e => setTargetId(e.target.value)}>
        <option value="">— target —</option>
        {otherObjects.map(o => <option key={o.id} value={o.id}>{o.name}</option>)}
      </select>
      <button onClick={handleAdd} style={{ padding: '2px 8px' }}>+</button>
    </div>
  )
}
