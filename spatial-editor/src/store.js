/**
 * Zustand store with undo/redo history
 *
 * Every mutation is a reversible action stored in a history stack.
 * The scene state is the single source of truth for the entire app.
 */

import { create } from 'zustand'
import { immer } from 'zustand/middleware/immer'
import { createScene, createObject, createReference, generateId } from './schema'

const MAX_HISTORY = 100

const useStore = create(
  immer((set, get) => ({
    // --- Scene state ---
    scene: createScene(),
    selectedId: null,
    transformMode: 'translate', // 'translate' | 'rotate' | 'scale'

    // --- History ---
    history: [],
    historyIndex: -1,

    _pushHistory(label) {
      const { history, historyIndex, scene } = get()
      const snapshot = JSON.parse(JSON.stringify(scene))
      const newHistory = history.slice(0, historyIndex + 1)
      newHistory.push({ label, scene: snapshot })
      if (newHistory.length > MAX_HISTORY) newHistory.shift()
      set(state => {
        state.history = newHistory
        state.historyIndex = newHistory.length - 1
      })
    },

    undo() {
      const { history, historyIndex } = get()
      if (historyIndex <= 0) return
      const prev = history[historyIndex - 1]
      set(state => {
        state.scene = JSON.parse(JSON.stringify(prev.scene))
        state.historyIndex = historyIndex - 1
      })
    },

    redo() {
      const { history, historyIndex } = get()
      if (historyIndex >= history.length - 1) return
      const next = history[historyIndex + 1]
      set(state => {
        state.scene = JSON.parse(JSON.stringify(next.scene))
        state.historyIndex = historyIndex + 1
      })
    },

    // --- Selection ---
    select(id) {
      set(state => { state.selectedId = id })
    },

    deselect() {
      set(state => { state.selectedId = null })
    },

    setTransformMode(mode) {
      set(state => { state.transformMode = mode })
    },

    // --- Object CRUD ---
    addObject(type, overrides = {}) {
      const obj = createObject(type, overrides)
      get()._pushHistory(`Add ${type}`)
      set(state => {
        state.scene.objects.push(obj)
        state.selectedId = obj.id
      })
      return obj.id
    },

    removeObject(id) {
      const { scene } = get()
      const obj = scene.objects.find(o => o.id === id)
      if (!obj) return
      get()._pushHistory(`Remove ${obj.name}`)
      set(state => {
        // Remove the object
        state.scene.objects = state.scene.objects.filter(o => o.id !== id)
        // Remove any references involving this object
        state.scene.references = state.scene.references.filter(
          r => r.from !== id && r.to !== id
        )
        // Remove from parent's children list
        state.scene.objects.forEach(o => {
          o.children = o.children.filter(c => c !== id)
        })
        // Unparent children
        state.scene.objects.forEach(o => {
          if (o.parent === id) o.parent = null
        })
        if (state.selectedId === id) state.selectedId = null
      })
    },

    updateObject(id, updates) {
      get()._pushHistory(`Update object`)
      set(state => {
        const obj = state.scene.objects.find(o => o.id === id)
        if (!obj) return
        Object.entries(updates).forEach(([key, value]) => {
          if (key === 'tags') {
            obj.tags = { ...obj.tags, ...value }
          } else {
            obj[key] = value
          }
        })
      })
    },

    updateObjectPosition(id, position) {
      // No history push for continuous drag — push on drag end
      set(state => {
        const obj = state.scene.objects.find(o => o.id === id)
        if (obj) obj.position = [...position]
      })
    },

    updateObjectRotation(id, rotation) {
      set(state => {
        const obj = state.scene.objects.find(o => o.id === id)
        if (obj) obj.rotation = [...rotation]
      })
    },

    commitTransform(id) {
      // Called on drag end — push history
      get()._pushHistory(`Transform object`)
    },

    // --- References ---
    addReference(fromId, relation, toId) {
      const ref = createReference(fromId, relation, toId)
      get()._pushHistory(`Add reference: ${relation}`)
      set(state => {
        state.scene.references.push(ref)
      })
      return ref.id
    },

    removeReference(id) {
      get()._pushHistory(`Remove reference`)
      set(state => {
        state.scene.references = state.scene.references.filter(r => r.id !== id)
      })
    },

    // --- Scene management ---
    loadScene(sceneData) {
      set(state => {
        state.scene = sceneData
        state.selectedId = null
        state.history = [{ label: 'Load scene', scene: JSON.parse(JSON.stringify(sceneData)) }]
        state.historyIndex = 0
      })
    },

    updateSceneMeta(updates) {
      get()._pushHistory(`Update scene metadata`)
      set(state => {
        Object.entries(updates).forEach(([key, value]) => {
          if (key === 'coordinate_system') {
            state.scene.coordinate_system = { ...state.scene.coordinate_system, ...value }
          } else {
            state.scene[key] = value
          }
        })
      })
    },

    replaceScene(sceneData) {
      // Replace from JSON editor — push history
      get()._pushHistory(`Edit JSON`)
      set(state => {
        state.scene = sceneData
      })
    },

    // --- Helpers ---
    getObject(id) {
      return get().scene.objects.find(o => o.id === id) || null
    },

    getObjectsByRole(role) {
      return get().scene.objects.filter(o => o.tags?.role === role)
    },

    getReferencesFor(id) {
      return get().scene.references.filter(r => r.from === id || r.to === id)
    },

    getSelectedObject() {
      const { scene, selectedId } = get()
      if (!selectedId) return null
      return scene.objects.find(o => o.id === selectedId) || null
    },
  }))
)

export default useStore
