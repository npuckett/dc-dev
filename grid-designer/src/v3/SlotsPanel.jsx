/**
 * grid-designer v3 — named save slots, backed by persistence.js.
 *
 * Same component as v2's src/components/SlotsPanel.jsx (kept, untouched, for
 * v2), rebound to the v3 store. persistence.js is schema-agnostic — a slot
 * saved by a v2 build now fails its embedded `config.version` check against
 * the bumped `EXPECTED_CONFIG_VERSION = 3` and is reported as unreadable
 * rather than resurrected, exactly like the working-config autosave.
 */

import { useMemo, useState } from 'react'
import useStoreV3 from './store.js'
import { deleteSlot, listSlots, loadSlot, saveSlot } from '../persistence.js'

export default function SlotsPanel() {
  const config = useStoreV3((s) => s.config)
  const setConfig = useStoreV3((s) => s.setConfig)

  const [open, setOpen] = useState(false)
  const [name, setName] = useState('')
  const [note, setNote] = useState(null) // { kind: 'ok'|'err', message }
  const [tick, setTick] = useState(0)

  // eslint-disable-next-line react-hooks/exhaustive-deps -- `tick` is the deliberate re-read trigger
  const slots = useMemo(() => listSlots(), [tick])

  const onSave = () => {
    const trimmed = name.trim()
    if (!trimmed) {
      setNote({ kind: 'err', message: 'name the slot before saving' })
      return
    }
    if (saveSlot(trimmed, config)) {
      setTick((t) => t + 1)
      setNote({ kind: 'ok', message: `saved current design as "${trimmed}"` })
    } else {
      setNote({ kind: 'err', message: 'save failed — storage unavailable or full' })
    }
  }

  const onLoad = (slotName) => {
    const loaded = loadSlot(slotName)
    if (!loaded) {
      setNote({
        kind: 'err',
        message: `"${slotName}" is unreadable or from a schema version this build no longer accepts`,
      })
      return
    }
    if (setConfig(loaded)) {
      setNote({ kind: 'ok', message: `loaded "${slotName}"` })
    } else {
      setNote({ kind: 'err', message: `"${slotName}" failed validation — see the errors above` })
    }
  }

  const onDelete = (slotName) => {
    deleteSlot(slotName)
    setTick((t) => t + 1)
    setNote({ kind: 'ok', message: `deleted "${slotName}"` })
  }

  const onToggle = (e) => setOpen(e.currentTarget.open)

  return (
    <details className="json-panel" data-testid="slots-panel" open={open} onToggle={onToggle}>
      <summary data-testid="slots-toggle">saved designs</summary>

      <div className="slots-save-row">
        <input
          type="text"
          className="slots-name-input"
          data-testid="slots-name-input"
          placeholder="slot name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') onSave()
          }}
        />
        <button type="button" className="preset-btn" data-testid="slots-save" onClick={onSave}>
          Save
        </button>
      </div>

      {slots.length > 0 ? (
        <ul className="slots-list" data-testid="slots-list">
          {slots.map((s) => (
            <li key={s.name} className="slot-row">
              <button
                type="button"
                className="slot-load-btn"
                data-testid={`slot-load-${s.name}`}
                title={s.savedAt ? `load — saved ${new Date(s.savedAt).toLocaleString()}` : 'load'}
                onClick={() => onLoad(s.name)}
              >
                {s.name}
              </button>
              <button
                type="button"
                className="slot-delete-btn"
                data-testid={`slot-delete-${s.name}`}
                title={`delete "${s.name}"`}
                onClick={() => onDelete(s.name)}
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      ) : (
        <p className="slots-empty" data-testid="slots-empty">
          no saved slots yet — Save parks the current design under a name
        </p>
      )}

      {note && (
        <p
          className={`json-note${note.kind === 'err' ? ' json-note-err' : ''}`}
          data-testid="slots-note"
        >
          {note.message}
        </p>
      )}
    </details>
  )
}
