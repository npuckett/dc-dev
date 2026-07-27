/**
 * grid-designer v4 — named save slots, backed by persistence.js.
 *
 * Ported from src/v3/SlotsPanel.jsx, rebound to the v4 store. Everything that
 * WRITES design truth goes through the store's slot actions; `listSlots` stays a
 * direct read here because listing is not a mutation and the store has no reason
 * to own it.
 *
 * persistence.js is schema-agnostic — a slot saved by a v3 build now fails its
 * embedded `config.version` check against the bumped `EXPECTED_CONFIG_VERSION =
 * 4` and is reported as unreadable rather than resurrected, exactly like the
 * working-config autosave. That is the documented mechanism (V4_SPEC §6), not a
 * regression, and the message below says so in the user's words.
 */

import { useMemo, useState } from 'react'
import useStoreV4 from './store.js'
import { listSlots } from '../persistence.js'

export default function SlotsPanel() {
  const saveToSlot = useStoreV4((s) => s.saveToSlot)
  const loadFromSlot = useStoreV4((s) => s.loadFromSlot)
  const removeSlot = useStoreV4((s) => s.removeSlot)

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
    if (saveToSlot(trimmed)) {
      setTick((t) => t + 1)
      setNote({ kind: 'ok', message: `saved current design as "${trimmed}"` })
    } else {
      setNote({ kind: 'err', message: 'save failed — storage unavailable or full' })
    }
  }

  const onLoad = (slotName) => {
    const result = loadFromSlot(slotName)
    if (result === 'ok') setNote({ kind: 'ok', message: `loaded "${slotName}"` })
    else if (result === 'unreadable') {
      setNote({
        kind: 'err',
        message: `"${slotName}" is unreadable or from a schema version this build no longer accepts`,
      })
    } else setNote({ kind: 'err', message: `"${slotName}" failed validation — see the errors above` })
  }

  const onDelete = (slotName) => {
    removeSlot(slotName)
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
        <p className={`json-note${note.kind === 'err' ? ' json-note-err' : ''}`} data-testid="slots-note">
          {note.message}
        </p>
      )}
    </details>
  )
}
