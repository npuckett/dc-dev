/**
 * grid-designer v4 — paste a config in, copy the current one out.
 *
 * Ported from src/v3/JsonPanel.jsx with the same idiom: the textarea is NOT
 * bound live to the store — it is refreshed only when the disclosure opens and
 * after a successful Apply — and "Apply" runs `normalizeConfig` +
 * `validateConfig` through the store's `loadJson`, committing only when valid.
 * JSON syntax errors are reported inline here; validation errors land in the
 * store's `lastErrors` for AppV4 to show.
 *
 * A v1/v2/v3 config pasted here is REJECTED outright rather than migrated
 * (V4_SPEC §6): a tiled drift surface and an open fold chain describe physically
 * different objects and do not map onto one another.
 */

import { useState } from 'react'
import useStoreV4 from './store.js'

const pretty = (config) => JSON.stringify(config, null, 2)

export default function JsonPanel() {
  const config = useStoreV4((s) => s.config)
  const loadJson = useStoreV4((s) => s.loadJson)

  const [open, setOpen] = useState(false)
  const [text, setText] = useState(() => pretty(config))
  const [note, setNote] = useState(null) // { kind: 'ok'|'err', message }

  const onToggle = (e) => {
    const isOpen = e.currentTarget.open
    setOpen(isOpen)
    if (isOpen) {
      setText(pretty(config))
      setNote(null)
    }
  }

  const onApply = () => {
    let parsed
    try {
      parsed = JSON.parse(text)
    } catch (err) {
      setNote({ kind: 'err', message: `not valid JSON — ${err.message}` })
      return
    }
    if (loadJson(parsed)) {
      setText(pretty(useStoreV4.getState().config))
      setNote({ kind: 'ok', message: 'applied — defaults filled in where omitted' })
    } else {
      setNote({ kind: 'err', message: 'rejected by validation — see the errors above' })
    }
  }

  const onCopy = async () => {
    const payload = pretty(config)
    try {
      await navigator.clipboard.writeText(payload)
      setNote({ kind: 'ok', message: 'current config copied to the clipboard' })
    } catch (err) {
      setText(payload)
      setNote({ kind: 'err', message: `clipboard unavailable (${err.message}) — copy from the box` })
    }
  }

  return (
    <details className="json-panel" data-testid="json-panel" open={open} onToggle={onToggle}>
      <summary data-testid="json-toggle">config JSON</summary>

      <textarea
        className="json-text"
        data-testid="json-text"
        spellCheck={false}
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder='{"version":4,"strip":{"count":1,"units":9},"gap":2,"angleDeg":30,…}'
      />

      <div className="json-actions">
        <button type="button" className="preset-btn" data-testid="json-apply" onClick={onApply}>
          Apply
        </button>
        <button type="button" className="preset-btn" data-testid="json-copy" onClick={onCopy}>
          Copy
        </button>
      </div>

      {note && (
        <p className={`json-note${note.kind === 'err' ? ' json-note-err' : ''}`} data-testid="json-note">
          {note.message}
        </p>
      )}
    </details>
  )
}
