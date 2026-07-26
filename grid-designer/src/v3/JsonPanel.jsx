/**
 * grid-designer v3 — paste a config in, copy the current one out.
 *
 * Same idiom as v2's src/components/JsonPanel.jsx (kept, untouched, for v2):
 * the textarea is NOT bound live to the store — it is refreshed only when the
 * disclosure opens and after a successful Apply — and "Apply" runs
 * `normalizeConfig` + `validateConfig` through the v3 store's `setConfig`,
 * committing only when valid. JSON syntax errors are reported inline here;
 * validation errors land in the store's `lastErrors` for ReportPanel/AppV3 to
 * show (a v1/v2 config pasted here is REJECTED outright — V3_SPEC.md §6).
 */

import { useState } from 'react'
import useStoreV3 from './store.js'

const pretty = (config) => JSON.stringify(config, null, 2)

export default function JsonPanel() {
  const config = useStoreV3((s) => s.config)
  const setConfig = useStoreV3((s) => s.setConfig)

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
    if (setConfig(parsed)) {
      setText(pretty(useStoreV3.getState().config))
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
        placeholder='{"version":3,"units":"cm","sheet":{"cols":6,"rows":8},…}'
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
        <p
          className={`json-note${note.kind === 'err' ? ' json-note-err' : ''}`}
          data-testid="json-note"
        >
          {note.message}
        </p>
      )}
    </details>
  )
}
