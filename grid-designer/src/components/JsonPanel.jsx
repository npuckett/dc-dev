/**
 * grid-designer — paste a config in, copy the current one out.
 *
 * The textarea is intentionally NOT bound live to the store: re-serializing on
 * every slider tick would fight whatever the user is typing. It is refreshed
 * exactly when the disclosure is opened (and after a successful Apply), which is
 * the moment you actually want to see the current state.
 *
 * "Apply" hands the parsed object to the store's `setConfig`, which runs
 * `normalizeConfig` + `validateConfig` and commits only when valid — so a
 * minimal hand- or LLM-written config works (defaults fill in), and a broken one
 * leaves the design untouched with its errors in the control panel's error box.
 * JSON syntax errors never reach the store; they are reported inline here.
 */

import { useState } from 'react'
import useStore from '../store.js'

const pretty = (config) => JSON.stringify(config, null, 2)

export default function JsonPanel() {
  const config = useStore((s) => s.config)
  const setConfig = useStore((s) => s.setConfig)

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
      setText(pretty(useStore.getState().config))
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
      // Clipboard access can be denied (insecure origin, no permission); show
      // the config in the textarea so it can still be selected by hand.
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
        placeholder='{"version":2,"units":"cm","columns":[{"foldsDeg":[…]},…]}'
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
