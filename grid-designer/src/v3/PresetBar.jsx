/**
 * The drift preset bar.
 *
 * Each preset is a chosen point in a three-way trade the model forces and
 * cannot resolve on its own — height vs. joint deviation vs. modularity — so
 * the trade-off text is surfaced on hover rather than hidden in the source.
 * Picking a preset is a design decision, and the UI should say what it costs.
 */

import { PRESETS } from '../core/v3/presets.js'
import useStore from './store.js'

export default function PresetBar() {
  const applyPreset = useStore((s) => s.applyPreset)
  const current = useStore((s) => s.config.meta?.preset ?? null)

  return (
    <div className="v3-presets">
      <div className="v3-presets-label">DRIFT PRESETS</div>
      <div className="v3-presets-row">
        {PRESETS.map((p) => (
          <button
            key={p.id}
            type="button"
            className={`preset-btn${current === p.id ? ' is-active' : ''}`}
            title={`${p.label}\n\n${p.trade}`}
            onClick={() => applyPreset(p.id)}
          >
            {p.id}
          </button>
        ))}
      </div>
      {current && (
        <div className="v3-preset-trade">
          {PRESETS.find((p) => p.id === current)?.trade}
        </div>
      )}
    </div>
  )
}
