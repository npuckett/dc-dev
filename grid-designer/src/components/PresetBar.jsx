/**
 * grid-designer — preset buttons.
 *
 * The active preset is read from `config.meta.preset`, which the presets write
 * as provenance — so hand-edited configs (whose meta still names the preset they
 * started from) keep the highlight, and an imported config without meta shows
 * none. The seeded 'random' preset gets a seed field plus a reroll that bumps
 * the seed and re-applies immediately.
 *
 * The buttons are grouped by `PRESETS[].family` with a thin divider between the
 * CALM family (flat fronts on the shore, 5 rows, everything grounded) and the
 * STORM family (pitched fronts everywhere, deeper grids, and — for 'wallcrash' —
 * a column bracketed to the −X wall). The two families follow different rules, so
 * the bar says so rather than presenting eight interchangeable buttons.
 */

import useStore from '../store.js'
import { PRESETS } from '../core/presets.js'

export default function PresetBar() {
  const config = useStore((s) => s.config)
  const seed = useStore((s) => s.seed)
  const applyPreset = useStore((s) => s.applyPreset)
  const setSeed = useStore((s) => s.setSeed)
  const active = config.meta?.preset

  return (
    <div className="preset-bar">
      {PRESETS.map((p, i) => (
        <span key={p.id} className="preset-slot">
          {i > 0 && PRESETS[i - 1].family !== p.family && (
            <span
              className="preset-divider"
              data-testid={`preset-divider-${p.family}`}
              title={`the ${p.family} family — different rules: pitched fronts in every column, deeper grids, and the −X wall in play`}
              aria-hidden="true"
            />
          )}
          <button
            type="button"
            data-testid={`preset-${p.id}`}
            className={`preset-btn${active === p.id ? ' preset-active' : ''}${p.family === 'storm' ? ' preset-storm' : ''}`}
            title={p.description}
            onClick={() => applyPreset(p.id, p.seeded ? seed : undefined)}
          >
            {p.label}
          </button>
        </span>
      ))}

      <span className="seed-group">
        <label className="seed-label" htmlFor="seed-input">
          seed
        </label>
        <input
          id="seed-input"
          data-testid="seed-input"
          type="number"
          className="seed-input"
          value={seed}
          step={1}
          onChange={(e) => {
            const n = Number(e.target.value)
            if (!Number.isFinite(n)) return
            setSeed(n)
            // Keep the view live while typing, but only when random is active.
            if (active === 'random') applyPreset('random', n)
          }}
        />
        <button
          type="button"
          data-testid="reroll"
          className="preset-btn"
          title="bump the seed and rebuild the random preset"
          onClick={() => applyPreset('random', seed + 1)}
        >
          reroll
        </button>
      </span>
    </div>
  )
}
