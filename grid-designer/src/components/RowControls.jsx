/**
 * grid-designer — controls for one row: base zig-zag, per-joint overrides, and
 * (for every row but the last) the dihedral fold to the next row.
 *
 * Row 0 is the SHORE at the storefront window: it is locked flat, so its
 * zig-zag slider and joint inputs are disabled. (The store would reject the
 * change anyway — E_SHORE_NOT_FLAT — but disabling says why up front.)
 *
 * Sliders commit on every `input` event: the solve is memoized per config and a
 * whole 6×5 solve + joint report is well under a millisecond, so live dragging
 * is fine.
 */

import { useEffect, useState } from 'react'
import useStore from '../store.js'
import {
  MAX_ROW_FOLD_DEG,
  MAX_ZIGZAG_DEG,
  removedJoints,
} from '../core/schema.js'

/** Cumulative row pitch ψ_r = Σ_{k<r} rowFoldsDeg[k]. */
function cumulativePitch(config, r) {
  const folds = Array.isArray(config.rowFoldsDeg) ? config.rowFoldsDeg : []
  let sum = 0
  for (let k = 0; k < r; k++) sum += Number(folds[k]) || 0
  return sum
}

function rowSuffix(r, rowCount) {
  if (r === 0) return ' — shore'
  if (r === rowCount - 1) return ' — back'
  return ''
}

// -----------------------------------------------------------------------------
// One per-joint signed override input
// -----------------------------------------------------------------------------
function JointInput({ r, j, value, placeholder, disabled, removed }) {
  const setJointOverride = useStore((s) => s.setJointOverride)
  const [text, setText] = useState(value === undefined ? '' : String(value))

  // Re-sync when the store's value changes underneath us (preset applied, ✕
  // pressed). Compared numerically so an in-progress entry like "-" survives.
  useEffect(() => {
    const cur = text.trim()
    const curNum = cur === '' || !Number.isFinite(Number(cur)) ? null : Number(cur)
    const ext = value === undefined ? null : value
    if (curNum !== ext) setText(ext === null ? '' : String(ext))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value])

  const onChange = (e) => {
    const t = e.target.value
    setText(t)
    const trimmed = t.trim()
    if (trimmed === '') setJointOverride(r, j, null)
    else if (Number.isFinite(Number(trimmed))) setJointOverride(r, j, Number(trimmed))
  }

  return (
    <div className={`joint-cell${removed ? ' joint-cell-removed' : ''}`}>
      <span className="joint-index">j{j}</span>
      <input
        type="number"
        className="joint-input"
        data-testid={`joint-${r}-${j}`}
        value={removed ? '' : text}
        placeholder={removed ? '—' : placeholder}
        disabled={disabled || removed}
        step={1}
        min={-MAX_ZIGZAG_DEG}
        max={MAX_ZIGZAG_DEG}
        onChange={onChange}
        title={
          removed
            ? 'joint removed by a horizontal rect — the plate is continuous here'
            : `signed override for joint ${j} (blank = alternating ±${Math.abs(Number(placeholder)) || 0}°)`
        }
      />
      <button
        type="button"
        className="joint-clear"
        aria-label={`clear override for joint ${j}`}
        disabled={disabled || removed || value === undefined}
        onClick={() => setJointOverride(r, j, null)}
      >
        ✕
      </button>
    </div>
  )
}

// -----------------------------------------------------------------------------
// Row block
// -----------------------------------------------------------------------------
export default function RowControls({ r }) {
  const config = useStore((s) => s.config)
  const setRowZigzag = useStore((s) => s.setRowZigzag)
  const setRowFold = useStore((s) => s.setRowFold)

  const rowCount = config.grid.rows
  const cols = config.grid.cols
  const row = config.rows[r]
  const isShore = r === 0
  const overrides = row.jointOverridesDeg ?? {}
  const removed = removedJoints(config, r)
  const overrideCount = Object.keys(overrides).length
  const pitch = cumulativePitch(config, r)
  const hasFold = r < rowCount - 1
  const foldDeg = hasFold ? Number(config.rowFoldsDeg[r]) || 0 : 0

  return (
    <>
      <section className="row-block" data-testid={`row-${r}`}>
        <header className="row-head">
          <span className="row-title">
            Row {r}
            <span className="row-suffix">{rowSuffix(r, rowCount)}</span>
          </span>
          <span className="row-pitch" title="cumulative row pitch (sum of folds up to this row)">
            pitch {pitch.toFixed(0)}°
          </span>
        </header>

        <div className="slider-row">
          <label className="slider-label" htmlFor={`zigzag-${r}`}>
            zig-zag
          </label>
          <input
            id={`zigzag-${r}`}
            data-testid={`zigzag-${r}`}
            type="range"
            min={-MAX_ZIGZAG_DEG}
            max={MAX_ZIGZAG_DEG}
            step={1}
            value={Number(row.zigzagDeg) || 0}
            disabled={isShore}
            onChange={(e) => setRowZigzag(r, Number(e.target.value))}
          />
          <output className="slider-value">{Number(row.zigzagDeg) || 0}°</output>
        </div>

        {isShore ? (
          <p className="row-note">shore — fixed flat</p>
        ) : (
          <details className="joints-disclosure">
            <summary>
              joints{overrideCount > 0 ? ` · ${overrideCount} override${overrideCount > 1 ? 's' : ''}` : ''}
            </summary>
            <div className="joints-grid">
              {Array.from({ length: Math.max(0, cols - 1) }, (_, j) => (
                <JointInput
                  key={j}
                  r={r}
                  j={j}
                  value={
                    Object.prototype.hasOwnProperty.call(overrides, String(j))
                      ? Number(overrides[String(j)])
                      : undefined
                  }
                  placeholder={String((j % 2 === 0 ? 1 : -1) * (Number(row.zigzagDeg) || 0))}
                  disabled={isShore}
                  removed={removed.has(j)}
                />
              ))}
            </div>
            <p className="joints-hint">blank = alternating ±zig-zag · filled = signed absolute angle</p>
          </details>
        )}
      </section>

      {hasFold && (
        <div className="fold-row" data-testid={`fold-row-${r}`}>
          <label className="slider-label" htmlFor={`fold-${r}`}>
            Fold {r}→{r + 1}
          </label>
          <input
            id={`fold-${r}`}
            data-testid={`fold-${r}`}
            type="range"
            min={-MAX_ROW_FOLD_DEG}
            max={MAX_ROW_FOLD_DEG}
            step={1}
            value={foldDeg}
            onChange={(e) => setRowFold(r, Number(e.target.value))}
          />
          <output className="slider-value">{foldDeg}°</output>
        </div>
      )}
    </>
  )
}
