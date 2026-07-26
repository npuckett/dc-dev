/**
 * grid-designer v3 — the drift-surface controls.
 *
 * Every range is read from the constants src/core/v3/schema.js EXPORTS (never
 * hardcoded) — SHEET_COLS_MIN/MAX, SHEET_ROWS_MIN/MAX, GAP_MIN/MAX,
 * AMPLITUDE_MIN/MAX, CREST_MIN/MAX, RIDGE_SHEAR_MIN/MAX, TOE_SHARP_MIN/MAX,
 * PLATE_FIT_TOLERANCE_MIN/MAX, and the TILING_STRATEGIES / PLACEMENT_TREES /
 * PLACEMENT_MODES enums — with ONE documented exception:
 *
 *   `form.angularity` (0..1) and `form.facetCells` (1..4) are clamped by
 *   src/core/v3/form.js's `normalizeForm`, but their MIN/MAX constants are
 *   module-private there (`ANGULARITY_MIN/MAX`, `FACET_CELLS_MIN/MAX` — no
 *   `export` keyword) and schema.js never re-exports them or range-checks
 *   them in `validateConfig` (every other form knob gets an `E_RANGE` check
 *   against the raw input; these two do not). Since src/core/v3/ is frozen
 *   for this package, the two ranges below are hand-mirrored from form.js's
 *   values rather than imported — flagged here, and in the P5 report, as a
 *   real gap in the frozen core rather than silently worked around.
 */

import useStoreV3, { getDerived } from './store.js'
import {
  SHEET_COLS_MIN,
  SHEET_COLS_MAX,
  SHEET_ROWS_MIN,
  SHEET_ROWS_MAX,
  GAP_MIN,
  GAP_MAX,
  AMPLITUDE_MIN,
  AMPLITUDE_MAX,
  CREST_MIN,
  CREST_MAX,
  RIDGE_SHEAR_MIN,
  RIDGE_SHEAR_MAX,
  TOE_SHARP_MIN,
  TOE_SHARP_MAX,
  PLATE_FIT_TOLERANCE_MIN,
  PLATE_FIT_TOLERANCE_MAX,
  TILING_STRATEGIES,
  PLACEMENT_TREES,
  PLACEMENT_MODES,
} from '../core/v3/schema.js'

/** See the file header: NOT exported by schema.js. Mirrored from form.js. */
const ANGULARITY_MIN = 0
const ANGULARITY_MAX = 1
const FACET_CELLS_MIN = 1
const FACET_CELLS_MAX = 4

const STRATEGY_HINTS = {
  'flat-lie': 'a rigid plate needs a flat place to lie — plates go where the surface has the least curvature',
  'ridge-aligned': 'plates run along the crest, squares take the bends',
  'toe-bands': 'the two grounded edges (wall, window) are banded by plates',
}
const TREE_HINTS = {
  'bfs-corner': 'BFS from the grounded wall+window corner — error spreads roughly evenly outward',
  'comb-v': 'a spine along the window edge, ribs run back — reproduces the v2 column-chain model, dumping all error on the cross-sheet joints',
  'comb-u': 'the transpose of comb-v: a spine along the wall edge',
}

function SliderRow({ testId, label, value, min, max, step, onChange, format }) {
  const display = format ? format(value) : value
  return (
    <div className="slider-row">
      <span className="slider-label">{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        data-testid={testId}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <span className="slider-value">{display}</span>
    </div>
  )
}

const fixed = (n) => (v) => v.toFixed(n)
const deg = (v) => `${Math.round(v)}°`
const cm1 = fixed(1)
const num2 = fixed(2)
const int0 = (v) => String(Math.round(v))

export default function FormPanel() {
  const config = useStoreV3((s) => s.config)
  const setForm = useStoreV3((s) => s.setForm)
  const setSheetCols = useStoreV3((s) => s.setSheetCols)
  const setSheetRows = useStoreV3((s) => s.setSheetRows)
  const setTilingStrategy = useStoreV3((s) => s.setTilingStrategy)
  const setPlateFitTolerance = useStoreV3((s) => s.setPlateFitTolerance)
  const setPlacementMode = useStoreV3((s) => s.setPlacementMode)
  const setPlacementTree = useStoreV3((s) => s.setPlacementTree)
  const setGap = useStoreV3((s) => s.setGap)
  const undo = useStoreV3((s) => s.undo)
  const redo = useStoreV3((s) => s.redo)
  const canUndo = useStoreV3((s) => s.canUndo)
  const canRedo = useStoreV3((s) => s.canRedo)
  const resetConfig = useStoreV3((s) => s.resetConfig)

  const { form, sheet, tiling, placement, gap } = config
  const { report } = getDerived(config)

  return (
    <section className="form-panel" data-testid="form-panel">
      <header className="form-panel-head">
        <span className="grid-map-title">drift form</span>
        <div className="form-panel-history">
          <button type="button" className="tool-btn" disabled={!canUndo} onClick={() => undo()} title="undo (Cmd/Ctrl+Z)">
            undo
          </button>
          <button type="button" className="tool-btn" disabled={!canRedo} onClick={() => redo()} title="redo (Shift+Cmd/Ctrl+Z)">
            redo
          </button>
          <button type="button" className="tool-btn" onClick={() => resetConfig()} title="reset to the shipped default drift">
            reset
          </button>
        </div>
      </header>

      <div className="col-profile form-block">
        <SliderRow
          testId="form-amplitude"
          label="amplitude"
          value={form.amplitude}
          min={AMPLITUDE_MIN}
          max={AMPLITUDE_MAX}
          step={1}
          onChange={(v) => setForm('amplitude', v)}
          format={(v) => `${Math.round(v)}cm`}
        />
        <SliderRow
          testId="form-crestX"
          label="crest X"
          value={form.crestX}
          min={CREST_MIN}
          max={CREST_MAX}
          step={0.01}
          onChange={(v) => setForm('crestX', v)}
          format={num2}
        />
        <SliderRow
          testId="form-crestZ"
          label="crest Z"
          value={form.crestZ}
          min={CREST_MIN}
          max={CREST_MAX}
          step={0.01}
          onChange={(v) => setForm('crestZ', v)}
          format={num2}
        />
        <SliderRow
          testId="form-ridgeShear"
          label="ridge shear"
          value={form.ridgeShear}
          min={RIDGE_SHEAR_MIN}
          max={RIDGE_SHEAR_MAX}
          step={0.01}
          onChange={(v) => setForm('ridgeShear', v)}
          format={num2}
        />
        <SliderRow
          testId="form-toeSharpX"
          label="toe sharp X"
          value={form.toeSharpX}
          min={TOE_SHARP_MIN}
          max={TOE_SHARP_MAX}
          step={0.01}
          onChange={(v) => setForm('toeSharpX', v)}
          format={num2}
        />
        <SliderRow
          testId="form-toeSharpZ"
          label="toe sharp Z"
          value={form.toeSharpZ}
          min={TOE_SHARP_MIN}
          max={TOE_SHARP_MAX}
          step={0.01}
          onChange={(v) => setForm('toeSharpZ', v)}
          format={num2}
        />

        <SliderRow
          testId="form-angularity"
          label="angularity"
          value={form.angularity}
          min={ANGULARITY_MIN}
          max={ANGULARITY_MAX}
          step={0.01}
          onChange={(v) => setForm('angularity', v)}
          format={num2}
        />
        <p className="form-annotation">
          0 = smooth drift, 1 = faceted to the panel lattice. Faceting lets the panels <b>BE</b> the
          surface instead of approximating it: it closes the joints and relieves housing collisions,
          but lifts the graded edges.
        </p>

        <SliderRow
          testId="form-facetCells"
          label="facet cells"
          value={form.facetCells}
          min={FACET_CELLS_MIN}
          max={FACET_CELLS_MAX}
          step={1}
          onChange={(v) => setForm('facetCells', v)}
          format={int0}
        />
        <p className="form-annotation">how many cells share one plane; 1 = a fold at every joint.</p>
      </div>

      <div className="col-profile form-block">
        <SliderRow
          testId="form-cols"
          label="cols (i)"
          value={sheet.cols}
          min={SHEET_COLS_MIN}
          max={SHEET_COLS_MAX}
          step={1}
          onChange={(v) => setSheetCols(v)}
          format={int0}
        />
        <SliderRow
          testId="form-rows"
          label="rows (j)"
          value={sheet.rows}
          min={SHEET_ROWS_MIN}
          max={SHEET_ROWS_MAX}
          step={1}
          onChange={(v) => setSheetRows(v)}
          format={int0}
        />
        <SliderRow
          testId="form-gap"
          label="gap"
          value={gap}
          min={GAP_MIN}
          max={GAP_MAX}
          step={0.1}
          onChange={(v) => setGap(v)}
          format={(v) => `${cm1(v)}cm`}
        />
      </div>

      <div className="col-profile form-block">
        <label className="form-select-row">
          <span className="slider-label">tiling</span>
          <select
            className="form-select"
            data-testid="form-tiling-strategy"
            value={tiling.strategy}
            onChange={(e) => setTilingStrategy(e.target.value)}
          >
            {TILING_STRATEGIES.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>
        <p className="form-annotation">{STRATEGY_HINTS[tiling.strategy]}</p>

        <SliderRow
          testId="form-plateFitTolerance"
          label="plate fit tol"
          value={tiling.plateFitToleranceCm}
          min={PLATE_FIT_TOLERANCE_MIN}
          max={PLATE_FIT_TOLERANCE_MAX}
          step={0.1}
          onChange={(v) => setPlateFitTolerance(v)}
          format={(v) => `${cm1(v)}cm`}
        />
      </div>

      <div className="col-profile form-block">
        <label className="form-select-row">
          <span className="slider-label">mode</span>
          <select
            className="form-select"
            data-testid="form-placement-mode"
            value={placement.mode}
            onChange={(e) => setPlacementMode(e.target.value)}
          >
            {PLACEMENT_MODES.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </label>
        <p className="form-annotation">
          <b>surface-fit</b> shares the misfit across all joints (what the connectors really do).{' '}
          <b>chain</b> makes tree joints exact and dumps everything on the cycle-closing edges.
        </p>

        {placement.mode === 'chain' && (
          <>
            <label className="form-select-row">
              <span className="slider-label">tree</span>
              <select
                className="form-select"
                data-testid="form-placement-tree"
                value={placement.tree}
                onChange={(e) => setPlacementTree(e.target.value)}
              >
                {PLACEMENT_TREES.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
            </label>
            <p className="form-annotation">{TREE_HINTS[placement.tree]}</p>
          </>
        )}
      </div>

      {report.warnings.length > 0 && (
        <div className="msg-list msg-warnings" data-testid="form-warnings">
          {report.warnings.map((w, i) => (
            <p key={i}>
              <code>{w.code}</code> {w.message}
            </p>
          ))}
        </div>
      )}
    </section>
  )
}
