/**
 * grid-designer v4 — the shape controls.
 *
 * Every range is read from the constants src/core/v4/schema.js EXPORTS, never
 * hardcoded — and unlike v3's FormPanel there is no exception to that rule here,
 * because v4's schema range-checks every knob it declares (schema.js's header,
 * closing HANDOFF §5.2's gap).
 *
 * =============================================================================
 * TWO KNOBS ARE THE DESIGN; THE REST ARE SETTINGS
 * =============================================================================
 * `strip.units` and `angleDeg` are what the user actually drives, so they get the
 * top of the panel and the most room. Everything below them — gap, phase, strip
 * count, the offsets, the connector knobs — changes the design in ways that are
 * real but secondary, and is laid out to read as such.
 *
 * =============================================================================
 * THE LIMITS ARE SHOWN BESIDE THE ANGLE, LIVE
 * =============================================================================
 * This is the panel's reason for existing in the form it has. v3 could tell you
 * an angle was wrong; it could not tell you which angle would be right
 * (HANDOFF §0). `envelope.maxAngleDeg` and `envelope.frontBar.concaveLimitDeg`
 * are exactly that missing sentence, so they sit next to the slider that moves
 * against them rather than three panels away in the report.
 *
 * They are DIFFERENT KINDS OF LIMIT and are never merged into one number:
 *   - `maxAngleDeg` is where the CONNECTOR runs out. Nothing in the part can be
 *     redesigned past it (HANDOFF §2.20) — it is a limit on the form.
 *   - `concaveLimitDeg` is where a flat front bar bites the bezels in a valley.
 *     It is a limit on the PART, fixable with a relief or a narrower section, and
 *     it does not move when the gap does. Collapsing the two would hide the one
 *     that governs.
 * The angle slider itself is NOT clamped to either: the store clamps to the
 * schema range only, on the same principle — an envelope is a permission you can
 * see yourself exceeding, not a fence.
 *
 * The GLYPH READOUT (`_/-\_/-\_`) is the most compact true statement about the
 * strip there is: it is the pattern of V4_SPEC §1 written out for the units this
 * strip actually has, with any role override already applied, so a forced role
 * shows up in it immediately.
 */

import useStoreV4, { getDerived } from './store.js'
import { ROLE_GLYPH } from '../core/v4/lattice.js'
import {
  LATTICE_COLS_MIN,
  LATTICE_COLS_MAX,
  LATTICE_ROWS_MIN,
  LATTICE_ROWS_MAX,
  WALL_ANCHORS,
  GAP_MIN,
  GAP_MAX,
  ANGLE_MIN,
  ANGLE_MAX,
  PHASE_MIN,
  PHASE_MAX,
  WALL_OFFSET_MIN,
  WALL_OFFSET_MAX,
  WINDOW_OFFSET_MIN,
  WINDOW_OFFSET_MAX,
  GROUND_CLEARANCE_MIN,
  GROUND_CLEARANCE_MAX,
  CONNECTOR_LENGTH_MIN,
  CONNECTOR_LENGTH_MAX,
  CONNECTOR_SPACING_MIN,
  CONNECTOR_SPACING_MAX,
  CONNECTOR_MIN_PER_JOINT_MIN,
  CONNECTOR_MIN_PER_JOINT_MAX,
  CONNECTOR_BIN_SPAN_MIN,
  CONNECTOR_BIN_SPAN_MAX,
  CONNECTOR_BIN_ANGLE_MIN,
  CONNECTOR_BIN_ANGLE_MAX,
  CONNECTOR_POWER_EDGES,
  CONNECTOR_SUPPLY_MODES,
  PATTERN_KINDS,
  SCRUNCH_MIN,
  SCRUNCH_MAX,
  ATTRACTOR_MIN,
  ATTRACTOR_MAX,
} from '../core/v4/schema.js'

const fixed = (n) => (v) => v.toFixed(n)
const cm1 = fixed(1)
const num2 = fixed(2)
const int0 = (v) => String(Math.round(v))

function SliderRow({ testId, label, value, min, max, step, onChange, format, disabled = false }) {
  return (
    <div className={`slider-row${disabled ? ' slider-row-disabled' : ''}`}>
      <span className="slider-label">{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        data-testid={testId}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <span className="slider-value">{format ? format(value) : value}</span>
    </div>
  )
}

/** A limit stated as a sentence, coloured by whether the design is past it. */
function LimitLine({ testId, over, children }) {
  return (
    <p className={`limit-line${over ? ' limit-line-over' : ''}`} data-testid={testId}>
      {children}
    </p>
  )
}

export default function StripPanel() {
  const config = useStoreV4((s) => s.config)
  const setCols = useStoreV4((s) => s.setCols)
  const setRows = useStoreV4((s) => s.setRows)
  const setAngleDeg = useStoreV4((s) => s.setAngleDeg)
  const setGap = useStoreV4((s) => s.setGap)
  const setPhase = useStoreV4((s) => s.setPhase)
  const setPatternKind = useStoreV4((s) => s.setPatternKind)
  const setWaveKnob = useStoreV4((s) => s.setWaveKnob)
  const setWallAnchor = useStoreV4((s) => s.setWallAnchor)
  const setObstacleField = useStoreV4((s) => s.setObstacleField)
  const setWallOffset = useStoreV4((s) => s.setWallOffset)
  const setWindowOffset = useStoreV4((s) => s.setWindowOffset)
  const setGroundToFloor = useStoreV4((s) => s.setGroundToFloor)
  const setGroundClearance = useStoreV4((s) => s.setGroundClearance)
  const setConnectorKnob = useStoreV4((s) => s.setConnectorKnob)
  const undo = useStoreV4((s) => s.undo)
  const redo = useStoreV4((s) => s.redo)
  const canUndo = useStoreV4((s) => s.canUndo)
  const canRedo = useStoreV4((s) => s.canRedo)
  const resetConfig = useStoreV4((s) => s.resetConfig)

  const { lattice, gap, angleDeg, pattern, placement, connectors } = config
  const { chain, report } = getDerived(config)
  const spacers = report.spacers
  const { envelope } = report
  const counts = report.metrics.counts
  const isWave = pattern.kind === 'wave'
  const waveCfg = pattern.wave ?? { scrunchX: 0, scrunchZ: 0, attractorX: 1, attractorZ: 1 }
  // The SOLVED wave, not the config — the readout has to show what the angles
  // came out at, which is the thing the knobs do not say.
  const waveOut = chain.lattice.wave
  const pinned = waveOut ? waveOut.warnings.length : 0

  // The glyph readout is now a COLUMN of the lattice — i = 0 read from the
  // window back — which is exactly the ribbon it grew out of, and still the
  // shortest true statement of what the shape is.
  const glyphs = (chain.lattice.levels[0] ?? [])
    .flatMap((lvl, j) => (j === 0 ? [ROLE_GLYPH[lvl ? 'high' : 'ground']]
      : [ROLE_GLYPH[lvl ? 'rise' : 'fall'], ROLE_GLYPH[lvl ? 'high' : 'ground']]))
    .join('')
  const bar = envelope.frontBar

  return (
    <section className="form-panel" data-testid="strip-panel">
      <header className="form-panel-head">
        <span className="grid-map-title">the network</span>
        <div className="form-panel-history">
          <button type="button" className="tool-btn" disabled={!canUndo} onClick={() => undo()} title="undo (Cmd/Ctrl+Z)">
            undo
          </button>
          <button type="button" className="tool-btn" disabled={!canRedo} onClick={() => redo()} title="redo (Shift+Cmd/Ctrl+Z)">
            redo
          </button>
          <button type="button" className="tool-btn" onClick={() => resetConfig()} title="reset to the shipped default ribbon">
            reset
          </button>
        </div>
      </header>

      {/* --- the knobs the design is actually driven by ---------------------- */}
      <div className="col-profile form-block form-block-primary">
        <div className="stepper-row">
          <span className="slider-label">cells x</span>
          <div className="rows-stepper">
            <button type="button" className="rows-btn" data-testid="lattice-cols-dec"
              disabled={lattice.cols <= LATTICE_COLS_MIN} onClick={() => setCols(lattice.cols - 1)}
              title="one fewer column of flat cells">−</button>
            <span className="rows-value" data-testid="lattice-cols-value">{lattice.cols}</span>
            <button type="button" className="rows-btn" data-testid="lattice-cols-inc"
              disabled={lattice.cols >= LATTICE_COLS_MAX} onClick={() => setCols(lattice.cols + 1)}
              title="one more column of flat cells">+</button>
          </div>
          <input type="range" className="stepper-slider" min={LATTICE_COLS_MIN} max={LATTICE_COLS_MAX}
            step={1} value={lattice.cols} data-testid="lattice-cols"
            onChange={(e) => setCols(Number(e.target.value))} />
        </div>

        <div className="stepper-row">
          <span className="slider-label">cells z</span>
          <div className="rows-stepper">
            <button type="button" className="rows-btn" data-testid="lattice-rows-dec"
              disabled={lattice.rows <= LATTICE_ROWS_MIN} onClick={() => setRows(lattice.rows - 1)}
              title="one fewer row of flat cells">−</button>
            <span className="rows-value" data-testid="lattice-rows-value">{lattice.rows}</span>
            <button type="button" className="rows-btn" data-testid="lattice-rows-inc"
              disabled={lattice.rows >= LATTICE_ROWS_MAX} onClick={() => setRows(lattice.rows + 1)}
              title="one more row of flat cells">+</button>
          </div>
          <input type="range" className="stepper-slider" min={LATTICE_ROWS_MIN} max={LATTICE_ROWS_MAX}
            step={1} value={lattice.rows} data-testid="lattice-rows"
            onChange={(e) => setRows(Number(e.target.value))} />
        </div>

        <p className="form-annotation">
          the lattice is counted in <b>flat cells</b> — the angled panels are derived, one per edge
          between two present cells. {lattice.cols} × {lattice.rows} is{' '}
          <b>{counts.cells} cells + {counts.ramps} ramps = {counts.panels} panels</b>.{' '}
          {isWave
            ? `The heights are h(i,j) = f(i) + g(j), so this network has ${
              chain.lattice.wave?.storeyCount ?? 0} distinct storeys rather than two.`
            : 'Levels alternate like a checkerboard, so every cell\'s neighbours are the opposite '
              + 'level and every edge has somewhere to go.'}
        </p>

        {/* The glyph readout has a two-symbol alphabet for the flats, so it can
            only describe a two-level field. Under the wave it would draw a row of
            `-` and claim the column was flat at one height, which is worse than
            drawing nothing — the wave's profile is the height row below it. */}
        {isWave ? (
          <div className="glyph-readout" data-testid="strip-glyphs" title="column 0's heights, from the window back">
            {(chain.lattice.wave?.heights?.[0] ?? []).map((h) => Math.round(h)).join(' · ') || '—'}
          </div>
        ) : (
          <div className="glyph-readout" data-testid="strip-glyphs" title="the strip's pattern, unit 1 at the window">
            {glyphs || '—'}
          </div>
        )}

        <div className="slider-row">
          <span className="slider-label">angle θ</span>
          <input
            type="range"
            min={ANGLE_MIN}
            max={ANGLE_MAX}
            step={0.5}
            value={angleDeg}
            data-testid="strip-angle"
            onChange={(e) => setAngleDeg(Number(e.target.value))}
          />
          <input
            type="number"
            className="num-input"
            min={ANGLE_MIN}
            max={ANGLE_MAX}
            step={0.5}
            value={angleDeg}
            data-testid="strip-angle-number"
            onChange={(e) => setAngleDeg(e.target.value)}
          />
        </div>
        <p className="form-annotation">
          the single shape parameter. <b>rise</b> and <b>fall</b> share it and are symmetric, so every
          fold in the pattern is θ rather than 2θ — the flats between them split each direction change
          in half, which is why this pattern buys height cheaply.
        </p>

        {/* The permission, stated where the knob is. See the file header. */}
        <div className="limit-block" data-testid="strip-limits">
          {/* Under the wave `maxAngleDeg` is the largest BASE angle the design
              admits, not the largest fold the connector takes — every joint has
              its own fold, so the two are different numbers and merging them
              would be the "silently keep quoting a single-angle boundary"
              failure. The per-joint reading is stated beside it. */}
          {envelope.angleIsPerJoint && envelope.perJoint && (
            <LimitLine testId="limit-per-joint" over={!envelope.perJoint.allClean}>
              {envelope.perJoint.distinctFolds} distinct folds over{' '}
              {envelope.perJoint.jointCount} joints
              {envelope.perJoint.allClean
                ? ' — every one of them inside the connector envelope'
                : `; ${envelope.perJoint.dirtyJointCount} outside it, worst at ` +
                  `${envelope.perJoint.worst.foldDeg.toFixed(2)}° on ${envelope.perJoint.worst.id}` +
                  `${envelope.perJoint.worst.flags.length ? ` (${envelope.perJoint.worst.flags.join(', ')})` : ''}`}
              . The limit below is on the <b>base</b> angle, not on any one fold.
            </LimitLine>
          )}
          {envelope.maxAngleDeg === null ? (
            <LimitLine testId="limit-connector" over>
              no angle works at this gap — the connector is outside its envelope even flat
              {envelope.limitedBy.length > 0 ? ` (${envelope.limitedBy.join(', ')})` : ''}
            </LimitLine>
          ) : angleDeg > envelope.maxAngleDeg ? (
            <LimitLine testId="limit-connector" over>
              {angleDeg}° — past the {envelope.maxAngleDeg.toFixed(2)}° the connector can take at a{' '}
              {cm1(gap)}cm gap
            </LimitLine>
          ) : (
            <LimitLine testId="limit-connector">
              connector limit {envelope.maxAngleDeg.toFixed(2)}°
              {envelope.angleAtRangeLimit
                ? ' — the top of the schema range; the gap is not what stops you'
                : ` — ${envelope.angleHeadroomDeg.toFixed(2)}° of headroom left`}
            </LimitLine>
          )}
          {bar.clears ? (
            <LimitLine testId="limit-frontbar">
              front bar clears — a flat bar lies across a {bar.worstConcaveDeg.toFixed(1)}° valley,
              and it bites past {bar.concaveLimitDeg.toFixed(2)}°
            </LimitLine>
          ) : (
            <LimitLine testId="limit-frontbar" over>
              {bar.worstConcaveDeg.toFixed(1)}° — past the {bar.concaveLimitDeg.toFixed(2)}° a flat
              front bar can lie across a valley. That is a limit on the PART, not on the form: a
              relief or a narrower bar on concave stations moves it, and widening the gap does not.
            </LimitLine>
          )}
        </div>
      </div>

      {/* --- the joint width -------------------------------------------------- */}
      <div className="col-profile form-block">
        <SliderRow
          testId="strip-gap"
          label="gap"
          value={gap}
          min={GAP_MIN}
          max={GAP_MAX}
          step={0.1}
          onChange={(v) => setGap(v)}
          format={(v) => `${cm1(v)}cm`}
        />
        <div className="limit-block">
          {envelope.minGapCm === null ? (
            <LimitLine testId="limit-gap" over>
              no gap in the band admits this angle
            </LimitLine>
          ) : (
            <LimitLine testId="limit-gap" over={envelope.gapHeadroomCm < 0}>
              smallest gap this angle admits: {envelope.minGapCm.toFixed(2)}cm
              {envelope.gapAtRangeLimit
                ? ' — the bottom of the schema range'
                : ` — ${envelope.gapHeadroomCm.toFixed(2)}cm of room`}
            </LimitLine>
          )}
        </div>
        <p className="form-annotation">
          in v4 this is an <b>input the geometry honours exactly</b>, not an outcome that gets
          measured: the bisector construction makes every rim-to-rim span equal <code>gap</code> at
          every joint, at every angle. There is nothing to deviate, so there is no tolerance on it.
        </p>
      </div>

      {/* --- the pattern ------------------------------------------------------
          The KIND is a change of model, not a knob: the checkerboard has two
          levels and one angle, the wave has a separable height field and one
          angle per lattice edge. They do not interpolate, and the controls below
          the switch change entirely with it — `phase` means nothing on a wave and
          the scrunch means nothing on a checkerboard, so neither is shown when it
          is dead. */}
      <div className="col-profile form-block">
        <div className="form-check-row">
          <span className="slider-label">pattern</span>
          <div className="seg-group" data-testid="pattern-kind">
            {PATTERN_KINDS.map((k) => (
              <button
                key={k}
                type="button"
                className={`seg-btn${pattern.kind === k ? ' seg-btn-on' : ''}`}
                data-testid={`pattern-kind-${k}`}
                onClick={() => setPatternKind(k)}
                title={
                  k === 'trapezoid'
                    ? 'the two-level checkerboard — one angle everywhere'
                    : 'a separable height field with a per-edge angle, so the grid can scrunch'
                }
              >
                {k}
              </button>
            ))}
          </div>
        </div>

        {isWave ? (
          <>
            <SliderRow
              testId="wave-scrunch-x"
              label="scrunch x"
              value={waveCfg.scrunchX}
              min={SCRUNCH_MIN}
              max={SCRUNCH_MAX}
              step={0.01}
              onChange={(v) => setWaveKnob('scrunchX', v)}
              format={(v) => `${(v * 100).toFixed(0)}%`}
            />
            <SliderRow
              testId="wave-attractor-x"
              label="attractor x"
              value={waveCfg.attractorX}
              min={ATTRACTOR_MIN}
              max={ATTRACTOR_MAX}
              step={0.05}
              onChange={(v) => setWaveKnob('attractorX', v)}
              format={(v) => (v === 0 ? 'uniform' : `${(v * 100).toFixed(0)}% along`)}
            />
            <SliderRow
              testId="wave-scrunch-z"
              label="scrunch z"
              value={waveCfg.scrunchZ}
              min={SCRUNCH_MIN}
              max={SCRUNCH_MAX}
              step={0.01}
              onChange={(v) => setWaveKnob('scrunchZ', v)}
              format={(v) => `${(v * 100).toFixed(0)}%`}
            />
            <SliderRow
              testId="wave-attractor-z"
              label="attractor z"
              value={waveCfg.attractorZ}
              min={ATTRACTOR_MIN}
              max={ATTRACTOR_MAX}
              step={0.05}
              onChange={(v) => setWaveKnob('attractorZ', v)}
              format={(v) => (v === 0 ? 'uniform' : `${(v * 100).toFixed(0)}% along`)}
            />

            {/* What the knobs actually did, in the units of the thing being
                compressed. The scrunch is a percentage of PLAN ADVANCE, and the
                angles are whatever delivers it — so quoting the angle range
                beside the run is the only way to read the control honestly. */}
            <div className="limit-block" data-testid="wave-readout">
              {['x', 'z'].map((ax) => {
                const w = waveOut?.[ax]
                if (!w || w.edgeCount === 0) {
                  return (
                    <LimitLine key={ax} testId={`wave-run-${ax}`}>
                      no {ax} edges — a single line of cells has nothing to scrunch
                    </LimitLine>
                  )
                }
                const lo = Math.min(...w.angleDeg)
                const hi = Math.max(...w.angleDeg)
                return (
                  <LimitLine key={ax} testId={`wave-run-${ax}`}>
                    <b>{ax}</b> · θ {lo.toFixed(1)}°→{hi.toFixed(1)}° · pitch{' '}
                    {w.pitchCm[0].toFixed(1)}→{w.pitchCm[w.pitchCm.length - 1].toFixed(1)}cm · run{' '}
                    {w.planRunCm.toFixed(1)} of {w.unscrunchedRunCm.toFixed(1)}cm (
                    {((1 - w.compression) * 100).toFixed(1)}% shorter)
                  </LimitLine>
                )
              })}
              {pinned > 0 && (
                <LimitLine testId="wave-pinned" over>
                  {pinned} edge{pinned === 1 ? '' : 's'} could not compress that far and{' '}
                  {pinned === 1 ? 'is' : 'are'} pinned at {ANGLE_MAX}° — the scrunch asked for a plan
                  advance no angle in the band delivers.
                </LimitLine>
              )}
            </div>

            <p className="form-annotation">
              the <b>angle above is the base angle, at the front</b>, and it is the lowest in the
              design — scrunching only ever steepens. What varies linearly is the <b>plan advance</b>,
              not the angle: each edge gives up its share of{' '}
              <code>60·cos θ + 2·gap·cos(θ/2)</code> and the angle that delivers it is solved for. The
              attractor is where full scrunch is reached; past it the grid stays uniformly tight.
            </p>
            <p className="form-hint">
              the wave is a <b>different shape</b>, not a setting on the checkerboard. Its height
              field is <code>h(i,j) = f(i) + g(j)</code>, which is the only family whose cycles close
              at a varying angle — a two-level checkerboard forces every angle equal, and there is no
              gap that buys it back. So even at 0% this is an <b>egg-crate with three storeys</b>,
              not the two-level board. <b>Braced</b> builds no wall anchors here: there is no single
              level rise for one to descend.
            </p>
          </>
        ) : (
          <>
            <SliderRow
              testId="strip-phase"
              label="phase"
              value={pattern.phase}
              min={PHASE_MIN}
              max={PHASE_MAX}
              step={1}
              onChange={(v) => setPhase(v)}
              format={(v) => (v ? '1 · high at the wall' : '0 · ground at the wall')}
            />
            <p className="form-annotation">
              which level cell (0, 0) sits on. The checkerboard has period 2, so this simply swaps
              ground and high across the whole lattice.
            </p>
          </>
        )}

        <div className="form-check-row">
          <span className="slider-label">brace to wall</span>
          <div className="seg-group" data-testid="wall-anchor">
            {WALL_ANCHORS.map((mode) => (
              <button
                key={mode}
                type="button"
                className={`seg-btn${placement.wallAnchor === mode ? ' seg-btn-on' : ''}`}
                onClick={() => setWallAnchor(mode)}
              >
                {mode}
              </button>
            ))}
          </div>
        </div>
        <p className="form-hint">
          <b>braced</b> runs one extra ramp down to the floor from every high cell in the column
          nearest the wall ({counts.anchorRamps} of them here), so the network is propped against
          the wall line rather than cantilevered off its own edge. <b>What the toe fixes to is not
          modelled</b> — the geometry only says where it lands.
        </p>
      </div>

      {/* --- the room's obstacles ---------------------------------------------- */}
      {(config.obstacles ?? []).map((o) => {
        const solved = report.obstacles.find((r) => r.id === o.id)
        const x0 = solved?.extents.min[0]
        const z0 = solved?.extents.min[2]
        return (
          <div className="col-profile form-block" key={o.id} data-testid={`obstacle-${o.id}`}>
            <div className="form-check-row">
              <span className="slider-label">{o.label}</span>
              <div className="seg-group">
                {['corner', 'centre'].map((a) => (
                  <button
                    key={a}
                    type="button"
                    className={`seg-btn${o.anchor === a ? ' seg-btn-on' : ''}`}
                    onClick={() => setObstacleField(o.id, 'anchor', a)}
                    title={a === 'corner' ? 'x/z locate its near corner' : 'x/z locate its centre'}
                  >
                    {a}
                  </button>
                ))}
              </div>
            </div>
            <div className="obstacle-fields">
              {[['xCm', 'x'], ['zCm', 'z'], ['widthCm', 'w'], ['depthCm', 'd']].map(([f, lbl]) => (
                <label key={f} className="obstacle-field">
                  <span>{lbl}</span>
                  <input
                    type="number"
                    value={o[f]}
                    step={5}
                    data-testid={`obstacle-${o.id}-${f}`}
                    onChange={(e) => setObstacleField(o.id, f, e.target.value)}
                  />
                </label>
              ))}
            </div>
            <p className={solved?.hitCount ? 'form-annotation obstacle-hit' : 'form-annotation'}>
              measured from the window/wall corner, so it occupies{' '}
              <b>x {x0}–{solved?.extents.max[0]}, z {z0}–{solved?.extents.max[2]}cm</b>.{' '}
              {solved?.hitCount
                ? `${solved.hitCount} panel${solved.hitCount === 1 ? '' : 's'} currently run through it — they are outlined red in the plan.`
                : `Nothing touches it; the nearest panel is ${solved?.nearestClearanceCm?.toFixed(0) ?? '—'}cm away.`}
              {' '}<b>corner</b> reads x/z as its near face, <b>centre</b> as its middle — a 25cm
              difference on a 50cm column, so check the extents above match the tape.
            </p>
          </div>
        )
      })}

      {/* --- where it sits ---------------------------------------------------- */}
      <div className="col-profile form-block">
        <SliderRow
          testId="strip-wall-offset"
          label="from wall (x)"
          value={placement.wallOffsetCm}
          min={WALL_OFFSET_MIN}
          max={WALL_OFFSET_MAX}
          step={1}
          onChange={(v) => setWallOffset(v)}
          format={(v) => `${Math.round(v)}cm`}
        />
        <SliderRow
          testId="strip-window-offset"
          label="from window (z)"
          value={placement.windowOffsetCm}
          min={WINDOW_OFFSET_MIN}
          max={WINDOW_OFFSET_MAX}
          step={1}
          onChange={(v) => setWindowOffset(v)}
          format={(v) => `${Math.round(v)}cm`}
        />
        <label className="form-check-row">
          <input
            type="checkbox"
            data-testid="strip-ground"
            checked={placement.groundToFloor}
            onChange={(e) => setGroundToFloor(e.target.checked)}
          />
          <span className="slider-label">sit on the floor</span>
        </label>
        <SliderRow
          testId="strip-ground-clearance"
          label="ground clearance"
          value={placement.groundClearanceCm}
          min={GROUND_CLEARANCE_MIN}
          max={GROUND_CLEARANCE_MAX}
          step={0.5}
          disabled={!placement.groundToFloor}
          onChange={(v) => setGroundClearance(v)}
          format={(v) => `${cm1(v)}cm`}
        />
        <p className="form-annotation">
          on, the lowest point of any present panel is dropped to the <b>ground clearance</b> — a
          property of the whole network, so removing whichever panel was lowest legitimately moves
          everything. Off, cell (0,0)'s reference plane sits at y = 0, panels may go below the floor
          (which the report flags), and no spacers are solved.
          {spacers.grounded && spacers.count > 0 && (
            <>
              {' '}The gap is held open by <b>{spacers.count} spacers</b> under{' '}
              {spacers.cellCount} floor-resting cell{spacers.cellCount === 1 ? '' : 's'} —{' '}
              {spacers.perEdge} per edge, placed by the same spacing rule as the connectors.
            </>
          )}
        </p>
      </div>

      {/* --- connectors -------------------------------------------------------
          Length / spacing / count are about the STRUCTURE — these parts are the
          structure, there is no substructure. The bins are about the PRINT QUEUE
          and change nothing physical. Kept in one block but read in that order. */}
      <div className="col-profile form-block">
        <SliderRow
          testId="strip-connector-length"
          label="part length"
          value={connectors.lengthCm}
          min={CONNECTOR_LENGTH_MIN}
          max={CONNECTOR_LENGTH_MAX}
          step={0.5}
          onChange={(v) => setConnectorKnob('lengthCm', v)}
          format={(v) => `${cm1(v)}cm`}
        />
        <SliderRow
          testId="strip-connector-spacing"
          label="max unsupported"
          value={connectors.spacingCm}
          min={CONNECTOR_SPACING_MIN}
          max={CONNECTOR_SPACING_MAX}
          step={5}
          onChange={(v) => setConnectorKnob('spacingCm', v)}
          format={(v) => `${cm1(v)}cm`}
        />
        <SliderRow
          testId="strip-connector-min-per-joint"
          label="min per joint"
          value={connectors.minPerJoint}
          min={CONNECTOR_MIN_PER_JOINT_MIN}
          max={CONNECTOR_MIN_PER_JOINT_MAX}
          step={1}
          onChange={(v) => setConnectorKnob('minPerJoint', v)}
          format={int0}
        />
        <p className="form-annotation">
          {connectors.minPerJoint === 1
            ? 'one part on a joint is a HINGE — it is free to rotate about it, and there is no substructure to stop it'
            : 'a v4 joint is 60cm of straight, parallel rim, so parts sit where the spacing rule puts them'}
        </p>
        <SliderRow
          testId="strip-connector-bin-span"
          label="span bin"
          value={connectors.binSpanCm}
          min={CONNECTOR_BIN_SPAN_MIN}
          max={CONNECTOR_BIN_SPAN_MAX}
          step={0.05}
          onChange={(v) => setConnectorKnob('binSpanCm', v)}
          format={(v) => `${num2(v)}cm`}
        />
        <SliderRow
          testId="strip-connector-bin-angle"
          label="angle bin"
          value={connectors.binAngleDeg}
          min={CONNECTOR_BIN_ANGLE_MIN}
          max={CONNECTOR_BIN_ANGLE_MAX}
          step={0.5}
          onChange={(v) => setConnectorKnob('binAngleDeg', v)}
          format={(v) => `${cm1(v)}°`}
        />
        <label className="form-select-row">
          <span className="slider-label">supply edge</span>
          <select
            className="form-select"
            data-testid="strip-power-edge"
            value={connectors.powerEdge}
            onChange={(e) => setConnectorKnob('powerEdge', e.target.value)}
          >
            {CONNECTOR_POWER_EDGES.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        </label>
        <label className="form-select-row">
          <span className="slider-label">supply mode</span>
          <select
            className="form-select"
            data-testid="strip-supply-mode"
            value={connectors.supplyMode}
            onChange={(e) => setConnectorKnob('supplyMode', e.target.value)}
          >
            {CONNECTOR_SUPPLY_MODES.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </label>
        <p className="form-hint">
          <b>relief</b> — the supply is flush with the flange to within 1mm, so a relief in the lip
          clears it and the joint carries a part that bears on the supply housing. <b>block</b> treats
          it as solid, which is the stricter reading and the way to measure what the constraint costs.
        </p>
      </div>
    </section>
  )
}
