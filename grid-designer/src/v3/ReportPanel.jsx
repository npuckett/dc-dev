/**
 * grid-designer v3 — the numbers: what the connectors have to absorb.
 *
 * Straight off `buildReport` (src/core/v3/report.js), which is the tool's
 * primary output (see that file's header) — this panel's whole job is to make
 * a bad number impossible to miss rather than buried in a scrollable list.
 * Shows:
 *   - worst/mean joint gap deviation vs `gapTolerance`, and how many joints
 *     are flagged;
 *   - the HOLONOMY block, only in 'chain' mode (tree-edge worst vs
 *     cycle-edge worst — the contrast is the point). Nothing is shown in
 *     'surface-fit' mode rather than a fake zero — `report.holonomy` itself
 *     carries `treeEdges: null, cycleEdges: null` there, so "nothing" is the
 *     honest render, not a placeholder this component invents;
 *   - shape residual sigma (how far the realised panels sit from the target);
 *   - plate/tile counts and the worst plate sagitta;
 *   - collision count with the deepest overlap (a hard buildability failure —
 *     styled the same alarming red as the viewport's collision highlight);
 *   - per-edge (wall / window) grounding clearance — V3_SPEC.md §5 calls this
 *     out as the brief's own requirement, so it gets real space, not an
 *     afterthought;
 *   - PINNED vs. algorithm-chosen tile counts (P8, manual overrides —
 *     `tile.pinned`), and any `W_PLATE_OVERRIDE_MISFIT` warning in its own
 *     styled-as-bad section, pulled OUT of the generic warnings list so a
 *     plate the user forced into a place it does not fit cannot get lost in
 *     it — see tiling.js's "MANUAL OVERRIDES" for why that warning exists and
 *     why it is never a validation error (a manual override is placed and
 *     reported, never refused);
 *   - warnings / violations, `E_UNSUPPORTED` especially (the "does it stand
 *     up" check — styled as an error, not a warning, regardless of which
 *     array it arrives in).
 */

import useStoreV3, { getDerived } from './store.js'
import { OVERRIDE_MISFIT_CODE } from '../core/v3/tiling.js'

const cm = (v, d = 2) => `${v.toFixed(d)}cm`
const deg = (v) => `${v.toFixed(1)}°`

/**
 * A joint-report metric badge, red past `bad`, amber past `warn`, else calm.
 * `raw` is the NUMBER the threshold compares against; `value` is the already
 * -formatted display string (e.g. "2.34cm") — kept separate so formatting
 * never corrupts the comparison (a string >= number comparison coerces the
 * string to NaN, silently defeating the threshold).
 */
function Metric({ label, value, testId, raw, bad, warn }) {
  const cls =
    raw !== undefined && bad !== undefined && raw >= bad
      ? 'metric-bad'
      : raw !== undefined && warn !== undefined && raw >= warn
        ? 'metric-warn'
        : 'metric-ok'
  return (
    <div className={`report-metric ${cls}`} data-testid={testId}>
      <span className="report-metric-label">{label}</span>
      <span className="report-metric-value">{value}</span>
    </div>
  )
}

export default function ReportPanel() {
  const config = useStoreV3((s) => s.config)
  const { report, layout } = getDerived(config)
  const { summary, holonomy, fit, collisions, connectors: conn, support, warnings, violations } = report

  const worstDeviationBad = summary.gapToleranceCm > 0 ? summary.gapToleranceCm : 1
  const unsupported = violations.some((v) => v.code === 'E_UNSUPPORTED')

  // --- P8: manual overrides — pinned vs. algorithm-chosen, and any misfit ---
  // pulled out of the generic warnings list so it cannot be buried (see the
  // file header). `otherWarnings` is what the generic list below renders.
  const pinnedCount = layout.tiles.filter((t) => t.pinned).length
  // null = unlimited. Shown as "used / budget" so a build against a real plate
  // inventory can be read at a glance.
  const budget = config?.tiling?.maxPlates ?? null
  const overrideMisfits = warnings.filter((w) => w.code === OVERRIDE_MISFIT_CODE)
  const otherWarnings = warnings.filter((w) => w.code !== OVERRIDE_MISFIT_CODE)

  return (
    <section className="report-panel" data-testid="report-panel">
      <header className="grid-map-head">
        <span className="grid-map-title">joint / fit report</span>
      </header>

      {/* --- headline verdict — impossible to miss ------------------------- */}
      <div
        className={`report-verdict${collisions.length > 0 || unsupported || conn.summary.infeasible > 0 ? ' report-verdict-bad' : ''}`}
        data-testid="report-verdict"
      >
        {collisions.length > 0
          ? `${collisions.length} panel collision${collisions.length === 1 ? '' : 's'} — not buildable as drawn`
          : unsupported
            ? 'assembly is unsupported — it tips over'
            // A part that cannot exist is the same class of failure as two
            // panels occupying the same space, so it is said at the same volume.
            : conn.summary.infeasible > 0
              ? `${conn.summary.infeasible} connector${conn.summary.infeasible === 1 ? '' : 's'} cannot be built — the gap is too narrow for the fold`
              : summary.worst > worstDeviationBad
                ? `worst joint deviation ${cm(summary.worst)} exceeds tolerance ${cm(summary.gapToleranceCm)}`
                : `no collisions · joints within tolerance · ${conn.summary.partTypes} connector type${conn.summary.partTypes === 1 ? '' : 's'}`}
      </div>

      {/* --- joints --------------------------------------------------------- */}
      <div className="report-section">
        <h4 className="report-section-title">joints</h4>
        <div className="report-metric-grid">
          <Metric
            testId="report-worst-gap"
            label="worst gap dev"
            value={cm(summary.worst)}
            raw={summary.worst}
            bad={worstDeviationBad}
          />
          <Metric testId="report-mean-gap" label="mean gap dev" value={cm(summary.mean)} />
          <Metric testId="report-gap-tol" label="tolerance" value={cm(summary.gapToleranceCm)} />
          <Metric
            testId="report-flagged"
            label="flagged"
            value={`${summary.flagged} / ${summary.count}`}
            raw={summary.flagged}
            bad={1}
          />
        </div>
        <p className="report-detail">
          worst dihedral {deg(summary.worstDihedralDeg)} · worst skew {deg(summary.worstSkewDeg)}
          {summary.pinched > 0 ? ` · ${summary.pinched} pinched joint${summary.pinched === 1 ? '' : 's'}` : ''}
        </p>
      </div>

      {/* --- holonomy — chain mode only, never a fake zero in surface-fit --- */}
      {holonomy.mode === 'chain' && (
        <div className="report-section" data-testid="report-holonomy">
          <h4 className="report-section-title">holonomy (chain mode)</h4>
          <p className="report-detail">
            tree edges are exact by construction; the closure error concentrates on the
            cycle-closing edges — that contrast <b>is</b> the measurement.
          </p>
          <div className="report-metric-grid">
            <Metric testId="report-holonomy-tree" label="tree edges worst" value={cm(holonomy.treeEdges.worst)} />
            <Metric
              testId="report-holonomy-cycle"
              label="cycle edges worst"
              value={cm(holonomy.cycleEdges.worst)}
              raw={holonomy.cycleEdges.worst}
              bad={worstDeviationBad}
            />
          </div>
          {holonomy.worstJoint && (
            <p className="report-detail">
              worst cycle joint: <code>{holonomy.worstJoint.a}</code> ↔ <code>{holonomy.worstJoint.b}</code> at{' '}
              {cm(holonomy.worstJoint.deviationCm)}
            </p>
          )}
        </div>
      )}

      {/* --- fit against the target ------------------------------------------ */}
      <div className="report-section">
        <h4 className="report-section-title">fit vs. target</h4>
        <div className="report-metric-grid">
          <Metric testId="report-sigma" label="shape residual σ" value={cm(fit.shapeResidualSigmaCm)} />
          <Metric testId="report-tiles" label="tiles" value={`${fit.tileCount}`} />
          <Metric
            testId="report-plates"
            label={budget === null ? 'plates' : 'plates / budget'}
            value={budget === null ? `${fit.plateCount}` : `${fit.plateCount} / ${budget}`}
            raw={fit.plateCount}
            bad={budget === null ? undefined : budget + 1}
          />
          <Metric
            testId="report-sagitta"
            label="worst plate sagitta"
            value={cm(fit.worstPlateSagittaCm)}
            raw={fit.worstPlateSagittaCm}
            bad={fit.plateFitToleranceCm}
          />
          <Metric
            testId="report-pinned"
            label="pinned / algorithm"
            value={`${pinnedCount} / ${fit.tileCount - pinnedCount}`}
          />
        </div>
        <p className="report-detail">
          angularity {fit.angularity.toFixed(2)} · {fit.facetCount} facet plane{fit.facetCount === 1 ? '' : 's'} ·
          plate fit tol {cm(fit.plateFitToleranceCm)}
        </p>
      </div>

      {/* --- manual overrides that missed tolerance — pulled out of the generic
          warnings list on purpose (P8): a plate the user forced into a place
          it does not fit must not be buried. See tiling.js's "MANUAL
          OVERRIDES" — it is placed regardless, this only reports the cost. */}
      {overrideMisfits.length > 0 && (
        <div className="report-section report-section-bad" data-testid="report-override-misfits">
          <h4 className="report-section-title">manual overrides — misfit</h4>
          <p className="report-detail report-bad-text">
            {overrideMisfits.length} manually-placed plate{overrideMisfits.length === 1 ? '' : 's'} exceed
            {overrideMisfits.length === 1 ? 's' : ''} the fit tolerance — placed anyway, as requested
          </p>
          <ul className="report-list">
            {overrideMisfits.map((w, i) => (
              <li key={i}>
                <code>{w.tile}</code> — sagitta {cm(w.sagittaCm)} vs. tolerance {cm(w.toleranceCm)}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* --- collisions — a hard buildability failure ------------------------ */}
      <div className={`report-section${collisions.length > 0 ? ' report-section-bad' : ''}`} data-testid="report-collisions">
        <h4 className="report-section-title">collisions</h4>
        {collisions.length === 0 ? (
          <p className="report-detail report-good">none</p>
        ) : (
          <>
            <p className="report-detail report-bad-text">
              {collisions.length} pair{collisions.length === 1 ? '' : 's'} interpenetrating — deepest{' '}
              {cm(collisions[0].depthCm)} (<code>{collisions[0].a}</code> ↔ <code>{collisions[0].b}</code>)
            </p>
            <ul className="report-list">
              {collisions.slice(0, 6).map((c, i) => (
                <li key={i}>
                  <code>{c.a}</code> ↔ <code>{c.b}</code> — {cm(c.depthCm)}
                </li>
              ))}
              {collisions.length > 6 && <li>… and {collisions.length - 6} more</li>}
            </ul>
          </>
        )}
      </div>

      {/* --- connectors: the printed kit (P11) -------------------------------
          The counterpart to the plate budget. `partTypes` is the number of
          distinct things that have to come off a printer, and it is set by the
          bin knobs, not by the design — so the two are shown together, and the
          worst forced deviation says what the coarser bins cost. */}
      <div
        className={`report-section${conn.summary.clashes > 0 || conn.summary.infeasible > 0 ? ' report-section-bad' : ''}`}
        data-testid="report-connectors"
      >
        <h4 className="report-section-title">connectors</h4>
        <div className="report-metric-grid">
          <Metric testId="report-conn-count" label="parts" value={`${conn.summary.count}`} />
          <Metric testId="report-conn-types" label="unique types" value={`${conn.summary.partTypes}`} />
          <Metric
            testId="report-conn-span"
            label="gap span"
            value={`${conn.summary.spanCm.min.toFixed(1)}–${conn.summary.spanCm.max.toFixed(1)}cm`}
            raw={conn.summary.spanCm.max}
            bad={conn.limits.maxSpanCm}
          />
          <Metric testId="report-conn-fold" label="worst fold" value={deg(conn.summary.worstFoldDeg)} />
          <Metric
            testId="report-conn-spread"
            label="worst wedge / part"
            value={cm(conn.summary.worstSpanSpreadCm)}
            raw={conn.summary.worstSpanSpreadCm}
            bad={conn.limits.maxSpanSpreadCm}
          />
          <Metric
            testId="report-conn-flagged"
            label="flagged"
            value={`${conn.summary.flagged} / ${conn.summary.count}`}
            raw={conn.summary.flagged}
            warn={1}
          />
        </div>
        <p className="report-detail">
          {conn.summary.lengthCm}cm parts, centred in the gap · bins {conn.summary.binSpanCm}cm /{' '}
          {conn.summary.binAngleDeg}° · worst forced fit {cm(conn.summary.worstBinSpanErrorCm)} and{' '}
          {deg(conn.summary.worstBinFoldErrorDeg)}
        </p>

        {/* The power supply. Given its own line above the clashes because it is
            not a warning about a part — it is a joint that gets no part at all,
            and that is a structural hole rather than a tolerance. */}
        {conn.summary.blockedJoints > 0 && (
          <p className="report-detail report-bad-text" data-testid="report-conn-blocked">
            {conn.summary.blockedJoints} of {conn.summary.jointCount} joints carry NO connector — a panel
            power supply sits on the flange behind them, leaving only{' '}
            {cm(conn.summary.clearEndCm, 0)} of usable rim at each end of that edge. A part must be{' '}
            {cm(conn.summary.clearEndCm, 0)} or shorter to fit there.
          </p>
        )}
        {conn.summary.reducedJoints > 0 && (
          <p className="report-detail" data-testid="report-conn-reduced">
            {conn.summary.reducedJoints} more joint{conn.summary.reducedJoints === 1 ? '' : 's'} carry fewer
            parts than asked for, for the same reason.
          </p>
        )}

        {conn.summary.clashes > 0 && (
          <p className="report-detail report-bad-text" data-testid="report-conn-clashes">
            {conn.summary.clashes} part clash{conn.summary.clashes === 1 ? '' : 'es'} — deepest{' '}
            {cm(conn.clashes[0].depthCm)} (<code>{conn.clashes[0].station}</code> ↔{' '}
            <code>{conn.clashes[0].against}</code>)
          </p>
        )}
        {conn.summary.infeasible > 0 && (
          <p className="report-detail report-bad-text" data-testid="report-conn-infeasible">
            {conn.summary.infeasible} part{conn.summary.infeasible === 1 ? '' : 's'} cannot be built — the gap
            is too narrow for the fold, so the two hooks would pass through each other
          </p>
        )}

        {/* The kit itself. Truncated because a tight bin can produce one type
            per part, and that IS the answer sometimes — the count above is the
            headline, this is the detail. */}
        <ul className="report-list" data-testid="report-conn-kit">
          {conn.kit.slice(0, 8).map((part) => (
            <li key={part.partId}>
              <code>{part.partId}</code> ×{part.count} — gap{' '}
              {part.spanStartCm === part.spanEndCm
                ? cm(part.spanStartCm, 1)
                : `${part.spanStartCm.toFixed(1)}→${part.spanEndCm.toFixed(1)}cm`}
              , fold {deg(part.foldDeg)}
            </li>
          ))}
          {conn.kit.length > 8 && <li>… and {conn.kit.length - 8} more types</li>}
        </ul>
      </div>

      {/* --- grounding: per-edge clearance, the brief's own requirement ------ */}
      <div className="report-section">
        <h4 className="report-section-title">grounding (wall / window edges)</h4>
        {support.edges.length === 0 ? (
          <p className="report-detail">no edge tiles reachable</p>
        ) : (
          <div className="report-metric-grid report-metric-grid-wide">
            {support.edges.map((e) => (
              <div key={e.edge} className="report-edge-card" data-testid={`report-edge-${e.edge}`}>
                <div className="report-edge-title">{e.edge}</div>
                <p className="report-detail">
                  {e.grounded} / {e.tiles} tiles grounded · {e.flatTiles} too flat
                </p>
                <p className="report-detail">
                  max clearance {cm(e.maxClearanceCm)} · mean {cm(e.meanClearanceCm)}
                </p>
              </div>
            ))}
          </div>
        )}
        <p className="report-detail" data-testid="report-support-hull">
          {support.comInsideHull ? 'centre of mass inside support hull ✓' : 'centre of mass OUTSIDE support hull'} ·{' '}
          {support.contacts.length} ground contact point{support.contacts.length === 1 ? '' : 's'}
        </p>
      </div>

      {/* --- violations (hard) then warnings (soft) -------------------------- */}
      {violations.length > 0 && (
        <div className="msg-list msg-errors" data-testid="report-violations">
          <strong>buildability violations</strong>
          {violations.map((v, i) => (
            <p key={i}>
              <code>{v.code}</code> {v.message}
            </p>
          ))}
        </div>
      )}
      {otherWarnings.length > 0 && (
        <div className="msg-list msg-warnings" data-testid="report-warnings">
          {otherWarnings.map((w, i) => (
            <p key={i}>
              <code>{w.code}</code> {w.message}
            </p>
          ))}
        </div>
      )}

      <p className="report-detail" data-testid="report-bounds">
        overall {Math.round(layout.bounds.size[0])} × {Math.round(layout.bounds.size[1])} ×{' '}
        {Math.round(layout.bounds.size[2])} cm (W × peak H × D)
      </p>
    </section>
  )
}
