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
 *   - warnings / violations, `E_UNSUPPORTED` especially (the "does it stand
 *     up" check — styled as an error, not a warning, regardless of which
 *     array it arrives in).
 */

import useStoreV3, { getDerived } from './store.js'

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
  const { summary, holonomy, fit, collisions, support, warnings, violations } = report

  const worstDeviationBad = summary.gapToleranceCm > 0 ? summary.gapToleranceCm : 1
  const unsupported = violations.some((v) => v.code === 'E_UNSUPPORTED')

  return (
    <section className="report-panel" data-testid="report-panel">
      <header className="grid-map-head">
        <span className="grid-map-title">joint / fit report</span>
      </header>

      {/* --- headline verdict — impossible to miss ------------------------- */}
      <div className={`report-verdict${collisions.length > 0 || unsupported ? ' report-verdict-bad' : ''}`} data-testid="report-verdict">
        {collisions.length > 0
          ? `${collisions.length} panel collision${collisions.length === 1 ? '' : 's'} — not buildable as drawn`
          : unsupported
            ? 'assembly is unsupported — it tips over'
            : summary.worst > worstDeviationBad
              ? `worst joint deviation ${cm(summary.worst)} exceeds tolerance ${cm(summary.gapToleranceCm)}`
              : 'no collisions · joints within tolerance'}
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
          <Metric testId="report-plates" label="plates" value={`${fit.plateCount}`} />
          <Metric
            testId="report-sagitta"
            label="worst plate sagitta"
            value={cm(fit.worstPlateSagittaCm)}
            raw={fit.worstPlateSagittaCm}
            bad={fit.plateFitToleranceCm}
          />
        </div>
        <p className="report-detail">
          angularity {fit.angularity.toFixed(2)} · {fit.facetCount} facet plane{fit.facetCount === 1 ? '' : 's'} ·
          plate fit tol {cm(fit.plateFitToleranceCm)}
        </p>
      </div>

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
      {warnings.length > 0 && (
        <div className="msg-list msg-warnings" data-testid="report-warnings">
          {warnings.map((w, i) => (
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
