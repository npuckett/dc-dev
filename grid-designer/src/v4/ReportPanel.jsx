/**
 * grid-designer v4 — the report, led by the envelope.
 *
 * =============================================================================
 * THE ENVELOPE GOES FIRST BECAUSE IT IS THE THING v3 COULD NOT SAY
 * =============================================================================
 * v3's verdict on itself (HANDOFF §0) was that the tool became very good at
 * saying no and never acquired a way to say yes. Its report was a list of
 * measured damage — every number correct, none of them a route to a design that
 * works. A v4 design is inside the connector envelope BY CONSTRUCTION, so the
 * joint table below is nearly always clean and would be nearly worthless on its
 * own. What this panel owes the user is the boundary:
 *
 *     maxAngleDeg   the largest θ this gap admits with every joint clean
 *     minGapCm      the smallest gap that admits this angle
 *
 * and the headroom to each. "You have 3.6° left" is a sentence v3 could not say
 * about anything, so it is the first thing on the panel and it is stated as a
 * permission rather than as a complaint.
 *
 * =============================================================================
 * THE FRONT BAR IS READ SEPARATELY, AND SAYS SO
 * =============================================================================
 * `envelope.frontBar` is deliberately NOT folded into `maxAngleDeg`, and
 * report.js's header gives both reasons. The one this panel has to communicate
 * is the second: **it is a limit on the PART, not on the form.** A flat front bar
 * bites the bezels in a valley past ~12.37° — at every gap, because the overhang
 * that collides is the bar's own lip and does not care how wide the joint is —
 * and that is fixable with a relief, a chamfer, or a narrower bar on concave
 * stations. A limit you can design away must never be shown in the same number
 * as one you cannot (the connector's), or the user will trade the wrong thing.
 *
 * =============================================================================
 * EVERY CODE GETS PLAIN LANGUAGE
 * =============================================================================
 * The core's warnings already carry good sentences, and the per-station FLAGS do
 * not — `connectorStationFlags` returns bare codes. `FLAG_TEXT` below is that
 * missing half, written in the UI layer because the codes come from v3's frozen
 * connectors.js. Warnings are grouped by code rather than listed flat: on a
 * trapezoid wave every valley raises the same front-bar warning, and eight
 * identical paragraphs say less than one paragraph and a count.
 */

import useStoreV4, { getDerived } from './store.js'
import { ADVISORY_FLAGS, getConnectorKit } from './exportAdapter.js'

const cm = (v, d = 2) => `${v.toFixed(d)}cm`
const deg = (v, d = 2) => `${v.toFixed(d)}°`

/**
 * What each flag actually means, in the room. The codes come from v3's frozen
 * `connectorStationFlags` / the v4 report; these sentences do not exist there.
 */
const FLAG_TEXT = {
  W_CONNECTOR_INFEASIBLE:
    "the part's own section self-intersects — the gap is too narrow for this fold, so the two hooks would pass through each other",
  W_CONNECTOR_PINCH: 'the gap is narrower than anything can be fitted into',
  W_CONNECTOR_SPAN: 'the gap is wider than the spine can strap — it becomes a beam',
  W_CONNECTOR_TWIST: 'the part wedges too much along its own length (impossible on a v4 joint: the rims are parallel)',
  W_FASTENER_PINCHED:
    'no room for the bolt at the depth the insert sits — a convex joint closes with depth, so a fastener that clears at the rim can still be pinched below it',
  W_BEARS_ON_POWER_SUPPLY:
    "the panel's power supply is behind this flange, so the lip bears on the supply housing rather than the panel frame",
  W_BACK_HALF_FOULS_PANEL: 'the back half bites into a panel it is gripping',
  W_FRONT_BAR_FOULS_PANEL: 'the front bar bites a panel (dormant — bar widths are assigned after flagging)',
  W_FRONT_BAR_FOULS_BEZEL:
    'a flat front bar cannot lie across this valley — it meets the rising bezels. Fixable in the part, not in the form',
  W_PANELS_COLLIDE_AT_JOINT: 'the two panels themselves meet at this joint before the connector does',
  W_CONNECTOR_CLASH: 'this part occupies the same space as another part or a panel it does not grip',
  W_CONNECTOR_CROWDED: 'the parts on this joint are shortened from their asked-for length to fit',
  W_JOINT_FLIP_MISMATCH:
    'one panel is flipped and one is not, so their back flanges are on opposite sides — no connector of this family can grip both',
  W_JOINT_BLOCKED_BY_POWER_SUPPLY: 'a power supply leaves too little usable rim — this joint carries NO connector',
  W_JOINT_REDUCED_BY_POWER_SUPPLY: 'a power supply costs this joint some of the parts it asked for',
  W_BELOW_FLOOR: 'a panel reaches below y = 0 — grounding is off, or the y offset is negative',
  W_THROUGH_WALL: 'a panel reaches through the wall plane at x = 0',
  W_OUTSIDE_ENVELOPE: 'this design is outside the connector envelope as drawn',
}

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
  const config = useStoreV4((s) => s.config)
  const hoveredUnitId = useStoreV4((s) => s.hoveredUnitId)
  const { chain, connectors, report } = getDerived(config)
  const { envelope, joints, collisions, warnings } = report
  const kit = getConnectorKit(config, chain, connectors)
  const bar = envelope.frontBar

  // Grouped, because a trapezoid wave raises the same warning once per valley.
  const grouped = new Map()
  for (const w of warnings) {
    if (!grouped.has(w.code)) grouped.set(w.code, [])
    grouped.get(w.code).push(w)
  }

  // Advisory flags are excluded from the count for the reason in exportAdapter's
  // `ADVISORY_FLAGS`: every joint of a powered strip carries one at every angle,
  // so "8 of 8 flagged" would be an alarm about a clean design.
  const flaggedJoints = joints.filter((j) => j.flags.some((f) => !ADVISORY_FLAGS.has(f))).length

  return (
    <section className="report-panel" data-testid="report-panel">
      <header className="grid-map-head">
        <span className="grid-map-title">envelope / joints</span>
      </header>

      {/* --- the verdict, and it is allowed to be a yes ---------------------- */}
      <div
        className={`report-verdict${envelope.clean ? '' : ' report-verdict-bad'}`}
        data-testid="report-verdict"
      >
        {collisions.length > 0
          ? `${collisions.length} panel collision${collisions.length === 1 ? '' : 's'} — not buildable as drawn`
          : !envelope.clean
            ? `outside the connector envelope — ${envelope.limitedBy.map((f) => FLAG_TEXT[f] ?? f).join('; ')}`
            : envelope.maxAngleDeg === null
              ? 'inside the envelope'
              : `inside the envelope · ${deg(envelope.angleHeadroomDeg)} of angle and ${cm(envelope.gapHeadroomCm)} of gap left`}
      </div>

      {/* --- the envelope ---------------------------------------------------- */}
      <div className="report-section" data-testid="report-envelope">
        <h4 className="report-section-title">the envelope — how much room is left</h4>
        <div className="report-metric-grid">
          <Metric
            testId="envelope-max-angle"
            label="max angle at this gap"
            value={envelope.maxAngleDeg === null ? 'none' : deg(envelope.maxAngleDeg)}
            raw={envelope.maxAngleDeg === null ? 1 : 0}
            bad={1}
          />
          <Metric
            testId="envelope-angle-headroom"
            label="angle headroom"
            value={envelope.angleHeadroomDeg === null ? '—' : deg(envelope.angleHeadroomDeg)}
            raw={envelope.angleHeadroomDeg === null ? 1 : -envelope.angleHeadroomDeg}
            bad={0}
          />
          <Metric
            testId="envelope-min-gap"
            label="min gap at this angle"
            value={envelope.minGapCm === null ? 'none' : cm(envelope.minGapCm)}
            raw={envelope.minGapCm === null ? 1 : 0}
            bad={1}
          />
          <Metric
            testId="envelope-gap-headroom"
            label="gap headroom"
            value={envelope.gapHeadroomCm === null ? '—' : cm(envelope.gapHeadroomCm)}
            raw={envelope.gapHeadroomCm === null ? 1 : -envelope.gapHeadroomCm}
            bad={0}
          />
        </div>
        <p className="report-detail">
          {envelope.angleAtRangeLimit
            ? `every angle in the schema range is clean at a ${cm(envelope.gapCm, 1)} gap — the connector is not what stops you.`
            : envelope.maxAngleDeg === null
              ? 'this design is outside the envelope at every angle, so the angle is not what is wrong with it.'
              : `at ${cm(envelope.gapCm, 1)} the connector takes ${deg(envelope.maxAngleDeg)} of fold; you are asking for ${deg(envelope.angleDeg, 1)}.`}{' '}
          Found by bisection on the FLAGS, in {envelope.bisectionSteps} fixed steps — the same
          function that judges the real design, so the boundary and the verdict cannot disagree.
        </p>
        {envelope.limitedBy.length > 0 && (
          <ul className="report-list" data-testid="envelope-limited-by">
            {envelope.limitedBy.map((f) => (
              <li key={f}>
                <code>{f}</code> — {FLAG_TEXT[f] ?? 'no plain-language text for this code'}
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* --- the front bar, on its own, because it is a different KIND of limit */}
      <div
        className={`report-section${bar.clears ? '' : ' report-section-bad'}`}
        data-testid="report-front-bar"
      >
        <h4 className="report-section-title">the front bar — a limit on the part</h4>
        <div className="report-metric-grid">
          <Metric testId="frontbar-limit" label="bar clears to" value={deg(bar.concaveLimitDeg)} />
          <Metric
            testId="frontbar-worst"
            label="worst valley"
            value={deg(bar.worstConcaveDeg, 1)}
            raw={bar.clears ? 0 : 1}
            bad={1}
          />
        </div>
        <p className="report-detail">
          {bar.clears
            ? `a flat bar sized for this gap lies across every valley here, with ${deg(bar.headroomDeg)} to spare.`
            : `a flat bar sized for this gap bites the bezels ${deg(-bar.headroomDeg)} into the deepest valley.`}{' '}
          It measures the same ~12.37° at every gap from 1 to 4cm, and that constancy is the tell:
          the overhang that collides is the bar's own lip, so <b>widening the gap does not buy a
          degree</b>. This is a limit on the PART — a relief or a chamfer on the underside, or a
          narrower bar on concave stations, moves it — which is exactly why it is kept out of the
          connector's envelope above.
        </p>
      </div>

      {/* --- the joints ------------------------------------------------------ */}
      <div className="report-section" data-testid="report-joints">
        <h4 className="report-section-title">
          joints — {joints.length} total, {flaggedJoints} flagged
        </h4>
        {joints.length === 0 ? (
          <p className="report-detail">no joints — a joint needs two present neighbours</p>
        ) : (
          <div className="metrics-scroll">
            <table className="metrics-table" data-testid="joints-table">
              <thead>
                <tr>
                  <th>joint</th>
                  <th>units</th>
                  <th>span</th>
                  <th>fold</th>
                  <th>sense</th>
                  <th>parts</th>
                  <th>flags</th>
                </tr>
              </thead>
              <tbody>
                {joints.map((j) => (
                  <tr
                    key={j.id}
                    className={[
                      j.flipMismatch ? 'metrics-row-bad' : '',
                      hoveredUnitId === j.a || hoveredUnitId === j.b ? 'metrics-row-hover' : '',
                    ]
                      .filter(Boolean)
                      .join(' ')}
                  >
                    <td>{j.jointIndex + 1}</td>
                    <td>
                      {j.unitA}–{j.unitB}
                    </td>
                    <td>{j.spanCm.toFixed(2)}</td>
                    <td>
                      {j.foldDeg > 0 ? '+' : ''}
                      {j.foldDeg.toFixed(1)}
                    </td>
                    <td className={j.sense === 'convex' ? 'joint-convex' : j.sense === 'concave' ? 'joint-concave' : undefined}>
                      {j.sense}
                    </td>
                    <td>{j.stationCount}</td>
                    {/* Hover carries the sentences; the cell carries the count,
                        split hard / advisory so a joint with nothing but a
                        power-supply note does not read as a problem. */}
                    <td className="joint-flags" title={j.flags.map((f) => FLAG_TEXT[f] ?? f).join('\n')}>
                      {j.flags.length === 0
                        ? '—'
                        : `${j.flags.filter((f) => !ADVISORY_FLAGS.has(f)).length} / ${j.flags.length}`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <p className="form-hint">
          span is cm and is <b>exactly the gap at every joint</b> — the bisector construction makes
          the two rims parallel, so there is nothing to deviate. <span className="joint-convex">convex</span>{' '}
          is a ridge (the housings pinch, which is the direction that closes on the connector);{' '}
          <span className="joint-concave">concave</span> is a valley, and those are the free ones as
          far as the connector is concerned — and the ones the front bar cannot lie across. The
          flags column reads <b>faults / all</b>; hover it for what each one says.
        </p>
      </div>

      {/* --- the printed kit -------------------------------------------------- */}
      <div
        className={`report-section${kit.summary.clashes > 0 || kit.summary.infeasible > 0 ? ' report-section-bad' : ''}`}
        data-testid="report-kit"
      >
        <h4 className="report-section-title">the printed kit</h4>
        <div className="report-metric-grid">
          <Metric testId="kit-count" label="parts" value={`${kit.summary.count}`} />
          <Metric testId="kit-types" label="unique types" value={`${kit.summary.partTypes}`} />
          {/* `hardFlagged`, not `flagged`: in 'relief' mode every station of every
              powered joint carries W_BEARS_ON_POWER_SUPPLY at every angle, so
              "16 / 16 flagged" would be an alarm about a design with nothing
              wrong with it. The advisory count is stated below instead. */}
          <Metric
            testId="kit-flagged"
            label="flagged"
            value={`${kit.summary.hardFlagged} / ${kit.summary.count}`}
            raw={kit.summary.hardFlagged}
            warn={1}
          />
          <Metric
            testId="kit-clashes"
            label="part clashes"
            value={`${kit.summary.clashes}`}
            raw={kit.summary.clashes}
            bad={1}
          />
        </div>
        <p className="report-detail">
          {kit.summary.backHalfTypes} back-half type{kit.summary.backHalfTypes === 1 ? '' : 's'} and{' '}
          {kit.summary.frontBarTypes} bar width{kit.summary.frontBarTypes === 1 ? '' : 's'} · bins{' '}
          {kit.summary.binSpanCm}cm / {kit.summary.binAngleDeg}° · worst forced fit{' '}
          {cm(kit.summary.worstBinSpanErrorCm)} and {deg(kit.summary.worstBinFoldErrorDeg, 1)}
        </p>
        {kit.summary.flagged > kit.summary.hardFlagged && (
          <p className="report-detail" data-testid="kit-advisory">
            {kit.summary.flagged - kit.summary.hardFlagged} part
            {kit.summary.flagged - kit.summary.hardFlagged === 1 ? '' : 's'} bear on a panel's power
            supply rather than on its frame — a note about what the lip rests on, not a fault, and it
            fires at every angle in <code>relief</code> mode.
          </p>
        )}
        <ul className="report-list" data-testid="kit-list">
          {kit.kit.slice(0, 8).map((part) => (
            <li key={part.partId}>
              <code>{part.partId}</code> ×{part.count} — gap{' '}
              {part.spanStartCm === part.spanEndCm
                ? cm(part.spanStartCm, 1)
                : `${part.spanStartCm.toFixed(1)}→${part.spanEndCm.toFixed(1)}cm`}
              , fold {deg(part.foldDeg, 1)}
            </li>
          ))}
          {kit.kit.length > 8 && <li>… and {kit.kit.length - 8} more types</li>}
        </ul>
      </div>

      {/* --- collisions — a hard buildability failure -------------------------- */}
      <div
        className={`report-section${collisions.length > 0 ? ' report-section-bad' : ''}`}
        data-testid="report-collisions"
      >
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
        <p className="form-hint">
          non-adjacent panels only. Two neighbours meeting across their own joint is the joint's
          business, and <code>sectionFouling</code> judges that exactly where a box pair cannot.
        </p>
      </div>

      {/* --- warnings, grouped by code ---------------------------------------- */}
      {grouped.size > 0 && (
        <div className="msg-list msg-warnings" data-testid="report-warnings">
          {[...grouped.entries()].map(([code, list]) => (
            <div key={code} className="warning-group">
              <p>
                <code>{code}</code>
                {list.length > 1 && <b> ×{list.length}</b>} — {FLAG_TEXT[code] ?? 'no plain-language text for this code'}
              </p>
              <p className="warning-detail">{list[0].message}</p>
              {list.length > 1 && (
                <p className="warning-detail">
                  also at{' '}
                  {list
                    .slice(1)
                    .map((w) => (w.joint !== undefined ? `joint ${w.joint + 1}` : (w.unit ?? '?')))
                    .join(', ')}
                </p>
              )}
            </div>
          ))}
        </div>
      )}
    </section>
  )
}
