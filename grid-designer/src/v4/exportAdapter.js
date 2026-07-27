/**
 * grid-designer v4 — the adapter between v4's core and everything that was
 * written against v3's report shape.
 *
 * =============================================================================
 * WHY THIS FILE EXISTS AT ALL
 * =============================================================================
 * `src/core/v4/connectors.js` emits stations that are FIELD-FOR-FIELD v3's — same
 * keys, same frame, same meanings — but it deliberately stops there. It places
 * stations; it does not judge them, does not bin them into printable types, and
 * does not size a front bar. All of that is v3's part machinery, which v4 imports
 * unchanged (V4_SPEC §0), and the pieces of it that v3's `buildReport` used to
 * assemble have no home in `core/v4/report.js` — that module's job is the
 * ENVELOPE, and adding a print queue to it would have widened the frozen core to
 * carry something only the UI and the exporters read.
 *
 * So the assembly happens here, in the UI layer, and this header is the record of
 * exactly what the core does not expose:
 *
 *   flags per station     `connectorStationFlags` — v4's report merges them per
 *                         JOINT, which is the right unit for a verdict but loses
 *                         which station carries what. The viewport colours parts
 *                         individually, so it needs the per-station answer.
 *   the front bar's width `frontBarWidthFor` — v4 never assigns one (report.js's
 *                         header explains why: `sectionFouling` only tests the bar
 *                         when a station carries a `barWidthCm`, and v4 asks that
 *                         question separately, per gap, in `envelope.frontBar`).
 *                         A bar still has to be drawn and printed.
 *   the kit               v3's `buildKit`, reproduced here verbatim in effect —
 *                         the binning rule is a PRINT-QUEUE policy, not geometry,
 *                         and v4's core has no opinion about it.
 *   part clashes          v4's report checks panel-on-panel collisions and the
 *                         joints' own fouling; it does not check whether two
 *                         printed parts occupy the same space. Computed here from
 *                         `connectorOBB` + `collide.js` rather than left out, so
 *                         the manifest's `clashes: 0` is a measurement instead of
 *                         a silence.
 *
 * =============================================================================
 * ONE KNOWN COSMETIC GAP IN THE MANIFEST
 * =============================================================================
 * `src/utils/connectorExport.js` is shared with the (unmounted but intact) v3 UI
 * and was not in this package's scope to change, so the manifest it writes still
 * labels itself `grid-designer v3` and still has a `design.sheet` slot that a v4
 * config has nothing to put in (v4 has strips, not a sheet). The PARTS and their
 * quantities — the whole point of the document — are correct.
 */

import {
  CONNECTOR_LIMITS,
  connectorOBB,
  connectorStationFlags,
  frontBarWidthFor,
} from '../core/v3/connectors.js'
import { aabbOverlap, findCollisions, obbPenetration } from '../core/v3/collide.js'

/** Ignore contact shallower than this when calling something a part clash —
 *  the same threshold v3's report used, for the same reason: parts are meant to
 *  sit close and a zero-tolerance test flags floating-point noise. */
export const CONNECTOR_CLASH_MIN_DEPTH_CM = 0.05

/** Flags that are a NOTE rather than a fault. `W_BEARS_ON_POWER_SUPPLY` fires on
 *  every station of every powered joint in 'relief' mode, at every angle — see
 *  core/v4/report.js's "WHAT CLEAN MEANS", which excludes it from the envelope on
 *  exactly this reasoning. Whether a driver housing is something to clamp against
 *  is a hardware question this model cannot answer. */
export const ADVISORY_FLAGS = new Set(['W_BEARS_ON_POWER_SUPPLY'])

const r = (v) => {
  const out = Math.round(v * 1e9) / 1e9
  return out === 0 ? 0 : out
}

// -----------------------------------------------------------------------------
// The kit — v3's binning rule, restated for v4 stations
// -----------------------------------------------------------------------------
/**
 * Group stations into printable part types.
 *
 * A part is described by `(spanStart, spanEnd, fold, length)`, and two stations
 * share a part when those round into the same bin. Two canonicalizations, both
 * because the physical part allows them: the section is mirror-symmetric about
 * the gap's centre line (so which panel is `a` does not distinguish two parts),
 * and a part fits either way round along the joint (so the span pair is ordered).
 * The FOLD is not canonicalized — convex and concave are genuinely different
 * parts, one closing the hooks and the other spreading them.
 *
 * On a v4 design this usually collapses very hard: every joint has the same span
 * by construction and only two folds (+θ and −θ), so a nine-unit ribbon needs two
 * back-half types where a drift needed dozens. That collapse is the pivot's
 * result showing up in the print queue, and it is worth being able to read.
 */
export function buildKitV4(stations, { binSpanCm, binAngleDeg }) {
  const groups = new Map()

  for (const st of stations) {
    const lo = Math.min(st.spanStartCm, st.spanEndCm)
    const hi = Math.max(st.spanStartCm, st.spanEndCm)
    const bLo = Math.round(lo / binSpanCm)
    const bHi = Math.round(hi / binSpanCm)
    const bFold = Math.round(st.foldDeg / binAngleDeg)
    const bLen = Math.round(st.lengthCm / 0.1)
    const key = `${bLo}|${bHi}|${bFold}|${bLen}`

    if (!groups.has(key)) {
      groups.set(key, {
        key,
        // The representative part is the BIN CENTRE, not the first station that
        // landed in it — otherwise the kit would depend on iteration order and
        // half the group could sit further from the part than the bin allows.
        spanStartCm: r(bLo * binSpanCm),
        spanEndCm: r(bHi * binSpanCm),
        foldDeg: r(bFold * binAngleDeg),
        lengthCm: r(bLen * 0.1),
        stations: [],
      })
    }
    groups.get(key).stations.push(st)
  }

  const ordered = [...groups.values()].sort(
    (a, b) => b.stations.length - a.stations.length || (a.key < b.key ? -1 : 1),
  )

  return ordered.map((g, idx) => {
    let worstSpan = 0
    let worstFold = 0
    for (const st of g.stations) {
      const lo = Math.min(st.spanStartCm, st.spanEndCm)
      const hi = Math.max(st.spanStartCm, st.spanEndCm)
      worstSpan = Math.max(worstSpan, Math.abs(lo - g.spanStartCm), Math.abs(hi - g.spanEndCm))
      worstFold = Math.max(worstFold, Math.abs(st.foldDeg - g.foldDeg))
    }
    return {
      partId: `P${String(idx).padStart(2, '0')}`,
      count: g.stations.length,
      spanStartCm: g.spanStartCm,
      spanEndCm: g.spanEndCm,
      foldDeg: g.foldDeg,
      lengthCm: g.lengthCm,
      worstSpanErrorCm: r(worstSpan),
      worstFoldErrorDeg: r(worstFold),
      stationIds: g.stations.map((s) => s.id),
      joints: [...new Set(g.stations.map((s) => s.jointIndex))].sort((a, b) => a - b),
    }
  })
}

// -----------------------------------------------------------------------------
// The whole printed-kit picture, from a v4 chain + stations
// -----------------------------------------------------------------------------
/**
 * Flags, bar widths, the kit, the bars and the clashes — everything the frozen
 * core leaves to the caller, in the shape v3's consumers already read.
 *
 * @param {object} config a normalized v4 config
 * @param {object} chain  `solveChain` output
 * @param {object} connectors `solveConnectorsV4` output
 * @returns {{stations, perJoint, kit, bars, clashes, summary, limits}}
 */
export function buildConnectorKit(config, chain, connectors) {
  const cfg = config.connectors
  const raw = connectors.stations

  // --- clash: a part against a panel it does NOT grip, or against another part
  const boxes = raw.map((st) => connectorOBB(st))
  const present = chain.panels.filter((u) => u.present)
  const panelIndex = new Map(present.map((u, i) => [u.id, i]))
  const clashes = []
  raw.forEach((st, i) => {
    for (let t = 0; t < present.length; t++) {
      if (t === panelIndex.get(st.a) || t === panelIndex.get(st.b)) continue
      if (!aabbOverlap(boxes[i], present[t].obb)) continue
      const pen = obbPenetration(boxes[i], present[t].obb)
      if (pen && pen.depthCm > CONNECTOR_CLASH_MIN_DEPTH_CM) {
        clashes.push({ station: st.id, against: present[t].id, kind: 'panel', depthCm: r(pen.depthCm) })
      }
    }
  })
  for (const hit of findCollisions(boxes, { minDepthCm: CONNECTOR_CLASH_MIN_DEPTH_CM })) {
    clashes.push({
      station: raw[hit.i].id,
      against: raw[hit.j].id,
      kind: 'connector',
      depthCm: r(hit.depthCm),
    })
  }
  clashes.sort((x, y) => y.depthCm - x.depthCm || (x.station < y.station ? -1 : 1))
  const clashed = new Set(clashes.map((c) => c.station))

  // --- the two part families ------------------------------------------------
  // Back halves bin on geometry: they sit on the flange at the joint's own fold.
  // Front bars do not — a bar is a plain rectangle bearing on two bezels, so one
  // width serves a whole band of gaps. Every station still gets its own bar, cut
  // to its own gap; the binning below is a REPORTING fact about how few distinct
  // widths the design needs, not a constraint the geometry has to meet.
  const kit = buildKitV4(raw, cfg)
  const barBin = cfg.binSpanCm
  const widthOf = (st) => r(Math.round(frontBarWidthFor(Math.max(st.spanStartCm, st.spanEndCm)) / barBin) * barBin)
  const barWidths = new Map()
  for (const st of raw) {
    const w = widthOf(st)
    if (!barWidths.has(w)) barWidths.set(w, { widthCm: w, stationIds: [] })
  }
  const bars = [...barWidths.values()]
    .sort((a, b) => a.widthCm - b.widthCm)
    .map((b, i) => ({ ...b, barId: `B${String(i).padStart(2, '0')}` }))
  const barByWidth = new Map(bars.map((b) => [b.widthCm, b]))

  const partOf = new Map()
  for (const p of kit) for (const sid of p.stationIds) partOf.set(sid, p.partId)

  const stations = raw.map((st) => {
    const flags = connectorStationFlags(st, CONNECTOR_LIMITS)
    if (clashed.has(st.id)) flags.push('W_CONNECTOR_CLASH')
    const bar = barByWidth.get(widthOf(st)) ?? null
    if (bar) bar.stationIds.push(st.id)
    return {
      ...st,
      flags,
      partId: partOf.get(st.id) ?? null,
      barId: bar?.barId ?? null,
      barWidthCm: bar?.widthCm ?? null,
    }
  })

  const flagged = stations.filter((st) => st.flags.length > 0).length
  // Split out, because on a powered joint in 'relief' mode EVERY station carries
  // `W_BEARS_ON_POWER_SUPPLY` at every angle — so a bare "16 / 16 flagged" reads
  // as an alarm about a design that has nothing wrong with it. It is a note about
  // what the lip rests on, and report.js excludes it from the envelope for the
  // same reason. `hardFlagged` is the count that means something changed.
  const hardFlagged = stations.filter((st) => st.flags.some((f) => !ADVISORY_FLAGS.has(f))).length
  const infeasible = stations.filter((st) => st.flags.includes('W_CONNECTOR_INFEASIBLE')).length
  const spans = stations.map((st) => st.spanCm)

  return {
    stations,
    perJoint: connectors.perJoint,
    kit,
    bars,
    clashes,
    limits: CONNECTOR_LIMITS,
    summary: {
      count: stations.length,
      jointCount: connectors.perJoint.length,
      // Both families: a design needs one of each back-half bin plus the handful
      // of bar widths, and that total is what actually goes on a printer.
      partTypes: kit.length + bars.length,
      backHalfTypes: kit.length,
      frontBarTypes: bars.length,
      flagged,
      hardFlagged,
      infeasible,
      clashes: clashes.length,
      lengthCm: config.connectors.lengthCm,
      binSpanCm: cfg.binSpanCm,
      binAngleDeg: cfg.binAngleDeg,
      spanCm: {
        min: spans.length ? r(Math.min(...spans)) : 0,
        max: spans.length ? r(Math.max(...spans)) : 0,
      },
      worstFoldDeg: stations.length ? r(Math.max(...stations.map((st) => Math.abs(st.foldDeg)))) : 0,
      worstSpanSpreadCm: stations.length ? r(Math.max(...stations.map((st) => st.spanSpreadCm))) : 0,
      worstBinSpanErrorCm: r(kit.length ? Math.max(...kit.map((k) => k.worstSpanErrorCm)) : 0),
      worstBinFoldErrorDeg: r(kit.length ? Math.max(...kit.map((k) => k.worstFoldErrorDeg)) : 0),
    },
  }
}

/**
 * `buildConnectorKit`, memoized on the STATIONS OBJECT IDENTITY — which the
 * store's `getDerived` already makes stable per config. Three consumers want the
 * same answer (the viewport colours parts by it, the report counts it, the
 * exporters print it), and it does all-pairs OBB work; computing it three times
 * per render would be three times more than necessary.
 */
const kitCache = new WeakMap()

export function getConnectorKit(config, chain, connectors) {
  let entry = kitCache.get(connectors)
  if (!entry) {
    entry = buildConnectorKit(config, chain, connectors)
    kitCache.set(connectors, entry)
  }
  return entry
}

// -----------------------------------------------------------------------------
// OBJ export
// -----------------------------------------------------------------------------
/**
 * Adapt a v4 chain to the shape `src/utils/exporters.js` expects.
 *
 * `buildExportGroup` only ever reads `layout.panels` and each panel's `.type` /
 * `.position` / `.quaternion` to bake a world transform. Two of those three a v4
 * unit already carries under exactly that name; the third it does not — a unit's
 * footprint is `panelType`, because `type` on a v4 unit would compete with
 * `role`, which is the more interesting thing a unit is a type OF. So it is
 * mapped, along with the NAMING fields `exportPanelName` reads (`.row` / `.col` /
 * `.rectOrientation`), and nothing in the exporter has to change: a unit's `unit`
 * number is its row along the chain and its `strip` is its column across the
 * wall, which gives `panel_r{unit}_c{strip}` — unique, and readable as the
 * position in the ribbon that it is.
 *
 * ABSENT UNITS ARE NOT EXPORTED. `present: false` means the panel is not there;
 * the chain still carries its record so removal stays non-destructive, but the
 * OBJ is a description of what gets built.
 *
 * @param {object} chain `solveChain` output
 * @param {Array} [stations] decorated stations; connectors are omitted if absent
 */
export function toExportableLayout(chain, stations = null) {
  const panels = chain.panels
    .filter((u) => u.present)
    .map((u) => ({
      ...u,
      type: u.panelType,
      row: u.unit,
      col: u.strip,
      rectOrientation: undefined,
    }))
  return { panels, connectors: stations ?? [] }
}
