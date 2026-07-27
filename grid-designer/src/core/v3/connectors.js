/**
 * grid-designer v3 — where the 3D-printed connectors go, and what each one has
 * to be shaped like.
 *
 * HEADLESS ZONE (src/core/): pure functions, importable from plain node.
 *   - explicit `.js` extensions on ALL relative imports
 *   - may import `three` math classes only; never components / store / DOM
 *   - same layout in → byte-identical output out
 *
 * =============================================================================
 * WHAT THIS IS FOR
 * =============================================================================
 * report.js measures what the connectors must absorb. This module is the next
 * step: it decides WHERE the parts sit and hands each one the local geometry it
 * has to be built to. It produces no solid — that is connectorGeometry.js (P10)
 * — only stations and their frames.
 *
 * There is NO SUBSTRUCTURE. These parts are the structure, which is why station
 * COUNT is a first-class knob rather than a detail: one part on a joint is a
 * hinge, two make it rigid.
 *
 * =============================================================================
 * WHY SHORT PARTS, CENTRED IN THE GAP
 * =============================================================================
 * v1 (3dprintFiles/) used ONE part everywhere: a Y-junction edge clamp holding a
 * fixed 62° dihedral, sitting on the assembly's OUTSIDE edges. It could be one
 * part because every v1 joint was the same angle.
 *
 * v3's joints are all different, and worse, a single joint is different ALONG
 * ITS OWN LENGTH. Measured across the six presets:
 *
 *     station dihedral        0 – 34.8°
 *     station skew            0 – 11.5°
 *     rim-to-rim span      0.60 – 15.82 cm
 *     span swing along ONE whole joint          up to 12.77 cm
 *     span swing inside ONE 10 cm window   0.02 – 0.15 cm mean, 2.12 cm worst
 *
 * That last pair is the whole argument. A joint that wedges from 3 cm to 16 cm
 * end to end cannot be held by any one rigid part, but every 10 cm slice of it
 * is very nearly a constant-span, constant-angle problem. Short local parts
 * convert one intractable joint into a handful of easy ones — the same move as
 * HANDOFF §2.1 (stop asking a rigid thing to be a curved thing), applied to the
 * hardware instead of the surface.
 *
 * The consequence of "centred" rather than "on the outside edges" is physical
 * and belongs in the part design: a connector at mid-edge cannot be SLID onto
 * the rim from an open end, so it has to snap over it.
 *
 * =============================================================================
 * WHAT A STATION CARRIES
 * =============================================================================
 * Everything the part generator needs, and nothing derived from a scene:
 *
 *   spanCm       rim-to-rim distance at the station centre, on the LIT FACE.
 *                Front-mounted parts clip that rim, so this is the dimension the
 *                spine has to be. NOT the same as the nominal `gap`: it is what
 *                the joint actually opened to.
 *   dihedralDeg  fold between the two panels' lit normals — the arms' angle.
 *   twistDeg     angle between the two rim LINES. Constant along a joint (both
 *                tiles are rigid and planar, so each rim direction is fixed), so
 *                it equals report.js's per-joint `skewDeg` — recorded per
 *                station anyway because the part is per station.
 *   aFrame/bFrame  per side: the rim point, the unit vector the rim runs along,
 *                the panel's lit normal, and `inward` — which way the panel
 *                material lies from its own rim, i.e. which way the channel has
 *                to reach.
 *
 * `spanMinCm`/`spanMaxCm` bracket the span over the part's own footprint rather
 * than over the whole joint. A rigid part has to swallow that difference, and it
 * is the number that says whether `lengthCm` is short enough.
 */

import * as THREE from 'three'
import { normalizeConfig } from './schema.js'
import { solveLayout, jointEdgePoint, jointEdgeInward, jointEdgeRun } from './placement.js'
import { PANEL_PROFILE, POWER_SUPPLY, poweredEdgeBlockedSpan } from '../../config.js'

const DEG = 180 / Math.PI

/** Samples across one part's footprint when bracketing its local span. */
export const SPAN_SAMPLES = 5

// =============================================================================
// THE PART'S CROSS-SECTION — A RIM CLAMP
// =============================================================================
/**
 * How the connector grips a panel.
 *
 * REBUILT against the measured section (`updatedPanelGeo/`, read through
 * config.js). The previous grip was a wedge hook that engaged a taper starting
 * at the outer wall — a feature the real panel does not have. The real rim is:
 *
 *     bezel        1.5cm, chamfered, rising inboard to the front-most plane
 *     outer wall   1.10cm, vertical
 *     FLANGE       3.0cm, essentially flat, open air behind it
 *     taper        1.62cm at ~60deg, down to the back plate
 *
 * So the part is now a C that CLAMPS THE RIM: a short lip over the bezel, the
 * full outer wall, and a long lip over the flange. The flange is the load-
 * bearing half — 3cm of flat material to bear against — and the bezel lip only
 * has to stop the clamp rotating off. That is also why the part sits mostly on
 * the BACK: the front lip covers at most `frontGripCm` of a bezel that is
 * already a frame, and nothing covers the diffuser.
 *
 * NOTHING HERE RESTATES A PANEL DIMENSION. The outline is traced off
 * `PANEL_PROFILE` every time, so refining the panel moves the grip with it —
 * which is the whole point of the profile being parametric (config.js).
 *
 *   frontGripCm  lip over the bezel. Must stay under `bezelWidth` or the clamp
 *                overhangs the diffuser.
 *   backGripCm   lip over the flange — the real grip. Must stay under
 *                `flangeWidth` or it fouls the taper.
 *   jawCm        clamp wall thickness, and the spine's thickness with it.
 *   clearanceCm  slop between the clamp's inner face and the panel. 0 models the
 *                nominal fit; print tolerance is the printer's call.
 *
 * STILL OPEN: whether a rim clamp is the right part at all. This is a faithful
 * port of "grip the panel" onto the corrected geometry, not a redesign — see
 * HANDOFF.
 */
export const CONNECTOR_PROFILE = {
  frontGripCm: 1.0,
  backGripCm: 2.4,
  jawCm: 0.4,
  clearanceCm: 0,
}

/**
 * The panel's own rim surface, as a function of how far inboard you are.
 * Both are read straight off PANEL_PROFILE — see the note above about not
 * restating panel dimensions.
 */
/** Depth of the bezel surface `i` cm inboard (0 at the edge → 0 at the peak). */
export function bezelDepthAt(i, profile = PANEL_PROFILE) {
  const p = { ...PANEL_PROFILE, ...profile }
  return p.bezelDrop * (1 - Math.min(i, p.bezelWidth) / p.bezelWidth)
}

/** Depth of the flange surface `i` cm inboard of the edge. */
export function flangeDepthAt(i, profile = PANEL_PROFILE) {
  const p = { ...PANEL_PROFILE, ...profile }
  return p.outerWallDepth + p.flangeDrop * (Math.min(i, p.flangeWidth) / p.flangeWidth)
}

/**
 * The part's cross-section, as a simple closed polygon in the plane
 * perpendicular to the joint.
 *
 * Coordinates are `(p, q)` about the midpoint of the two rims: `p` runs from
 * panel A's rim toward panel B's, `q` is "up" (roughly the average lit normal).
 * Panel A's rim sits at `p = -span/2`, panel B's at `+span/2`, and each panel's
 * face tilts away from the p-axis by half the fold — so a positive `foldDeg`
 * (convex, a ridge) has both faces falling away and the lit faces diverging.
 *
 * The polygon is traversed counter-clockwise:
 *
 *      ┌────────────────────────────────────────────┐   ← one continuous strap:
 *      │  jaw A   ╎        spine        ╎   jaw B   │     both upper jaws + spine
 *      ├──────┐   ╎                     ╎   ┌───────┤
 *      │ slot │←── rim A inserts here   ╎   │ slot  │
 *      ├──────┘   ╎                     ╎   └───────┤
 *      │ hook A   ╎                     ╎   hook B  │
 *      └──────────┘                     └───────────┘
 *
 * The two notches are the slots. Nothing spans the gap on the underside — the
 * spine is the top band only, so the part never reaches into the space behind
 * the panels, which is the space that CLOSES on a convex joint.
 *
 * @param {object} opts
 * @param {number} opts.spanCm rim-to-rim distance at this cross-section
 * @param {number} opts.foldDeg signed dihedral; positive is convex (a ridge)
 * @param {object} [opts.profile] overrides for CONNECTOR_PROFILE
 * @returns {{ points: Array<[number, number]>, extents: object }}
 */
export function connectorProfile({ spanCm, foldDeg, profile = CONNECTOR_PROFILE, panel = PANEL_PROFILE }) {
  const c = { ...CONNECTOR_PROFILE, ...profile }
  const pp = { ...PANEL_PROFILE, ...panel }
  const phi = (foldDeg * Math.PI) / 180 / 2
  const cs = Math.cos(phi)
  const sn = Math.sin(phi)
  const j = c.jawCm
  const cl = c.clearanceCm

  // Per side: the rim datum, the inward direction, and "deeper" (away from the
  // lit side). The two sides are mirror images across p = 0. The rim datum is
  // (inboard 0, depth 0) — the front-most plane at the panel's outer edge,
  // which is exactly the point placement.js's jointEdgePoint returns.
  const sides = [
    { rim: [-spanCm / 2, 0], inward: [-cs, -sn], deeper: [sn, -cs] },  // A, -p side
    { rim: [spanCm / 2, 0], inward: [cs, -sn], deeper: [-sn, -cs] },   // B, +p side
  ]
  const at = (side, i, d) => [
    side.rim[0] + i * side.inward[0] + d * side.deeper[0],
    side.rim[1] + i * side.inward[1] + d * side.deeper[1],
  ]

  // One clamp, traced as a C opening INBOARD: down the bezel lip's inner face,
  // around the outer wall, out along the flange lip, then back along the
  // outside. Every inner-face point sits on the panel's own rim surface (offset
  // by the clearance), so the clamp cannot bite into the panel by construction.
  const clamp = (side) => {
    const fg = Math.min(c.frontGripCm, pp.bezelWidth)
    const bg = Math.min(c.backGripCm, pp.flangeWidth)
    return [
      at(side, fg, bezelDepthAt(fg, pp) - cl),                 // front lip, inner tip
      at(side, 0, pp.bezelDrop - cl),                          // front outer corner
      at(side, 0, pp.outerWallDepth + cl),                     // back outer corner
      at(side, bg, flangeDepthAt(bg, pp) + cl),                // flange lip, inner tip
      at(side, bg, flangeDepthAt(bg, pp) + cl + j),            // ...its thickness
      at(side, 0, pp.outerWallDepth + cl + j),                 // spine, back face
      at(side, 0, pp.bezelDrop - cl - j),                      // spine, front face
      at(side, fg, bezelDepthAt(fg, pp) - cl - j),             // front lip, outer face
    ]
  }

  const A = clamp(sides[0])
  const B = clamp(sides[1])

  // A's frame mirrors B's, so traversing both in the same LOCAL order would wind
  // them oppositely; B is therefore reversed. The SPINE appears implicitly, as
  // the two hops across the gap at indices 5 and 6 — it fills the gap over the
  // whole depth of the outer wall plus both jaw thicknesses, which is a far
  // stiffer section than a strap across the front.
  //
  // Its faces sit at the panels' own edges (inset 0), NOT proud of them. One
  // shared piece of material bridges the gap; giving each clamp its own outboard
  // wall instead invented a minimum gap of twice the wall thickness, which shut
  // out every joint under 0.8cm for no physical reason.
  const points = [
    A[0], A[1], A[2], A[3], A[4], A[5],
    B[5], B[4], B[3], B[2], B[1], B[0], B[7], B[6],
    A[6], A[7],
  ]

  let pMin = Infinity
  let pMax = -Infinity
  let qMin = Infinity
  let qMax = -Infinity
  for (const [p, q] of points) {
    if (p < pMin) pMin = p
    if (p > pMax) pMax = p
    if (q < qMin) qMin = q
    if (q > qMax) qMax = q
  }

  return { points, extents: { pMin, pMax, qMin, qMax, width: pMax - pMin, height: qMax - qMin } }
}

/**
 * The two cross-sections a part is lofted between — its `r̂ = -length/2` end and
 * its `+length/2` end. They differ only in span: both tiles are rigid planes, so
 * the fold is constant along the joint and the rims' separation varies linearly.
 * That is also why "twist" needs no separate term in the solid — a twisted joint
 * IS one whose two ends have different spans.
 */
export function connectorEndProfiles(station, profile = CONNECTOR_PROFILE) {
  return {
    start: connectorProfile({ spanCm: station.spanStartCm, foldDeg: station.foldDeg, profile }),
    end: connectorProfile({ spanCm: station.spanEndCm, foldDeg: station.foldDeg, profile }),
  }
}

/**
 * A station's part as an oriented bounding box, in the shape collide.js
 * consumes — so the SAT that already checks panel-against-panel can check
 * part-against-panel and part-against-part without a second implementation.
 *
 * The box encloses BOTH end profiles. Its local axes are the station frame
 * (p̂, q̂, r̂), which is right-handed by construction in `solveConnectors`; a
 * left-handed basis here would corrupt every overlap test exactly as it did for
 * plates in HANDOFF §6.
 */
export function connectorOBB(station, profile = CONNECTOR_PROFILE) {
  const { start, end } = connectorEndProfiles(station, profile)
  const pMin = Math.min(start.extents.pMin, end.extents.pMin)
  const pMax = Math.max(start.extents.pMax, end.extents.pMax)
  const qMin = Math.min(start.extents.qMin, end.extents.qMin)
  const qMax = Math.max(start.extents.qMax, end.extents.qMax)

  const p = new THREE.Vector3(...station.frame.p)
  const q = new THREE.Vector3(...station.frame.q)
  const rr = new THREE.Vector3(...station.frame.r)
  const centre = new THREE.Vector3(...station.mid)
    .addScaledVector(p, (pMin + pMax) / 2)
    .addScaledVector(q, (qMin + qMax) / 2)

  const m = new THREE.Matrix4().makeBasis(p, q, rr)
  const quat = new THREE.Quaternion().setFromRotationMatrix(m)

  return {
    center: rv(centre),
    halfExtents: [r((pMax - pMin) / 2), r((qMax - qMin) / 2), r(station.lengthCm / 2)],
    quaternion: [r(quat.x), r(quat.y), r(quat.z), r(quat.w)],
  }
}

/** Signed area of a closed polygon; positive means counter-clockwise. */
export function polygonArea(points) {
  let a = 0
  for (let k = 0; k < points.length; k++) {
    const [x0, y0] = points[k]
    const [x1, y1] = points[(k + 1) % points.length]
    a += x0 * y1 - x1 * y0
  }
  return a / 2
}

const sideOf = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

/**
 * Does a closed polygon cross itself? Non-adjacent edge pairs only, proper
 * crossings only — touching endpoints are legal.
 *
 * This is the part's HARD feasibility gate, and it is a physical statement
 * rather than a modelling nicety: the profile self-intersects when the two
 * hooks, swinging under the joint as it folds, run into each other. Measured
 * boundary (see tests) —
 *
 *     span 0.4cm → ±14° convex   span 1.0cm → ±36°   span ≥2.5cm → anything
 *
 * — which is the same fact the housings already report, seen from the connector
 * side: a convex joint closes BEHIND the panels, and a narrow gap leaves the
 * hardware nowhere to be. Deriving the gate from the geometry rather than
 * tabulating it means editing CONNECTOR_PROFILE moves the boundary honestly.
 */
export function profileSelfIntersects(points) {
  const n = points.length
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (j === i || (j + 1) % n === i || (i + 1) % n === j) continue
      const p1 = points[i]
      const p2 = points[(i + 1) % n]
      const p3 = points[j]
      const p4 = points[(j + 1) % n]
      const d1 = sideOf(p3, p4, p1)
      const d2 = sideOf(p3, p4, p2)
      const d3 = sideOf(p1, p2, p3)
      const d4 = sideOf(p1, p2, p4)
      if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) {
        return true
      }
    }
  }
  return false
}

/**
 * Printability limits. Unlike the self-intersection gate above these are
 * JUDGEMENT, not geometry, so they live as data and are stated as such:
 *
 *   minSpanCm       below this there is no room to work the part onto the rims,
 *                   whatever the profile says.
 *   maxSpanCm       the spine is a `jawCm`-thick strap. Past roughly 8cm it is a
 *                   beam being asked to act like a strap and wants ribbing or a
 *                   thicker section — neither of which this first pass models.
 *   maxSpanSpreadCm how much a single part may wedge along its own length before
 *                   the loft stops being a mild taper. This is the number
 *                   `connectors.lengthCm` exists to keep small.
 *
 * Consumed by report.js (P11), which flags rather than refuses.
 */
export const CONNECTOR_LIMITS = {
  minSpanCm: 0.4,
  maxSpanCm: 8,
  maxSpanSpreadCm: 1.0,
}

/**
 * Everything wrong with one station's part that can be decided from the station
 * alone. Clash needs the rest of the assembly, so report.js adds that.
 *
 * Pure and exported rather than inlined into the report because
 * `W_CONNECTOR_INFEASIBLE` is not reachable from any real config — see the note
 * on it below — and a rule with no test is a rule that quietly stops working.
 *
 *   W_CONNECTOR_INFEASIBLE  the two hooks pass through each other. GEOMETRY, and
 *        the only hard one here. Never yet observed on a real design, and the
 *        reason is a real property rather than luck: where the surface folds
 *        hard the gap has already wedged open, and where the gap is tight the
 *        surface is nearly flat. The two ways to make a part impossible do not
 *        co-occur on a drift. Kept, and tested against a synthetic station,
 *        because that correlation is a property of these forms and not a law.
 *   W_CONNECTOR_PINCH   the gap is narrower than anything can be fitted into.
 *   W_CONNECTOR_SPAN    the spine is a beam being asked to act as a strap.
 *   W_CONNECTOR_TWIST   the part wedges too much along its own length; this is
 *        the one `connectors.lengthCm` directly controls.
 */
export function connectorStationFlags(station, limits = CONNECTOR_LIMITS) {
  const flags = []
  const { start, end } = connectorEndProfiles(station)
  if (profileSelfIntersects(start.points) || profileSelfIntersects(end.points)) {
    flags.push('W_CONNECTOR_INFEASIBLE')
  }
  if (station.spanMinCm < limits.minSpanCm) flags.push('W_CONNECTOR_PINCH')
  if (station.spanMaxCm > limits.maxSpanCm) flags.push('W_CONNECTOR_SPAN')
  if (station.spanSpreadCm > limits.maxSpanSpreadCm) flags.push('W_CONNECTOR_TWIST')
  return flags
}

/**
 * A joint too short to seat `count` parts at full length. The parts are still
 * placed — shortened to fit — because refusing to connect a joint is not an
 * option a structure-bearing part gets to take. Same contract as
 * `W_PLATE_OVERRIDE_MISFIT`: report the cost, do not veto.
 */
export const CROWDED_CODE = 'W_CONNECTOR_CROWDED'

function r(v) {
  const out = Math.round(v * 1e9) / 1e9
  return out === 0 ? 0 : out
}

const rv = (v) => [r(v.x), r(v.y), r(v.z)]

// =============================================================================
// THE POWER SUPPLY — an edge a connector cannot use
// =============================================================================
/**
 * Which of a tile's edges carries the power supply, as `{ axis, boundary }` in
 * the same terms an adjacency record uses: `axis` is the material direction the
 * edge RUNS along, `boundary` its coordinate on the other axis.
 *
 * The supply is on a 60cm edge on both panel types, so for a plate it is one of
 * the two short ENDS, never a long side. `policy` picks which end:
 *   'low'  the end at the tile's minimum coordinate (default)
 *   'high' the far end
 *   'none' no supply — for studying what the constraint costs
 *
 * A GLOBAL CONVENTION, not a per-tile choice. Which way each panel faces is a
 * real design freedom that nothing in the tool models yet; until it does, every
 * panel is assumed oriented the same way, and that assumption is visible here
 * rather than buried.
 */
export function poweredEdgeOf(tile, policy = 'low') {
  if (policy === 'none') return null
  // The short edges run ACROSS the tile's long axis. A square has no long axis;
  // by convention its powered edge runs along u, matching a v-axis plate.
  const alongU = !(tile.type === '2x4' && tile.axis === 'u')
  const lo = alongU ? tile.uv.v0 : tile.uv.u0
  const len = alongU ? tile.uv.vLen : tile.uv.uLen
  return { axis: alongU ? 'u' : 'v', boundary: policy === 'low' ? lo : lo + len }
}

const EDGE_EPS = 1e-6

/**
 * The intervals of a joint that a flange-gripping connector CANNOT use, because
 * one of the two panels has its power supply behind that stretch of rim.
 *
 * Returned in the joint's own run-parameter space (the same coordinates as
 * `edge.edge.from`/`to`), merged and clipped to the joint.
 */
export function blockedSpansOnJoint(edge, A, B, policy = 'low', supply = POWER_SUPPLY) {
  const out = []
  for (const [tile, isA] of [[A, true], [B, false]]) {
    const powered = poweredEdgeOf(tile, policy)
    if (!powered || powered.axis !== edge.axis) continue
    const myBoundary = isA ? edge.edge.a : edge.edge.b
    if (Math.abs(powered.boundary - myBoundary) > EDGE_EPS) continue
    // The supply is centred on ITS OWN edge, which is the full run-extent of the
    // tile along `edge.axis` — not the joint, which may be a partial overlap.
    const runLo = edge.axis === 'u' ? tile.uv.u0 : tile.uv.v0
    const runLen = edge.axis === 'u' ? tile.uv.uLen : tile.uv.vLen
    const [from, to] = poweredEdgeBlockedSpan(runLen, supply)
    out.push([runLo + from, runLo + to])
  }
  return mergeIntervals(out, edge.edge.from, edge.edge.to)
}

/** Merge overlapping intervals and clip them to [lo, hi]. */
function mergeIntervals(spans, lo, hi) {
  const clipped = spans
    .map(([a, b]) => [Math.max(a, lo), Math.min(b, hi)])
    .filter(([a, b]) => b - a > EDGE_EPS)
    .sort((x, y) => x[0] - y[0])
  const out = []
  for (const s of clipped) {
    const last = out[out.length - 1]
    if (last && s[0] <= last[1] + EDGE_EPS) last[1] = Math.max(last[1], s[1])
    else out.push([...s])
  }
  return out
}

/** The complement of `blocked` within [lo, hi] — where a part may actually go. */
export function clearSpans(blocked, lo, hi) {
  const out = []
  let cursor = lo
  for (const [a, b] of blocked) {
    if (a - cursor > EDGE_EPS) out.push([cursor, a])
    cursor = Math.max(cursor, b)
  }
  if (hi - cursor > EDGE_EPS) out.push([cursor, hi])
  return out
}

/** A joint whose usable rim is too short for even one part. */
export const BLOCKED_CODE = 'W_JOINT_BLOCKED_BY_POWER_SUPPLY'
/** A joint that lost parts to the power supply but still carries some. */
export const REDUCED_CODE = 'W_JOINT_REDUCED_BY_POWER_SUPPLY'

/**
 * How many parts a joint of `materialLength` gets.
 *
 * `spacingCm` is stated as the most unsupported joint allowed between parts, so
 * the count is a ceiling, not a rounding: at the default 50 cm a 60 cm edge gets
 * 2 and a 121 cm plate edge gets 3. `minPerJoint` is a floor beneath that, and
 * the reason it defaults to 2 is structural — see schema.js.
 */
export function stationCount(materialLength, { spacingCm, minPerJoint }) {
  const bySpacing = Math.ceil(materialLength / spacingCm - 1e-9)
  return Math.max(minPerJoint, bySpacing, 1)
}

/**
 * Place connectors on every joint of a solved layout.
 *
 * @param {object} config raw or normalized v3 config
 * @param {object} [layout] a layout from solveLayout; solved here if omitted
 * @returns {{ stations: Array, perJoint: Array, warnings: Array }}
 */
export function solveConnectors(config, layout = null) {
  const cfg = normalizeConfig(config)
  const L = layout ?? solveLayout(cfg)
  const byId = new Map(L.tiles.map((t) => [t.id, t]))
  const { lengthCm, spacingCm, minPerJoint, powerEdge } = cfg.connectors

  const stations = []
  const perJoint = []
  const warnings = []

  L.adjacency.forEach((edge, jointIndex) => {
    const A = byId.get(edge.a)
    const B = byId.get(edge.b)
    // Same guard as report.js: an unplaced tile has no rim to clip.
    if (!A?.position || !B?.position) return

    const span = edge.materialLength
    const wanted = stationCount(span, { spacingCm, minPerJoint })

    // --- where the power supply forbids a part ----------------------------
    // A connector grips the FLANGE, and on a powered edge the supply sits on
    // the flange for 50 of its 60cm. So the usable rim is only what is left
    // outside that, and stations are placed in those clear stretches rather
    // than evenly along a joint that cannot receive them.
    const blocked = blockedSpansOnJoint(edge, A, B, powerEdge)
    const clear = clearSpans(blocked, edge.edge.from, edge.edge.to)
    const clearLength = clear.reduce((n, [a, b]) => n + (b - a), 0)

    // Each clear stretch gets its own parts, sized by the same spacing rule.
    // A stretch shorter than one part gets none — it cannot hold one.
    const usable = clear.filter(([a, b]) => b - a >= lengthCm - 1e-9)
    const perStretch = usable.map(([a, b]) => Math.max(1, Math.ceil((b - a) / spacingCm - 1e-9)))
    // `minPerJoint` is a floor on the JOINT, so once spacing has had its say,
    // top up wherever there is still room for another part. Without this the
    // floor silently stopped applying as soon as a joint was split into
    // stretches by a power supply.
    let total = perStretch.reduce((n, k) => n + k, 0)
    // The floor is a FLOOR: parts are added until it is met even when that means
    // shortening them, and `W_CONNECTOR_CROWDED` reports the cost. Refusing
    // instead would silently drop a structural requirement — same contract as a
    // manual plate override, which is placed and reported rather than vetoed.
    while (total < minPerJoint && usable.length > 0) {
      let best = 0
      let bestRoom = -Infinity
      usable.forEach(([a, b], k) => {
        const room = (b - a) / (perStretch[k] + 1)
        if (room > bestRoom) { bestRoom = room; best = k }
      })
      perStretch[best]++
      total++
    }
    const count = total

    if (blocked.length > 0) {
      const code = count === 0 ? BLOCKED_CODE : count < wanted ? REDUCED_CODE : null
      if (code) {
        warnings.push({
          code,
          joint: jointIndex,
          a: edge.a,
          b: edge.b,
          message:
            count === 0
              ? `joint ${edge.a}–${edge.b} has a power supply behind it and only ${r(clearLength)}cm of ` +
                `usable rim in stretches too short for a ${lengthCm}cm part — it carries NO connector`
              : `joint ${edge.a}–${edge.b} has a power supply behind it: ${count} parts fit where ` +
                `${wanted} were wanted`,
          wanted,
          placed: count,
          clearLengthCm: r(clearLength),
          blockedCm: r(span - clearLength),
        })
      }
    }

    if (count === 0) {
      perJoint.push({
        jointIndex,
        a: edge.a,
        b: edge.b,
        materialLength: r(span),
        count: 0,
        lengthCm: 0,
        blockedCm: r(span - clearLength),
      })
      return
    }

    // Each part occupies `partLength` of its stretch, centred on its station.
    const room = Math.min(...usable.map(([a, b], k) => (b - a) / perStretch[k]))
    const partLength = Math.min(lengthCm, room)
    if (partLength < lengthCm - 1e-9) {
      warnings.push({
        code: CROWDED_CODE,
        joint: jointIndex,
        a: edge.a,
        b: edge.b,
        message:
          `joint ${edge.a}–${edge.b} takes ${count} parts in ${usable.length} usable stretch` +
          `${usable.length === 1 ? '' : 'es'} — they are shortened from ${lengthCm}cm to ${r(partLength)}cm to fit`,
        requestedLengthCm: lengthCm,
        placedLengthCm: r(partLength),
        count,
      })
    }

    // Station centres: within each usable stretch, evenly spaced and symmetric.
    const centres = []
    usable.forEach(([a, b], k) => {
      const n = perStretch[k]
      for (let m = 0; m < n; m++) centres.push(a + (b - a) * ((m + 0.5) / n))
    })
    centres.sort((x, y) => x - y)

    // Rim directions and normals are constant along a joint — both tiles are
    // rigid and planar — so they are computed once per joint, not per station.
    const runA = jointEdgeRun(A, edge)
    const runB = jointEdgeRun(B, edge)
    const nA = new THREE.Vector3(...A.normal)
    const nB = new THREE.Vector3(...B.normal)
    const inA = jointEdgeInward(A, edge, true)
    const inB = jointEdgeInward(B, edge, false)

    const twistDeg = Math.acos(Math.min(1, Math.max(-1, runA.dot(runB)))) * DEG
    const dihedralDeg = Math.acos(Math.min(1, Math.max(-1, nA.dot(nB)))) * DEG

    for (let k = 0; k < count; k++) {
      const s = centres[k]
      const pa = jointEdgePoint(A, edge, true, s)
      const pb = jointEdgePoint(B, edge, false, s)
      const spanCm = pa.distanceTo(pb)

      // --- the part's own frame ------------------------------------------
      // p̂ from rim A toward rim B, perpendicular to the joint; q̂ "up"; r̂ along
      // the joint, chosen so (p̂, q̂, r̂) is RIGHT-HANDED and q̂ agrees with the
      // panels' lit side. Deriving r̂ rather than taking it from a tile is what
      // makes the fold's SIGN meaningful — a frame picked up from whichever
      // tile happened to be `a` would flip sign with tile ordering.
      let rHat = runA.clone()
      let pHat = pb.clone().sub(pa)
      pHat.addScaledVector(rHat, -pHat.dot(rHat)).normalize()
      let qHat = rHat.clone().cross(pHat)
      if (qHat.dot(nA.clone().add(nB)) < 0) {
        rHat.negate()
        qHat = rHat.clone().cross(pHat)
      }

      // Signed fold about r̂. Convex — a ridge, lit faces diverging, housings
      // pinching — comes out POSITIVE, which is the reading the drift wants.
      const cross = nA.clone().cross(nB)
      const foldDeg = -Math.atan2(cross.dot(rHat), nA.dot(nB)) * DEG

      // Bracket the span over THIS PART's footprint, not the whole joint: a
      // rigid part only has to swallow the variation it actually straddles.
      // start/end are ordered along r̂, so the lofted solid is built the same
      // way round as it is placed.
      const along = rHat.dot(runA) >= 0 ? 1 : -1
      let spanMin = Infinity
      let spanMax = -Infinity
      const spanAt = (sm) => jointEdgePoint(A, edge, true, sm).distanceTo(jointEdgePoint(B, edge, false, sm))
      for (let m = 0; m < SPAN_SAMPLES; m++) {
        const f = (m / (SPAN_SAMPLES - 1)) - 0.5
        const d = spanAt(s + f * partLength)
        if (d < spanMin) spanMin = d
        if (d > spanMax) spanMax = d
      }
      const spanStart = spanAt(s - along * partLength / 2)
      const spanEnd = spanAt(s + along * partLength / 2)

      stations.push({
        id: `J${jointIndex}S${k}`,
        jointIndex,
        a: edge.a,
        b: edge.b,
        axis: edge.axis,
        index: k,
        of: count,
        s: r(s),
        tAlong: r((k + 0.5) / count),
        lengthCm: r(partLength),
        spanCm: r(spanCm),
        spanMinCm: r(spanMin),
        spanMaxCm: r(spanMax),
        spanSpreadCm: r(spanMax - spanMin),
        // The part's two ends, ordered along r̂ — what the lofted solid is built
        // from. spanMin/Max are the unordered bracket; these are the signed pair.
        spanStartCm: r(spanStart),
        spanEndCm: r(spanEnd),
        dihedralDeg: r(dihedralDeg),
        // Signed: positive is convex (a ridge). dihedralDeg is its magnitude,
        // kept because report.js has always reported the unsigned angle.
        foldDeg: r(foldDeg),
        twistDeg: r(twistDeg),
        // Midpoint of the two rims: where the part's spine sits.
        mid: rv(pa.clone().add(pb).multiplyScalar(0.5)),
        frame: { p: rv(pHat), q: rv(qHat), r: rv(rHat) },
        aFrame: { point: rv(pa), run: rv(runA), normal: rv(nA), inward: rv(inA) },
        bFrame: { point: rv(pb), run: rv(runB), normal: rv(nB), inward: rv(inB) },
      })
    }

    perJoint.push({
      jointIndex,
      a: edge.a,
      b: edge.b,
      materialLength: r(span),
      count,
      lengthCm: r(partLength),
      blockedCm: r(span - clearLength),
    })
  })

  return { stations, perJoint, warnings }
}
