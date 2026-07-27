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
import { PANEL_PROFILE, BODY_INSET } from '../../config.js'

const DEG = 180 / Math.PI

/** Samples across one part's footprint when bracketing its local span. */
export const SPAN_SAMPLES = 5

// =============================================================================
// THE PART'S CROSS-SECTION
// =============================================================================
/**
 * How the connector grips a panel, and why it is not v1's slot.
 *
 * v1 recorded a parallel-sided channel: 9.5mm wide, 8.5mm deep (its overall
 * 28.5mm thickness is exactly three of those 9.5mm — jaw, slot, jaw). A
 * parallel slot assumes the rim is a parallel-sided plate. It is not. Reading
 * the real profile out of config.js:
 *
 *     flange top face      y = 0, exposed inward for lipWidth = 2.5cm
 *     outer wall           y = 0 down to y = -outerThickness (1.0cm)
 *     BELOW THAT, A TAPER  y(i) = -1.0 - 0.675·i, receding inward to the body
 *
 * So the undercut a hook has to engage is a WEDGE that opens with depth — only
 * 1.7mm of it 2.5mm in, 5.7mm of it 8.5mm in. A parallel-sided jaw of any
 * useful reach bites straight into the taper. The lower jaw here is therefore
 * bounded above by the taper plane itself, which turns the grip into a wedge
 * engagement that tightens as the part is pushed on rather than a friction fit
 * that relies on interference.
 *
 * The consequence for assembly, and it is the direct price of "centred in the
 * gap" rather than "on the outside edges": a part at mid-edge cannot be slid on
 * from an open end. It goes on by hooking the lower wedge under the rim first
 * and rotating the upper jaw down onto the flange.
 *
 *   gripCm   how far the channel reaches in over the 2.5cm flange. v1's 8.5mm.
 *   wallCm   the back wall closing the C, inboard of the grip.
 *   jawCm    material above the flange. v1's 9.5mm, and also the spine's
 *            thickness — the upper jaws and the spine are one continuous strap
 *            across the joint, which is what makes the part stiff in the
 *            direction that matters.
 *   hookCm   material below the taper plane: the undercut hook.
 *   slotClearanceCm  slop on both slot faces. 0 models the nominal fit exactly;
 *            NEGATIVE is a press fit (v1's 9.5mm slot on a 10mm rim was
 *            -0.5mm). Left at 0 because print tolerance is the printer's call,
 *            not the model's.
 */
export const CONNECTOR_PROFILE = {
  gripCm: 0.85,
  wallCm: 0.3,
  jawCm: 0.95,
  hookCm: 0.6,
  slotClearanceCm: 0,
}

/** Depth of the taper below the rim plane, `i` cm inboard of the panel edge. */
export function taperDepthAt(i) {
  return ((PANEL_PROFILE.overallThickness - PANEL_PROFILE.outerThickness) / BODY_INSET) * i
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
export function connectorProfile({ spanCm, foldDeg, profile = CONNECTOR_PROFILE }) {
  const { gripCm, wallCm, jawCm, hookCm, slotClearanceCm } = { ...CONNECTOR_PROFILE, ...profile }
  const outer = PANEL_PROFILE.outerThickness
  const phi = (foldDeg * Math.PI) / 180 / 2
  const c = Math.cos(phi)
  const s = Math.sin(phi)
  const back = gripCm + wallCm

  // Per side: the rim point, the inward direction, and the lit normal, all in
  // (p, q). The two sides are mirror images across p = 0.
  const sides = [
    { rim: [-spanCm / 2, 0], inward: [-c, -s], up: [-s, c] },  // A, on the -p side
    { rim: [spanCm / 2, 0], inward: [c, -s], up: [s, c] },     // B, on the +p side
  ]

  // A local (i, n) point on one side, mapped into (p, q).
  const at = (side, i, n) => [
    side.rim[0] + i * side.inward[0] + n * side.up[0],
    side.rim[1] + i * side.inward[1] + n * side.up[1],
  ]

  // One channel's outline in its own (i, n), from the mouth's top corner around
  // the outside and back through the slot notch. `-taperDepthAt(i)` is the slot
  // floor: the hook's upper face IS the taper, which is the whole point.
  const cl = slotClearanceCm
  const channel = (side) => [
    at(side, 0, jawCm),                                  // mouth, top of the jaw
    at(side, back, jawCm),                               // inboard, top
    at(side, back, -outer - taperDepthAt(back) - hookCm), // inboard, bottom of the hook
    at(side, 0, -outer - hookCm),                        // mouth, bottom of the hook
    at(side, 0, -outer - cl),                            // mouth, slot floor
    at(side, gripCm, -outer - taperDepthAt(gripCm) - cl), // slot floor, inboard end
    at(side, gripCm, cl),                                // slot ceiling, inboard end
    at(side, 0, cl),                                     // mouth, slot ceiling
  ]

  const A = channel(sides[0])
  const B = channel(sides[1])

  // A's frame is a mirror of B's, so traversing both in the same LOCAL order
  // would wind them oppositely in (p, q). B is therefore reversed. The spine
  // appears implicitly, as the two straight hops across the gap: A's slot
  // ceiling → B's slot ceiling underneath, and B's jaw top → A's jaw top over.
  const points = [
    A[1], A[2], A[3], A[4], A[5], A[6], A[7],  // channel A, ending at its mouth ceiling
    B[7], B[6], B[5], B[4], B[3], B[2], B[1],  // spine underside, then channel B reversed
    B[0], A[0],                                // B's mouth top, spine top, A's mouth top
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
  const { lengthCm, spacingCm, minPerJoint } = cfg.connectors

  const stations = []
  const perJoint = []
  const warnings = []

  L.adjacency.forEach((edge, jointIndex) => {
    const A = byId.get(edge.a)
    const B = byId.get(edge.b)
    // Same guard as report.js: an unplaced tile has no rim to clip.
    if (!A?.position || !B?.position) return

    const span = edge.materialLength
    const count = stationCount(span, { spacingCm, minPerJoint })

    // Each part occupies `partLength` of the joint centred on its station. With
    // stations at (k + 0.5)/count the first centre sits at span/(2·count), so
    // "no part runs off the end and none overlaps its neighbour" is the single
    // condition partLength ≤ span/count.
    const room = span / count
    const partLength = Math.min(lengthCm, room)
    if (partLength < lengthCm - 1e-9) {
      warnings.push({
        code: CROWDED_CODE,
        joint: jointIndex,
        a: edge.a,
        b: edge.b,
        message:
          `joint ${edge.a}–${edge.b} is ${r(span)}cm and takes ${count} parts, leaving ${r(room)}cm ` +
          `each — they are shortened from ${lengthCm}cm to ${r(partLength)}cm to fit`,
        requestedLengthCm: lengthCm,
        placedLengthCm: r(partLength),
        count,
      })
    }

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
      const s = edge.edge.from + span * ((k + 0.5) / count)
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
    })
  })

  return { stations, perJoint, warnings }
}
