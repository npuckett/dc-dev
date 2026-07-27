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

const DEG = 180 / Math.PI

/** Samples across one part's footprint when bracketing its local span. */
export const SPAN_SAMPLES = 5

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

      // Bracket the span over THIS PART's footprint, not the whole joint: a
      // rigid part only has to swallow the variation it actually straddles.
      let spanMin = Infinity
      let spanMax = -Infinity
      for (let m = 0; m < SPAN_SAMPLES; m++) {
        const f = (m / (SPAN_SAMPLES - 1)) - 0.5
        const sm = s + f * partLength
        const d = jointEdgePoint(A, edge, true, sm).distanceTo(jointEdgePoint(B, edge, false, sm))
        if (d < spanMin) spanMin = d
        if (d > spanMax) spanMax = d
      }

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
        dihedralDeg: r(dihedralDeg),
        twistDeg: r(twistDeg),
        // Midpoint of the two rims: where the part's spine sits.
        mid: rv(pa.clone().add(pb).multiplyScalar(0.5)),
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
