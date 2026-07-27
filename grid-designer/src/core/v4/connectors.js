/**
 * grid-designer v4 — where the connectors go on a folded network.
 *
 * HEADLESS ZONE (src/core/): pure functions, importable from plain node.
 *   - explicit `.js` extensions on ALL relative imports
 *   - may import `three` math classes only; never components / store / DOM
 *   - same lattice in → byte-identical output out
 *
 * =============================================================================
 * WHAT THIS IS, AND WHAT IT DELIBERATELY IS NOT
 * =============================================================================
 * This module places STATIONS. It does not design a part, does not decide
 * whether a part is feasible, and does not know what a flange is. All of that
 * is `core/v3/connectors.js` — the two-piece bolted rim clamp, its section, the
 * fastener-at-depth check, the section-level fouling test, and the joint
 * feasibility envelope — and it is imported here unchanged. HANDOFF §5.3 calls
 * that file the thing the new direction should be built on top of; this is what
 * building on top of it looks like.
 *
 * The station objects below are therefore FIELD-FOR-FIELD the ones
 * `solveConnectors` emits: same keys, same meanings, same frame construction.
 * That is a hard requirement rather than a convenience — `connectorEndProfiles`,
 * `connectorOBB`, `connectorStationFlags` and `geometry/connectorGeometry.js`
 * all consume them, and a v4 station that is merely "similar" would produce a
 * part solid that looks right and is wrong.
 *
 * =============================================================================
 * A JOINT NO LONGER RUNS ALONG WORLD X
 * =============================================================================
 * On the ribbon every joint ran across the strip along world +X, and both
 * panels' width axes agreed with it. On the network a joint runs along
 * `w = Ŷ × ê` — world X for a z-axis lattice edge, world Z for an x-axis one —
 * so three things that were free before have to be got right here:
 *
 *   THE RUN AXIS. `station.s` is still a WORLD COORDINATE (the power supply's
 *        blocked span is stated in the same units, and a conversion between two
 *        parameterizations is a thing that can only ever be got wrong), but
 *        which world coordinate is now `joint.runAxis`. lattice.js's
 *        `jointRimPoint` replaces that component rather than always the x one.
 *   WHICH LOCAL AXIS EACH PANEL'S RIM RUNS ALONG. A flat cell meets a z-axis
 *        edge on its local X and an x-axis edge on its local Z. lattice.js takes
 *        `runA`/`runB` through each panel's own placement accordingly; nothing
 *        here may assume local X.
 *   THE TWIST. Because `ê` carries the ramp's direction, a descending ramp's
 *        width axis is the negative of its neighbouring cells' — the same
 *        physical line, numbered the other way. `twistDeg` is therefore the
 *        angle between the two rim LINES, not between two rays: see below.
 *
 * `station.axis` is 'x' or 'z' accordingly. v3 used that field for the material
 * direction on its adjacency lattice; here it names the world axis, which is the
 * nearest true thing a network can say.
 *
 * =============================================================================
 * WHAT A v4 JOINT MAKES EASY, AND WHY THE CODE STILL DOES THE HARD VERSION
 * =============================================================================
 * On a drift, a single joint wedges from 3cm to 16cm end to end and the whole
 * apparatus of `spanMinCm` / `spanMaxCm` / `spanStartCm` / `spanEndCm` /
 * `spanSpreadCm` exists to describe that. On a v4 joint the two rims are
 * PARALLEL by construction (lattice.js's half-angle step), so every one of those
 * numbers is `gap` and the spread is 0.
 *
 * They are still computed by measuring the actual rim points at the actual
 * sample positions, not written down as `gap`. Two reasons, and the second is
 * the real one:
 *   - the part machinery is fed measurements everywhere else, and a module that
 *     asserts its own inputs is a module that cannot detect its own bug;
 *   - `tests/test-v4-lattice.mjs` asserts span == gap to 1e-9 as the model's
 *     central claim. That assertion is only worth anything if the number came
 *     out of the geometry.
 *
 * =============================================================================
 * THE FOLD, AND WHY IT IS NOT THE CLOSED FORM
 * =============================================================================
 * `foldDeg` is `−θ` at every joint on a ramp's GROUND end and `+θ` at every
 * joint on its HIGH end, and that IS what the tests assert against. (The
 * ribbon's `α_A − α_B` does not survive the generalisation: a descending ramp is
 * built rising out of its own ground cell, so the A/B ordering no longer tracks
 * the sign of the tilt. The `end` field does, and it is the physical statement —
 * a ramp meets the floor in a valley and the high level on a ridge.) It is not
 * what this file computes. The sign convention that
 * `backHalfProfile` reads is defined by v3's p̂/q̂/r̂ construction — r̂ along the
 * joint, negated if q̂ disagrees with n_A + n_B, then
 * `foldDeg = −atan2((n_A × n_B)·r̂, n_A·n_B)` — and a part lofted from a profile
 * built at the wrong sign is a part with its hooks on backwards. So the
 * construction is reproduced exactly and the closed form is used only to check
 * it from the outside.
 *
 * A consequence worth stating: because r̂ is chosen so that q̂ agrees with the
 * LIT side, flipping BOTH panels of a joint negates `foldDeg`. That is correct
 * and physical — a ridge seen from the lit side is a valley seen from the
 * housing side, and the housings are what the part grips.
 *
 * =============================================================================
 * A JOINT WHOSE PANELS FACE OPPOSITE WAYS GETS NO PART
 * =============================================================================
 * The connector grips the BACK FLANGE of both panels. If one panel is flipped
 * and its neighbour is not, their flanges are on opposite sides of the
 * reference plane and no member of this part family can reach both — the back
 * half would have to pass through the joint to get from one flange to the
 * other. There is no shape of clamp that fixes this, so v4 emits NO STATION and
 * raises `W_JOINT_FLIP_MISMATCH` against the joint. Faking a part for it would
 * be worse than reporting nothing: it would produce an STL someone could print.
 */

import * as THREE from 'three'
import { normalizeConfig } from './schema.js'
import { solveLattice, jointRimPoint } from './lattice.js'
import {
  stationCount,
  clearSpans,
  SPAN_SAMPLES,
  CROWDED_CODE,
  BLOCKED_CODE,
  REDUCED_CODE,
} from '../v3/connectors.js'
import { poweredEdgeBlockedSpan, POWER_SUPPLY } from '../../config.js'

const DEG = 180 / Math.PI

/** A joint whose two panels face opposite ways — see the file header. */
export const FLIP_MISMATCH_CODE = 'W_JOINT_FLIP_MISMATCH'

function r(v) {
  const out = Math.round(v * 1e9) / 1e9
  return out === 0 ? 0 : out
}

const rv = (v) => [r(v.x), r(v.y), r(v.z)]

// =============================================================================
// THE POWER SUPPLY — a rim a connector cannot use
// =============================================================================
/**
 * Which SIDE of a joint carries the power supply.
 *
 * v3's `poweredEdgeOf` answers the same question but is shaped for a tile on a
 * material lattice — it reads `tile.uv` and returns a `{ axis, boundary }` pair
 * in adjacency coordinates, none of which a v4 panel has. Rather than widen a v3
 * function that v3 tests pin down, this is the small v4 equivalent: a joint has
 * exactly two 60cm rims facing each other, A's and B's, and the policy picks
 * between them.
 *
 * A GLOBAL CONVENTION, exactly as in v3, and on the network it is a coarser
 * assumption than it was on the ribbon: a flat cell has FOUR rims and can carry
 * its supply on only one of them, so "every panel is turned the same way" is a
 * statement the 2-D model can no longer even express. Which way each panel is
 * turned is a real design freedom that nothing in the tool models yet (HANDOFF
 * §4, question 6c); the assumption is visible here rather than buried, and it is
 * conservative — it blocks one rim of every joint rather than one rim of every
 * panel.
 *
 *   'low'   the rim on the joint's A side
 *   'high'  the rim on its B side
 *   'none'  no supply — for measuring what the constraint costs
 */
export function poweredRimOf(policy = 'low') {
  if (policy === 'none') return null
  return policy === 'low' ? 'start' : 'end'
}

/**
 * The interval of a joint that a flange-gripping connector cannot use, in the
 * joint's own run coordinates (`runFrom`..`runTo` on `joint.runAxis`).
 *
 * Under a global policy exactly ONE of the two panels contributes a blocked
 * span. The supply is centred on its own rim, so the blocked stretch is the
 * middle 50cm of the 60 and only ~5cm at each end survives.
 */
export function blockedSpansOnJointV4(joint, policy = 'low', supply = POWER_SUPPLY) {
  const rim = poweredRimOf(policy)
  if (rim === null) return []
  const [from, to] = poweredEdgeBlockedSpan(joint.lengthCm, supply)
  return [[joint.runFrom + from, joint.runFrom + to]]
}

// =============================================================================
// THE JOINT FRAME
// =============================================================================
/**
 * The station frame and the signed fold for a whole joint.
 *
 * v3 rebuilds this per station because the span varies along a drift joint and
 * the frame is assembled from the local rim points. On a v4 joint the rims are
 * parallel, so every station's frame is the same frame — it is computed once
 * and shared, which is not an optimisation but a statement: if two stations on
 * one v4 joint ever disagreed about their frame, something upstream would have
 * stopped being parallel.
 *
 * The construction is v3's, verbatim in effect:
 *   p̂  from rim A toward rim B, perpendicular to the joint
 *   q̂  "up", agreeing with the panels' lit side
 *   r̂  along the joint, chosen so (p̂, q̂, r̂) is RIGHT-HANDED
 * Deriving r̂ rather than taking it from a panel is what makes the fold's SIGN
 * meaningful — a frame picked up from whichever unit happened to be `a` would
 * flip sign with unit ordering.
 *
 * DEGENERATE CASE: two units with the same tilt and opposite flips have
 * n_A + n_B = 0, so q̂ has nothing to agree with and r̂ is not negated. The fold
 * then reads ±180°, which is honest (the lit faces ARE antiparallel) but is not
 * a fold any part spans — and such a joint gets no station anyway.
 */
export function jointFrame(joint) {
  const nA = new THREE.Vector3(...joint.normalA)
  const nB = new THREE.Vector3(...joint.normalB)
  const runA = new THREE.Vector3(...joint.runA)
  const runB = new THREE.Vector3(...joint.runB)

  const pa = jointRimPoint(joint, true, joint.runFrom)
  const pb = jointRimPoint(joint, false, joint.runFrom)

  let rHat = runA.clone()
  const pHat = pb.clone().sub(pa)
  pHat.addScaledVector(rHat, -pHat.dot(rHat)).normalize()
  let qHat = rHat.clone().cross(pHat)
  if (qHat.dot(nA.clone().add(nB)) < 0) {
    rHat.negate()
    qHat = rHat.clone().cross(pHat)
  }

  // Signed fold about r̂. Convex — a ridge, lit faces diverging, housings
  // pinching — comes out POSITIVE, which is the reading backHalfProfile wants.
  const cross = nA.clone().cross(nB)
  const foldDeg = -Math.atan2(cross.dot(rHat), nA.dot(nB)) * DEG
  // v3 takes the dihedral independently, as `acos(n_A·n_B)`. Here it is the
  // magnitude of the signed fold instead — the same quantity (V4_SPEC §3), but
  // acos is badly conditioned exactly where v4 spends most of its time: near a
  // flat joint its derivative is unbounded, so the 1e-9 rounding on the normals
  // turns into ~1e-4° of noise at θ → 0, while atan2 stays exact there. Taking
  // one from the other also stops the two disagreeing at the 1e-8 level, which
  // an independent acos does.
  const dihedralDeg = Math.abs(foldDeg)
  // The angle between the two rim LINES, each taken through its own panel's
  // placement. 0 on a v4 joint — the fold is a rotation about the width axis, so
  // it cannot turn one rim relative to the other — which is why a v4 part never
  // wedges. Measured rather than asserted, for the reason in the file header.
  //
  // |runA·runB|, not runA·runB: a rim is a LINE, not a ray, and on a 2-D lattice
  // the two panels at a joint legitimately number their width axes in opposite
  // senses — a descending ramp's `w = Ŷ × ê` is the negative of its cells' local
  // X. The signed form would read 180° on exactly half the joints of every
  // network and mean nothing by it. On the ribbon, where both are +X, the two
  // forms agree exactly.
  const twistDeg = Math.acos(Math.min(1, Math.abs(runA.dot(runB)))) * DEG

  return { pHat, qHat, rHat, nA, nB, runA, runB, foldDeg, dihedralDeg, twistDeg }
}

// =============================================================================
// SOLVE
// =============================================================================
/**
 * Place connectors on every joint of a solved network.
 *
 * @param {object} config raw or normalized v4 config
 * @param {object} [lattice] a network from solveLattice; solved here if omitted
 * @returns {{ stations: Array, perJoint: Array, warnings: Array }}
 */
export function solveConnectorsV4(config, lattice = null) {
  const cfg = normalizeConfig(config)
  const C = lattice ?? solveLattice(cfg)
  const { lengthCm, spacingCm, minPerJoint, powerEdge, supplyMode } = cfg.connectors

  const stations = []
  const perJoint = []
  const warnings = []

  for (const joint of C.joints) {
    const jointIndex = joint.jointIndex

    // --- the one joint no part in this family can span --------------------
    if (joint.flippedA !== joint.flippedB) {
      warnings.push({
        code: FLIP_MISMATCH_CODE,
        joint: jointIndex,
        a: joint.a,
        b: joint.b,
        message:
          `joint ${joint.a}–${joint.b} has one panel flipped and one not, so their back flanges are on ` +
          'opposite sides of the joint — no connector of this family can grip both, and none is placed',
      })
      perJoint.push({
        jointIndex,
        a: joint.a,
        b: joint.b,
        materialLength: r(joint.lengthCm),
        count: 0,
        lengthCm: 0,
        blockedCm: 0,
      })
      continue
    }

    const span = joint.lengthCm
    const wanted = stationCount(span, { spacingCm, minPerJoint })

    // --- where the power supply forbids a part ----------------------------
    // In 'relief' mode (the default) the supply is NOT an obstruction: it is
    // flush with the flange to within 1mm, so a relief in the lip clears it and
    // the part is placed normally — but flagged, because the lip then bears on
    // the supply housing rather than the panel frame. 'block' keeps the older,
    // stricter reading so the difference can be measured. Identical policy to
    // v3, and for the same physical reason.
    const supplySpans = blockedSpansOnJointV4(joint, powerEdge)
    const blocked = supplyMode === 'block' ? supplySpans : []
    const clear = clearSpans(blocked, joint.runFrom, joint.runTo)
    const clearLength = clear.reduce((n, [a, b]) => n + (b - a), 0)

    // Each clear stretch gets its own parts, sized by the same spacing rule. A
    // stretch shorter than one part gets none — it cannot hold one.
    const usable = clear.filter(([a, b]) => b - a >= lengthCm - 1e-9)
    const perStretch = usable.map(([a, b]) => Math.max(1, Math.ceil((b - a) / spacingCm - 1e-9)))
    let total = perStretch.reduce((n, k) => n + k, 0)
    // `minPerJoint` is a FLOOR on the joint: parts are added until it is met
    // even when that means shortening them, and W_CONNECTOR_CROWDED reports the
    // cost. Refusing instead would silently drop a structural requirement.
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
          a: joint.a,
          b: joint.b,
          message:
            count === 0
              ? `joint ${joint.a}–${joint.b} has a power supply behind it and only ${r(clearLength)}cm of ` +
                `usable rim in stretches too short for a ${lengthCm}cm part — it carries NO connector`
              : `joint ${joint.a}–${joint.b} has a power supply behind it: ${count} parts fit where ` +
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
        a: joint.a,
        b: joint.b,
        materialLength: r(span),
        count: 0,
        lengthCm: 0,
        blockedCm: r(span - clearLength),
      })
      continue
    }

    const room = Math.min(...usable.map(([a, b], k) => (b - a) / perStretch[k]))
    const partLength = Math.min(lengthCm, room)
    if (partLength < lengthCm - 1e-9) {
      warnings.push({
        code: CROWDED_CODE,
        joint: jointIndex,
        a: joint.a,
        b: joint.b,
        message:
          `joint ${joint.a}–${joint.b} takes ${count} parts in ${usable.length} usable stretch` +
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

    const frame = jointFrame(joint)
    const { pHat, qHat, rHat, nA, nB, runA, runB, foldDeg, dihedralDeg, twistDeg } = frame
    const inA = new THREE.Vector3(...joint.inwardA)
    const inB = new THREE.Vector3(...joint.inwardB)

    const spanAt = (sm) =>
      jointRimPoint(joint, true, sm).distanceTo(jointRimPoint(joint, false, sm))

    for (let k = 0; k < count; k++) {
      const s = centres[k]
      const pa = jointRimPoint(joint, true, s)
      const pb = jointRimPoint(joint, false, s)
      const spanCm = pa.distanceTo(pb)

      // Bracket the span over THIS PART's footprint. Constant on a v4 joint, and
      // sampled anyway — see the file header.
      const along = rHat.dot(runA) >= 0 ? 1 : -1
      let spanMin = Infinity
      let spanMax = -Infinity
      for (let m = 0; m < SPAN_SAMPLES; m++) {
        const f = (m / (SPAN_SAMPLES - 1)) - 0.5
        const d = spanAt(s + f * partLength)
        if (d < spanMin) spanMin = d
        if (d > spanMax) spanMax = d
      }
      const spanStart = spanAt(s - along * partLength / 2)
      const spanEnd = spanAt(s + along * partLength / 2)

      const half = partLength / 2
      const bearsOnPowerSupply = supplySpans.some(([lo, hi]) => s + half > lo && s - half < hi)

      stations.push({
        id: `J${jointIndex}S${k}`,
        bearsOnPowerSupply,
        jointIndex,
        a: joint.a,
        b: joint.b,
        // v3's `axis` is the material direction the joint RUNS along, named on
        // its adjacency lattice. A v4 joint runs along `Ŷ × ê` — world X for a
        // z-axis lattice edge, world Z for an x-axis one — so it is named for
        // the world axis instead, which is the nearest true thing to say.
        axis: joint.runAxis === 0 ? 'x' : 'z',
        index: k,
        of: count,
        s: r(s),
        tAlong: r((k + 0.5) / count),
        lengthCm: r(partLength),
        spanCm: r(spanCm),
        spanMinCm: r(spanMin),
        spanMaxCm: r(spanMax),
        spanSpreadCm: r(spanMax - spanMin),
        spanStartCm: r(spanStart),
        spanEndCm: r(spanEnd),
        dihedralDeg: r(dihedralDeg),
        foldDeg: r(foldDeg),
        twistDeg: r(twistDeg),
        mid: rv(pa.clone().add(pb).multiplyScalar(0.5)),
        frame: { p: rv(pHat), q: rv(qHat), r: rv(rHat) },
        aFrame: { point: rv(pa), run: rv(runA), normal: rv(nA), inward: rv(inA) },
        bFrame: { point: rv(pb), run: rv(runB), normal: rv(nB), inward: rv(inB) },
      })
    }

    perJoint.push({
      jointIndex,
      a: joint.a,
      b: joint.b,
      materialLength: r(span),
      count,
      lengthCm: r(partLength),
      blockedCm: r(span - clearLength),
    })
  }

  return { stations, perJoint, warnings }
}
