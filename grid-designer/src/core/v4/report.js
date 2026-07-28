/**
 * grid-designer v4 — the report: joints, the envelope, collisions, metrics.
 *
 * HEADLESS ZONE (src/core/): pure functions, importable from plain node.
 *   - explicit `.js` extensions on ALL relative imports
 *   - may import `three` math classes only; never components / store / DOM
 *   - same config in → byte-identical output out
 *
 * =============================================================================
 * THE THING v3 COULD NEVER PRODUCE
 * =============================================================================
 * v3's verdict on itself (HANDOFF §0) was that **the tool became very good at
 * saying no and never acquired a way to say yes**. Its report was a list of
 * measured damage — 58 of 60 joints flagged, 32 unspannable, 0 plates placeable
 * — every number correct, none of them a route to a design that works.
 *
 * A v4 design is inside the connector envelope by construction, so the joint
 * table is nearly always clean and would be nearly worthless on its own. What
 * this module owes the user instead is the boundary:
 *
 *     maxAngleDeg(gap)  the largest θ this gap admits with every joint clean
 *     minGapCm(θ)       the smallest gap that admits this angle
 *
 * plus the headroom to each. Those two numbers are the pivot's whole point, and
 * they are a PERMISSION rather than a complaint: "you have 3.6° left" is a
 * sentence v3 could not say about anything.
 *
 * =============================================================================
 * WHY THE ENVELOPE IS BISECTED AND NOT LOOKED UP
 * =============================================================================
 * `foldLimitDeg(gap)` in core/v3/connectors.js already gives a fold limit, and
 * it is the wrong number to report. It is where the two panels' own back corners
 * meet — the PANEL's limit. The connector fouls 3–6° before that (HANDOFF §2.20),
 * and other rules bind earlier still: at a 2cm gap it is the FASTENER, pinched
 * at the depth the insert sits at, that runs out first. The honest limit is
 * where the FLAGS start, so this module finds it by building real stations at a
 * trial θ and asking `connectorStationFlags` — the same function that judges the
 * real design, so the boundary and the verdict can never disagree.
 *
 * The bisection assumes the predicate is MONOTONE — clean below the boundary,
 * dirty above — and it is, for a reason rather than by luck: every rule that
 * binds here (fastener gap at depth, back-half self-intersection, panel-on-panel
 * fouling at the joint) tightens with |fold| and loosens with span, and none of
 * them is non-monotone in either. Concave folds are unconstrained by
 * panel-on-panel contact — the housings DIVERGE — so only the convex joints
 * bind, which is why the pattern gets more angle than a zig-zag would.
 *
 * The network does not change this at all, and that is worth stating: every
 * joint on the lattice has the same |fold| = θ and the same span = gap, so the
 * envelope of a 10 × 10 network is the envelope of a 1 × 2 one. It is still
 * measured over the real design rather than assumed, because a module that
 * asserts its own inputs cannot detect its own bug — but it means the envelope
 * is a property of (gap, θ) and nothing else, which is what makes it usable as a
 * permission BEFORE a design exists.
 *
 * THE WAVE BREAKS THAT SENTENCE, and the report has to stop saying it.
 *
 * Under `pattern.kind: 'wave'` every joint has its OWN fold — that is the point
 * of the mode — so `maxAngleDeg` is no longer a property of `(gap, θ)`. It is
 * still a true and useful number, because every angle in the design is derived
 * from `angleDeg` and steepens with it, so the bisection is bisecting the one
 * knob the user actually holds: the answer means "the largest BASE angle this
 * design admits", not "the largest fold the connector takes". What must not
 * happen is the report quoting it as though it were the second thing.
 *
 * So under 'wave' the envelope carries `perJoint` as well: the worst fold in the
 * design, which joint it is on, how many joints are dirty, and how many folds
 * there actually are. `angleIsPerJoint` is the flag that says which reading
 * applies. Under 'trapezoid' none of it is emitted and the old sentence stands —
 * `tests/test-v4-report.mjs` §2 still asserts the 1 × 2 and the 4 × 4 report the
 * same boundary, and that has to keep passing.
 *
 * A fixed 60 steps, never "until converged": determinism is the standing rule in
 * this core, and 60 halvings of a 75° range is 6.5e-17°, far below anything that
 * could matter.
 *
 * =============================================================================
 * WHAT "CLEAN" MEANS, EXACTLY
 * =============================================================================
 * No station on any joint carries a flag in `ENVELOPE_HARD_FLAGS`. Two
 * deliberate exclusions, both because they do not move with θ or gap and
 * including them would make every design dirty at every angle, which would make
 * the envelope report a constant instead of a boundary:
 *
 *   W_BEARS_ON_POWER_SUPPLY  fires on every station of every powered joint in
 *        'relief' mode, at any angle. It is a note about what the lip bears on,
 *        and whether a driver housing is something to clamp against is a
 *        hardware question this model cannot answer.
 *   W_JOINT_FLIP_MISMATCH    is not a station flag at all, and a flip is not an
 *        angle. A mismatched joint carries no part at ANY θ, so folding it into
 *        the envelope would report "no angle works" for a problem no angle can
 *        fix. It is reported against the joint instead.
 *
 * `W_FRONT_BAR_FOULS_PANEL` is silent for a structural reason worth stating:
 * `sectionFouling` only tests the bar when the station carries a `barWidthCm`,
 * and bar widths are assigned AFTER flagging (this is v3's ordering too — see
 * its report.js). So the flag never fires, in v3 or here, and folding it into
 * the envelope would silently change what the envelope means.
 *
 * =============================================================================
 * THE FRONT BAR AND THE VALLEYS — reported separately, on purpose
 * =============================================================================
 * Ask `sectionFouling` the question anyway, with a bar width assigned, and it
 * says something this pattern needs to hear: **a flat front bar bites the
 * bezels on a CONCAVE joint past 12.37° of fold.** In a valley the two lit
 * faces tilt up toward the bar while the bar stays flat across the gap, so its
 * overhang meets the rising bezel peaks.
 *
 * Measured by bisection at gaps 1, 1.5, 2, 3 and 4cm: **12.37° at every one of
 * them.** That constancy is the tell that it is a property of the BAR, not of
 * the joint — the overhang past the gap is `frontLipCm` whatever the gap is,
 * and it is the overhang that collides. Widening the gap does not buy a degree.
 *
 * It is kept OUT of `ENVELOPE_HARD_FLAGS` and given its own reading
 * (`envelope.frontBar`, and `W_FRONT_BAR_FOULS_BEZEL` per joint) for two
 * reasons:
 *
 *   - it would collapse `maxAngleDeg` from ~33.6° to 12.37° and hide the limit
 *     that governs the CONNECTOR, which is the one a fold has to respect;
 *   - it is fixable in the part rather than in the form. A relief or a chamfer
 *     on the bar's underside, or a narrower bar on concave stations, moves it.
 *     A limit you can design away does not belong in the same number as one you
 *     cannot.
 *
 * The network makes this MORE unavoidable than the ribbon did, not less: every
 * ramp meets its ground cell in a valley, so exactly half of every network's
 * joints are concave, at any θ.
 *
 * =============================================================================
 * THE CORNER CLEARANCE — the one number with no 1-D analogue
 * =============================================================================
 * A ground cell's +x ramp occupies the corridor beyond its +x edge; its +z ramp
 * occupies the one beyond its +z edge. The region beyond BOTH is occupied by
 * neither, so four ramps meet at a lattice corner without touching and leave a
 * diamond opening. That is intended — the surface stays open, as asked.
 *
 * Whether their HOUSINGS clear at large θ is a question, not an assumption. Two
 * ramps off the same cell are not joined to each other, so the ordinary
 * non-adjacent collision pass already tests them and would report an overlap;
 * but by the time it does, the design is already broken. `cornerClearance`
 * reports the distance BEFORE it becomes a collision, because it is the quantity
 * that decides how far θ can go in two dimensions and there was nothing like it
 * on the ribbon.
 *
 * It is a conservative number, in two compounding ways, and BOTH matter when
 * reading it:
 *
 *   THE SAT BOUND. The separation is the largest gap over the same 15 axes
 *        `collide.js` uses, which is a LOWER BOUND on the true distance between
 *        two boxes — exact when the closest features are a face-face or
 *        edge-edge pair, an underestimate for a vertex-vertex one. It never
 *        claims more room than there is, which is the direction a clearance
 *        number has to err in.
 *   THE BOX ITSELF, which is the big one. A panel's OBB is the full 4.1cm
 *        housing everywhere, and the real section is only `outerWallDepth`
 *        (1.2cm) deep at the rim, reaching full thickness `bodyInset` (4.62cm)
 *        inboard. The two ramps off a cell approach each other at their
 *        CORNERS, where the real solid is thin in both directions — so the
 *        overlap this reports is between box corners that are mostly air.
 *        Measured on the default 3 × 5: at 30° the boxes overlap by 0.12cm
 *        while the two BACK PLATES are still 9.43cm apart, and they are still
 *        7.19cm apart at 50°.
 *
 * So `worstCm` going negative means "the boxes have met", not "the panels have".
 * The right answer is a section-level test — the `sectionFouling` analogue for a
 * pair that shares no joint — and there is no such function in `core/v3/`. Until
 * there is, a positive clearance is a GUARANTEE and a negative one is a QUESTION,
 * and `W_CORNER_RAMPS_MEET` says so in as many words.
 *
 * =============================================================================
 * THE SPACERS — a cross-check, not a restatement
 * =============================================================================
 * `report.spacers` carries the count, the count per cell, the clearance asked
 * for, and the DISTINCT HEIGHTS the posts came out at. That last field is the
 * whole point of the section: the clearance is applied by `lattice.js` as a
 * translation of the network and the heights are measured by `spacers.js` off
 * the resulting undersides, so the two are independent and their agreement is
 * evidence rather than tautology. One distinct height, equal to the clearance,
 * is what a correct grounded design looks like; anything else raises
 * `W_SPACER_MISMATCH` and names the worst post.
 */

import { normalizeConfig, ANGLE_MIN, ANGLE_MAX, GAP_MIN, GAP_MAX } from './schema.js'
import { solveLattice } from './lattice.js'
import { solveConnectorsV4, jointFrame } from './connectors.js'
import * as THREE from 'three'
import {
  connectorStationFlags,
  frontBarWidthFor,
  sectionFouling,
  CONNECTOR_LIMITS,
} from '../v3/connectors.js'
import { findCollisions } from '../v3/collide.js'
import { solveObstacles, OBSTACLE_HIT_CODE } from './obstacles.js'
import { solveSpacers, SPACER_MISMATCH_CODE } from './spacers.js'

/** Ignore contact shallower than this when calling something a collision. The
 *  panels are MEANT to nearly touch across the joint, so a zero-tolerance test
 *  would flag a healthy design on floating-point noise alone. */
export const COLLISION_MIN_DEPTH_CM = 0.05

/** How far below y = 0 / x = 0 counts as through the floor or the wall. Panels
 *  are placed to land exactly on those planes, so the tolerance is numerical
 *  rather than physical. */
const PLANE_EPSILON = 1e-6

/** Fixed, never "until converged" — see the file header. */
export const BISECTION_STEPS = 60

/** How far past a found boundary to probe when asking what stopped you. Large
 *  enough to be outside a boundary located to 1e-16, small enough that the
 *  flags it reports are the ones that bind AT the boundary. */
const PROBE = 0.01

/**
 * The flags that make a design dirty for envelope purposes. See "WHAT CLEAN
 * MEANS" in the file header for the two that are deliberately absent.
 */
export const ENVELOPE_HARD_FLAGS = [
  'W_CONNECTOR_INFEASIBLE',
  'W_CONNECTOR_PINCH',
  'W_CONNECTOR_SPAN',
  'W_CONNECTOR_TWIST',
  'W_FASTENER_PINCHED',
  'W_BACK_HALF_FOULS_PANEL',
  'W_FRONT_BAR_FOULS_PANEL',
  'W_PANELS_COLLIDE_AT_JOINT',
]

/** Below this a fold is reported as flat rather than convex or concave. It is a
 *  numerical threshold, not a physical one: the fold is computed by atan2 and is
 *  exactly 0 at θ = 0, so anything above the rounding floor is a real fold. */
const FLAT_FOLD_DEG = 1e-9

function r(v) {
  const out = Math.round(v * 1e9) / 1e9
  return out === 0 ? 0 : out
}

/**
 * The same 1e-9 rounding, but always AWAY from the permitted side.
 *
 * `maxAngleDeg` and `minGapCm` are the largest / smallest value the bisection
 * proved CLEAN, and `r()` can move either of them by half a nanometre — which
 * is enough to land on the dirty side of a boundary located to 1e-16. Then
 * `isClean(reportedValue)` is false and the tool has issued a permission it
 * would itself refuse. Rounding an upper bound down and a lower bound up keeps
 * the reported number inside the region it describes, at a cost (a nanodegree,
 * a picometre) that is four orders below anything the design cares about.
 */
const rDown = (v) => Math.floor(v * 1e9) / 1e9
const rUp = (v) => Math.ceil(v * 1e9) / 1e9

// =============================================================================
// THE ENVELOPE
// =============================================================================
/**
 * Is every station of this design clean? See "WHAT CLEAN MEANS" above.
 *
 * A design with no joints at all (a single cell) is vacuously clean, and the
 * envelope then correctly reports the range limit: there is nothing to fold, so
 * no angle is forbidden.
 */
export function isClean(config) {
  const cfg = normalizeConfig(config)
  const C = solveLattice(cfg)
  const K = solveConnectorsV4(cfg, C)
  for (const st of K.stations) {
    const flags = connectorStationFlags(st, CONNECTOR_LIMITS)
    for (const f of flags) if (ENVELOPE_HARD_FLAGS.includes(f)) return false
  }
  return true
}

/** Every hard flag this design's stations carry, sorted and de-duplicated. */
function hardFlagsOf(config) {
  const cfg = normalizeConfig(config)
  const K = solveConnectorsV4(cfg, solveLattice(cfg))
  const out = new Set()
  for (const st of K.stations) {
    for (const f of connectorStationFlags(st, CONNECTOR_LIMITS)) {
      if (ENVELOPE_HARD_FLAGS.includes(f)) out.add(f)
    }
  }
  return [...out].sort()
}

/**
 * The largest θ this config's gap admits with every joint clean.
 *
 * Returns `{ value, atRangeLimit }`. `value` is null when the design is dirty
 * even flat — at that point the angle is not what is wrong with it, and
 * reporting a number would suggest otherwise.
 */
function solveMaxAngle(cfg) {
  const at = (angleDeg) => isClean({ ...cfg, angleDeg })
  if (!at(ANGLE_MIN)) return { value: null, atRangeLimit: false }
  if (at(ANGLE_MAX)) return { value: ANGLE_MAX, atRangeLimit: true }
  let lo = ANGLE_MIN // clean
  let hi = ANGLE_MAX // dirty
  for (let k = 0; k < BISECTION_STEPS; k++) {
    const mid = (lo + hi) / 2
    if (at(mid)) lo = mid
    else hi = mid
  }
  // The largest angle KNOWN clean, not the midpoint of the bracket: the number
  // is a permission, and a permission that has not been tested is not one.
  return { value: lo, atRangeLimit: false }
}

/**
 * The smallest gap this config's angle admits, same construction — but bracketed
 * from the design's OWN gap when that is already clean, rather than from
 * GAP_MAX.
 *
 * Not an optimisation. `minGapCm` is the bottom of the band, and the top of the
 * band has its own, unrelated boundary: `CONNECTOR_LIMITS.maxSpanCm` is 8cm and
 * so is `GAP_MAX`, so a design sitting exactly on GAP_MAX is on that limit and
 * reads as dirty. Bracketing from GAP_MAX would then make the whole lower
 * boundary unreportable for a reason that has nothing to do with it.
 */
function solveMinGap(cfg) {
  const at = (gap) => isClean({ ...cfg, gap })
  if (at(GAP_MIN)) return { value: GAP_MIN, atRangeLimit: true }
  const ceiling = at(cfg.gap) ? cfg.gap : GAP_MAX
  if (!at(ceiling)) return { value: null, atRangeLimit: false }
  let lo = GAP_MIN // dirty
  let hi = ceiling // clean
  for (let k = 0; k < BISECTION_STEPS; k++) {
    const mid = (lo + hi) / 2
    if (at(mid)) hi = mid
    else lo = mid
  }
  // The smallest gap KNOWN clean, for the same reason.
  return { value: hi, atRangeLimit: false }
}

/** The code for a joint whose valley the front bar cannot lie across. Distinct
 *  from v3's dormant `W_FRONT_BAR_FOULS_PANEL` so the two can never be read as
 *  the same finding — see "THE FRONT BAR AND THE VALLEYS" in the file header. */
export const FRONT_BAR_CODE = 'W_FRONT_BAR_FOULS_BEZEL'

/** Does a bar sized for `gapCm` clear the bezels at this concave fold? */
function frontBarClears(gapCm, concaveFoldDeg) {
  // sectionFouling reads the fold's SIGN, so the probe has to be genuinely
  // negative — handing it a magnitude would silently test a ridge instead.
  const station = {
    spanCm: gapCm,
    spanStartCm: gapCm,
    spanEndCm: gapCm,
    foldDeg: -Math.abs(concaveFoldDeg),
    barWidthCm: frontBarWidthFor(gapCm),
  }
  return !sectionFouling(station).frontBar
}

/**
 * The concave fold at which a front bar sized for this gap starts biting the
 * bezels. Same fixed bisection as the envelope, and monotone for the same kind
 * of reason: the bezels rise toward the bar as the valley deepens and never
 * fall back.
 *
 * Measures ~12.37° at every gap in 1–4cm, because the overhang that collides is
 * `frontLipCm` and does not depend on the gap. Solved rather than hardcoded so
 * that a change to the bar profile shows up here instead of going unnoticed.
 */
export function solveFrontBarLimit(gapCm) {
  if (!frontBarClears(gapCm, 0)) return 0
  if (frontBarClears(gapCm, ANGLE_MAX)) return ANGLE_MAX
  let lo = 0
  let hi = ANGLE_MAX
  for (let k = 0; k < BISECTION_STEPS; k++) {
    const mid = (lo + hi) / 2
    if (frontBarClears(gapCm, mid)) lo = mid
    else hi = mid
  }
  return lo
}

/**
 * The per-joint reading the wave needs: which joint is worst, and whether every
 * one of them is clean.
 *
 * The flags are taken from the REAL stations of the REAL design — the same
 * `connectorStationFlags` the envelope bisection asks — so "every joint is clean"
 * here and `envelope.clean` can never disagree. `worst` is the largest |fold|,
 * because that is the joint the connector is closest to running out on, and it is
 * named so that a design past the limit says WHERE rather than only THAT.
 */
function perJointEnvelope(cfg, C) {
  const K = solveConnectorsV4(cfg, C)
  const dirty = new Map()
  for (const st of K.stations) {
    const bad = connectorStationFlags(st, CONNECTOR_LIMITS).filter((f) => ENVELOPE_HARD_FLAGS.includes(f))
    if (bad.length === 0) continue
    const have = dirty.get(st.jointIndex) ?? new Set()
    for (const f of bad) have.add(f)
    dirty.set(st.jointIndex, have)
  }

  // The worst joint is the largest CONVEX fold, not the largest |fold|.
  //
  // That is not a preference, it is what binds: on a ridge the two housings pinch
  // toward each other and every rule in `ENVELOPE_HARD_FLAGS` tightens; in a
  // valley they diverge and the connector is unconstrained by them (this file's
  // header, "WHY THE ENVELOPE IS BISECTED"). Ranking by magnitude picked the
  // deepest valley on the first design this was run against — a 55° joint
  // carrying no flags at all, reported as the worst thing in a network that had
  // 18 genuinely dirty ridges. Signed max is the question actually being asked.
  let worst = null
  for (const j of C.joints) {
    const fold = jointFrame(j).foldDeg
    if (worst === null || fold > worst.foldDeg) {
      worst = { id: j.id, jointIndex: j.jointIndex, foldDeg: fold, edge: j.edge, end: j.end }
    }
  }

  // How many DISTINCT folds there are, to the degree the part types are binned
  // at — a wave whose joints all land in one bin is, for connector purposes, a
  // trapezoid, and the number says so without anyone having to eyeball a table.
  const folds = new Set(C.joints.map((j) => r(Math.abs(jointFrame(j).foldDeg))))

  return {
    jointCount: C.joints.length,
    distinctFolds: folds.size,
    dirtyJointCount: dirty.size,
    allClean: dirty.size === 0,
    worst: worst && {
      ...worst,
      foldDeg: r(worst.foldDeg),
      flags: [...(dirty.get(worst.jointIndex) ?? [])].sort(),
    },
    dirty: [...dirty.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([jointIndex, flags]) => {
        const j = C.joints.find((x) => x.jointIndex === jointIndex)
        return { jointIndex, id: j?.id ?? null, foldDeg: r(jointFrame(j).foldDeg), flags: [...flags].sort() }
      }),
  }
}

/**
 * Where this design sits inside the connector envelope, and how much room it has
 * left in each direction.
 *
 * `limitedBy` is what actually stops you: the flags raised just past the
 * boundary. When the design is already dirty it is the flags it carries now,
 * because that is the honest answer to "what is stopping this".
 */
export function solveEnvelope(config) {
  const cfg = normalizeConfig(config)
  const clean = isClean(cfg)
  const angle = solveMaxAngle(cfg)
  const gap = solveMinGap(cfg)

  const limits = new Set()
  if (!clean) {
    for (const f of hardFlagsOf(cfg)) limits.add(f)
  } else {
    if (angle.value !== null && !angle.atRangeLimit) {
      for (const f of hardFlagsOf({ ...cfg, angleDeg: Math.min(ANGLE_MAX, angle.value + PROBE) })) {
        limits.add(f)
      }
    }
    if (gap.value !== null && !gap.atRangeLimit) {
      for (const f of hardFlagsOf({ ...cfg, gap: Math.max(GAP_MIN, gap.value - PROBE) })) limits.add(f)
    }
  }

  // The front bar's own limit, reported beside the envelope rather than inside
  // it — see "THE FRONT BAR AND THE VALLEYS". `worstConcaveDeg` comes from the
  // network's actual joints, not from θ.
  const barLimit = solveFrontBarLimit(cfg.gap)
  const C = solveLattice(cfg)
  let worstConcave = 0
  for (const j of C.joints) {
    const fold = jointFrame(j).foldDeg
    if (fold < -FLAT_FOLD_DEG) worstConcave = Math.max(worstConcave, -fold)
  }

  return {
    ...(cfg.pattern.kind === 'wave' ? { angleIsPerJoint: true, perJoint: perJointEnvelope(cfg, C) } : {}),
    angleDeg: cfg.angleDeg,
    gapCm: cfg.gap,
    clean,
    frontBar: {
      concaveLimitDeg: r(barLimit),
      worstConcaveDeg: r(worstConcave),
      clears: worstConcave <= barLimit,
      headroomDeg: r(barLimit - worstConcave),
    },
    // Rounded away from the permitted side — see `rDown` / `rUp`. The headrooms
    // are taken from the SAME rounded numbers the report quotes, so a reader
    // adding them back never disagrees with the tool by a nanometre.
    maxAngleDeg: angle.value === null ? null : rDown(angle.value),
    angleHeadroomDeg: angle.value === null ? null : r(rDown(angle.value) - cfg.angleDeg),
    angleAtRangeLimit: angle.atRangeLimit,
    minGapCm: gap.value === null ? null : rUp(gap.value),
    gapHeadroomCm: gap.value === null ? null : r(cfg.gap - rUp(gap.value)),
    gapAtRangeLimit: gap.atRangeLimit,
    limitedBy: [...limits].sort(),
    bisectionSteps: BISECTION_STEPS,
  }
}

// =============================================================================
// THE CORNER CLEARANCE
// =============================================================================
/**
 * How far apart two OBBs are, as the largest separation over the 15 SAT axes.
 *
 * A LOWER BOUND on the true distance, and deliberately so — see the file header.
 * Negative means they interpenetrate, in which case the magnitude is a lower
 * bound on the penetration depth rather than a distance.
 *
 * It lives here rather than in `collide.js` because `core/v3/` is frozen (that
 * file is the shipped, tested SAT and nothing in this package may touch it), and
 * because a "clearance" is a reporting question rather than a collision one: the
 * collision pass already answers whether two boxes overlap, exactly.
 */
export function obbClearance(a, b) {
  const axesOf = (q) => {
    const quat = new THREE.Quaternion(q[0], q[1], q[2], q[3]).normalize()
    return [
      new THREE.Vector3(1, 0, 0).applyQuaternion(quat),
      new THREE.Vector3(0, 1, 0).applyQuaternion(quat),
      new THREE.Vector3(0, 0, 1).applyQuaternion(quat),
    ]
  }
  const axesA = axesOf(a.quaternion)
  const axesB = axesOf(b.quaternion)
  const T = new THREE.Vector3(
    b.center[0] - a.center[0],
    b.center[1] - a.center[1],
    b.center[2] - a.center[2],
  )
  const candidates = [...axesA, ...axesB]
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      const cross = new THREE.Vector3().crossVectors(axesA[i], axesB[j])
      const len2 = cross.lengthSq()
      // Near-parallel edge pairs are skipped rather than normalized, for exactly
      // the reason collide.js's header gives: normalizing a near-zero vector
      // amplifies float noise into a meaningless direction, and here that would
      // fabricate a clearance rather than merely a separation verdict.
      if (len2 < 1e-12) continue
      candidates.push(cross.divideScalar(Math.sqrt(len2)))
    }
  }
  const radius = (halfExtents, axes, L) =>
    halfExtents[0] * Math.abs(axes[0].dot(L)) +
    halfExtents[1] * Math.abs(axes[1].dot(L)) +
    halfExtents[2] * Math.abs(axes[2].dot(L))

  let best = -Infinity
  for (const L of candidates) {
    const sep = Math.abs(T.dot(L)) - radius(a.halfExtents, axesA, L) - radius(b.halfExtents, axesB, L)
    if (sep > best) best = sep
  }
  return best
}

/**
 * The tightest place two ramps off the same cell come to each other.
 *
 * Ramps are grouped by the CELL they are joined to, taken from the joint table
 * rather than from the edge names, so a ramp whose joint does not exist (an
 * absent cell, a switched-off edge) is not in any group. Every unordered pair
 * within a group is measured; a pair can only ever appear in one group, since
 * two distinct edges share at most one cell.
 *
 * Note that this includes COLLINEAR pairs — the two ramps a cell in the middle
 * of a column carries, one to each side. They are in no danger of touching (they
 * are a pitch apart) and including them costs nothing, while writing a rule to
 * exclude them would mean deciding what "a corner" is, which is exactly the sort
 * of special case that later turns out to have been wrong.
 *
 * Exported because it is a headline number of the report (V4_SPEC §9.6) and
 * because bisecting on it directly — rather than on a whole `buildReportV4`,
 * which drags the 130-solve envelope along with it — is the only affordable way
 * to ask "at what θ does the corner close".
 */
export function solveCornerClearance(C) {
  const byId = new Map(C.panels.map((p) => [p.id, p]))
  const rampsAtCell = new Map()
  for (const jt of C.joints) {
    const cellId = jt.panelA === 'cell' ? jt.a : jt.b
    const rampId = jt.panelA === 'ramp' ? jt.a : jt.b
    if (!rampsAtCell.has(cellId)) rampsAtCell.set(cellId, [])
    rampsAtCell.get(cellId).push(rampId)
  }

  const pairs = []
  for (const [cellId, ramps] of rampsAtCell) {
    const sorted = [...new Set(ramps)].sort()
    for (let a = 0; a < sorted.length; a++) {
      for (let b = a + 1; b < sorted.length; b++) {
        const A = byId.get(sorted[a])
        const B = byId.get(sorted[b])
        if (!A || !B) continue
        pairs.push({
          cell: cellId,
          a: A.id,
          b: B.id,
          clearanceCm: r(obbClearance(A.obb, B.obb)),
        })
      }
    }
  }
  pairs.sort((x, y) => x.clearanceCm - y.clearanceCm || (x.a < y.a ? -1 : 1))
  return {
    pairCount: pairs.length,
    worstCm: pairs.length ? pairs[0].clearanceCm : null,
    worst: pairs.length ? { cell: pairs[0].cell, a: pairs[0].a, b: pairs[0].b } : null,
    touchingPairs: pairs.filter((p) => p.clearanceCm <= 0).length,
    pairs,
  }
}

// =============================================================================
// THE REPORT
// =============================================================================
/**
 * @param {object} config raw or normalized v4 config
 * @param {object} [lattice] a network from solveLattice; solved here if omitted
 * @param {object} [connectors] stations from solveConnectorsV4; solved if omitted
 * @param {object} [spacers] posts from solveSpacers; solved if omitted
 */
export function buildReportV4(config, lattice = null, connectors = null, spacers = null) {
  const cfg = normalizeConfig(config)
  const C = lattice ?? solveLattice(cfg)
  const K = connectors ?? solveConnectorsV4(cfg, C)
  const S = spacers ?? solveSpacers(cfg, C)

  const warnings = [...K.warnings]

  // --- the wave's own findings ----------------------------------------------
  // Both are about a REQUEST THE MODEL DID NOT HONOUR, which is the one category
  // that must never be silent: an edge pinned at ANGLE_MAX because the scrunch
  // asked for more compression than the angle band can deliver, and a wall anchor
  // asked for on a field that has no level for it to descend to (lattice.js's
  // header). Neither is a defect in the design — they are the tool declining to
  // invent geometry, and saying so.
  if (C.lattice.wave) {
    for (const w of C.lattice.wave.warnings) warnings.push(w)
    if (cfg.placement.wallAnchor === 'braced') {
      warnings.push({
        code: 'W_WAVE_NO_ANCHOR',
        message:
          'placement.wallAnchor is "braced", and the wave builds no anchor ramps. The anchor of ' +
          'V4_SPEC §9.4 descends exactly one level rise to the floor, and the wave\'s height field ' +
          'h(i,j) = f(i) + g(j) has no such level — the toe would land in mid-air. The wall column ' +
          'cantilevers off its own edge here, exactly as it does with "free"',
      })
    }
  }

  // --- joints ---------------------------------------------------------------
  const stationsByJoint = new Map()
  for (const st of K.stations) {
    if (!stationsByJoint.has(st.jointIndex)) stationsByJoint.set(st.jointIndex, [])
    stationsByJoint.get(st.jointIndex).push(st)
  }

  const joints = C.joints.map((j) => {
    const frame = jointFrame(j)
    const mine = stationsByJoint.get(j.jointIndex) ?? []
    // Merged across the joint's stations. On a v4 joint every station sees the
    // same span and the same fold, so this is a de-duplication rather than a
    // union of differing verdicts — but it is written as a union because that is
    // what it means, and because a joint split by a power supply can carry parts
    // of different LENGTHS, which W_CONNECTOR_TWIST does depend on.
    const flags = new Set()
    for (const st of mine) for (const f of connectorStationFlags(st, CONNECTOR_LIMITS)) flags.add(f)
    // The span, measured rather than assumed — the model's central claim.
    const spanCm = Math.hypot(j.rimB[0] - j.rimA[0], j.rimB[1] - j.rimA[1], j.rimB[2] - j.rimA[2])
    return {
      id: j.id,
      jointIndex: j.jointIndex,
      a: j.a,
      b: j.b,
      // Which lattice edge's ramp this is, and which of its two ends. `end` is
      // the physical statement the fold's sign follows from: a ramp meets its
      // ground cell in a valley and its high cell on a ridge, whichever way it
      // happens to run.
      edge: j.edge,
      end: j.end,
      anchor: j.anchor,
      axis: j.runAxis === 0 ? 'x' : 'z',
      spanCm: r(spanCm),
      foldDeg: r(frame.foldDeg),
      dihedralDeg: r(frame.dihedralDeg),
      twistDeg: r(frame.twistDeg),
      // Convex is a ridge — the lit faces diverge and the housings pinch, which
      // is the direction that closes on the connector. Concave joints are the
      // free ones (V4_SPEC §3).
      sense: frame.foldDeg > FLAT_FOLD_DEG ? 'convex' : frame.foldDeg < -FLAT_FOLD_DEG ? 'concave' : 'flat',
      flipMismatch: j.flippedA !== j.flippedB,
      stationCount: mine.length,
      flags: [...flags].sort(),
    }
  })

  // --- collisions -----------------------------------------------------------
  // Adjacent panels are excluded: their overlap is the JOINT, and
  // `sectionFouling` judges it exactly (W_PANELS_COLLIDE_AT_JOINT) where an OBB
  // pair cannot. On the ribbon "adjacent" was consecutive unit numbers; on the
  // network it is JOINED BY A JOINT, read off the joint table. That is the same
  // rule stated more directly, and it keeps the property the ribbon relied on:
  // two panels either side of a hole are still checked, because nothing connects
  // them.
  const present = C.panels.filter((p) => p.present)
  const joined = new Set(C.joints.map((j) => (j.a < j.b ? `${j.a}|${j.b}` : `${j.b}|${j.a}`)))
  const boxes = present.map((p) => p.obb)
  const hits = findCollisions(boxes, { minDepthCm: COLLISION_MIN_DEPTH_CM })

  // CORNER PAIRS ARE SPLIT OUT, and this is a judgement rather than a filter.
  //
  // Two ramps off the same cell meet at a lattice corner box-corner to
  // box-corner. There the panel is 1.2cm of outer wall, not the 4.1cm the box
  // claims — the OBB overstates the solid by the whole depth of the taper
  // exactly where these two touch. Measured on a back-plate-only box the same
  // pair is 9.4cm apart at 30°.
  //
  // So calling this a COLLISION would be asserting something the primitive
  // cannot support, and it is the loud kind of wrong: 16 red pairs on a design
  // whose panels are nowhere near each other. Calling it nothing would be the
  // quiet kind. It gets its own list and its own number
  // (`metrics.cornerClearance`, `W_CORNER_RAMPS_MEET`) and stays out of the
  // headline count until there is a section-level test that can settle it —
  // §9.6, and the one real gap this pass leaves open.
  const cornerPairs = new Set(
    (solveCornerClearance(C).pairs ?? []).map(({ a, b }) => (a < b ? `${a}|${b}` : `${b}|${a}`)),
  )
  const unjoined = hits.filter((h) => {
    const A = present[h.i].id
    const B = present[h.j].id
    return !joined.has(A < B ? `${A}|${B}` : `${B}|${A}`)
  })
  const asPair = (h) => ({ a: present[h.i].id, b: present[h.j].id, depthCm: r(h.depthCm) })
  const byDepth = (x, y) => y.depthCm - x.depthCm || (x.a < y.a ? -1 : 1)
  const isCorner = (h) => {
    const A = present[h.i].id
    const B = present[h.j].id
    return cornerPairs.has(A < B ? `${A}|${B}` : `${B}|${A}`)
  }
  const collisions = unjoined.filter((h) => !isCorner(h)).map(asPair).sort(byDepth)
  const cornerContacts = unjoined.filter(isCorner).map(asPair).sort(byDepth)

  for (const p of present) {
    // The solid's corners ARE the OBB's corners — the panel envelope is a box —
    // so the floor and wall tests read them directly rather than reconstructing
    // the box from its centre and quaternion.
    let yMin = Infinity
    let xMin = Infinity
    for (const c of p.corners) {
      if (c[1] < yMin) yMin = c[1]
      if (c[0] < xMin) xMin = c[0]
    }
    if (yMin < -PLANE_EPSILON) {
      warnings.push({
        code: 'W_BELOW_FLOOR',
        panel: p.id,
        yMinCm: r(yMin),
        message: `${p.id} reaches ${r(-yMin)}cm below the floor — only reachable with grounding off`,
      })
    }
    if (xMin < -PLANE_EPSILON) {
      warnings.push({
        code: 'W_THROUGH_WALL',
        panel: p.id,
        xMinCm: r(xMin),
        message: `${p.id} reaches ${r(-xMin)}cm through the wall plane at x = 0`,
      })
    }
  }

  // --- metrics --------------------------------------------------------------
  const metrics = buildMetrics(cfg, C)

  // --- envelope -------------------------------------------------------------
  const envelope = solveEnvelope(cfg)
  if (!envelope.clean) {
    warnings.push({
      code: 'W_OUTSIDE_ENVELOPE',
      message:
        envelope.maxAngleDeg === null
          ? `this design is outside the connector envelope at every angle: ${envelope.limitedBy.join(', ')}`
          : `this design is outside the connector envelope — ${envelope.limitedBy.join(', ')}; the ` +
            `largest angle this ${cfg.gap}cm gap admits is ${envelope.maxAngleDeg}°`,
      limitedBy: envelope.limitedBy,
    })
  }

  // --- the front bar, per valley --------------------------------------------
  // Named per joint rather than once for the design, because the fix is likely
  // to be per station (a relief, or a narrower bar on concave joints) and the
  // list of which joints need it is the actionable part.
  for (const j of joints) {
    if (j.sense !== 'concave') continue
    if (j.dihedralDeg <= envelope.frontBar.concaveLimitDeg) continue
    j.flags = [...new Set([...j.flags, FRONT_BAR_CODE])].sort()
    warnings.push({
      code: FRONT_BAR_CODE,
      joint: j.jointIndex,
      a: j.a,
      b: j.b,
      dihedralDeg: j.dihedralDeg,
      limitDeg: envelope.frontBar.concaveLimitDeg,
      message:
        `joint ${j.a}–${j.b} is a ${j.dihedralDeg}° valley; a flat front bar sized for a ` +
        `${cfg.gap}cm gap bites the bezels past ${envelope.frontBar.concaveLimitDeg.toFixed(2)}°. The bar ` +
        `needs a relief or a narrower section here — widening the gap does not help`,
    })
  }

  // --- the corner holes -----------------------------------------------------
  // Reported as a number, and flagged only when two BOXES meet. See the file
  // header for why that is not the same as two panels meeting: the boxes touch
  // at their corners, where the real section is thin, and the back plates are
  // still metres apart. A positive clearance is a guarantee; a negative one is a
  // question the model cannot currently answer.
  if (metrics.cornerClearance.touchingPairs > 0) {
    const worst = metrics.cornerClearance.pairs[0]
    warnings.push({
      code: 'W_CORNER_RAMPS_MEET',
      a: worst.a,
      b: worst.b,
      cell: worst.cell,
      clearanceCm: worst.clearanceCm,
      count: metrics.cornerClearance.touchingPairs,
      message:
        `${metrics.cornerClearance.touchingPairs} pair(s) of ramps off the same cell no longer clear ` +
        `each other at the lattice corner — worst is ${worst.a} against ${worst.b} at cell ` +
        `${worst.cell} (${worst.clearanceCm}cm). This is a BOUNDING-BOX verdict at the panels' ` +
        'corners, where the real section is only the outer wall — the housings proper are still well ' +
        'clear. It is the 2-D limit on θ, it has no ribbon analogue, and unlike the front bar the gap ' +
        'buys it back: widening the joint moves this limit directly',
    })
  }

  // --- obstacles ------------------------------------------------------------
  // Room facts, not design (obstacles.js). Reported and never enforced: a panel
  // through the column is named so it can be switched off in the plan editor,
  // which is the same "report the cost, do not veto" contract the connector
  // flags follow.
  const obstacles = solveObstacles(C, cfg.obstacles)
  for (const o of obstacles) {
    if (!o.hitCount) continue
    warnings.push({
      code: OBSTACLE_HIT_CODE,
      obstacle: o.id,
      count: o.hitCount,
      panels: o.hits.map((h) => h.id),
      message:
        `${o.hitCount} panel${o.hitCount === 1 ? '' : 's'} run through the ${o.label} at ` +
        `x ${o.extents.min[0]}–${o.extents.max[0]}, z ${o.extents.min[2]}–${o.extents.max[2]}cm ` +
        `(${o.hits.map((h) => h.id).join(', ')}) — switch them off in the plan, or move the network`,
    })
  }

  // --- the ground spacers ---------------------------------------------------
  // The posts that hold the flat cells off the floor (spacers.js). Two numbers
  // that are worth stating separately: the CLEARANCE asked for, and the heights
  // the posts actually came out. They are computed in different modules from
  // different quantities — `lattice.js` translates the network, `spacers.js`
  // measures the underside it landed at — so a disagreement between them is a
  // real bug in one of the two and not a rounding question. There should be
  // exactly ONE distinct height on any grounded design.
  const spacerReport = {
    grounded: S.grounded,
    clearanceCm: S.clearanceCm,
    sectionCm: S.sectionCm,
    perEdge: S.perEdge,
    count: S.spacers.length,
    cellCount: S.perCell.length,
    perCell: S.perCell,
    heightsCm: S.heightsCm,
  }
  const offBy = S.spacers.filter((sp) => Math.abs(sp.heightCm - S.clearanceCm) > 1e-6)
  if (offBy.length > 0) {
    const worst = offBy.reduce(
      (a, b) => (Math.abs(b.heightCm - S.clearanceCm) > Math.abs(a.heightCm - S.clearanceCm) ? b : a),
    )
    warnings.push({
      code: SPACER_MISMATCH_CODE,
      count: offBy.length,
      clearanceCm: S.clearanceCm,
      worstHeightCm: worst.heightCm,
      spacer: worst.id,
      cell: worst.cell,
      message:
        `${offBy.length} ground spacer(s) do not span the ${S.clearanceCm}cm clearance — worst is ` +
        `${worst.id} under ${worst.cell} at ${worst.heightCm}cm. Grounding puts the network's LOWEST ` +
        'material at the clearance, so this means the lowest flat cells are not the lowest thing in ' +
        'the design (a ramp toe or an anchor is below them) and the posts under those cells are the ' +
        'wrong length for the gap they are meant to hold open',
    })
  }

  return { joints, envelope, collisions, cornerContacts, obstacles, spacers: spacerReport, metrics, warnings }
}

/**
 * The measuring box, with the two-dimensional breakdown it hides.
 *
 * Bounds are over PRESENT panels' solids only — an absent panel is a hole in the
 * network and must not inflate its box.
 *
 * A COLUMN `i` is its cells, the z-axis ramps that run between them (which lie
 * inside the column's own x band), and any wall anchors at i = 0. A ROW `j` is
 * its cells, the x-axis ramps between them, and the same anchors. The x-axis
 * ramps are deliberately absent from the column figures and the z-axis ones from
 * the row figures: a ramp that crosses BETWEEN two columns belongs to neither,
 * and counting it in both would make the columns' extents overlap and stop
 * meaning "how long is this column".
 */
function buildMetrics(cfg, C) {
  const present = C.panels.filter((p) => p.present)
  const { cols, rows } = cfg.lattice

  const extentOf = (panels) => {
    const min = [Infinity, Infinity, Infinity]
    const max = [-Infinity, -Infinity, -Infinity]
    for (const p of panels) {
      for (const c of p.corners) {
        for (let k = 0; k < 3; k++) {
          if (c[k] < min[k]) min[k] = c[k]
          if (c[k] > max[k]) max[k] = c[k]
        }
      }
    }
    return Number.isFinite(min[0]) ? { min, max } : null
  }

  const box = (panels) => {
    const ext = extentOf(panels)
    return {
      xMinCm: r(ext ? ext.min[0] : 0),
      xMaxCm: r(ext ? ext.max[0] : 0),
      widthCm: r(ext ? ext.max[0] - ext.min[0] : 0),
      yMinCm: r(ext ? ext.min[1] : 0),
      yMaxCm: r(ext ? ext.max[1] : 0),
      heightCm: r(ext ? ext.max[1] - ext.min[1] : 0),
      zMinCm: r(ext ? ext.min[2] : 0),
      zMaxCm: r(ext ? ext.max[2] : 0),
      planRunCm: r(ext ? ext.max[2] - ext.min[2] : 0),
    }
  }

  const columns = []
  for (let i = 0; i < cols; i++) {
    const mine = present.filter((p) =>
      (p.kind === 'cell' && p.i === i) ||
      (p.kind === 'ramp' && !p.anchor && p.axis === 'z' && p.i === i) ||
      (p.anchor && i === 0))
    columns.push({
      i,
      cellSlots: rows,
      presentCells: mine.filter((p) => p.kind === 'cell').length,
      panelCount: mine.length,
      ...box(mine),
    })
  }

  const rowsOut = []
  for (let j = 0; j < rows; j++) {
    const mine = present.filter((p) =>
      (p.kind === 'cell' && p.j === j) ||
      (p.kind === 'ramp' && !p.anchor && p.axis === 'x' && p.j === j) ||
      (p.anchor && p.j === j))
    rowsOut.push({
      j,
      cellSlots: cols,
      presentCells: mine.filter((p) => p.kind === 'cell').length,
      panelCount: mine.length,
      ...box(mine),
    })
  }

  const cells = present.filter((p) => p.kind === 'cell')
  const ramps = present.filter((p) => p.kind === 'ramp' && !p.anchor)
  const anchors = present.filter((p) => p.anchor)
  const counts = {
    panels: present.length,
    slots: C.panels.length,
    absent: C.panels.length - present.length,
    cells: cells.length,
    groundCells: cells.filter((p) => p.role === 'ground').length,
    highCells: cells.filter((p) => p.role === 'high').length,
    ramps: ramps.length,
    rampsRising: ramps.filter((p) => p.role === 'rise').length,
    rampsFalling: ramps.filter((p) => p.role === 'fall').length,
    anchorRamps: anchors.length,
    joints: C.joints.length,
  }

  const L = C.lattice.lengthCm
  const W = C.lattice.widthCm
  // The 2-D analogue of the ribbon's compression ratio: how much sheet is
  // standing up rather than lying down. 1.0 would be a flat floor of panels.
  const materialAreaCm2 = counts.panels * L * W
  const planAreaCm2 = C.bounds.size[0] * C.bounds.size[2]
  const panels = C.panels.map((p) => ({
    id: p.id,
    kind: p.kind,
    i: p.i,
    j: p.j,
    axis: p.axis,
    anchor: p.anchor,
    level: p.level,
    // Wave only — carried through rather than recomputed, so the table and the
    // 3-D view can never quote two different heights for one panel.
    ...(p.heightCm !== undefined ? { heightCm: p.heightCm } : {}),
    role: p.role,
    glyph: p.glyph,
    tiltDeg: p.tiltDeg,
    present: p.present,
    flipped: p.flipped,
    // Taken from the REFERENCE segment, not from the solid: these describe what
    // the panel does, and adding half a housing thickness to them would make the
    // rise disagree with the tilt it is supposed to illustrate.
    xStartCm: p.refStart[0],
    xEndCm: p.refEnd[0],
    zStartCm: p.refStart[2],
    zEndCm: p.refEnd[2],
    planRunCm: r(Math.hypot(p.refEnd[0] - p.refStart[0], p.refEnd[2] - p.refStart[2])),
    yStartCm: p.refStart[1],
    yEndCm: p.refEnd[1],
    riseCm: r(p.refEnd[1] - p.refStart[1]),
  }))

  return {
    overall: C.bounds,
    lattice: {
      cols,
      rows,
      phase: C.lattice.phase,
      wallAnchor: C.lattice.wallAnchor,
      pitchCm: C.lattice.pitchCm,
      riseCm: C.lattice.riseCm,
      stepPlanCm: C.lattice.stepPlanCm,
      stepRiseCm: C.lattice.stepRiseCm,
      levels: C.lattice.levels,
      shiftCm: C.lattice.shiftCm,
      // Wave only. `pitchCm` / `riseCm` above are the BASE angle's numbers under
      // the wave — what edge 0 does — and this is what every other edge does.
      ...(C.lattice.wave ? { kind: 'wave', wave: C.lattice.wave } : {}),
    },
    counts,
    material: {
      materialAreaCm2: r(materialAreaCm2),
      planAreaCm2: r(planAreaCm2),
      coverageRatio: materialAreaCm2 > 0 ? r(planAreaCm2 / materialAreaCm2) : null,
    },
    columns,
    rows: rowsOut,
    cornerClearance: solveCornerClearance(C),
    panels,
  }
}
