/**
 * grid-designer v4 — the network: a checkerboard of flat cells, with the
 * angled panels between them derived.
 *
 * HEADLESS ZONE (src/core/): pure functions, importable from plain node.
 *   - explicit `.js` extensions on ALL relative imports
 *   - may import `three` math classes only; never components / store / DOM
 *   - same config in → byte-identical output out
 *
 * =============================================================================
 * WHAT THIS REPLACES, AND WHAT IT KEEPS
 * =============================================================================
 * This file replaces `chain.js`, and the replacement is a GENERALISATION rather
 * than a rewrite: at `cols: 1` it reproduces the shipped ribbon panel for panel
 * (V4_SPEC §9's opening claim, and `tests/test-v4-lattice.mjs` §1 asserts it
 * against the ribbon's own hardcoded numbers). Every joint, envelope, flip and
 * connector rule of §1–§8 carries over untouched. What changes is the UNIT OF
 * DESIGN: you no longer author a sequence of panels, you switch flat cells on
 * and off and the angled panels between them follow.
 *
 * =============================================================================
 * WHY THE LEVEL FIELD IS A CHECKERBOARD, AND WHY THAT IS NOT A STYLE CHOICE
 * =============================================================================
 * The brief is: a flat panel on the ground may add an angled panel UP at any
 * face, and an angled panel is followed by a flat; a flat above the ground may
 * add an angled panel DOWN at any face, likewise. Those rules admit exactly two
 * levels and make every flat cell's four neighbours the opposite level, so
 *
 *     level(i, j) = (i + j + phase) mod 2          0 = ground, 1 = high
 *
 * and every edge between two flat cells carries exactly one angled panel.
 *
 * The alternative — allowing two same-level flats to abut — would put a
 * `gap`-wide edge and a ~56cm edge on the same lattice, and the pitch would stop
 * being uniform unless every level change ran in an unbroken line across the
 * whole grid. That is HANDOFF §3's planar-quad trap arriving by a different
 * door, and it is the thing this model exists not to walk into.
 *
 * =============================================================================
 * WHY IT CLOSES EXACTLY — the one property everything else rests on
 * =============================================================================
 * §2's bisector step collapses to a half-angle here. For a joint between a
 * horizontal panel and one tilted by θ,
 *
 *     normalize(u_flat + u_tilt) = ( cos(θ/2), sin(θ/2) )   in (plan, y)
 *
 * because 1 + cos θ = 2cos²(θ/2), sin θ = 2 sin(θ/2) cos(θ/2), and the norm is
 * 2cos(θ/2). So EVERY level change costs the same plan distance and the same
 * rise, in x and in z alike:
 *
 *     gap step      plan  gap·cos(θ/2)      rise  gap·sin(θ/2)
 *     level rise    R = 2·gap·sin(θ/2) + 60·sin θ
 *     CELL PITCH    P = 60 + 2·gap·cos(θ/2) + 60·cos θ      identical in x and z
 *
 * The flat cells therefore sit on a UNIFORM SQUARE LATTICE and every cycle
 * closes with zero residual — which is why the network needs no solver, exactly
 * as the ribbon needed none. `latticeStep` below is the single place those two
 * numbers are computed, and the suite checks a ramp's far end against its high
 * cell's edge from the ramp's OWN geometry rather than trusting the identity.
 *
 * This is also the reason the whole file is written in the half-angle form
 * rather than by re-running the bisector: the bisector is what PROVES the step,
 * and the step is what the geometry is built from. The two are asserted equal in
 * the tests, which is the honest way to keep a deliberate restatement from
 * drifting (the same discipline schema.js applies to the gap band).
 *
 * =============================================================================
 * HANDEDNESS — the failure this file is most likely to have
 * =============================================================================
 * A ramp's width direction is FIXED by right-handedness, not chosen:
 *
 *     u = cos θ · ê + sin θ · Ŷ      the chain direction, rising
 *     n = −sin θ · ê + cos θ · Ŷ     the lit normal
 *     w = Ŷ × ê                      the WIDTH direction,  w × n = u
 *
 * `ê = +Z` gives `w = +X`, which is exactly the ribbon; `ê = +X` gives `w = −Z`.
 * Taking `+Z` there instead yields `−u` and a LEFT-HANDED basis — invisible in
 * the corner geometry of a square panel, and it silently corrupts every
 * collision box. That is precisely what happened to v3's plates (HANDOFF §6) and
 * made a perfectly flat grid report 21 interpenetrating pairs, so the suite
 * asserts `X × Y = Z` on every panel in all four ramp directions.
 *
 * The quaternion is built as `R_y(φ_ê) · R_x(−θ)` (φ_ê the yaw taking +Z to ê)
 * rather than from a basis matrix. Same rotation, and it makes the ribbon case
 * BIT-identical: at φ = 0 the composition with the identity yaw is exact, so the
 * `cols: 1` lattice hands back the very quaternion `chain.js` used to.
 *
 * =============================================================================
 * ONE PLACE THE LATTICE HONESTLY DISAGREES WITH THE RIBBON
 * =============================================================================
 * `ê` carries the direction, so there is no separate "fall" case — a ramp is
 * always defined RISING from its ground cell. On a descending ramp that makes
 * `w = −X` where the ribbon used `+X`: the same physical panel, turned 180° in
 * its own plane. Position, lit normal, solid corners and OBB are identical to
 * the ribbon's; the quaternion differs by that half turn about the normal. The
 * consequence downstream is that the two panels at a joint can number their
 * width axes in opposite senses, which is why `twistDeg` is measured between the
 * two rim LINES rather than between two rays — see `connectors.js`.
 *
 * =============================================================================
 * OUTPUT
 * =============================================================================
 *   solveLattice(config) → {
 *     config,                    the normalized config it was solved from
 *     lattice: { cols, rows, phase, panelType, widthCm, lengthCm,
 *                pitchCm, riseCm, stepPlanCm, stepRiseCm,
 *                levels, shiftCm },
 *     panels: [{ id, kind, i, j, axis, anchor, role, glyph, level, tiltDeg,
 *                present, flipped, panelType, widthCm, lengthCm,
 *                position, quaternion, normal, refStart, refEnd, corners, obb }],
 *     joints: [{ id, jointIndex, a, b, panelA, panelB, edge, end, axis,
 *                lengthCm, runAxis, run, runFrom, runTo, runA, runB,
 *                rimA, rimB, normalA, normalB, inwardA, inwardB,
 *                flippedA, flippedB, tiltADeg, tiltBDeg }],
 *     bounds: { min, max, size },
 *   }
 *
 * `i` runs along x FROM THE WALL, `j` along z FROM THE WINDOW, both 0-based.
 * An edge is named by its LOW-INDEX cell and its axis: `(i, j, 'x')` joins
 * `(i,j)`–`(i+1,j)`, `(i, j, 'z')` joins `(i,j)`–`(i,j+1)`. That is NOT the same
 * as the edge's ground cell, which is whichever of the two the checkerboard puts
 * at level 0 — see `edgeGeometry` below.
 */

import * as THREE from 'three'
import { normalizeConfig } from './schema.js'
import { PANEL_DIMENSIONS, PANEL_PROFILE } from '../../config.js'

const RAD = Math.PI / 180

// -----------------------------------------------------------------------------
// Determinism helpers — carried over from chain.js unchanged, including the
// 1e-9 / 1e-12 split.
//
// Trig produces 3.7000000000000004 and −0 depending on which branch generated
// them; both break deepStrictEqual / JSON.stringify comparisons without changing
// the geometry. This rounding IS the determinism guarantee.
// -----------------------------------------------------------------------------
function r(v) {
  const out = Math.round(v * 1e9) / 1e9
  return out === 0 ? 0 : out
}

const rv = (v) => [r(v.x), r(v.y), r(v.z)]

/**
 * The same device at 1e-12, for the JOINT record only.
 *
 * A joint's rim points and normals are not display values — they are the inputs
 * to `spanMaxCm > maxSpanCm` and to `atan2`, and at 1e-9 the rounding is
 * measurable in both. HANDOFF §8.5 records the two cases that showed up: a
 * picometre of rim-point error made an 8.0cm gap measure 8.000000001 and trip
 * `W_CONNECTOR_SPAN`, and rounding cos θ at 1e-9 capped any angle derived from
 * it at ~3e-8°. 1e-12 puts both four orders of magnitude below anything that
 * could matter while keeping the property the rounding exists for.
 */
function rFine(v) {
  const out = Math.round(v * 1e12) / 1e12
  return out === 0 ? 0 : out
}

const rvFine = (v) => [rFine(v.x), rFine(v.y), rFine(v.z)]

// =============================================================================
// THE LEVEL FIELD
// =============================================================================
/** Ground cells sit on the floor; high cells sit one `riseCm` above it. */
export const CELL_ROLES = ['ground', 'high']
/** A ramp is named for what it does as i or j INCREASES — a readout only. */
export const RAMP_ROLES = ['rise', 'fall']

/** How each role is drawn in a profile through the network: `_ / - \`. */
export const ROLE_GLYPH = { ground: '_', high: '-', rise: '/', fall: '\\' }

/**
 * `level(i, j) = (i + j + phase) mod 2`. 0 is ground, 1 is high.
 *
 * The double modulo is not superstition: `phase` is clamped non-negative by the
 * schema but `i` reaches −1 for the wall anchor's virtual ground cell, and
 * JavaScript's `%` keeps the sign of the dividend.
 */
export function levelAt(i, j, phase = 0) {
  return (((i + j + phase) % 2) + 2) % 2
}

/**
 * The lattice's two defining lengths, from the gap and the angle alone.
 *
 * `pitchCm` is identical in x and z, which is the whole content of "the flat
 * cells sit on a uniform square lattice". Exported because the report quotes it
 * and the tests check it against the closed form independently.
 */
export function latticeStep({ gapCm, angleDeg, lengthCm }) {
  const theta = angleDeg * RAD
  const half = theta / 2
  const stepPlanCm = gapCm * Math.cos(half)
  const stepRiseCm = gapCm * Math.sin(half)
  return {
    stepPlanCm,
    stepRiseCm,
    pitchCm: lengthCm + 2 * stepPlanCm + lengthCm * Math.cos(theta),
    riseCm: 2 * stepRiseCm + lengthCm * Math.sin(theta),
  }
}

// -----------------------------------------------------------------------------
// The four plan directions a ramp can run in, and the yaw that reaches each.
//
// `yaw` takes +Z to `dir`, so `R_y(yaw) · R_x(−θ)` is the panel's rotation and
// the +Z entry (yaw 0) is the ribbon's own quaternion, exactly.
// -----------------------------------------------------------------------------
const PLAN_DIRS = {
  '+z': { vec: [0, 0, 1], yaw: 0 },
  '+x': { vec: [1, 0, 0], yaw: Math.PI / 2 },
  '-z': { vec: [0, 0, -1], yaw: Math.PI },
  '-x': { vec: [-1, 0, 0], yaw: -Math.PI / 2 },
}

/** Which world axis a direction lies on: 0 for x, 2 for z. */
const AXIS_INDEX = { x: 0, z: 2 }

const dirKey = (axis, sign) => `${sign > 0 ? '+' : '-'}${axis}`

// =============================================================================
// SOLVE
// =============================================================================
/**
 * Build the whole network and place it in the world.
 *
 * @param {object} config raw or normalized v4 config
 */
export function solveLattice(config) {
  const cfg = normalizeConfig(config)
  const dims = PANEL_DIMENSIONS[cfg.lattice.panelType]
  const W = dims.width     // across the panel
  const L = dims.height    // along the panel, and along the lattice
  // A rectangular panel would give a rectangular lattice — the pitch is `L` in
  // BOTH axes below, which is only the same object when the panel is square.
  // Plates are V4_SPEC §9.8, deliberately not in this pass, and the schema only
  // admits '2x2'; this is the tripwire for the day that changes.
  if (W !== L) {
    throw new Error(
      `solveLattice: panel type ${JSON.stringify(cfg.lattice.panelType)} is ${W}×${L} — the lattice ` +
      'pitch is square by construction (V4_SPEC §9.1) and a rectangular panel needs a rectangular ' +
      'lattice, which is §9.8 and is not built',
    )
  }

  const { cols, rows } = cfg.lattice
  const phase = cfg.pattern.phase
  const gap = cfg.gap
  const theta = cfg.angleDeg * RAD
  const T = PANEL_PROFILE.overallThickness
  const step = latticeStep({ gapCm: gap, angleDeg: cfg.angleDeg, lengthCm: L })
  const { pitchCm: P, riseCm: R, stepPlanCm, stepRiseCm } = step

  const Y = new THREE.Vector3(0, 1, 0)
  const AX = new THREE.Vector3(1, 0, 0)

  // --- overrides, indexed --------------------------------------------------
  const cellOv = new Map()
  for (const ov of cfg.overrides.cells) cellOv.set(`${ov.i},${ov.j}`, ov)
  const edgeOv = new Map()
  for (const ov of cfg.overrides.edges) edgeOv.set(`${ov.i},${ov.j},${ov.axis}`, ov)

  const levels = []
  const cellPresent = []
  const cellFlipped = []
  for (let i = 0; i < cols; i++) {
    levels.push([])
    cellPresent.push([])
    cellFlipped.push([])
    for (let j = 0; j < rows; j++) {
      const ov = cellOv.get(`${i},${j}`)
      levels[i].push(levelAt(i, j, phase))
      cellPresent[i].push(ov ? ov.present : true)
      cellFlipped[i].push(ov ? ov.flipped : false)
    }
  }

  /** The plan centre of cell (i, j), at its own reference height. */
  const cellCentre = (i, j) =>
    new THREE.Vector3(L / 2 + i * P, levelAt(i, j, phase) === 1 ? R : 0, L / 2 + j * P)

  // ---------------------------------------------------------------------------
  // A panel, built from its reference segment and its frame.
  //
  // Everything below — corners, OBB, normal — is derived from the SEGMENT and
  // the quaternion, never from the local axes twice over, so a flip cannot be
  // applied to the same thing in two places. That is chain.js's rule and the
  // reason its flip test is a one-liner.
  // ---------------------------------------------------------------------------
  const built = []
  function push(spec) {
    const { refStart, refEnd, w, yaw, tiltMagnitude, flipped } = spec
    // A rotation about world +Y to aim the panel, then about its own +X to tilt
    // it. Flipping is a further half turn about that same +X — the panel stays
    // in its own plane and keeps its footprint (it is centred and square); what
    // changes is which way the lit face points and which side the housing
    // extrudes to. See the file header for why this form rather than a basis.
    const quat = new THREE.Quaternion()
      .setFromAxisAngle(Y, yaw)
      .multiply(new THREE.Quaternion().setFromAxisAngle(
        AX, flipped ? Math.PI - tiltMagnitude : -tiltMagnitude,
      ))
    // The LIT normal: local +Y through the placement, so it is already negated
    // for a flipped panel and no caller has to remember to do it.
    const normal = new THREE.Vector3(0, 1, 0).applyQuaternion(quat)
    const centre = new THREE.Vector3().addVectors(refStart, refEnd).multiplyScalar(0.5)

    // The 8 corners of the solid: 4 on the reference plane and those 4 pushed
    // `overallThickness` along −n̂.
    const along = new THREE.Vector3().subVectors(refEnd, refStart).multiplyScalar(0.5)
    const across = w.clone().multiplyScalar(W / 2)
    const corners = []
    for (const depth of [0, -T]) {
      for (const [sa, sb] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
        corners.push(
          centre.clone().addScaledVector(across, sa).addScaledVector(along, sb)
            .addScaledVector(normal, depth),
        )
      }
    }
    built.push({ ...spec, quat, normal, centre, corners })
    return built[built.length - 1]
  }

  // --- the flat cells ------------------------------------------------------
  // Every cell of the grid is emitted, present or not: `present: false` opens a
  // hole, it does not re-solve the network. That is §2's removal rule, and here
  // it is free rather than earned — the lattice is GENERATED from (i, j), so
  // nothing downstream of a removed cell could move even if it wanted to.
  for (let i = 0; i < cols; i++) {
    for (let j = 0; j < rows; j++) {
      const c = cellCentre(i, j)
      const level = levels[i][j]
      push({
        id: `Ci${i}j${j}`,
        kind: 'cell',
        i,
        j,
        axis: null,
        anchor: false,
        level,
        role: level === 1 ? 'high' : 'ground',
        tiltDeg: 0,
        present: cellPresent[i][j],
        flipped: cellFlipped[i][j],
        refStart: c.clone().add(new THREE.Vector3(0, 0, -L / 2)),
        refEnd: c.clone().add(new THREE.Vector3(0, 0, L / 2)),
        w: AX.clone(),
        yaw: 0,
        tiltMagnitude: 0,
        // Filled for ramps only; a cell is joined by whatever ramps reach it.
        lo: null,
        hi: null,
        e: null,
      })
    }
  }

  /**
   * Everything an edge's ramp needs, from the edge's name.
   *
   * The edge is NAMED by its low-INDEX cell; the ramp is DEFINED from its
   * low-LEVEL cell, and those are different cells half the time. `ê` runs from
   * the ground cell toward the high one, which is what removes the "fall" case
   * (V4_SPEC §9.3).
   */
  function edgeGeometry(i, j, axis) {
    const a = { i, j }
    const b = axis === 'x' ? { i: i + 1, j } : { i, j: j + 1 }
    const aIsGround = levelAt(a.i, a.j, phase) === 0
    const lo = aIsGround ? a : b
    const hi = aIsGround ? b : a
    // ê points from the ground cell to the high one, so it is +axis when the
    // low-index cell is the ground one and −axis otherwise.
    const sign = aIsGround ? 1 : -1
    const dir = PLAN_DIRS[dirKey(axis, sign)]
    const e = new THREE.Vector3(...dir.vec)
    const u = e.clone().multiplyScalar(Math.cos(theta)).addScaledVector(Y, Math.sin(theta))
    const w = new THREE.Vector3().crossVectors(Y, e)
    return { lo, hi, e, u, w, yaw: dir.yaw, sign }
  }

  /** The reference segment of the ramp rising out of `loCentre` along `ê`. */
  function rampSegment(loCentre, e, u) {
    const refStart = loCentre.clone()
      .addScaledVector(e, L / 2)
      .addScaledVector(e, stepPlanCm)
      .addScaledVector(Y, stepRiseCm)
    return { refStart, refEnd: refStart.clone().addScaledVector(u, L) }
  }

  // --- the ramps, one per lattice edge -------------------------------------
  // Emitted for EVERY in-grid edge, so the table is complete and an edge's
  // geometry does not depend on whether it is switched on.
  //
  // A RAMP NEEDS ONE CELL, NOT TWO. The first cut of this required both, which
  // made removing a flat silently take up to four angled panels with it — so
  // the design could not be edited panel by panel, which is the whole point of
  // the plan editor. It also contradicted the model's own precedent: a WALL
  // ANCHOR (§9.4) is exactly a ramp with nothing at its far end, and it is a
  // perfectly good panel.
  //
  // So a ramp exists when it is switched on and AT LEAST ONE of its cells is
  // there. It then cantilevers off its single joint, which is what an
  // open-ended folded surface is made of. With NEITHER cell it would float, so
  // that case is still absent — the one rule the geometry actually requires.
  for (const axis of ['x', 'z']) {
    const iMax = axis === 'x' ? cols - 1 : cols
    const jMax = axis === 'z' ? rows - 1 : rows
    for (let i = 0; i < iMax; i++) {
      for (let j = 0; j < jMax; j++) {
        const { lo, hi, e, u, w, yaw, sign } = edgeGeometry(i, j, axis)
        const { refStart, refEnd } = rampSegment(cellCentre(lo.i, lo.j), e, u)
        const ov = edgeOv.get(`${i},${j},${axis}`)
        const cellsPresent = cellPresent[lo.i][lo.j] || cellPresent[hi.i][hi.j]
        push({
          id: `Ei${i}j${j}${axis}`,
          kind: 'ramp',
          i,
          j,
          axis,
          anchor: false,
          level: null,
          // `sign` is the readout: a ramp that gains height as the index grows
          // reads as a rise, one that loses it as a fall. The GEOMETRY does not
          // distinguish them — both are built rising out of their ground cell.
          role: sign > 0 ? 'rise' : 'fall',
          tiltDeg: sign * cfg.angleDeg,
          present: cellsPresent && (ov ? ov.present : true),
          flipped: false,
          refStart,
          refEnd,
          w,
          yaw,
          tiltMagnitude: theta,
          lo,
          hi,
          e,
        })
      }
    }
  }

  // --- the wall anchor -----------------------------------------------------
  // One extra ramp per present HIGH cell in column i = 0, running down to ground
  // level on the wall side so the network is propped against the wall line
  // rather than cantilevered off its own edge (V4_SPEC §9.4). Ground cells at
  // i = 0 need nothing — they are already on the floor.
  //
  // It is an ordinary ramp on the edge that WOULD join a virtual ground cell at
  // i = −1, so `ê = +X`: §9.3 defines ê from the ground cell toward the high
  // one, and here the ground cell is the virtual one on the wall side. (§9.4
  // says "ê = −X", which cannot be right under §9.3's own definition — taken
  // literally it builds a ramp two pitches out from the wall that touches
  // nothing. The description "descending in −x" is what the geometry does.)
  //
  // What these attach to at the wall is NOT modelled and is an open hardware
  // question. The geometry says where the toe lands, nothing more.
  if (cfg.placement.wallAnchor === 'braced') {
    for (let j = 0; j < rows; j++) {
      if (levels[0][j] !== 1) continue
      const e = new THREE.Vector3(1, 0, 0)
      const u = e.clone().multiplyScalar(Math.cos(theta)).addScaledVector(Y, Math.sin(theta))
      const w = new THREE.Vector3().crossVectors(Y, e)
      // The virtual ground cell: one pitch back from the high cell, on the floor.
      const virtualLo = cellCentre(0, j).clone().setX(L / 2 - P).setY(0)
      const { refStart, refEnd } = rampSegment(virtualLo, e, u)
      push({
        id: `Aj${j}`,
        kind: 'ramp',
        i: -1,
        j,
        axis: 'x',
        anchor: true,
        level: null,
        role: 'rise',
        tiltDeg: cfg.angleDeg,
        present: cellPresent[0][j],
        flipped: false,
        refStart,
        refEnd,
        w,
        yaw: PLAN_DIRS['+x'].yaw,
        tiltMagnitude: theta,
        lo: null, // there is no cell here — that is what makes it an anchor
        hi: { i: 0, j },
        e,
      })
    }
  }

  // --- grounding and the offsets ------------------------------------------
  // All three are a rigid translation of the FINISHED network, taken over the
  // material of PRESENT panels only:
  //
  //   y   `groundToFloor` puts the lowest solid point on the floor
  //   x   `wallOffsetCm` is what a tape measure reads from the wall plane to the
  //       nearest material — INCLUDING an anchor ramp's toe, which is what makes
  //       the number mean the same thing braced and free (§9.5)
  //   z   `windowOffsetCm`, the same in z
  //
  // Taken over present panels, so removing the panel that happens to be lowest
  // or nearest legitimately moves everything else. That is the one case in which
  // editing is not local, it is inherited from the ribbon's grounding, and it is
  // the price of the offsets meaning what a tape measure reads.
  const mins = [Infinity, Infinity, Infinity]
  for (const p of built) {
    if (!p.present) continue
    for (const c of p.corners) {
      if (c.x < mins[0]) mins[0] = c.x
      if (c.y < mins[1]) mins[1] = c.y
      if (c.z < mins[2]) mins[2] = c.z
    }
  }
  const anyPresent = Number.isFinite(mins[0])
  const shift = new THREE.Vector3(
    anyPresent ? cfg.placement.wallOffsetCm - mins[0] : 0,
    anyPresent && cfg.placement.groundToFloor ? -mins[1] : 0,
    anyPresent ? cfg.placement.windowOffsetCm - mins[2] : 0,
  )

  // --- emit ----------------------------------------------------------------
  const panels = []
  const exact = new Map()
  for (const p of built) {
    const refStart = p.refStart.clone().add(shift)
    const refEnd = p.refEnd.clone().add(shift)
    const centre = p.centre.clone().add(shift)
    const corners = p.corners.map((c) => c.clone().add(shift))
    const record = {
      id: p.id,
      kind: p.kind,
      i: p.i,
      j: p.j,
      axis: p.axis,
      anchor: p.anchor,
      level: p.level,
      role: p.role,
      glyph: ROLE_GLYPH[p.role],
      tiltDeg: r(p.tiltDeg),
      present: p.present,
      flipped: p.flipped,
      panelType: cfg.lattice.panelType,
      widthCm: r(W),
      lengthCm: r(L),
      position: rv(centre),
      quaternion: [r(p.quat.x), r(p.quat.y), r(p.quat.z), r(p.quat.w)],
      normal: rv(p.normal),
      refStart: rv(refStart),
      refEnd: rv(refEnd),
      corners: corners.map(rv),
      // The OBB in the exact shape collide.js consumes. Note the re-centring:
      // `position` is the centre of the LIT FACE, while the solid's centre sits
      // half the housing thickness behind it along −n̂ — so a flipped panel's box
      // is on the other side of the reference plane, which is the whole physical
      // content of a flip. The thickness is derived from PANEL_PROFILE and is
      // never written down here (config.js's header, HANDOFF §2.19).
      obb: {
        center: rv(centre.clone().addScaledVector(p.normal, -T / 2)),
        halfExtents: [r(W / 2), r(T / 2), r(L / 2)],
        quaternion: [r(p.quat.x), r(p.quat.y), r(p.quat.z), r(p.quat.w)],
      },
    }
    panels.push(record)
    exact.set(p.id, { ...p, refStart, refEnd, centre, record })
  }

  // --- joints --------------------------------------------------------------
  // A ramp has two joints — one to its ground cell, one to its high cell — and
  // each exists when the ramp and that cell are both present. An anchor ramp has
  // only the high one; its toe joins nothing this model knows about.
  //
  // The rim is the SHARED LINE, taken once and handed to both sides: A's rim
  // point is the cell edge or the ramp's segment end, B's is the other, and the
  // vector between them is the gap step. One source of truth for the line two
  // modules measure — the general form of the bug in HANDOFF §2.8.
  const joints = []
  const cellAt = (i, j) => exact.get(`Ci${i}j${j}`)

  for (const p of built) {
    if (p.kind !== 'ramp' || !p.present) continue
    const ramp = exact.get(p.id)
    const e = p.e
    const u = new THREE.Vector3().subVectors(ramp.refEnd, ramp.refStart).normalize()
    // Which local axis of a CELL the rim runs along: a z-axis edge meets the
    // cell's north/south rim, which runs along its local X; an x-axis edge meets
    // its east/west rim, which runs along its local Z. Getting this from the
    // panel's own placement rather than assuming world X is the whole content of
    // "a joint no longer runs along X".
    const cellRimLocal = p.axis === 'z'
      ? new THREE.Vector3(1, 0, 0)
      : new THREE.Vector3(0, 0, 1)
    const runAxis = AXIS_INDEX[p.axis === 'z' ? 'x' : 'z']

    const ends = []
    if (p.lo) {
      const lo = cellAt(p.lo.i, p.lo.j)
      if (lo && lo.present) {
        ends.push({
          end: 'lo',
          A: lo,
          B: ramp,
          rimA: lo.centre.clone().addScaledVector(e, L / 2),
          rimB: ramp.refStart.clone(),
          runLocalA: cellRimLocal,
          runLocalB: new THREE.Vector3(1, 0, 0),
          // Which way each panel's material lies from the shared rim: the cell
          // reaches back against ê, the ramp forward along its own direction.
          inwardA: e.clone().negate(),
          inwardB: u.clone(),
        })
      }
    }
    {
      const hi = cellAt(p.hi.i, p.hi.j)
      if (hi && hi.present) {
        ends.push({
          end: 'hi',
          A: ramp,
          B: hi,
          rimA: ramp.refEnd.clone(),
          rimB: hi.centre.clone().addScaledVector(e, -L / 2),
          runLocalA: new THREE.Vector3(1, 0, 0),
          runLocalB: cellRimLocal,
          inwardA: u.clone().negate(),
          inwardB: e.clone(),
        })
      }
    }

    for (const g of ends) {
      // Each panel's OWN rim direction, taken through its OWN placement rather
      // than from the joint — so `twistDeg` downstream is a measurement of two
      // independently-placed panels agreeing, not a tautology. On a 2-D lattice
      // the two can come out ANTIPARALLEL (see the file header), which is why
      // the twist is measured between lines rather than rays.
      const runA = g.runLocalA.clone().applyQuaternion(g.A.quat)
      const runB = g.runLocalB.clone().applyQuaternion(g.B.quat)
      joints.push({
        id: `${g.A.id}>${g.B.id}`,
        jointIndex: joints.length,
        a: g.A.id,
        b: g.B.id,
        panelA: g.A.record.kind,
        panelB: g.B.record.kind,
        edge: { i: p.i, j: p.j, axis: p.axis },
        end: g.end,
        anchor: p.anchor,
        lengthCm: r(W),
        // The world axis the rim is PARAMETERIZED along, and the interval of it
        // the rim occupies. `s` is a WORLD COORDINATE, not a normalized
        // parameter — the power supply's blocked span is stated in the same
        // coordinates, so keeping them the same avoids a conversion that could
        // only ever be got wrong. In the ribbon this was always world X; here it
        // is X for a z-axis edge and Z for an x-axis one.
        runAxis,
        run: runAxis === 0 ? [1, 0, 0] : [0, 0, 1],
        runFrom: r(g.rimA.getComponent(runAxis) - W / 2),
        runTo: r(g.rimA.getComponent(runAxis) + W / 2),
        runA: rvFine(runA),
        runB: rvFine(runB),
        rimA: rvFine(g.rimA),
        rimB: rvFine(g.rimB),
        normalA: rvFine(g.A.normal),
        normalB: rvFine(g.B.normal),
        inwardA: rvFine(g.inwardA),
        inwardB: rvFine(g.inwardB),
        flippedA: g.A.flipped,
        flippedB: g.B.flipped,
        tiltADeg: r(g.A.tiltDeg),
        tiltBDeg: r(g.B.tiltDeg),
      })
    }
  }

  return {
    config: cfg,
    lattice: {
      cols,
      rows,
      phase,
      panelType: cfg.lattice.panelType,
      widthCm: r(W),
      lengthCm: r(L),
      pitchCm: r(P),
      riseCm: r(R),
      stepPlanCm: r(stepPlanCm),
      stepRiseCm: r(stepRiseCm),
      wallAnchor: cfg.placement.wallAnchor,
      levels,
      shiftCm: rv(shift),
    },
    panels,
    joints,
    bounds: latticeBounds(panels),
  }
}

/**
 * The world point at run-parameter `s` along a joint's rim, on one side of it.
 *
 * `s` is a WORLD COORDINATE on `joint.runAxis`, in [runFrom, runTo]. Exact by
 * construction: both rims are straight lines parallel to that axis, so the point
 * is the stored rim with one component replaced. No sampling, no interpolation.
 */
export function jointRimPoint(joint, isA, s) {
  const rim = isA ? joint.rimA : joint.rimB
  const p = new THREE.Vector3(rim[0], rim[1], rim[2])
  p.setComponent(joint.runAxis, s)
  return p
}

/**
 * Axis-aligned extent of every PRESENT panel solid — the physical envelope
 * including housings, which is what the measuring box reports. Absent panels are
 * holes in the network and must not inflate its box.
 */
export function latticeBounds(panels) {
  const min = [Infinity, Infinity, Infinity]
  const max = [-Infinity, -Infinity, -Infinity]
  for (const p of panels) {
    if (!p.present) continue
    for (const c of p.corners) {
      for (let k = 0; k < 3; k++) {
        if (c[k] < min[k]) min[k] = c[k]
        if (c[k] > max[k]) max[k] = c[k]
      }
    }
  }
  if (!Number.isFinite(min[0])) return { min: [0, 0, 0], max: [0, 0, 0], size: [0, 0, 0] }
  return {
    min: min.map(r),
    max: max.map(r),
    size: [r(max[0] - min[0]), r(max[1] - min[1]), r(max[2] - min[2])],
  }
}
