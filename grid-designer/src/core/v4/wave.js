/**
 * grid-designer v4 — the WAVE: a varying angle that keeps the plan grid square.
 *
 * HEADLESS ZONE (src/core/): pure functions, importable from plain node.
 *   - explicit `.js` extensions on ALL relative imports
 *   - may import `three` math classes only; never components / store / DOM
 *   - same config in → byte-identical output out
 *
 * =============================================================================
 * THE REQUIREMENT
 * =============================================================================
 * "a wave based angle, but in the simplest way possible. base angle at the front
 * which will be the lowest, but the mesh needs to compress in a simple linear
 * fashion. nothing gets out of basic alignment, but there should be % scrunching
 * in the rows and columns towards a basic attractor. everything stays linear,
 * with the new gap"
 *
 * Four claims, and each one pins down a piece of the model:
 *
 *   NOTHING GETS OUT OF BASIC ALIGNMENT   the plan grid stays a PRODUCT grid —
 *        the x of column `i` does not depend on `j`, the z of row `j` does not
 *        depend on `i`. Every cell stays on the crossing of one x line and one z
 *        line, so the network still reads as a grid from above.
 *   BASE ANGLE AT THE FRONT, THE LOWEST    `config.angleDeg` is the SMALLEST
 *        angle in the design and it belongs to edge 0 of each axis. Scrunching
 *        only ever steepens.
 *   COMPRESS IN A SIMPLE LINEAR FASHION    the thing that varies linearly is the
 *        PLAN ADVANCE, not the angle. `E(θ(k)) = E(base)·(1 − factor(k))` with
 *        `factor` a straight ramp — so the pitch shrinks by equal steps, which is
 *        what "scrunch" reads as on the floor. The angles that produce it are
 *        whatever they have to be.
 *   TOWARDS A BASIC ATTRACTOR              the ramp reaches full scrunch at a
 *        normalized edge position and stays there. `attractor = 1` spreads the
 *        compression over the whole axis; `0.5` finishes it half way and leaves
 *        the far half uniformly tight.
 *
 * =============================================================================
 * WHY THE CHECKERBOARD CANNOT DO THIS — an impossibility, not a difficulty
 * =============================================================================
 * This module exists because `pattern.kind: 'trapezoid'` PROVABLY cannot carry a
 * varying angle, and the next person to read this will otherwise try to make it.
 *
 * Alignment forces the x-ramp angle to depend only on `i` and the z-ramp angle
 * only on `j`. Walk any 4-cycle of the lattice,
 *
 *     (i,j) → (i+1,j) → (i+1,j+1) → (i,j+1) → (i,j)
 *
 * On a two-level CHECKERBOARD the levels alternate around that cycle, so the four
 * height steps are `+R(θx(i))`, `−R(θz(j))`, `+R(θx(i))`, `−R(θz(j))` and closure
 * demands
 *
 *     2·R(θx(i)) − 2·R(θz(j)) = 0        ⟹        every angle equal.
 *
 * Measured at gap 2, lengths in cm: an x-ramp at 30° against a z-ramp at 35°
 * leaves the loop open by **9.164**; against 40°, by **17.800**. The cycle does
 * not close, which means there is no such network — not that it is hard to find.
 *
 * The escape hatch of varying the GAP to hold `R` constant is dead too. Holding
 * `R = 31.035` (the 30°/2cm value) needs gap **30.27cm at 20°**, **13.12 at 25°**,
 * **−5.62 at 35°**, **−11.01 at 40°** — every one of them outside the connector
 * envelope's 1–8cm, and two of them negative.
 *
 * =============================================================================
 * WHAT DOES WORK — the separable family
 * =============================================================================
 * Drop the checkerboard and make the height field SEPARABLE:
 *
 *     h(i, j) = f(i) + g(j)
 *
 * Then the 4-cycle's residual is `(f(i+1) − f(i)) + (g(j+1) − g(j)) − (f(i+1) −
 * f(i)) − (g(j+1) − g(j)) = 0` identically, for ANY f and g. Closure stops being
 * a constraint on the angles and becomes an algebraic identity. Measured over a
 * lattice with 5 different x-angles and 6 different z-angles: worst 4-cycle
 * height residual **7.1e-15 cm**.
 *
 * This is exactly the family HANDOFF §3 rejected — "additive / translational
 * height fields for exact planar quads" — and the rejection still stands FOR THE
 * PROBLEM IT WAS ABOUT. v3 needed a height field that was zero along two
 * intersecting edges and a mound in between, and the separable family cannot be
 * that. v4 is not asking for a mound. It is asking for a foldable network, and
 * separability is the exact condition for one. The same fact, read as a
 * permission instead of a refusal (HANDOFF §5.3's closing paragraph predicted
 * this would be the move).
 *
 * =============================================================================
 * SO THE WAVE IS NOT THE CHECKERBOARD, EVEN AT ZERO SCRUNCH
 * =============================================================================
 * With f and g each zig-zagging by ±R, `h(i,j) = f(i) + g(j)` takes THREE values
 * — 0, R and 2R — not two. Cell (1,1) is two rises up, where the checkerboard
 * would put it back on the floor. That is an egg-crate, and it is the honest
 * shape of the only family that admits a varying angle.
 *
 * It is expected, it is documented, and it must not be "fixed": forcing it back
 * to two levels is precisely the constraint the impossibility proof above says
 * kills the wave. `trapezoid` remains the default and is untouched, so nothing
 * that was designed on the checkerboard moves.
 *
 * =============================================================================
 * ZIG-ZAG SIGNS, AND WHY THE SURFACE DRIFTS DOWN
 * =============================================================================
 *     f(0) = 0 ;  f(i+1) = f(i) + σ(i)·R(θx(i)) ,  σ(i) = (−1)^i
 *
 * At a uniform angle the partial sums are 0, R, 0, R … and the axis is level. As
 * soon as the angles differ they are 0, R₀, R₀−R₁, R₀−R₁+R₂ … and because
 * scrunching only ever STEEPENS (R is increasing in θ), the negative terms
 * outweigh the positive ones and the run drifts DOWNWARD as it compresses.
 *
 * That is a real property of the shape, not an artefact: a run of folds whose
 * far end is steeper than its near end sheds height. `lattice.js` then grounds
 * the finished network as usual, so what the drift actually does is tilt the
 * whole assembly. It is reported (the per-axis tables carry every `f(i)`) rather
 * than corrected, because correcting it would mean breaking the zig-zag and the
 * zig-zag is what makes the panels alternate up and down.
 *
 * =============================================================================
 * OUTPUT
 * =============================================================================
 *   solveWave(cfg, lengthCm) → {
 *     x: axis, z: axis,            // see `solveAxis`
 *     warnings: [{ code, axis, index, ... }],
 *   }
 */

import { ANGLE_MAX, clamp } from './schema.js'

const RAD = Math.PI / 180

/** Fixed, never "until converged" — the standing rule in this core (report.js). */
export const WAVE_BISECTION_STEPS = 60

/** An edge whose asked-for plan advance is past what `ANGLE_MAX` can deliver. */
export const SCRUNCH_UNREACHABLE_CODE = 'W_SCRUNCH_UNREACHABLE'

/**
 * The PLAN ADVANCE across one ramp: how much further the next cell line sits,
 * over and above the panel's own length.
 *
 * `E(θ) = L·cos θ + 2·gap·cos(θ/2)`. The two terms are the ramp's own plan run
 * and the two half-angle gap steps either side of it (V4_SPEC §9.1) — so the
 * cell pitch is `L + E(θ)` and at a uniform angle this reproduces `latticeStep`'s
 * `pitchCm` exactly. Strictly decreasing on [0°, 90°], which is what makes the
 * bisection below well posed.
 */
export function planAdvanceCm(angleDeg, gapCm, lengthCm) {
  const theta = angleDeg * RAD
  return lengthCm * Math.cos(theta) + 2 * gapCm * Math.cos(theta / 2)
}

/**
 * The RISE across one ramp, `R(θ) = L·sin θ + 2·gap·sin(θ/2)` — `latticeStep`'s
 * `riseCm`, restated per angle. Strictly increasing on [0°, 90°].
 */
export function riseCm(angleDeg, gapCm, lengthCm) {
  const theta = angleDeg * RAD
  return lengthCm * Math.sin(theta) + 2 * gapCm * Math.sin(theta / 2)
}

/**
 * How much of the base plan advance is taken away at normalized position `t`.
 *
 * A straight ramp to `scrunch` at `t = attractor`, flat after it. This — and not
 * the angle — is the thing the brief calls linear, and it is why the compression
 * reads as even spacing on the floor rather than as even degrees.
 *
 * `attractor = 0` is the degenerate reading "everything is already past the
 * attractor", i.e. a UNIFORM compression at full strength. It is kept rather than
 * clamped away because it is a real thing to ask for (a tighter grid at one
 * angle), but note it is the one setting where θ(0) is NOT the base angle.
 */
export function scrunchFactor(t, scrunch, attractor) {
  if (scrunch <= 0) return 0
  if (attractor <= 0) return scrunch
  return scrunch * Math.min(1, t / attractor)
}

/**
 * The angle whose plan advance is `targetCm`, found by bisection on [base, MAX].
 *
 * Fixed `WAVE_BISECTION_STEPS`, never "until converged": determinism is the
 * standing rule in this core and 60 halvings of a 75° bracket is 6.5e-17°.
 *
 * Returns the angle KNOWN to advance by at least the target — the same
 * "the number is a permission, and an untested permission is not one" convention
 * report.js's envelope uses. `reached` is false when even `ANGLE_MAX` cannot
 * compress that far, and the caller raises `W_SCRUNCH_UNREACHABLE`.
 */
export function angleForAdvance(targetCm, baseAngleDeg, gapCm, lengthCm) {
  const atBase = planAdvanceCm(baseAngleDeg, gapCm, lengthCm)
  // No compression asked for (or asked for backwards): the base angle IS the
  // answer, exactly, with no bisection to round it. This is what makes
  // `θ(0) === angleDeg` an identity rather than a tolerance.
  if (targetCm >= atBase) return { angleDeg: baseAngleDeg, reached: true }
  const atMax = planAdvanceCm(ANGLE_MAX, gapCm, lengthCm)
  if (targetCm <= atMax) return { angleDeg: ANGLE_MAX, reached: false, floorCm: atMax }

  let lo = baseAngleDeg // advances by at least the target
  let hi = ANGLE_MAX    // advances by less
  for (let k = 0; k < WAVE_BISECTION_STEPS; k++) {
    const mid = (lo + hi) / 2
    if (planAdvanceCm(mid, gapCm, lengthCm) >= targetCm) lo = mid
    else hi = mid
  }
  return { angleDeg: lo, reached: true }
}

/**
 * One axis of the wave: its edge angles, its plan lines, and its height run.
 *
 * @param {object} spec
 * @param {number} spec.edgeCount    edges on this axis — `cols − 1` or `rows − 1`
 * @param {number} spec.baseAngleDeg θ at edge 0, the front, the lowest
 * @param {number} spec.gapCm
 * @param {number} spec.lengthCm     the panel's length along the axis
 * @param {number} spec.scrunch      0..0.9, the fraction of plan advance removed
 * @param {number} spec.attractor    0..1, where full scrunch is reached
 * @param {string} spec.axis         'x' | 'z', for the warnings only
 * @returns {{
 *   axis: string, edgeCount: number,
 *   angleDeg: number[], advanceCm: number[], riseCm: number[], factor: number[],
 *   sign: number[], lineCm: number[], heightCm: number[],
 *   baseAdvanceCm: number, planRunCm: number, unscrunchedRunCm: number,
 *   warnings: object[],
 * }}
 *
 * `lineCm[i]` is the plan coordinate of cell line `i` measured from line 0, and
 * `heightCm[i]` is that line's contribution to the height field. Both are
 * CUMULATIVE and depend on one index only — which is the whole content of
 * "nothing gets out of basic alignment".
 */
export function solveAxis({ edgeCount, baseAngleDeg, gapCm, lengthCm, scrunch, attractor, axis }) {
  const n = Math.max(0, edgeCount)
  const baseAdvanceCm = planAdvanceCm(baseAngleDeg, gapCm, lengthCm)
  const angleDeg = []
  const advanceCm = []
  const rises = []
  const factor = []
  const sign = []
  const warnings = []

  for (let k = 0; k < n; k++) {
    // A single edge has no ramp to ramp toward, so it sits at the base angle
    // rather than jumping straight to full scrunch. `t` is the position ALONG
    // the run of edges, and with one edge there is no run.
    const t = n > 1 ? k / (n - 1) : 0
    const fac = scrunchFactor(t, scrunch, attractor)
    const target = baseAdvanceCm * (1 - fac)
    const found = angleForAdvance(target, baseAngleDeg, gapCm, lengthCm)
    if (!found.reached) {
      warnings.push({
        code: SCRUNCH_UNREACHABLE_CODE,
        axis,
        index: k,
        wantedAdvanceCm: target,
        floorCm: found.floorCm,
        angleDeg: ANGLE_MAX,
        message:
          `${axis} edge ${k} was asked to advance ${target.toFixed(3)}cm (${(fac * 100).toFixed(1)}% ` +
          `scrunch off ${baseAdvanceCm.toFixed(3)}cm) and the steepest angle the schema allows, ` +
          `${ANGLE_MAX}°, only gets down to ${found.floorCm.toFixed(3)}cm. The edge is pinned at ` +
          `${ANGLE_MAX}° and the plan run is longer than the scrunch asked for`,
      })
    }
    angleDeg.push(found.angleDeg)
    advanceCm.push(planAdvanceCm(found.angleDeg, gapCm, lengthCm))
    rises.push(riseCm(found.angleDeg, gapCm, lengthCm))
    factor.push(fac)
    // The zig-zag. Even edges go up, odd ones come back down — the alternation
    // is what makes a fold pattern rather than a staircase, and it is fixed by
    // parity rather than measured so that θ = 0 (every rise zero) still names a
    // direction for every ramp.
    sign.push(k % 2 === 0 ? 1 : -1)
  }

  const lineCm = [0]
  const heightCm = [0]
  for (let k = 0; k < n; k++) {
    lineCm.push(lineCm[k] + lengthCm + advanceCm[k])
    heightCm.push(heightCm[k] + sign[k] * rises[k])
  }

  return {
    axis,
    edgeCount: n,
    angleDeg,
    advanceCm,
    riseCm: rises,
    factor,
    sign,
    lineCm,
    heightCm,
    baseAdvanceCm,
    /** Line 0 to line n — what a tape reads across the cell centres. */
    planRunCm: lineCm[n],
    /** The same run with no scrunch at all, so the compression is quotable. */
    unscrunchedRunCm: n * (lengthCm + baseAdvanceCm),
    warnings,
  }
}

/**
 * Both axes of the wave, from a NORMALIZED config.
 *
 * @param {object} cfg normalized v4 config with `pattern.kind === 'wave'`
 * @param {number} lengthCm the panel's length (the lattice is square, so one)
 */
export function solveWave(cfg, lengthCm) {
  const w = cfg.pattern.wave
  const x = solveAxis({
    edgeCount: cfg.lattice.cols - 1,
    baseAngleDeg: cfg.angleDeg,
    gapCm: cfg.gap,
    lengthCm,
    scrunch: w.scrunchX,
    attractor: w.attractorX,
    axis: 'x',
  })
  const z = solveAxis({
    edgeCount: cfg.lattice.rows - 1,
    baseAngleDeg: cfg.angleDeg,
    gapCm: cfg.gap,
    lengthCm,
    scrunch: w.scrunchZ,
    attractor: w.attractorZ,
    axis: 'z',
  })
  return { x, z, warnings: [...x.warnings, ...z.warnings] }
}

/**
 * The worst 4-cycle residual of a field of RAMP STEPS, over every cell of the
 * grid.
 *
 * `step(i, j, axis)` is the height the ramp on that edge climbs, going from
 * `(i,j)` to its `+axis` neighbour — the quantity the GEOMETRY actually builds,
 * not a difference of two heights. That distinction is the whole point: summing
 * differences of a height field around a loop telescopes to zero for any field
 * whatsoever and would assert nothing. Summing the four ramps' own rises asks
 * whether the network closes, which is a question that can be answered no.
 *
 *     residual(i,j) = step(i,j,'x') + step(i+1,j,'z') − step(i,j+1,'x') − step(i,j,'z')
 *
 * `tests/test-v4-wave.mjs` runs this over the wave's steps (residual ~1e-15) and
 * over a CHECKERBOARD assembled from the same varying angles (residual ~9cm),
 * which is the non-vacuous negative proving the check can fail.
 *
 * @param {(i: number, j: number, axis: 'x'|'z') => number} step
 */
export function worstCycleResidual(step, cols, rows) {
  let worst = 0
  for (let i = 0; i + 1 < cols; i++) {
    for (let j = 0; j + 1 < rows; j++) {
      const loop =
        step(i, j, 'x') + step(i + 1, j, 'z') - step(i, j + 1, 'x') - step(i, j, 'z')
      worst = Math.max(worst, Math.abs(loop))
    }
  }
  return worst
}

/** Clamp helper re-exported so callers do not have to reach into the schema for
 *  the one thing they need when building a wave block by hand. */
export { clamp }
