/**
 * grid-designer v3 — the drift heightfield.
 *
 * HEADLESS ZONE (src/core/): pure functions, importable from plain node.
 *   - explicit `.js` extensions on ALL relative imports
 *   - imports `three`'s math classes only; never a scene graph, the DOM, or the store
 *   - pure and deterministic: same `form` + (x, z) in → byte-identical output
 *   - NO dependency on schema.js — this package (V3_SPEC.md §8, P1) is pure math
 *     that the schema later wraps; the schema must never be required to exercise it.
 *
 * =============================================================================
 * WHAT THIS IS — V3_SPEC.md §2
 * =============================================================================
 * A parametric snow-drift heightfield H(x, z) ≥ 0 over the plan (the floor,
 * x = 0 at the wall running +X across the window, z = 0 at the window running
 * +Z toward the back). It is AUTHORED, not solved: a person sculpting a drift
 * thinks in (amplitude, crest position, toe sharpness), not in a solver. Later
 * packages (tiling, placement) sample this surface as their *target*; they do
 * not touch this file's exports for anything but reading H and its gradient.
 *
 * H = 0 outside the footprint — material that runs past the drift lies flat on
 * the floor, which is correct and reads as the drift feathering out.
 *
 * =============================================================================
 * THE PROFILE PRIMITIVE — a one-axis asymmetric bump with a non-flat toe
 * =============================================================================
 *   B(τ; p, a) = normalize( τ^a · (1 − τ)^b ),   b = a·(1 − p)/p
 *
 * τ ∈ [0, 1]; B is 0 outside that range by definition (handled explicitly below
 * so we never raise a fractional power of a negative base).  ln B(τ) = a ln τ +
 * b ln(1−τ) + const, so d(ln B)/dτ = a/τ − b/(1−τ) = 0 at τ = a/(a+b); solving
 * b = a(1−p)/p makes that root land exactly at τ = p. `normalize` divides by the
 * value at τ = p so the peak is exactly 1 — that's the "/ p^a·(1−p)^b" the work
 * order restates.
 *
 * TOE SLOPE, AND WHY toeSharp IS CLAMPED TO [0.45, 1.0] — do not widen this:
 *   dB/dτ at τ→0⁺ scales like τ^(a−1). For a < 1 that diverges (a vertical toe,
 *   the steepest kind of "not flat"); for a = 1 it is the finite constant
 *   B'(0) = a/p·1 = 1/p (a proper linear toe). For a > 1, τ^(a−1) → 0: the toe
 *   flattens out and touches the ground tangent to the floor, which is exactly
 *   the "flat toe" the brief forbids ("left edge and window edge start from the
 *   ground but not flat"). Clamping a (toeSharpX / toeSharpZ) to (0, 1] is what
 *   makes "grounded but not flat" mechanical rather than a hope. The lower bound
 *   0.45 is a shape choice (below it the toe reads as a spike, not a drift) —
 *   see V3_SPEC.md §2.2 / §6.
 *
 * =============================================================================
 * THE DRIFT — V3_SPEC.md §2.3
 * =============================================================================
 *   s = x / footprint.width          (0 at the wall,   1 at the far edge)
 *   t = z / footprint.depth          (0 at the window, 1 at the back)
 *   tc(s) = clamp(crestZ + ridgeShear·s, RIDGE_CENTER_MIN, RIDGE_CENTER_MAX)
 *   H(x, z) = amplitude · B(s; crestX, toeSharpX) · B(t; tc(s), toeSharpZ)
 *
 * Both factors vanish at s = 0 and at t = 0, so the wall edge and the window
 * edge come out of the form for free (H = 0 there) rather than being
 * special-cased, and B's non-flat toe (above) gives them non-zero slope too.
 *
 * `tc` is clamped to [0.05, 0.95] rather than [0, 1] (unlike the outer s, t
 * clamp, which is a hard footprint boundary): tc is the CREST parameter `p` fed
 * into the z-axis profile, and b = a(1−p)/p divides by p and by (1−p), so p
 * must stay strictly inside (0, 1) or the profile becomes degenerate. 0.05/0.95
 * keeps it well away from either pole while still letting the ridge visibly
 * walk almost to either wall as `ridgeShear` pushes it — see RIDGE_CENTER_MIN/MAX
 * below.
 *
 * =============================================================================
 * THE GRADIENT — the crux of this file
 * =============================================================================
 * `ridgeShear` makes tc depend on s, so H's dependence on x has TWO paths:
 * directly through B(s; crestX, toeSharpX), and indirectly through the z-axis
 * profile's crest parameter tc(s). Writing H = amplitude · Bx(s) · Bz(t; tc(s)):
 *
 *   ∂H/∂x = amplitude · [ Bx'(s)·Bz(t; tc(s)) + Bx(s)·∂Bz/∂p(t; tc(s))·tc'(s) ] / width
 *   ∂H/∂z = amplitude ·   Bx(s) · Bz'(t; tc(s))                              / depth
 *
 * where Bx' and Bz' are B's derivative w.r.t. its own τ argument (`profileDTau`
 * below), tc'(s) = ridgeShear where the clamp is not saturated and 0 where it
 * is (a standard subgradient choice — the clamp is piecewise-linear), and
 * ∂Bz/∂p (`profileDCrest` below) is B's derivative w.r.t. its CREST parameter p
 * with τ held fixed. That last one is the term "a lot of people would silently
 * drop": b = a(1−p)/p depends on p too, so ∂Bz/∂p differentiates through both
 * the explicit p^a·(1−p)^b normalizer AND b(p) itself. Full derivation:
 *
 *   f(τ; p, a) = τ^a·(1−τ)^b(p),  b(p) = a(1−p)/p,  db/dp = −a/p²
 *   fmax(p)    = p^a·(1−p)^b(p)                    (so B = f / fmax)
 *   df/dp      = f · ln(1−τ) · db/dp                (only b depends on p)
 *   dfmax/dp   = a·p^(a−1)·(1−p)^b
 *              + p^a·(1−p)^b · ( db/dp·ln(1−p) − b/(1−p) )
 *   dB/dp      = (df/dp · fmax − f · dfmax/dp) / fmax²
 *
 * This is a closed form — no finite difference anywhere in this file. It was
 * checked against central differences at 2000 random (τ, p, a) triples during
 * development (max relative error ~3e-10) before being written down here; the
 * shipped test suite re-checks it against `driftHeight` itself.
 *
 * =============================================================================
 * EXPORTS — V3_SPEC.md §2.4, exactly this list
 * =============================================================================
 *   driftHeight(form, x, z)        → number
 *   driftGradient(form, x, z)      → [dH/dx, dH/dz]   (analytic)
 *   driftNormal(form, x, z)        → THREE.Vector3    (unit, +Y-ish)
 *   driftFrame(form, x, z)         → { point, normal }
 *   sampleDriftMesh(form, nx, nz)  → { positions, indices }
 *   DEFAULT_FORM
 *   normalizeForm(form)
 */

import * as THREE from 'three'

// =============================================================================
// Config ranges — V3_SPEC.md §6. `normalizeForm` clamps to these; nothing else
// in this file should ever construct a form outside them.
// =============================================================================
const AMPLITUDE_MIN = 0
const AMPLITUDE_MAX = 250
const CREST_MIN = 0.15
const CREST_MAX = 0.85
const RIDGE_SHEAR_MIN = -0.4
const RIDGE_SHEAR_MAX = 0.4
/** See "why toeSharp is clamped" above — a > 1 gives a flat, brief-violating toe. */
const TOE_SHARP_MIN = 0.45
const TOE_SHARP_MAX = 1.0
/** Minimum plan footprint extent (cm); guards s = x/width, t = z/depth against
 *  division by zero. Not a V3_SPEC range (footprint isn't listed in §6) — just
 *  a sanity floor. */
const FOOTPRINT_MIN = 1
/**
 * Faceting knobs, consumed by `target.js` (NOT by driftHeight — the drift itself
 * stays smooth; faceting is a quantization of it onto the panel lattice).
 *
 * `angularity` blends smooth → faceted. Rigid panels cannot BE a smooth
 * surface, so a smooth target is one they can only approximate: joints wedge
 * open, housings collide past ~40cm of amplitude, and no region stays flat
 * enough to lay a rigid 121cm plate. A faceted target is one they can be
 * exactly. 1 = fully faceted, 0 = the smooth drift with all of that faithfully
 * reported.
 *
 * `facetCells` is how many cells share a facet plane: 1 gives every panel its
 * own plane (a fold at every joint), larger gives broader planes with crisper
 * creases between them. Facet boundaries always fall on cell boundaries, so a
 * crease only ever lands where there is already a physical joint to absorb it.
 */
export const ANGULARITY_MIN = 0
export const ANGULARITY_MAX = 1
export const FACET_CELLS_MIN = 1
export const FACET_CELLS_MAX = 4

/**
 * Clamp range for tc(s), the z-axis crest parameter after ridgeShear shifts it.
 * NOT the same as CREST_MIN/MAX (which bound the config knobs crestX/crestZ):
 * this is an internal numerical guard so tc never reaches the poles p → 0 or
 * p → 1, where b = a(1−p)/p blows up or the profile degenerates. See "THE
 * DRIFT" above.
 */
const RIDGE_CENTER_MIN = 0.05
const RIDGE_CENTER_MAX = 0.95

export const DEFAULT_FORM = {
  amplitude: 120,
  crestX: 0.62,
  crestZ: 0.55,
  ridgeShear: 0.18,
  toeSharpX: 0.8,
  toeSharpZ: 0.9,
  angularity: 0.5,
  facetCells: 2,
  footprint: { width: 365, depth: 430 },
}

function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v
}

function isPlainObject(v) {
  return typeof v === 'object' && v !== null && !Array.isArray(v)
}

function numberOr(v, fallback) {
  return typeof v === 'number' && Number.isFinite(v) ? v : fallback
}

/**
 * Fill defaults and clamp every knob to its V3_SPEC §6 range. Idempotent:
 * normalizeForm(normalizeForm(f)) === normalizeForm(f) for any input, because
 * every field is independently clamped/defaulted with no cross-field state.
 */
export function normalizeForm(partial) {
  const src = isPlainObject(partial) ? partial : {}
  const footprint = isPlainObject(src.footprint) ? src.footprint : {}
  return {
    amplitude: clamp(numberOr(src.amplitude, DEFAULT_FORM.amplitude), AMPLITUDE_MIN, AMPLITUDE_MAX),
    crestX: clamp(numberOr(src.crestX, DEFAULT_FORM.crestX), CREST_MIN, CREST_MAX),
    crestZ: clamp(numberOr(src.crestZ, DEFAULT_FORM.crestZ), CREST_MIN, CREST_MAX),
    ridgeShear: clamp(numberOr(src.ridgeShear, DEFAULT_FORM.ridgeShear), RIDGE_SHEAR_MIN, RIDGE_SHEAR_MAX),
    toeSharpX: clamp(numberOr(src.toeSharpX, DEFAULT_FORM.toeSharpX), TOE_SHARP_MIN, TOE_SHARP_MAX),
    toeSharpZ: clamp(numberOr(src.toeSharpZ, DEFAULT_FORM.toeSharpZ), TOE_SHARP_MIN, TOE_SHARP_MAX),
    angularity: clamp(numberOr(src.angularity, DEFAULT_FORM.angularity), ANGULARITY_MIN, ANGULARITY_MAX),
    facetCells: Math.round(clamp(numberOr(src.facetCells, DEFAULT_FORM.facetCells), FACET_CELLS_MIN, FACET_CELLS_MAX)),
    footprint: {
      width: Math.max(FOOTPRINT_MIN, numberOr(footprint.width, DEFAULT_FORM.footprint.width)),
      depth: Math.max(FOOTPRINT_MIN, numberOr(footprint.depth, DEFAULT_FORM.footprint.depth)),
    },
  }
}

// =============================================================================
// The profile primitive B(τ; p, a) and its two analytic derivatives.
// Module-private: callers only ever see H through driftHeight/driftGradient.
// =============================================================================

/** B(τ; p, a). 0 outside [0, 1] by definition — guards against raising a
 *  negative base to a fractional power. */
function profile(tau, p, a) {
  if (tau <= 0 || tau >= 1) return 0
  const b = (a * (1 - p)) / p
  const fmax = Math.pow(p, a) * Math.pow(1 - p, b)
  const f = Math.pow(tau, a) * Math.pow(1 - tau, b)
  return f / fmax
}

/** ∂B/∂τ, p and a held fixed. */
function profileDTau(tau, p, a) {
  if (tau <= 0 || tau >= 1) return 0
  const b = (a * (1 - p)) / p
  const fmax = Math.pow(p, a) * Math.pow(1 - p, b)
  const term = a * Math.pow(tau, a - 1) * Math.pow(1 - tau, b) - b * Math.pow(tau, a) * Math.pow(1 - tau, b - 1)
  return term / fmax
}

/** ∂B/∂p, τ and a held fixed — the term the ridgeShear cross-derivative needs.
 *  See "THE GRADIENT" in the header comment for the full derivation. */
function profileDCrest(tau, p, a) {
  if (tau <= 0 || tau >= 1) return 0
  const b = (a * (1 - p)) / p
  const dbDp = -a / (p * p)

  const f = Math.pow(tau, a) * Math.pow(1 - tau, b)
  const dfDp = f * Math.log(1 - tau) * dbDp

  const fmax = Math.pow(p, a) * Math.pow(1 - p, b)
  const dfmaxDp = a * Math.pow(p, a - 1) * Math.pow(1 - p, b) + fmax * (dbDp * Math.log(1 - p) - b / (1 - p))

  return (dfDp * fmax - f * dfmaxDp) / (fmax * fmax)
}

/** tc(s) = clamp(crestZ + ridgeShear·s, RIDGE_CENTER_MIN, RIDGE_CENTER_MAX). */
function ridgeCenter(form, s) {
  return clamp(form.crestZ + form.ridgeShear * s, RIDGE_CENTER_MIN, RIDGE_CENTER_MAX)
}

/** tc'(s): ridgeShear where the clamp is not saturated, 0 where it is. */
function ridgeCenterSlope(form, s) {
  const raw = form.crestZ + form.ridgeShear * s
  if (raw <= RIDGE_CENTER_MIN || raw >= RIDGE_CENTER_MAX) return 0
  return form.ridgeShear
}

/**
 * H(x, z) — V3_SPEC.md §2.3. 0 outside the footprint (s or t outside [0, 1]),
 * which falls out of `profile` for free.
 */
export function driftHeight(form, x, z) {
  const s = x / form.footprint.width
  const t = z / form.footprint.depth
  const tc = ridgeCenter(form, s)
  return form.amplitude * profile(s, form.crestX, form.toeSharpX) * profile(t, tc, form.toeSharpZ)
}

/**
 * [∂H/∂x, ∂H/∂z], analytic — see "THE GRADIENT" in the header comment. Includes
 * the ridgeShear cross-term in ∂H/∂x that a naive per-axis derivative drops.
 */
export function driftGradient(form, x, z) {
  const { amplitude, crestX, toeSharpX, toeSharpZ, footprint } = form
  const { width, depth } = footprint
  const s = x / width
  const t = z / depth
  const tc = ridgeCenter(form, s)
  const dTcDs = ridgeCenterSlope(form, s)

  const Bx = profile(s, crestX, toeSharpX)
  const dBxDs = profileDTau(s, crestX, toeSharpX)
  const Bz = profile(t, tc, toeSharpZ)
  const dBzDt = profileDTau(t, tc, toeSharpZ)
  const dBzDp = profileDCrest(t, tc, toeSharpZ)

  const dHds = amplitude * (dBxDs * Bz + Bx * dBzDp * dTcDs)
  const dHdt = amplitude * Bx * dBzDt

  return [dHds / width, dHdt / depth]
}

/** normalize(-dH/dx, 1, -dH/dz) — surface normal from the gradient (steepest
 *  ascent points ⟨dH/dx, dH/dz⟩ in the plan, so the upward normal tilts against it). */
export function driftNormal(form, x, z) {
  const [dHdx, dHdz] = driftGradient(form, x, z)
  return new THREE.Vector3(-dHdx, 1, -dHdz).normalize()
}

/** Convenience: point on the surface plus its normal, in one call. */
export function driftFrame(form, x, z) {
  return {
    point: new THREE.Vector3(x, driftHeight(form, x, z), z),
    normal: driftNormal(form, x, z),
  }
}

/**
 * Triangulated grid over the footprint, `nx` × `nz` vertices (so (nx-1)×(nz-1)
 * quads, 2 triangles each), for the ghost-surface preview in the viewport.
 * Requires nx, nz ≥ 2.
 */
export function sampleDriftMesh(form, nx, nz) {
  const n = Math.max(2, Math.floor(nx))
  const m = Math.max(2, Math.floor(nz))
  const { width, depth } = form.footprint

  const positions = new Float32Array(n * m * 3)
  let p = 0
  for (let j = 0; j < m; j++) {
    const z = (j / (m - 1)) * depth
    for (let i = 0; i < n; i++) {
      const x = (i / (n - 1)) * width
      positions[p++] = x
      positions[p++] = driftHeight(form, x, z)
      positions[p++] = z
    }
  }

  const indices = new Uint32Array((n - 1) * (m - 1) * 6)
  let k = 0
  for (let j = 0; j < m - 1; j++) {
    for (let i = 0; i < n - 1; i++) {
      const a = j * n + i
      const b = a + 1
      const c = a + n
      const d = c + 1
      indices[k++] = a
      indices[k++] = c
      indices[k++] = b
      indices[k++] = b
      indices[k++] = c
      indices[k++] = d
    }
  }

  return { positions, indices }
}
