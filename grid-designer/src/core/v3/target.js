/**
 * grid-designer v3 — the TARGET SURFACE: what the panels are actually asked to be.
 *
 * HEADLESS ZONE (src/core/): pure functions, importable from plain node.
 *   - explicit `.js` extensions on ALL relative imports
 *   - may import `three` math classes only; never components / store / DOM
 *   - same config in → byte-identical output out
 *
 * =============================================================================
 * WHY THIS MODULE EXISTS
 * =============================================================================
 * `form.js` authors a SMOOTH drift. Rigid flat panels cannot be a smooth drift:
 * a 60cm panel deviates from a curved target by roughly (30²/2)·curvature
 * wherever you put it, which is what forced the graded edges to hover a few cm
 * off the floor, opened the joints, drove the housings into each other past
 * ~40cm of amplitude, and starved the tiler of any place flat enough to lay a
 * rigid 121cm plate.
 *
 * Every one of those is an artifact of panelizing a shape chosen without
 * reference to the panels. So don't do that. **The target is allowed to be
 * angular.** Quantize the drift into PLANAR FACETS aligned to the panel
 * lattice, and the panels can be the surface exactly rather than approximate it:
 *
 *   - tiles inside one facet are COPLANAR, so the joints between them stay at
 *     exactly `gap` instead of wedging open
 *   - all the folding collects onto facet BOUNDARIES, which is where the
 *     physical joints and their connectors already are
 *   - a plate's sagitta inside a facet is ZERO, so plates become placeable at
 *     any amplitude instead of vanishing as the drift steepens
 *   - the graded edges can actually reach the floor, because a planar facet has
 *     a straight edge and the floor is straight
 *
 * This is the system considered as a whole rather than a shape run through a
 * panelizer — and a wind-carved drift genuinely is faceted, so the angular
 * reading is the more honest one anyway.
 *
 * `form.angularity` blends between the two: 0 = the smooth drift (every problem
 * above, faithfully reported), 1 = fully faceted. `form.facetCells` sets how
 * many cells share a facet — 1 gives every panel its own plane (maximum
 * articulation, a fold at every joint), larger values give broader planes with
 * crisper creases between them.
 *
 * =============================================================================
 * THE UNROLL, AND WHY IT IS SOLVED TWICE
 * =============================================================================
 * The sheet is longer than its shadow, so material → plan is an ARC-LENGTH map
 * (dx/du = 1/√(1 + (∂H/∂x)²)); see `buildUnroll`. That map depends on the
 * surface's gradient, and the faceted surface's gradient depends on where the
 * facets fall in plan, which depends on the map. Rather than iterate to a
 * fixed point (and lose byte-reproducibility), it is solved in exactly two
 * stages:
 *
 *   1. unroll against the SMOOTH gradient → gives plan positions good enough to
 *      place the facet boundaries
 *   2. fit the facet planes, then unroll AGAIN against the FACETED gradient →
 *      the map the placement actually uses
 *
 * Two stages, always, so the result is deterministic.
 */

import * as THREE from 'three'
import { normalizeForm, driftHeight, driftGradient, driftNormal } from './form.js'

/** Sub-steps per unroll cell; see `buildUnroll`. */
const UNROLL_STEPS = 64
/** Passes of the coupled u/v arc-length relaxation. Fixed for determinism. */
const UNROLL_PASSES = 3
/**
 * Plane-fit samples per facet, per axis. **2 means corners only, and that is
 * deliberate.**
 *
 * Adjacent facets share two corners. A plane fitted to a facet's CORNERS passes
 * (nearly) through those shared points, so neighbouring facets meet in a
 * CREASE. A plane fitted to the facet's interior instead drifts away from the
 * corners, and neighbours then meet in a STEP — a height discontinuity that the
 * panels straddling the boundary have to swallow as an open joint. Measured on
 * the default drift at amplitude 60, interior fitting (5×5) made the worst
 * joint deviation WORSE than no faceting at all.
 *
 * Exact continuity would need the four corners to be coplanar, which in general
 * they are not; the residual non-coplanarity over one facet is what is left, and
 * it is small compared with the step it replaces.
 */
const FACET_SAMPLES = 2

/**
 * Build the material → plan arc-length map for an arbitrary gradient field.
 *
 * `gradAt(x, z)` returns `[∂H/∂x, ∂H/∂z]`. The two integrations are coupled —
 * x(u) depends on which z you walk along and vice versa — so they are relaxed
 * together for a FIXED number of passes.
 *
 * Both graded edges are fixed points of the map whenever the surface is zero
 * along them: arc length equals material length there, so u = x and v = z
 * exactly. That is why the wall and window edges land where the brief says.
 */
export function buildUnroll(gradAt, uMax, vMax) {
  const nu = UNROLL_STEPS
  const nv = UNROLL_STEPS
  const du = uMax / nu
  const dv = vMax / nv
  const idx = (iu, iv) => iu * (nv + 1) + iv
  const xs = new Float64Array((nu + 1) * (nv + 1))
  const zs = new Float64Array((nu + 1) * (nv + 1))
  for (let iu = 0; iu <= nu; iu++) {
    for (let iv = 0; iv <= nv; iv++) {
      xs[idx(iu, iv)] = iu * du
      zs[idx(iu, iv)] = iv * dv
    }
  }
  for (let pass = 0; pass < UNROLL_PASSES; pass++) {
    for (let iv = 0; iv <= nv; iv++) {
      let x = 0
      xs[idx(0, iv)] = 0
      for (let iu = 1; iu <= nu; iu++) {
        const gx = gradAt(x, zs[idx(iu - 1, iv)])[0]
        x += du / Math.sqrt(1 + gx * gx)
        xs[idx(iu, iv)] = x
      }
    }
    for (let iu = 0; iu <= nu; iu++) {
      let z = 0
      zs[idx(iu, 0)] = 0
      for (let iv = 1; iv <= nv; iv++) {
        const gz = gradAt(xs[idx(iu, iv - 1)], z)[1]
        z += dv / Math.sqrt(1 + gz * gz)
        zs[idx(iu, iv)] = z
      }
    }
  }
  return function planAt(u, v) {
    const fu = Math.min(nu, Math.max(0, u / du))
    const fv = Math.min(nv, Math.max(0, v / dv))
    const iu = Math.min(nu - 1, Math.floor(fu))
    const iv = Math.min(nv - 1, Math.floor(fv))
    const tu = fu - iu
    const tv = fv - iv
    const s = (arr) =>
      arr[idx(iu, iv)] * (1 - tu) * (1 - tv) +
      arr[idx(iu + 1, iv)] * tu * (1 - tv) +
      arr[idx(iu, iv + 1)] * (1 - tu) * tv +
      arr[idx(iu + 1, iv + 1)] * tu * tv
    return { x: s(xs), z: s(zs) }
  }
}

/**
 * Least-squares plane y = a·x + b·z + c through sampled points.
 * Falls back to a horizontal plane at the mean height if the system is
 * degenerate (a facet collapsed to a line or a point).
 */
function fitPlane(pts) {
  let sxx = 0, sxz = 0, sx = 0, szz = 0, sz = 0, n = 0
  let sxy = 0, szy = 0, sy = 0
  for (const [x, z, y] of pts) {
    sxx += x * x; sxz += x * z; sx += x
    szz += z * z; sz += z
    sxy += x * y; szy += z * y; sy += y
    n++
  }
  const m = [[sxx, sxz, sx], [sxz, szz, sz], [sx, sz, n]]
  const rhs = [sxy, szy, sy]
  // 3×3 solve by Cramer's rule.
  const det = (M) =>
    M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
    M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
    M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0])
  const D = det(m)
  if (!Number.isFinite(D) || Math.abs(D) < 1e-9) {
    return { a: 0, b: 0, c: n > 0 ? sy / n : 0 }
  }
  const col = (k) => m.map((row, i) => row.map((v, j) => (j === k ? rhs[i] : v)))
  return { a: det(col(0)) / D, b: det(col(1)) / D, c: det(col(2)) / D }
}

/**
 * The target surface the panels are asked to realise.
 *
 * @param {object} cfg a NORMALIZED v3 config (caller normalizes; this module
 *   does not import schema.js, to keep the dependency direction one-way)
 * @returns {{
 *   planAt: (u:number, v:number) => {x:number, z:number},
 *   frameAtMaterial: (u:number, v:number) => {point:number[], normal:THREE.Vector3},
 *   heightAtMaterial: (u:number, v:number) => number,
 *   facetIndexAt: (u:number, v:number) => string,
 *   angularity: number, facetCells: number, facetCount: number,
 * }}
 */
export function buildTarget(cfg) {
  const form = normalizeForm(cfg.form)
  const pitch = cfg.cell.size + cfg.gap
  const { cols, rows } = cfg.sheet
  const uMax = cols * pitch - cfg.gap
  const vMax = rows * pitch - cfg.gap
  const angularity = form.angularity
  const facetCells = form.facetCells

  // --- stage 1: unroll against the smooth gradient -------------------------
  const smoothGrad = (x, z) => driftGradient(form, x, z)
  let planAt = buildUnroll(smoothGrad, uMax, vMax)

  // --- facet planes, indexed by their cell block ---------------------------
  // Facet boundaries sit on CELL boundaries in material space, so they coincide
  // with real panel joints — that is the whole point: creases land where there
  // is already a physical joint to absorb them.
  const facetKey = (u, v) => {
    const i = Math.min(cols - 1, Math.max(0, Math.floor(u / pitch)))
    const j = Math.min(rows - 1, Math.max(0, Math.floor(v / pitch)))
    return `${Math.floor(i / facetCells)},${Math.floor(j / facetCells)}`
  }

  const buildPlanes = (map) => {
    const planes = new Map()
    const nfi = Math.ceil(cols / facetCells)
    const nfj = Math.ceil(rows / facetCells)
    for (let fi = 0; fi < nfi; fi++) {
      for (let fj = 0; fj < nfj; fj++) {
        const i0 = fi * facetCells
        const j0 = fj * facetCells
        const i1 = Math.min(cols, i0 + facetCells)
        const j1 = Math.min(rows, j0 + facetCells)
        const u0 = i0 * pitch
        const v0 = j0 * pitch
        const u1 = (i1 - 1) * pitch + cfg.cell.size
        const v1 = (j1 - 1) * pitch + cfg.cell.size
        const pts = []
        for (let a = 0; a < FACET_SAMPLES; a++) {
          for (let b = 0; b < FACET_SAMPLES; b++) {
            const u = u0 + ((u1 - u0) * a) / (FACET_SAMPLES - 1)
            const v = v0 + ((v1 - v0) * b) / (FACET_SAMPLES - 1)
            const { x, z } = map(u, v)
            pts.push([x, z, driftHeight(form, x, z)])
          }
        }
        planes.set(`${fi},${fj}`, fitPlane(pts))
      }
    }
    return planes
  }

  let planes = buildPlanes(planAt)

  // --- stage 2: re-unroll against the FACETED gradient ---------------------
  // The faceted surface has a different slope field, so the arc-length map
  // differs. Solved exactly once more — see the header on why not to iterate.
  if (angularity > 0) {
    const facetedGrad = (x, z) => {
      // Invert approximately: the plan point's facet is found via the stage-1
      // map, which is close enough to pick the right block.
      const pl = planes.get(facetKey(x, z)) ?? { a: 0, b: 0 }
      const sg = smoothGrad(x, z)
      return [
        sg[0] * (1 - angularity) + pl.a * angularity,
        sg[1] * (1 - angularity) + pl.b * angularity,
      ]
    }
    planAt = buildUnroll(facetedGrad, uMax, vMax)
    planes = buildPlanes(planAt)
  }

  const facetIndexAt = (u, v) => facetKey(u, v)

  function frameAtMaterial(u, v) {
    const { x, z } = planAt(u, v)
    const smoothY = driftHeight(form, x, z)
    const smoothN = driftNormal(form, x, z).clone().normalize()
    if (angularity <= 0) return { point: [x, smoothY, z], normal: smoothN }

    const pl = planes.get(facetKey(u, v)) ?? { a: 0, b: 0, c: 0 }
    const facetY = pl.a * x + pl.b * z + pl.c
    const facetN = new THREE.Vector3(-pl.a, 1, -pl.b).normalize()

    if (angularity >= 1) return { point: [x, facetY, z], normal: facetN }
    const y = smoothY * (1 - angularity) + facetY * angularity
    const n = smoothN.clone().multiplyScalar(1 - angularity)
      .addScaledVector(facetN, angularity)
    if (n.lengthSq() < 1e-12) n.copy(facetN)
    return { point: [x, y, z], normal: n.normalize() }
  }

  return {
    planAt,
    frameAtMaterial,
    heightAtMaterial: (u, v) => frameAtMaterial(u, v).point[1],
    facetIndexAt,
    angularity,
    facetCells,
    facetCount: planes.size,
  }
}
