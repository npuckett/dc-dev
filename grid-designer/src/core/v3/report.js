/**
 * grid-designer v3 — the joint / fit report: what the connectors have to absorb.
 *
 * HEADLESS ZONE (src/core/): pure functions, importable from plain node.
 *   - explicit `.js` extensions on ALL relative imports
 *   - may import `three` math classes only; never components / store / DOM
 *   - same layout in → byte-identical output out
 *
 * =============================================================================
 * WHAT THIS IS FOR
 * =============================================================================
 * The durable finding of the whole project (HANDOFF.md §2.1) is: don't solve
 * rigid origami — place panels deterministically and MEASURE what the physical
 * connectors have to take up. This module is that measurement, and it is the
 * tool's primary output. Everything else exists to feed it.
 *
 * Four families of number, each answering a different buildability question:
 *
 *   JOINTS      — per adjacent tile pair: how far from the nominal `gap` does
 *                 this joint actually sit, how skewed is it, how far does it
 *                 fold. This is the connector spec, one row per connector.
 *   HOLONOMY    — in 'chain' placement the tree edges are exact by
 *                 construction, so all the closure error lands on the
 *                 cycle-closing edges. Summarised separately, because a mean
 *                 over all joints hides it completely.
 *   FIT         — how far the realised panels sit from the target surface, and
 *                 how hard each plate is working (its sagitta).
 *   COLLISIONS  — panels interpenetrating. v2 never needed this; v3 does, and
 *                 it is a hard buildability limit rather than a warning.
 *
 * A NOTE ON THE THREE GAP NUMBERS (inherited quirk, HANDOFF.md §5.3): for a
 * strongly skewed joint `gapMid` can be SMALLER than `gapMin`, because the
 * midpoints of two edges can be closer than their endpoints. That is correct.
 * Do not present min/mid/max as an ordered triple.
 */

import * as THREE from 'three'
import { normalizeConfig } from './schema.js'
import { buildTarget } from './target.js'
import { solveLayout, tileOBB } from './placement.js'
import { findCollisions } from './collide.js'

const DEG = 180 / Math.PI

/** Ignore contact shallower than this when calling something a collision. The
 *  panels are MEANT to nearly touch across a ~1cm joint, so a zero-tolerance
 *  overlap test would flag an entire healthy design. */
export const COLLISION_MIN_DEPTH_CM = 0.05

function r(v) {
  const out = Math.round(v * 1e9) / 1e9
  return out === 0 ? 0 : out
}

/**
 * The world-space segment where a tile's shared material edge lies, on its lit
 * face. Exact: a tile is rigid and planar, spanning `uLen` along êu and `vLen`
 * along êv, so any material point maps to world by construction.
 */
function edgeSegment(tile, edge, isA, samples) {
  const runAxis = edge.axis
  const sepAxis = runAxis === 'u' ? 'v' : 'u'
  const sepBoundary = isA ? edge.edge.a : edge.edge.b
  const uc = tile.uv.u0 + tile.uv.uLen / 2
  const vc = tile.uv.v0 + tile.uv.vLen / 2
  const sepCenter = sepAxis === 'u' ? uc : vc
  const runCenter = runAxis === 'u' ? uc : vc
  const eSep = sepAxis === 'u' ? tile.eu : tile.ev
  const eRun = runAxis === 'u' ? tile.eu : tile.ev
  const out = []
  for (let k = 0; k < samples; k++) {
    const f = samples === 1 ? 0.5 : k / (samples - 1)
    const s = edge.edge.from + f * (edge.edge.to - edge.edge.from)
    out.push(new THREE.Vector3(
      tile.position[0] + (sepBoundary - sepCenter) * eSep[0] + (s - runCenter) * eRun[0],
      tile.position[1] + (sepBoundary - sepCenter) * eSep[1] + (s - runCenter) * eRun[1],
      tile.position[2] + (sepBoundary - sepCenter) * eSep[2] + (s - runCenter) * eRun[2],
    ))
  }
  return out
}

/** Samples along a joint when measuring its gap profile. */
const JOINT_SAMPLES = 5

/**
 * Measure every joint, plus whole-surface fit and collisions.
 *
 * @param {object} config raw or normalized v3 config
 * @param {object} [layout] a layout from solveLayout; solved here if omitted
 * @returns {object} see the file header
 */
export function buildReport(config, layout = null) {
  const cfg = normalizeConfig(config)
  const L = layout ?? solveLayout(cfg)
  const target = buildTarget(cfg)
  const byId = new Map(L.tiles.map((t) => [t.id, t]))
  const treeEdges = new Set(L.tree.treeEdges)
  const tol = cfg.gapTolerance

  // --- joints --------------------------------------------------------------
  const joints = []
  L.adjacency.forEach((edge, idx) => {
    const A = byId.get(edge.a)
    const B = byId.get(edge.b)
    if (!A?.position || !B?.position) return

    const pa = edgeSegment(A, edge, true, JOINT_SAMPLES)
    const pb = edgeSegment(B, edge, false, JOINT_SAMPLES)
    const dists = pa.map((p, k) => p.distanceTo(pb[k]))
    const gapMin = Math.min(...dists)
    const gapMax = Math.max(...dists)
    const gapMid = dists[(JOINT_SAMPLES - 1) / 2 | 0]

    // Skew: angle between the two edge directions.
    const da = pa[pa.length - 1].clone().sub(pa[0])
    const db = pb[pb.length - 1].clone().sub(pb[0])
    const skewDeg = (da.lengthSq() > 1e-12 && db.lengthSq() > 1e-12)
      ? Math.acos(Math.min(1, Math.max(-1, da.normalize().dot(db.normalize())))) * DEG
      : 0

    // Dihedral: fold across the joint, from the two face normals.
    const na = new THREE.Vector3(...A.normal)
    const nb = new THREE.Vector3(...B.normal)
    const dihedralDeg = Math.acos(Math.min(1, Math.max(-1, na.dot(nb)))) * DEG

    const deviationCm = Math.max(Math.abs(gapMin - cfg.gap), Math.abs(gapMax - cfg.gap))
    const flags = []
    if (deviationCm > tol) flags.push('W_GAP_OUT_OF_TOLERANCE')
    if (gapMin < 0.05) flags.push('W_JOINT_PINCHED')

    joints.push({
      index: idx,
      a: edge.a,
      b: edge.b,
      axis: edge.axis,
      treeEdge: treeEdges.has(idx),
      materialLength: r(edge.materialLength),
      gapMin: r(gapMin),
      gapMid: r(gapMid),
      gapMax: r(gapMax),
      deviationCm: r(deviationCm),
      skewDeg: r(skewDeg),
      dihedralDeg: r(dihedralDeg),
      flags,
    })
  })

  const devs = joints.map((j) => j.deviationCm)
  const stat = (xs) => xs.length
    ? { worst: r(Math.max(...xs)), mean: r(xs.reduce((a, b) => a + b, 0) / xs.length), count: xs.length }
    : { worst: 0, mean: 0, count: 0 }

  // --- holonomy ------------------------------------------------------------
  // Only meaningful in 'chain' mode, where tree edges are exact by construction
  // and the closure error therefore concentrates on the cycle-closing edges. In
  // 'surface-fit' every joint shares the error, so there is no tree to compare
  // against and the split is reported as null rather than as a misleading zero.
  const holonomy = L.mode === 'chain'
    ? {
      mode: 'chain',
      treeEdges: stat(joints.filter((j) => j.treeEdge).map((j) => j.deviationCm)),
      cycleEdges: stat(joints.filter((j) => !j.treeEdge).map((j) => j.deviationCm)),
      worstJoint: joints.filter((j) => !j.treeEdge)
        .sort((a, b) => b.deviationCm - a.deviationCm)[0] ?? null,
    }
    : { mode: L.mode, treeEdges: null, cycleEdges: null, worstJoint: null }

  // --- fit against the target ---------------------------------------------
  // Measured on the tile's UNDERSIDE, which is the face the target describes
  // (the lit face sits one housing thickness in front of it). Reported as a
  // spread about the mean, because the assembly is settled onto the floor by a
  // global rigid lift and an absolute residual would mostly measure that lift.
  const residuals = []
  for (const t of L.tiles) {
    if (!t.position) continue
    const u = t.uv.u0 + t.uv.uLen / 2
    const v = t.uv.v0 + t.uv.vLen / 2
    residuals.push(t.position[1] - target.frameAtMaterial(u, v).point[1])
  }
  const rMean = residuals.length ? residuals.reduce((a, b) => a + b, 0) / residuals.length : 0
  const rSigma = residuals.length
    ? Math.sqrt(residuals.reduce((a, b) => a + (b - rMean) ** 2, 0) / residuals.length)
    : 0

  const plates = L.tiles.filter((t) => t.type === '2x4')
  const fit = {
    shapeResidualSigmaCm: r(rSigma),
    plateCount: plates.length,
    tileCount: L.tiles.length,
    worstPlateSagittaCm: r(plates.length ? Math.max(...plates.map((t) => t.sagittaCm ?? 0)) : 0),
    plateFitToleranceCm: cfg.tiling.plateFitToleranceCm,
    angularity: cfg.form.angularity,
    facetCells: cfg.form.facetCells,
    facetCount: target.facetCount,
  }

  // --- collisions ----------------------------------------------------------
  const placed = L.tiles.filter((t) => t.position)
  const hits = findCollisions(placed.map(tileOBB), { minDepthCm: COLLISION_MIN_DEPTH_CM })
  const collisions = hits.map((h) => ({
    a: placed[h.i].id,
    b: placed[h.j].id,
    depthCm: r(h.depthCm),
  })).sort((x, y) => y.depthCm - x.depthCm || (x.a < y.a ? -1 : 1))

  return {
    joints,
    summary: {
      gap: cfg.gap,
      gapToleranceCm: tol,
      ...stat(devs),
      flagged: joints.filter((j) => j.flags.length > 0).length,
      pinched: joints.filter((j) => j.flags.includes('W_JOINT_PINCHED')).length,
      worstDihedralDeg: r(joints.length ? Math.max(...joints.map((j) => j.dihedralDeg)) : 0),
      worstSkewDeg: r(joints.length ? Math.max(...joints.map((j) => j.skewDeg)) : 0),
    },
    holonomy,
    fit,
    collisions,
    support: L.support,
    bounds: L.bounds,
    warnings: L.warnings,
    violations: L.violations,
  }
}
