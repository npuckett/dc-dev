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
import { solveLayout, tileOBB, jointEdgePoint } from './placement.js'
import { findCollisions, aabbOverlap, obbPenetration } from './collide.js'
import {
  solveConnectors,
  connectorOBB,
  connectorStationFlags,
  CONNECTOR_LIMITS,
  BLOCKED_CODE,
  REDUCED_CODE,
} from './connectors.js'
import { POWER_SUPPLY } from '../../config.js'

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
 * face, sampled `samples` times end to end.
 *
 * The per-point mapping now lives in placement.js as `jointEdgePoint` — moved
 * there when connectors.js needed the same line, so the two cannot drift apart.
 */
function edgeSegment(tile, edge, isA, samples) {
  const out = []
  for (let k = 0; k < samples; k++) {
    const f = samples === 1 ? 0.5 : k / (samples - 1)
    const s = edge.edge.from + f * (edge.edge.to - edge.edge.from)
    out.push(jointEdgePoint(tile, edge, isA, s))
  }
  return out
}

/** Samples along a joint when measuring its gap profile. */
const JOINT_SAMPLES = 5

// =============================================================================
// CONNECTORS (P11)
// =============================================================================
/**
 * A connector's OBB necessarily overlaps the two panels it GRIPS — the channel
 * closes around the rim, and the slot is a void inside the bounding box. So
 * those two pairs are excluded by construction, and what remains is what a
 * clash actually means here: a part fouling a THIRD panel, or two parts fouling
 * each other. Both are reachable at a tight fold near a panel corner.
 */
const CONNECTOR_CLASH_MIN_DEPTH_CM = 0.05

/**
 * Group the stations into printable part types.
 *
 * A part is fully described by `(spanStart, spanEnd, fold, length)`, and two
 * stations can share a part when those round into the same bin. Two
 * canonicalizations before binning, both because the physical part allows it:
 *
 *   - the cross-section is MIRROR-SYMMETRIC about the gap's centre line, so
 *     which panel is `a` and which is `b` does not distinguish two parts;
 *   - a part can be fitted either way round along the joint, so `(2cm, 5cm)`
 *     and `(5cm, 2cm)` are the same object rotated. Ordering the pair collapses
 *     them.
 *
 * The FOLD is not canonicalized. Convex and concave are genuinely different
 * parts — one closes the hooks, the other spreads them.
 *
 * This is the plate budget's question in a different currency (README, "the
 * trade"): tight bins mean every joint gets geometry that fits it and you print
 * a lot of unique things; loose bins mean a handful of types and some joints
 * forced onto a neighbour's shape. The tool measures the forcing. It cannot
 * choose the tolerance.
 */
function buildKit(stations, { binSpanCm, binAngleDeg }) {
  const groups = new Map()

  for (const st of stations) {
    const lo = Math.min(st.spanStartCm, st.spanEndCm)
    const hi = Math.max(st.spanStartCm, st.spanEndCm)
    const bLo = Math.round(lo / binSpanCm)
    const bHi = Math.round(hi / binSpanCm)
    const bFold = Math.round(st.foldDeg / binAngleDeg)
    const bLen = Math.round(st.lengthCm / 0.1)
    const key = `${bLo}|${bHi}|${bFold}|${bLen}`

    if (!groups.has(key)) {
      groups.set(key, {
        key,
        // The representative part IS the bin centre, not the first station that
        // landed in it — otherwise the kit would depend on iteration order and
        // half the group could sit further from the part than the bin allows.
        spanStartCm: r(bLo * binSpanCm),
        spanEndCm: r(bHi * binSpanCm),
        foldDeg: r(bFold * binAngleDeg),
        lengthCm: r(bLen * 0.1),
        stations: [],
      })
    }
    groups.get(key).stations.push(st)
  }

  // Sorted by count (a kit is read biggest-first), then by the bin key, so the
  // ids are stable for a given design.
  const ordered = [...groups.values()].sort(
    (a, b) => b.stations.length - a.stations.length || (a.key < b.key ? -1 : 1),
  )

  return ordered.map((g, idx) => {
    let worstSpan = 0
    let worstFold = 0
    for (const st of g.stations) {
      const lo = Math.min(st.spanStartCm, st.spanEndCm)
      const hi = Math.max(st.spanStartCm, st.spanEndCm)
      worstSpan = Math.max(worstSpan, Math.abs(lo - g.spanStartCm), Math.abs(hi - g.spanEndCm))
      worstFold = Math.max(worstFold, Math.abs(st.foldDeg - g.foldDeg))
    }
    return {
      partId: `P${String(idx).padStart(2, '0')}`,
      count: g.stations.length,
      spanStartCm: g.spanStartCm,
      spanEndCm: g.spanEndCm,
      foldDeg: g.foldDeg,
      lengthCm: g.lengthCm,
      // What each joint using this part is forced to absorb by not getting its
      // own exact geometry. Bounded by the bin half-width by construction; the
      // number is here so the bin size can be chosen against a consequence.
      worstSpanErrorCm: r(worstSpan),
      worstFoldErrorDeg: r(worstFold),
      stationIds: g.stations.map((s) => s.id),
      joints: [...new Set(g.stations.map((s) => s.jointIndex))].sort((a, b) => a - b),
    }
  })
}

/**
 * Everything about the printed parts: where they go, whether each is buildable,
 * what fouls what, and how few distinct ones the design can be built from.
 */
function buildConnectorReport(cfg, L, placedTiles, tileBoxes) {
  const C = solveConnectors(cfg, L)
  const limits = CONNECTOR_LIMITS

  const boxes = C.stations.map((st) => connectorOBB(st))

  // --- clash: a part against a panel it does NOT grip, or against another part
  const tileIndex = new Map(placedTiles.map((t, i) => [t.id, i]))
  const clashes = []
  C.stations.forEach((st, i) => {
    for (let t = 0; t < tileBoxes.length; t++) {
      if (t === tileIndex.get(st.a) || t === tileIndex.get(st.b)) continue
      if (!aabbOverlap(boxes[i], tileBoxes[t])) continue
      const pen = obbPenetration(boxes[i], tileBoxes[t])
      if (pen && pen.depthCm > CONNECTOR_CLASH_MIN_DEPTH_CM) {
        clashes.push({ station: st.id, against: placedTiles[t].id, kind: 'panel', depthCm: r(pen.depthCm) })
      }
    }
  })
  for (let i = 0; i < boxes.length; i++) {
    for (let j = i + 1; j < boxes.length; j++) {
      if (!aabbOverlap(boxes[i], boxes[j])) continue
      const pen = obbPenetration(boxes[i], boxes[j])
      if (pen && pen.depthCm > CONNECTOR_CLASH_MIN_DEPTH_CM) {
        clashes.push({
          station: C.stations[i].id,
          against: C.stations[j].id,
          kind: 'connector',
          depthCm: r(pen.depthCm),
        })
      }
    }
  }
  clashes.sort((x, y) => y.depthCm - x.depthCm || (x.station < y.station ? -1 : 1))
  const clashedStations = new Set(clashes.map((c) => c.station))

  // --- per-station flags ---------------------------------------------------
  // Everything decidable from the station alone lives in connectors.js, so it
  // can be tested against synthetic stations; clash is the only rule that needs
  // the rest of the assembly and so is added here.
  const stations = C.stations.map((st) => {
    const flags = connectorStationFlags(st, limits)
    if (clashedStations.has(st.id)) flags.push('W_CONNECTOR_CLASH')
    return { ...st, flags }
  })

  // A joint held by ONE part is free to rotate about it, and with no
  // substructure that is a real degree of freedom rather than a detail.
  const singles = C.perJoint.filter((pj) => pj.count < 2)

  const kit = buildKit(C.stations, cfg.connectors)

  return {
    stations,
    perJoint: C.perJoint,
    kit,
    clashes,
    warnings: [
      ...C.warnings,
      ...singles.map((pj) => ({
        code: 'W_JOINT_SINGLE_CONNECTOR',
        joint: pj.jointIndex,
        a: pj.a,
        b: pj.b,
        message: `joint ${pj.a}–${pj.b} carries one part, so it is a hinge rather than a fixture`,
      })),
    ],
    limits,
    summary: {
      count: stations.length,
      jointCount: C.perJoint.length,
      partTypes: kit.length,
      lengthCm: cfg.connectors.lengthCm,
      binSpanCm: cfg.connectors.binSpanCm,
      binAngleDeg: cfg.connectors.binAngleDeg,
      flagged: stations.filter((s) => s.flags.length > 0).length,
      infeasible: stations.filter((s) => s.flags.includes('W_CONNECTOR_INFEASIBLE')).length,
      clashes: clashes.length,
      singleConnectorJoints: singles.length,
      // Joints the power supply leaves with no connector at all, and joints it
      // merely thins out. Separated because they are different failures: one is
      // a structural hole, the other is a reduction.
      blockedJoints: C.warnings.filter((w) => w.code === BLOCKED_CODE).length,
      reducedJoints: C.warnings.filter((w) => w.code === REDUCED_CODE).length,
      clearEndCm: r((60 - POWER_SUPPLY.length) / 2),
      spanCm: stations.length
        ? { min: r(Math.min(...stations.map((s) => s.spanMinCm))), max: r(Math.max(...stations.map((s) => s.spanMaxCm))) }
        : { min: 0, max: 0 },
      worstFoldDeg: r(stations.length ? Math.max(...stations.map((s) => Math.abs(s.foldDeg))) : 0),
      worstTwistDeg: r(stations.length ? Math.max(...stations.map((s) => s.twistDeg)) : 0),
      // The number `lengthCm` exists to keep small — how much a single part has
      // to wedge along its own length.
      worstSpanSpreadCm: r(stations.length ? Math.max(...stations.map((s) => s.spanSpreadCm)) : 0),
      worstBinSpanErrorCm: r(kit.length ? Math.max(...kit.map((k) => k.worstSpanErrorCm)) : 0),
      worstBinFoldErrorDeg: r(kit.length ? Math.max(...kit.map((k) => k.worstFoldErrorDeg)) : 0),
    },
  }
}

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
  const tileBoxes = placed.map(tileOBB)
  const hits = findCollisions(tileBoxes, { minDepthCm: COLLISION_MIN_DEPTH_CM })
  const collisions = hits.map((h) => ({
    a: placed[h.i].id,
    b: placed[h.j].id,
    depthCm: r(h.depthCm),
  })).sort((x, y) => y.depthCm - x.depthCm || (x.a < y.a ? -1 : 1))

  // --- connectors ----------------------------------------------------------
  const connectors = buildConnectorReport(cfg, L, placed, tileBoxes)

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
    connectors,
    support: L.support,
    bounds: L.bounds,
    warnings: L.warnings,
    violations: L.violations,
  }
}
