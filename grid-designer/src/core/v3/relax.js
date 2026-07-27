/**
 * grid-designer v3 — relaxing a placed layout into the connectors' envelope.
 *
 * HEADLESS ZONE (src/core/): pure functions, importable from plain node.
 *   - explicit `.js` extensions on ALL relative imports
 *   - may import `three` math classes only; never components / store / DOM
 *   - same layout in → byte-identical output out
 *
 * =============================================================================
 * WHY THIS EXISTS, AND WHY IT IS THE ONLY LEVER LEFT
 * =============================================================================
 * The connector cannot be redesigned out of its hard cases. Measured in
 * connectors.js: the part fouls a panel only 3–6° of fold before the two PANELS
 * collide with each other anyway (15° vs 19° at a 0.4cm gap, 45° vs 49° at 1cm),
 * and no connector dimension moves that boundary. So a joint outside the
 * envelope is not a hardware problem — it is a placement problem, and the only
 * remaining lever is to move the panels.
 *
 * =============================================================================
 * WHAT IS RELAXED, AND WHAT IS NOT
 * =============================================================================
 * THE PLACEMENTS, NEVER THE FORM. `form` is authored intent; moving it would
 * quietly redesign the drift the user asked for. What moves is where the rigid
 * panels sit on it — which `surface-fit` already does per tile, deciding each
 * tile independently. This does the same thing GLOBALLY: every joint pulls on
 * its two tiles, every tile pulls back toward where the surface put it, and the
 * iteration finds where those balance.
 *
 * The tiling is untouched too. Which cells are plates is a kit decision, not a
 * geometric one, and re-tiling mid-relaxation would make the result depend on
 * the order of two unrelated solvers.
 *
 * =============================================================================
 * DETERMINISM
 * =============================================================================
 * Gauss–Seidel, a FIXED iteration count, joints visited in adjacency-index order
 * and tiles in tile order. No convergence test, no randomness, no early exit —
 * the same reasoning as placement.js's "exactly 3 fixed-point iterations, not
 * until converged". A relaxation that stops when it feels finished is one whose
 * output depends on floating-point noise.
 *
 * =============================================================================
 * IT MUST BE ABLE TO FAIL
 * =============================================================================
 * A relaxation that always succeeds has stopped telling you anything. Sometimes
 * the honest answer is "this form is too aggressive for a 1cm joint", and the
 * tool's whole reason for existing is to say so rather than smooth it away. So:
 *
 *   - `unresolved` lists every joint still outside the envelope afterwards,
 *     with what it is still short by;
 *   - `moved` reports how far each tile was displaced from where the surface
 *     put it, because that displacement is the PRICE, and a design that only
 *     works after 8cm of shoving is not the design that was authored.
 *
 * Both are reported. Neither is a veto — same contract as every other
 * measurement in this project.
 */

import * as THREE from 'three'
import { finalizeTileGeometry, jointEdgePoint, jointEdgeRun } from './placement.js'
import {
  fastenerGapNeededCm,
  foldLimitDeg,
  CONNECTOR_PROFILE,
} from './connectors.js'

const DEG = 180 / Math.PI

function r(v) {
  const out = Math.round(v * 1e9) / 1e9
  return out === 0 ? 0 : out
}
const rv = (v) => [r(v.x), r(v.y), r(v.z)]

/**
 * What a joint has to satisfy for its connector to exist. Read straight off
 * connectors.js — the envelope is the CONNECTOR's, and if the part changes this
 * follows it rather than restating it.
 */
export function jointEnvelope(profile = CONNECTOR_PROFILE) {
  return {
    minGapCm: fastenerGapNeededCm(profile),
    foldLimitDeg: (gapCm) => foldLimitDeg(gapCm, profile),
  }
}

/**
 * How close to the envelope counts as inside it.
 *
 * 0.05mm — fifteen times finer than the shim, and far below any print or
 * assembly tolerance. A tighter test would report a joint 0.001mm short as a
 * failure, which is measuring floating point rather than buildability; the
 * first run of this module did exactly that and called 25 resolved joints
 * unresolved.
 */
export const RESOLVED_TOLERANCE_CM = 0.005

/**
 * How far INSIDE the envelope the correction aims.
 *
 * A spring relaxation settles where correction and restoration balance, which
 * is always a little short of what it was aiming at — measured 0.14mm short on
 * `modular` when aiming exactly at the envelope. Aiming 0.5mm inside puts the
 * balance point ON it. This biases the TARGET, never the acceptance test: the
 * envelope a joint is judged against is still the connector's own.
 */
export const RELAX_MARGIN_CM = 0.05

/** The measured state of one joint: its gap at the midpoint and its fold. */
function measureJoint(edge, A, B) {
  const s = (edge.edge.from + edge.edge.to) / 2
  const pa = jointEdgePoint(A, edge, true, s)
  const pb = jointEdgePoint(B, edge, false, s)
  const nA = new THREE.Vector3(...A.normal)
  const nB = new THREE.Vector3(...B.normal)

  let rHat = jointEdgeRun(A, edge).clone()
  const pHat = pb.clone().sub(pa)
  pHat.addScaledVector(rHat, -pHat.dot(rHat))
  const gapCm = pHat.length()
  if (gapCm > 1e-9) pHat.normalize()
  let qHat = rHat.clone().cross(pHat)
  if (qHat.dot(nA.clone().add(nB)) < 0) {
    rHat = rHat.negate()
    qHat = rHat.clone().cross(pHat)
  }
  const foldDeg = -Math.atan2(nA.clone().cross(nB).dot(rHat), nA.dot(nB)) * DEG
  return { gapCm, foldDeg, pHat, rHat, mid: pa.clone().add(pb).multiplyScalar(0.5) }
}

/** Rotate a tile's frame in place, about `axis` through the tile's centre. */
function rotateTile(tile, axis, angleRad) {
  if (Math.abs(angleRad) < 1e-12) return
  const q = new THREE.Quaternion().setFromAxisAngle(axis, angleRad)
  const eu = new THREE.Vector3(...tile.eu).applyQuaternion(q)
  const ev = new THREE.Vector3(...tile.ev).applyQuaternion(q)
  const n = new THREE.Vector3(...tile.normal).applyQuaternion(q)
  tile.eu = [eu.x, eu.y, eu.z]
  tile.ev = [ev.x, ev.y, ev.z]
  tile.normal = [n.x, n.y, n.z]
}

/**
 * Relax a solved layout toward the connector envelope.
 *
 * @param {object} layout a `solveLayout` result — NOT mutated
 * @param {object} cfg normalized config
 * @returns {object} a new layout, plus a `relax` report
 */
export function relaxLayout(layout, cfg) {
  const opts = cfg.placement.relax
  const env = jointEnvelope(cfg.connectors)
  const iterations = opts.iterations
  const pull = opts.targetWeight

  // Work on clones; `home` is where the surface put each tile, and the thing
  // the restoration pulls back toward. Using the SOLVED pose rather than
  // re-deriving from the target keeps this honest — the reported displacement is
  // exactly "how far from what surface-fit produced".
  const tiles = layout.tiles.map((t) => ({ ...t, eu: [...t.eu ?? []], ev: [...t.ev ?? []], normal: [...t.normal ?? []], position: t.position ? [...t.position] : null }))
  const byId = new Map(tiles.map((t) => [t.id, t]))
  const home = new Map(tiles.filter((t) => t.position).map((t) => [t.id, {
    position: new THREE.Vector3(...t.position),
    eu: new THREE.Vector3(...t.eu),
    ev: new THREE.Vector3(...t.ev),
    normal: new THREE.Vector3(...t.normal),
  }]))

  for (let iter = 0; iter < iterations; iter++) {
    // --- joint pass: push each violating joint back toward the envelope ----
    layout.adjacency.forEach((edge) => {
      const A = byId.get(edge.a)
      const B = byId.get(edge.b)
      if (!A?.position || !B?.position) return
      const m = measureJoint(edge, A, B)

      // GAP. Split the deficit between the two tiles, along the joint normal.
      const deficit = env.minGapCm + RELAX_MARGIN_CM - m.gapCm
      if (deficit > 0 && m.gapCm > 1e-9) {
        const d = (deficit / 2) * opts.stiffness
        A.position = [A.position[0] - m.pHat.x * d, A.position[1] - m.pHat.y * d, A.position[2] - m.pHat.z * d]
        B.position = [B.position[0] + m.pHat.x * d, B.position[1] + m.pHat.y * d, B.position[2] + m.pHat.z * d]
      }

      // FOLD. Rotate both tiles about the joint axis, toward flatter.
      const limit = env.foldLimitDeg(Math.max(m.gapCm, env.minGapCm))
      const excess = Math.abs(m.foldDeg) - (limit - RELAX_MARGIN_CM)
      if (excess > 0) {
        const step = ((excess / 2) * opts.stiffness) / DEG
        const dir = Math.sign(m.foldDeg)
        rotateTile(A, m.rHat, -dir * step)
        rotateTile(B, m.rHat, dir * step)
      }
    })

    // --- restoration pass: pull every tile back toward where it belongs ---
    for (const tile of tiles) {
      if (!tile.position) continue
      const h = home.get(tile.id)
      const p = new THREE.Vector3(...tile.position).lerp(h.position, pull)
      tile.position = [p.x, p.y, p.z]
      for (const key of ['eu', 'ev', 'normal']) {
        const v = new THREE.Vector3(...tile[key]).lerp(h[key], pull).normalize()
        tile[key] = [v.x, v.y, v.z]
      }
    }

    // Re-orthonormalize: repeated lerping drifts the frame off orthogonal, and
    // a non-orthonormal frame silently skews every downstream measurement.
    for (const tile of tiles) {
      if (!tile.position) continue
      const n = new THREE.Vector3(...tile.normal).normalize()
      const eu = new THREE.Vector3(...tile.eu)
      eu.addScaledVector(n, -eu.dot(n)).normalize()
      const ev = eu.clone().cross(n).normalize()
      tile.eu = [eu.x, eu.y, eu.z]
      tile.ev = [ev.x, ev.y, ev.z]
      tile.normal = [n.x, n.y, n.z]
    }
  }

  // --- settle back onto the floor, as solveLayout does ---------------------
  for (const tile of tiles) {
    if (!tile.position) continue
    tile.position = tile.position.map(r)
    tile.eu = tile.eu.map(r)
    tile.ev = tile.ev.map(r)
    tile.normal = tile.normal.map(r)
    finalizeTileGeometry(tile, cfg.groundTolerance)
  }
  const placed = tiles.filter((t) => t.position)
  const lift = placed.length ? Math.min(...placed.map((t) => t.minY)) : 0
  if (Math.abs(lift) > 1e-9) {
    for (const tile of placed) {
      tile.position = [tile.position[0], r(tile.position[1] - lift), tile.position[2]]
      finalizeTileGeometry(tile, cfg.groundTolerance)
    }
  }

  // --- what it cost, and what it could not fix ---------------------------
  const moved = placed.map((t) => {
    const h = home.get(t.id)
    return {
      id: t.id,
      displacementCm: r(new THREE.Vector3(...t.position).distanceTo(h.position)),
      turnedDeg: r(Math.acos(Math.min(1, Math.max(-1, new THREE.Vector3(...t.normal).dot(h.normal)))) * DEG),
    }
  }).sort((a, b) => b.displacementCm - a.displacementCm)

  const unresolved = []
  layout.adjacency.forEach((edge, idx) => {
    const A = byId.get(edge.a)
    const B = byId.get(edge.b)
    if (!A?.position || !B?.position) return
    const m = measureJoint(edge, A, B)
    const limit = env.foldLimitDeg(Math.max(m.gapCm, env.minGapCm))
    const gapShort = env.minGapCm - m.gapCm
    const foldOver = Math.abs(m.foldDeg) - limit
    if (gapShort > RESOLVED_TOLERANCE_CM || foldOver > RESOLVED_TOLERANCE_CM) {
      unresolved.push({
        joint: idx,
        a: edge.a,
        b: edge.b,
        gapCm: r(m.gapCm),
        foldDeg: r(m.foldDeg),
        gapShortCm: r(Math.max(0, gapShort)),
        foldOverDeg: r(Math.max(0, foldOver)),
      })
    }
  })

  return {
    ...layout,
    tiles,
    relax: {
      iterations,
      targetWeight: pull,
      stiffness: opts.stiffness,
      minGapCm: r(env.minGapCm),
      worstDisplacementCm: moved.length ? moved[0].displacementCm : 0,
      meanDisplacementCm: moved.length
        ? r(moved.reduce((n, m2) => n + m2.displacementCm, 0) / moved.length) : 0,
      worstTurnDeg: moved.length ? r(Math.max(...moved.map((m2) => m2.turnedDeg))) : 0,
      moved: moved.slice(0, 12),
      unresolved,
      resolvedCount: layout.adjacency.length - unresolved.length,
      jointCount: layout.adjacency.length,
    },
  }
}
