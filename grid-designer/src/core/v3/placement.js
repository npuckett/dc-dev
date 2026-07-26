/**
 * grid-designer v3 — spanning-tree placement of a tiled 3D surface.
 *
 * HEADLESS ZONE (src/core/): pure functions, importable from plain node.
 *   - explicit `.js` extensions on ALL relative imports
 *   - may import `three` math classes only; never components / store / DOM
 *   - same config in → byte-identical output out
 *
 * =============================================================================
 * THE IDEA
 * =============================================================================
 * A rigid tile attached to an already-placed neighbour across a shared edge, at
 * a fixed gap, has EXACTLY ONE degree of freedom: the dihedral about that edge.
 *
 * So place tiles in the order of a SPANNING TREE of the tile adjacency graph.
 * Every tree edge is then exact — exactly `gap` cm apart, zero skew, no lateral
 * slip — and each tile's single remaining freedom is spent matching the target
 * drift surface as closely as it can.
 *
 * The error consequently has nowhere to hide. It lands entirely on the
 * NON-TREE edges, the ones that close cycles in the graph. That residual is the
 * HOLONOMY of tiling a doubly-curved surface with rigid flat panels: it is not
 * an implementation artifact, it is the geometry telling you what the physical
 * connectors have to absorb. Measuring it is the whole point of the tool.
 *
 * A v2 column chain was exactly this algorithm on a 1-dimensional graph (a path
 * is its own spanning tree, so v2 had no cycles and therefore no visible
 * holonomy — it dumped everything onto the un-modelled cross-column joints).
 * v3 is the same idea one dimension up, with the error made explicit.
 *
 * Because a developable surface (a plane, a cylinder) CAN be tiled without
 * closure error, holonomy near zero on a cylinder and non-zero on a doubly
 * curved drift is the signature that this is real geometry. tests/ asserts
 * exactly that pair.
 *
 * =============================================================================
 * FRAMES — the part that is easy to get wrong
 * =============================================================================
 * Every placed tile carries an orthonormal world frame (êu, êv, n̂), the images
 * of the material u and v directions and the outward (lit) normal:
 *
 *     n̂ = êv × êu          êv = êu × n̂          êu = êv × n̂ ... (no: see below)
 *
 * Concretely, and consistently with the FLAT reference configuration where
 * material (u,v) maps to world (x,z) with the lit face up:
 *
 *     êu = +X,  êv = +Z,  n̂ = +Y      and indeed  êv × êu = Z × X = +Y ✓
 *
 * so the two relations actually used are
 *
 *     n̂  = êv × êu
 *     êv = êu × n̂                       (X × Y = Z ✓)
 *
 * A tile is rigid and planar, spanning `uLen` along êu and `vLen` along êv, so
 * ANY material point (u,v) inside its footprint maps to world exactly:
 *
 *     world(u,v) = center + (u − uCenter)·êu + (v − vCenter)·êv
 *
 * This exactness is what makes the tree edges exact, and it is used throughout.
 *
 * PANEL LOCAL FRAME (matches geometry/panelGeometry.js and v2):
 *   lit face in the local XZ plane at local y = 0, centered; width along local
 *   X, length along local Z, lit face looking along +Y, housing extending back
 *   to local y = −PANEL_PROFILE.overallThickness.
 *
 * Mapping material → local depends on which way a plate's long side runs:
 *   - square, or plate with axis 'v'  →  localX ↔  êu,  localZ ↔ êv
 *   - plate with axis 'u'             →  localX ↔ −êv,  localZ ↔ êu
 * Both must be right-handed (localX × localY = localZ, with localY = n̂). The
 * NEGATION in the second case is load-bearing: (êv, n̂, êu) is a reflection, not
 * a yaw. See `frameQuaternion`.
 *
 * `position` is the centre of the LIT FACE (as in v2), not of the solid. The
 * OBB handed to collide.js is re-centred on the solid — see `tileOBB`.
 *
 * =============================================================================
 * OUTPUT
 * =============================================================================
 *   solveLayout(config) → {
 *     tiles: [{ id, cells, type, axis, uv, sagittaCm,
 *               position: [x,y,z], quaternion: [x,y,z,w],
 *               eu, ev, normal,            // world frame, for the report
 *               width, length,             // footprint along localX / localZ
 *               corners: [[x,y,z] × 8],    // solid: 4 lit + 4 housing
 *               minY, grounded, tiltDeg }],
 *     tree: { rootId, parent: {childId: parentId}, edgeKeys: Set-like array,
 *             strategy },
 *     support: { contacts, hull, comXZ, comInsideHull },
 *     bounds: { min, max, size },
 *     warnings: [{ code, ... }],
 *     violations: [{ code, ... }],
 *   }
 *
 * Layout warnings / violations (reported, never forced — a slider drag
 * necessarily passes through bad states and the store only commits configs that
 * VALIDATE, so geometry problems must not be validation errors):
 *   E_UNSUPPORTED     centre of mass does not project inside the ground-contact
 *                     hull — the assembly tips over. Per-column grounding could
 *                     never express this; a tiled shell needs it.
 *   W_EDGE_FLOATING   a tile on the wall edge (i = 0) or window edge (j = 0) is
 *                     not grounded. The brief pins those two edges to the floor.
 *   W_TOE_FLAT        a grounded boundary tile is within TOE_FLAT_DEG of
 *                     horizontal. The brief says grounded BUT NOT FLAT.
 *   W_BELOW_FLOOR     a tile's solid dips below y = 0 — nothing to stand on.
 */

import * as THREE from 'three'
import { normalizeConfig } from './schema.js'
import { buildTarget } from './target.js'
import { solveTiling } from './tiling.js'
import { PANEL_PROFILE } from '../../config.js'

const DEG = 180 / Math.PI

/** A grounded boundary tile flatter than this trips W_TOE_FLAT (the brief). */
export const TOE_FLAT_DEG = 6

/** Below this the solid counts as dug into the floor. */
const FLOOR_EPSILON = 1e-6

/** Degenerate-projection guard when the target normal is parallel to the hinge. */
const AXIS_EPS = 1e-9

// -----------------------------------------------------------------------------
// Determinism helpers
// -----------------------------------------------------------------------------

/**
 * Round to 1e-9 and normalize −0 → 0.
 *
 * Trig produces values like 3.7000000000000004 and −0 depending on which branch
 * generated them; both break deepStrictEqual / JSON.stringify comparisons
 * without changing the geometry. This rounding IS the determinism guarantee.
 */
function r(v) {
  const out = Math.round(v * 1e9) / 1e9
  return out === 0 ? 0 : out
}

const rv = (v) => [r(v.x), r(v.y), r(v.z)]

// -----------------------------------------------------------------------------
// Tile material helpers
// -----------------------------------------------------------------------------

const uCenterOf = (t) => t.uv.u0 + t.uv.uLen / 2
const vCenterOf = (t) => t.uv.v0 + t.uv.vLen / 2
const centerOn = (t, axis) => (axis === 'u' ? uCenterOf(t) : vCenterOf(t))

/**
 * Footprint of a tile along its own local X (width) and local Z (length).
 * Derived from the MATERIAL extents, so a config with a non-default
 * cell.plateLength stays consistent instead of silently using the constant.
 */
function tileDims(tile) {
  const longAlongU = tile.type === '2x4' && tile.axis === 'u'
  return longAlongU
    ? { width: tile.uv.vLen, length: tile.uv.uLen, swapped: true }
    : { width: tile.uv.uLen, length: tile.uv.vLen, swapped: false }
}

// -----------------------------------------------------------------------------
// Frames
// -----------------------------------------------------------------------------

const WORLD_X = new THREE.Vector3(1, 0, 0)
const WORLD_Z = new THREE.Vector3(0, 0, 1)

/**
 * An orthonormal surface frame at a plan point, matching the target drift.
 *
 * êu is world +X projected into the tangent plane (falling back to +Z if the
 * surface is so steep that +X is parallel to the normal, which the drift's
 * clamps make unreachable but which must not produce NaN), and êv = êu × n̂ so
 * that n̂ = êv × êu holds — see the FRAMES note in the file header.
 */
function frameFromNormal(n) {
  let eu = WORLD_X.clone().addScaledVector(n, -WORLD_X.dot(n))
  if (eu.lengthSq() < AXIS_EPS) {
    eu = WORLD_Z.clone().addScaledVector(n, -WORLD_Z.dot(n))
  }
  return eu.normalize()
}

function surfaceFrame(target, u, v) {
  const n = target.frameAtMaterial(u, v).normal.clone().normalize()
  const eu = frameFromNormal(n)
  const ev = eu.clone().cross(n).normalize()
  return { eu, ev, n }
}

/**
 * Quaternion taking panel-local axes to world, given the tile's material frame.
 * Columns of the rotation matrix are (localX, localY, localZ) in world.
 */
function frameQuaternion(eu, ev, n, swapped) {
  // Handedness matters and is easy to get wrong. The basis must satisfy
  // localX × localY = localZ, with localY = n̂:
  //   unswapped:  êu × n̂ = X × Y = Z = êv          ✓
  //   swapped:    êv × n̂ = Z × Y = −X = −êu        ✗  ← a REFLECTION
  // so the swapped case must negate êu's partner: localX = −êv gives
  //   (−êv) × n̂ = +êu = localZ                     ✓
  // Feeding a reflection to setFromRotationMatrix yields a garbage quaternion,
  // which is invisible in the corner geometry (the panel is symmetric across
  // its width) but corrupts every downstream OBB — it made a perfectly FLAT
  // grid report 21 interpenetrating pairs before this was fixed.
  const localX = swapped ? ev.clone().negate() : eu.clone()
  const localZ = swapped ? eu.clone() : ev.clone()
  const m = new THREE.Matrix4().makeBasis(localX, n.clone(), localZ)
  return new THREE.Quaternion().setFromRotationMatrix(m)
}

/**
 * The 8 corners of a placed tile's SOLID: 4 on the lit face (local y = 0) and
 * those 4 pushed `overallThickness` along −n̂ (the back of the housing).
 * Order: lit face CCW from (−w/2,−l/2), then the housing four in the same order.
 */
export function panelSolidCorners(tile) {
  const eu = new THREE.Vector3(...tile.eu)
  const ev = new THREE.Vector3(...tile.ev)
  const n = new THREE.Vector3(...tile.normal)
  const c = new THREE.Vector3(...tile.position)
  const hu = tile.uv.uLen / 2
  const hv = tile.uv.vLen / 2
  const th = PANEL_PROFILE.overallThickness

  const out = []
  // Corners are enumerated on the MATERIAL axes, which is equivalent to the
  // local axes up to the swap and avoids duplicating the swap logic here.
  const signs = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
  for (const depth of [0, -th]) {
    for (const [su, sv] of signs) {
      const p = c.clone()
        .addScaledVector(eu, su * hu)
        .addScaledVector(ev, sv * hv)
        .addScaledVector(n, depth)
      out.push(rv(p))
    }
  }
  return out
}

/**
 * The tile's solid as an oriented bounding box in the exact shape
 * `src/core/v3/collide.js` consumes. Note the re-centring: `position` is the
 * centre of the LIT FACE, while the solid's centre sits half the housing
 * thickness behind it along −n̂.
 */
export function tileOBB(tile) {
  const n = new THREE.Vector3(...tile.normal)
  const c = new THREE.Vector3(...tile.position).addScaledVector(n, -PANEL_PROFILE.overallThickness / 2)
  const { width, length } = tileDims(tile)
  return {
    center: rv(c),
    halfExtents: [r(width / 2), r(PANEL_PROFILE.overallThickness / 2), r(length / 2)],
    quaternion: tile.quaternion.slice(),
  }
}

// -----------------------------------------------------------------------------
// Spanning tree
// -----------------------------------------------------------------------------

/**
 * Edge weight per tree strategy. Lower is preferred, and the search is a
 * priority-first traversal keyed on (weight, discovery order), so:
 *
 *   bfs-corner : all weights equal → plain FIFO BFS from the grounded corner;
 *                error spreads roughly evenly outward.
 *   comb-v     : edges running ALONG the window edge are taken first, forming a
 *                spine at j = 0, then ribs run back in v. This reproduces the
 *                v2 column-strip model — every rib is an independent chain — so
 *                all the error is dumped onto the cross-sheet joints. Kept
 *                deliberately: it is the honest way to show what the pivot
 *                bought.
 *   comb-u     : the transpose, spine along the wall edge at i = 0.
 */
function edgeWeight(strategy, edge, tileById) {
  if (strategy === 'bfs-corner') return 0
  const A = tileById.get(edge.a)
  const B = tileById.get(edge.b)
  // `edge.axis` is the axis the shared boundary RUNS along; the tiles are
  // separated along the other one.
  const sepAxis = edge.axis === 'u' ? 'v' : 'u'
  if (strategy === 'comb-v') {
    const onSpine = minCell(A, 1) === 0 && minCell(B, 1) === 0
    if (sepAxis === 'u' && onSpine) return 0
    return sepAxis === 'v' ? 1 : 2
  }
  // comb-u
  const onSpine = minCell(A, 0) === 0 && minCell(B, 0) === 0
  if (sepAxis === 'v' && onSpine) return 0
  return sepAxis === 'u' ? 1 : 2
}

const minCell = (tile, k) => Math.min(...tile.cells.map((c) => c[k]))

/**
 * Priority-first spanning tree. Deterministic: candidates are compared on
 * (weight, insertion order, edge index), all of which are fixed by the input.
 */
function buildSpanningTree(tiles, adjacency, strategy, rootId) {
  const tileById = new Map(tiles.map((t) => [t.id, t]))
  const incident = new Map(tiles.map((t) => [t.id, []]))
  adjacency.forEach((e, idx) => {
    incident.get(e.a).push(idx)
    incident.get(e.b).push(idx)
  })

  const placed = new Set([rootId])
  const parent = new Map()
  const parentEdge = new Map()
  const order = [rootId]
  const treeEdges = new Set()

  // Frontier of candidate edges from placed → unplaced.
  const frontier = []
  let seq = 0
  const pushIncident = (id) => {
    for (const idx of incident.get(id)) {
      const e = adjacency[idx]
      const other = e.a === id ? e.b : e.a
      if (placed.has(other)) continue
      frontier.push({ idx, from: id, to: other, w: edgeWeight(strategy, e, tileById), seq: seq++ })
    }
  }
  pushIncident(rootId)

  while (placed.size < tiles.length) {
    let best = -1
    for (let k = 0; k < frontier.length; k++) {
      const f = frontier[k]
      if (placed.has(f.to)) continue
      if (best === -1) { best = k; continue }
      const b = frontier[best]
      if (f.w < b.w || (f.w === b.w && f.seq < b.seq)) best = k
    }
    if (best === -1) break // disconnected graph; caller reports it
    const chosen = frontier[best]
    frontier.splice(best, 1)
    placed.add(chosen.to)
    parent.set(chosen.to, chosen.from)
    parentEdge.set(chosen.to, chosen.idx)
    treeEdges.add(chosen.idx)
    order.push(chosen.to)
    pushIncident(chosen.to)
  }

  return { order, parent, parentEdge, treeEdges }
}

// -----------------------------------------------------------------------------
// The dihedral solve
// -----------------------------------------------------------------------------

/**
 * Signed angle from `a` to `b` about `axis` (all unit, a and b ⊥ axis).
 */
function signedAngle(a, b, axis) {
  const cross = new THREE.Vector3().crossVectors(a, b)
  return Math.atan2(cross.dot(axis), a.dot(b))
}

/**
 * Place `child` off `parent` across their shared material edge.
 *
 * The child's mating edge is the parent's edge translated by `gap` along the
 * parent's outward in-plane normal; the child is then rotated about that edge
 * by the dihedral θ. Exactly one degree of freedom, so the gap and the skew on
 * this edge are EXACT by construction — no optimisation, no drift.
 *
 * θ is chosen closed-form to align the child's face normal with the target
 * surface normal sampled at the child's MATERIAL centre — see the long comment
 * at the sampling site for why material and not plan coordinates, which is the
 * single most important decision in this file.
 */
function placeChild(target, parentTile, childTile, edge, gap) {
  const runAxis = edge.axis
  const sepAxis = runAxis === 'u' ? 'v' : 'u'

  const parentIsA = edge.a === parentTile.id
  const parentBoundary = parentIsA ? edge.edge.a : edge.edge.b
  const childBoundary = parentIsA ? edge.edge.b : edge.edge.a

  const pEu = new THREE.Vector3(...parentTile.eu)
  const pEv = new THREE.Vector3(...parentTile.ev)
  const pN = new THREE.Vector3(...parentTile.normal)
  const pC = new THREE.Vector3(...parentTile.position)

  const pSep = sepAxis === 'u' ? pEu : pEv
  const pRun = runAxis === 'u' ? pEu : pEv
  const pSepCenter = centerOn(parentTile, sepAxis)
  const pRunCenter = centerOn(parentTile, runAxis)

  // Outward direction from parent toward child, in the parent's plane.
  const sign = childBoundary > parentBoundary ? 1 : -1
  const outward = pSep.clone().multiplyScalar(sign)

  // The hinge axis is the shared edge's direction — parallel to êrun, and
  // unchanged by a rotation about itself.
  const axis = pRun.clone().normalize()

  // Anchor: the world point where the child's material point
  // (sep = childBoundary, run = childRunCenter) must sit. It lies on the line
  // containing the mating edge, offset from the parent's edge by exactly `gap`.
  const cRunCenter = centerOn(childTile, runAxis)
  const cSepCenter = centerOn(childTile, sepAxis)
  const anchor = pC.clone()
    .addScaledVector(pSep, parentBoundary - pSepCenter)
    .addScaledVector(pRun, cRunCenter - pRunCenter)
    .addScaledVector(outward, gap)

  const sepOffset = cSepCenter - childBoundary

  // The target normal is sampled in MATERIAL coordinates — at the child's own
  // (u, v) centre — NOT at its world plan position.
  //
  // This is the crux of the whole solver, and sampling in plan coordinates is
  // the natural-looking mistake. A tilted tile spans 60·cos θ of PLAN for every
  // 60 cm of MATERIAL, so a sheet whose material origin is pinned at the wall
  // falls progressively short of the plan position its material index implies,
  // rides up the slope, and lifts clean off the floor — the grounded wall and
  // window edges the brief demands never touch down. Sampling in material
  // coordinates removes the ambiguity: the walk advances in material space, so
  // the target field must be defined there too.
  //
  // It also makes the solve a direct discrete integration of a PRESCRIBED
  // rotation field, which is why no fixed-point iteration is needed here: the
  // child's material centre is known before it is placed, so the dihedral is a
  // single closed-form solve rather than a chicken-and-egg between orientation
  // and position.
  const aim = target.frameAtMaterial(centerOn(childTile, 'u'), centerOn(childTile, 'v'))
    .normal.clone().normalize()
  const perp = aim.clone().addScaledVector(axis, -aim.dot(axis))
  let theta = 0
  if (perp.lengthSq() >= AXIS_EPS) {
    perp.normalize()
    theta = signedAngle(pN, perp, axis)
  }

  const q = new THREE.Quaternion().setFromAxisAngle(axis, theta)
  const eu = pEu.clone().applyQuaternion(q)
  const ev = pEv.clone().applyQuaternion(q)
  const n = pN.clone().applyQuaternion(q)
  const sepDir = sepAxis === 'u' ? eu : ev
  const center = anchor.clone().addScaledVector(sepDir, sepOffset)
  return { eu, ev, n, center, theta }
}

// -----------------------------------------------------------------------------
// Support hull
// -----------------------------------------------------------------------------

/** Andrew's monotone chain, CCW, on [x, z] points. */
function convexHull(points) {
  if (points.length < 3) return points.slice()
  const pts = points.slice().sort((p, q) => (p[0] - q[0]) || (p[1] - q[1]))
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
  const half = (src) => {
    const h = []
    for (const p of src) {
      while (h.length >= 2 && cross(h[h.length - 2], h[h.length - 1], p) <= 0) h.pop()
      h.push(p)
    }
    h.pop()
    return h
  }
  return half(pts).concat(half(pts.slice().reverse()))
}

function pointInHull(pt, hull) {
  if (hull.length < 3) return false
  for (let i = 0; i < hull.length; i++) {
    const a = hull[i]
    const b = hull[(i + 1) % hull.length]
    const c = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
    if (c < -1e-9) return false
  }
  return true
}

// -----------------------------------------------------------------------------
// solveLayout
// -----------------------------------------------------------------------------

/**
 * @param {object} config raw or normalized v3 config
 * @returns {object} see the OUTPUT block in the file header
 */
export function solveLayout(config) {
  const cfg = normalizeConfig(config)
  const target = buildTarget(cfg)
  const tiling = solveTiling(cfg)
  const gap = cfg.gap
  const groundTol = cfg.groundTolerance
  const warnings = []
  const violations = []

  const tiles = tiling.tiles.map((t) => ({ ...t }))
  const byId = new Map(tiles.map((t) => [t.id, t]))

  const mode = cfg.placement.mode ?? 'surface-fit'
  const root = tiles.find((t) => t.cells.some(([i, j]) => i === 0 && j === 0)) ?? tiles[0]

  let tree
  if (mode === 'surface-fit') {
    // Every tile is placed independently against the target surface, so the
    // joints absorb the incompatibility instead of one edge absorbing all of
    // it. See MODES in the file header.
    for (const tile of tiles) {
      // The TARGET, not the raw smooth form: at angularity > 0 it is quantized
      // into planar facets on the panel lattice, so every tile sharing a facet
      // gets the SAME plane and is therefore exactly coplanar with its
      // neighbours — the joints between them stay at `gap` instead of wedging
      // open, and all the folding collects onto facet boundaries, which is
      // where the physical joints already are. See target.js.
      const { point, normal } = target.frameAtMaterial(uCenterOf(tile), vCenterOf(tile))
      const [x, hy, z] = point
      const n = normal.clone().normalize()
      const eu = frameFromNormal(n)
      const ev = eu.clone().cross(n).normalize()
      tile.eu = rv(eu)
      tile.ev = rv(ev)
      tile.normal = rv(n)
      // The drift surface describes where the assembly's UNDERSIDE sits, so the
      // lit face is offset a full housing thickness along +n̂. Seat the housing
      // on the surface rather than the lit face, and where H = 0 the housing
      // rests on the floor instead of being buried 3.7cm under it. This is v2's
      // `frontRestY` (y0 = 3.7·cos θ — "the panel rests on its housing edge")
      // generalized from a 1D chain to an arbitrarily oriented tile.
      const p = new THREE.Vector3(x, hy, z)
        .addScaledVector(n, PANEL_PROFILE.overallThickness)
      tile.position = rv(p)
    }
    // Settle the whole assembly as one rigid body so the lowest solid corner
    // rests on the floor. A global settle, not a per-tile one: the shell rests
    // on whatever touches first, and moving tiles individually would fake a
    // fit the panels cannot actually achieve.
    let lowest = Infinity
    for (const tile of tiles) {
      for (const c of panelSolidCorners(tile)) lowest = Math.min(lowest, c[1])
    }
    if (Number.isFinite(lowest) && lowest !== 0) {
      for (const tile of tiles) {
        tile.position = [tile.position[0], r(tile.position[1] - lowest), tile.position[2]]
      }
    }
    tree = { order: tiles.map((t) => t.id), parent: new Map(), parentEdge: new Map(), treeEdges: new Set() }
  } else {
  // --- root: the tile owning material cell (0,0), the wall+window corner ----
  {
    const { eu, ev, n } = surfaceFrame(target, uCenterOf(root), vCenterOf(root))
    root.eu = rv(eu)
    root.ev = rv(ev)
    root.normal = rv(n)
    // Anchor by the CORNER, not the centre: the sheet's material origin (0,0)
    // is pinned to the world origin, which is the wall line (x = 0) meeting the
    // window line (z = 0). That is the corner the brief pins to the ground, and
    // anchoring the centre instead would let the tilted tile float inboard of
    // both edges.
    const corner = new THREE.Vector3(0, 0, 0)
    const center = corner.clone()
      .addScaledVector(eu, root.uv.uLen / 2)
      .addScaledVector(ev, root.uv.vLen / 2)
    root.position = rv(center)
    // Then settle it so the lowest solid corner rests exactly on the floor:
    // pitched and rolled, but touching — "grounded but not flat".
    const lift = Math.min(...panelSolidCorners(root).map((c) => c[1]))
    root.position = [root.position[0], r(root.position[1] - lift), root.position[2]]
  }

  tree = buildSpanningTree(tiles, tiling.adjacency, cfg.placement.tree, root.id)

  for (const id of tree.order) {
    if (id === root.id) continue
    const tile = byId.get(id)
    const parentTile = byId.get(tree.parent.get(id))
    const edge = tiling.adjacency[tree.parentEdge.get(id)]
    const p = placeChild(target, parentTile, tile, edge, gap)
    tile.eu = rv(p.eu)
    tile.ev = rv(p.ev)
    tile.normal = rv(p.n)
    tile.position = rv(p.center)
  }
  }

  if (tree.order.length < tiles.length) {
    warnings.push({
      code: 'W_DISCONNECTED',
      message: `only ${tree.order.length} of ${tiles.length} tiles are reachable from the root; the tiling graph is disconnected`,
    })
  }

  // --- per-tile derived geometry ------------------------------------------
  const contacts = []
  for (const tile of tiles) {
    if (!tile.position) {
      // Unreachable tile (disconnected graph). Leave it out of the geometry
      // rather than emitting a fake placement.
      tile.corners = []
      tile.minY = null
      tile.grounded = false
      tile.tiltDeg = null
      continue
    }
    const dims = tileDims(tile)
    tile.width = r(dims.width)
    tile.length = r(dims.length)
    tile.quaternion = (() => {
      const q = frameQuaternion(
        new THREE.Vector3(...tile.eu),
        new THREE.Vector3(...tile.ev),
        new THREE.Vector3(...tile.normal),
        dims.swapped,
      )
      return [r(q.x), r(q.y), r(q.z), r(q.w)]
    })()
    tile.corners = panelSolidCorners(tile)
    tile.minY = r(Math.min(...tile.corners.map((c) => c[1])))
    tile.grounded = tile.minY <= groundTol
    tile.tiltDeg = r(Math.acos(Math.min(1, Math.abs(tile.normal[1]))) * DEG)

    if (tile.grounded) {
      for (const c of tile.corners) {
        if (c[1] <= groundTol) contacts.push([c[0], c[2]])
      }
    }
    if (tile.minY < -FLOOR_EPSILON) {
      warnings.push({
        code: 'W_BELOW_FLOOR',
        tile: tile.id,
        depthCm: r(-tile.minY),
        message: `tile ${tile.id} dips ${(-tile.minY).toFixed(1)}cm below the floor`,
      })
    }

    const onWallEdge = minCell(tile, 0) === 0
    const onWindowEdge = minCell(tile, 1) === 0
    if ((onWallEdge || onWindowEdge) && !tile.grounded) {
      warnings.push({
        code: 'W_EDGE_FLOATING',
        tile: tile.id,
        clearanceCm: tile.minY,
        edge: onWallEdge ? 'wall' : 'window',
        message: `${onWallEdge ? 'wall' : 'window'}-edge tile ${tile.id} floats ${tile.minY.toFixed(1)}cm above the floor; the brief pins that edge down`,
      })
    }
    if ((onWallEdge || onWindowEdge) && tile.grounded && tile.tiltDeg < TOE_FLAT_DEG) {
      warnings.push({
        code: 'W_TOE_FLAT',
        tile: tile.id,
        tiltDeg: tile.tiltDeg,
        message: `grounded edge tile ${tile.id} is only ${tile.tiltDeg.toFixed(1)}° off horizontal; the brief asks for grounded but NOT flat`,
      })
    }
  }

  // --- support: does it stand up? -----------------------------------------
  const hull = convexHull(contacts)
  let comX = 0
  let comZ = 0
  let mass = 0
  for (const tile of tiles) {
    if (!tile.position) continue
    const a = tile.width * tile.length
    comX += tile.position[0] * a
    comZ += tile.position[2] * a
    mass += a
  }
  const comXZ = mass > 0 ? [r(comX / mass), r(comZ / mass)] : [0, 0]
  const comInsideHull = pointInHull(comXZ, hull)
  if (hull.length < 3) {
    // Fewer than three contact points is not "tips over", it is "barely
    // touches" — a different problem with a different fix, so say so rather
    // than reporting a stability verdict the geometry can't support.
    warnings.push({
      code: 'W_NO_SUPPORT',
      contacts: hull.length,
      message: `only ${hull.length} ground-contact point(s); there is no support polygon to test stability against`,
    })
  } else if (!comInsideHull) {
    violations.push({
      code: 'E_UNSUPPORTED',
      comXZ,
      message: `the assembly's centre of mass at (${comXZ[0].toFixed(0)}, ${comXZ[1].toFixed(0)}) does not project inside the ground-contact hull — it tips over`,
    })
  }

  // --- clearance along the two edges the brief pins to the floor -----------
  // Reported as a profile, not a yes/no. A rigid 60cm panel deviates from a
  // doubly-curved target by roughly (30²/2)·curvature wherever you put it, so
  // "the wall edge touches the ground" is achieved to within a few cm and no
  // closer. The number IS the deliverable: it is what shims and connectors have
  // to take up, and it is how you tell whether a form is too aggressive.
  const edgeStats = (pred, label) => {
    const es = tiles.filter((t) => t.position && pred(t))
    if (!es.length) return null
    const cl = es.map((t) => t.minY)
    return {
      edge: label,
      tiles: es.length,
      grounded: es.filter((t) => t.grounded).length,
      maxClearanceCm: r(Math.max(...cl)),
      meanClearanceCm: r(cl.reduce((a, b) => a + b, 0) / cl.length),
      flatTiles: es.filter((t) => t.grounded && t.tiltDeg < TOE_FLAT_DEG).length,
    }
  }
  const edges = [
    edgeStats((t) => minCell(t, 0) === 0, 'wall'),
    edgeStats((t) => minCell(t, 1) === 0, 'window'),
  ].filter(Boolean)

  return {
    tiles,
    mode,
    tree: {
      rootId: root.id,
      strategy: mode === 'surface-fit' ? null : cfg.placement.tree,
      parent: Object.fromEntries(tree.parent),
      treeEdges: Array.from(tree.treeEdges).sort((a, b) => a - b),
    },
    adjacency: tiling.adjacency,
    pattern: tiling.pattern,
    support: { contacts: contacts.map(([x, z]) => [r(x), r(z)]), hull, comXZ, comInsideHull, edges },
    bounds: layoutBounds({ tiles }),
    warnings: warnings.concat(tiling.warnings ?? []),
    violations,
  }
}

/**
 * Axis-aligned extent of every panel SOLID — the physical envelope including
 * housings, which is what the measuring box reports.
 */
export function layoutBounds(layout) {
  const min = [Infinity, Infinity, Infinity]
  const max = [-Infinity, -Infinity, -Infinity]
  for (const tile of layout.tiles) {
    for (const c of tile.corners ?? []) {
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
