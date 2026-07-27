/**
 * grid-designer v3 — where the 3D-printed connectors go, and what each one has
 * to be shaped like.
 *
 * HEADLESS ZONE (src/core/): pure functions, importable from plain node.
 *   - explicit `.js` extensions on ALL relative imports
 *   - may import `three` math classes only; never components / store / DOM
 *   - same layout in → byte-identical output out
 *
 * =============================================================================
 * WHAT THIS IS FOR
 * =============================================================================
 * report.js measures what the connectors must absorb. This module is the next
 * step: it decides WHERE the parts sit and hands each one the local geometry it
 * has to be built to. It produces no solid — that is connectorGeometry.js (P10)
 * — only stations and their frames.
 *
 * There is NO SUBSTRUCTURE. These parts are the structure, which is why station
 * COUNT is a first-class knob rather than a detail: one part on a joint is a
 * hinge, two make it rigid.
 *
 * =============================================================================
 * WHY SHORT PARTS, CENTRED IN THE GAP
 * =============================================================================
 * v1 (3dprintFiles/) used ONE part everywhere: a Y-junction edge clamp holding a
 * fixed 62° dihedral, sitting on the assembly's OUTSIDE edges. It could be one
 * part because every v1 joint was the same angle.
 *
 * v3's joints are all different, and worse, a single joint is different ALONG
 * ITS OWN LENGTH. Measured across the six presets:
 *
 *     station dihedral        0 – 34.8°
 *     station skew            0 – 11.5°
 *     rim-to-rim span      0.60 – 15.82 cm
 *     span swing along ONE whole joint          up to 12.77 cm
 *     span swing inside ONE 10 cm window   0.02 – 0.15 cm mean, 2.12 cm worst
 *
 * That last pair is the whole argument. A joint that wedges from 3 cm to 16 cm
 * end to end cannot be held by any one rigid part, but every 10 cm slice of it
 * is very nearly a constant-span, constant-angle problem. Short local parts
 * convert one intractable joint into a handful of easy ones — the same move as
 * HANDOFF §2.1 (stop asking a rigid thing to be a curved thing), applied to the
 * hardware instead of the surface.
 *
 * The consequence of "centred" rather than "on the outside edges" is physical
 * and belongs in the part design: a connector at mid-edge cannot be SLID onto
 * the rim from an open end, so it has to snap over it.
 *
 * =============================================================================
 * WHAT A STATION CARRIES
 * =============================================================================
 * Everything the part generator needs, and nothing derived from a scene:
 *
 *   spanCm       rim-to-rim distance at the station centre, on the LIT FACE.
 *                Front-mounted parts clip that rim, so this is the dimension the
 *                spine has to be. NOT the same as the nominal `gap`: it is what
 *                the joint actually opened to.
 *   dihedralDeg  fold between the two panels' lit normals — the arms' angle.
 *   twistDeg     angle between the two rim LINES. Constant along a joint (both
 *                tiles are rigid and planar, so each rim direction is fixed), so
 *                it equals report.js's per-joint `skewDeg` — recorded per
 *                station anyway because the part is per station.
 *   aFrame/bFrame  per side: the rim point, the unit vector the rim runs along,
 *                the panel's lit normal, and `inward` — which way the panel
 *                material lies from its own rim, i.e. which way the channel has
 *                to reach.
 *
 * `spanMinCm`/`spanMaxCm` bracket the span over the part's own footprint rather
 * than over the whole joint. A rigid part has to swallow that difference, and it
 * is the number that says whether `lengthCm` is short enough.
 */

import * as THREE from 'three'
import { normalizeConfig } from './schema.js'
import { solveLayout, jointEdgePoint, jointEdgeInward, jointEdgeRun } from './placement.js'
import { PANEL_PROFILE, POWER_SUPPLY, poweredEdgeBlockedSpan } from '../../config.js'

const DEG = 180 / Math.PI

/** Samples across one part's footprint when bracketing its local span. */
export const SPAN_SAMPLES = 5

// =============================================================================
// THE PART — A TWO-PIECE BOLTED RIM CLAMP
// =============================================================================
/**
 * Two printed pieces per station, pulled together by three small bolts.
 *
 *   FRONT BAR   a plain rectangular bar lying across the gap, bearing on both
 *               panels' bezels, with three countersunk holes. UNIVERSAL: its
 *               only variable is its width, so one bar serves a whole BAND of
 *               gaps (see `frontBarBand`).
 *   BACK HALF   fills the gap behind the bar and reaches under both flanges,
 *               with three heat-set inserts. Per-station geometry — it has to
 *               sit on the flange at the joint's own fold.
 *
 * The bolts do the clamping, so the two pieces never have to snap or rotate on:
 * the back half goes on from behind, the bar drops on from the front, three
 * bolts pull the rims between them. That was the point of splitting it.
 *
 * WHY THE BAR CAN BE UNIVERSAL. It bears on two bezel surfaces and spans
 * whatever is between them. A bar of width W leaves a lip of `(W − gap)/2` on
 * each side, so it serves every gap for which that lip is between
 * `frontMinLipCm` (enough to bear on) and the bezel's own width (beyond which
 * it would overhang the diffuser). That is a band `2·(bezelWidth − minLip)`
 * wide — 2.2cm at the measured panel. Measured over the presets: `closed` and
 * `modular` need ONE bar, `shelf` and `drift` two.
 *
 * ITS UNDERSIDE IS FLAT, so it bears on a LINE rather than a face — at the
 * panel's outer corner on a convex joint, nearer the bezel peak on a concave
 * one. That is the price of universality and it is real; a per-station bar
 * would bear on the full lip. Three bolts rather than one exist partly to
 * spread the load that line has to carry.
 *
 * THE BAR STANDS PROUD of the panel's front plane by `crownCm`. It cannot sit
 * flush: the bezel peak IS the front plane, so a lip over it has nowhere to be
 * except above it.
 *
 * NOTHING HERE RESTATES A PANEL DIMENSION — every surface is traced off
 * PANEL_PROFILE, so refining the panel moves the clamp with it.
 */
export const BOLT_M3 = {
  name: 'M3',
  shankCm: 0.30,
  headCm: 0.60,
  headDepthCm: 0.24,
  insertOdCm: 0.40,
  insertLenCm: 0.57,
  lengthCm: 1.2,
}

export const CONNECTOR_PROFILE = {
  /**
   * SHIM CLEARANCE — how far every clamp face stands off the panel it bears on.
   *
   * The parts never touch the panels: a rubber shim goes in the gap on the real
   * build, so the printed geometry must leave room for one. It applies to all
   * three bearing faces — the bar over the bezel, the back half's lip under the
   * flange, and its body against the outer walls — because all three get a shim.
   *
   * 0.075cm sits in the middle of the 0.5–1mm band. It is also what stops the
   * model claiming a face-to-face fit the build will never have.
   */
  shimCm: 0.075,
  /** Least bearing the front bar may keep on a bezel before it is not holding. */
  frontMinLipCm: 0.4,
  /**
   * The lip a per-station bar is sized to give. The UNIVERSAL bar is retired for
   * now (it could not be made to work across the extremes), so every station
   * gets a bar cut to its own gap: width = gap + 2·frontLipCm. `solveFrontBars`
   * still exists and still bins them, so the kit reports how few distinct WIDTHS
   * the design needs — the sharing is now a reporting fact rather than a
   * constraint the geometry has to satisfy.
   */
  frontLipCm: 1.0,
  /** The bar's own thickness. */
  frontThicknessCm: 0.4,
  /** Where the two pieces part. Must lie inside the outer wall — see below. */
  splitDepthCm: 0.6,
  /** The back half's lip on the flange: the load-bearing grip. */
  backGripCm: 2.4,
  /** Material below the flange lip. */
  backFloorCm: 0.45,
  /** Material around a fastener. */
  wallCm: 0.2,
  boltCount: 3,
  bolt: BOLT_M3,
}

/** Depth of the bezel surface `i` cm inboard (bezelDrop at the edge → 0 at the peak). */
export function bezelDepthAt(i, panel = PANEL_PROFILE) {
  const p = { ...PANEL_PROFILE, ...panel }
  return p.bezelDrop * (1 - Math.min(i, p.bezelWidth) / p.bezelWidth)
}

/** Depth of the flange surface `i` cm inboard of the edge. */
export function flangeDepthAt(i, panel = PANEL_PROFILE) {
  const p = { ...PANEL_PROFILE, ...panel }
  return p.outerWallDepth + p.flangeDrop * (Math.min(i, p.flangeWidth) / p.flangeWidth)
}

/**
 * THE SPLIT PLANE HAS ONLY THE OUTER WALL TO LIVE IN. It must sit below the
 * bezel the front bar grips and above the flange the back half grips, so its
 * whole latitude is the 1.1cm outer wall. Reported rather than clamped, on the
 * usual contract.
 */
export function splitPlaneRange(panel = PANEL_PROFILE) {
  const p = { ...PANEL_PROFILE, ...panel }
  return [p.bezelDrop, p.outerWallDepth]
}

/** The width of the bar for a station of this gap: its own lip, both sides. */
export function frontBarWidthFor(gapCm, profile = CONNECTOR_PROFILE) {
  const c = { ...CONNECTOR_PROFILE, ...profile }
  return gapCm + 2 * c.frontLipCm
}

/** The band of gaps one front bar of width `barWidthCm` can serve. */
export function frontBarBand(barWidthCm, profile = CONNECTOR_PROFILE, panel = PANEL_PROFILE) {
  const c = { ...CONNECTOR_PROFILE, ...profile }
  const p = { ...PANEL_PROFILE, ...panel }
  return [barWidthCm - 2 * p.bezelWidth, barWidthCm - 2 * c.frontMinLipCm]
}

/**
 * Cover a set of gaps with as few front bars as possible.
 *
 * Greedy from the narrowest gap up: each bar is sized so the narrowest gap it
 * serves gets the FULL bezel lip, and it then covers everything up to the top
 * of its band. Greedy is optimal for covering points on a line with fixed-width
 * intervals, so this is the true minimum, not a heuristic.
 */
export function solveFrontBars(gapsCm, profile = CONNECTOR_PROFILE, panel = PANEL_PROFILE) {
  const c = { ...CONNECTOR_PROFILE, ...profile }
  const p = { ...PANEL_PROFILE, ...panel }
  const sorted = [...gapsCm].sort((x, y) => x - y)
  const bars = []
  let i = 0
  while (i < sorted.length) {
    const widthCm = r(sorted[i] + 2 * p.bezelWidth)
    const [, hi] = frontBarBand(widthCm, c, p)
    const serves = []
    while (i < sorted.length && sorted[i] <= hi + 1e-9) serves.push(sorted[i++])
    bars.push({
      widthCm,
      gapMinCm: r(serves[0]),
      gapMaxCm: r(serves[serves.length - 1]),
      bandCm: [r(widthCm - 2 * p.bezelWidth), r(hi)],
      count: serves.length,
    })
  }
  return bars
}

/**
 * The FRONT BAR's cross-section: a rectangle, centred on the gap.
 *
 * Genuinely just a rectangle — that is what makes it universal. Its underside
 * is flat at `frontFlatDepthCm`, its top at `-crownCm`, and it spans its own
 * width regardless of what the joint underneath is doing.
 */
export function frontBarProfile(barWidthCm, profile = CONNECTOR_PROFILE) {
  const c = { ...CONNECTOR_PROFILE, ...profile }
  const h = barWidthCm / 2
  // Its underside floats a shim above the panel's front-most plane — the bezel
  // peak — so it clears the bezel by at least the shim everywhere along its
  // reach, and by more as the bezel falls away toward the panel edge.
  const bot = c.shimCm
  const top = c.shimCm + c.frontThicknessCm
  const points = [[-h, bot], [h, bot], [h, top], [-h, top]]
  return { points, extents: extentsOf(points) }
}

/**
 * The BACK HALF's cross-section: fills the gap behind the split plane and
 * reaches under both flanges. Per-station, because the flanges tilt with the
 * joint's fold.
 */
export function backHalfProfile({ spanCm, foldDeg, profile = CONNECTOR_PROFILE, panel = PANEL_PROFILE }) {
  const c = { ...CONNECTOR_PROFILE, ...profile }
  const pp = { ...PANEL_PROFILE, ...panel }
  const phi = (foldDeg * Math.PI) / 180 / 2
  const cs = Math.cos(phi)
  const sn = Math.sin(phi)
  const bg = Math.min(c.backGripCm, pp.flangeWidth)

  const sides = [
    { rim: [-spanCm / 2, 0], inward: [-cs, -sn], deeper: [sn, -cs] },
    { rim: [spanCm / 2, 0], inward: [cs, -sn], deeper: [-sn, -cs] },
  ]
  const at = (side, i, d) => [
    side.rim[0] + i * side.inward[0] + d * side.deeper[0],
    side.rim[1] + i * side.inward[1] + d * side.deeper[1],
  ]

  // Every panel-facing point is pushed off by the shim: the lips sit a shim
  // BELOW the flange, and the body stands a shim OFF each outer wall (negative
  // inboard offset is out into the gap). Nothing touches the panel.
  const sh = c.shimCm
  const aLip = at(sides[0], bg, flangeDepthAt(bg, pp) + sh)
  const bLip = at(sides[1], bg, flangeDepthAt(bg, pp) + sh)
  const aWall = at(sides[0], -sh, pp.outerWallDepth + sh)
  const bWall = at(sides[1], -sh, pp.outerWallDepth + sh)
  // The floor must clear EVERY point above it, not just the lips: on a concave
  // joint the walls tilt outward and their back corners drop below the lips, and
  // a floor taken from the lips alone left them hanging through it — a
  // self-intersecting outline, which lofts into a torn shell.
  const floor = Math.min(aLip[1], bLip[1], aWall[1], bWall[1]) - c.backFloorCm
  // Counter-clockwise in (p, q): along the floor, up the far side, back over the
  // top face, down the near side. The loft's winding rule depends on it, and a
  // clockwise outline produces a shell that is inside-out — invisible on screen,
  // fatal at the slicer.
  // The top surface FOLLOWS THE PANEL: up the outer wall from the back corner to
  // the split plane, across the gap, and back down the other wall. Cutting
  // straight from the split plane to the flange lip instead would run the piece
  // clean through the panel's own section.
  const points = [
    [aLip[0], floor],
    [bLip[0], floor],
    bLip,
    bWall,
    at(sides[1], -sh, c.splitDepthCm),
    at(sides[0], -sh, c.splitDepthCm),
    aWall,
    aLip,
  ]
  return { points, extents: extentsOf(points) }
}

function extentsOf(points) {
  let pMin = Infinity
  let pMax = -Infinity
  let qMin = Infinity
  let qMax = -Infinity
  for (const [p, q] of points) {
    if (p < pMin) pMin = p
    if (p > pMax) pMax = p
    if (q < qMin) qMin = q
    if (q > qMax) qMax = q
  }
  return { pMin, pMax, qMin, qMax, width: pMax - pMin, height: qMax - qMin }
}

/**
 * The gap a fastener needs, and where it is narrowest.
 *
 * The binding dimension is the countersink head or the insert's outside
 * diameter, whichever is larger, plus a wall each side. And the check must be
 * made at DEPTH, not at the face: on a convex joint the gap narrows as
 * `gap − 2·d·sin(fold/2)`, and the insert sits at the bottom of the back half.
 * This is the first constraint in the tool that couples gap and fold.
 */
export function fastenerGapNeededCm(profile = CONNECTOR_PROFILE) {
  const c = { ...CONNECTOR_PROFILE, ...profile }
  return Math.max(c.bolt.headCm, c.bolt.insertOdCm) + 2 * c.wallCm
}

/**
 * The most a joint of this gap can fold before the two panels' own back corners
 * meet — the hard limit, and the one relax.js pulls joints back inside.
 *
 * It belongs to the PANEL, not the connector: the corners sit a shim off the
 * wall in both axes, so the separation is
 *
 *     gap − 2·shim·cos(φ) − 2·(outerWallDepth + shim)·sin(φ),   φ = fold/2
 *
 * solved here by bisection. The connector fouls 3–6° before this, so staying
 * inside it is necessary and very nearly sufficient.
 */
export function foldLimitDeg(gapCm, profile = CONNECTOR_PROFILE, panel = PANEL_PROFILE) {
  const c = { ...CONNECTOR_PROFILE, ...profile }
  const p = { ...PANEL_PROFILE, ...panel }
  const sep = (phi) => gapCm - 2 * c.shimCm * Math.cos(phi) - 2 * (p.outerWallDepth + c.shimCm) * Math.sin(phi)
  if (sep(Math.PI / 4) > 0) return 90
  if (sep(0) <= 0) return 0
  let lo = 0
  let hi = Math.PI / 4
  for (let k = 0; k < 60; k++) {
    const mid = (lo + hi) / 2
    if (sep(mid) > 0) lo = mid
    else hi = mid
  }
  return (((lo + hi) / 2) * 2 * 180) / Math.PI
}

export function gapAtDepthCm(spanCm, foldDeg, depthCm) {
  return spanCm - 2 * depthCm * Math.sin(Math.abs((foldDeg * Math.PI) / 180 / 2)) * (foldDeg > 0 ? 1 : -1)
}

/**
 * The two cross-sections a part is lofted between — its `r̂ = -length/2` end and
 * its `+length/2` end. They differ only in span: both tiles are rigid planes, so
 * the fold is constant along the joint and the rims' separation varies linearly.
 * That is also why "twist" needs no separate term in the solid — a twisted joint
 * IS one whose two ends have different spans.
 */
export function connectorEndProfiles(station, profile = CONNECTOR_PROFILE) {
  return {
    start: backHalfProfile({ spanCm: station.spanStartCm, foldDeg: station.foldDeg, profile }),
    end: backHalfProfile({ spanCm: station.spanEndCm, foldDeg: station.foldDeg, profile }),
  }
}

/**
 * A station's part as an oriented bounding box, in the shape collide.js
 * consumes — so the SAT that already checks panel-against-panel can check
 * part-against-panel and part-against-part without a second implementation.
 *
 * The box encloses BOTH end profiles. Its local axes are the station frame
 * (p̂, q̂, r̂), which is right-handed by construction in `solveConnectors`; a
 * left-handed basis here would corrupt every overlap test exactly as it did for
 * plates in HANDOFF §6.
 */
export function connectorOBB(station, profile = CONNECTOR_PROFILE) {
  return pieceOBB(station, connectorEndProfiles(station, profile))
}

/** The front bar's box. A constant rectangle, so both ends are the same. */
export function frontBarOBB(station, barWidthCm, profile = CONNECTOR_PROFILE) {
  const prof = frontBarProfile(barWidthCm, profile)
  return pieceOBB(station, { start: prof, end: prof })
}

function pieceOBB(station, { start, end }) {
  const pMin = Math.min(start.extents.pMin, end.extents.pMin)
  const pMax = Math.max(start.extents.pMax, end.extents.pMax)
  const qMin = Math.min(start.extents.qMin, end.extents.qMin)
  const qMax = Math.max(start.extents.qMax, end.extents.qMax)

  const p = new THREE.Vector3(...station.frame.p)
  const q = new THREE.Vector3(...station.frame.q)
  const rr = new THREE.Vector3(...station.frame.r)
  const centre = new THREE.Vector3(...station.mid)
    .addScaledVector(p, (pMin + pMax) / 2)
    .addScaledVector(q, (qMin + qMax) / 2)

  const m = new THREE.Matrix4().makeBasis(p, q, rr)
  const quat = new THREE.Quaternion().setFromRotationMatrix(m)

  return {
    center: rv(centre),
    halfExtents: [r((pMax - pMin) / 2), r((qMax - qMin) / 2), r(station.lengthCm / 2)],
    quaternion: [r(quat.x), r(quat.y), r(quat.z), r(quat.w)],
  }
}

/**
 * Does the BACK HALF fold so far that it closes on itself? The two flange lips
 * swing toward each other as a convex joint folds, exactly as the old one-piece
 * clamp's did. The front bar cannot self-intersect — it is a rectangle.
 */
/** Signed area of a closed polygon; positive means counter-clockwise. */
export function polygonArea(points) {
  let a = 0
  for (let k = 0; k < points.length; k++) {
    const [x0, y0] = points[k]
    const [x1, y1] = points[(k + 1) % points.length]
    a += x0 * y1 - x1 * y0
  }
  return a / 2
}

const sideOf = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

/**
 * Does a closed polygon cross itself? Non-adjacent edge pairs only, proper
 * crossings only — touching endpoints are legal.
 *
 * This is the part's HARD feasibility gate, and it is a physical statement
 * rather than a modelling nicety: the profile self-intersects when the two
 * hooks, swinging under the joint as it folds, run into each other. Measured
 * boundary (see tests) —
 *
 *     span 0.4cm → ±14° convex   span 1.0cm → ±36°   span ≥2.5cm → anything
 *
 * — which is the same fact the housings already report, seen from the connector
 * side: a convex joint closes BEHIND the panels, and a narrow gap leaves the
 * hardware nowhere to be. Deriving the gate from the geometry rather than
 * tabulating it means editing CONNECTOR_PROFILE moves the boundary honestly.
 */
export function profileSelfIntersects(points) {
  const n = points.length
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (j === i || (j + 1) % n === i || (i + 1) % n === j) continue
      const p1 = points[i]
      const p2 = points[(i + 1) % n]
      const p3 = points[j]
      const p4 = points[(j + 1) % n]
      const d1 = sideOf(p3, p4, p1)
      const d2 = sideOf(p3, p4, p2)
      const d3 = sideOf(p1, p2, p3)
      const d4 = sideOf(p1, p2, p4)
      if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) {
        return true
      }
    }
  }
  return false
}

/**
 * THE PANEL'S OWN SECTION on one side of a joint, in the joint's (p, q) frame —
 * the same frame the connector profiles are built in, so the two can be tested
 * against each other directly.
 *
 * `farCm` is how far inboard to carry it. It only has to reach past anything the
 * connector could touch, which is the flange and the taper behind it.
 */
export function panelSectionAt(sign, spanCm, foldDeg, panel = PANEL_PROFILE, farCm = 8) {
  const p = { ...PANEL_PROFILE, ...panel }
  const phi = (foldDeg * Math.PI) / 180 / 2
  const cs = Math.cos(phi)
  const sn = Math.sin(phi)
  const rim = [(sign * spanCm) / 2, 0]
  const inw = [sign * cs, -sn]
  const dp = [-sign * sn, -cs]
  const at = (i, d) => [rim[0] + i * inw[0] + d * dp[0], rim[1] + i * inw[1] + d * dp[1]]
  return [
    at(0, p.bezelDrop),
    at(p.bezelWidth, 0),
    at(farCm, 0),
    at(farCm, p.overallThickness),
    at(p.flangeWidth + p.taperWidth, p.overallThickness),
    at(p.flangeWidth, flangeDepthAt(p.flangeWidth, p)),
    at(0, p.outerWallDepth),
  ]
}

const pointInPolygon = (poly, x, y) => {
  let n = false
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i]
    const [xj, yj] = poly[j]
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) n = !n
  }
  return n
}

/**
 * Do two simple polygons overlap? Proper edge crossings, plus containment either
 * way round for the case where one sits entirely inside the other.
 *
 * WHY THIS AND NOT THE OBB TEST. Both the panel and the connector are SWEPT
 * SOLIDS along the joint, so a section test is exact for them where a bounding
 * box is not even close: a connector's box always encloses the panel rim it
 * wraps, which is why the OBB check had to exclude the two panels it grips and
 * could therefore never see a part biting into its own panel. This can.
 */
export function polygonsOverlap(A, B) {
  for (let i = 0; i < A.length; i++) {
    const a1 = A[i]
    const a2 = A[(i + 1) % A.length]
    for (let j = 0; j < B.length; j++) {
      const b1 = B[j]
      const b2 = B[(j + 1) % B.length]
      const d1 = sideOf(b1, b2, a1)
      const d2 = sideOf(b1, b2, a2)
      const d3 = sideOf(a1, a2, b1)
      const d4 = sideOf(a1, a2, b2)
      if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) return true
    }
  }
  if (A.some(([x, y]) => pointInPolygon(B, x, y))) return true
  if (B.some(([x, y]) => pointInPolygon(A, x, y))) return true
  return false
}

/**
 * Does either piece of a station's connector bite into either panel?
 *
 * Checked at BOTH ends of the loft, because the span differs there and a part
 * can clear at one end and foul at the other. Returns which pieces foul.
 *
 * MEASURED HEADROOM: the connector fouls only 3–6° of fold before the two
 * PANELS collide with each other anyway — 15° vs 19° at a 0.4cm gap, 45° vs 49°
 * at 1cm. So the part is already within a few degrees of the hard geometric
 * limit, and no redesign of it buys meaningful range in that corner.
 */
export function sectionFouling(station, profile = CONNECTOR_PROFILE, panel = PANEL_PROFILE) {
  const out = { backHalf: false, frontBar: false, panelsCollide: false }
  for (const span of [station.spanStartCm, station.spanEndCm]) {
    const sides = [
      panelSectionAt(-1, span, station.foldDeg, panel),
      panelSectionAt(1, span, station.foldDeg, panel),
    ]
    if (polygonsOverlap(sides[0], sides[1])) out.panelsCollide = true
    const back = backHalfProfile({ spanCm: span, foldDeg: station.foldDeg, profile, panel }).points
    if (sides.some((sec) => polygonsOverlap(sec, back))) out.backHalf = true
    if (station.barWidthCm) {
      const bar = frontBarProfile(station.barWidthCm, profile).points
      if (sides.some((sec) => polygonsOverlap(sec, bar))) out.frontBar = true
    }
  }
  return out
}

/**
 * Printability limits. Unlike the self-intersection gate above these are
 * JUDGEMENT, not geometry, so they live as data and are stated as such:
 *
 *   minSpanCm       below this there is no room to work the part onto the rims,
 *                   whatever the profile says.
 *   maxSpanCm       the spine is a `jawCm`-thick strap. Past roughly 8cm it is a
 *                   beam being asked to act like a strap and wants ribbing or a
 *                   thicker section — neither of which this first pass models.
 *   maxSpanSpreadCm how much a single part may wedge along its own length before
 *                   the loft stops being a mild taper. This is the number
 *                   `connectors.lengthCm` exists to keep small.
 *
 * Consumed by report.js (P11), which flags rather than refuses.
 */
export const CONNECTOR_LIMITS = {
  minSpanCm: 0.4,
  maxSpanCm: 8,
  maxSpanSpreadCm: 1.0,
}

/**
 * Everything wrong with one station's part that can be decided from the station
 * alone. Clash needs the rest of the assembly, so report.js adds that.
 *
 * Pure and exported rather than inlined into the report because
 * `W_CONNECTOR_INFEASIBLE` is not reachable from any real config — see the note
 * on it below — and a rule with no test is a rule that quietly stops working.
 *
 *   W_CONNECTOR_INFEASIBLE  the two hooks pass through each other. GEOMETRY, and
 *        the only hard one here. Never yet observed on a real design, and the
 *        reason is a real property rather than luck: where the surface folds
 *        hard the gap has already wedged open, and where the gap is tight the
 *        surface is nearly flat. The two ways to make a part impossible do not
 *        co-occur on a drift. Kept, and tested against a synthetic station,
 *        because that correlation is a property of these forms and not a law.
 *   W_CONNECTOR_PINCH   the gap is narrower than anything can be fitted into.
 *   W_CONNECTOR_SPAN    the spine is a beam being asked to act as a strap.
 *   W_CONNECTOR_TWIST   the part wedges too much along its own length; this is
 *        the one `connectors.lengthCm` directly controls.
 */
export function connectorStationFlags(station, limits = CONNECTOR_LIMITS, profile = CONNECTOR_PROFILE) {
  const flags = []
  const { start, end } = connectorEndProfiles(station, profile)
  if (profileSelfIntersects(start.points) || profileSelfIntersects(end.points)) {
    flags.push('W_CONNECTOR_INFEASIBLE')
  }
  if (station.spanMinCm < limits.minSpanCm) flags.push('W_CONNECTOR_PINCH')
  if (station.spanMaxCm > limits.maxSpanCm) flags.push('W_CONNECTOR_SPAN')
  if (station.spanSpreadCm > limits.maxSpanSpreadCm) flags.push('W_CONNECTOR_TWIST')

  // THE FASTENER, checked where the gap is NARROWEST — at the bottom of the
  // back half, not at the face. On a convex joint the gap closes with depth, so
  // a bolt that clears at the rim can still be pinched at the insert.
  const need = fastenerGapNeededCm(profile)
  const c = { ...CONNECTOR_PROFILE, ...profile }
  const deep = flangeDepthAt(c.backGripCm) + c.backFloorCm
  const atFace = station.spanMinCm
  const atDepth = gapAtDepthCm(station.spanMinCm, station.foldDeg, deep)
  if (Math.min(atFace, atDepth) < need) flags.push('W_FASTENER_PINCHED')

  // The panel's power supply sits on the flange this half grips, so the lip
  // bears on the SUPPLY HOUSING rather than the panel frame.
  if (station.bearsOnPowerSupply) flags.push('W_BEARS_ON_POWER_SUPPLY')

  // SECTION-LEVEL fouling — exact for these swept solids, and able to see a part
  // biting into a panel it grips, which the OBB test structurally cannot.
  const foul = sectionFouling(station, profile)
  if (foul.backHalf) flags.push('W_BACK_HALF_FOULS_PANEL')
  if (foul.frontBar) flags.push('W_FRONT_BAR_FOULS_PANEL')
  if (foul.panelsCollide) flags.push('W_PANELS_COLLIDE_AT_JOINT')
  return flags
}

/**
 * A joint too short to seat `count` parts at full length. The parts are still
 * placed — shortened to fit — because refusing to connect a joint is not an
 * option a structure-bearing part gets to take. Same contract as
 * `W_PLATE_OVERRIDE_MISFIT`: report the cost, do not veto.
 */
export const CROWDED_CODE = 'W_CONNECTOR_CROWDED'

function r(v) {
  const out = Math.round(v * 1e9) / 1e9
  return out === 0 ? 0 : out
}

const rv = (v) => [r(v.x), r(v.y), r(v.z)]

// =============================================================================
// THE POWER SUPPLY — an edge a connector cannot use
// =============================================================================
/**
 * Which of a tile's edges carries the power supply, as `{ axis, boundary }` in
 * the same terms an adjacency record uses: `axis` is the material direction the
 * edge RUNS along, `boundary` its coordinate on the other axis.
 *
 * The supply is on a 60cm edge on both panel types, so for a plate it is one of
 * the two short ENDS, never a long side. `policy` picks which end:
 *   'low'  the end at the tile's minimum coordinate (default)
 *   'high' the far end
 *   'none' no supply — for studying what the constraint costs
 *
 * A GLOBAL CONVENTION, not a per-tile choice. Which way each panel faces is a
 * real design freedom that nothing in the tool models yet; until it does, every
 * panel is assumed oriented the same way, and that assumption is visible here
 * rather than buried.
 */
export function poweredEdgeOf(tile, policy = 'low') {
  if (policy === 'none') return null
  // The short edges run ACROSS the tile's long axis. A square has no long axis;
  // by convention its powered edge runs along u, matching a v-axis plate.
  const alongU = !(tile.type === '2x4' && tile.axis === 'u')
  const lo = alongU ? tile.uv.v0 : tile.uv.u0
  const len = alongU ? tile.uv.vLen : tile.uv.uLen
  return { axis: alongU ? 'u' : 'v', boundary: policy === 'low' ? lo : lo + len }
}

const EDGE_EPS = 1e-6

/**
 * The intervals of a joint that a flange-gripping connector CANNOT use, because
 * one of the two panels has its power supply behind that stretch of rim.
 *
 * Returned in the joint's own run-parameter space (the same coordinates as
 * `edge.edge.from`/`to`), merged and clipped to the joint.
 */
export function blockedSpansOnJoint(edge, A, B, policy = 'low', supply = POWER_SUPPLY) {
  const out = []
  for (const [tile, isA] of [[A, true], [B, false]]) {
    const powered = poweredEdgeOf(tile, policy)
    if (!powered || powered.axis !== edge.axis) continue
    const myBoundary = isA ? edge.edge.a : edge.edge.b
    if (Math.abs(powered.boundary - myBoundary) > EDGE_EPS) continue
    // The supply is centred on ITS OWN edge, which is the full run-extent of the
    // tile along `edge.axis` — not the joint, which may be a partial overlap.
    const runLo = edge.axis === 'u' ? tile.uv.u0 : tile.uv.v0
    const runLen = edge.axis === 'u' ? tile.uv.uLen : tile.uv.vLen
    const [from, to] = poweredEdgeBlockedSpan(runLen, supply)
    out.push([runLo + from, runLo + to])
  }
  return mergeIntervals(out, edge.edge.from, edge.edge.to)
}

/** Merge overlapping intervals and clip them to [lo, hi]. */
function mergeIntervals(spans, lo, hi) {
  const clipped = spans
    .map(([a, b]) => [Math.max(a, lo), Math.min(b, hi)])
    .filter(([a, b]) => b - a > EDGE_EPS)
    .sort((x, y) => x[0] - y[0])
  const out = []
  for (const s of clipped) {
    const last = out[out.length - 1]
    if (last && s[0] <= last[1] + EDGE_EPS) last[1] = Math.max(last[1], s[1])
    else out.push([...s])
  }
  return out
}

/** The complement of `blocked` within [lo, hi] — where a part may actually go. */
export function clearSpans(blocked, lo, hi) {
  const out = []
  let cursor = lo
  for (const [a, b] of blocked) {
    if (a - cursor > EDGE_EPS) out.push([cursor, a])
    cursor = Math.max(cursor, b)
  }
  if (hi - cursor > EDGE_EPS) out.push([cursor, hi])
  return out
}

/**
 * A joint whose usable rim is too short for even one part. Only reachable when
 * the supply is treated as a hard obstruction (`powerEdge: 'block'`).
 */
export const BLOCKED_CODE = 'W_JOINT_BLOCKED_BY_POWER_SUPPLY'
/** A joint that lost parts to the power supply but still carries some. */
export const REDUCED_CODE = 'W_JOINT_REDUCED_BY_POWER_SUPPLY'

/**
 * HOW FAR THE SUPPLY STANDS PROUD OF THE FLANGE, at `i` cm inboard.
 *
 * Re-measured, and it corrects an earlier over-statement in this file. The
 * supply's top face sits at the flange's OUTER depth (1.20cm) while the flange
 * itself falls away to 1.30cm going inboard — so the supply is FLUSH at the
 * panel edge and at most 1mm proud at the flange's inner edge. It is set into
 * the housing behind the flange, not a box sitting on top of it.
 *
 * Consequences, and they pull in opposite directions:
 *   - the interference is tiny, so a 1mm RELIEF in the back half's lip clears
 *     it and a powered edge can carry a connector after all;
 *   - but the lip then bears on the SUPPLY HOUSING rather than on the panel
 *     frame, because the supply occupies the flange it would otherwise sit on.
 *
 * The tool reports that rather than choosing: `W_BEARS_ON_POWER_SUPPLY`.
 * Whether a driver housing is something to clamp against is a hardware question
 * this model cannot answer.
 */
export function supplyProudAt(i, panel = PANEL_PROFILE, supply = POWER_SUPPLY) {
  const p = { ...PANEL_PROFILE, ...panel }
  const s = { ...POWER_SUPPLY, ...supply }
  if (i < s.edgeInset || i > s.edgeInset + s.depth) return 0
  return Math.max(0, flangeDepthAt(i, p) - p.outerWallDepth)
}

/** The relief a lip reaching `gripCm` inboard needs to clear the supply. */
export function supplyReliefCm(gripCm, panel = PANEL_PROFILE, supply = POWER_SUPPLY) {
  let worst = 0
  for (let k = 0; k <= 20; k++) worst = Math.max(worst, supplyProudAt((gripCm * k) / 20, panel, supply))
  return worst
}

/**
 * How many parts a joint of `materialLength` gets.
 *
 * `spacingCm` is stated as the most unsupported joint allowed between parts, so
 * the count is a ceiling, not a rounding: at the default 50 cm a 60 cm edge gets
 * 2 and a 121 cm plate edge gets 3. `minPerJoint` is a floor beneath that, and
 * the reason it defaults to 2 is structural — see schema.js.
 */
export function stationCount(materialLength, { spacingCm, minPerJoint }) {
  const bySpacing = Math.ceil(materialLength / spacingCm - 1e-9)
  return Math.max(minPerJoint, bySpacing, 1)
}

/**
 * Place connectors on every joint of a solved layout.
 *
 * @param {object} config raw or normalized v3 config
 * @param {object} [layout] a layout from solveLayout; solved here if omitted
 * @returns {{ stations: Array, perJoint: Array, warnings: Array }}
 */
export function solveConnectors(config, layout = null) {
  const cfg = normalizeConfig(config)
  const L = layout ?? solveLayout(cfg)
  const byId = new Map(L.tiles.map((t) => [t.id, t]))
  const { lengthCm, spacingCm, minPerJoint, powerEdge, supplyMode } = cfg.connectors

  const stations = []
  const perJoint = []
  const warnings = []

  L.adjacency.forEach((edge, jointIndex) => {
    const A = byId.get(edge.a)
    const B = byId.get(edge.b)
    // Same guard as report.js: an unplaced tile has no rim to clip.
    if (!A?.position || !B?.position) return

    const span = edge.materialLength
    const wanted = stationCount(span, { spacingCm, minPerJoint })

    // --- where the power supply forbids a part ----------------------------
    // A connector grips the FLANGE, and on a powered edge the supply sits on
    // the flange for 50 of its 60cm. So the usable rim is only what is left
    // outside that, and stations are placed in those clear stretches rather
    // than evenly along a joint that cannot receive them.
    // In 'relief' mode (the default) the supply is NOT an obstruction: it is
    // flush with the flange to within 1mm, so a relief in the lip clears it and
    // the part is placed normally — but flagged, because the lip then bears on
    // the supply housing rather than the panel frame. 'block' keeps the older,
    // stricter reading so the difference can be measured.
    const supplySpans = blockedSpansOnJoint(edge, A, B, powerEdge)
    const blocked = supplyMode === 'block' ? supplySpans : []
    const clear = clearSpans(blocked, edge.edge.from, edge.edge.to)
    const clearLength = clear.reduce((n, [a, b]) => n + (b - a), 0)

    // Each clear stretch gets its own parts, sized by the same spacing rule.
    // A stretch shorter than one part gets none — it cannot hold one.
    const usable = clear.filter(([a, b]) => b - a >= lengthCm - 1e-9)
    const perStretch = usable.map(([a, b]) => Math.max(1, Math.ceil((b - a) / spacingCm - 1e-9)))
    // `minPerJoint` is a floor on the JOINT, so once spacing has had its say,
    // top up wherever there is still room for another part. Without this the
    // floor silently stopped applying as soon as a joint was split into
    // stretches by a power supply.
    let total = perStretch.reduce((n, k) => n + k, 0)
    // The floor is a FLOOR: parts are added until it is met even when that means
    // shortening them, and `W_CONNECTOR_CROWDED` reports the cost. Refusing
    // instead would silently drop a structural requirement — same contract as a
    // manual plate override, which is placed and reported rather than vetoed.
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
          a: edge.a,
          b: edge.b,
          message:
            count === 0
              ? `joint ${edge.a}–${edge.b} has a power supply behind it and only ${r(clearLength)}cm of ` +
                `usable rim in stretches too short for a ${lengthCm}cm part — it carries NO connector`
              : `joint ${edge.a}–${edge.b} has a power supply behind it: ${count} parts fit where ` +
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
        a: edge.a,
        b: edge.b,
        materialLength: r(span),
        count: 0,
        lengthCm: 0,
        blockedCm: r(span - clearLength),
      })
      return
    }

    // Each part occupies `partLength` of its stretch, centred on its station.
    const room = Math.min(...usable.map(([a, b], k) => (b - a) / perStretch[k]))
    const partLength = Math.min(lengthCm, room)
    if (partLength < lengthCm - 1e-9) {
      warnings.push({
        code: CROWDED_CODE,
        joint: jointIndex,
        a: edge.a,
        b: edge.b,
        message:
          `joint ${edge.a}–${edge.b} takes ${count} parts in ${usable.length} usable stretch` +
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

    // Rim directions and normals are constant along a joint — both tiles are
    // rigid and planar — so they are computed once per joint, not per station.
    const runA = jointEdgeRun(A, edge)
    const runB = jointEdgeRun(B, edge)
    const nA = new THREE.Vector3(...A.normal)
    const nB = new THREE.Vector3(...B.normal)
    const inA = jointEdgeInward(A, edge, true)
    const inB = jointEdgeInward(B, edge, false)

    const twistDeg = Math.acos(Math.min(1, Math.max(-1, runA.dot(runB)))) * DEG
    const dihedralDeg = Math.acos(Math.min(1, Math.max(-1, nA.dot(nB)))) * DEG

    for (let k = 0; k < count; k++) {
      const s = centres[k]
      const pa = jointEdgePoint(A, edge, true, s)
      const pb = jointEdgePoint(B, edge, false, s)
      const spanCm = pa.distanceTo(pb)

      // --- the part's own frame ------------------------------------------
      // p̂ from rim A toward rim B, perpendicular to the joint; q̂ "up"; r̂ along
      // the joint, chosen so (p̂, q̂, r̂) is RIGHT-HANDED and q̂ agrees with the
      // panels' lit side. Deriving r̂ rather than taking it from a tile is what
      // makes the fold's SIGN meaningful — a frame picked up from whichever
      // tile happened to be `a` would flip sign with tile ordering.
      let rHat = runA.clone()
      let pHat = pb.clone().sub(pa)
      pHat.addScaledVector(rHat, -pHat.dot(rHat)).normalize()
      let qHat = rHat.clone().cross(pHat)
      if (qHat.dot(nA.clone().add(nB)) < 0) {
        rHat.negate()
        qHat = rHat.clone().cross(pHat)
      }

      // Signed fold about r̂. Convex — a ridge, lit faces diverging, housings
      // pinching — comes out POSITIVE, which is the reading the drift wants.
      const cross = nA.clone().cross(nB)
      const foldDeg = -Math.atan2(cross.dot(rHat), nA.dot(nB)) * DEG

      // Bracket the span over THIS PART's footprint, not the whole joint: a
      // rigid part only has to swallow the variation it actually straddles.
      // start/end are ordered along r̂, so the lofted solid is built the same
      // way round as it is placed.
      const along = rHat.dot(runA) >= 0 ? 1 : -1
      let spanMin = Infinity
      let spanMax = -Infinity
      const spanAt = (sm) => jointEdgePoint(A, edge, true, sm).distanceTo(jointEdgePoint(B, edge, false, sm))
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
        a: edge.a,
        b: edge.b,
        axis: edge.axis,
        index: k,
        of: count,
        s: r(s),
        tAlong: r((k + 0.5) / count),
        lengthCm: r(partLength),
        spanCm: r(spanCm),
        spanMinCm: r(spanMin),
        spanMaxCm: r(spanMax),
        spanSpreadCm: r(spanMax - spanMin),
        // The part's two ends, ordered along r̂ — what the lofted solid is built
        // from. spanMin/Max are the unordered bracket; these are the signed pair.
        spanStartCm: r(spanStart),
        spanEndCm: r(spanEnd),
        dihedralDeg: r(dihedralDeg),
        // Signed: positive is convex (a ridge). dihedralDeg is its magnitude,
        // kept because report.js has always reported the unsigned angle.
        foldDeg: r(foldDeg),
        twistDeg: r(twistDeg),
        // Midpoint of the two rims: where the part's spine sits.
        mid: rv(pa.clone().add(pb).multiplyScalar(0.5)),
        frame: { p: rv(pHat), q: rv(qHat), r: rv(rHat) },
        aFrame: { point: rv(pa), run: rv(runA), normal: rv(nA), inward: rv(inA) },
        bFrame: { point: rv(pb), run: rv(runB), normal: rv(nB), inward: rv(inB) },
      })
    }

    perJoint.push({
      jointIndex,
      a: edge.a,
      b: edge.b,
      materialLength: r(span),
      count,
      lengthCm: r(partLength),
      blockedCm: r(span - clearLength),
    })
  })

  return { stations, perJoint, warnings }
}
