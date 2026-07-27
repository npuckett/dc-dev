/**
 * grid-designer — the panel: footprint, cross-section, and the power supply.
 *
 * PROVENANCE
 * =============================================================================
 * Began as a copy of panel-designer/src/config.js (do not edit that original).
 * The cross-section is no longer that file's: it was re-derived from the
 * measured models in `updatedPanelGeo/` (`2x2panel.obj`, `2x4panel.obj`), which
 * describe a materially different panel from the four numbers inherited here.
 * The old section had NO back flange — its taper began at the outer wall — and
 * was 3.7cm thick. Both are wrong; see PANEL_PROFILE below.
 *
 * THE NUMBERS ARE INPUTS, NOT FACTS
 * =============================================================================
 * The source models are close but not exact, and will be refined. So every
 * dimension below is an independently measurable PARAMETER, and everything else
 * — the solid, the collision boxes, the connector's grip — is DERIVED from them
 * through `panelSectionRings()`. Refining the panel means editing numbers here
 * and nothing else. Nothing downstream may hardcode a section dimension; if it
 * needs one, it derives it, or the two will drift apart the moment a caliper
 * disagrees with this file.
 *
 * Units: centimetres throughout, matching the installation's
 * world_coordinates.json / hardware.py conventions. The source OBJs are a tenth
 * of that (their "6.000" square is the 60cm panel), so they were scaled by 10.
 *
 * THE SECTION (looking along an edge; identical on all four edges)
 * =============================================================================
 * Depths are measured DOWN FROM THE FRONT-MOST PLANE, which is local y = 0 —
 * the convention core/v3/placement.js already reasons in, so it is unchanged.
 *
 *      inboard  ←──────────────────────────  0 = the outer edge
 *
 *   bezel peak ──┐                                          depth 0
 *                │╲___                                      bezelDrop 0.10
 *      diffuser  │    ╲______________ front outer corner
 *      (recessed)│                   │                      diffuserDepth 0.363
 *                │                   │  outer wall, 1.10 high
 *                │                   │
 *                │     ______________│  back outer corner   outerWallDepth 1.20
 *                │    /   flange 3.0cm — nearly flat        flangeDrop 0.10
 *                │   /
 *                │  /  taper 1.62cm at ~60°
 *                │_/                                        overallThickness 4.10
 *                   back plate
 *
 * THE BACK FLANGE IS THE POINT. It is 3cm of essentially flat material with open
 * air behind it, and it is what a connector should grip. The front is a 1.5cm
 * chamfered bezel rising to a peak over a recessed diffuser — a decorative
 * surface on the lit face, and a poor thing to clamp.
 *
 * Panel local frame (unchanged):
 *   - The panel face lies in the local XZ-plane, centred at the origin.
 *   - Width runs along local X. Height runs along local Z.
 *   - +Y is "out" (the lit direction); the housing runs back to
 *     y = −overallThickness.
 *
 *       north (edge 2, +Z side)
 *   ┌──────────────────────────┐
 *   │                          │
 * w │        front face         │  height (Z)
 * e │       faces +Y (out)      │
 * s │                          │
 *   └──────────────────────────┘
 *       south (edge 0, -Z side)
 *   west(edge3,−X)   east(edge1,+X)
 */

// =============================================================================
// PANEL TYPES — footprint sizes (cm)
// =============================================================================
export const PANEL_DIMENSIONS = {
  '2x2': {
    width: 60,   // short edge
    height: 60,  // long edge (= short for square)
    label: '2×2 (60×60cm)',
  },
  '2x4': {
    width: 60,   // short edge
    height: 121, // long edge
    label: '2×4 (60×121cm)',
  },
}

// =============================================================================
// PANEL PROFILE — the measured cross-section (cm)
//
// Every value is a caliper measurement off `updatedPanelGeo/`, taken as a DEPTH
// below the front-most plane (local y = 0) or as an INBOARD offset from the
// panel edge. These are the refinement surface: change a number here and the
// solid, the collision boxes and the connector grip all follow.
// =============================================================================
export const PANEL_PROFILE = {
  // --- front ---------------------------------------------------------------
  /** How far in the front bezel runs before dropping to the diffuser. */
  bezelWidth: 1.5,
  /** The bezel falls this far from its inner peak out to the panel edge. */
  bezelDrop: 0.1,
  /** The lit surface, recessed below the front-most plane. */
  diffuserDepth: 0.363,

  // --- edge ----------------------------------------------------------------
  /** Back outer corner. The outer wall runs bezelDrop → here: 1.10cm of it. */
  outerWallDepth: 1.2,

  // --- back ----------------------------------------------------------------
  /**
   * THE FLANGE. 3cm of nearly-flat material around the whole perimeter with
   * open air behind it — the feature a connector grips. Its absence from the
   * old inherited section is why the first connector design made no sense.
   */
  flangeWidth: 3.0,
  /** The flange falls this far across its width — 0.1 over 3.0, about 1.9°. */
  flangeDrop: 0.1,
  /** Horizontal run of the steep (~60°) taper from the flange to the back plate. */
  taperWidth: 1.62,
  /** Front-most plane to the back of the housing. */
  overallThickness: 4.1,
}

/**
 * Values implied by PANEL_PROFILE. Never write one of these down as a constant
 * — derive it, or it will disagree with the profile the moment that is refined.
 */
export const PANEL_METRICS = {
  /** Height of the vertical outer wall — what a rim clamp closes across. */
  get outerWallHeight() {
    return PANEL_PROFILE.outerWallDepth - PANEL_PROFILE.bezelDrop
  },
  /** Depth of the flange at its inner edge, where the taper takes over. */
  get flangeInnerDepth() {
    return PANEL_PROFILE.outerWallDepth + PANEL_PROFILE.flangeDrop
  },
  /** How far in the back plate begins. */
  get bodyInset() {
    return PANEL_PROFILE.flangeWidth + PANEL_PROFILE.taperWidth
  },
  /** How far the diffuser sits below the bezel peak. */
  get diffuserRecess() {
    return PANEL_PROFILE.diffuserDepth
  },
}

/** Inset of the back plate from the panel edge. Derived — see PANEL_METRICS. */
export const BODY_INSET = PANEL_METRICS.bodyInset

/**
 * The section as a sequence of rings, each `{ inset, depth }`: how far inboard
 * of the panel edge the ring sits, and how far below the front-most plane.
 *
 * THE SINGLE SOURCE OF TRUTH FOR THE SECTION SHAPE. `geometry/panelGeometry.js`
 * sweeps it into the solid and `core/v3/connectors.js` reads it to place a grip
 * on it. Two modules deriving the same physical surface from one description is
 * the standing rule here (HANDOFF §2.8).
 *
 * Ordered front → outward → back, which is the traversal that yields outward
 * face normals when consecutive rings are connected.
 */
export function panelSectionRings(profile = PANEL_PROFILE) {
  const p = { ...PANEL_PROFILE, ...profile }
  return [
    { name: 'diffuser', inset: p.bezelWidth, depth: p.diffuserDepth },
    { name: 'bezelPeak', inset: p.bezelWidth, depth: 0 },
    { name: 'frontEdge', inset: 0, depth: p.bezelDrop },
    { name: 'backEdge', inset: 0, depth: p.outerWallDepth },
    { name: 'flangeInner', inset: p.flangeWidth, depth: p.outerWallDepth + p.flangeDrop },
    { name: 'backPlate', inset: p.flangeWidth + p.taperWidth, depth: p.overallThickness },
  ]
}

// =============================================================================
// POWER SUPPLY — the obstruction on one edge of every panel
//
// A box on the BACK, running along one 60cm edge on both panel types. It sits
// directly on the flange (1.5mm to 36.5mm inboard, against a flange spanning 0
// to 30mm), so on that edge the flange is covered for 50 of its 60cm and only
// ~5cm at each end is clear.
//
// That is a hard placement constraint, not a detail: a connector gripping the
// flange cannot go anywhere along the middle of a powered edge. It is very
// likely why v1's connectors sat out at the corners.
// =============================================================================
export const POWER_SUPPLY = {
  /** Along the edge. Both panel types carry the same 50cm box. */
  length: 50,
  /** Inboard from the panel edge — deeper than the 3cm flange, so it covers it. */
  depth: 3.5,
  /** Down from the flange plane toward the back plate. */
  height: 2.7,
  /** Gap between the panel edge and the near face of the box. */
  edgeInset: 0.15,
}

/**
 * The span of a powered edge that a flange-gripping connector CANNOT use,
 * as `[from, to]` in cm measured along that edge from one end.
 *
 * Centred, because the box is centred on the edge in both source models.
 */
export function poweredEdgeBlockedSpan(edgeLengthCm, supply = POWER_SUPPLY) {
  const clear = Math.max(0, (edgeLengthCm - supply.length) / 2)
  return [clear, edgeLengthCm - clear]
}

// =============================================================================
// EDGES — index convention (0–3), CCW from south
//
//   edge 0 = south (-Z)    runs along +X  (width)
//   edge 1 = east  (+X)    runs along +Z  (height)
//   edge 2 = north (+Z)    runs along -X  (width)
//   edge 3 = west  (-X)    runs along -Z  (height)
// =============================================================================
export const EDGE_NAMES = ['south', 'east', 'north', 'west']

/**
 * Get the two endpoints (local coords) of a panel edge.
 * Returns [start, end] as [x, z] pairs in panel-local frame (centered at origin).
 * Traversed counter-clockwise.
 */
export function edgeEndpoints(edge, panelType, dims = PANEL_DIMENSIONS) {
  const hw = dims[panelType].width / 2
  const hh = dims[panelType].height / 2
  switch (edge) {
    case 0: return [[-hw, -hh], [hw, -hh]]   // south → +X
    case 1: return [[hw, -hh], [hw, hh]]     // east  → +Z
    case 2: return [[hw, hh], [-hw, hh]]     // north → -X
    case 3: return [[-hw, hh], [-hw, -hh]]   // west  → -Z
    default: return null
  }
}

/**
 * Direction vector (local XZ) along which an edge runs (CCW).
 */
export function edgeDirection(edge) {
  switch (edge) {
    case 0: return [1, 0]    // south → +X
    case 1: return [0, 1]    // east  → +Z
    case 2: return [-1, 0]   // north → -X
    case 3: return [0, -1]   // west  → -Z
    default: return [1, 0]
  }
}

/**
 * Outward-pointing normal (local XZ) of an edge (away from panel center).
 */
export function edgeOutwardNormal(edge) {
  switch (edge) {
    case 0: return [0, -1]   // south → -Z
    case 1: return [1, 0]    // east  → +X
    case 2: return [0, 1]    // north → +Z
    case 3: return [-1, 0]   // west  → -X
    default: return [0, -1]
  }
}
