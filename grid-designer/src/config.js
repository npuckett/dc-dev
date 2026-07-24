/**
 * Copied from panel-designer/src/config.js — do not edit the original.
 *
 * Trimmed to what grid-designer needs: panel footprint dimensions, the
 * cross-section profile, and the edge helpers used by panelGeometry.js and
 * (later) the joint report. The panel-designer-only helpers
 * (perpendicularEdges, edgeIsWidth, and the connector/tree config) were dropped.
 *
 * Units: centimeters throughout (matches the existing installation's
 * world_coordinates.json / hardware.py conventions).
 *
 * Panel profile (side view, single-sided, looking along an edge):
 *
 *      front face (lit side)
 *      ────────────────────  y = 0            (flat display surface)
 *      │   outer rim       │  y = -1.0cm       (outerThickness)
 *      └────┐              │
 *      lip  │   bevel      │  y ramps -1.0 → -3.7 over slantWidth
 *      2.5  ╲              │
 *           ╲______________│  y = -3.7cm      (overallThickness — housing body)
 *
 *  3 sides have the 2.5cm lip + bevel.  1 side (powerSupplyEdge) is flush —
 *  the full 3.7cm body extends straight to the edge with no lip.
 *
 * Panel local frame:
 *   - The panel face lies in the local XZ-plane, centered at origin.
 *   - Width  runs along local X.   Height runs along local Z.
 *   - +Y is "out" (the lit/display direction); -Y is the housing/back.
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
// PANEL PROFILE — cross-section depths & lip geometry (cm)
// =============================================================================
export const PANEL_PROFILE = {
  overallThickness: 3.7,  // full depth front-face → back of housing
  outerThickness:   1.0,  // depth of the outer rim at the very edge
  lipWidth:         2.5,  // width of the flat shallow lip on 3 sides
  slantWidth:       1.5,  // horizontal width of the bevel ramp
}

// Derived: inset of the raised body from the panel edge on lip sides.
export const BODY_INSET = PANEL_PROFILE.lipWidth + PANEL_PROFILE.slantWidth // 4.0 cm

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
