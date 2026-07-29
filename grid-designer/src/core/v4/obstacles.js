/**
 * grid-designer v4 — things in the room the network has to be planned around.
 *
 * HEADLESS ZONE (src/core/): pure functions, importable from plain node.
 *   - explicit `.js` extensions on ALL relative imports
 *   - may import `three` math classes only; never components / store / DOM
 *   - same config in → byte-identical output out
 *
 * =============================================================================
 * WHAT AN OBSTACLE IS, AND WHAT IT IS NOT
 * =============================================================================
 * An axis-aligned box standing on the floor. Two kinds, drawn differently and
 * tested identically (schema.js's `OBSTACLE_KINDS`):
 *
 *   'solid'  material in the room — a structural column, a duct, a plinth.
 *   'zone'   RESERVED EMPTY SPACE the installation must keep out of — the
 *            heating run along the window, a service route, a swing clearance.
 *
 * Neither is part of the design and nothing about either is derived from the
 * lattice — they are facts about the room, in the same category as the wall
 * plane at x = 0.
 *
 * So this module only ever ANSWERS QUESTIONS about the design; it never changes
 * it. Nothing here removes a panel, moves the lattice, or refuses a
 * configuration. A column that the network runs through is reported, in as many
 * words, and the panels it hits are named so they can be switched off in the
 * plan editor — which is the same "report the cost, do not veto" contract the
 * connector flags and the plate overrides already follow.
 *
 * =============================================================================
 * WHERE THE NUMBERS ARE MEASURED FROM
 * =============================================================================
 * The room datum is **x = 0 at the wall's room-side face, y = 0 at the floor,
 * z = 0 at the window side** — the origin every other dimension in the tool is
 * quoted from, and the one `RibbonViewport`'s origin triad draws.
 *
 * **z = 0 IS A REFERENCE PLANE, NOT THE GLASS.** It marks the window SIDE of
 * the room, and the actual window — its position, its mullions — is being
 * measured in as a set of elements sitting at NEGATIVE z. So obstacles legally
 * take negative coordinates and the ranges are signed; anything here that
 * assumed the room lives in the positive quadrant would be wrong. The heating
 * run is the first of these: x −81.3 → 512.7, z −59.7 → 0, which also reaches
 * past the wall face at x = 0.
 *
 * Obstacles carry a `baseYCm`, so they need not stand on the floor — the window
 * mullions start 25cm BELOW it. y is given outright rather than anchored:
 * "how far up does it start" has no corner/centre ambiguity to resolve.
 *
 * `anchor` says which part of the box `(xCm, zCm)` locates, because the two
 * readings differ by half its width and there is no way to guess from a pair of
 * numbers alone:
 *
 *   'corner'  (default) the box's MINIMUM x and z corner — what a tape measure
 *             from the datum to the near face of a column gives you.
 *   'centre'  the box's centre line.
 *
 * The UI shows the resulting extents next to the inputs so that a wrong reading
 * is visible immediately rather than after a collision report that looks
 * plausible. Getting this wrong is a 25cm error on a 50cm column, which is
 * exactly the size of mistake that survives a review.
 *
 * =============================================================================
 * THE OVERLAP TEST IS DELIBERATELY CONSERVATIVE
 * =============================================================================
 * Panels are tested as their full OBBs, the same boxes `collide.js` uses. That
 * box overstates the real panel near its rim, where the section is only the
 * 1.2cm outer wall rather than the full 4.1cm (see report.js's note on the
 * corner clearance, where the same overstatement made a verdict unreportable).
 *
 * Here that bias is the RIGHT WAY ROUND and is left alone. A structural column
 * is not something to clear by a millimetre, and a test that says "this panel
 * fouls the column" slightly too eagerly costs one click in the plan editor,
 * while one that says it slightly too late costs a site visit. Clearance is
 * reported the same way — as a lower bound.
 */

import { obbPenetration } from '../v3/collide.js'

/** A column standing floor-to-ceiling: tall enough that only its FOOTPRINT
 *  decides anything, which is the assumption the report's wording relies on. */
export const DEFAULT_OBSTACLE_HEIGHT_CM = 300

function r(v) {
  const out = Math.round(v * 1e9) / 1e9
  return out === 0 ? 0 : out
}

/**
 * An obstacle's world extents, resolving `anchor`.
 * @returns {{ min: number[], max: number[], centre: number[], size: number[] }}
 */
export function obstacleExtents(o) {
  const w = o.widthCm
  const d = o.depthCm
  const h = o.heightCm ?? DEFAULT_OBSTACLE_HEIGHT_CM
  // `baseYCm` is where the box STARTS in y, and it is routinely negative — a
  // mullion runs down past floor level. This used to be hardcoded to 0, which
  // silently pinned everything to the floor; the only way to place something
  // starting below it was to inflate `heightCm`, which puts the TOP in the
  // wrong place. Anchoring applies to x and z only: y is stated outright,
  // because "how far up does it start" has no corner/centre ambiguity.
  const y0 = o.baseYCm ?? 0
  const x0 = o.anchor === 'centre' ? o.xCm - w / 2 : o.xCm
  const z0 = o.anchor === 'centre' ? o.zCm - d / 2 : o.zCm
  return {
    min: [r(x0), r(y0), r(z0)],
    max: [r(x0 + w), r(y0 + h), r(z0 + d)],
    centre: [r(x0 + w / 2), r(y0 + h / 2), r(z0 + d / 2)],
    size: [r(w), r(h), r(d)],
  }
}

/**
 * An obstacle as an OBB in the shape `collide.js` consumes. Axis-aligned, so
 * the quaternion is identity — which is also why `obbPenetration` is exact here
 * with no special case: a box is just an OBB that happens not to be rotated.
 */
export function obstacleOBB(o) {
  const e = obstacleExtents(o)
  return {
    center: e.centre,
    halfExtents: [e.size[0] / 2, e.size[1] / 2, e.size[2] / 2],
    quaternion: [0, 0, 0, 1],
  }
}

/** A panel's world axis-aligned bounds, read off the corners lattice.js emits. */
function panelAabb(panel) {
  const min = [Infinity, Infinity, Infinity]
  const max = [-Infinity, -Infinity, -Infinity]
  for (const c of panel.corners) {
    for (let k = 0; k < 3; k++) {
      if (c[k] < min[k]) min[k] = c[k]
      if (c[k] > max[k]) max[k] = c[k]
    }
  }
  return { min, max }
}

/**
 * The gap between an obstacle and a panel, as a LOWER BOUND.
 *
 * Measured between the obstacle's box and the panel's world AABB. Because the
 * obstacle is axis-aligned, an axis-aligned gap can only UNDERSTATE the true
 * separation of a tilted panel — never overstate it. That is the safe direction
 * for a structural column (see the file header), and saying "at least 40cm"
 * is a useful sentence where an exact figure computed wrongly is not.
 *
 * Returns 0 when the boxes overlap or touch in plan and height.
 */
export function obstacleClearanceCm(o, panel) {
  const e = obstacleExtents(o)
  const p = panelAabb(panel)
  let sq = 0
  for (let k = 0; k < 3; k++) {
    const gap = Math.max(e.min[k] - p.max[k], p.min[k] - e.max[k], 0)
    sq += gap * gap
  }
  return Math.sqrt(sq)
}

/** The code a panel intersecting an obstacle is reported under. */
export const OBSTACLE_HIT_CODE = 'W_PANEL_HITS_OBSTACLE'

/**
 * Test every present panel against every obstacle.
 *
 * @param {object} lattice a solved lattice
 * @param {Array} obstacles normalized obstacle records
 * @returns {Array} one entry per obstacle: its extents, the panels it hits, and
 *   the closest clearance to anything that misses it
 */
export function solveObstacles(lattice, obstacles) {
  return (obstacles ?? []).map((o) => {
    const box = obstacleOBB(o)
    const extents = obstacleExtents(o)
    const hits = []
    let nearestCm = Infinity
    let nearest = null

    for (const panel of lattice.panels) {
      if (!panel.present) continue
      const pen = obbPenetration(box, panel.obb)
      if (pen) {
        hits.push({ id: panel.id, kind: panel.kind, depthCm: r(pen.depthCm) })
        continue
      }
      const gap = obstacleClearanceCm(o, panel)
      if (gap < nearestCm) {
        nearestCm = gap
        nearest = panel.id
      }
    }

    hits.sort((a, b) => b.depthCm - a.depthCm || (a.id < b.id ? -1 : 1))
    return {
      id: o.id,
      label: o.label ?? o.id,
      // Carried through for the viewport: a reserved ZONE and a SOLID are drawn
      // differently and tested identically (schema.js's OBSTACLE_KINDS).
      kind: o.kind ?? 'solid',
      // Repeated sub-elements suppress their name — see schema.js.
      // The FLAG decides, not the text: sanitizeObstacles falls an empty label
      // back to the id (so nothing is ever nameless in a warning), which means
      // emptiness cannot be the signal.
      labelled: o.labelled !== false,
      extents,
      hits,
      hitCount: hits.length,
      // Only meaningful when nothing is hit; a design that already runs through
      // the column has no clearance to quote and the field says so.
      nearestPanel: hits.length ? null : nearest,
      nearestClearanceCm: hits.length || !Number.isFinite(nearestCm) ? null : r(nearestCm),
    }
  })
}
