/**
 * grid-designer v4 — the ground spacer: what holds the flat cells off the floor.
 *
 * HEADLESS ZONE (src/core/): pure functions, importable from plain node.
 *   - explicit `.js` extensions on ALL relative imports
 *   - may import `three` math classes only; never components / store / DOM
 *   - same config in → byte-identical output out
 *
 * =============================================================================
 * THE REQUIREMENT, AND WHAT IT ACTUALLY ASKS FOR
 * =============================================================================
 * "Everything laying flat on the ground needs a 15cm spacer or gap, matching the
 * pattern of the other connectors."
 *
 * Two separate things, and they are built in two separate places:
 *
 *   THE GAP is `placement.groundClearanceCm`, and it belongs to GROUNDING, not
 *        here. `lattice.js` already translates the finished network in y so its
 *        lowest material lands on the floor; the clearance simply moves that
 *        target from 0 to 15. Nothing else about grounding changes, and in
 *        particular the whole network moves — a spacer under one cell that left
 *        the others where they were would tilt a rigid assembly.
 *   THE SPACER is a PART, and it is this module. It is the physical thing that
 *        holds the gap open, so it has to be counted, positioned, collided
 *        against and exported like any other part.
 *
 * =============================================================================
 * WHY ONLY THE LOWEST CELLS GET ONE
 * =============================================================================
 * A high cell is held up by the four ramps that reach it — that is what the
 * checkerboard is FOR. Putting a post under one would be inventing a second,
 * redundant load path to the floor and (at any realistic θ) a post two thirds of
 * a metre tall standing in the middle of the room. Only the cells that are
 * actually resting on the floor are propped.
 *
 * "The lowest cells" is decided by MEASURING each present cell's underside and
 * keeping the ones within 1e-6 of the lowest, rather than by testing
 * `level === 0`. The two agree on every design the checkerboard can make — cells
 * of one level are built from the same construction and come out bit-identical —
 * but the measurement keeps meaning the right thing if the level field ever
 * grows a third level (V4_SPEC §9.8), where `level === 0` would silently start
 * propping the middle storey as well.
 *
 * =============================================================================
 * "MATCHING THE PATTERN OF THE OTHER CONNECTORS" IS A REUSED RULE, NOT A LOOK
 * =============================================================================
 * The stations along a cell's edge are placed by `stationCount` from
 * `core/v3/connectors.js` — the same function, reading the same
 * `config.connectors.spacingCm` / `minPerJoint` the joint connectors read — and
 * spaced by the same "evenly, and symmetric within the stretch" rule
 * `solveConnectorsV4` uses: centres at `(m + 0.5) / n` of the edge.
 *
 * That is deliberate and it is testable: turning `connectors.spacingCm` down
 * has to move the spacer count exactly as it moves the connector count, and
 * `tests/test-v4-spacers.mjs` §6 asserts precisely that. A second spacing rule —
 * or, worse, a hardcoded four-per-cell — would look identical on the default
 * design and drift apart the first time the knob moved.
 *
 * A cell has FOUR edges and every one of them gets stations. A cell in the
 * middle of the network is joined on all four; one at a ragged edge may be
 * joined on none, and it is exactly that cell that most needs the floor.
 *
 * =============================================================================
 * WHERE THE POST STANDS, AND WHY IT IS NOT ON THE EDGE LINE
 * =============================================================================
 * A post CENTRED on the rim line would have half its section hanging in space
 * outside the panel altogether. So it is set in by half its own section: its
 * outer face is flush with the rim, and the whole of it is under material.
 *
 * The section is `PANEL_PROFILE.overallThickness` square — derived rather than
 * chosen, so the post is as wide as the housing it stands under and a change to
 * the measured panel flows through here with nothing to update (config.js's
 * header; HANDOFF §2.19 on inherited constants with no provenance).
 *
 * NOT MODELLED, and worth saying rather than implying: what the post is made of,
 * how it fixes to the panel, and whether the rim can take the load there. The
 * outer wall is only 1.2cm deep at the very edge and the section does not reach
 * full thickness until `bodyInset` inboard, so a foot bearing on the rim bears
 * on the thin part. The geometry says where the post stands and how tall it is;
 * it does not say it is strong enough. Same contract as the wall anchor (§9.4).
 *
 * =============================================================================
 * NO GROUNDING, NO SPACERS
 * =============================================================================
 * With `placement.groundToFloor` off the network is not resting on anything —
 * unit 1's reference plane sits at y = 0 and panels are free to go below the
 * floor, which `W_BELOW_FLOOR` already reports. A post from the floor to the
 * underside of a cell that is under the floor is not a part, it is an artefact
 * of asking a question that does not apply, so the solve returns nothing and
 * says why in `grounded`.
 */

import { stationCount } from '../v3/connectors.js'
import { normalizeConfig } from './schema.js'
import { solveLattice } from './lattice.js'
import { PANEL_PROFILE } from '../../config.js'

/** The code for a spacer whose height is not the clearance it was asked for. */
export const SPACER_MISMATCH_CODE = 'W_SPACER_MISMATCH'

/**
 * The post's square section. As wide as the housing it stands under — see the
 * file header on why this is derived rather than picked.
 */
export const SPACER_SECTION_CM = PANEL_PROFILE.overallThickness

/** How close two undersides must be to count as the same level, in cm. Cells of
 *  one level are built identically and agree exactly; this is float slack, not a
 *  physical tolerance. */
const LEVEL_EPSILON = 1e-6

/**
 * The four edges of a flat cell, named by the world direction they face.
 *
 * `axis` is the world axis the edge RUNS along, which is the one `s` is a
 * coordinate on — the same convention as `joint.runAxis`, so a spacer's `s` and
 * a connector's `s` are the same kind of number and never need converting
 * between parameterizations (connectors.js's header on why that matters).
 *
 * Order is fixed, because it is emission order and emission order is the
 * determinism guarantee.
 */
export const SPACER_EDGES = [
  { edge: '-x', fixed: 0, side: -1, axis: 'z', run: 2 },
  { edge: '+x', fixed: 0, side: 1, axis: 'z', run: 2 },
  { edge: '-z', fixed: 2, side: -1, axis: 'x', run: 0 },
  { edge: '+z', fixed: 2, side: 1, axis: 'x', run: 0 },
]

function r(v) {
  const out = Math.round(v * 1e9) / 1e9
  return out === 0 ? 0 : out
}

/** A panel's world axis-aligned bounds, read off the corners lattice.js emits. */
function aabb(panel) {
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
 * Put a post under every present flat cell that is resting on the floor.
 *
 * @param {object} config raw or normalized v4 config
 * @param {object} [lattice] a network from solveLattice; solved here if omitted
 * @returns {{
 *   clearanceCm: number, grounded: boolean, sectionCm: number,
 *   perEdge: number, cells: string[],
 *   spacers: Array<{ id, cell, i, j, edge, axis, s, position, heightCm, obb }>,
 *   perCell: Array<{ cell, i, j, count }>,
 *   heightsCm: number[],
 * }}
 */
export function solveSpacers(config, lattice = null) {
  const cfg = normalizeConfig(config)
  const C = lattice ?? solveLattice(cfg)
  const clearanceCm = cfg.placement.groundClearanceCm
  const { spacingCm, minPerJoint } = cfg.connectors

  const empty = {
    clearanceCm: r(clearanceCm),
    grounded: cfg.placement.groundToFloor,
    sectionCm: r(SPACER_SECTION_CM),
    perEdge: 0,
    cells: [],
    spacers: [],
    perCell: [],
    heightsCm: [],
  }
  // See "NO GROUNDING, NO SPACERS" in the file header.
  if (!cfg.placement.groundToFloor) return empty

  const flats = C.panels.filter((p) => p.kind === 'cell' && p.present)
  if (flats.length === 0) return empty

  // The lowest UNDERSIDE, measured — see the file header on why this is not
  // `level === 0`.
  const boxes = new Map(flats.map((p) => [p.id, aabb(p)]))
  let lowest = Infinity
  for (const p of flats) lowest = Math.min(lowest, boxes.get(p.id).min[1])
  const resting = flats.filter((p) => boxes.get(p.id).min[1] - lowest <= LEVEL_EPSILON)

  const spacers = []
  const perCell = []
  let perEdge = 0

  for (const cell of resting) {
    const box = boxes.get(cell.id)
    // The height the post SPANS, measured from the floor to the underside it
    // holds up — never written down as `clearanceCm`. That is what makes
    // W_SPACER_MISMATCH able to catch grounding and this module disagreeing; a
    // spacer that reports the number it was asked for cannot detect anything.
    const heightCm = box.min[1]
    let count = 0

    for (const spec of SPACER_EDGES) {
      const from = box.min[spec.run]
      const to = box.max[spec.run]
      const n = stationCount(to - from, { spacingCm, minPerJoint })
      perEdge = n
      // Set in by half the section, so the post's outer face is flush with the
      // rim rather than half of it hanging outside the panel.
      const fixed = spec.side < 0
        ? box.min[spec.fixed] + SPACER_SECTION_CM / 2
        : box.max[spec.fixed] - SPACER_SECTION_CM / 2

      for (let m = 0; m < n; m++) {
        // The same "evenly spaced, symmetric within the stretch" rule
        // solveConnectorsV4 places its own station centres by.
        const s = from + (to - from) * ((m + 0.5) / n)
        const position = [0, 0, 0]
        position[spec.fixed] = r(fixed)
        position[spec.run] = r(s)
        position[1] = r(heightCm)

        spacers.push({
          id: `Sp${cell.id}${spec.edge}${m}`,
          cell: cell.id,
          i: cell.i,
          j: cell.j,
          edge: spec.edge,
          axis: spec.axis,
          index: m,
          of: n,
          s: r(s),
          // The underside point the post bears on. Its FOOT is the same x/z at
          // y = 0, which is what makes `heightCm` the whole story.
          position,
          heightCm: r(heightCm),
          // In the exact shape collide.js consumes. Axis-aligned — a flat cell
          // is flat whichever way up it is turned — so the quaternion is
          // identity and `obbPenetration` is exact here with no special case,
          // the same reasoning obstacles.js gives for the room's column.
          obb: {
            center: [position[0], r(heightCm / 2), position[2]],
            halfExtents: [r(SPACER_SECTION_CM / 2), r(Math.abs(heightCm) / 2), r(SPACER_SECTION_CM / 2)],
            quaternion: [0, 0, 0, 1],
          },
        })
        count++
      }
    }
    perCell.push({ cell: cell.id, i: cell.i, j: cell.j, count })
  }

  // Distinct heights, sorted. There is exactly one on any design where grounding
  // and this module agree, and the report says so out loud rather than leaving
  // the reader to notice a second entry.
  const heightsCm = [...new Set(spacers.map((s) => s.heightCm))].sort((a, b) => a - b)

  return {
    clearanceCm: r(clearanceCm),
    grounded: true,
    sectionCm: r(SPACER_SECTION_CM),
    perEdge,
    cells: resting.map((p) => p.id),
    spacers,
    perCell,
    heightsCm,
  }
}
