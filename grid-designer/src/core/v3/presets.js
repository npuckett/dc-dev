/**
 * grid-designer v3 — drift presets.
 *
 * HEADLESS ZONE (src/core/): pure functions, importable from plain node.
 *   - explicit `.js` extensions on ALL relative imports
 *   - may import `three` math classes only; never components / store / DOM
 *   - same id in → byte-identical config out
 *
 * =============================================================================
 * WHAT THE PRESETS ARE FOR
 * =============================================================================
 * These are not decoration. Each one pins a different answer to the question
 * the v3 model forces and cannot answer on its own:
 *
 *   HOW MUCH JOINT DEVIATION IS THE INSTALLATION WILLING TO BUILD?
 *
 * Three physical facts collide, and every preset is a chosen point among them:
 *
 * 1. HEIGHT COSTS JOINT DEVIATION. Rigid 60cm panels on a curved target diverge
 *    at their shared edges by roughly (panel size)·(curvature). Push the crest up
 *    and the joints wedge open. This is geometry, not an implementation limit.
 *
 * 2. THE NOMINAL GAP BUYS HEIGHT — AND COSTS MODULARITY. On convex curvature the
 *    lit faces open while the HOUSINGS converge, so a 1cm joint runs the panels
 *    into each other well before the drift gets interesting (first collision at
 *    ~40cm of amplitude). Widening the gap relieves that directly: at gap 2 the
 *    same form reaches a 95cm peak with no collisions at all.
 *    BUT `2·size + gap` is what makes a 60×121 plate an exact drop-in for two
 *    squares plus their joint — 60+1+60 = 121 EXACTLY (HANDOFF.md §2.2). At gap
 *    2 that becomes 122 against a 121cm plate: a 1cm mismatch per plate, which
 *    `W_PLATE_LENGTH` reports and which does not go away. The hardware plate is a
 *    standard 121cm drop-in size, so the mismatch is real, not a config error.
 *
 * 3. FACETING CLOSES THE JOINTS AND LIFTS THE EDGES. An angular target is one
 *    rigid panels can BE rather than approximate (see target.js), so faceting
 *    collapses joint deviation — but a broad facet plane cannot hug the toe, so
 *    the graded edges the brief pins to the floor rise a few cm to ~20cm.
 *
 * Every figure quoted below was MEASURED, not estimated. Re-measure with
 * tests/test-v3-presets.mjs if any of the core math changes.
 */

import { DEFAULT_CONFIG, normalizeConfig } from './schema.js'

/**
 * @typedef {object} PresetDef
 * @property {string} id
 * @property {string} label
 * @property {string} trade   what this preset chooses, and what it gives up
 */

const PRESET_DEFS = [
  {
    id: 'shelf',
    label: 'shelf — the graded edges actually land',
    trade:
      'The brief-compliant build, and the only one where the graded edges really ' +
      'land: 4 of 5 tiles touching on the wall edge at 2.1cm worst clearance, ' +
      '4 of 4 on the window edge at 1.6cm, and no collisions. ONE facet per panel ' +
      'is what does it — a broad facet plane cannot hug the toe. Costs height ' +
      '(peak ~44cm) and ~4cm of worst joint deviation, and trips W_TOE_FLAT at ' +
      'the wall/window corner unavoidably, since a plane grounded along two ' +
      'intersecting floor lines IS the floor.',
    sheet: { cols: 6, rows: 7 },
    gap: 3,
    form: {
      amplitude: 36, crestX: 0.6, crestZ: 0.5, ridgeShear: 0.35,
      toeSharpX: 1.0, toeSharpZ: 1.0, angularity: 0.5, facetCells: 1,
    },
  },
  {
    id: 'closed',
    label: 'closed — not one joint out of tolerance',
    trade:
      'The only preset with ZERO flagged joints: every joint sits inside ' +
      'gapTolerance and nothing collides. Broad 4-cell facets are what buy that, ' +
      'and they are also why the graded edges sit ~8cm up rather than on the ' +
      'floor. Runs a 3cm joint, so each plate carries a 2cm modular mismatch.',
    sheet: { cols: 6, rows: 7 },
    gap: 3,
    form: {
      amplitude: 40, crestX: 0.6, crestZ: 0.5, ridgeShear: 0.24,
      toeSharpX: 0.8, toeSharpZ: 0.8, angularity: 0.75, facetCells: 4,
    },
  },
  {
    id: 'drift',
    label: 'drift — the balanced default',
    trade:
      'The recommended starting point. Peak ~62cm with worst joint deviation ' +
      'under 2.5cm and no collisions, at the price of a 3cm joint and graded ' +
      'edges ~14cm off the floor.',
    sheet: { cols: 6, rows: 8 },
    gap: 3,
    form: {
      amplitude: 60, crestX: 0.6, crestZ: 0.5, ridgeShear: 0.24,
      toeSharpX: 1.0, toeSharpZ: 1.0, angularity: 1, facetCells: 4,
    },
  },
  {
    id: 'dune',
    label: 'dune — tall, still closed',
    trade:
      'Pushes the crest to ~87cm and holds the joints under about 9cm by ' +
      'faceting hard. The graded edges pay for it, rising to ~19cm: broad facet ' +
      'planes cannot hug the toe.',
    sheet: { cols: 6, rows: 8 },
    gap: 2,
    form: {
      amplitude: 90, crestX: 0.6, crestZ: 0.5, ridgeShear: 0.35,
      toeSharpX: 1.0, toeSharpZ: 1.0, angularity: 1, facetCells: 4,
    },
  },
  {
    id: 'modular',
    label: 'modular — exact 1cm joint, no plate mismatch',
    trade:
      'The only preset that keeps 60+1+60 = 121 exactly, so a plate is a true ' +
      'drop-in for two squares and the kit stays modular. That 1cm joint is what ' +
      'limits it: the housings converge on curvature, so the crest has to stay ' +
      'near ~50cm to keep panels out of each other.',
    sheet: { cols: 6, rows: 8 },
    gap: 1,
    form: {
      amplitude: 45, crestX: 0.6, crestZ: 0.5, ridgeShear: 0.24,
      toeSharpX: 1.0, toeSharpZ: 1.0, angularity: 1, facetCells: 4,
    },
  },
  {
    id: 'crest',
    label: 'crest — v2 height, and what it costs',
    trade:
      'Reaches the ~105cm neighbourhood of the v2 swell, deliberately, so the ' +
      'comparison is on the record. It is NOT a closed design: worst joint ' +
      'deviation is around 27cm and a pair of panels still collide. Kept because ' +
      'v2 shipped presets with 49cm deviation, so this is the honest ' +
      'like-for-like — and because the report should be able to say no.',
    sheet: { cols: 6, rows: 7 },
    gap: 3,
    form: {
      amplitude: 100, crestX: 0.6, crestZ: 0.5, ridgeShear: 0.35,
      toeSharpX: 1.0, toeSharpZ: 1.0, angularity: 1, facetCells: 4,
    },
  },
]

export const PRESETS = PRESET_DEFS.map((p) => ({ id: p.id, label: p.label, trade: p.trade }))
export const PRESET_IDS = PRESET_DEFS.map((p) => p.id)
export const DEFAULT_PRESET = 'drift'

/**
 * Build a preset's config. Deterministic and normalized, so the store can
 * commit it directly.
 *
 * `form.footprint` is deliberately OMITTED so `normalizeConfig` derives it from
 * the sheet — a drift shorter than its sheet leaves a slope discontinuity the
 * straddling panels cannot follow (see schema.js's withDefaults).
 *
 * @param {string} id one of PRESET_IDS
 * @returns {object|null} normalized v3 config, or null for an unknown id
 */
export function buildPreset(id) {
  const def = PRESET_DEFS.find((p) => p.id === id)
  if (!def) return null
  return normalizeConfig({
    ...DEFAULT_CONFIG,
    name: def.id,
    sheet: { ...def.sheet },
    gap: def.gap,
    form: { ...def.form },
    tiling: { ...DEFAULT_CONFIG.tiling },
    placement: { ...DEFAULT_CONFIG.placement },
    meta: { preset: def.id, notes: def.trade },
  })
}
