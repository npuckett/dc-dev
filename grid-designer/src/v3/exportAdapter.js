/**
 * grid-designer v3 — adapts a v3 `solveLayout` output to the shape
 * `src/utils/exporters.js` expects.
 *
 * `src/utils/exporters.js` (v2, UNTOUCHED per the P5 work order) is genuinely
 * model-agnostic where it matters: `buildExportGroup` only ever reads
 * `layout.panels`, and each panel's `.type` / `.position` / `.quaternion` to
 * bake its world transform — nothing v2-specific. But its panel NAMING
 * (`exportPanelName`) reads `.rectOrientation` / `.row` / `.col`, fields v3's
 * tiles do not carry (v3 has `.axis` ('u'|'v') and `.cells` ([[i,j],...])
 * instead — V3_SPEC.md §1's material coordinates). So the "call site" the P5
 * work order asks to fix is exactly this: adapt field NAMES, not the exporter
 * itself.
 *
 * @param {object} layout a v3 `solveLayout()` result
 * @returns {{ panels: Array }} shape `buildExportGroup`/`objPayload` expect
 */
export function toExportableLayout(layout) {
  const panels = layout.tiles
    .filter((t) => Array.isArray(t.position)) // unreachable (disconnected-graph) tiles carry no position
    .map((t) => {
      const [i, j] = t.cells[0]
      return {
        ...t,
        // material i ~ wall-to-window axis ('u'), j ~ window-to-back axis ('v') —
        // reused here purely as a stable, human-readable OBJ object name, not as
        // a claim about v2 row/column semantics (V3_SPEC.md §1: "the code must
        // call them i/j or u/v, never row/col").
        row: j,
        col: i,
        rectOrientation: t.type === '2x4' ? (t.axis === 'u' ? 'horizontal' : 'vertical') : undefined,
      }
    })
  return { panels }
}
