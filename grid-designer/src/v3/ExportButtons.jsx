/**
 * grid-designer v3 — OBJ + JSON download buttons.
 *
 * `src/utils/exporters.js` (v2, untouched) bakes world transforms and is
 * model-agnostic — it only reads `.type` / `.position` / `.quaternion` off
 * each "panel" plus a couple of naming fields. v3's tiles carry the same
 * geometric fields under the same names (`solveLayout` in
 * src/core/v3/placement.js), so only the NAMING fields differ; `toExportableLayout`
 * (exportAdapter.js) bridges that gap. See its header for the full reasoning.
 */

import useStoreV3, { getDerived } from './store.js'
import { exportConfigJSON, exportOBJ } from '../utils/exporters.js'
import { toExportableLayout } from './exportAdapter.js'

export default function ExportButtons() {
  const config = useStoreV3((s) => s.config)
  const { layout } = getDerived(config)
  const exportable = toExportableLayout(layout)

  return (
    <div className="export-bar">
      <button
        type="button"
        className="preset-btn"
        data-testid="export-obj"
        title={`bake ${exportable.panels.length} tiles into a Wavefront OBJ (one named object each)`}
        onClick={() => exportOBJ(exportable)}
      >
        Export OBJ
      </button>
      <button
        type="button"
        className="preset-btn"
        data-testid="export-json"
        title="download this config — re-import it to restore the design exactly"
        onClick={() => exportConfigJSON(config)}
      >
        Download JSON
      </button>
    </div>
  )
}
