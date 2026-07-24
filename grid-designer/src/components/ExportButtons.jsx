/**
 * grid-designer — OBJ + JSON download buttons.
 *
 * Both act on the CURRENT state: the OBJ is baked from the derived layout, the
 * JSON is the config itself (which re-solves to a byte-identical layout, so the
 * pair is a complete, reversible record of a design).
 */

import useStore, { getDerived } from '../store.js'
import { exportConfigJSON, exportOBJ } from '../utils/exporters.js'

export default function ExportButtons() {
  const config = useStore((s) => s.config)
  const { layout } = getDerived(config)

  return (
    <div className="export-bar">
      <button
        type="button"
        className="preset-btn"
        data-testid="export-obj"
        title={`bake ${layout.panels.length} panels into a Wavefront OBJ (one named object each)`}
        onClick={() => exportOBJ(layout)}
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
