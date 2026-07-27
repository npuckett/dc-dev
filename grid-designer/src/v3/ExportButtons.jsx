/**
 * grid-designer v3 — download buttons.
 *
 * `src/utils/exporters.js` (v2, untouched at P5) bakes world transforms and is
 * model-agnostic — it only reads `.type` / `.position` / `.quaternion` off
 * each "panel" plus a couple of naming fields. v3's tiles carry the same
 * geometric fields under the same names (`solveLayout` in
 * src/core/v3/placement.js), so only the NAMING fields differ; `toExportableLayout`
 * (exportAdapter.js) bridges that gap. See its header for the full reasoning.
 *
 * Four things leave this tool, and they answer different questions:
 *
 *   OBJ       the assembly — every panel AND every connector, world-baked, for
 *             looking at it somewhere else.
 *   JSON      the config — the single source of truth; re-import restores the
 *             design exactly.
 *   STL       one of each unique connector type, in MILLIMETRES, oriented for
 *             printing. What goes to the slicer.
 *   manifest  how many of each to run, and what each is forced to absorb by not
 *             getting its own exact geometry. What goes with the STL — the STL
 *             alone cannot say that P00 is needed 46 times.
 */

import useStoreV3, { getDerived } from './store.js'
import { exportConfigJSON, exportOBJ } from '../utils/exporters.js'
import { exportConnectorPlateSTL, exportConnectorManifest } from '../utils/connectorExport.js'
import { toExportableLayout } from './exportAdapter.js'

export default function ExportButtons() {
  const config = useStoreV3((s) => s.config)
  const { layout, report } = getDerived(config)
  const exportable = toExportableLayout(layout, report.connectors)
  const conn = report.connectors.summary

  return (
    <div className="export-bar">
      <button
        type="button"
        className="preset-btn"
        data-testid="export-obj"
        title={`bake ${exportable.panels.length} tiles and ${conn.count} connectors into a Wavefront OBJ (one named object each)`}
        onClick={() => exportOBJ(exportable)}
      >
        Export OBJ
      </button>
      <button
        type="button"
        className="preset-btn"
        data-testid="export-connector-stl"
        title={`one of each of the ${conn.partTypes} unique connector types, in millimetres, laid out flat for printing`}
        onClick={() => exportConnectorPlateSTL(report)}
      >
        Connector STL
      </button>
      <button
        type="button"
        className="preset-btn"
        data-testid="export-connector-manifest"
        title={`how many of each type to print (${conn.count} parts over ${conn.jointCount} joints) and what each one is forced to absorb`}
        onClick={() => exportConnectorManifest(config, report)}
      >
        Connector manifest
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
