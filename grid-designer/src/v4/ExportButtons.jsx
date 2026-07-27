/**
 * grid-designer v4 — download buttons.
 *
 * `src/utils/exporters.js` and `src/utils/connectorExport.js` are model-agnostic
 * where it matters — they read `.type` / `.position` / `.quaternion` off each
 * panel and `.mid` / `.frame` off each station, all of which a v4 unit and a v4
 * station carry under exactly those names. `exportAdapter.js` bridges the rest;
 * see its header for the full inventory of what the frozen core does not expose
 * and is assembled there instead.
 *
 * Four things leave this tool, and they answer different questions:
 *
 *   OBJ       the assembly — every PRESENT panel and every connector,
 *             world-baked, for looking at it somewhere else.
 *   JSON      the config — the single source of truth; re-import restores the
 *             design exactly.
 *   STL       one of each unique connector type, in MILLIMETRES, oriented for
 *             printing. What goes to the slicer.
 *   manifest  how many of each to run, and what each is forced to absorb by not
 *             getting its own exact geometry. What goes with the STL — the STL
 *             alone cannot say that P00 is needed sixteen times.
 */

import useStoreV4, { getDerived } from './store.js'
import { exportConfigJSON, exportOBJ } from '../utils/exporters.js'
import { exportConnectorPlateSTL, exportConnectorManifest } from '../utils/connectorExport.js'
import { getConnectorKit, toExportableLayout } from './exportAdapter.js'

export default function ExportButtons() {
  const config = useStoreV4((s) => s.config)
  const { chain, connectors } = getDerived(config)
  const kit = getConnectorKit(config, chain, connectors)
  const exportable = toExportableLayout(chain, kit.stations)
  // The two connector exporters read `report.connectors.*`, so they are handed a
  // v3-shaped report whose `connectors` block IS the assembled kit. Nothing in
  // them needs to know which model produced it.
  const reportShape = { connectors: kit }
  const summary = kit.summary

  return (
    <div className="export-bar">
      <button
        type="button"
        className="preset-btn"
        data-testid="export-obj"
        title={`bake ${exportable.panels.length} panels and ${summary.count} connectors into a Wavefront OBJ (one named object each)`}
        onClick={() => exportOBJ(exportable)}
      >
        Export OBJ
      </button>
      <button
        type="button"
        className="preset-btn"
        data-testid="export-connector-stl"
        title={`one of each of the ${summary.partTypes} unique connector types, in millimetres, laid out flat for printing`}
        onClick={() => exportConnectorPlateSTL(reportShape)}
      >
        Connector STL
      </button>
      <button
        type="button"
        className="preset-btn"
        data-testid="export-connector-manifest"
        title={`how many of each type to print (${summary.count} parts over ${summary.jointCount} joints) and what each one is forced to absorb`}
        onClick={() => exportConnectorManifest(config, reportShape)}
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
