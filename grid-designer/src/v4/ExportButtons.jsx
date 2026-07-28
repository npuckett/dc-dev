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
 * Five things leave this tool, and they answer different questions:
 *
 *   OBJ+MTL   the assembly — every PRESENT panel and every connector,
 *             world-baked, for looking at it somewhere else. TWO FILES: the OBJ
 *             names its materials and the MTL defines them, and an importer that
 *             splits a mesh by material needs both or it merges the lot into one
 *             surface. They download together for that reason; a zip would be a
 *             dependency to solve a problem two clicks already solve.
 *   GLB       the same assembly, one self-contained binary, one material per
 *             object with a real emissive strength. The better import for
 *             per-panel brightness — see src/v4/glbExport.js for why not FBX.
 *   JSON      the config — the single source of truth; re-import restores the
 *             design exactly.
 *   STL       one of each unique connector type, in MILLIMETRES, oriented for
 *             printing. What goes to the slicer.
 *   manifest  how many of each to run, and what each is forced to absorb by not
 *             getting its own exact geometry. What goes with the STL — the STL
 *             alone cannot say that P00 is needed sixteen times.
 */

import useStoreV4, { getDerived } from './store.js'
import { downloadBlob, downloadText, exportConfigJSON, timestamp } from '../utils/exporters.js'
import { exportConnectorPlateSTL, exportConnectorManifest } from '../utils/connectorExport.js'
import { getConnectorKit } from './exportAdapter.js'
import { objMtlPairV4 } from './objExport.js'
import { glbPayloadV4 } from './glbExport.js'

export default function ExportButtons() {
  const config = useStoreV4((s) => s.config)
  const { chain, connectors } = getDerived(config)
  const kit = getConnectorKit(config, chain, connectors)
  const panelCount = chain.panels.filter((p) => p.present).length
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
        title={
          `${panelCount} diffusers as individual objects, ready for per-panel brightness, plus ` +
          `merged groups: frame, connectors (${summary.count} parts) ` +
          'and power supplies. Downloads TWO files — keep the .mtl beside the .obj or it imports as one surface'
        }
        onClick={() => {
          // One basename for both, so the OBJ's `mtllib` names a file that is
          // actually on disk next to it. Two downloads rather than a zip.
          const pair = objMtlPairV4(config, chain, connectors, `drop-ceiling_${timestamp()}`)
          downloadText(pair.mtl, pair.mtlName, 'model/mtl')
          downloadText(pair.obj, pair.objName, 'model/obj')
        }}
      >
        Export OBJ + MTL
      </button>
      <button
        type="button"
        className="preset-btn"
        data-testid="export-glb"
        title={
          `the same assembly as one self-contained .glb — ${panelCount} named diffuser objects, ` +
          'each with its own emissive material, plus the merged frame, connector and supply groups'
        }
        onClick={async () => {
          const buffer = await glbPayloadV4(config, chain, connectors)
          downloadBlob(new Blob([buffer], { type: 'model/gltf-binary' }), `drop-ceiling_${timestamp()}.glb`)
        }}
      >
        Export GLB
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
