/**
 * grid-designer — file exporters (OBJ mesh + config JSON).
 *
 * Adapted from panel-designer/src/utils/exporters.js — do not edit the original.
 * Carried over: the `downloadBlob` / `downloadText` / `timestamp` helpers and the
 * OBJ "bake the world matrix into a cloned geometry" pattern from its
 * `exportMesh(..., individualObjects: true)` branch, which is what makes OBJ
 * emit one correctly-placed named object per panel (OBJExporter flattens a
 * transform hierarchy otherwise).
 *
 * Deliberately NOT ported: `exportPanelLayoutJSON`. It computes a panel's world
 * normal as `[0, 1, 0].applyQuaternion(q)` on a plain Array, which has no such
 * method — the call throws. grid-designer exports the config instead: it is the
 * single source of truth, round-trips through `normalizeConfig`/`validateConfig`,
 * and re-solves to a byte-identical layout.
 *
 * `buildExportGroup` is deliberately DOM-free so tests/test-obj.mjs can run
 * OBJExporter over it in plain node; only the `export*` wrappers touch
 * `document` / `URL.createObjectURL`. All relative imports carry explicit `.js`
 * extensions for the same reason.
 */

import * as THREE from 'three'
import { OBJExporter } from 'three/examples/jsm/exporters/OBJExporter.js'
import { buildPanelGeometry } from '../geometry/panelGeometry.js'

// -----------------------------------------------------------------------------
// Download helpers (browser only)
// -----------------------------------------------------------------------------
/**
 * Trigger a browser download of a Blob.
 *
 * @param {Blob} blob
 * @param {string} filename
 */
export function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

/**
 * Trigger a browser download of a text payload.
 *
 * @param {string} text
 * @param {string} filename
 * @param {string} [mime]
 */
export function downloadText(text, filename, mime = 'application/octet-stream') {
  downloadBlob(new Blob([text], { type: mime }), filename)
}

/**
 * Local-time `YYYYMMDD_HHMM` stamp for filenames.
 *
 * @returns {string}
 */
export function timestamp() {
  const d = new Date()
  const pad = (n) => String(n).padStart(2, '0')
  return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}_${pad(d.getHours())}${pad(d.getMinutes())}`
}

// -----------------------------------------------------------------------------
// Headless: layout → THREE.Group of world-baked meshes
// -----------------------------------------------------------------------------
/** One base geometry per panel type, cloned per instance before baking. */
const baseGeometries = new Map()

function baseGeometry(type) {
  if (!baseGeometries.has(type)) {
    baseGeometries.set(type, buildPanelGeometry({ type }))
  }
  return baseGeometries.get(type)
}

/**
 * Object name for a layout panel: `panel_r{row}_c{col}` for squares,
 * `rect_h_…` / `rect_v_…` for the two-cell plates. `row`/`col` are the panel's
 * ANCHOR cell (a plate's lower / nearer-the-shore cell), so names are unique.
 *
 * @param {object} panel a layout panel
 * @returns {string}
 */
export function exportPanelName(panel) {
  const kind =
    panel.rectOrientation === 'horizontal'
      ? 'rect_h'
      : panel.rectOrientation === 'vertical'
        ? 'rect_v'
        : 'panel'
  return `${kind}_r${panel.row}_c${panel.col}`
}

/**
 * Build an export-ready group from a solved layout. Pure and DOM-free.
 *
 * Each panel gets a CLONE of its type's geometry with the panel's world matrix
 * baked in (`applyMatrix4`), and sits at the identity transform. That is what
 * makes each object independently correct in formats that ignore or flatten
 * hierarchy — and it is why the exported vertices can be compared directly
 * against `placement.js`'s `panelWorldCorners`.
 *
 * The panel's stored quaternion is rounded to 1e-9 for determinism, so it is
 * normalized before composing (an un-normalized quaternion scales the baked
 * geometry by |q|²) — the same correction `placement.js` applies on read.
 *
 * @param {{panels: Array}} layout output of core/placement.js solveLayout
 * @returns {THREE.Group}
 */
export function buildExportGroup(layout) {
  const group = new THREE.Group()
  group.name = 'grid_surface'

  const position = new THREE.Vector3()
  const quaternion = new THREE.Quaternion()
  const scale = new THREE.Vector3(1, 1, 1)

  for (const panel of layout.panels) {
    const geometry = baseGeometry(panel.type).clone()
    const matrix = new THREE.Matrix4().compose(
      position.fromArray(panel.position),
      quaternion.fromArray(panel.quaternion).normalize(),
      scale,
    )
    geometry.applyMatrix4(matrix)
    geometry.computeBoundingBox()
    geometry.computeBoundingSphere()

    const mesh = new THREE.Mesh(geometry)
    mesh.name = exportPanelName(panel)
    mesh.updateMatrixWorld(true)
    group.add(mesh)
  }

  group.updateMatrixWorld(true)
  return group
}

/**
 * Serialize a layout to OBJ text (no download). Headless.
 *
 * @param {{panels: Array}} layout
 * @returns {string} OBJ text, one `o <name>` block per panel
 */
export function objPayload(layout) {
  return new OBJExporter().parse(buildExportGroup(layout))
}

/**
 * Serialize a config to the exact text `exportConfigJSON` downloads. Headless,
 * so the round-trip (parse → validate → solve) is testable without a browser.
 *
 * @param {object} config
 * @returns {string} pretty-printed JSON, trailing newline
 */
export function configJSONPayload(config) {
  return `${JSON.stringify(config, null, 2)}\n`
}

// -----------------------------------------------------------------------------
// Browser: downloads
// -----------------------------------------------------------------------------
/**
 * Download the solved surface as a Wavefront OBJ.
 *
 * @param {{panels: Array}} layout
 * @param {string} [filename]
 * @returns {string} the filename used
 */
export function exportOBJ(layout, filename = `grid-design_${timestamp()}.obj`) {
  downloadText(objPayload(layout), filename, 'model/obj')
  return filename
}

/**
 * Download the current config as pretty-printed JSON.
 *
 * @param {object} config
 * @param {string} [filename]
 * @returns {string} the filename used
 */
export function exportConfigJSON(config, filename = `grid-design_${timestamp()}.json`) {
  downloadText(configJSONPayload(config), filename, 'application/json')
  return filename
}
