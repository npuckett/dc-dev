/**
 * grid-designer — the printable output: a plate of connector parts, and the
 * manifest that says how many of each to run.
 *
 * Kept separate from `exporters.js` because that file is a carried-over v2
 * module the P5 work order froze; this is new v3 surface and there is no reason
 * to entangle them. Same conventions though: everything except the `export*`
 * wrappers is DOM-free, so the whole payload is testable in plain node, and all
 * relative imports carry explicit `.js` extensions.
 *
 * =============================================================================
 * UNITS — MILLIMETRES
 * =============================================================================
 * The rest of this project is centimetres (README, "World conventions"). STL
 * carries no unit, and every slicer in existence assumes millimetres, so the
 * plate and the manifest are BOTH in mm and both say so. Getting this wrong
 * produces a part one tenth the size that still slices happily, which is the
 * worst kind of error to find at the printer.
 *
 * =============================================================================
 * PRINT ORIENTATION
 * =============================================================================
 * Parts are laid with the SLOT AXIS VERTICAL — the channel opens sideways, the
 * cross-section lies in the bed plane, and the part's length runs along the bed.
 *
 * That is what v1 did (`3dprintFiles/`: 94.9 × 156.7 × 28.5 mm, and 28.5 mm is
 * exactly jaw + slot + jaw), and the reasons carry over unchanged: the footprint
 * is the part's largest face so it will not tip, the slot becomes a horizontal
 * groove needing no support, and no layer boundary runs across a jaw where the
 * grip load is.
 *
 * The mapping is a proper +90° rotation about X — `(p, q, r) → (X, −Z, Y)` — not
 * an axis swap. Swapping two axes is a reflection, which would mirror every part
 * and produce a kit that grips nothing. `test-v3-connector-export.mjs` checks
 * the plate's signed volume for exactly this reason.
 *
 * =============================================================================
 * ONE OF EACH, NOT ONE OF EVERY
 * =============================================================================
 * The plate carries ONE instance of each unique part type, laid out in kit
 * order. Quantities live in the manifest. A plate with all 82 parts on it would
 * be unprintable and unreadable, and the interesting number — how few distinct
 * things this design needs — is exactly the plate's part count.
 */

import * as THREE from 'three'
import { buildConnectorGeometry } from '../geometry/connectorGeometry.js'
import { CONNECTOR_PROFILE } from '../core/v3/connectors.js'
import { PANEL_PROFILE, PANEL_METRICS } from '../config.js'
import { downloadBlob, downloadText, timestamp } from './exporters.js'

/** cm → mm. See "UNITS" above. */
export const MM_PER_CM = 10

/** Clearance between parts on the plate, mm. */
export const PLATE_GAP_MM = 8

/**
 * A representative station for a kit part — enough for the geometry builder,
 * built from the part's BIN CENTRE rather than from any one station that landed
 * in it, so what gets printed is what the manifest claims.
 */
export function partStation(part) {
  return {
    id: part.partId,
    lengthCm: part.lengthCm,
    spanStartCm: part.spanStartCm,
    spanEndCm: part.spanEndCm,
    foldDeg: part.foldDeg,
  }
}

/**
 * One instance of every unique part type, laid out in a row on Z = 0, in
 * millimetres, oriented for printing.
 *
 * @param {Array} kit `report.connectors.kit`
 * @returns {{ positions: Float32Array, indices: Uint32Array, parts: Array }}
 */
export function buildConnectorPlate(kit) {
  const positions = []
  const indices = []
  const parts = []
  let cursorX = 0
  let vertexBase = 0

  for (const part of kit) {
    const geo = buildConnectorGeometry(partStation(part))
    const pos = geo.getAttribute('position')
    const idx = geo.getIndex()

    // (p, q, r) → (X, −Z, Y): a +90° rotation about X, then scaled to mm.
    // A bare axis swap here would mirror the part — see the file header.
    let minX = Infinity
    let maxX = -Infinity
    let minY = Infinity
    let maxY = -Infinity
    let minZ = Infinity
    let maxZ = -Infinity
    const xs = new Array(pos.count)
    const ys = new Array(pos.count)
    const zs = new Array(pos.count)
    for (let v = 0; v < pos.count; v++) {
      const x = pos.getX(v) * MM_PER_CM
      const y = -pos.getZ(v) * MM_PER_CM
      const z = pos.getY(v) * MM_PER_CM
      xs[v] = x
      ys[v] = y
      zs[v] = z
      if (x < minX) minX = x
      if (x > maxX) maxX = x
      if (y < minY) minY = y
      if (y > maxY) maxY = y
      if (z < minZ) minZ = z
      if (z > maxZ) maxZ = z
    }

    // Seat it on the bed and butt it against the previous part.
    const dx = cursorX - minX
    const dy = -minY
    const dz = -minZ
    for (let v = 0; v < pos.count; v++) positions.push(xs[v] + dx, ys[v] + dy, zs[v] + dz)
    for (let t = 0; t < idx.count; t++) indices.push(idx.array[t] + vertexBase)

    parts.push({
      partId: part.partId,
      quantity: part.count,
      originMm: [Math.round((cursorX) * 1e4) / 1e4, 0, 0],
      sizeMm: [
        Math.round((maxX - minX) * 1e4) / 1e4,
        Math.round((maxY - minY) * 1e4) / 1e4,
        Math.round((maxZ - minZ) * 1e4) / 1e4,
      ],
    })

    cursorX += maxX - minX + PLATE_GAP_MM
    vertexBase += pos.count
    geo.dispose()
  }

  return { positions: new Float32Array(positions), indices: new Uint32Array(indices), parts }
}

/**
 * Binary STL. Chosen over ASCII deliberately: it is what slicers expect, it is
 * an order of magnitude smaller, and its fixed 50-byte record makes the test's
 * independent re-parse trivial rather than a text-format exercise.
 *
 * Facet normals are computed from the winding rather than written as zeros —
 * some slicers use them to decide inside from outside.
 *
 * @param {{positions: Float32Array, indices: Uint32Array}} mesh
 * @param {string} [header] up to 80 bytes, truncated
 * @returns {ArrayBuffer}
 */
export function stlPayload({ positions, indices }, header = 'grid-designer connector plate (mm)') {
  const triangles = indices.length / 3
  const buf = new ArrayBuffer(84 + triangles * 50)
  const view = new DataView(buf)
  const bytes = new Uint8Array(buf)

  const head = header.slice(0, 79)
  for (let i = 0; i < head.length; i++) bytes[i] = head.charCodeAt(i) & 0x7f
  view.setUint32(80, triangles, true)

  const a = new THREE.Vector3()
  const b = new THREE.Vector3()
  const c = new THREE.Vector3()
  const n = new THREE.Vector3()
  const ab = new THREE.Vector3()
  const ac = new THREE.Vector3()

  for (let t = 0; t < triangles; t++) {
    const o = 84 + t * 50
    const ia = indices[t * 3] * 3
    const ib = indices[t * 3 + 1] * 3
    const ic = indices[t * 3 + 2] * 3
    a.set(positions[ia], positions[ia + 1], positions[ia + 2])
    b.set(positions[ib], positions[ib + 1], positions[ib + 2])
    c.set(positions[ic], positions[ic + 1], positions[ic + 2])
    n.crossVectors(ab.subVectors(b, a), ac.subVectors(c, a))
    if (n.lengthSq() > 0) n.normalize()

    view.setFloat32(o, n.x, true)
    view.setFloat32(o + 4, n.y, true)
    view.setFloat32(o + 8, n.z, true)
    const verts = [a, b, c]
    for (let v = 0; v < 3; v++) {
      view.setFloat32(o + 12 + v * 12, verts[v].x, true)
      view.setFloat32(o + 16 + v * 12, verts[v].y, true)
      view.setFloat32(o + 20 + v * 12, verts[v].z, true)
    }
    view.setUint16(o + 48, 0, true)
  }

  return buf
}

/**
 * What to print, how many, and what each one is being asked to absorb by not
 * getting its own exact geometry. This is the document that goes with the STL.
 *
 * @param {object} config the design
 * @param {object} report `buildReport` output
 * @param {Array} plateParts `buildConnectorPlate(...).parts`
 * @returns {object}
 */
export function connectorManifest(config, report, plateParts) {
  const conn = report.connectors
  const byId = new Map(plateParts.map((p) => [p.partId, p]))
  const mm = (v) => Math.round(v * MM_PER_CM * 1e4) / 1e4

  return {
    generated: new Date().toISOString(),
    tool: 'grid-designer v3',
    units: 'millimetres',
    design: {
      name: config.name ?? null,
      preset: config.meta?.preset ?? null,
      sheet: config.sheet,
      gapCm: config.gap,
      connectors: config.connectors,
    },
    // The grip, and the panel rim it is derived from. Recorded so a part can be
    // checked against the hardware without opening the source — and so a manifest
    // from an older panel profile is identifiable as such.
    grip: {
      note:
        'two pieces bolted together: a universal front BAR bearing on both bezels, and a per-joint ' +
        'BACK HALF reaching under both flanges, pulled together by three countersunk bolts into ' +
        'heat-set inserts. Every panel-facing dimension derives from PANEL_PROFILE — see src/config.js',
      backLipMm: mm(CONNECTOR_PROFILE.backGripCm),
      frontMinLipMm: mm(CONNECTOR_PROFILE.frontMinLipCm),
      splitDepthMm: mm(CONNECTOR_PROFILE.splitDepthCm),
      boltCount: CONNECTOR_PROFILE.boltCount,
      bolt: CONNECTOR_PROFILE.bolt,
      panelRim: {
        bezelWidthMm: mm(PANEL_PROFILE.bezelWidth),
        outerWallHeightMm: mm(PANEL_METRICS.outerWallHeight),
        flangeWidthMm: mm(PANEL_PROFILE.flangeWidth),
        overallThicknessMm: mm(PANEL_PROFILE.overallThickness),
      },
      v1Reference: { slotDepthMm: 8.5, slotWidthMm: 9.5, fixedDihedralDeg: 62 },
    },
    totals: {
      partsToPrint: conn.summary.count,
      uniqueTypes: conn.summary.partTypes,
      jointsServed: conn.summary.jointCount,
      flagged: conn.summary.flagged,
      infeasible: conn.summary.infeasible,
      clashes: conn.summary.clashes,
      worstForcedSpanMm: mm(conn.summary.worstBinSpanErrorCm),
      worstForcedFoldDeg: conn.summary.worstBinFoldErrorDeg,
    },
    parts: conn.kit.map((part) => ({
      partId: part.partId,
      quantity: part.count,
      gapStartMm: mm(part.spanStartCm),
      gapEndMm: mm(part.spanEndCm),
      foldDeg: part.foldDeg,
      lengthMm: mm(part.lengthCm),
      // How far the worst joint using this part is from the part's own geometry.
      // Bounded by half a bin by construction; here so a coarser bin can be
      // chosen against a consequence rather than a feeling.
      worstForcedSpanMm: mm(part.worstSpanErrorCm),
      worstForcedFoldDeg: part.worstFoldErrorDeg,
      plateOriginMm: byId.get(part.partId)?.originMm ?? null,
      plateSizeMm: byId.get(part.partId)?.sizeMm ?? null,
      joints: part.joints,
      stations: part.stationIds,
    })),
  }
}

/** The manifest as the exact text `exportConnectorManifest` downloads. */
export function connectorManifestPayload(config, report, plateParts) {
  return `${JSON.stringify(connectorManifest(config, report, plateParts), null, 2)}\n`
}

// -----------------------------------------------------------------------------
// Browser: downloads
// -----------------------------------------------------------------------------
/** Download one of each unique connector type as a printable STL plate. */
export function exportConnectorPlateSTL(report, filename = `connectors_${timestamp()}.stl`) {
  const plate = buildConnectorPlate(report.connectors.kit)
  downloadBlob(new Blob([stlPayload(plate)], { type: 'model/stl' }), filename)
  return filename
}

/** Download the manifest that says how many of each to run. */
export function exportConnectorManifest(config, report, filename = `connectors_${timestamp()}.json`) {
  const plate = buildConnectorPlate(report.connectors.kit)
  downloadText(connectorManifestPayload(config, report, plate.parts), filename, 'application/json')
  return filename
}
