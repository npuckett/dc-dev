/**
 * tests/test-v3-connector-export.mjs — the printable output.
 *
 * These bytes go to a printer, so the checks are about the things that are
 * invisible on screen and expensive at the machine:
 *
 *   - MILLIMETRES. The whole project is centimetres and STL carries no unit. A
 *     factor-of-ten error slices perfectly happily and produces a part a tenth
 *     the size, which nothing downstream would catch.
 *   - HANDEDNESS. The print orientation is a rotation, not an axis swap. A swap
 *     is a reflection: every part would be mirrored, and a mirrored C-channel
 *     grips nothing. Caught by signed volume, which a screenshot cannot show.
 *   - The STL is re-parsed from its own bytes rather than compared against the
 *     structure that wrote it, so a writer bug cannot agree with itself.
 */

import * as THREE from 'three'
import {
  buildConnectorPlate,
  stlPayload,
  connectorManifest,
  partStation,
  MM_PER_CM,
  PLATE_GAP_MM,
} from '../src/utils/connectorExport.js'
import { objPayload } from '../src/utils/exporters.js'
import { toExportableLayout } from '../src/v3/exportAdapter.js'
import { buildConnectorGeometry } from '../src/geometry/connectorGeometry.js'
import { buildReport } from '../src/core/v3/report.js'
import { solveLayout } from '../src/core/v3/placement.js'
import { buildPreset, PRESET_IDS } from '../src/core/v3/presets.js'

let passed = 0
let failed = 0
const ok = (c, m) => { if (c) passed++; else { failed++; console.log(`  FAIL: ${m}`) } }
const near = (a, b, tol, m) => ok(Math.abs(a - b) <= tol, `${m} (got ${a}, want ${b} ±${tol})`)

/** Independent binary-STL reader — deliberately not sharing code with the writer. */
function readSTL(buf) {
  const view = new DataView(buf)
  const count = view.getUint32(80, true)
  const tris = []
  for (let t = 0; t < count; t++) {
    const o = 84 + t * 50
    const normal = [view.getFloat32(o, true), view.getFloat32(o + 4, true), view.getFloat32(o + 8, true)]
    const v = []
    for (let k = 0; k < 3; k++) {
      v.push([
        view.getFloat32(o + 12 + k * 12, true),
        view.getFloat32(o + 16 + k * 12, true),
        view.getFloat32(o + 20 + k * 12, true),
      ])
    }
    tris.push({ normal, v })
  }
  return { count, tris, byteLength: buf.byteLength }
}

const signedVolume = (tris) => {
  let vol = 0
  for (const { v } of tris) {
    const a = new THREE.Vector3(...v[0])
    const b = new THREE.Vector3(...v[1])
    const c = new THREE.Vector3(...v[2])
    vol += a.dot(new THREE.Vector3().crossVectors(b, c)) / 6
  }
  return vol
}

console.log('=== test-v3-connector-export ===')

const report = buildReport(buildPreset('drift'))
const kit = report.connectors.kit

// -----------------------------------------------------------------------------
// 1. The plate carries one of each type, seated on the bed, not overlapping.
// -----------------------------------------------------------------------------
console.log('1. plate layout')
{
  const plate = buildConnectorPlate(kit)
  ok(plate.parts.length === kit.length, `one instance per unique type (${kit.length})`)
  ok(plate.indices.length % 3 === 0, 'the index buffer is whole triangles')
  ok(plate.positions.length / 3 > 0, 'the plate has vertices')

  // Everything sits on or above Z = 0 — nothing sunk into the bed.
  let belowBed = 0
  let minZ = Infinity
  for (let i = 2; i < plate.positions.length; i += 3) {
    if (plate.positions[i] < -1e-4) belowBed++
    minZ = Math.min(minZ, plate.positions[i])
  }
  ok(belowBed === 0, 'no vertex sits below the bed')
  near(minZ, 0, 1e-4, 'the plate touches the bed')

  // Parts butt in a row with exactly PLATE_GAP_MM between them, and so cannot
  // overlap: each part's X interval starts where the last one ended plus the gap.
  let overlaps = 0
  let wrongGap = 0
  for (let k = 1; k < plate.parts.length; k++) {
    const prev = plate.parts[k - 1]
    const cur = plate.parts[k]
    const prevEnd = prev.originMm[0] + prev.sizeMm[0]
    if (cur.originMm[0] < prevEnd - 1e-6) overlaps++
    if (Math.abs(cur.originMm[0] - prevEnd - PLATE_GAP_MM) > 1e-3) wrongGap++
  }
  ok(overlaps === 0, 'no two parts on the plate overlap')
  ok(wrongGap === 0, `parts are spaced exactly ${PLATE_GAP_MM}mm apart`)
  ok(plate.parts.every((p) => p.sizeMm.every((s) => s > 0)), 'every part has positive extent')
}

// -----------------------------------------------------------------------------
// 2. MILLIMETRES, and the orientation is a rotation rather than a reflection.
// -----------------------------------------------------------------------------
console.log('2. units and handedness')
{
  // A single-part plate, so the numbers can be traced back to one known part.
  const one = kit[0]
  const plate = buildConnectorPlate([one])
  const sizes = plate.parts[0].sizeMm

  // The part's own bounding box in cm, straight off the geometry.
  const geo = buildConnectorGeometry(partStation(one))
  geo.computeBoundingBox()
  const bb = geo.boundingBox
  const cmSize = [bb.max.x - bb.min.x, bb.max.y - bb.min.y, bb.max.z - bb.min.z] // p, q, r

  // (p, q, r) → (X, −Z, Y): plate X is the cross-section width, plate Y is the
  // part's length along the joint, plate Z is the cross-section height.
  near(sizes[0], cmSize[0] * MM_PER_CM, 1e-3, 'plate X is the cross-section width, in mm')
  near(sizes[1], cmSize[2] * MM_PER_CM, 1e-3, 'plate Y is the part length, in mm')
  near(sizes[2], cmSize[1] * MM_PER_CM, 1e-3, 'plate Z is the cross-section height, in mm')

  // The length is the knob's value, in mm — the single clearest unit check.
  near(sizes[1], one.lengthCm * MM_PER_CM, 1e-3,
    `a ${one.lengthCm}cm part is ${one.lengthCm * MM_PER_CM}mm long on the plate`)
  ok(sizes[1] > 50, 'and it is emphatically not centimetres (would be ~10)')

  // Handedness: a reflection flips the sign of the volume. This is the check
  // that a bare axis swap would fail while looking perfectly correct on screen.
  const vol = signedVolume(readSTL(stlPayload(plate)).tris)
  ok(vol > 0, `the printed part is not mirrored (signed volume ${vol.toFixed(1)}mm³)`)

  // And its magnitude is the geometry's own volume scaled by 10³.
  const pos = geo.getAttribute('position')
  const idx = geo.getIndex()
  let cmVol = 0
  for (let t = 0; t < idx.count; t += 3) {
    const p = [0, 1, 2].map((e) => {
      const k = idx.array[t + e] * 3
      return new THREE.Vector3(pos.array[k], pos.array[k + 1], pos.array[k + 2])
    })
    cmVol += p[0].dot(new THREE.Vector3().crossVectors(p[1], p[2])) / 6
  }
  near(vol, cmVol * MM_PER_CM ** 3, Math.abs(cmVol) * 1e-2, 'volume scales by exactly 10³')
  geo.dispose()
}

// -----------------------------------------------------------------------------
// 3. The STL round-trips through its own bytes.
// -----------------------------------------------------------------------------
console.log('3. binary STL')
{
  const plate = buildConnectorPlate(kit)
  const buf = stlPayload(plate)
  const parsed = readSTL(buf)

  ok(parsed.count === plate.indices.length / 3, 'the header triangle count matches the payload')
  ok(parsed.byteLength === 84 + parsed.count * 50, 'the file is exactly the binary-STL size')
  ok(parsed.tris.every((t) => t.v.every((v) => v.every(Number.isFinite))), 'every vertex is finite')

  // Facet normals must agree with the winding — some slicers use them to decide
  // inside from outside, so writing zeros (or the wrong sign) is a real defect.
  let badNormal = 0
  for (const { normal, v } of parsed.tris) {
    const a = new THREE.Vector3(...v[0])
    const ab = new THREE.Vector3(...v[1]).sub(a)
    const ac = new THREE.Vector3(...v[2]).sub(a)
    const n = new THREE.Vector3().crossVectors(ab, ac)
    if (n.lengthSq() < 1e-12) continue          // degenerate triangle, normal is 0
    n.normalize()
    if (n.dot(new THREE.Vector3(...normal)) < 0.99) badNormal++
  }
  ok(badNormal === 0, 'every facet normal agrees with its winding')

  // Deterministic.
  const again = stlPayload(buildConnectorPlate(kit))
  ok(Buffer.compare(Buffer.from(buf), Buffer.from(again)) === 0, 'the same kit writes byte-identical STL')
}

// -----------------------------------------------------------------------------
// 4. The manifest is the document the STL cannot be.
// -----------------------------------------------------------------------------
console.log('4. manifest')
{
  const cfg = buildPreset('drift')
  const plate = buildConnectorPlate(kit)
  const m = connectorManifest(cfg, report, plate.parts)

  ok(m.units === 'millimetres', 'the manifest states its units')
  ok(m.parts.length === kit.length, 'one row per unique type')
  ok(m.totals.uniqueTypes === kit.length, 'totals agree with the rows')

  // The count the STL cannot carry: how many of each to actually run.
  const printed = m.parts.reduce((n, p) => n + p.quantity, 0)
  ok(printed === report.connectors.summary.count,
    `quantities sum to every part in the design (${printed})`)
  ok(printed > kit.length, 'and that is more than one of each — which is the point of the manifest')

  // Every part is locatable on the plate, or the manifest cannot say which is which.
  ok(m.parts.every((p) => Array.isArray(p.plateOriginMm) && Array.isArray(p.plateSizeMm)),
    'every part carries its position on the plate')

  // mm throughout, cross-checked against the cm source.
  near(m.parts[0].lengthMm, kit[0].lengthCm * MM_PER_CM, 1e-6, 'part length is in mm')
  near(m.grip.gripDepthMm, 8.5, 1e-6, "the grip depth is v1's 8.5mm, in mm")
  ok(m.grip.v1Reference.fixedDihedralDeg === 62, "v1's fixed angle is recorded for comparison")

  // Every joint in the design is served by exactly one part type.
  const joints = new Set()
  for (const p of m.parts) for (const j of p.joints) joints.add(j)
  ok(joints.size === report.connectors.summary.jointCount, 'every joint is covered by the kit')

  ok(JSON.stringify(connectorManifest(cfg, report, plate.parts).parts) === JSON.stringify(m.parts),
    'the manifest body is deterministic (only `generated` is a timestamp)')
}

// -----------------------------------------------------------------------------
// 5. The assembly OBJ gains the connectors — and is unchanged without them.
// -----------------------------------------------------------------------------
console.log('5. assembly OBJ')
{
  const cfg = buildPreset('drift')
  const layout = solveLayout(cfg)

  const without = objPayload(toExportableLayout(layout))
  const withConn = objPayload(toExportableLayout(layout, report.connectors))

  const names = (obj) => obj.split('\n').filter((l) => l.startsWith('o ')).map((l) => l.slice(2))
  const panelNames = names(without)
  const allNames = names(withConn)

  ok(panelNames.every((n) => !n.startsWith('connector_')), 'without connectors, none appear')
  ok(allNames.length === panelNames.length + report.connectors.summary.count,
    `with connectors, ${report.connectors.summary.count} more objects appear`)
  ok(allNames.filter((n) => n.startsWith('connector_')).length === report.connectors.summary.count,
    'each connector is its own named object')
  ok(new Set(allNames).size === allNames.length, 'every object name is unique')

  // Omitting connectors leaves the old output byte-identical, so every existing
  // caller and the P5-era OBJ test are unaffected.
  ok(without === objPayload(toExportableLayout(layout)), 'the panel-only OBJ is unchanged and deterministic')

  // The connectors land where the report puts them: an exported connector's
  // vertices must bracket its station midpoint.
  const lines = withConn.split('\n')
  const st = report.connectors.stations[0]
  const start = lines.findIndex((l) => l === `o connector_${st.id}`)
  ok(start >= 0, `connector_${st.id} is present in the OBJ`)
  const verts = []
  for (let k = start + 1; k < lines.length && !lines[k].startsWith('o '); k++) {
    if (lines[k].startsWith('v ')) verts.push(lines[k].slice(2).trim().split(/\s+/).map(Number))
  }
  ok(verts.length > 0, 'and it has vertices')
  for (let axis = 0; axis < 3; axis++) {
    const lo = Math.min(...verts.map((v) => v[axis]))
    const hi = Math.max(...verts.map((v) => v[axis]))
    ok(st.mid[axis] >= lo - 1e-3 && st.mid[axis] <= hi + 1e-3,
      `axis ${axis}: the exported part brackets its station's midpoint`)
  }
}

// -----------------------------------------------------------------------------
// 6. Every preset exports.
// -----------------------------------------------------------------------------
console.log('6. preset sweep')
{
  for (const id of PRESET_IDS) {
    const cfg = buildPreset(id)
    const R = buildReport(cfg)
    const plate = buildConnectorPlate(R.connectors.kit)
    const buf = stlPayload(plate)
    const parsed = readSTL(buf)
    ok(parsed.count > 0, `${id}: the plate has triangles (${parsed.count})`)
    ok(signedVolume(parsed.tris) > 0, `${id}: the plate is not mirrored`)
    const m = connectorManifest(cfg, R, plate.parts)
    ok(m.parts.reduce((n, p) => n + p.quantity, 0) === R.connectors.summary.count,
      `${id}: manifest quantities are complete`)
  }
}

console.log(`\ntest-v3-connector-export: ${passed} checks passed, ${failed} failed`)
process.exit(failed ? 1 : 0)
