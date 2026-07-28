/**
 * tests/test-v4-obj.mjs — headless checks for src/v4/objExport.js.
 *
 * Plain node script, no test framework. Exits non-zero on any failure.
 *   node tests/test-v4-obj.mjs
 *
 * The export exists so a renderer can be given ONE object per diffuser (each
 * gets its own brightness) and one object each for the frame, the connectors and
 * the power supplies. So the subject here is the GROUPING and the SPLIT, not
 * three.js's OBJ serializer:
 *
 *   §1 the object list is exactly that shape, and it tracks the design;
 *   §2 the diffuser/frame cut is LOSSLESS — every triangle of the panel solid
 *      lands in exactly one of the two, which is the way a geometry split
 *      silently goes wrong;
 *   §3 the world transform survives the split, checked against the panel's own
 *      position rather than against the exporter's other output.
 */

import * as THREE from 'three'
import {
  buildSceneGroup,
  objPayloadV4,
  objObjectNames,
  GROUP_FRAME,
  GROUP_CONNECTORS,
  GROUP_SUPPLIES,
  DIFFUSER_PREFIX,
} from '../src/v4/objExport.js'
import { solveLattice } from '../src/core/v4/lattice.js'
import { solveConnectorsV4 } from '../src/core/v4/connectors.js'
import { normalizeConfig, DEFAULT_CONFIG } from '../src/core/v4/schema.js'
import { buildPanelGeometry, buildPowerSupplyGeometry } from '../src/geometry/panelGeometry.js'

let passed = 0
let failed = 0
const ok = (c, m) => { if (c) passed++; else { failed++; console.log(`  FAIL: ${m}`) } }
const near = (a, b, tol, m) => ok(Math.abs(a - b) <= tol, `${m} (got ${a}, want ${b} ±${tol})`)

const solve = (patch = {}) => {
  const cfg = normalizeConfig({ ...DEFAULT_CONFIG, ...patch })
  const chain = solveLattice(cfg)
  return { cfg, chain, connectors: solveConnectorsV4(cfg, chain) }
}
const tris = (mesh) => mesh.geometry.getIndex().count / 3

console.log('=== test-v4-obj ===')

// -----------------------------------------------------------------------------
// 1. THE OBJECT LIST
// -----------------------------------------------------------------------------
console.log('1. one object per diffuser, plus exactly four merged groups')
{
  const { cfg, chain, connectors } = solve()
  const present = chain.panels.filter((p) => p.present)
  const names = objObjectNames(cfg, chain, connectors)

  const diffusers = names.filter((n) => n.startsWith(`${DIFFUSER_PREFIX}_`))
  ok(diffusers.length === present.length,
    `one diffuser object per present panel (${diffusers.length} vs ${present.length})`)
  ok(names.length === present.length + 3, 'and exactly three more objects than that')
  for (const g of [GROUP_FRAME, GROUP_CONNECTORS, GROUP_SUPPLIES]) {
    ok(names.filter((n) => n === g).length === 1, `exactly one "${g}" object`)
  }
  ok(new Set(names).size === names.length, 'every object name is unique')

  // The names have to be usable: they carry the panel id, and sort in build
  // order rather than alphabetically (which interleaves cells and ramps).
  ok(diffusers.every((n, i) => n.includes(present[i].id)), 'each diffuser names its panel')
  const sorted = [...diffusers].sort()
  ok(sorted.every((n, i) => n === diffusers[i]), 'and they are already in sorted order')

  // It tracks the design, not the rectangle: switch a panel off and the object
  // goes. A test that only counted 37 would pass against a hardcoded loop.
  const cut = solve({ overrides: { cells: [{ i: 1, j: 2, present: false }], edges: [] } })
  const cutNames = objObjectNames(cut.cfg, cut.chain, cut.connectors)
  ok(cutNames.length === names.length - 1, 'removing a panel removes exactly one object')
  ok(!cutNames.some((n) => n.includes('Ci1j2')), 'and it is that panel')
  ok(cutNames.includes(GROUP_FRAME), 'the merged groups survive')

  const grown = solve({ lattice: { cols: 4, rows: 5, panelType: '2x2' } })
  ok(objObjectNames(grown.cfg, grown.chain, grown.connectors).length > names.length,
    'a bigger network exports more objects')

}

// -----------------------------------------------------------------------------
// 2. THE SPLIT IS LOSSLESS
//
// The failure mode of cutting a geometry by material group is losing or
// duplicating triangles, and it is invisible in a render. So this counts them.
// -----------------------------------------------------------------------------
console.log('2. every triangle of the panel solid lands in exactly one group')
{
  const { cfg, chain, connectors } = solve()
  const group = buildSceneGroup(cfg, chain, connectors)
  const present = chain.panels.filter((p) => p.present)

  const solid = buildPanelGeometry({ type: '2x2' })
  const solidTris = solid.getIndex().count / 3
  const diffuserGroup = solid.groups.find((g) => g.materialIndex === 0)
  const housingGroup = solid.groups.find((g) => g.materialIndex === 1)

  const diffusers = group.children.filter((c) => c.name.startsWith(`${DIFFUSER_PREFIX}_`))
  const frame = group.children.find((c) => c.name === GROUP_FRAME)

  ok(diffusers.every((m) => tris(m) === diffuserGroup.count / 3),
    `every diffuser carries the solid's diffuser group (${diffuserGroup.count / 3} triangles)`)
  ok(tris(frame) === (housingGroup.count / 3) * present.length,
    `the frame carries every panel's housing group (${tris(frame)} triangles)`)
  near(tris(diffusers[0]) + tris(frame) / present.length, solidTris, 0,
    'and diffuser + frame per panel is exactly the whole panel — nothing lost, nothing doubled')

  // Supplies: one box per panel, and a box is 12 triangles.
  const supplies = group.children.find((c) => c.name === GROUP_SUPPLIES)
  const oneBox = buildPowerSupplyGeometry({ type: '2x2', edge: 0 })
  const boxTris = (oneBox.getIndex()?.count ?? oneBox.getAttribute('position').count) / 3
  near(tris(supplies), boxTris * present.length, 0, 'one power supply per panel')

  // 'none' means the connectors were solved as though there is no supply, so
  // exporting one would contradict the rest of the model.
  const off = solve({ connectors: { ...DEFAULT_CONFIG.connectors, powerEdge: 'none' } })
  ok(!objObjectNames(off.cfg, off.chain, off.connectors).includes(GROUP_SUPPLIES),
    "powerEdge 'none' exports no supply object at all")

  // Connectors: both printed pieces per station, merged into one object.
  const parts = group.children.find((c) => c.name === GROUP_CONNECTORS)
  ok(tris(parts) > 0, 'the connectors object is not empty')
  const fewer = solve({ connectors: { ...DEFAULT_CONFIG.connectors, minPerJoint: 1, spacingCm: 60 } })
  const fewerGroup = buildSceneGroup(fewer.cfg, fewer.chain, fewer.connectors)
  ok(tris(fewerGroup.children.find((c) => c.name === GROUP_CONNECTORS)) < tris(parts),
    'and it shrinks when the design asks for fewer parts')
}

// -----------------------------------------------------------------------------
// 3. THE WORLD TRANSFORM SURVIVES THE SPLIT
// -----------------------------------------------------------------------------
console.log('3. the split geometry is still where the panel is')
{
  const { cfg, chain, connectors } = solve()
  const group = buildSceneGroup(cfg, chain, connectors)
  const present = chain.panels.filter((p) => p.present)

  let worst = 0
  for (let n = 0; n < present.length; n++) {
    const panel = present[n]
    const mesh = group.children.find((c) => c.name.includes(panel.id))
    mesh.geometry.computeBoundingBox()
    const bb = mesh.geometry.boundingBox
    const centre = new THREE.Vector3().addVectors(bb.min, bb.max).multiplyScalar(0.5)
    // The diffuser is RECESSED below the panel's front plane by diffuserDepth,
    // along the panel's own normal — so the offset is not zero, and checking for
    // zero would be checking the wrong thing.
    const expected = new THREE.Vector3().fromArray(panel.position)
      .addScaledVector(new THREE.Vector3().fromArray(panel.normal), -0.363)
    worst = Math.max(worst, centre.distanceTo(expected))
  }
  near(worst, 0, 1e-4,
    'every diffuser sits on its panel, recessed by exactly the measured diffuserDepth')

  // A flipped panel's diffuser must follow it to the other side.
  const flip = solve({ overrides: { cells: [{ i: 0, j: 0, flipped: true }], edges: [] } })
  const flipMesh = buildSceneGroup(flip.cfg, flip.chain, flip.connectors)
    .children.find((c) => c.name.includes('Ci0j0'))
  flipMesh.geometry.computeBoundingBox()
  const plainMesh = group.children.find((c) => c.name.includes('Ci0j0'))
  plainMesh.geometry.computeBoundingBox()
  ok(flipMesh.geometry.boundingBox.min.y !== plainMesh.geometry.boundingBox.min.y,
    'a flipped panel exports its diffuser on the other face')
}

// -----------------------------------------------------------------------------
// 4. THE FILE ITSELF
// -----------------------------------------------------------------------------
console.log('4. the OBJ text carries the grouping')
{
  const { cfg, chain, connectors } = solve()
  const text = objPayloadV4(cfg, chain, connectors)
  const names = objObjectNames(cfg, chain, connectors)

  const objLines = text.match(/^o .*/gm) ?? []
  ok(objLines.length === names.length, `one "o" block per object (${objLines.length})`)
  ok(objLines.map((l) => l.slice(2).trim()).every((n, i) => n === names[i]),
    'in the same order, with the same names')
  // Named materials are what make the grouping survive an import.
  ok((text.match(/^usemtl /gm) ?? []).length === names.length, 'and every object carries a usemtl')
  ok(text.includes(`usemtl ${GROUP_FRAME}`), 'the frame names its material')
  ok(!text.includes('NaN'), 'no NaN anywhere in the file')

  // Deterministic: the same design exports byte-identically.
  ok(text === objPayloadV4(cfg, chain, connectors), 'two exports of one design are identical')
}

console.log(`\ntest-v4-obj: ${passed} checks passed, ${failed} failed`)
process.exit(failed ? 1 : 0)
