/**
 * tests/test-geometry.mjs — the panel solid.
 *
 * Plain node script, no test framework, NO browser. Exits non-zero on failure.
 *   node tests/test-geometry.mjs
 *
 * =============================================================================
 * WHY THIS SUITE EXISTS
 * =============================================================================
 * The panel solid inherited from panel-designer had its front and back CAPS
 * wound inside-out: under FrontSide materials every panel was missing its lit
 * face AND its housing back, and nothing noticed, because the vertex COUNT was
 * right and the OBJ still round-tripped. So the checks here are about ORIENTATION
 * and CLOSURE rather than counts.
 *
 * A second failure mode now matters as much. The section was replaced wholesale
 * when the panels were re-measured (`updatedPanelGeo/`), and the old one had no
 * back flange and the wrong thickness. So §3 does not check remembered numbers —
 * it derives every expectation from `panelSectionRings()`, including a
 * closed-form volume summed frustum by frustum. That is what makes the profile
 * genuinely parametric: refine `PANEL_PROFILE` and the mesh and this test move
 * together, and neither can quietly agree with a stale constant.
 *
 *   1. CLOSED 2-MANIFOLD — every undirected edge borders exactly two triangles.
 *   2. OUTWARD ORIENTATION — every DIRECTED edge appears exactly once, and the
 *      divergence-theorem signed volume is positive.
 *   3. THE SECTION IS THE ONE IN config.js — volume and every ring's footprint.
 *   4. THE POWER SUPPLY sits where config.js says, on the edge asked for.
 */

import * as THREE from 'three'
import {
  buildPanelGeometry,
  buildPowerSupplyGeometry,
  DIFFUSER_MATERIAL_INDEX,
  HOUSING_MATERIAL_INDEX,
} from '../src/geometry/panelGeometry.js'
import {
  PANEL_DIMENSIONS,
  PANEL_PROFILE,
  PANEL_METRICS,
  POWER_SUPPLY,
  panelSectionRings,
} from '../src/config.js'

let passed = 0
let failed = 0
const ok = (c, m) => { if (c) passed++; else { failed++; console.log(`  FAIL  ${m}`) } }
const near = (a, b, tol, m) => ok(Math.abs(a - b) <= tol, `${m} (got ${a}, want ${b} ±${tol})`)

console.log('=== test-geometry ===')

/**
 * Closed-form volume of the swept section, summed frustum by frustum.
 *
 * Between two rings the horizontal cross-section is a rectangle whose inset
 * varies linearly, so its area is a quadratic in the sweep parameter and
 * integrates exactly. Segments where the depth DECREASES (the diffuser recess)
 * contribute negative volume, which is correct — that is what the recess removes.
 */
function sectionVolume(W, H, rings = panelSectionRings()) {
  let v = 0
  for (let i = 0; i < rings.length - 1; i++) {
    const a0 = rings[i].inset
    const da = rings[i + 1].inset - a0
    const P0 = W - 2 * a0
    const Q0 = H - 2 * a0
    const areaIntegral = P0 * Q0 - da * (P0 + Q0) + (4 * da * da) / 3
    v += areaIntegral * (rings[i + 1].depth - rings[i].depth)
  }
  return v
}

const meshVolume = (g) => {
  const pos = g.getAttribute('position')
  const idx = g.getIndex()
  let v = 0
  for (let t = 0; t < idx.count; t += 3) {
    const p = [0, 1, 2].map((e) => {
      const k = idx.array[t + e] * 3
      return new THREE.Vector3(pos.array[k], pos.array[k + 1], pos.array[k + 2])
    })
    v += p[0].dot(new THREE.Vector3().crossVectors(p[1], p[2])) / 6
  }
  return v
}

// -----------------------------------------------------------------------------
// 1–3. Every panel type.
// -----------------------------------------------------------------------------
for (const type of Object.keys(PANEL_DIMENSIONS)) {
  const tag = `[${type}]`
  const { width: W, height: H } = PANEL_DIMENSIONS[type]
  const g = buildPanelGeometry({ type })
  const pos = g.getAttribute('position')
  const idx = g.getIndex()

  const key = (i) => {
    const k = i * 3
    return `${pos.array[k].toFixed(6)},${pos.array[k + 1].toFixed(6)},${pos.array[k + 2].toFixed(6)}`
  }
  const dir = new Map()
  const und = new Map()
  for (let t = 0; t < idx.count; t += 3) {
    const v = [key(idx.array[t]), key(idx.array[t + 1]), key(idx.array[t + 2])]
    for (let e = 0; e < 3; e++) {
      const a = v[e]
      const b = v[(e + 1) % 3]
      dir.set(`${a}|${b}`, (dir.get(`${a}|${b}`) ?? 0) + 1)
      const u = a < b ? `${a}|${b}` : `${b}|${a}`
      und.set(u, (und.get(u) ?? 0) + 1)
    }
  }
  ok([...und.values()].every((c) => c === 2), `${tag} closed 2-manifold — every edge borders 2 triangles`)
  ok([...dir.values()].every((c) => c === 1), `${tag} consistently wound — no directed edge appears twice`)

  const vol = meshVolume(g)
  const expected = sectionVolume(W, H)
  ok(vol > 0, `${tag} signed volume is POSITIVE — the shell faces outward`)
  near(vol, expected, Math.abs(expected) * 1e-5, `${tag} volume matches the section summed frustum by frustum`)

  g.computeBoundingBox()
  const bb = g.boundingBox
  near(bb.max.x - bb.min.x, W, 1e-4, `${tag} spans the full width`)
  near(bb.max.z - bb.min.z, H, 1e-4, `${tag} spans the full height`)
  near(bb.max.y, 0, 1e-9, `${tag} the front-most surface is the plane y = 0`)
  // 1e-5, not 1e-9: positions live in a Float32BufferAttribute, so 4.1 is only
  // representable to about seven figures. A tighter bound would be testing
  // float32, not the geometry.
  near(bb.min.y, -PANEL_PROFILE.overallThickness, 1e-5, `${tag} the housing reaches y = −overallThickness`)

  // Every ring in the section must appear in the mesh, at its own depth and on
  // its own footprint — the check that ties the solid to config.js.
  for (const ring of panelSectionRings()) {
    const y = -ring.depth
    const atDepth = []
    for (let i = 0; i < pos.count; i++) {
      if (Math.abs(pos.getY(i) - y) < 1e-6) atDepth.push([pos.getX(i), pos.getZ(i)])
    }
    ok(atDepth.length > 0, `${tag} ring "${ring.name}" appears at depth ${ring.depth}`)
    near(Math.max(...atDepth.map(([x]) => Math.abs(x))), W / 2 - ring.inset, 1e-4,
      `${tag} ring "${ring.name}" is inset ${ring.inset} in width`)
    near(Math.max(...atDepth.map(([, z]) => Math.abs(z))), H / 2 - ring.inset, 1e-4,
      `${tag} ring "${ring.name}" is inset ${ring.inset} in height`)
  }

  const nrm = g.getAttribute('normal')
  let bad = 0
  for (let i = 0; i < nrm.count; i++) {
    const n = new THREE.Vector3(nrm.getX(i), nrm.getY(i), nrm.getZ(i))
    if (!Number.isFinite(n.length()) || Math.abs(n.length() - 1) > 1e-3) bad++
  }
  ok(bad === 0, `${tag} every vertex normal is finite and unit`)

  ok(g.groups.length === 2, `${tag} two material groups`)
  ok(g.groups[0].materialIndex === DIFFUSER_MATERIAL_INDEX, `${tag} group 0 is the diffuser`)
  ok(g.groups[1].materialIndex === HOUSING_MATERIAL_INDEX, `${tag} group 1 is the housing`)
  ok(g.groups[0].start === 0 && g.groups[0].count + g.groups[1].count === idx.count,
    `${tag} the groups partition the index buffer with no gap or overlap`)

  ok(g.groups[0].count / 3 === 2, `${tag} the diffuser is 2 triangles`)
  let diffuserUp = true
  for (let t = 0; t < g.groups[0].count; t += 3) {
    const p = [0, 1, 2].map((e) => {
      const k = idx.array[t + e] * 3
      return new THREE.Vector3(pos.array[k], pos.array[k + 1], pos.array[k + 2])
    })
    const n = new THREE.Vector3()
      .crossVectors(p[1].clone().sub(p[0]), p[2].clone().sub(p[0])).normalize()
    if (Math.abs(n.y - 1) > 1e-6) diffuserUp = false
    for (const q of p) if (Math.abs(q.y + PANEL_PROFILE.diffuserDepth) > 1e-6) diffuserUp = false
  }
  ok(diffuserUp, `${tag} the diffuser faces +Y at the diffuser depth`)
}

// -----------------------------------------------------------------------------
// 4. The derived metrics really are derived, and the profile really is a knob.
// -----------------------------------------------------------------------------
near(PANEL_METRICS.outerWallHeight, PANEL_PROFILE.outerWallDepth - PANEL_PROFILE.bezelDrop, 1e-12,
  'outerWallHeight is derived from the profile')
near(PANEL_METRICS.bodyInset, PANEL_PROFILE.flangeWidth + PANEL_PROFILE.taperWidth, 1e-12,
  'bodyInset is derived from the profile')
ok(PANEL_METRICS.flangeInnerDepth > PANEL_PROFILE.outerWallDepth,
  'the flange falls away from the outer wall rather than rising')
ok(PANEL_PROFILE.flangeWidth > 0, 'there IS a back flange — the feature a connector grips')
{
  const base = meshVolume(buildPanelGeometry({ type: '2x2' }))
  const thicker = { ...PANEL_PROFILE, overallThickness: PANEL_PROFILE.overallThickness + 1 }
  const g = buildPanelGeometry({ type: '2x2', profile: thicker })
  g.computeBoundingBox()
  near(g.boundingBox.min.y, -thicker.overallThickness, 1e-5, 'overriding the profile moves the solid')
  const wider = { ...PANEL_PROFILE, flangeWidth: PANEL_PROFILE.flangeWidth + 1 }
  ok(Math.abs(meshVolume(buildPanelGeometry({ type: '2x2', profile: wider })) - base) > 1,
    'widening the flange changes the solid')
}

// -----------------------------------------------------------------------------
// 5. The power supply.
// -----------------------------------------------------------------------------
{
  const g = buildPowerSupplyGeometry({ type: '2x2', edge: 0 })
  g.computeBoundingBox()
  const b = g.boundingBox
  near(b.max.x - b.min.x, POWER_SUPPLY.length, 1e-6, 'power supply runs its full length along the edge')
  near(b.max.z - b.min.z, POWER_SUPPLY.depth, 1e-6, 'power supply is as deep as specified')
  near(b.max.y - b.min.y, POWER_SUPPLY.height, 1e-6, 'power supply is as tall as specified')
  near(b.max.y, -PANEL_PROFILE.outerWallDepth, 1e-6, 'it hangs from the flange plane')
  near(b.min.z, -30 + POWER_SUPPLY.edgeInset, 1e-6, 'edge 0 places it inboard of the −Z edge')
  ok(b.min.y > -PANEL_PROFILE.overallThickness, 'it stays within the panel envelope')

  // It covers the flange, which is the whole reason it constrains connectors.
  ok(POWER_SUPPLY.edgeInset + POWER_SUPPLY.depth > PANEL_PROFILE.flangeWidth,
    'the supply reaches past the flange — so it covers the connector grip entirely')

  const centres = [0, 1, 2, 3].map((edge) => {
    const gg = buildPowerSupplyGeometry({ type: '2x2', edge })
    gg.computeBoundingBox()
    return gg.boundingBox.getCenter(new THREE.Vector3())
  })
  ok(centres[0].z < -20 && centres[2].z > 20, 'edges 0 and 2 are the −Z and +Z sides')
  ok(centres[3].x < -20 && centres[1].x > 20, 'edges 3 and 1 are the −X and +X sides')
  let threw = false
  try { buildPowerSupplyGeometry({ type: '2x2', edge: 9 }) } catch { threw = true }
  ok(threw, 'an out-of-range edge is refused rather than silently placed')

  const plate = buildPowerSupplyGeometry({ type: '2x4', edge: 0 })
  plate.computeBoundingBox()
  near(plate.boundingBox.max.x - plate.boundingBox.min.x, POWER_SUPPLY.length, 1e-6,
    'on a plate the supply still runs along a 60cm edge')
}

console.log(failed === 0 ? `PASS — ${passed}/${passed + failed} checks` : `FAIL — ${failed} of ${passed + failed}`)
process.exit(failed ? 1 : 0)
