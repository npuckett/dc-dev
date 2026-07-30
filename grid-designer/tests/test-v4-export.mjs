/**
 * tests/test-v4-export.mjs — the export files, PARSED BACK.
 *
 * Plain node script, no test framework. Exits non-zero on any failure.
 *   node tests/test-v4-export.mjs
 *
 * `test-v4-obj.mjs` already pins the GROUPING — which objects exist, that the
 * geometry split is lossless, that the transforms survive. This file is about
 * the thing that grouping was useless without: whether an importer can SEE it.
 *
 * The reported bug was that the OBJ "is read as a single surface, not split into
 * parts". The `o` blocks were correct the whole time; what was missing was the
 * material library they referred to. So nothing here asserts that an exporter
 * ran. Every check reads the produced bytes back and asserts on what a reader
 * would find:
 *
 *   §1 every `usemtl` in the OBJ resolves to a `newmtl` in the MTL, and the
 *      `mtllib` names the file that is actually written. THIS IS THE BUG. A
 *      dangling `usemtl` must fail this suite forever after.
 *   §2 the OBJ's face indices are still global, cumulative and never forward —
 *      the property that makes the `o` blocks separable at all.
 *   §3 the GLB container is decoded by hand — header, chunk lengths, JSON — and
 *      the glTF is asserted on: 41 materials, not 1, and the 37 diffuser ones
 *      distinct, because a shared material is exactly "one brightness for all".
 *   §4 the two writers describe the SAME scene, so they cannot drift apart.
 */

import * as THREE from 'three'
import {
  buildSceneGroup,
  objMtlPairV4,
  objPayloadV4,
  mtlPayloadV4,
  objObjectNames,
  DEFAULT_MTL_NAME,
  GROUP_FRAME,
  GROUP_CONNECTORS,
  GROUP_SUPPLIES,
  DIFFUSER_PREFIX,
} from '../src/v4/objExport.js'
import { solveLattice } from '../src/core/v4/lattice.js'
import { solveConnectorsV4 } from '../src/core/v4/connectors.js'
import { normalizeConfig, DEFAULT_CONFIG } from '../src/core/v4/schema.js'

// GLTFExporter packs its buffer through a FileReader, which node has no reason
// to define. This is the entire browser surface it touches in binary mode (Blob
// and TextEncoder are already global in node 18+), so the shim is four lines
// rather than a DOM emulator. It exists ONLY so the container can be decoded
// here; nothing in src/ depends on it.
if (typeof globalThis.FileReader === 'undefined') {
  globalThis.FileReader = class FileReader {
    readAsArrayBuffer(blob) {
      blob.arrayBuffer().then((r) => { this.result = r; if (this.onloadend) this.onloadend() })
    }
  }
}
const { glbPayloadV4 } = await import('../src/v4/glbExport.js')

let passed = 0
let failed = 0
const ok = (c, m) => { if (c) passed++; else { failed++; console.log(`  FAIL: ${m}`) } }

const solve = (patch = {}) => {
  const cfg = normalizeConfig({ ...DEFAULT_CONFIG, ...patch })
  const chain = solveLattice(cfg)
  return { cfg, chain, connectors: solveConnectorsV4(cfg, chain) }
}

/** Decode a .glb the way a reader does: header, then the two chunks. */
function readGLB(buffer) {
  const dv = new DataView(buffer)
  const magic = dv.getUint32(0, true)
  const version = dv.getUint32(4, true)
  const declaredTotal = dv.getUint32(8, true)

  let offset = 12
  let json = null
  let binLength = null
  while (offset < buffer.byteLength) {
    const chunkLength = dv.getUint32(offset, true)
    const chunkType = dv.getUint32(offset + 4, true)
    const body = new Uint8Array(buffer, offset + 8, chunkLength)
    if (chunkType === 0x4e4f534a) json = JSON.parse(new TextDecoder().decode(body))
    if (chunkType === 0x004e4942) binLength = chunkLength
    offset += 8 + chunkLength
  }
  return { magic, version, declaredTotal, json, binLength, consumed: offset }
}

console.log('=== test-v4-export ===')

const base = solve()
const pair = objMtlPairV4(base.cfg, base.chain, base.connectors, 'drop-ceiling_TEST')
const glb = await glbPayloadV4(base.cfg, base.chain, base.connectors)
const gltf = readGLB(glb).json
const names = objObjectNames(base.cfg, base.chain, base.connectors)

// -----------------------------------------------------------------------------
// 1. THE MATERIAL LIBRARY EXISTS AND RESOLVES — the bug
// -----------------------------------------------------------------------------
console.log('1. every usemtl in the OBJ resolves to a newmtl in the MTL')
{
  const mtllib = pair.obj.match(/^mtllib (.+)$/m)
  ok(!!mtllib, 'the OBJ has a mtllib line at all')
  ok(mtllib && mtllib[1].trim() === pair.mtlName,
    `and it names the file actually written (${mtllib?.[1]} vs ${pair.mtlName})`)
  ok(pair.obj.indexOf('mtllib') < pair.obj.indexOf('\no '),
    'the mtllib comes before the first object, where a reader looks for it')

  const used = [...pair.obj.matchAll(/^usemtl (.+)$/gm)].map((m) => m[1].trim())
  const defined = new Set([...pair.mtl.matchAll(/^newmtl (.+)$/gm)].map((m) => m[1].trim()))
  ok(used.length === names.length, `one usemtl per object (${used.length} vs ${names.length})`)
  ok(defined.size === names.length, `one newmtl per object (${defined.size} vs ${names.length})`)

  const dangling = used.filter((u) => !defined.has(u))
  ok(dangling.length === 0,
    `no usemtl points at a material that does not exist (dangling: ${dangling.slice(0, 3).join(', ')})`)
  const unused = [...defined].filter((d) => !used.includes(d))
  ok(unused.length === 0, `and no newmtl is defined that nothing uses (${unused.slice(0, 3).join(', ')})`)

  // Non-vacuous: the check above must actually reject a dangling reference.
  // If it cannot, it proves nothing when it passes.
  const broken = pair.obj.replace(`usemtl ${GROUP_FRAME}`, 'usemtl frame_typo')
  const brokenUsed = [...broken.matchAll(/^usemtl (.+)$/gm)].map((m) => m[1].trim())
  ok(brokenUsed.some((u) => !defined.has(u)), 'the resolution check rejects a deliberately broken reference')

  // The MTL blocks are well-formed, not just present.
  const blocks = pair.mtl.split(/^newmtl /m).slice(1)
  ok(blocks.every((b) => /^Kd [\d.]+ [\d.]+ [\d.]+$/m.test(b)), 'every material declares a diffuse colour')
  ok(blocks.every((b) => /^illum \d+$/m.test(b)), 'every material declares an illumination model')
  ok(blocks.every((b) => /^Ke [\d.]+ [\d.]+ [\d.]+$/m.test(b)), 'every material declares emission')
  ok(!pair.mtl.includes('NaN'), 'no NaN in the library')

  // The diffusers are the lit ones and the merged groups are not — the whole
  // reason the diffusers are individual is that they take brightness.
  const keOf = (name) => {
    const b = pair.mtl.split(/^newmtl /m).find((s) => s.startsWith(`${name}\n`))
    return b.match(/^Ke ([\d.]+)/m)[1]
  }
  const firstDiffuser = names.find((n) => n.startsWith(`${DIFFUSER_PREFIX}_`))
  ok(Number(keOf(firstDiffuser)) > 0, 'a diffuser emits')
  ok(Number(keOf(GROUP_FRAME)) === 0, 'the frame does not')
  ok(Number(keOf(GROUP_CONNECTORS)) === 0, 'nor do the connectors')

  // The default-name path still produces a resolvable pair, since objPayloadV4
  // is what the older suite calls.
  ok(objPayloadV4(base.cfg, base.chain, base.connectors).includes(`mtllib ${DEFAULT_MTL_NAME}`),
    'the standalone OBJ writer falls back to the default library name')
  const solo = mtlPayloadV4(base.cfg, base.chain, base.connectors)
  ok([...solo.matchAll(/^newmtl /gm)].length === names.length,
    'and the standalone MTL writer produces the same count')
}

// -----------------------------------------------------------------------------
// 2. THE INDICES ARE STILL SEPARABLE
//
// OBJ face indices are file-global and one-based. If an `o` block ever referred
// to a vertex declared after it, or restarted numbering, the blocks would not be
// independently loadable — which is the other way this export could look merged.
// -----------------------------------------------------------------------------
console.log('2. face indices stay global, cumulative and never forward')
{
  let vertexCount = 0
  let objectCount = 0
  let maxIndex = 0
  let forwardRefs = 0
  let zeroOrNegative = 0

  for (const line of pair.obj.split('\n')) {
    if (line.startsWith('v ')) vertexCount++
    else if (line.startsWith('o ')) objectCount++
    else if (line.startsWith('f ')) {
      for (const tok of line.slice(2).trim().split(/\s+/)) {
        const i = Number(tok.split('/')[0])
        if (!Number.isFinite(i) || i <= 0) { zeroOrNegative++; continue }
        if (i > vertexCount) forwardRefs++
        if (i > maxIndex) maxIndex = i
      }
    }
  }

  ok(objectCount === names.length, `${objectCount} o blocks, matching the scene`)
  ok(vertexCount > 0, `${vertexCount} vertices written`)
  ok(forwardRefs === 0, `no face refers to a vertex not yet declared (${forwardRefs})`)
  ok(zeroOrNegative === 0, 'no zero, negative or unparseable index (OBJ is one-based)')
  ok(maxIndex === vertexCount, `the last face uses the last vertex (${maxIndex} vs ${vertexCount})`)
}

// -----------------------------------------------------------------------------
// 3. THE GLB, DECODED
// -----------------------------------------------------------------------------
console.log('3. the GLB container decodes, and carries 41 materials rather than 1')
{
  const g = readGLB(glb)
  ok(g.magic === 0x46546c67, 'the magic is glTF')
  ok(g.version === 2, 'container version 2')
  ok(g.declaredTotal === glb.byteLength,
    `the header's total length matches the file (${g.declaredTotal} vs ${glb.byteLength})`)
  ok(g.consumed === glb.byteLength, 'the chunks account for every byte — no trailing slack')
  ok(g.json !== null, 'there is a JSON chunk')
  ok(g.binLength !== null, 'there is a BIN chunk')
  // The declared buffer is the payload; the chunk is padded to 4 bytes, so it is
  // at least as long and by less than a word.
  const declaredBuffer = g.json.buffers[0].byteLength
  ok(g.binLength >= declaredBuffer && g.binLength - declaredBuffer < 4,
    `the BIN chunk holds the declared buffer (${g.binLength} vs ${declaredBuffer})`)

  // The node graph. One root plus one node per exported object.
  const nodeNames = g.json.nodes.map((n) => n.name)
  ok(g.json.nodes.length === names.length + 1,
    `${g.json.nodes.length} nodes — one per object plus the root`)
  ok(nodeNames.includes('drop_ceiling'), 'the root node is named')
  const missing = names.filter((n) => !nodeNames.includes(n))
  ok(missing.length === 0, `every expected node name is present (missing: ${missing.slice(0, 3).join(', ')})`)
  for (const grp of [GROUP_FRAME, GROUP_CONNECTORS, GROUP_SUPPLIES]) {
    ok(nodeNames.filter((n) => n === grp).length === 1, `exactly one "${grp}" node`)
  }

  ok(g.json.meshes.length === names.length, `${g.json.meshes.length} meshes, one per object`)
  ok(g.json.materials.length === names.length,
    `${g.json.materials.length} materials — NOT 1, which is what a merged import looks like`)
  ok(g.json.materials.length > 1, 'and that is emphatically more than one')

  // Distinctness: 37 diffusers must own 37 different material INDICES, not 37
  // references to a shared one. A shared material means one brightness for all.
  const diffuserNodes = g.json.nodes.filter((n) => n.name?.startsWith(`${DIFFUSER_PREFIX}_`))
  const diffuserMatIdx = diffuserNodes.map((n) => g.json.meshes[n.mesh].primitives[0].material)
  ok(diffuserNodes.length === names.filter((n) => n.startsWith(`${DIFFUSER_PREFIX}_`)).length,
    `${diffuserNodes.length} diffuser nodes`)
  ok(new Set(diffuserMatIdx).size === diffuserNodes.length,
    `each diffuser points at its OWN material (${new Set(diffuserMatIdx).size} distinct of ${diffuserNodes.length})`)
  ok(new Set(diffuserMatIdx.map((i) => g.json.materials[i].name)).size === diffuserNodes.length,
    'and those materials have distinct names')

  // Brightness: emission survives, with a strength channel that can be driven.
  const dm = g.json.materials[diffuserMatIdx[0]]
  ok(Array.isArray(dm.emissiveFactor) && dm.emissiveFactor.some((c) => c > 0),
    'a diffuser material is emissive')
  // VERIFIED, not assumed: GLTFExporter writes KHR_materials_emissive_strength
  // only when emissiveIntensity !== 1 (GLTFExporter.js, the extension's
  // writeMaterialAsync returns early at exactly 1.0). The shipped default IS 1 —
  // a neutral starting point — so the extension is correctly ABSENT from the
  // default file, and asserting it were present would pin a bug. What matters is
  // that the channel exists and carries a driven value, so that is what is
  // checked, on a group built for the purpose.
  ok(!(g.json.extensionsUsed ?? []).includes('KHR_materials_emissive_strength'),
    'at the default intensity of 1 the strength extension is correctly absent')
  {
    const probe = buildSceneGroup(base.cfg, base.chain, base.connectors)
    const lit = probe.children.find((c) => c.name.startsWith(`${DIFFUSER_PREFIX}_`))
    lit.material.emissiveIntensity = 4.25
    const drivenJson = readGLB(await new (await import('three/examples/jsm/exporters/GLTFExporter.js'))
      .GLTFExporter().parseAsync(probe, { binary: true, onlyVisible: false })).json
    ok((drivenJson.extensionsUsed ?? []).includes('KHR_materials_emissive_strength'),
      'and turning one panel up writes KHR_materials_emissive_strength')
    const strengths = drivenJson.materials
      .map((m) => m.extensions?.KHR_materials_emissive_strength?.emissiveStrength)
      .filter((s) => s !== undefined)
    ok(strengths.length === 1 && strengths[0] === 4.25,
      `carrying exactly that panel's value and no other (${JSON.stringify(strengths)})`)
  }
  const frameNode = g.json.nodes.find((n) => n.name === GROUP_FRAME)
  const fm = g.json.materials[g.json.meshes[frameNode.mesh].primitives[0].material]
  ok(!fm.emissiveFactor || fm.emissiveFactor.every((c) => c === 0), 'the frame material is not emissive')

  // Every mesh has real geometry behind it — a node graph over empty primitives
  // would satisfy every count above.
  ok(g.json.meshes.every((m) => m.primitives.length > 0 && m.primitives[0].attributes.POSITION !== undefined),
    'every mesh has a POSITION attribute')
  const posCounts = g.json.meshes.map((m) => g.json.accessors[m.primitives[0].attributes.POSITION].count)
  ok(posCounts.every((c) => c > 0), 'and none of them is empty')
  ok(g.json.accessors.length >= g.json.meshes.length, 'accessors at least keep up with meshes')
}

// -----------------------------------------------------------------------------
// 4. THE TWO WRITERS CANNOT DRIFT
//
// Both consume buildSceneGroup, so today they agree by construction. This pins
// it, because "add a mesh to one path only" is a one-line mistake.
// -----------------------------------------------------------------------------
console.log('4. the OBJ and the GLB describe the same scene')
{
  const objNames = [...pair.obj.matchAll(/^o (.+)$/gm)].map((m) => m[1].trim())
  const gltfNames = gltf.nodes.map((n) => n.name).filter((n) => n !== 'drop_ceiling')

  ok(objNames.length === names.length && gltfNames.length === names.length,
    `both writers emit ${names.length} objects`)
  ok(objNames.every((n, i) => n === names[i]), 'the OBJ names match buildSceneGroup, in order')
  ok(gltfNames.every((n, i) => n === names[i]), 'the glTF node names match buildSceneGroup, in order')
  ok(new Set(objNames).size === objNames.length, 'and every name is unique, so nothing can collide on import')

  // It tracks the design in BOTH formats, not a constant. Drop a panel and both
  // shrink by exactly one — a suite that only ever solved the default config
  // would pass against a hardcoded list.
  const cut = solve({ overrides: { cells: [{ i: 1, j: 2, present: false }], edges: [] } })
  const cutPair = objMtlPairV4(cut.cfg, cut.chain, cut.connectors, null, 'cut')
  const cutGltf = readGLB(await glbPayloadV4(cut.cfg, cut.chain, cut.connectors)).json
  ok([...cutPair.obj.matchAll(/^o /gm)].length === names.length - 1, 'removing a panel removes one OBJ object')
  ok(cutGltf.nodes.length === names.length, 'and one glTF node')
  ok(cutGltf.materials.length === names.length - 1, 'and one glTF material')
  ok(!cutGltf.nodes.some((n) => n.name?.includes('Ci1j2')), 'and it is that panel that went')
  ok([...cutPair.mtl.matchAll(/^newmtl /gm)].length === names.length - 1,
    'the MTL follows too, so nothing dangles after an edit')

  // A conditional group vanishing must vanish from both. `power_supplies` is
  // the one group that is still conditional: 'none' solves the connectors as
  // though there is no driver, so exporting a box for one would contradict the
  // rest of the model.
  const un = solve({ connectors: { ...DEFAULT_CONFIG.connectors, powerEdge: 'none' } })
  const unPair = objMtlPairV4(un.cfg, un.chain, un.connectors, 'un')
  const unGltf = readGLB(await glbPayloadV4(un.cfg, un.chain, un.connectors)).json
  ok(names.includes(GROUP_SUPPLIES), 'the default export HAS a supplies object — the check below is not vacuous')
  ok(!unPair.obj.includes(`\no ${GROUP_SUPPLIES}`), "powerEdge 'none' removes the supplies object from the OBJ")
  ok(!unPair.mtl.includes(`newmtl ${GROUP_SUPPLIES}`), 'and its material from the MTL')
  ok(!unGltf.nodes.some((n) => n.name === GROUP_SUPPLIES), 'and its node from the GLB')
  ok(unGltf.materials.length === names.length - 1, 'and its material from the GLB')
}

// -----------------------------------------------------------------------------
// 5. THE SCENE GROUP'S MATERIALS ARE INSTANCES, NOT ONE SHARED OBJECT
//
// Checked at the source, because this is the property both writers depend on and
// the cheapest place for it to silently regress (`const M = new Material()`
// hoisted out of a loop looks like an optimisation).
// -----------------------------------------------------------------------------
console.log('5. buildSceneGroup gives every mesh its own material instance')
{
  const group = buildSceneGroup(base.cfg, base.chain, base.connectors)
  const mats = group.children.map((m) => m.material)
  ok(new Set(mats).size === mats.length, `${new Set(mats).size} distinct material objects for ${mats.length} meshes`)
  ok(new Set(mats.map((m) => m.uuid)).size === mats.length, 'distinct uuids, which is what the exporters key on')
  ok(mats.every((m, i) => m.name === group.children[i].name), 'each material is named for its object')
  const diffMats = group.children
    .filter((c) => c.name.startsWith(`${DIFFUSER_PREFIX}_`))
    .map((c) => c.material)
  ok(diffMats.every((m) => m.emissive instanceof THREE.Color && m.emissive.getHex() !== 0x000000),
    'every diffuser material carries emission')
  ok(diffMats.every((m) => m.emissiveIntensity > 0), 'and a non-zero intensity to scale it by')
  ok(group.children.find((c) => c.name === GROUP_SUPPLIES).material.emissive.getHex() === 0x000000,
    'the power supplies do not glow')
}

// -----------------------------------------------------------------------------
// 6. THE ROOM SHIPS AS OPT-IN, GROUPED BY FAMILY
// -----------------------------------------------------------------------------
console.log('6. env_* groups: opt-in, one mesh per family, every obstacle accounted for')
{
  const { environmentFamily, ENV_GROUPS } = await import('../src/v4/objExport.js')
  const { DEFAULT_OBSTACLES } = await import('../src/core/v4/schema.js')

  // OFF by default: an installation-only export must not carry the room, or the
  // downstream that reads it as "the parts" gets a room too.
  const plain = readGLB(await glbPayloadV4(base.cfg, base.chain, base.connectors)).json
  ok(!plain.nodes.some((n) => (n.name || '').startsWith('env_')),
    'the default GLB has no env_ nodes')

  // ON: one merged node per family that had any obstacles in it.
  const withRoom = readGLB(
    await glbPayloadV4(base.cfg, base.chain, base.connectors, { includeEnvironment: true }),
  ).json
  const envNodes = withRoom.nodes.filter((n) => (n.name || '').startsWith('env_')).map((n) => n.name)
  ok(envNodes.length > 0 && envNodes.every((n) => ENV_GROUPS.includes(n)),
    `every env_ node is a declared family (got ${envNodes.join(', ')})`)
  ok(new Set(envNodes).size === envNodes.length,
    'no family emits two nodes — the merging is per family, not per obstacle')
  ok(withRoom.nodes.length - plain.nodes.length === envNodes.length,
    `adding the room adds exactly one node per family (${withRoom.nodes.length - plain.nodes.length} new, ${envNodes.length} families)`)

  // EVERY DEFAULT OBSTACLE MAPS TO A FAMILY. The rule is closed on purpose —
  // an id with no rule silently vanishes, which is the failure this catches.
  const unmapped = DEFAULT_OBSTACLES.filter((o) => environmentFamily(o.id) === null)
  ok(unmapped.length === 0,
    `no default obstacle is unmapped (${unmapped.map((o) => o.id).join(', ') || 'none'})`)

  // ...and the families named in ENV_GROUPS actually have members, so we do not
  // ship phantom empty node names. `env_stair_band` is the one family NOT fed by
  // obstacles — it is the oriented raking beams from `stairBands()` — so it is
  // added to the used set explicitly.
  const used = new Set(DEFAULT_OBSTACLES.map((o) => environmentFamily(o.id)).filter(Boolean))
  // The stair edge families come from `stairBands()`, and the wall from
  // `slatWall()` — neither is an obstacle, so add them explicitly.
  used.add('env_stair_band')
  used.add('env_stair_balustrades')
  used.add('env_stair_handrail')
  used.add('env_wall')
  const extras = ENV_GROUPS.filter((g) => !used.has(g))
  ok(extras.length === 0, `no family is declared without members (${extras.join(', ') || 'none'})`)

  // Order is by ENV_GROUPS, not by whatever the obstacles happened to iterate
  // in — determinism is the standing rule and the emit order is the visible
  // proof of it.
  const expected = ENV_GROUPS.filter((g) => used.has(g))
  ok(envNodes.join('|') === expected.join('|'),
    `env_ nodes come out in ENV_GROUPS order (got ${envNodes.join('|')})`)

  // Every env_ material is a fresh instance — same reason the diffusers are.
  const envMats = withRoom.materials.filter((m) => (m.name || '').startsWith('env_'))
  ok(new Set(envMats.map((m) => m.name)).size === envMats.length,
    'and each env_ mesh carries a material of its own')
}

console.log(`\ntest-v4-export: ${passed} checks passed, ${failed} failed`)
process.exit(failed ? 1 : 0)
