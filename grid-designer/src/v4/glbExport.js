/**
 * grid-designer v4 — GLB export, the format that actually arrives split.
 *
 * =============================================================================
 * WHY A SECOND FORMAT AT ALL
 * =============================================================================
 * OBJ + MTL now names its materials properly and will import as parts. But OBJ
 * is two files that must travel together, its material model is from 1992, and
 * "brightness" in it is a `Ke` triplet with no strength — turn a panel up past
 * white and there is nowhere to put the number. GLB is one self-contained file,
 * carries a node graph with names, one material per mesh, and an emissive
 * strength that survives as `KHR_materials_emissive_strength`. For the stated
 * job — 37 diffusers each with their own brightness — it is the better answer,
 * and OBJ stays for the readers that only speak OBJ.
 *
 * =============================================================================
 * WHY NOT FBX, WHICH IS WHAT WAS ASKED FOR FIRST
 * =============================================================================
 * three.js ships an FBXLoader and NO FBXExporter — it never has. Writing one
 * means hand-emitting the binary FBX record tree (a documented-by-reverse-
 * engineering format), or taking a dependency to do it. Neither is worth it when
 * every target that reads FBX also reads GLB, so this is the route.
 *
 * =============================================================================
 * ONE SCENE, TWO WRITERS
 * =============================================================================
 * The geometry and the material list both come from `objExport.buildSceneGroup`.
 * This module owns no geometry and no palette; it is a serializer and nothing
 * else, which is why the OBJ and the GLB cannot describe different scenes.
 * `tests/test-v4-export.mjs` pins that by asserting the name lists match.
 */

import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js'
import { buildSceneGroup } from './objExport.js'

/**
 * Serialize to a binary glTF container.
 *
 * Async because `GLTFExporter` is: it hands the packed buffer through a
 * `FileReader`, so there is no synchronous form to wrap. Callers get the raw
 * ArrayBuffer rather than a Blob so this stays headless and the test can read
 * the container back byte by byte.
 *
 * `onlyVisible: false` because visibility is a viewport affordance in this app
 * (the plan grid dims things) and has never meant "do not export me" — the
 * `present` flag is what decides that, upstream in `buildSceneGroup`.
 *
 * @returns {Promise<ArrayBuffer>} a complete .glb
 */
export async function glbPayloadV4(config, chain, connectors) {
  const group = buildSceneGroup(config, chain, connectors)
  const out = await new GLTFExporter().parseAsync(group, {
    binary: true,
    onlyVisible: false,
  })
  // parseAsync resolves with a JSON object when binary is false; if it ever does
  // here, the caller would silently write a text file with a .glb extension.
  if (!(out instanceof ArrayBuffer)) throw new Error('glbPayloadV4: expected a binary glTF')
  return out
}
