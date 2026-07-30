/**
 * grid-designer v4 — OBJ export, organised for assigning materials downstream.
 *
 * =============================================================================
 * WHAT THE GROUPING IS FOR
 * =============================================================================
 * The old exporter emitted one object per panel and one per connector, which is
 * the right shape for checking geometry and the wrong shape for lighting a
 * scene. This one splits the model the way a renderer wants to receive it:
 *
 *   diffuser_NN_<id>   ONE OBJECT PER PANEL. These are the lit faces, and they
 *                      are kept individual precisely so each can be given its
 *                      own brightness — that is the whole reason for the split.
 *   frame              every panel's housing, merged into one object
 *   connectors         every printed part, back halves and front bars, merged
 *   power_supplies     every driver box, merged
 *
 * The merged groups are things that take ONE material each, so keeping
 * them apart as hundreds of objects is only clutter in the outliner.
 *
 * =============================================================================
 * WHERE THE DIFFUSER/FRAME SPLIT COMES FROM
 * =============================================================================
 * It is not a new decision. `panelGeometry.js` already emits the panel solid as
 * two material groups — `DIFFUSER_MATERIAL_INDEX` and `HOUSING_MATERIAL_INDEX`
 * — and the viewport has always rendered them as two materials. This module
 * just cuts the geometry along a line that was already drawn, so a change to the
 * measured section flows through here with nothing to update.
 *
 * =============================================================================
 * WHY NOT src/utils/exporters.js
 * =============================================================================
 * That module is shared with the retired v3 UI and its suite pins its output.
 * Its `buildExportGroup` bakes one mesh per panel with both material groups
 * intact, which is a different document from this one — so this is a parallel
 * builder rather than a flag on that one. Neither has to compromise.
 *
 * Every mesh carries a NAMED material, so the OBJ gets `usemtl` lines and the
 * grouping survives import into anything that reads them.
 *
 * =============================================================================
 * WHY THERE IS AN .MTL AT ALL
 * =============================================================================
 * The first cut of this emitted `usemtl <name>` for all 41 objects and shipped
 * no material library — no `mtllib` line, no `.mtl` file. Every one of those
 * names resolved to nothing, and an importer that splits a mesh BY MATERIAL
 * (which is how most of them do it) found no materials to split on and merged
 * the lot into one surface. That was the whole reported bug: the `o` blocks were
 * correct the entire time, the materials they named did not exist.
 *
 * So the library is written from the SAME `buildSceneGroup` the geometry comes
 * from — `mtlPayloadV4` walks the group's meshes and emits one `newmtl` per
 * material it actually finds. A material cannot appear in one file and not the
 * other, because there is only one list. Hardcoding a parallel palette here was
 * the obvious alternative and is exactly the drift that caused the bug.
 *
 * =============================================================================
 * WHY THE DIFFUSERS ARE EMISSIVE
 * =============================================================================
 * Per-panel brightness is the reason the diffusers are split out one-by-one, and
 * emission is the channel that carries brightness. Each panel therefore gets its
 * OWN material instance rather than 37 references to a shared one — a shared
 * material would make "give this panel its own brightness" mean "change all 37",
 * which defeats the split. The merged groups are non-emissive and given
 * distinguishable colours so the outliner is readable before anyone re-lights it.
 */

import * as THREE from 'three'
import { OBJExporter } from 'three/examples/jsm/exporters/OBJExporter.js'
import {
  buildPanelGeometry,
  buildPowerSupplyGeometry,
  DIFFUSER_MATERIAL_INDEX,
  HOUSING_MATERIAL_INDEX,
} from '../geometry/panelGeometry.js'
import { buildConnectorGeometry, buildFrontBarGeometry, connectorTransform } from '../geometry/connectorGeometry.js'
import { getConnectorKit } from './exportAdapter.js'
import { obstacleExtents } from '../core/v4/obstacles.js'
import { stairBands, slatWall, wallDepthCm } from '../core/v4/schema.js'

/** Object / material names. Kept as constants because they are the contract
 *  with whatever opens the file, not incidental strings. */
export const GROUP_FRAME = 'frame'
export const GROUP_CONNECTORS = 'connectors'
export const GROUP_SUPPLIES = 'power_supplies'
export const DIFFUSER_PREFIX = 'diffuser'

/**
 * The environment groups. Prefixed `env_` so nothing that reads only the
 * installation ever picks them up by accident, and named by FAMILY rather than
 * by kind — a stair tread and a mullion are both `solid` but nobody thinks of
 * them as the same thing.
 */
export const ENV_PREFIX = 'env_'
export const ENV_GROUPS = [
  'env_wall',
  'env_floor',
  'env_column',
  'env_mullions',
  'env_mullion_caps',
  'env_sills',
  'env_glass',
  'env_heating',
  'env_stair_treads',
  'env_stair_landing',
  'env_stair_band',
  'env_stair_balustrades',
  'env_stair_handrail',
]

/** The stair edge assembly's `part` → env family. */
const STAIR_PART_FAMILY = {
  band: 'env_stair_band',
  glass: 'env_stair_balustrades',
  handrail: 'env_stair_handrail',
}

/**
 * Which family an obstacle belongs to, by id.
 *
 * Explicit and small on purpose: obstacles are DERIVED, so the id vocabulary is
 * closed and adding one that needs its own family means updating this file too.
 * A rule matched by regex over `kind` would work today and fall apart the first
 * time a new solid did not want to be in `env_other`.
 *
 * Returns null for an id no rule matches; the caller drops it rather than
 * inventing a bucket for a naming mistake.
 */
export function environmentFamily(id) {
  if (id === 'column') return 'env_column'
  if (id === 'floor') return 'env_floor'
  if (id === 'stair-landing') return 'env_stair_landing'
  if (id.startsWith('heating')) return 'env_heating'
  if (/^stair-f\d+-step-/.test(id)) return 'env_stair_treads'
  if (/-cap$/.test(id) && id.startsWith('mullion-')) return 'env_mullion_caps'
  if (id.startsWith('mullion-')) return 'env_mullions'
  if (id === 'glass' || id === 'glass-return') return 'env_glass'
  if (id.startsWith('sill') || id.startsWith('spandrel')) return 'env_sills'
  return null
}

/** Default companion-library name, for callers that do not stamp their files. */
export const DEFAULT_MTL_NAME = 'drop-ceiling.mtl'

/**
 * The look of each group, as MeshStandardMaterial parameters.
 *
 * Colours are chosen only to be TELLABLE APART on import — nobody is shipping
 * these as the final render, and a plausible-looking aluminium would be a worse
 * default because it hides which object you clicked. The diffuser entry is the
 * one with meaning: `emissive` white at `emissiveIntensity` 1 is a neutral
 * starting point that the user turns up or down per panel.
 *
 * `emissiveIntensity` is deliberately left at exactly 1. Verified against
 * GLTFExporter: its KHR_materials_emissive_strength writer returns early when
 * the intensity is 1.0, so the default file carries a plain `emissiveFactor`
 * and no extension — which is right, since there is no strength to declare yet.
 * Set it to anything else and the extension appears with that value, so the
 * channel is there the moment a panel is actually driven.
 */
const DIFFUSER_LOOK = {
  color: 0xf4f4f2,
  emissive: 0xffffff,
  emissiveIntensity: 1,
  roughness: 0.9,
  metalness: 0,
}
const GROUP_LOOKS = {
  [GROUP_FRAME]: { color: 0x2e3033, roughness: 0.55, metalness: 0.4 },
  [GROUP_CONNECTORS]: { color: 0xd06a2c, roughness: 0.8, metalness: 0 },
  [GROUP_SUPPLIES]: { color: 0x3f8f5a, roughness: 0.6, metalness: 0.2 },
  // Environment. Visibly distinct starting looks; the whole point of exporting
  // them is so they can be re-shaded in the DCC. Glass and heating are
  // translucent so the installation reads through them by default.
  env_wall: { color: 0x6b4a2f, roughness: 0.8, metalness: 0 },
  env_floor: { color: 0x6d5642, roughness: 0.9, metalness: 0 },
  env_column: { color: 0x8a8f96, roughness: 0.7, metalness: 0.1 },
  env_mullions: { color: 0x1f242b, roughness: 0.4, metalness: 0.6 },
  env_mullion_caps: { color: 0x1f242b, roughness: 0.4, metalness: 0.6 },
  env_sills: { color: 0x8a8f96, roughness: 0.7, metalness: 0.1 },
  env_glass: { color: 0xb8dcee, roughness: 0.05, metalness: 0, opacity: 0.15, transparent: true },
  env_heating: { color: 0xd07a3a, roughness: 0.8, metalness: 0, opacity: 0.25, transparent: true },
  env_stair_treads: { color: 0xf1efe9, roughness: 0.6, metalness: 0 },
  env_stair_landing: { color: 0xf1efe9, roughness: 0.6, metalness: 0 },
  // The strong white structural band. Brightest white of the stair, since the
  // photos read it as the crisp painted fascia the glass sits on.
  env_stair_band: { color: 0xffffff, roughness: 0.5, metalness: 0 },
  // The handrail cap along the top of the glass — dark metal.
  env_stair_handrail: { color: 0x2a2d33, roughness: 0.4, metalness: 0.6 },
  env_stair_balustrades: { color: 0xb8dcee, roughness: 0.05, metalness: 0, opacity: 0.2, transparent: true },
}

/**
 * A fresh material for one exported object.
 *
 * ALWAYS A NEW INSTANCE, never a shared singleton, even for the merged groups:
 * GLTFExporter writes one glTF material per distinct instance, so sharing would
 * silently collapse the 37 diffusers into one entry and take per-panel
 * brightness with it. The merged groups gain nothing from sharing either — there
 * is exactly one mesh each.
 */
function materialFor(name) {
  const look = name.startsWith(`${DIFFUSER_PREFIX}_`) ? DIFFUSER_LOOK : GROUP_LOOKS[name]
  return new THREE.MeshStandardMaterial({ name, ...(look ?? {}) })
}

/**
 * Which panel-local edge carries the driver, from the same `connectors.powerEdge`
 * knob the connector solver reads — so the box is drawn where the parts were
 * placed around it, and 'none' exports no supplies at all rather than exporting
 * ones the rest of the model was solved as though absent.
 */
function supplyEdgeFor(policy) {
  if (policy === 'high') return 2 // north, +Z local
  if (policy === 'none') return null
  return 0 // 'low' — south, −Z local
}

/**
 * The world transform of a panel, as a matrix.
 *
 * The stored quaternion is rounded to 1e-9 for determinism, so it is normalized
 * before composing: an un-normalized quaternion scales the baked geometry by
 * |q|², which is a silent few-parts-per-billion error in every exported vertex.
 * `exporters.js` applies the same correction for the same reason.
 */
function worldMatrix(position, quaternion) {
  return new THREE.Matrix4().compose(
    new THREE.Vector3().fromArray(position),
    new THREE.Quaternion().fromArray(quaternion).normalize(),
    new THREE.Vector3(1, 1, 1),
  )
}

/**
 * One material group of an indexed geometry, as a standalone geometry.
 *
 * Vertices are REMAPPED rather than dereferenced, so the result stays indexed
 * and the file does not triple in size — a 37-panel network is ~90 objects and
 * the duplication is not free. Returns null for an empty group.
 */
function extractGroup(geometry, materialIndex) {
  const group = (geometry.groups ?? []).find((g) => g.materialIndex === materialIndex)
  if (!group || group.count === 0) return null

  const pos = geometry.getAttribute('position')
  const nor = geometry.getAttribute('normal')
  const idx = geometry.getIndex()
  if (!idx) throw new Error('extractGroup: expected an indexed geometry')

  const remap = new Map()
  const positions = []
  const normals = []
  const indices = []

  for (let k = group.start; k < group.start + group.count; k++) {
    const v = idx.getX(k)
    let n = remap.get(v)
    if (n === undefined) {
      n = positions.length / 3
      remap.set(v, n)
      positions.push(pos.getX(v), pos.getY(v), pos.getZ(v))
      if (nor) normals.push(nor.getX(v), nor.getY(v), nor.getZ(v))
    }
    indices.push(n)
  }

  const out = new THREE.BufferGeometry()
  out.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
  if (normals.length) out.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3))
  out.setIndex(indices)
  return out
}

/** Accumulates geometries into one merged, indexed geometry. */
function Merger() {
  const positions = []
  const normals = []
  const indices = []
  let base = 0

  return {
    add(geometry) {
      const pos = geometry.getAttribute('position')
      const nor = geometry.getAttribute('normal')
      const idx = geometry.getIndex()
      for (let v = 0; v < pos.count; v++) {
        positions.push(pos.getX(v), pos.getY(v), pos.getZ(v))
        if (nor) normals.push(nor.getX(v), nor.getY(v), nor.getZ(v))
      }
      if (idx) {
        for (let k = 0; k < idx.count; k++) indices.push(base + idx.getX(k))
      } else {
        for (let v = 0; v < pos.count; v++) indices.push(base + v)
      }
      base += pos.count
    },
    isEmpty() {
      return base === 0
    },
    build() {
      const g = new THREE.BufferGeometry()
      g.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
      // Only when EVERY source carried normals; a partial normal attribute is
      // worse than none, because it silently mis-shades whichever part is short.
      if (normals.length === positions.length) {
        g.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3))
      }
      g.setIndex(indices)
      return g
    },
  }
}

function meshOf(geometry, name) {
  const mesh = new THREE.Mesh(geometry, materialFor(name))
  mesh.name = name
  mesh.updateMatrixWorld(true)
  return mesh
}

/**
 * Add the room's obstacles to an export scene as ONE MERGED MESH PER FAMILY.
 *
 * Every obstacle is an axis-aligned box, so each becomes a `BoxGeometry` and is
 * translated to its own centre. Same-family boxes merge into one indexed
 * geometry, one material — a DCC importer then sees ten mullions as one
 * `env_mullions` object, forty stair treads as one `env_stair_treads`, and so
 * on. The families are declared in `ENV_GROUPS`; `environmentFamily` decides
 * which one an id belongs to.
 *
 * These groups exist ALONGSIDE the installation groups, prefixed `env_` — so
 * anything that reads only the design can filter them out by name.
 */
/** The window head off the glazing, for the wall's height — matches the
 *  viewport's `windowHeadCm`. */
function wallHeightCm(config) {
  let top = -Infinity
  for (const o of (config.obstacles ?? [])) {
    if (o.id === 'glass' || o.id === 'glass-return') top = Math.max(top, obstacleExtents(o).max[1])
  }
  return Number.isFinite(top) ? top : 375
}

function addEnvironment(group, config, chain) {
  const mergers = new Map()
  const seenIds = new Set()
  for (const obstacle of (config.obstacles ?? [])) {
    const fam = environmentFamily(obstacle.id)
    if (fam === null) continue
    if (seenIds.has(obstacle.id)) continue
    seenIds.add(obstacle.id)
    const e = obstacleExtents(obstacle)
    const box = new THREE.BoxGeometry(e.size[0], e.size[1], e.size[2])
    box.translate(e.centre[0], e.centre[1], e.centre[2])
    if (!mergers.has(fam)) mergers.set(fam, Merger())
    mergers.get(fam).add(box)
    box.dispose()
  }

  // The stair edge assembly is not obstacles — a raking stringer cannot be
  // axis-aligned. Its mitred `ribbon` profiles are extruded along x; its
  // straight `boxes` are plain. Each routes to a family by its `part`
  // (band / glass / handrail) via STAIR_PART_FAMILY.
  const stair = stairBands()
  const stairMerger = (part) => {
    const fam = STAIR_PART_FAMILY[part]
    if (!mergers.has(fam)) mergers.set(fam, Merger())
    return mergers.get(fam)
  }
  for (const r of stair.ribbons) {
    const shape = new THREE.Shape(r.polygon.map((p) => new THREE.Vector2(p[0], p[1])))
    const geo = new THREE.ExtrudeGeometry(shape, { depth: r.thickness, bevelEnabled: false })
    geo.rotateY(-Math.PI / 2)
    geo.translate(r.xCenter + r.thickness / 2, 0, 0)
    stairMerger(r.part).add(geo) // ExtrudeGeometry is already non-indexed
    geo.dispose()
  }
  for (const b of stair.boxes) {
    const box = new THREE.BoxGeometry(b.size[0], b.size[1], b.size[2])
    box.translate(b.center[0], b.center[1], b.center[2])
    stairMerger(b.part).add(box)
    box.dispose()
  }

  // The interior SLAT WALL — the same box geometry the viewport draws, from the
  // shared `slatWall()`. Frame and slats merge into one `env_wall` mesh.
  const wall = slatWall({
    thicknessCm: config.room?.wallThicknessCm ?? 8.9,
    heightCm: wallHeightCm(config),
    depthCm: wallDepthCm(chain),
  })
  const wallMerger = () => {
    if (!mergers.has('env_wall')) mergers.set('env_wall', Merger())
    return mergers.get('env_wall')
  }
  for (const m of [...wall.frame, ...wall.slats]) {
    const box = new THREE.BoxGeometry(Math.max(m.size[0], 0.01), m.size[1], m.size[2])
    box.translate(m.center[0], m.center[1], m.center[2])
    wallMerger().add(box)
    box.dispose()
  }

  // Emit in the declared order, so the export is deterministic even if the
  // obstacle list happens to iterate in a different one.
  for (const fam of ENV_GROUPS) {
    const m = mergers.get(fam)
    if (!m || m.isEmpty()) continue
    const geometry = m.build()
    geometry.computeBoundingBox()
    geometry.computeBoundingSphere()
    group.add(meshOf(geometry, fam))
  }
}

/**
 * Build the export scene: individual diffusers, then the merged groups.
 *
 * ABSENT PANELS ARE NOT EXPORTED. `present: false` means the panel is not there;
 * the chain keeps its record so removal stays non-destructive, but the OBJ is a
 * description of what actually gets built.
 *
 * With `{ includeEnvironment: true }` the room's obstacles are appended as
 * per-family merged meshes — the panels get placed against something. See
 * `addEnvironment` for how families are decided.
 *
 * @param {object} config normalized v4 config
 * @param {object} chain `solveLattice` output
 * @param {object} connectors `solveConnectorsV4` output
 * @param {{ includeEnvironment?: boolean }} [options]
 * @returns {THREE.Group}
 */
export function buildSceneGroup(config, chain, connectors, options = {}) {
  const group = new THREE.Group()
  group.name = 'drop_ceiling'

  const frame = Merger()
  const supplies = Merger()
  const parts = Merger()
  const supplyEdge = supplyEdgeFor(config.connectors?.powerEdge)

  const present = chain.panels.filter((p) => p.present)

  present.forEach((panel, n) => {
    const matrix = worldMatrix(panel.position, panel.quaternion)
    const solid = buildPanelGeometry({ type: panel.panelType })

    // The diffuser is its own object, one per panel — the point of the exercise.
    const diffuser = extractGroup(solid, DIFFUSER_MATERIAL_INDEX)
    if (diffuser) {
      diffuser.applyMatrix4(matrix)
      diffuser.computeBoundingBox()
      diffuser.computeBoundingSphere()
      // Zero-padded index first so the objects sort in build order rather than
      // alphabetically by id, which interleaves cells and ramps.
      group.add(meshOf(diffuser, `${DIFFUSER_PREFIX}_${String(n).padStart(3, '0')}_${panel.id}`))
    }

    const housing = extractGroup(solid, HOUSING_MATERIAL_INDEX)
    if (housing) {
      housing.applyMatrix4(matrix)
      frame.add(housing)
    }

    if (supplyEdge !== null) {
      const box = buildPowerSupplyGeometry({ type: panel.panelType, edge: supplyEdge })
      box.applyMatrix4(matrix)
      supplies.add(box)
    }

    solid.dispose()
  })

  // Both printed pieces per station, merged — they take one material between
  // them, and the kit's per-type colouring is a screen affordance, not geometry.
  const kit = getConnectorKit(config, chain, connectors)
  for (const st of kit.stations) {
    const { position, quaternion } = connectorTransform(st)
    const matrix = new THREE.Matrix4().compose(position, quaternion, new THREE.Vector3(1, 1, 1))

    const back = buildConnectorGeometry(st)
    back.applyMatrix4(matrix)
    parts.add(back)

    if (st.barWidthCm) {
      const bar = buildFrontBarGeometry(st.barWidthCm, st.lengthCm)
      bar.applyMatrix4(matrix)
      parts.add(bar)
    }
  }

  // Emitted only when non-empty: an empty `o frame` block is a trap in an
  // importer, and 'none' supply mode should leave no supply object at all.
  if (!frame.isEmpty()) group.add(meshOf(frame.build(), GROUP_FRAME))
  if (!parts.isEmpty()) group.add(meshOf(parts.build(), GROUP_CONNECTORS))
  if (!supplies.isEmpty()) group.add(meshOf(supplies.build(), GROUP_SUPPLIES))

  if (options.includeEnvironment) addEnvironment(group, config, chain)

  group.updateMatrixWorld(true)
  return group
}

/** sRGB triplet for an MTL colour statement. MTL predates colour management, and
 *  every reader treats these as display-referred, so the linear working values
 *  three.js stores internally are converted back on the way out. */
function mtlRGB(color) {
  const c = color.clone().convertLinearToSRGB()
  return `${c.r.toFixed(6)} ${c.g.toFixed(6)} ${c.b.toFixed(6)}`
}

/**
 * Serialize the material library that the OBJ's `usemtl` lines refer to.
 *
 * Derived from the scene group, so the set of `newmtl` blocks is by construction
 * the set of materials the geometry actually uses. `Ke` (emissive) is not in the
 * original Wavefront spec but is read by Blender, Maya, Cinema 4D and every
 * other target that matters here; a reader that ignores it still gets a valid
 * material to split the mesh on, which is the part that was broken.
 *
 * @returns {string} MTL text, one `newmtl` block per exported object
 */
export function mtlPayloadV4(config, chain, connectors) {
  return mtlFromGroup(buildSceneGroup(config, chain, connectors))
}

/** The library for an already-built group, so the pair builder can walk the
 *  scene once instead of solving it twice. */
function mtlFromGroup(group) {
  const lines = [
    '# grid-designer v4 — material library',
    '# One material per exported object. The diffusers are emissive (Ke) and',
    '# individual so each panel can be given its own brightness.',
    '',
  ]
  for (const mesh of group.children) {
    const m = mesh.material
    // Emission is folded into Ke: MTL has no strength multiplier, so the
    // intensity that glTF carries as KHR_materials_emissive_strength has to be
    // baked here or silently dropped. Clamped, because Ke > 1 is meaningless.
    const ke = m.emissive
      ? m.emissive.clone().multiplyScalar(Math.min(m.emissiveIntensity ?? 1, 1))
      : new THREE.Color(0, 0, 0)
    lines.push(
      `newmtl ${m.name}`,
      `Kd ${mtlRGB(m.color)}`,
      'Ka 0.000000 0.000000 0.000000',
      `Ks ${(0.5 * (1 - (m.roughness ?? 1))).toFixed(6)} ${(0.5 * (1 - (m.roughness ?? 1))).toFixed(6)} ${(0.5 * (1 - (m.roughness ?? 1))).toFixed(6)}`,
      `Ns ${(Math.max(1 - (m.roughness ?? 1), 0) ** 2 * 900 + 2).toFixed(6)}`,
      'd 1.000000',
      // illum 2 = colour on, ambient on, specular on. The reader needs a lighting
      // model declared or some importers skip the block entirely.
      'illum 2',
      `Ke ${mtlRGB(ke)}`,
      '',
    )
  }
  return lines.join('\n')
}

/**
 * Serialize to OBJ text. Headless — no DOM — so the grouping is testable.
 *
 * `mtlName` is written into the `mtllib` line and MUST be the filename the
 * companion `mtlPayloadV4` output is actually saved under, in the same folder.
 * A `mtllib` naming a file that is not there is the same failure as no `mtllib`
 * at all, so the UI derives both names from one stamp rather than typing them.
 *
 * @returns {string} OBJ text: a `mtllib`, then one `o` block per diffuser, then
 *   the merged groups
 */
export function objPayloadV4(config, chain, connectors, mtlName = DEFAULT_MTL_NAME) {
  const body = new OBJExporter().parse(buildSceneGroup(config, chain, connectors))
  return `mtllib ${mtlName}\n${body}`
}

/**
 * Both halves of the OBJ export, named consistently, from ONE solve.
 *
 * This is what the UI should call: the two files are not independently useful —
 * an OBJ whose `mtllib` names a file the user did not save is back to the
 * original bug — so they are produced together and their names derived from one
 * basename rather than assembled twice at the call site.
 *
 * @param {string} basename filename stem, no extension
 * @returns {{objName: string, mtlName: string, obj: string, mtl: string}}
 */
export function objMtlPairV4(config, chain, connectors, basename = 'drop-ceiling') {
  const group = buildSceneGroup(config, chain, connectors)
  const mtlName = `${basename}.mtl`
  return {
    objName: `${basename}.obj`,
    mtlName,
    obj: `mtllib ${mtlName}\n${new OBJExporter().parse(group)}`,
    mtl: mtlFromGroup(group),
  }
}

/**
 * The object names this export will produce, in order. Cheap enough to call for
 * a UI hint, and it is what the export tests assert against.
 */
export function objObjectNames(config, chain, connectors) {
  return buildSceneGroup(config, chain, connectors).children.map((m) => m.name)
}
