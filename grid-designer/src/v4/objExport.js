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
 *   spacers            every ground spacer post, merged
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
import { solveSpacers } from '../core/v4/spacers.js'

/** Object / material names. Kept as constants because they are the contract
 *  with whatever opens the file, not incidental strings. */
export const GROUP_FRAME = 'frame'
export const GROUP_CONNECTORS = 'connectors'
/** The posts under the floor-resting flat cells (core/v4/spacers.js). Their own
 *  group rather than part of `connectors`: they are a different part, in a
 *  different material, and merging them would make the connector object mean
 *  "printed parts and also some legs". */
export const GROUP_SPACERS = 'spacers'
export const GROUP_SUPPLIES = 'power_supplies'
export const DIFFUSER_PREFIX = 'diffuser'

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
  const mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ name }))
  mesh.name = name
  mesh.updateMatrixWorld(true)
  return mesh
}

/**
 * Build the export scene: individual diffusers, then the merged groups.
 *
 * ABSENT PANELS ARE NOT EXPORTED. `present: false` means the panel is not there;
 * the chain keeps its record so removal stays non-destructive, but the OBJ is a
 * description of what actually gets built.
 *
 * @param {object} config normalized v4 config
 * @param {object} chain `solveLattice` output
 * @param {object} connectors `solveConnectorsV4` output
 * @param {object} [spacers] `solveSpacers` output; solved here if omitted
 * @returns {THREE.Group}
 */
export function buildSceneGroup(config, chain, connectors, spacers = null) {
  const group = new THREE.Group()
  group.name = 'drop_ceiling'

  const frame = Merger()
  const supplies = Merger()
  const parts = Merger()
  const posts = Merger()
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

  // The ground spacers, merged. Built from the solver's OBB rather than from a
  // separate geometry module because a post IS its box — there is no profile to
  // loft — so a `buildSpacerGeometry` would be a second place to state the same
  // three numbers and a second place for them to drift.
  const S = spacers ?? solveSpacers(config, chain)
  for (const sp of S.spacers) {
    const [hx, hy, hz] = sp.obb.halfExtents
    const box = new THREE.BoxGeometry(hx * 2, hy * 2, hz * 2)
    box.applyMatrix4(new THREE.Matrix4().makeTranslation(...sp.obb.center))
    posts.add(box)
    box.dispose()
  }

  // Emitted only when non-empty: an empty `o frame` block is a trap in an
  // importer, 'none' supply mode should leave no supply object at all, and a
  // design with grounding off has no spacers to export.
  if (!frame.isEmpty()) group.add(meshOf(frame.build(), GROUP_FRAME))
  if (!parts.isEmpty()) group.add(meshOf(parts.build(), GROUP_CONNECTORS))
  if (!posts.isEmpty()) group.add(meshOf(posts.build(), GROUP_SPACERS))
  if (!supplies.isEmpty()) group.add(meshOf(supplies.build(), GROUP_SUPPLIES))

  group.updateMatrixWorld(true)
  return group
}

/**
 * Serialize to OBJ text. Headless — no DOM — so the grouping is testable.
 *
 * @returns {string} OBJ text: one `o` block per diffuser, then the merged groups
 */
export function objPayloadV4(config, chain, connectors, spacers = null) {
  return new OBJExporter().parse(buildSceneGroup(config, chain, connectors, spacers))
}

/**
 * The object names this export will produce, in order. Cheap enough to call for
 * a UI hint, and it is what the export tests assert against.
 */
export function objObjectNames(config, chain, connectors, spacers = null) {
  return buildSceneGroup(config, chain, connectors, spacers).children.map((m) => m.name)
}
