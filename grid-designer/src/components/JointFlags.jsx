/**
 * grid-designer — markers on the joints the report flagged.
 *
 *   red   → GAP_OUT_OF_TOL (a connector cannot physically span this joint)
 *   amber → SKEW only      (edges are within gap tolerance but not parallel)
 *
 * `jointReport` deliberately returns only measurements (no scene coordinates),
 * so the marker position is recomputed here from the two facing edges, using the
 * same `placement.js` helpers and the same edge/axis selection `report.js` uses
 * — the marker therefore sits exactly midway between the two measured edges.
 * Recomputed once per config change (memoized), not per frame.
 */

import { useMemo } from 'react'
import * as THREE from 'three'
import useStore, { getDerived } from '../store.js'
import { normalizeConfig } from '../core/schema.js'
import { cellEdgeWorld, negAxis, panelLocalAxes } from '../core/placement.js'

const MARKER_RADIUS = 4
const COLOR_GAP = '#ff3b30'
const COLOR_SKEW = '#ffb020'

/**
 * World-space midpoint of a joint: the mean of its two facing edges' midpoints.
 *
 * @param {object} joint a `jointReport` joint record
 * @param {Map<string, {panel: object, cellIndex: number}>} owner cell → panel
 * @param {object} cfg normalized config
 * @returns {THREE.Vector3 | null}
 */
function jointMidpoint(joint, owner, cfg) {
  const A = owner.get(`${joint.cellA[0]},${joint.cellA[1]}`)
  const B = owner.get(`${joint.cellB[0]},${joint.cellB[1]}`)
  if (!A || !B) return null

  const axA = panelLocalAxes(A.panel)
  const axB = panelLocalAxes(B.panel)

  let edgeA
  let edgeB
  if (joint.class === 'in-row') {
    edgeA = cellEdgeWorld(A.panel, A.cellIndex, axA.col, axA.row, cfg)
    edgeB = cellEdgeWorld(B.panel, B.cellIndex, negAxis(axB.col), axB.row, cfg)
  } else {
    edgeA = cellEdgeWorld(A.panel, A.cellIndex, axA.row, negAxis(axA.col), cfg)
    edgeB = cellEdgeWorld(B.panel, B.cellIndex, negAxis(axB.row), negAxis(axB.col), cfg)
  }

  return new THREE.Vector3()
    .add(edgeA[0])
    .add(edgeA[1])
    .add(edgeB[0])
    .add(edgeB[1])
    .multiplyScalar(0.25)
}

export default function JointFlags() {
  const config = useStore((s) => s.config)
  const { layout, report } = getDerived(config)

  const markers = useMemo(() => {
    const cfg = normalizeConfig(config)
    const owner = new Map()
    for (const panel of layout.panels) {
      panel.cells.forEach((cell, cellIndex) => {
        owner.set(`${cell[0]},${cell[1]}`, { panel, cellIndex })
      })
    }
    const out = []
    for (const joint of report.joints) {
      if (joint.ok) continue
      const p = jointMidpoint(joint, owner, cfg)
      if (!p) continue
      out.push({
        id: joint.id,
        position: [p.x, p.y, p.z],
        color: joint.flags.includes('GAP_OUT_OF_TOL') ? COLOR_GAP : COLOR_SKEW,
      })
    }
    return out
  }, [config, layout, report])

  return (
    <group>
      {markers.map((m) => (
        <mesh key={m.id} position={m.position}>
          <octahedronGeometry args={[MARKER_RADIUS, 0]} />
          <meshBasicMaterial color={m.color} toneMapped={false} transparent opacity={0.9} />
        </mesh>
      ))}
    </group>
  )
}
