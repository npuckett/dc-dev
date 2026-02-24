/**
 * ReferenceLines — Renders dashed lines between related objects
 */

import React, { useMemo } from 'react'
import * as THREE from 'three'
import { Text } from '@react-three/drei'
import useStore from '../store'

function getObjectCenter(obj) {
  if (obj.type === 'zone' && obj.bounds) {
    return [
      (obj.bounds.x[0] + obj.bounds.x[1]) / 2,
      (obj.bounds.y[0] + obj.bounds.y[1]) / 2,
      (obj.bounds.z[0] + obj.bounds.z[1]) / 2,
    ]
  }
  return obj.position || [0, 0, 0]
}

function ReferenceLine({ ref_, objects }) {
  const fromObj = objects.find(o => o.id === ref_.from)
  const toObj = objects.find(o => o.id === ref_.to)
  if (!fromObj || !toObj) return null

  const from = getObjectCenter(fromObj)
  const to = getObjectCenter(toObj)
  const mid = [
    (from[0] + to[0]) / 2,
    (from[1] + to[1]) / 2 + 5,
    (from[2] + to[2]) / 2,
  ]

  const points = useMemo(() => [
    new THREE.Vector3(...from),
    new THREE.Vector3(...to),
  ], [from[0], from[1], from[2], to[0], to[1], to[2]])

  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry().setFromPoints(points)
    return g
  }, [points])

  return (
    <group>
      <line geometry={geometry}>
        <lineDashedMaterial
          color="#606090"
          dashSize={6}
          gapSize={4}
          linewidth={1}
        />
      </line>
      <Text
        position={mid}
        fontSize={6}
        color="#707090"
        anchorX="center"
        anchorY="bottom"
      >
        {ref_.relation}
      </Text>
    </group>
  )
}

export default function ReferenceLines() {
  const references = useStore(s => s.scene.references)
  const objects = useStore(s => s.scene.objects)

  if (references.length === 0) return null

  return (
    <>
      {references.map(ref_ => (
        <ReferenceLine key={ref_.id} ref_={ref_} objects={objects} />
      ))}
    </>
  )
}
