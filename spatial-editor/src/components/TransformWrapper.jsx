/**
 * TransformWrapper — Shows transform gizmo on selected object
 */

import React, { useRef, useCallback, useEffect } from 'react'
import { TransformControls } from '@react-three/drei'
import * as THREE from 'three'
import useStore from '../store'

export default function TransformWrapper() {
  const selectedId = useStore(s => s.selectedId)
  const objects = useStore(s => s.scene.objects)
  const transformMode = useStore(s => s.transformMode)
  const updateObjectPosition = useStore(s => s.updateObjectPosition)
  const updateObjectRotation = useStore(s => s.updateObjectRotation)
  const commitTransform = useStore(s => s.commitTransform)
  const transformRef = useRef()

  const selectedObj = selectedId ? objects.find(o => o.id === selectedId) : null

  // Zones use bounds, not position — skip transform gizmo for them
  if (!selectedObj || selectedObj.type === 'zone') return null

  return (
    <TransformControlsProxy
      key={selectedId}
      obj={selectedObj}
      mode={transformMode}
      onTransform={(pos, rot) => {
        if (pos) updateObjectPosition(selectedObj.id, pos)
        if (rot) updateObjectRotation(selectedObj.id, rot)
      }}
      onTransformEnd={() => commitTransform(selectedObj.id)}
    />
  )
}

function TransformControlsProxy({ obj, mode, onTransform, onTransformEnd }) {
  const ref = useRef()
  const groupRef = useRef()

  useEffect(() => {
    if (!ref.current) return
    const controls = ref.current

    function handleChange() {
      if (!groupRef.current) return
      const pos = groupRef.current.position
      const rot = groupRef.current.rotation
      onTransform(
        [pos.x, pos.y, pos.z],
        [
          THREE.MathUtils.radToDeg(rot.x),
          THREE.MathUtils.radToDeg(rot.y),
          THREE.MathUtils.radToDeg(rot.z),
        ]
      )
    }

    function handleEnd() {
      onTransformEnd()
    }

    controls.addEventListener('change', handleChange)
    controls.addEventListener('mouseUp', handleEnd)
    return () => {
      controls.removeEventListener('change', handleChange)
      controls.removeEventListener('mouseUp', handleEnd)
    }
  }, [onTransform, onTransformEnd])

  const pos = obj.position || [0, 0, 0]
  const rot = (obj.rotation || [0, 0, 0]).map(d => d * Math.PI / 180)

  return (
    <>
      <group ref={groupRef} position={pos} rotation={rot}>
        {/* Invisible target for transform controls */}
        <mesh visible={false}>
          <boxGeometry args={[1, 1, 1]} />
        </mesh>
      </group>
      <TransformControls
        ref={ref}
        object={groupRef}
        mode={mode}
        size={0.8}
        space="world"
      />
    </>
  )
}
