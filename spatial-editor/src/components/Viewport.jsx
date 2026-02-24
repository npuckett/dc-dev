/**
 * Viewport — Three.js canvas with orbit controls, grid, and scene objects
 */

import React, { useRef, useCallback, Suspense } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, GizmoHelper, GizmoViewport, Grid } from '@react-three/drei'
import useStore from '../store'
import SceneObjects from './SceneObjects'
import ReferenceLines from './ReferenceLines'
import TransformWrapper from './TransformWrapper'

export default function Viewport() {
  const deselect = useStore(s => s.deselect)
  const controlsRef = useRef()

  const handlePointerMissed = useCallback((e) => {
    // Click on empty space → deselect
    if (e.type === 'click') {
      deselect()
    }
  }, [deselect])

  return (
    <Canvas
      camera={{ position: [300, 250, 400], fov: 50, near: 1, far: 10000 }}
      onPointerMissed={handlePointerMissed}
      gl={{ antialias: true }}
      style={{ background: '#12121e' }}
    >
      <ambientLight intensity={0.4} />
      <directionalLight position={[200, 400, 300]} intensity={0.6} />

      <Grid
        args={[2000, 2000]}
        cellSize={10}
        cellThickness={0.4}
        cellColor="#252540"
        sectionSize={100}
        sectionThickness={1}
        sectionColor="#303050"
        fadeDistance={1500}
        infiniteGrid
      />

      {/* Axis indicator at origin */}
      <axesHelper args={[50]} />

      <Suspense fallback={null}>
        <SceneObjects />
        <ReferenceLines />
        <TransformWrapper />
      </Suspense>

      <OrbitControls
        ref={controlsRef}
        makeDefault
        enableDamping
        dampingFactor={0.1}
        minDistance={20}
        maxDistance={3000}
      />

      <GizmoHelper alignment="bottom-right" margin={[60, 60]}>
        <GizmoViewport labelColor="white" axisHeadScale={0.8} />
      </GizmoHelper>
    </Canvas>
  )
}
