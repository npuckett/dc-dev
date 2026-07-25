/**
 * grid-designer — 3D viewport.
 *
 * Camera looks from the SHORE side (−Z, i.e. from the storefront window) back
 * into the room, so the surface reads the way a passer-by sees it: the flat
 * shore row nearest, the folded rows rising away behind it.
 *
 * World: cm, Y up. Columns along +X (0..365 for the default 6-column lattice at
 * the 61cm cell pitch = 60cm cell + 1cm joint gap), rows recede along +Z
 * (0..304). Grid centre ≈ (182.5, 0, 152).
 *
 * `preserveDrawingBuffer` is on so tests/screenshot.mjs can `drawImage()` the
 * WebGL canvas into a 2D canvas and sample pixels for a non-blank assertion.
 */

import { Canvas } from '@react-three/fiber'
import { Grid, Html, OrbitControls } from '@react-three/drei'
import SurfaceMeshes from './SurfaceMeshes.jsx'
import JointFlags from './JointFlags.jsx'

/** Nominal grid extent for the default 6×5 / 61cm-pitch config. */
const GRID_CENTER = [182.5, 0, 152]
const SHORE_X0 = -20
const SHORE_X1 = 390

export default function Viewport() {
  return (
    <Canvas
      className="viewport-canvas"
      dpr={[1, 2]}
      gl={{ preserveDrawingBuffer: true, antialias: true }}
      camera={{ position: [GRID_CENTER[0], 220, -320], fov: 45, near: 1, far: 6000 }}
    >
      <color attach="background" args={['#0d0d14']} />

      {/* --- lighting: soft, with the panels carrying their own emissive glow */}
      <ambientLight intensity={0.55} />
      <hemisphereLight args={['#9fb4d8', '#1a1a24', 0.5]} />
      <directionalLight position={[260, 420, -260]} intensity={1.15} />
      <directionalLight position={[-200, 240, 500]} intensity={0.35} />

      {/* --- ground plane / cm-scaled grid (one line per 61cm cell pitch) */}
      <Grid
        position={[GRID_CENTER[0], 0, GRID_CENTER[2]]}
        args={[976, 976]}
        cellSize={61}
        cellThickness={0.6}
        cellColor="#262636"
        sectionSize={305}
        sectionThickness={1.1}
        sectionColor="#3b3b55"
        fadeDistance={2200}
        fadeStrength={1}
        followCamera={false}
        infiniteGrid={false}
      />

      {/* --- shore / window indicator: blue line along z = 0 */}
      <mesh position={[(SHORE_X0 + SHORE_X1) / 2, 0.4, 0]}>
        <boxGeometry args={[SHORE_X1 - SHORE_X0, 0.8, 2.4]} />
        <meshBasicMaterial color="#3fa9ff" toneMapped={false} />
      </mesh>
      <Html
        position={[GRID_CENTER[0], 4, -22]}
        center
        distanceFactor={420}
        zIndexRange={[10, 0]}
      >
        <div className="shore-label">WINDOW / SHORE</div>
      </Html>

      {/* --- the surface itself */}
      <SurfaceMeshes />
      <JointFlags />

      <OrbitControls
        makeDefault
        target={[GRID_CENTER[0], 30, 150]}
        enableDamping
        dampingFactor={0.12}
        minDistance={60}
        maxDistance={2500}
      />
    </Canvas>
  )
}
