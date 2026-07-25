/**
 * grid-designer — 3D viewport.
 *
 * Camera looks from the SHORE side (−Z, i.e. from the storefront window) back
 * into the room, so the surface reads the way a passer-by sees it: the front row
 * nearest, the folded rows rising away behind it.
 *
 * World: cm, Y up. Columns along +X (0..365 for the default 6-column lattice at
 * the 61cm cell pitch = 60cm cell + 1cm joint gap), rows recede along +Z
 * (0..304 at 5 rows, further at 6–8). Grid centre ≈ (182.5, 0, 152).
 *
 * Two room features are drawn, both non-interactive and deliberately quiet:
 *   - the WINDOW / SHORE line at z = 0, in blue;
 *   - the WALL closing the −X edge at x = `WALL_X` = 0 (column 0's outer edge), in
 *     teal — the surface a column with `endSupport: 'wall'` is bracketed to. It
 *     runs window → back over the surface's own depth, 250cm high. Because the
 *     camera looks in from the window (+X to the left), the wall appears on the
 *     RIGHT of the 3D view, beside the nearest-to-camera-right strip, column 0.
 *
 * One measuring aid is drawn on top of them: the MEASURING BOX (`BoundsBox`) — a
 * low-opacity wireframe around the whole surface with its width / peak height /
 * depth in centimetres, toggled by the toolbar's "bounds" button.
 *
 * `preserveDrawingBuffer` is on so tests/screenshot.mjs can `drawImage()` the
 * WebGL canvas into a 2D canvas and sample pixels for a non-blank assertion.
 */

import { useEffect, useMemo } from 'react'
import * as THREE from 'three'
import { Canvas, useThree } from '@react-three/fiber'
import { Grid, Html, OrbitControls } from '@react-three/drei'
import useStore, { getDerived } from '../store.js'
import { WALL_X } from '../core/schema.js'
import SurfaceMeshes from './SurfaceMeshes.jsx'
import JointFlags from './JointFlags.jsx'

/** Nominal grid extent for the default 6×5 / 61cm-pitch config. */
const GRID_CENTER = [182.5, 0, 152]
const SHORE_X0 = -20
const SHORE_X1 = 390

/** Wall: teal, translucent, and thin enough to read as a plane rather than a slab. */
const WALL_HEIGHT_CM = 250
const WALL_THICKNESS_CM = 1.5
const WALL_COLOR = '#3ad0c0'
/** Never draw a stub wall: even a shallow design gets a wall of at least this depth. */
const WALL_MIN_DEPTH_CM = 200
/** A little past the deepest panel, so the wall visibly runs behind the surface. */
const WALL_DEPTH_MARGIN_CM = 40

/**
 * The wall at the −X edge of the grid — x = `WALL_X` = 0, column 0's outer edge —
 * running from the window (z = 0) straight back. Its depth follows the SOLVED
 * surface (rows 5..8 change how far the strips reach), so it is its own
 * store-subscribing component rather than part of `Viewport` — re-rendering the
 * whole `<Canvas>` on every fold drag would throw away the camera.
 */
function Wall() {
  const config = useStore((s) => s.config)
  const { layout } = getDerived(config)

  let deepest = 0
  for (const chain of layout.columnChains ?? []) {
    for (const [, z] of chain.points ?? []) if (Number.isFinite(z) && z > deepest) deepest = z
  }
  const depth = Math.max(deepest + WALL_DEPTH_MARGIN_CM, WALL_MIN_DEPTH_CM)

  return (
    <group>
      {/* the plane itself — its INNER face sits exactly on x = WALL_X, the slab
          extending AWAY from the grid into −x */}
      <mesh position={[WALL_X - WALL_THICKNESS_CM / 2, WALL_HEIGHT_CM / 2, depth / 2]}>
        <boxGeometry args={[WALL_THICKNESS_CM, WALL_HEIGHT_CM, depth]} />
        <meshBasicMaterial
          color={WALL_COLOR}
          toneMapped={false}
          transparent
          opacity={0.1}
          depthWrite={false}
        />
      </mesh>
      {/* two rules — the foot and the top edge — so the plane's extent reads even
          at 10% opacity, without brightening the fill enough to fight the panels */}
      <mesh position={[WALL_X, 0.4, depth / 2]}>
        <boxGeometry args={[2.4, 0.8, depth]} />
        <meshBasicMaterial color={WALL_COLOR} toneMapped={false} />
      </mesh>
      <mesh position={[WALL_X, WALL_HEIGHT_CM, depth / 2]}>
        <boxGeometry args={[1.6, 0.8, depth]} />
        <meshBasicMaterial color={WALL_COLOR} toneMapped={false} transparent opacity={0.45} />
      </mesh>
      <Html
        position={[WALL_X - 14, 120, depth * 0.55]}
        center
        distanceFactor={520}
        zIndexRange={[10, 0]}
      >
        <div className="wall-label">WALL</div>
      </Html>
    </group>
  )
}

/**
 * THE MEASURING BOX — the design's overall extent, with its three dimensions in
 * centimetres.
 *
 * The box is `getDerived(config).bounds` (core/placement.js `layoutBounds`): the
 * axis-aligned extent of every panel's SOLID, housings included, so it is the
 * volume the built surface actually occupies and not just the lit faces. It is
 * drawn as a low-opacity wireframe on purpose — it is a ruler held up against the
 * design, so it has to be legible without competing with it, and the "bounds"
 * toolbar button hides it outright (`showBounds`, plain UI state).
 *
 * LABEL SIDES. All three pills are pushed onto the +X / −Z quadrant — the LEFT and
 * NEAR sides of the 3D view — because the two labels that already exist live
 * elsewhere: WINDOW / SHORE sits low at the middle of z = 0 and WALL sits at
 * x ≈ 0, the RIGHT of the view. So width rides the top front edge, and height and
 * depth ride the front-left vertical and bottom-left edges.
 *
 * The HEIGHT pill reports `max.y` — the PEAK — rather than the extent, and is the
 * emphasized one: every column starts on the floor, so the peak is the number a
 * design gets tuned against. (min.y is 0 for any grounded design, which makes the
 * two equal; they part company only if a future design floats.)
 */
const BOUNDS_COLOR = '#aab4c8'
/** How far outside the box the pills sit, cm — clear of the wireframe, still on it. */
const BOUNDS_LABEL_OFFSET_CM = 20

function BoundsBox() {
  const config = useStore((s) => s.config)
  const showBounds = useStore((s) => s.showBounds)
  const { bounds } = getDerived(config)
  const [w, h, d] = bounds.size

  const edges = useMemo(() => {
    const box = new THREE.BoxGeometry(Math.max(w, 1e-3), Math.max(h, 1e-3), Math.max(d, 1e-3))
    const geometry = new THREE.EdgesGeometry(box)
    box.dispose()
    return geometry
  }, [w, h, d])
  useEffect(() => () => edges.dispose(), [edges])

  if (!showBounds || w <= 0 || h <= 0 || d <= 0) return null

  const [, minY] = bounds.min
  const [maxX, maxY, maxZ] = bounds.max
  const [cx, cy, cz] = bounds.center
  const off = BOUNDS_LABEL_OFFSET_CM
  const cm = (v) => `${Math.round(v)} cm`

  return (
    <group>
      <lineSegments geometry={edges} position={[cx, cy, cz]}>
        <lineBasicMaterial
          color={BOUNDS_COLOR}
          toneMapped={false}
          transparent
          opacity={0.3}
          depthWrite={false}
        />
      </lineSegments>
      {/* width (X) — the top BACK edge, out in the empty sky above the deep end. The
          front top edge reads as the middle of the surface from this camera. */}
      <Html position={[cx, maxY + off * 1.6, maxZ + off]} center distanceFactor={460} zIndexRange={[10, 0]}>
        <div className="bounds-label" data-testid="bounds-label-x">
          W {cm(w)}
        </div>
      </Html>
      {/* peak height (Y) — off the +X (screen-left) face, high up beside the crest */}
      <Html
        position={[maxX + off * 2, Math.max(cy, off * 1.5), cz * 0.55]}
        center
        distanceFactor={460}
        zIndexRange={[10, 0]}
      >
        <div className="bounds-label bounds-label-peak" data-testid="bounds-label-y">
          peak {cm(maxY)}
        </div>
      </Html>
      {/* depth (Z) — the bottom edge of the same face, at the floor, mid-depth */}
      <Html
        position={[maxX + off * 2, minY + 2, cz]}
        center
        distanceFactor={460}
        zIndexRange={[10, 0]}
      >
        <div className="bounds-label" data-testid="bounds-label-z">
          D {cm(d)}
        </div>
      </Html>
    </group>
  )
}

/**
 * Automation hook: `window.__gridDesignerCamera([x,y,z], [tx,ty,tz])` parks the
 * camera, in the same spirit as `window.__gridDesignerStore` in App.jsx. It
 * exists so tests/screenshot.mjs can frame a CLOSE-UP of two or three adjacent
 * panels (the visual proof that the frame / diffuser / tapered back read
 * correctly) without simulating orbit drags, which are timing-dependent.
 *
 * It has to live inside `<Canvas>` because that is the only place the r3f camera
 * and the drei OrbitControls instance (registered by `makeDefault`) are
 * reachable. Setting `controls.target` and calling `update()` keeps damping from
 * dragging the view back.
 */
function CameraHook() {
  const camera = useThree((s) => s.camera)
  const controls = useThree((s) => s.controls)

  useEffect(() => {
    window.__gridDesignerCamera = (position, target) => {
      camera.position.set(position[0], position[1], position[2])
      if (controls?.target) {
        controls.target.set(target[0], target[1], target[2])
        controls.update()
      } else {
        camera.lookAt(target[0], target[1], target[2])
      }
      camera.updateProjectionMatrix()
      camera.updateMatrixWorld(true)
      return { position: camera.position.toArray(), target: [...target] }
    }
    return () => {
      delete window.__gridDesignerCamera
    }
  }, [camera, controls])

  return null
}

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

      {/* --- the wall closing the −X edge (what 'wall' end-support brackets to) */}
      <Wall />

      {/* --- the surface itself */}
      <SurfaceMeshes />
      <JointFlags />

      {/* --- the overall measuring box + its cm dimensions (toolbar-dismissible) */}
      <BoundsBox />

      <OrbitControls
        makeDefault
        target={[GRID_CENTER[0], 30, 150]}
        enableDamping
        dampingFactor={0.12}
        minDistance={60}
        maxDistance={2500}
      />
      <CameraHook />
    </Canvas>
  )
}
