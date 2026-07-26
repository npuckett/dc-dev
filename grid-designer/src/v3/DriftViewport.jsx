/**
 * grid-designer v3 — the drift-surface 3D scene.
 *
 * Reuses v2's camera / orbit / lighting / grid setup (src/components/Viewport.jsx)
 * and its two-material-per-panel approach (src/components/SurfaceMeshes.jsx) —
 * both files are UNTOUCHED, this is a parallel v3 module. Panel geometry itself
 * is the same `buildPanelGeometry` v2 uses (`'2x2'` / `'2x4'` are the same
 * strings v3 tiles carry — src/core/v3/tiling.js's `makeSquareTile` /
 * `makePlateTile`), so no geometry work was needed here, only placement math.
 *
 * IMPORTANT (carried over from SurfaceMeshes.jsx, do not "fix"): the housing
 * material's `emissiveIntensity` is a deliberate visibility FLOOR (0.16), not
 * decoration. Both scene lights sit above the surface, and a folded-up panel's
 * housing back faces away from both of them — with a purely reflective
 * material it crushes to solid black. Every colour mode below routes its
 * housing look through `deriveLook`, which hard-codes that floor regardless of
 * mode, so this can never regress silently.
 *
 * =============================================================================
 * WHAT IS DRAWN
 * =============================================================================
 *   - every placed tile as its panel solid (`buildPanelGeometry`), positioned by
 *     `tile.position` + `tile.quaternion` from `solveLayout` — exactly the
 *     fields v2 panels also carry, so the geometry pipeline is unchanged;
 *   - a COLOUR MODE selector (`type` / `gap` / `clearance` / `facet` —
 *     V3_SPEC.md §5, the tool's main analytical readout in 3D);
 *   - a translucent GHOST of the target drift surface (`sampleDriftMesh`,
 *     form.js), toggleable — how far the panels sit from what was asked for;
 *   - COLLISION highlighting: any tile in `report.collisions` gets a saturated
 *     red look AND a bright red OBB wireframe (the exact box `collide.js`
 *     tested), because a collision is a hard buildability failure;
 *   - the floor grid, a WINDOW/SHORE line at z = 0, and a translucent WALL
 *     plane at x = 0 — THE MIRROR: the camera looks in from the window (−Z),
 *     so +X runs to screen-LEFT here, and the wall (x = 0) reads as the RIGHT
 *     side of this 3D view. TilingMap.jsx's plan view is the opposite: wall on
 *     the LEFT. This trips everyone; see both files' headers.
 *   - the measuring box (W × H × D, from `layout.bounds`), toggleable.
 *
 * `camera` is a MODULE-LEVEL constant, not an inline object literal: r3f
 * re-applies the `camera` prop via `applyProps` whenever its object identity
 * changes, which would reset the user's orbit on every unrelated store
 * update (colour-mode toggle, a hover) if it were recreated per render. v2's
 * Viewport.jsx recreates it inline and gets away with it only because neither
 * `Viewport` nor its parent ever subscribes to the store directly; this file
 * subscribes in a sibling toolbar, so the constant is cheap insurance.
 */

import { useEffect, useMemo } from 'react'
import * as THREE from 'three'
import { Canvas } from '@react-three/fiber'
import { Grid, Html, OrbitControls } from '@react-three/drei'
import useStoreV3, { getDerived } from './store.js'
import { buildPanelGeometry } from '../geometry/panelGeometry.js'
import { tileOBB } from '../core/v3/placement.js'
import { buildTarget } from '../core/v3/target.js'
import { normalizeForm, sampleDriftMesh } from '../core/v3/form.js'
import { normalizeConfig } from '../core/v3/schema.js'

// -----------------------------------------------------------------------------
// Nominal framing — DEFAULT_CONFIG's 6×8 sheet, footprint 365×487cm centre.
// Deliberately NOT reactive to the live sheet size (v2's GRID_CENTER is the
// same kind of fixed nominal reference), so changing cols/rows never yanks
// the camera out from under an orbit the user is mid-drag on.
// -----------------------------------------------------------------------------
const GRID_CENTER = [182.5, 0, 243.5]
const SHORE_X0 = -20
const SHORE_X1 = 400
const CAMERA_PROPS = { position: [GRID_CENTER[0], 260, -360], fov: 45, near: 1, far: 6000 }

/** Colour mode catalogue — V3_SPEC.md §5's "main analytical value". */
const COLOR_MODES = [
  { id: 'type', label: 'type', hint: 'plates one colour, squares another' },
  { id: 'gap', label: 'gap', hint: "colour each tile by the worst joint deviation touching it, from the report" },
  { id: 'clearance', label: 'clearance', hint: 'colour by height above the floor, so grounded tiles read instantly' },
  { id: 'facet', label: 'facet', hint: "colour by the target surface's facet plane at the tile's centre" },
]

// -----------------------------------------------------------------------------
// Geometry cache — one BufferGeometry per panel TYPE, exactly v2's pattern.
// -----------------------------------------------------------------------------
const geometryCache = new Map()
function geometryFor(type) {
  if (!geometryCache.has(type)) geometryCache.set(type, buildPanelGeometry({ type }))
  return geometryCache.get(type)
}

// -----------------------------------------------------------------------------
// Looks — diffuser (bright, emissive) + housing (dark, emissive FLOOR 0.16)
// -----------------------------------------------------------------------------
function deriveLook(hex) {
  const base = new THREE.Color(hex)
  const housingColor = base.clone().multiplyScalar(0.32)
  const housingEmissive = base.clone().multiplyScalar(0.22)
  return {
    diffuser: {
      color: `#${base.getHexString()}`,
      emissive: `#${base.getHexString()}`,
      emissiveIntensity: 0.55,
      roughness: 0.35,
      metalness: 0,
    },
    housing: {
      color: `#${housingColor.getHexString()}`,
      emissive: `#${housingEmissive.getHexString()}`,
      // NOT decoration — see file header. Never let a colour mode drop this.
      emissiveIntensity: 0.16,
      roughness: 0.72,
      metalness: 0.4,
    },
  }
}

/** 'type' mode's two looks, matching v2 SurfaceMeshes.jsx's palette exactly. */
const SQUARE_LOOK = {
  diffuser: { color: '#fbf4e4', emissive: '#ffd6a0', emissiveIntensity: 0.6, roughness: 0.35, metalness: 0 },
  housing: { color: '#8b857a', emissive: '#2c2318', emissiveIntensity: 0.16, roughness: 0.72, metalness: 0.4 },
}
const RECT_LOOK = {
  diffuser: { color: '#e7f0fd', emissive: '#8fb9ff', emissiveIntensity: 0.55, roughness: 0.35, metalness: 0 },
  housing: { color: '#7c828c', emissive: '#1b2230', emissiveIntensity: 0.16, roughness: 0.72, metalness: 0.4 },
}
/** A collision is a hard buildability failure — saturated, impossible to miss. */
const COLLISION_LOOK = {
  diffuser: { color: '#ff4d4d', emissive: '#ff0000', emissiveIntensity: 0.9, roughness: 0.4, metalness: 0 },
  housing: { color: '#5a1414', emissive: '#3a0a0a', emissiveIntensity: 0.3, roughness: 0.6, metalness: 0.3 },
}

function lerpHex(hexA, hexB, t) {
  const a = new THREE.Color(hexA)
  const b = new THREE.Color(hexB)
  return `#${a.clone().lerp(b, THREE.MathUtils.clamp(t, 0, 1)).getHexString()}`
}

/** Green at/under tolerance, through amber, to red at 3× tolerance or worse. */
function gapColorHex(devCm, tolCm) {
  const tol = tolCm > 0 ? tolCm : 1
  const t = devCm / tol
  if (t <= 1) return lerpHex('#3fd17c', '#ffd24d', t)
  return lerpHex('#ffd24d', '#ff3b3b', Math.min(1, (t - 1) / 2))
}

/** Cool blue near the floor (grounded), warm orange at the assembly's peak. */
function clearanceColorHex(y, maxY) {
  const t = THREE.MathUtils.clamp(y / (maxY > 0 ? maxY : 1), 0, 1)
  return lerpHex('#3fa9ff', '#ff9d3f', t)
}

/** Cheap deterministic string→hue, so each facet gets a stable, distinct tint. */
function hashHue(str) {
  let h = 0
  for (let i = 0; i < str.length; i++) h = (h * 31 + str.charCodeAt(i)) >>> 0
  return (h % 360) / 360
}

/**
 * Per-tile look for the active colour mode. `target` is only built when
 * `colorMode === 'facet'` (the caller memoizes it) — building it costs a
 * 64×64 relaxation pass and there is no reason to pay that on every config
 * change when a cheaper mode is showing.
 */
function computeLooks(layout, report, colorMode, target) {
  const looks = new Map()

  if (colorMode === 'gap') {
    const worst = new Map()
    for (const j of report.joints) {
      worst.set(j.a, Math.max(worst.get(j.a) ?? 0, j.deviationCm))
      worst.set(j.b, Math.max(worst.get(j.b) ?? 0, j.deviationCm))
    }
    for (const t of layout.tiles) {
      looks.set(t.id, deriveLook(gapColorHex(worst.get(t.id) ?? 0, report.summary.gapToleranceCm)))
    }
    return looks
  }

  if (colorMode === 'clearance') {
    const maxY = layout.bounds.size[1]
    for (const t of layout.tiles) {
      looks.set(t.id, deriveLook(clearanceColorHex(t.minY ?? 0, maxY)))
    }
    return looks
  }

  if (colorMode === 'facet' && target) {
    const keyHex = new Map()
    for (const t of layout.tiles) {
      const u = t.uv.u0 + t.uv.uLen / 2
      const v = t.uv.v0 + t.uv.vLen / 2
      const key = target.facetIndexAt(u, v)
      if (!keyHex.has(key)) {
        keyHex.set(key, `#${new THREE.Color().setHSL(hashHue(key), 0.55, 0.6).getHexString()}`)
      }
      looks.set(t.id, deriveLook(keyHex.get(key)))
    }
    return looks
  }

  // 'type' (default / fallback)
  for (const t of layout.tiles) looks.set(t.id, t.type === '2x4' ? RECT_LOOK : SQUARE_LOOK)
  return looks
}

// -----------------------------------------------------------------------------
// The placed tiles
// -----------------------------------------------------------------------------
function OBBOutline({ tile, unitBox, color, opacity = 0.9 }) {
  const obb = tileOBB(tile)
  return (
    <lineSegments
      position={obb.center}
      quaternion={obb.quaternion}
      scale={[obb.halfExtents[0] * 2, obb.halfExtents[1] * 2, obb.halfExtents[2] * 2]}
      renderOrder={10}
    >
      <edgesGeometry args={[unitBox]} />
      <lineBasicMaterial color={color} toneMapped={false} transparent opacity={opacity} depthTest={false} />
    </lineSegments>
  )
}

function DriftSurface() {
  const config = useStoreV3((s) => s.config)
  const colorMode = useStoreV3((s) => s.colorMode)
  const hoveredTileId = useStoreV3((s) => s.hoveredTileId)
  const setHoveredTile = useStoreV3((s) => s.setHoveredTile)
  const { layout, report } = getDerived(config)

  const target = useMemo(
    () => (colorMode === 'facet' ? buildTarget(normalizeConfig(config)) : null),
    [colorMode, config],
  )
  const looks = useMemo(() => computeLooks(layout, report, colorMode, target), [layout, report, colorMode, target])
  const collisionSet = useMemo(() => {
    const s = new Set()
    for (const c of report.collisions) {
      s.add(c.a)
      s.add(c.b)
    }
    return s
  }, [report])

  const geometries = useMemo(() => ({ '2x2': geometryFor('2x2'), '2x4': geometryFor('2x4') }), [])
  const unitBox = useMemo(() => new THREE.BoxGeometry(1, 1, 1), [])
  useEffect(() => () => unitBox.dispose(), [unitBox])

  return (
    <group>
      {layout.tiles
        .filter((t) => t.position)
        .map((tile) => {
          const colliding = collisionSet.has(tile.id)
          const hovered = hoveredTileId === tile.id
          const look = colliding ? COLLISION_LOOK : looks.get(tile.id) ?? SQUARE_LOOK
          return (
            <group key={tile.id}>
              <mesh
                geometry={geometries[tile.type]}
                position={tile.position}
                quaternion={tile.quaternion}
                castShadow={false}
                receiveShadow={false}
                onPointerOver={(e) => {
                  e.stopPropagation()
                  setHoveredTile(tile.id)
                }}
                onPointerOut={() => setHoveredTile(null)}
              >
                <meshStandardMaterial attach="material-0" {...look.diffuser} />
                <meshStandardMaterial attach="material-1" {...look.housing} />
              </mesh>
              {colliding && <OBBOutline tile={tile} unitBox={unitBox} color="#ff2d2d" opacity={0.95} />}
              {hovered && !colliding && <OBBOutline tile={tile} unitBox={unitBox} color="#ffffff" opacity={0.5} />}
            </group>
          )
        })}
    </group>
  )
}

// -----------------------------------------------------------------------------
// Ghost of the target surface (form.js `sampleDriftMesh`), toggleable
// -----------------------------------------------------------------------------
const GHOST_RES = 56

function GhostSurface() {
  const config = useStoreV3((s) => s.config)
  const showGhost = useStoreV3((s) => s.showGhost)

  const geometry = useMemo(() => {
    const form = normalizeForm(config.form)
    const { positions, indices } = sampleDriftMesh(form, GHOST_RES, GHOST_RES)
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
    geo.setIndex(new THREE.BufferAttribute(indices, 1))
    geo.computeVertexNormals()
    return geo
  }, [config.form])

  const wireGeometry = useMemo(() => new THREE.WireframeGeometry(geometry), [geometry])

  useEffect(
    () => () => {
      geometry.dispose()
      wireGeometry.dispose()
    },
    [geometry, wireGeometry],
  )

  if (!showGhost) return null
  return (
    <group>
      <mesh geometry={geometry}>
        <meshStandardMaterial
          color="#8fb9ff"
          transparent
          opacity={0.09}
          side={THREE.DoubleSide}
          depthWrite={false}
          roughness={1}
          metalness={0}
        />
      </mesh>
      <lineSegments geometry={wireGeometry}>
        <lineBasicMaterial color="#bcd9ff" transparent opacity={0.32} toneMapped={false} depthWrite={false} />
      </lineSegments>
    </group>
  )
}

// -----------------------------------------------------------------------------
// Wall (x = 0) / Window-Shore (z = 0) — same conventions as v2's Viewport.jsx
// -----------------------------------------------------------------------------
const WALL_HEIGHT_CM = 250
const WALL_THICKNESS_CM = 1.5
const WALL_COLOR = '#3ad0c0'
const WALL_MIN_DEPTH_CM = 200
const WALL_DEPTH_MARGIN_CM = 40

function Wall() {
  const config = useStoreV3((s) => s.config)
  const { layout } = getDerived(config)
  const deepest = Number.isFinite(layout.bounds.max?.[2]) ? layout.bounds.max[2] : 0
  const depth = Math.max(deepest + WALL_DEPTH_MARGIN_CM, WALL_MIN_DEPTH_CM)

  return (
    <group>
      <mesh position={[-WALL_THICKNESS_CM / 2, WALL_HEIGHT_CM / 2, depth / 2]}>
        <boxGeometry args={[WALL_THICKNESS_CM, WALL_HEIGHT_CM, depth]} />
        <meshBasicMaterial color={WALL_COLOR} toneMapped={false} transparent opacity={0.1} depthWrite={false} />
      </mesh>
      <mesh position={[0, 0.4, depth / 2]}>
        <boxGeometry args={[2.4, 0.8, depth]} />
        <meshBasicMaterial color={WALL_COLOR} toneMapped={false} />
      </mesh>
      <mesh position={[0, WALL_HEIGHT_CM, depth / 2]}>
        <boxGeometry args={[1.6, 0.8, depth]} />
        <meshBasicMaterial color={WALL_COLOR} toneMapped={false} transparent opacity={0.45} />
      </mesh>
      <Html position={[-14, 120, depth * 0.55]} center distanceFactor={520} zIndexRange={[10, 0]}>
        <div className="wall-label">WALL</div>
      </Html>
    </group>
  )
}

function ShoreLine() {
  return (
    <group>
      <mesh position={[(SHORE_X0 + SHORE_X1) / 2, 0.4, 0]}>
        <boxGeometry args={[SHORE_X1 - SHORE_X0, 0.8, 2.4]} />
        <meshBasicMaterial color="#3fa9ff" toneMapped={false} />
      </mesh>
      <Html position={[GRID_CENTER[0], 4, -22]} center distanceFactor={420} zIndexRange={[10, 0]}>
        <div className="shore-label">WINDOW / SHORE</div>
      </Html>
    </group>
  )
}

// -----------------------------------------------------------------------------
// Measuring box — W × H × D from layout.bounds, toggleable (matches v2)
// -----------------------------------------------------------------------------
const BOUNDS_COLOR = '#aab4c8'
const BOUNDS_LABEL_OFFSET_CM = 20

function MeasuringBox() {
  const config = useStoreV3((s) => s.config)
  const showBounds = useStoreV3((s) => s.showBounds)
  const { layout } = getDerived(config)
  const { bounds } = layout
  const [w, h, d] = bounds.size

  const edges = useMemo(() => {
    const box = new THREE.BoxGeometry(Math.max(w, 1e-3), Math.max(h, 1e-3), Math.max(d, 1e-3))
    const geo = new THREE.EdgesGeometry(box)
    box.dispose()
    return geo
  }, [w, h, d])
  useEffect(() => () => edges.dispose(), [edges])

  if (!showBounds || w <= 0 || h <= 0 || d <= 0) return null

  const [minX, minY, minZ] = bounds.min
  const [maxX, maxY, maxZ] = bounds.max
  const cx = (minX + maxX) / 2
  const cy = (minY + maxY) / 2
  const cz = (minZ + maxZ) / 2
  const off = BOUNDS_LABEL_OFFSET_CM
  const cm = (v) => `${Math.round(v)} cm`

  return (
    <group>
      <lineSegments geometry={edges} position={[cx, cy, cz]}>
        <lineBasicMaterial color={BOUNDS_COLOR} toneMapped={false} transparent opacity={0.3} depthWrite={false} />
      </lineSegments>
      <Html position={[cx, maxY + off * 1.6, maxZ + off]} center distanceFactor={460} zIndexRange={[10, 0]}>
        <div className="bounds-label" data-testid="bounds-label-x">
          W {cm(w)}
        </div>
      </Html>
      <Html position={[maxX + off * 2, Math.max(cy, off * 1.5), cz * 0.55]} center distanceFactor={460} zIndexRange={[10, 0]}>
        <div className="bounds-label bounds-label-peak" data-testid="bounds-label-y">
          peak {cm(maxY)}
        </div>
      </Html>
      <Html position={[maxX + off * 2, minY + 2, cz]} center distanceFactor={460} zIndexRange={[10, 0]}>
        <div className="bounds-label" data-testid="bounds-label-z">
          D {cm(d)}
        </div>
      </Html>
    </group>
  )
}

// -----------------------------------------------------------------------------
// The <Canvas> — deliberately reads NO store state directly (see file header):
// keeps the camera / OrbitControls instance stable across every toolbar toggle.
// -----------------------------------------------------------------------------
function Scene() {
  return (
    <Canvas
      className="viewport-canvas"
      dpr={[1, 2]}
      gl={{ preserveDrawingBuffer: true, antialias: true }}
      camera={CAMERA_PROPS}
    >
      <color attach="background" args={['#0d0d14']} />

      <ambientLight intensity={0.55} />
      <hemisphereLight args={['#9fb4d8', '#1a1a24', 0.5]} />
      <directionalLight position={[260, 420, -260]} intensity={1.15} />
      <directionalLight position={[-200, 240, 500]} intensity={0.35} />

      <Grid
        position={[GRID_CENTER[0], 0, GRID_CENTER[2]]}
        args={[1200, 1200]}
        cellSize={61}
        cellThickness={0.6}
        cellColor="#262636"
        sectionSize={305}
        sectionThickness={1.1}
        sectionColor="#3b3b55"
        fadeDistance={2400}
        fadeStrength={1}
        followCamera={false}
        infiniteGrid={false}
      />

      <ShoreLine />
      <Wall />
      <GhostSurface />
      <DriftSurface />
      <MeasuringBox />

      <OrbitControls
        makeDefault
        target={[GRID_CENTER[0], 40, GRID_CENTER[2] * 0.7]}
        enableDamping
        dampingFactor={0.12}
        minDistance={60}
        maxDistance={2500}
      />
    </Canvas>
  )
}

// -----------------------------------------------------------------------------
// Toolbar — colour mode + ghost/bounds toggles. Lives OUTSIDE <Canvas> so its
// store subscriptions never touch the Scene's render tree.
// -----------------------------------------------------------------------------
function ViewportToolbar() {
  const colorMode = useStoreV3((s) => s.colorMode)
  const setColorMode = useStoreV3((s) => s.setColorMode)
  const showGhost = useStoreV3((s) => s.showGhost)
  const toggleGhost = useStoreV3((s) => s.toggleGhost)
  const showBounds = useStoreV3((s) => s.showBounds)
  const toggleBounds = useStoreV3((s) => s.toggleBounds)

  return (
    <div className="viewport-toolbar" data-testid="viewport-toolbar">
      <span className="viewport-toolbar-label">colour</span>
      <div className="color-mode-group" role="group" aria-label="colour mode">
        {COLOR_MODES.map((m) => (
          <button
            key={m.id}
            type="button"
            className={`preset-btn${colorMode === m.id ? ' preset-active' : ''}`}
            data-testid={`colormode-${m.id}`}
            title={m.hint}
            onClick={() => setColorMode(m.id)}
          >
            {m.label}
          </button>
        ))}
      </div>
      <button
        type="button"
        className={`tool-btn${showGhost ? ' tool-btn-on' : ''}`}
        data-testid="toggle-ghost"
        title="translucent preview of the target drift surface (form.js), for judging how far the panels sit from what was asked"
        onClick={() => toggleGhost()}
      >
        ghost surface
      </button>
      <button
        type="button"
        className={`tool-btn${showBounds ? ' tool-btn-on' : ''}`}
        data-testid="toggle-bounds"
        title="overall measuring box (width × peak height × depth)"
        onClick={() => toggleBounds()}
      >
        bounds
      </button>
    </div>
  )
}

export default function DriftViewport() {
  return (
    <>
      <ViewportToolbar />
      <Scene />
    </>
  )
}
