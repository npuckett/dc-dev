/**
 * grid-designer v4 — the folded-ribbon 3D scene.
 *
 * Ported from src/v3/DriftViewport.jsx, which is UNTOUCHED. Everything that was
 * not about the drift model came across unchanged: the camera / orbit / lighting
 * setup, the floor grid, the window-shore line, the wall plane, the measuring
 * box, the toolbar, and the two-material-per-panel treatment. What was dropped —
 * the ghost drift surface, facet colouring, tiling and plate anything — described
 * a model that no longer exists (HANDOFF §0).
 *
 * =============================================================================
 * THE MIRROR — DO NOT "FIX" IT
 * =============================================================================
 * The camera looks in FROM THE WINDOW, at −Z. World +X therefore runs to
 * screen-LEFT, and the wall (x = 0) reads as the RIGHT side of this view while
 * the window (z = 0) is at the bottom. This trips everyone who comes to it fresh
 * and has been "corrected" before; it is the convention the whole project states
 * its geometry in (V4_SPEC §0, "the world conventions") and it is correct.
 *
 * =============================================================================
 * THE HOUSING'S emissiveIntensity IS A VISIBILITY FLOOR, NOT DECORATION
 * =============================================================================
 * Carried over verbatim from DriftViewport, which carried it from v2's
 * SurfaceMeshes: both scene lights sit above the assembly, and a panel folded up
 * at 30° presents its housing BACK to neither of them — with a purely reflective
 * material it crushes to solid black. Every colour mode below routes its housing
 * look through `deriveLook`, which hard-codes the 0.16 floor regardless of mode,
 * so this cannot regress silently. v4 makes it matter MORE than v3 did: half the
 * units in a trapezoid wave are tilted, and the `fall` units face away.
 *
 * =============================================================================
 * `camera` IS A MODULE-LEVEL CONSTANT
 * =============================================================================
 * r3f re-applies the `camera` prop via `applyProps` whenever its object identity
 * changes, which would reset the user's orbit on every unrelated store update — a
 * colour-mode toggle, a hover — if it were recreated per render. This file
 * subscribes to the store in a sibling toolbar, so the constant is not optional
 * insurance here.
 *
 * =============================================================================
 * WHAT IS DRAWN
 * =============================================================================
 *   - every PRESENT unit's panel solid (`buildPanelGeometry`) at its
 *     `position` / `quaternion` from `solveChain`. Absent units are not drawn:
 *     they are holes in the ribbon, and the whole point of removal is to see one.
 *   - the connector parts from `solveConnectorsV4`'s stations, back half plus
 *     front bar, toggleable. A joint whose panels face opposite ways carries no
 *     station, so the gap is visibly EMPTY — which is the honest render of "no
 *     part of this family can span this".
 *   - the GROUND SPACERS from `solveSpacers`, toggleable: plain posts standing
 *     on the floor under the flat cells that rest on it. They are what holds the
 *     network at its ground clearance, so a view without them shows an assembly
 *     floating with nothing under it.
 *   - COLLISION highlighting: a saturated red look plus the exact OBB
 *     `collide.js` tested, because a collision is a hard buildability failure.
 *   - hover / selection outlines driven by the store, so the units table and the
 *     scene point at the same panel.
 */

import { useEffect, useMemo } from 'react'
import * as THREE from 'three'
import { Canvas } from '@react-three/fiber'
import { Grid, Html, OrbitControls } from '@react-three/drei'
import useStoreV4, { getDerived } from './store.js'
import { ADVISORY_FLAGS, getConnectorKit } from './exportAdapter.js'
import { buildPanelGeometry } from '../geometry/panelGeometry.js'
import { buildConnectorGeometry, buildFrontBarGeometry, connectorTransform } from '../geometry/connectorGeometry.js'

// -----------------------------------------------------------------------------
// Nominal framing — DEFAULT_CONFIG's nine-unit ribbon: 60cm wide, ~523cm deep.
// Deliberately NOT reactive to the live design (v3's GRID_CENTER is the same kind
// of fixed nominal reference), so changing the unit count never yanks the camera
// out from under an orbit the user is mid-drag on.
// -----------------------------------------------------------------------------
const GRID_CENTER = [30, 0, 260]
const SHORE_X0 = -60
const SHORE_X1 = 300
/**
 * The default view is a three-quarter from in front of the window, and the two
 * halves of that pull against each other:
 *
 *   - it must sit at −Z looking +Z, because that is what puts the window at the
 *     bottom of the screen and the wall (x = 0) on the right — THE MIRROR in the
 *     file header, and the spatial convention the whole project reads in;
 *   - but the ribbon is ~523cm deep against 60cm wide, so a camera on that axis
 *     foreshortens it into a stripe and the folds stop reading at all.
 *
 * So: far enough out in +X to open the profile up, close enough that the near
 * units still read. This position was arrived at by looking; pulling further
 * back to fit the whole run in frame ([330, 330, −430] was tried) only shrinks
 * everything and lets the wall plane dominate, because the depth is what needs
 * the room and moving away does not buy any.
 *
 * A three-quarter can only ever be a compromise on a 523 × 60 object. **Orbit
 * round to +X for a true profile** — that is the view the fold pattern reads
 * in, and it is one drag away.
 */
const CAMERA_PROPS = { position: [210, 250, -330], fov: 45, near: 1, far: 6000 }
const ORBIT_TARGET = [GRID_CENTER[0], 40, GRID_CENTER[2] * 0.9]

/**
 * Colour modes. Each answers a question the v4 model actually raises; v3's
 * `gap`, `clearance` and `facet` modes are gone because there is no gap
 * deviation to colour, no target surface to facet, and height reads directly off
 * a ribbon standing in front of you.
 */
const COLOR_MODES = [
  { id: 'role', label: 'role', hint: 'ground / high cells and the ramps between them' },
  { id: 'fold', label: 'fold', hint: 'colour each panel by the worst fold of the joints touching it' },
  { id: 'flip', label: 'flip', hint: 'a flipped panel is unmistakable — its lit face points the other way' },
  { id: 'flags', label: 'flags', hint: 'panels touching a flagged joint' },
]

// -----------------------------------------------------------------------------
// Geometry cache — one BufferGeometry per panel TYPE, exactly v3's pattern.
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
      // NOT decoration — see the file header. Never let a colour mode drop this.
      emissiveIntensity: 0.16,
      roughness: 0.72,
      metalness: 0.4,
    },
  }
}

/** The pattern's four roles, each its own colour. `base` and `high` are both
 *  flat and are deliberately far apart: they do different jobs (one lands on the
 *  floor, one is a branch anchor) and reading them as the same thing is the
 *  misunderstanding this mode exists to prevent. */
const ROLE_HEX = {
  ground: '#fbf4e4',
  high: '#8fb9ff',
  rise: '#8fd9a8',
  fall: '#f2b07a',
  // A wall anchor is a ramp with nothing on its far end, so it reads apart from
  // the ones that land on a cell — otherwise the braced edge is invisible.
  anchor: '#3ad0c0',
}
const ROLE_LOOKS = Object.fromEntries(Object.entries(ROLE_HEX).map(([k, v]) => [k, deriveLook(v)]))

/** A collision is a hard buildability failure — saturated, impossible to miss. */
const COLLISION_LOOK = {
  diffuser: { color: '#ff4d4d', emissive: '#ff0000', emissiveIntensity: 0.9, roughness: 0.4, metalness: 0 },
  housing: { color: '#5a1414', emissive: '#3a0a0a', emissiveIntensity: 0.3, roughness: 0.6, metalness: 0.3 },
}

const FLIP_UPRIGHT_LOOK = deriveLook('#e9e4d6')
const FLIP_TURNED_LOOK = deriveLook('#c46ff0')
const FLAG_CLEAN_LOOK = deriveLook('#7f9bb5')
const FLAG_DIRTY_LOOK = deriveLook('#ffb020')

function lerpHex(hexA, hexB, t) {
  const a = new THREE.Color(hexA)
  const b = new THREE.Color(hexB)
  return `#${a.clone().lerp(b, THREE.MathUtils.clamp(t, 0, 1)).getHexString()}`
}

/** Flat through cool to hot as the fold deepens, scaled against the connector's
 *  own limit rather than an arbitrary constant — so the colour means "how much
 *  of your allowance this joint spends". */
function foldColorHex(foldDeg, limitDeg) {
  const lim = limitDeg > 0 ? limitDeg : 45
  const t = Math.abs(foldDeg) / lim
  if (t <= 1) return lerpHex('#3fa9ff', '#ffd24d', t)
  return lerpHex('#ffd24d', '#ff3b3b', Math.min(1, t - 1))
}

/**
 * Per-unit look for the active colour mode.
 *
 * The fold and flags modes both need the JOINTS a unit touches, which the report
 * carries per joint rather than per unit — so both start by folding the joint
 * table down onto its two endpoints. That is a UI-layer derivation and it is the
 * right place for it: a joint's verdict genuinely belongs to the joint, and only
 * a renderer needs it smeared onto the panels.
 */
function computeLooks(chain, report, colorMode) {
  const looks = new Map()

  if (colorMode === 'fold') {
    const worst = new Map()
    for (const j of report.joints) {
      const d = j.dihedralDeg
      worst.set(j.a, Math.max(worst.get(j.a) ?? 0, d))
      worst.set(j.b, Math.max(worst.get(j.b) ?? 0, d))
    }
    const limit = report.envelope.maxAngleDeg ?? 45
    for (const u of chain.panels) looks.set(u.id, deriveLook(foldColorHex(worst.get(u.id) ?? 0, limit)))
    return looks
  }

  if (colorMode === 'flip') {
    for (const u of chain.panels) looks.set(u.id, u.flipped ? FLIP_TURNED_LOOK : FLIP_UPRIGHT_LOOK)
    return looks
  }

  if (colorMode === 'flags') {
    const dirty = new Set()
    for (const j of report.joints) {
      if (j.flags.length === 0 && !j.flipMismatch) continue
      dirty.add(j.a)
      dirty.add(j.b)
    }
    for (const u of chain.panels) looks.set(u.id, dirty.has(u.id) ? FLAG_DIRTY_LOOK : FLAG_CLEAN_LOOK)
    return looks
  }

  // 'role' (default / fallback)
  for (const u of chain.panels) looks.set(u.id, ROLE_LOOKS[u.anchor ? 'anchor' : u.role] ?? ROLE_LOOKS.ground)
  return looks
}

// -----------------------------------------------------------------------------
// The placed panels
// -----------------------------------------------------------------------------
function OBBOutline({ obb, unitBox, color, opacity = 0.9 }) {
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

function Ribbon() {
  const config = useStoreV4((s) => s.config)
  const colorMode = useStoreV4((s) => s.colorMode)
  const hoveredUnitId = useStoreV4((s) => s.hoveredUnitId)
  const selectedUnitId = useStoreV4((s) => s.selectedUnitId)
  const setHoveredUnit = useStoreV4((s) => s.setHoveredUnit)
  const setSelectedUnit = useStoreV4((s) => s.setSelectedUnit)
  const { chain, report } = getDerived(config)

  const looks = useMemo(() => computeLooks(chain, report, colorMode), [chain, report, colorMode])
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
      {chain.panels
        .filter((u) => u.present)
        .map((unit) => {
          const colliding = collisionSet.has(unit.id)
          const hovered = hoveredUnitId === unit.id
          const selected = selectedUnitId === unit.id
          const look = colliding ? COLLISION_LOOK : looks.get(unit.id) ?? ROLE_LOOKS.ground
          return (
            <group key={unit.id}>
              <mesh
                geometry={geometries[unit.panelType]}
                position={unit.position}
                quaternion={unit.quaternion}
                castShadow={false}
                receiveShadow={false}
                onPointerOver={(e) => {
                  e.stopPropagation()
                  setHoveredUnit(unit.id)
                }}
                onPointerOut={() => setHoveredUnit(null)}
                onClick={(e) => {
                  e.stopPropagation()
                  setSelectedUnit(unit.id)
                }}
              >
                <meshStandardMaterial attach="material-0" {...look.diffuser} />
                <meshStandardMaterial attach="material-1" {...look.housing} />
              </mesh>
              {colliding && <OBBOutline obb={unit.obb} unitBox={unitBox} color="#ff2d2d" opacity={0.95} />}
              {selected && !colliding && <OBBOutline obb={unit.obb} unitBox={unitBox} color="#8fd9ff" opacity={0.8} />}
              {hovered && !selected && !colliding && (
                <OBBOutline obb={unit.obb} unitBox={unitBox} color="#ffffff" opacity={0.5} />
              )}
            </group>
          )
        })}
    </group>
  )
}

// -----------------------------------------------------------------------------
// The printed connectors, toggleable
//
// Coloured BY KIT PART TYPE, which is the whole point of showing them — and on a
// v4 design the answer is usually two, where a drift needed dozens. That collapse
// is the pivot's result and it should be visible rather than only counted.
//
// A flagged part goes amber and a clashing one red, on the same reasoning as the
// panel collision highlight.
// -----------------------------------------------------------------------------
const CONNECTOR_FLAG_COLOR = '#ffb020'
const CONNECTOR_CLASH_COLOR = '#ff2d2d'
/** The front bar — neutral, because there are only ever one or two widths. */
const FRONT_BAR_COLOR = '#c8ccd6'

function ConnectorParts() {
  const config = useStoreV4((s) => s.config)
  const showConnectors = useStoreV4((s) => s.showConnectors)
  const { chain, connectors } = getDerived(config)
  const kit = getConnectorKit(config, chain, connectors)

  // Keyed on the kit's identity, which is memoized per config — so the meshes are
  // rebuilt only when the design actually changes, not on every unrelated store
  // update. Each part is a distinct solid, so unlike the panels there is no
  // per-type geometry to share.
  const parts = useMemo(() => {
    if (!showConnectors) return []
    // Golden-ratio hue rotation on the kit INDEX rather than a hash of the part
    // id: ids are sequential ('P00', 'P01', …) and a rolling hash maps
    // neighbours to hues a degree apart. Stepping by φ⁻¹ spreads any number of
    // types as far apart as they can go, which is the entire reason for
    // colouring by type.
    const hexOf = new Map()
    kit.kit.forEach((p, i) => {
      hexOf.set(p.partId, `#${new THREE.Color().setHSL((i * 0.6180339887) % 1, 0.62, 0.58).getHexString()}`)
    })
    const out = []
    for (const st of kit.stations) {
      const { position, quaternion } = connectorTransform(st)
      const clash = st.flags.includes('W_CONNECTOR_CLASH') || st.flags.includes('W_CONNECTOR_INFEASIBLE')
      const flagged = st.flags.some((f) => !ADVISORY_FLAGS.has(f))
      const tint = clash
        ? CONNECTOR_CLASH_COLOR
        : flagged
          ? CONNECTOR_FLAG_COLOR
          : hexOf.get(st.partId) ?? '#d0d0d8'
      out.push({
        id: `${st.id}-back`,
        geometry: buildConnectorGeometry(st),
        position,
        quaternion,
        color: tint,
        emphasis: clash || flagged,
      })
      if (st.barWidthCm) {
        out.push({
          id: `${st.id}-bar`,
          geometry: buildFrontBarGeometry(st.barWidthCm, st.lengthCm),
          position,
          quaternion,
          color: clash ? CONNECTOR_CLASH_COLOR : FRONT_BAR_COLOR,
          emphasis: clash,
        })
      }
    }
    return out
  }, [kit, showConnectors])

  // BufferGeometry is not garbage collected — it holds GPU buffers — so every
  // rebuild has to dispose the set it replaced.
  useEffect(() => () => parts.forEach((p) => p.geometry.dispose()), [parts])

  if (!showConnectors) return null

  return (
    <group>
      {parts.map((p) => (
        <mesh key={p.id} geometry={p.geometry} position={p.position} quaternion={p.quaternion}>
          <meshStandardMaterial
            color={p.color}
            emissive={p.color}
            emissiveIntensity={p.emphasis ? 0.55 : 0.22}
            roughness={0.55}
            metalness={0.1}
          />
        </mesh>
      ))}
    </group>
  )
}

// -----------------------------------------------------------------------------
// The ground spacers, toggleable
//
// Simple posts standing on the floor under the flat cells that rest on it
// (core/v4/spacers.js). Drawn from the SOLVER's records — position, height and
// section — rather than reconstructed here, so the thing on screen is the thing
// the report counted and the OBJ exports.
//
// Deliberately a plain box and a neutral colour: a spacer is not a part type,
// there is only ever one of it, and the kit colouring the connectors get would
// be inventing a distinction that does not exist.
// -----------------------------------------------------------------------------
const SPACER_COLOR = '#9aa6bb'

function SpacerPosts() {
  const config = useStoreV4((s) => s.config)
  const showSpacers = useStoreV4((s) => s.showSpacers)
  const { spacers } = getDerived(config)

  // One BoxGeometry for all of them, scaled per post: every spacer on a design
  // is the same size, and a geometry per post would be hundreds of GPU buffers
  // to say one thing.
  const unitBox = useMemo(() => new THREE.BoxGeometry(1, 1, 1), [])
  useEffect(() => () => unitBox.dispose(), [unitBox])

  if (!showSpacers || spacers.spacers.length === 0) return null

  return (
    <group>
      {spacers.spacers.map((sp) => (
        <mesh
          key={sp.id}
          geometry={unitBox}
          position={sp.obb.center}
          scale={[sp.obb.halfExtents[0] * 2, sp.obb.halfExtents[1] * 2, sp.obb.halfExtents[2] * 2]}
        >
          <meshStandardMaterial
            color={SPACER_COLOR}
            emissive={SPACER_COLOR}
            emissiveIntensity={0.18}
            roughness={0.6}
            metalness={0.15}
          />
        </mesh>
      ))}
    </group>
  )
}

// -----------------------------------------------------------------------------
// Wall (x = 0) / Window-Shore (z = 0) — the world conventions, unchanged
// -----------------------------------------------------------------------------
const WALL_HEIGHT_CM = 250
const WALL_THICKNESS_CM = 1.5
const WALL_COLOR = '#3ad0c0'
const WALL_MIN_DEPTH_CM = 200
const WALL_DEPTH_MARGIN_CM = 40

function Wall() {
  const config = useStoreV4((s) => s.config)
  const { chain } = getDerived(config)
  const deepest = Number.isFinite(chain.bounds.max?.[2]) ? chain.bounds.max[2] : 0
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
      <Html position={[GRID_CENTER[0], 4, -30]} center distanceFactor={420} zIndexRange={[10, 0]}>
        <div className="shore-label">WINDOW / SHORE</div>
      </Html>
    </group>
  )
}

// -----------------------------------------------------------------------------
// Obstacles — the room's structural column and anything like it
//
// Drawn like the wall rather than like a panel, because that is what it is: a
// fact about the room that the design has to be planned around, not part of the
// design. It turns RED when panels run through it, which is the one state where
// looking at the 3D view has to be enough to notice.
// -----------------------------------------------------------------------------
const OBSTACLE_COLOR = '#7f8aa3'
const OBSTACLE_HIT_COLOR = '#ff4d4d'

function Obstacles() {
  const config = useStoreV4((s) => s.config)
  const { report } = getDerived(config)

  return (
    <group>
      {report.obstacles.map((o) => {
        const hit = o.hitCount > 0
        const color = hit ? OBSTACLE_HIT_COLOR : OBSTACLE_COLOR
        const [w, h, d] = o.extents.size
        const c = o.extents.centre
        return (
          <group key={o.id}>
            <mesh position={c}>
              <boxGeometry args={[w, h, d]} />
              <meshBasicMaterial
                color={color}
                toneMapped={false}
                transparent
                opacity={hit ? 0.3 : 0.16}
                depthWrite={false}
              />
            </mesh>
            {/* A solid skirt at the base: the footprint is the part that
                actually constrains anything, and a translucent column alone
                reads as fog from most angles. */}
            <mesh position={[c[0], 1, c[2]]}>
              <boxGeometry args={[w, 2, d]} />
              <meshBasicMaterial color={color} toneMapped={false} />
            </mesh>
            <Html position={[c[0], h * 0.42, c[2]]} center distanceFactor={520} zIndexRange={[10, 0]}>
              <div className={`obstacle-label${hit ? ' obstacle-label-hit' : ''}`}>
                {o.label.toUpperCase()}
                {hit ? ` · ${o.hitCount} PANEL${o.hitCount === 1 ? '' : 'S'} THROUGH IT` : ''}
              </div>
            </Html>
          </group>
        )
      })}
    </group>
  )
}

// -----------------------------------------------------------------------------
// Measuring box — W × H × D from chain.bounds, toggleable
// -----------------------------------------------------------------------------
const BOUNDS_COLOR = '#aab4c8'
const BOUNDS_LABEL_OFFSET_CM = 20

function MeasuringBox() {
  const config = useStoreV4((s) => s.config)
  const showBounds = useStoreV4((s) => s.showBounds)
  const { chain } = getDerived(config)
  const { bounds } = chain
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

      <Obstacles />

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
      <Ribbon />
      <ConnectorParts />
      <SpacerPosts />
      <MeasuringBox />

      <OrbitControls
        makeDefault
        target={ORBIT_TARGET}
        enableDamping
        dampingFactor={0.12}
        minDistance={60}
        maxDistance={2500}
      />
    </Canvas>
  )
}

// -----------------------------------------------------------------------------
// Toolbar — colour mode + connector/bounds toggles. Lives OUTSIDE <Canvas> so its
// store subscriptions never touch the Scene's render tree.
// -----------------------------------------------------------------------------
function ViewportToolbar() {
  const colorMode = useStoreV4((s) => s.colorMode)
  const setColorMode = useStoreV4((s) => s.setColorMode)
  const showBounds = useStoreV4((s) => s.showBounds)
  const toggleBounds = useStoreV4((s) => s.toggleBounds)
  const showConnectors = useStoreV4((s) => s.showConnectors)
  const toggleConnectors = useStoreV4((s) => s.toggleConnectors)
  const showSpacers = useStoreV4((s) => s.showSpacers)
  const toggleSpacers = useStoreV4((s) => s.toggleSpacers)

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
        className={`tool-btn${showConnectors ? ' tool-btn-on' : ''}`}
        data-testid="toggle-connectors"
        title="the 3D-printed parts — back halves coloured by kit type, front bars in neutral grey; amber is flagged, red cannot be built or fouls something"
        onClick={() => toggleConnectors()}
      >
        connectors
      </button>
      <button
        type="button"
        className={`tool-btn${showSpacers ? ' tool-btn-on' : ''}`}
        data-testid="toggle-spacers"
        title="the ground spacers — posts standing under the flat cells that rest on the floor, holding them at the ground clearance"
        onClick={() => toggleSpacers()}
      >
        spacers
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

export default function RibbonViewport() {
  return (
    <>
      <ViewportToolbar />
      <Scene />
    </>
  )
}
