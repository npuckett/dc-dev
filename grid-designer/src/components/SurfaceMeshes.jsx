/**
 * grid-designer — the folded surface: one mesh per placed panel.
 *
 * Geometry is memoized per panel TYPE only ('2x2' and '2x4') — exactly two
 * BufferGeometries for the whole scene. A horizontal rect is the same '2x4'
 * solid; its +90° yaw is already baked into the quaternion `solveLayout`
 * produced, so nothing extra is needed here.
 *
 * TWO MATERIALS PER PANEL
 * =============================================================================
 * `buildPanelGeometry` emits two material groups, so every mesh carries two
 * materials attached by slot (r3f's `attach="material-0"` / `"material-1"`):
 *
 *   slot 0 — THE DIFFUSER (`DIFFUSER_MATERIAL_INDEX`): the recessed glowing
 *     panel, the light source itself. Bright and clearly emissive, nearly
 *     smooth, non-metallic. This is where the square-vs-rect colour distinction
 *     now lives — warm near-white for the 60×60 squares, a cooler blue for the
 *     60×121 plates — so plate modularity stays legible at a glance across a
 *     whole folded surface.
 *
 *   slot 1 — FRAME + HOUSING (`HOUSING_MATERIAL_INDEX`): the flange around the
 *     diffuser, the outer side wall, the tapered tray and its back plate. Matte
 *     mid-grey aluminium — much darker than the diffuser, rough, slightly
 *     metallic — so the frame reads as a crisp border around the glow and a
 *     folded-up panel's back reads as a solid tray rather than a lit plate. It is
 *     tinted per type too (a hair warmer / cooler to match its diffuser).
 *
 *     Its small `emissiveIntensity` is deliberate and NOT decoration: the tray
 *     back faces away from both directional lights, and with a purely reflective
 *     material it crushed to solid black — the closed tapered housing was there
 *     but unreadable, which is the very thing this material has to prove. The
 *     emissive is a visibility floor, low enough that the frame never competes
 *     with the diffuser and the array still reads as light fixtures, not plastic.
 */

import { useMemo } from 'react'
import useStore, { getDerived } from '../store.js'
import { buildPanelGeometry } from '../geometry/panelGeometry.js'

/** Lazily built, then reused for the life of the page. */
const geometryCache = new Map()

function geometryFor(type) {
  if (!geometryCache.has(type)) {
    geometryCache.set(type, buildPanelGeometry({ type }))
  }
  return geometryCache.get(type)
}

/** Warm near-white glow for the 60×60 squares. */
const SQUARE = {
  diffuser: {
    color: '#fbf4e4',
    emissive: '#ffd6a0',
    emissiveIntensity: 0.6,
    roughness: 0.35,
    metalness: 0,
  },
  housing: {
    color: '#8b857a',
    emissive: '#2c2318',
    emissiveIntensity: 0.16,
    roughness: 0.72,
    metalness: 0.4,
  },
}

/** Cooler blue glow for the 60×121 plates, so the modules stay countable. */
const RECT = {
  diffuser: {
    color: '#e7f0fd',
    emissive: '#8fb9ff',
    emissiveIntensity: 0.55,
    roughness: 0.35,
    metalness: 0,
  },
  housing: {
    color: '#7c828c',
    emissive: '#1b2230',
    emissiveIntensity: 0.16,
    roughness: 0.72,
    metalness: 0.4,
  },
}

export default function SurfaceMeshes() {
  const config = useStore((s) => s.config)
  const { layout } = getDerived(config)

  const geometries = useMemo(
    () => ({ '2x2': geometryFor('2x2'), '2x4': geometryFor('2x4') }),
    [],
  )

  return (
    <group>
      {layout.panels.map((panel) => {
        const look = panel.type === '2x4' ? RECT : SQUARE
        return (
          <mesh
            key={panel.id}
            geometry={geometries[panel.type]}
            position={panel.position}
            quaternion={panel.quaternion}
            castShadow={false}
            receiveShadow={false}
          >
            {/* slot 0 — the diffuser (geometry group 0) */}
            <meshStandardMaterial
              attach="material-0"
              color={look.diffuser.color}
              emissive={look.diffuser.emissive}
              emissiveIntensity={look.diffuser.emissiveIntensity}
              roughness={look.diffuser.roughness}
              metalness={look.diffuser.metalness}
            />
            {/* slot 1 — frame flange, side walls, tapered tray, back plate */}
            <meshStandardMaterial
              attach="material-1"
              color={look.housing.color}
              emissive={look.housing.emissive}
              emissiveIntensity={look.housing.emissiveIntensity}
              roughness={look.housing.roughness}
              metalness={look.housing.metalness}
            />
          </mesh>
        )
      })}
    </group>
  )
}
