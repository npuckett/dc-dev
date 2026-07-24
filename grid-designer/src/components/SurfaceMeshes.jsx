/**
 * grid-designer — the folded surface: one mesh per placed panel.
 *
 * Geometry is memoized per panel TYPE only ('2x2' and '2x4') — exactly two
 * BufferGeometries for the whole scene. A horizontal rect is the same '2x4'
 * solid; its +90° yaw is already baked into the quaternion `solveLayout`
 * produced, so nothing extra is needed here.
 *
 * Materials read as light fixtures rather than plates: warm near-white with a
 * gentle emissive lift for the square panels, a cooler tint for the 60×121
 * rects so the modularity of the surface is legible at a glance.
 */

import { useMemo } from 'react'
import useStore, { getDerived } from '../store.js'
import { buildPanelGeometry } from '../geometry/panelGeometry.js'

/** Lazily built, then reused for the life of the page. */
const geometryCache = new Map()

function geometryFor(type) {
  if (!geometryCache.has(type)) {
    geometryCache.set(
      type,
      buildPanelGeometry({ type, sidedness: 'single', powerSupplyEdge: 0 }),
    )
  }
  return geometryCache.get(type)
}

const SQUARE = {
  color: '#f6efe2',
  emissive: '#ffd9a0',
  emissiveIntensity: 0.42,
}
const RECT = {
  color: '#e6eef8',
  emissive: '#8fb9ff',
  emissiveIntensity: 0.36,
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
            <meshStandardMaterial
              color={look.color}
              emissive={look.emissive}
              emissiveIntensity={look.emissiveIntensity}
              roughness={0.55}
              metalness={0.05}
            />
          </mesh>
        )
      })}
    </group>
  )
}
