/**
 * SceneObjects — Renders all objects in the scene as 3D meshes
 *
 * Color-coded by role:
 *   actuator → amber
 *   sensor   → blue
 *   zone     → green (translucent wireframe)
 *   structural → gray
 *   reference → purple
 */

import React, { useRef, useCallback, useMemo } from 'react'
import { Text } from '@react-three/drei'
import * as THREE from 'three'
import useStore from '../store'
import { ROLE_COLORS } from '../schema'

function getColor(obj) {
  const role = obj.tags?.role || 'default'
  return ROLE_COLORS[role] || ROLE_COLORS.default
}

/**
 * Individual object renderer — dispatches by type
 */
function SceneObject({ obj }) {
  const select = useStore(s => s.select)
  const selectedId = useStore(s => s.selectedId)
  const isSelected = selectedId === obj.id
  const color = getColor(obj)
  const meshRef = useRef()

  const handleClick = useCallback((e) => {
    e.stopPropagation()
    select(obj.id)
  }, [obj.id, select])

  const pos = obj.position || [0, 0, 0]
  const rot = (obj.rotation || [0, 0, 0]).map(d => d * Math.PI / 180)

  const commonProps = {
    ref: meshRef,
    position: pos,
    rotation: rot,
    onClick: handleClick,
  }

  return (
    <group>
      {renderByType(obj, commonProps, color, isSelected)}
      {/* Label */}
      <Text
        position={[pos[0], pos[1] + getLabelOffset(obj), pos[2]]}
        fontSize={8}
        color={isSelected ? '#ffffff' : '#808090'}
        anchorX="center"
        anchorY="bottom"
        outlineWidth={0.5}
        outlineColor="#000000"
      >
        {obj.name}
      </Text>
    </group>
  )
}

function getLabelOffset(obj) {
  if (obj.type === 'box') return (obj.dimensions?.[1] || 100) / 2 + 8
  if (obj.type === 'plane') return (obj.dimensions?.[1] || 100) / 2 + 8
  if (obj.type === 'zone') return (obj.bounds?.y?.[1] || 100) + 8
  if (obj.type === 'point') return (obj.radius || 5) + 8
  if (obj.type === 'mesh') {
    const s = obj.scale || [1, 1, 1]
    return ((obj.geometry?.size?.[1] || 100) / 2) * s[1] + 8
  }
  if (obj.type === 'sensor') return 10
  if (obj.type === 'camera') return 12
  return 15
}

function renderByType(obj, props, color, isSelected) {
  switch (obj.type) {
    case 'box':
      return <BoxObject obj={obj} color={color} isSelected={isSelected} {...props} />
    case 'plane':
      return <PlaneObject obj={obj} color={color} isSelected={isSelected} {...props} />
    case 'point':
      return <PointObject obj={obj} color={color} isSelected={isSelected} {...props} />
    case 'zone':
      return <ZoneObject obj={obj} color={color} isSelected={isSelected} />
    case 'group':
      return <GroupObject obj={obj} color={color} isSelected={isSelected} {...props} />
    case 'mesh':
      return <MeshObject obj={obj} color={color} isSelected={isSelected} {...props} />
    case 'sensor':
      return <SensorObject obj={obj} color={color} isSelected={isSelected} {...props} />
    case 'camera':
      return <CameraObject obj={obj} color={color} isSelected={isSelected} {...props} />
    default:
      return null
  }
}

function BoxObject({ obj, color, isSelected, ...props }) {
  const [w, h, d] = obj.dimensions || [100, 100, 100]
  return (
    <mesh {...props}>
      <boxGeometry args={[w, h, d]} />
      <meshStandardMaterial
        color={color}
        transparent
        opacity={isSelected ? 0.6 : 0.35}
        wireframe={false}
      />
      {/* Wireframe outline */}
      <lineSegments>
        <edgesGeometry args={[new THREE.BoxGeometry(w, h, d)]} />
        <lineBasicMaterial color={isSelected ? '#ffffff' : color} linewidth={1} />
      </lineSegments>
    </mesh>
  )
}

function PlaneObject({ obj, color, isSelected, ...props }) {
  const [w, h] = obj.dimensions || [100, 100]
  return (
    <mesh {...props}>
      <planeGeometry args={[w, h]} />
      <meshStandardMaterial
        color={color}
        transparent
        opacity={isSelected ? 0.7 : 0.5}
        side={THREE.DoubleSide}
      />
      {/* Wireframe outline */}
      <lineSegments>
        <edgesGeometry args={[new THREE.PlaneGeometry(w, h)]} />
        <lineBasicMaterial color={isSelected ? '#ffffff' : color} linewidth={1} />
      </lineSegments>
    </mesh>
  )
}

function PointObject({ obj, color, isSelected, ...props }) {
  const r = obj.radius || 5
  return (
    <mesh {...props}>
      <sphereGeometry args={[r, 16, 16]} />
      <meshStandardMaterial
        color={color}
        transparent
        opacity={isSelected ? 0.9 : 0.7}
        emissive={color}
        emissiveIntensity={isSelected ? 0.3 : 0.1}
      />
    </mesh>
  )
}

function ZoneObject({ obj, color, isSelected }) {
  const bounds = obj.bounds || { x: [-50, 50], y: [0, 100], z: [-50, 50] }
  const cx = (bounds.x[0] + bounds.x[1]) / 2
  const cy = (bounds.y[0] + bounds.y[1]) / 2
  const cz = (bounds.z[0] + bounds.z[1]) / 2
  const sx = bounds.x[1] - bounds.x[0]
  const sy = bounds.y[1] - bounds.y[0]
  const sz = bounds.z[1] - bounds.z[0]

  const select = useStore(s => s.select)
  const handleClick = useCallback((e) => {
    e.stopPropagation()
    select(obj.id)
  }, [obj.id, select])

  return (
    <group position={[cx, cy, cz]} onClick={handleClick}>
      {/* Translucent fill */}
      <mesh>
        <boxGeometry args={[sx, sy, sz]} />
        <meshStandardMaterial
          color={color}
          transparent
          opacity={isSelected ? 0.15 : 0.06}
        />
      </mesh>
      {/* Wireframe */}
      <lineSegments>
        <edgesGeometry args={[new THREE.BoxGeometry(sx, sy, sz)]} />
        <lineBasicMaterial color={isSelected ? '#ffffff' : color} />
      </lineSegments>
    </group>
  )
}

function GroupObject({ obj, color, isSelected, ...props }) {
  // Render as a small wireframe cube marker
  return (
    <mesh {...props}>
      <octahedronGeometry args={[8]} />
      <meshStandardMaterial
        color={color}
        wireframe
        transparent
        opacity={isSelected ? 0.8 : 0.4}
      />
    </mesh>
  )
}

function MeshObject({ obj, color, isSelected, ...props }) {
  const scale = obj.scale || [1, 1, 1]

  const geometry = useMemo(() => {
    if (!obj.geometry?.vertices?.length || !obj.geometry?.indices?.length) return null
    const geo = new THREE.BufferGeometry()
    const positions = new Float32Array(obj.geometry.vertices)
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geo.setIndex(obj.geometry.indices)
    geo.computeVertexNormals()
    return geo
  }, [obj.geometry])

  const edgesGeo = useMemo(() => {
    if (!geometry) return null
    return new THREE.EdgesGeometry(geometry, 30)
  }, [geometry])

  if (!geometry) return null

  return (
    <group {...props} scale={scale}>
      <mesh geometry={geometry}>
        <meshStandardMaterial
          color={color}
          transparent
          opacity={isSelected ? 0.6 : 0.35}
          side={THREE.DoubleSide}
        />
      </mesh>
      <lineSegments geometry={edgesGeo}>
        <lineBasicMaterial color={isSelected ? '#ffffff' : color} />
      </lineSegments>
    </group>
  )
}

function SensorObject({ obj, color, isSelected, ...props }) {
  const range = obj.range || 200
  const halfAngle = ((obj.angle || 60) / 2) * Math.PI / 180
  const radius = Math.tan(halfAngle) * range

  const coneGeo = useMemo(() => new THREE.ConeGeometry(radius, range, 16, 1, true), [radius, range])
  const edgesGeo = useMemo(() => new THREE.EdgesGeometry(coneGeo, 30), [coneGeo])

  return (
    <group {...props}>
      {/* Body — small sphere at the sensor origin */}
      <mesh>
        <sphereGeometry args={[4, 12, 12]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={isSelected ? 0.4 : 0.15} />
      </mesh>
      {/* Cone — detection range, pointing along -Z (forward) */}
      <group rotation={[Math.PI / 2, 0, 0]} position={[0, 0, -range / 2]}>
        <mesh geometry={coneGeo}>
          <meshStandardMaterial
            color={color}
            transparent
            opacity={isSelected ? 0.15 : 0.06}
            side={THREE.DoubleSide}
          />
        </mesh>
        <lineSegments geometry={edgesGeo}>
          <lineBasicMaterial color={isSelected ? '#ffffff' : color} transparent opacity={0.5} />
        </lineSegments>
      </group>
    </group>
  )
}

function CameraObject({ obj, color, isSelected, ...props }) {
  const fov = obj.fov || 70
  const range = obj.range || 500
  const aspect = obj.aspect || 1.78

  // Compute frustum dimensions at the far plane
  const halfFovRad = (fov / 2) * Math.PI / 180
  const farH = Math.tan(halfFovRad) * range
  const farW = farH * aspect

  const frustumLines = useMemo(() => {
    const pts = [
      // Near corners (small rect at origin)
      new THREE.Vector3(-2, -1.5, 0),
      new THREE.Vector3( 2, -1.5, 0),
      new THREE.Vector3( 2,  1.5, 0),
      new THREE.Vector3(-2,  1.5, 0),
      // Far corners
      new THREE.Vector3(-farW, -farH, -range),
      new THREE.Vector3( farW, -farH, -range),
      new THREE.Vector3( farW,  farH, -range),
      new THREE.Vector3(-farW,  farH, -range),
    ]
    const indices = [
      // Near rect
      0,1, 1,2, 2,3, 3,0,
      // Far rect
      4,5, 5,6, 6,7, 7,4,
      // Connecting edges
      0,4, 1,5, 2,6, 3,7,
    ]
    const geo = new THREE.BufferGeometry()
    const positions = new Float32Array(indices.length * 3)
    for (let i = 0; i < indices.length; i++) {
      positions[i * 3]     = pts[indices[i]].x
      positions[i * 3 + 1] = pts[indices[i]].y
      positions[i * 3 + 2] = pts[indices[i]].z
    }
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    return geo
  }, [farW, farH, range])

  return (
    <group {...props}>
      {/* Camera body — small box */}
      <mesh>
        <boxGeometry args={[6, 4, 8]} />
        <meshStandardMaterial
          color={color}
          transparent
          opacity={isSelected ? 0.8 : 0.6}
          emissive={color}
          emissiveIntensity={isSelected ? 0.3 : 0.1}
        />
      </mesh>
      {/* Lens indicator */}
      <mesh position={[0, 0, -5]}>
        <cylinderGeometry args={[2, 2.5, 3, 12]} />
        <meshStandardMaterial color={color} transparent opacity={isSelected ? 0.7 : 0.5} />
      </mesh>
      {/* FOV frustum */}
      <lineSegments geometry={frustumLines}>
        <lineBasicMaterial color={isSelected ? '#ffffff' : color} transparent opacity={isSelected ? 0.5 : 0.25} />
      </lineSegments>
    </group>
  )
}

/**
 * SceneObjects — Maps all scene objects to 3D representations
 */
export default function SceneObjects() {
  const objects = useStore(s => s.scene.objects)
  return (
    <>
      {objects.map(obj => (
        <SceneObject key={obj.id} obj={obj} />
      ))}
    </>
  )
}
