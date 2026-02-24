/**
 * Context Export — Generates a structured natural-language summary
 * of the scene that can be pasted directly into an agent prompt.
 *
 * Groups objects by role, lists positions and relationships,
 * and formats everything as readable Markdown.
 */

import { ROLES } from '../schema'

export function exportSceneContext(scene) {
  const lines = []

  // Header
  lines.push(`Scene: ${scene.name} (${scene.units})`)
  if (scene.coordinate_system) {
    const cs = scene.coordinate_system
    lines.push(`Origin: ${cs.origin}`)
    lines.push(`Axes: X=${cs.x_axis}, Y=${cs.y_axis}, Z=${cs.z_axis}`)
  }
  lines.push('')

  // Group objects by role
  const byRole = {}
  for (const obj of scene.objects) {
    const role = obj.tags?.role || 'structural'
    if (!byRole[role]) byRole[role] = []
    byRole[role].push(obj)
  }

  // Output each role group
  const roleOrder = ['actuator', 'sensor', 'zone', 'reference', 'structural']
  for (const role of roleOrder) {
    const objs = byRole[role]
    if (!objs || objs.length === 0) continue

    const label = role.charAt(0).toUpperCase() + role.slice(1) + 's'
    lines.push(`${label} (${objs.length}):`)

    for (const obj of objs) {
      const parts = [`  ${obj.name}`]

      // Position / bounds
      if (obj.type === 'zone' && obj.bounds) {
        const b = obj.bounds
        parts.push(`x[${b.x[0]}, ${b.x[1]}] y[${b.y[0]}, ${b.y[1]}] z[${b.z[0]}, ${b.z[1]}]`)
      } else if (obj.position) {
        parts.push(`at (${obj.position.join(', ')})`)
      }

      // Type
      parts.push(`[${obj.type}]`)

      // Dimensions
      if (obj.dimensions) {
        parts.push(`size: ${obj.dimensions.join('×')}`)
      }

      // Mesh info
      if (obj.type === 'mesh' && obj.geometry) {
        parts.push(`mesh: ${obj.geometry.vertexCount}v/${obj.geometry.faceCount}f`)
        if (obj.geometry.size) {
          parts.push(`bbox: ${obj.geometry.size.map(v => Math.round(v)).join('×')}`)
        }
      }

      // Sensor info
      if (obj.type === 'sensor') {
        parts.push(`range: ${obj.range || 200}, angle: ${obj.angle || 60}°`)
      }

      // Camera info
      if (obj.type === 'camera') {
        parts.push(`fov: ${obj.fov || 70}°, range: ${obj.range || 500}, aspect: ${obj.aspect || 1.78}`)
      }

      // Extra tags (excluding role)
      const extraTags = Object.entries(obj.tags || {})
        .filter(([k]) => k !== 'role')
      if (extraTags.length > 0) {
        parts.push(`{${extraTags.map(([k, v]) => `${k}: ${v}`).join(', ')}}`)
      }

      lines.push(parts.join(' — '))
    }
    lines.push('')
  }

  // References
  if (scene.references && scene.references.length > 0) {
    lines.push('Relationships:')
    for (const ref of scene.references) {
      const fromObj = scene.objects.find(o => o.id === ref.from)
      const toObj = scene.objects.find(o => o.id === ref.to)
      const fromName = fromObj?.name || ref.from
      const toName = toObj?.name || ref.to
      lines.push(`  ${fromName} → ${ref.relation} → ${toName}`)
    }
    lines.push('')
  }

  // Summary stats
  lines.push('---')
  lines.push(`Total objects: ${scene.objects.length}`)
  lines.push(`Total relationships: ${scene.references?.length || 0}`)

  return lines.join('\n')
}
