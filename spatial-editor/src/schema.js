/**
 * Spatial Scene Editor — Schema Definition
 *
 * Every object in a scene has geometry, a role, tags, and optional relationships.
 * This schema is designed to be:
 *  - Human-readable as a JSON file
 *  - Agent-parseable for spatial context
 *  - Extensible toward runtime behavior (post-V1)
 */

/**
 * Create a new empty scene
 */
export function createScene(name = 'Untitled Scene', units = 'centimeters') {
  return {
    name,
    version: '1.0',
    units,
    coordinate_system: {
      origin: 'User-defined',
      x_axis: 'Right',
      y_axis: 'Up',
      z_axis: 'Forward',
    },
    objects: [],
    references: [],
  }
}

let _nextId = 1
export function generateId() {
  return `obj_${_nextId++}_${Date.now().toString(36)}`
}

/**
 * Object types:
 *  - box:    3D volume (width, height, depth)
 *  - plane:  flat surface (width, height, no depth)
 *  - point:  position marker (rendered as sphere)
 *  - zone:   trigger volume defined by bounds (translucent wireframe)
 *  - group:  logical container for child objects
 *  - mesh:   imported OBJ geometry
 *  - sensor: directional sensor (cone shape — distance sensor, mic, etc.)
 *  - camera: camera with field-of-view frustum
 */
export const OBJECT_TYPES = ['box', 'plane', 'point', 'zone', 'group', 'mesh', 'sensor', 'camera']

/**
 * Roles determine how an object functions within an interactive system:
 *  - actuator:   something that outputs (light panel, speaker, motor)
 *  - sensor:     something that inputs (camera, microphone, distance sensor)
 *  - zone:       a spatial trigger region (presence detection area)
 *  - structural: physical structure with no interactive role (wall, column, floor)
 *  - reference:  a calibration or alignment marker
 */
export const ROLES = ['actuator', 'sensor', 'zone', 'structural', 'reference']

/**
 * Role → color mapping for viewport rendering
 */
export const ROLE_COLORS = {
  actuator: '#f0a030',   // amber
  sensor: '#3090f0',     // blue
  zone: '#30c070',       // green
  structural: '#808080', // gray
  reference: '#c050c0',  // purple
  default: '#a0a0a0',    // fallback
}

/**
 * Create a new scene object with sensible defaults
 */
export function createObject(type, overrides = {}) {
  const id = generateId()
  const base = {
    id,
    name: overrides.name || `${type}_${id.slice(4, 8)}`,
    type,
    position: [0, 0, 0],
    rotation: [0, 0, 0],
    tags: { role: type === 'zone' ? 'zone' : 'structural' },
    parent: null,
    children: [],
  }

  // Type-specific geometry defaults
  switch (type) {
    case 'box':
      base.dimensions = [100, 100, 100]
      break
    case 'plane':
      base.dimensions = [100, 100]
      break
    case 'point':
      base.radius = 5
      break
    case 'zone':
      base.bounds = {
        x: [-50, 50],
        y: [0, 100],
        z: [-50, 50],
      }
      break
    case 'group':
      break
    case 'mesh':
      base.geometry = { vertices: [], indices: [], vertexCount: 0, faceCount: 0, size: [0, 0, 0] }
      base.scale = [1, 1, 1]
      break
    case 'sensor':
      base.range = 200
      base.angle = 60
      base.tags = { role: 'sensor' }
      break
    case 'camera':
      base.fov = 70
      base.range = 500
      base.aspect = 1.78
      base.tags = { role: 'sensor' }
      break
  }

  return { ...base, ...overrides, id, tags: { ...base.tags, ...overrides.tags } }
}

/**
 * Create a reference (relationship) between two objects
 */
export function createReference(fromId, relation, toId) {
  return {
    id: generateId(),
    from: fromId,
    relation,
    to: toId,
  }
}

/**
 * Relation types used between objects
 */
export const RELATION_TYPES = [
  'observes',    // sensor → zone/object
  'triggers',    // zone → actuator
  'aligned_with', // object → object (spatial alignment)
  'contains',    // group/zone → object
  'controls',    // sensor/zone → actuator (future: behavior wiring)
]
