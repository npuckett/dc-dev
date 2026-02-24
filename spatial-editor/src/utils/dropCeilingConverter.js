/**
 * Drop Ceiling Converter
 *
 * Converts IO/world_coordinates.json into the spatial editor scene schema.
 * This validates the schema against a real project and provides a demo scene.
 */

export function convertWorldCoordinates(wc) {
  const scene = {
    name: 'Drop Ceiling v1',
    version: '1.0',
    units: wc.units || 'centimeters',
    coordinate_system: {
      origin: wc.coordinate_system?.origin || 'User-defined',
      x_axis: wc.coordinate_system?.x_axis || 'Right',
      y_axis: wc.coordinate_system?.y_axis || 'Up',
      z_axis: wc.coordinate_system?.z_axis || 'Forward',
    },
    objects: [],
    references: [],
  }

  let refId = 1

  // --- Panel units and subpanels ---
  const units = wc.panels?.units || {}
  const subpanels = wc.panels?.subpanels?.local_positions || {}
  const panelWidth = wc.panels?.panel_width || 60
  const panelDepth = wc.panels?.panel_depth || 60

  for (const [unitIdx, unitData] of Object.entries(units)) {
    const unitId = `panel_unit_${unitIdx}`
    const unitCenter = unitData.center || [0, 0, 0]

    // Create group for unit
    const unitObj = {
      id: unitId,
      name: unitId,
      type: 'group',
      position: [...unitCenter],
      rotation: [0, 0, 0],
      tags: {
        role: 'actuator',
        description: unitData.description || '',
        unit_index: unitIdx,
      },
      parent: null,
      children: [],
    }

    // Create subpanels
    for (const [subIdx, subData] of Object.entries(subpanels)) {
      const panelId = `panel_${unitIdx}_${subIdx}`
      const channelNum = parseInt(unitIdx) * 3 + parseInt(subIdx)

      const panelObj = {
        id: panelId,
        name: panelId,
        type: 'plane',
        position: [unitCenter[0], subData.y || 0, subData.z || 0],
        rotation: [subData.angle_deg || 0, 0, 0],
        dimensions: [panelWidth, panelDepth],
        tags: {
          role: 'actuator',
          dmx_channel: String(channelNum),
          subpanel: subIdx,
          protocol: '0-10v',
        },
        parent: unitId,
        children: [],
      }

      unitObj.children.push(panelId)
      scene.objects.push(panelObj)
    }

    scene.objects.push(unitObj)
  }

  // --- Cameras ---
  const cameras = wc.cameras || {}
  for (const [camKey, camData] of Object.entries(cameras)) {
    if (typeof camData !== 'object' || !camData.position) continue

    const camId = camKey
    scene.objects.push({
      id: camId,
      name: camKey.replace('_', ' '),
      type: 'point',
      position: [...camData.position],
      rotation: [0, 0, 0],
      radius: 8,
      tags: {
        role: 'sensor',
        sensor_type: 'camera',
        model: cameras.model || '',
        fov_h: String(camData.fov?.horizontal || ''),
        fov_v: String(camData.fov?.vertical || ''),
        description: camData.description || '',
        coverage: camData.coverage || '',
      },
      parent: null,
      children: [],
    })
  }

  // --- Calibration markers ---
  const markers = wc.calibration_markers?.markers || {}
  for (const [mIdx, mData] of Object.entries(markers)) {
    const markerSize = wc.calibration_markers?.marker_size || 20
    scene.objects.push({
      id: `marker_${mIdx}`,
      name: `marker ${mIdx}`,
      type: mData.orientation === 'vertical' ? 'plane' : 'plane',
      position: [...mData.position],
      rotation: mData.orientation === 'horizontal' ? [90, 0, 0] : [0, 0, 0],
      dimensions: [markerSize, markerSize],
      tags: {
        role: 'reference',
        marker_id: mIdx,
        orientation: mData.orientation || 'horizontal',
        description: mData.description || '',
        visible_to: (mData.visible_to || []).join(', '),
      },
      parent: null,
      children: [],
    })
  }

  // --- Tracking zones ---
  const zones = wc.tracking_zones || {}
  for (const [zoneKey, zoneData] of Object.entries(zones)) {
    const zoneId = `${zoneKey}_zone`
    scene.objects.push({
      id: zoneId,
      name: `${zoneKey} zone`,
      type: 'zone',
      bounds: {
        x: zoneData.bounds?.x || [-50, 50],
        y: zoneData.bounds?.y || [0, 100],
        z: zoneData.bounds?.z || [-50, 50],
      },
      tags: {
        role: 'zone',
        trigger_type: 'presence',
        description: zoneData.description || '',
      },
      parent: null,
      children: [],
    })
  }

  // --- Wander box ---
  const wb = wc.light_behavior?.wander_box
  if (wb) {
    scene.objects.push({
      id: 'wander_box',
      name: 'wander box',
      type: 'zone',
      bounds: {
        x: wb.bounds?.x || [-280, -20],
        y: wb.bounds?.y || [0, 150],
        z: wb.bounds?.z || [-28, 32],
      },
      tags: {
        role: 'zone',
        trigger_type: 'movement_bounds',
        description: wb.description || 'Light movement boundary',
      },
      parent: null,
      children: [],
    })
  }

  // --- Reference levels as point markers ---
  const refLevels = wc.reference_levels || {}
  for (const [levelKey, levelData] of Object.entries(refLevels)) {
    scene.objects.push({
      id: `ref_${levelKey}`,
      name: `${levelKey} level`,
      type: 'point',
      position: [-150, levelData.y || 0, 0],
      rotation: [0, 0, 0],
      radius: 3,
      tags: {
        role: 'reference',
        description: levelData.description || '',
      },
      parent: null,
      children: [],
    })
  }

  // --- References (relationships) ---
  // Cameras observe tracking zones
  for (const camKey of Object.keys(cameras)) {
    if (typeof cameras[camKey] !== 'object' || !cameras[camKey].position) continue
    scene.references.push({
      id: `ref_${refId++}`,
      from: camKey,
      relation: 'observes',
      to: 'active_zone',
    })
    scene.references.push({
      id: `ref_${refId++}`,
      from: camKey,
      relation: 'observes',
      to: 'passive_zone',
    })
  }

  // Zones trigger actuator groups
  for (const unitIdx of Object.keys(units)) {
    scene.references.push({
      id: `ref_${refId++}`,
      from: 'active_zone',
      relation: 'triggers',
      to: `panel_unit_${unitIdx}`,
    })
  }

  return scene
}

/**
 * Hardcoded Drop Ceiling demo scene for when world_coordinates.json isn't available
 */
export const DROP_CEILING_WORLD_COORDINATES = {
  "version": "2.1",
  "units": "centimeters",
  "coordinate_system": {
    "origin": "Back right corner of Panel Unit 0, at floor level",
    "x_axis": "Negative toward Unit 3 (left when facing panels)",
    "y_axis": "Positive upward",
    "z_axis": "Positive forward into tracking zone (away from panels)"
  },
  "panels": {
    "unit_spacing": 80,
    "panel_width": 60,
    "panel_depth": 60,
    "units": {
      "0": { "center": [-30, 0, 0], "description": "Rightmost unit" },
      "1": { "center": [-110, 0, 0], "description": "Second from right" },
      "2": { "center": [-190, 0, 0], "description": "Second from left" },
      "3": { "center": [-270, 0, 0], "description": "Leftmost unit" }
    },
    "subpanels": {
      "local_positions": {
        "1": { "y": 120, "z": -23, "angle_deg": -30 },
        "2": { "y": 60, "z": 0, "angle_deg": 0 },
        "3": { "y": 0, "z": -23, "angle_deg": 30 }
      }
    }
  },
  "cameras": {
    "camera_1": {
      "position": [-30, -15, 78],
      "description": "Right camera",
      "coverage": "Right half of tracking zone",
      "fov": { "horizontal": 80, "vertical": 48 }
    },
    "camera_2": {
      "position": [-270, -15, 78],
      "description": "Left camera",
      "coverage": "Left half of tracking zone",
      "fov": { "horizontal": 80, "vertical": 48 }
    },
    "model": "Reolink RLC-520A"
  },
  "calibration_markers": {
    "marker_size": 20,
    "markers": {
      "0": { "position": [-30, -66, 168], "orientation": "horizontal", "description": "Right front", "visible_to": ["camera_1"] },
      "1": { "position": [-150, -66, 168], "orientation": "horizontal", "description": "Center front", "visible_to": ["camera_1", "camera_2"] },
      "2": { "position": [-270, -66, 168], "orientation": "horizontal", "description": "Left front", "visible_to": ["camera_2"] },
      "3": { "position": [-30, -66, 219], "orientation": "horizontal", "description": "Right back", "visible_to": ["camera_1"] },
      "4": { "position": [-270, -66, 219], "orientation": "horizontal", "description": "Left back", "visible_to": ["camera_2"] },
      "5": { "position": [-150, -15, 628], "orientation": "vertical", "description": "Subway wall", "visible_to": ["camera_1", "camera_2"] },
      "6": { "position": [-150, -66, 219], "orientation": "horizontal", "description": "Center back", "visible_to": ["camera_1", "camera_2"] }
    }
  },
  "tracking_zones": {
    "active": {
      "description": "Primary tracking area",
      "bounds": { "x": [-350, 50], "y": [-66, 234], "z": [78, 283] }
    },
    "passive": {
      "description": "Sidewalk passersby",
      "bounds": { "x": [-350, 50], "y": [-66, 234], "z": [283, 553] }
    }
  },
  "light_behavior": {
    "wander_box": {
      "description": "Light movement boundary",
      "bounds": { "x": [-280, -20], "y": [0, 150], "z": [-28, 32] }
    }
  },
  "reference_levels": {
    "floor": { "y": 0, "description": "Storefront floor level" },
    "street": { "y": -66, "description": "Street/sidewalk level" },
    "camera_ledge": { "y": -15, "description": "Camera mounting height" }
  }
}
