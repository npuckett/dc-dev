/**
 * OBJ Parser — Parses Wavefront .obj text into geometry data
 * for the spatial editor's mesh objects.
 *
 * Handles: vertices (v), normals (vn), faces (f) with fan triangulation.
 * Centers geometry at origin; returns centroid as the position offset.
 */

export function parseOBJ(text) {
  const rawVerts = []   // flat: [x, y, z, x, y, z, ...]
  const indices = []    // triangle vertex indices

  for (const raw of text.split('\n')) {
    const line = raw.trim()
    if (!line || line[0] === '#') continue
    const parts = line.split(/\s+/)

    switch (parts[0]) {
      case 'v':
        rawVerts.push(+parts[1] || 0, +parts[2] || 0, +parts[3] || 0)
        break
      case 'f': {
        const face = []
        for (let i = 1; i < parts.length; i++) {
          const tok = parts[i].split('/')[0]
          let vi = parseInt(tok)
          if (isNaN(vi)) continue
          // OBJ supports negative (relative) indices
          if (vi < 0) vi = rawVerts.length / 3 + vi
          else vi -= 1 // OBJ is 1-indexed
          face.push(vi)
        }
        // Fan triangulation for n-gons
        for (let i = 1; i < face.length - 1; i++) {
          indices.push(face[0], face[i], face[i + 1])
        }
        break
      }
      // We skip vt, vn, mtllib, usemtl, etc. — geometry only for v1
    }
  }

  if (rawVerts.length === 0) {
    throw new Error('No vertices found in OBJ file')
  }

  // Compute bounding box
  let minX = Infinity, minY = Infinity, minZ = Infinity
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity

  for (let i = 0; i < rawVerts.length; i += 3) {
    const x = rawVerts[i], y = rawVerts[i + 1], z = rawVerts[i + 2]
    if (x < minX) minX = x; if (x > maxX) maxX = x
    if (y < minY) minY = y; if (y > maxY) maxY = y
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z
  }

  // Centroid — used as the object's position in the scene
  const cx = (minX + maxX) / 2
  const cy = (minY + maxY) / 2
  const cz = (minZ + maxZ) / 2

  // Center the vertex data so the mesh origin is at its centroid
  const vertices = new Array(rawVerts.length)
  for (let i = 0; i < rawVerts.length; i += 3) {
    vertices[i]     = rawVerts[i]     - cx
    vertices[i + 1] = rawVerts[i + 1] - cy
    vertices[i + 2] = rawVerts[i + 2] - cz
  }

  return {
    vertices,                          // centered vertex positions (flat)
    indices,                           // triangle face indices
    vertexCount: rawVerts.length / 3,
    faceCount: indices.length / 3,
    centroid: [cx, cy, cz],            // original center → becomes object position
    size: [maxX - minX, maxY - minY, maxZ - minZ],
  }
}
