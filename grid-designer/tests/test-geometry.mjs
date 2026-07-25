/**
 * tests/test-geometry.mjs — headless checks for geometry/panelGeometry.js.
 *
 * Plain node script, no test framework, NO browser. Exits non-zero on any
 * failure.
 *   node tests/test-geometry.mjs
 *
 * =============================================================================
 * WHY THIS SUITE EXISTS
 * =============================================================================
 * The panel solid inherited from panel-designer had its front and back CAPS
 * wound inside-out: the triangles at y = 0 carried a −Y geometric normal and
 * those at y = −overallThickness a +Y one. Under FrontSide materials both were
 * culled, so every panel was missing its lit face AND its housing back — and
 * nothing in the test suite noticed, because the vertex COUNT was right and the
 * OBJ still round-tripped. These are the checks that would have caught it, and
 * they are deliberately about ORIENTATION and CLOSURE rather than counts:
 *
 *   1. closed 2-manifold — every undirected edge is shared by exactly two
 *      triangles, and those two traverse it in OPPOSITE directions (so the
 *      surface is consistently oriented, with no flipped face anywhere)
 *   2. outward orientation — the divergence-theorem signed volume
 *      Σ a·(b×c)/6 is POSITIVE (an inside-out shell gives a negative one) and
 *      equals the closed-form volume of the cross-section, derived here from
 *      config.js's constants rather than hardcoded
 *   3. face-normal spot checks on the three horizontal surfaces that carry the
 *      design intent: the diffuser cap (+Y, at y = −t, on the LIP footprint),
 *      the frame flange top ring (+Y, at y = 0) and the back plate (−Y, at
 *      y = −D, on the BODY footprint); plus no NaN / degenerate normal anywhere
 *   4. extents — bbox y spans exactly [−D, 0] (so core/placement.js's face plane
 *      at y = 0 and thickness D are untouched), x/z span the FULL footprint, the
 *      diffuser area equals the LIP footprint, and the visible frame ring is
 *      lipWidth wide on all four sides
 *   5. material groups — exactly two, partitioning the index buffer with no gap
 *      or overlap, group 0 being the two diffuser triangles
 *   6. the structural counts (72 vertices / 36 triangles), so an accidental
 *      change to the ring sequence is loud rather than silent
 *
 * Vertices are duplicated per quad (by design — that is what makes the frame
 * edges hard), so the manifold check keys edges on QUANTIZED positions rather
 * than on vertex indices.
 */

import { PANEL_DIMENSIONS, PANEL_PROFILE, BODY_INSET } from '../src/config.js'
import {
  DIFFUSER_MATERIAL_INDEX,
  HOUSING_MATERIAL_INDEX,
  buildPanelGeometry,
} from '../src/geometry/panelGeometry.js'

let passed = 0
const failures = []

function check(name, condition, detail = '') {
  if (condition) {
    passed++
  } else {
    failures.push(`${name}${detail ? ` — ${detail}` : ''}`)
    console.error(`FAIL  ${name}${detail ? ` — ${detail}` : ''}`)
  }
}

// -----------------------------------------------------------------------------
// profile constants (the geometry is a pure function of these)
// -----------------------------------------------------------------------------
const T = PANEL_PROFILE.outerThickness     // 1.0  — flange depth / diffuser recess
const D = PANEL_PROFILE.overallThickness   // 3.7  — full depth
const LW = PANEL_PROFILE.lipWidth          // 2.5  — visible frame ring width
const INSET = BODY_INSET                   // 4.0  — tapered wall horizontal run

const EXPECT_VERTS = 72
const EXPECT_TRIS = 36
const EXPECT_INDICES = EXPECT_TRIS * 3

/**
 * Closed-form volume of the panel profile, from the constants above.
 *
 * Slice the solid by depth:
 *   - y ∈ [−t, 0]  the FRAME FLANGE slab: the full footprint MINUS the diffuser
 *     opening (which is open air — that recess is what makes the frame read).
 *   - y ∈ [−D, −t] the TAPERED TRAY: a filled rectangle whose footprint shrinks
 *     linearly from FULL at y = −t to FULL inset by BODY_INSET at y = −D. It is
 *     filled because the diffuser cap closes the cavity's top at y = −t.
 *
 * The tray integral, with s the per-side inset (0 → INSET over a depth of D−t):
 *   ∫ (W−2s)(H−2s) dy = ((D−t)/INSET) · ∫₀^INSET (W−2s)(H−2s) ds
 *                     = ((D−t)/INSET) · [W·H·a − (W+H)a² + (4/3)a³],  a = INSET
 *
 * @param {number} W panel width (cm)
 * @param {number} H panel height (cm)
 * @returns {number} volume in cm³
 */
function profileVolume(W, H) {
  const flange = T * (W * H - (W - 2 * LW) * (H - 2 * LW))
  const a = INSET
  const tray = ((D - T) / INSET) * (W * H * a - (W + H) * a * a + (4 / 3) * a * a * a)
  return flange + tray
}

// -----------------------------------------------------------------------------
// small vector / mesh helpers (written from scratch — nothing from src/)
// -----------------------------------------------------------------------------
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
]
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
const len = (a) => Math.hypot(a[0], a[1], a[2])

/** Quantize a position to a 1e-6 grid so duplicated corners collide exactly. */
const key = (p) =>
  `${Math.round(p[0] * 1e6) / 1e6},${Math.round(p[1] * 1e6) / 1e6},${Math.round(p[2] * 1e6) / 1e6}`
const key2 = (x, z) => `${Math.round(x * 1e6) / 1e6},${Math.round(z * 1e6) / 1e6}`

/** Pull an indexed geometry apart into `{ verts, tris }` of plain arrays. */
function readMesh(geometry) {
  const pos = geometry.getAttribute('position')
  const index = geometry.getIndex()
  const verts = []
  for (let i = 0; i < pos.count; i++) verts.push([pos.getX(i), pos.getY(i), pos.getZ(i)])
  const tris = []
  for (let i = 0; i < index.count; i += 3) {
    tris.push([index.getX(i), index.getX(i + 1), index.getX(i + 2)])
  }
  return { verts, tris }
}

/** Unit geometric normal of a triangle, or null if degenerate / non-finite. */
function faceNormal(verts, tri) {
  const [a, b, c] = tri.map((i) => verts[i])
  const n = cross(sub(b, a), sub(c, a))
  const l = len(n)
  if (!Number.isFinite(l) || l < 1e-9) return null
  return [n[0] / l, n[1] / l, n[2] / l]
}

/** Area of a triangle. */
const triArea = (verts, tri) => {
  const [a, b, c] = tri.map((i) => verts[i])
  return len(cross(sub(b, a), sub(c, a))) / 2
}

// =============================================================================
// Per-type checks
// =============================================================================
for (const type of ['2x2', '2x4']) {
  const W = PANEL_DIMENSIONS[type].width
  const H = PANEL_DIMENSIONS[type].height
  const hw = W / 2
  const hh = H / 2
  const tag = `[${type}]`

  const geometry = buildPanelGeometry({ type })
  const { verts, tris } = readMesh(geometry)

  // --- 6. structural counts -------------------------------------------------
  check(
    `${tag} 72 vertices (18 quads × 4, duplicated per quad for hard edges)`,
    verts.length === EXPECT_VERTS,
    `got ${verts.length}`,
  )
  check(
    `${tag} 36 triangles / ${EXPECT_INDICES} indices`,
    tris.length === EXPECT_TRIS && geometry.getIndex().count === EXPECT_INDICES,
    `got ${tris.length} triangles, ${geometry.getIndex().count} indices`,
  )

  // --- 3b. no NaN / degenerate normals --------------------------------------
  const badNormals = tris.filter((tri) => faceNormal(verts, tri) === null)
  check(
    `${tag} every triangle has a finite, non-zero normal (no NaN, no degenerate quad)`,
    badNormals.length === 0,
    `${badNormals.length} bad triangle(s)`,
  )
  check(
    `${tag} no vertex coordinate is NaN or infinite`,
    verts.every((v) => v.every(Number.isFinite)),
  )

  // --- 1. closed 2-manifold, consistently oriented --------------------------
  // Undirected edges keyed on quantized endpoint POSITIONS (vertices are
  // duplicated, so index-keyed edges would never match up).
  const edges = new Map() // "min|max" → { forward, backward }
  for (const tri of tris) {
    const p = tri.map((i) => verts[i])
    for (let e = 0; e < 3; e++) {
      const ka = key(p[e])
      const kb = key(p[(e + 1) % 3])
      const undirected = ka < kb ? `${ka}|${kb}` : `${kb}|${ka}`
      let entry = edges.get(undirected)
      if (!entry) {
        entry = { forward: 0, backward: 0 }
        edges.set(undirected, entry)
      }
      if (ka < kb) entry.forward++
      else entry.backward++
    }
  }
  const wrongCount = [...edges.entries()].filter(
    ([, e]) => e.forward + e.backward !== 2,
  )
  check(
    `${tag} closed: every undirected edge is shared by exactly 2 triangles`,
    wrongCount.length === 0,
    wrongCount.length
      ? `${wrongCount.length} bad edge(s), e.g. ${wrongCount[0][0]} used ${wrongCount[0][1].forward + wrongCount[0][1].backward}×`
      : `${edges.size} edges`,
  )
  const wrongOrientation = [...edges.entries()].filter(
    ([, e]) => !(e.forward === 1 && e.backward === 1),
  )
  check(
    `${tag} consistently oriented: the 2 triangles on each edge traverse it in OPPOSITE directions`,
    wrongOrientation.length === 0,
    wrongOrientation.length
      ? `${wrongOrientation.length} inconsistent edge(s), e.g. ${wrongOrientation[0][0]} (fwd ${wrongOrientation[0][1].forward}, bwd ${wrongOrientation[0][1].backward})`
      : `${edges.size} edges`,
  )
  // A closed triangulated surface of genus 0: V − E + F = 2 on the WELDED mesh.
  const weldedVerts = new Set(verts.map(key)).size
  check(
    `${tag} Euler characteristic V − E + F = 2 (a single closed shell, genus 0)`,
    weldedVerts - edges.size + tris.length === 2,
    `V=${weldedVerts} E=${edges.size} F=${tris.length} → ${weldedVerts - edges.size + tris.length}`,
  )

  // --- 2. outward orientation via signed volume -----------------------------
  let signedVolume = 0
  for (const tri of tris) {
    const [a, b, c] = tri.map((i) => verts[i])
    signedVolume += dot(a, cross(b, c)) / 6
  }
  const expectedVolume = profileVolume(W, H)
  check(
    `${tag} signed volume is POSITIVE — the shell faces OUTWARD (an inside-out cap makes this negative)`,
    signedVolume > 0,
    `${signedVolume.toFixed(3)} cm³`,
  )
  const relErr = Math.abs(signedVolume - expectedVolume) / expectedVolume
  check(
    `${tag} signed volume matches the closed-form profile volume (${expectedVolume.toFixed(2)} cm³)`,
    relErr < 1e-4,
    `got ${signedVolume.toFixed(3)} cm³, relative error ${relErr.toExponential(2)}`,
  )

  // --- classify the horizontal surfaces by the plane all 3 verts lie in -----
  const atY = (tri, y) => tri.every((i) => Math.abs(verts[i][1] - y) < 1e-4)
  const diffuserTris = tris.filter((tri) => atY(tri, -T))
  const flangeTris = tris.filter((tri) => atY(tri, 0))
  const backTris = tris.filter((tri) => atY(tri, -D))

  // --- 3. face-normal spot checks -------------------------------------------
  const isPlusY = (n) => n && Math.abs(n[0]) < 1e-6 && Math.abs(n[2]) < 1e-6 && n[1] > 0.999999
  const isMinusY = (n) => n && Math.abs(n[0]) < 1e-6 && Math.abs(n[2]) < 1e-6 && n[1] < -0.999999

  check(
    `${tag} the DIFFUSER is 2 triangles, all at y = −${T}, all facing exactly +Y`,
    diffuserTris.length === 2 && diffuserTris.every((tri) => isPlusY(faceNormal(verts, tri))),
    `${diffuserTris.length} triangle(s), normals ${JSON.stringify(diffuserTris.map((tri) => faceNormal(verts, tri)))}`,
  )
  const lipCorners = new Set([
    key2(-(hw - LW), -(hh - LW)),
    key2(hw - LW, -(hh - LW)),
    key2(hw - LW, hh - LW),
    key2(-(hw - LW), hh - LW),
  ])
  const diffuserFootprint = new Set(
    diffuserTris.flatMap((tri) => tri.map((i) => key2(verts[i][0], verts[i][2]))),
  )
  check(
    `${tag} the diffuser sits on the LIP footprint (${W - 2 * LW}×${H - 2 * LW})`,
    diffuserFootprint.size === 4 && [...diffuserFootprint].every((k) => lipCorners.has(k)),
    JSON.stringify([...diffuserFootprint]),
  )
  check(
    `${tag} the FRAME FLANGE top ring is 8 triangles at y = 0, all facing exactly +Y`,
    flangeTris.length === 8 && flangeTris.every((tri) => isPlusY(faceNormal(verts, tri))),
    `${flangeTris.length} triangle(s)`,
  )
  check(
    `${tag} the BACK PLATE is 2 triangles at y = −${D}, both facing exactly −Y`,
    backTris.length === 2 && backTris.every((tri) => isMinusY(faceNormal(verts, tri))),
    `${backTris.length} triangle(s), normals ${JSON.stringify(backTris.map((tri) => faceNormal(verts, tri)))}`,
  )
  const bodyCorners = new Set([
    key2(-(hw - INSET), -(hh - INSET)),
    key2(hw - INSET, -(hh - INSET)),
    key2(hw - INSET, hh - INSET),
    key2(-(hw - INSET), hh - INSET),
  ])
  const backFootprint = new Set(
    backTris.flatMap((tri) => tri.map((i) => key2(verts[i][0], verts[i][2]))),
  )
  check(
    `${tag} the back plate sits on the BODY footprint (${W - 2 * INSET}×${H - 2 * INSET})`,
    backFootprint.size === 4 && [...backFootprint].every((k) => bodyCorners.has(k)),
    JSON.stringify([...backFootprint]),
  )

  // --- 4. extents -----------------------------------------------------------
  const xs = verts.map((v) => v[0])
  const ys = verts.map((v) => v[1])
  const zs = verts.map((v) => v[2])
  check(
    `${tag} bbox y spans exactly [−${D}, 0] — the lit face plane is still y = 0, so placement math is untouched`,
    Math.abs(Math.min(...ys) + D) < 1e-4 && Math.abs(Math.max(...ys)) < 1e-9,
    `y ∈ [${Math.min(...ys)}, ${Math.max(...ys)}]`,
  )
  check(
    `${tag} bbox x/z span the FULL footprint (${W}×${H})`,
    Math.abs(Math.min(...xs) + hw) < 1e-4 &&
      Math.abs(Math.max(...xs) - hw) < 1e-4 &&
      Math.abs(Math.min(...zs) + hh) < 1e-4 &&
      Math.abs(Math.max(...zs) - hh) < 1e-4,
    `x ∈ [${Math.min(...xs)}, ${Math.max(...xs)}]  z ∈ [${Math.min(...zs)}, ${Math.max(...zs)}]`,
  )
  const diffuserArea = diffuserTris.reduce((s, tri) => s + triArea(verts, tri), 0)
  const lipArea = (W - 2 * LW) * (H - 2 * LW)
  check(
    `${tag} the diffuser's area equals the LIP footprint area (${lipArea} cm²)`,
    Math.abs(diffuserArea - lipArea) / lipArea < 1e-5,
    `${diffuserArea.toFixed(4)} vs ${lipArea}`,
  )
  // The visible frame ring: the y = 0 surface runs from the FULL rim inward to
  // the LIP edge, so its distinct footprint corners are exactly FULL ∪ LIP and
  // the ring is lipWidth wide on every side.
  const flangeFootprint = new Set(
    flangeTris.flatMap((tri) => tri.map((i) => key2(verts[i][0], verts[i][2]))),
  )
  const fullCorners = [
    key2(-hw, -hh), key2(hw, -hh), key2(hw, hh), key2(-hw, hh),
  ]
  check(
    `${tag} the frame ring's footprint is exactly FULL ∪ LIP (8 distinct corners)`,
    flangeFootprint.size === 8 &&
      fullCorners.every((k) => flangeFootprint.has(k)) &&
      [...lipCorners].every((k) => flangeFootprint.has(k)),
    `${flangeFootprint.size} corners: ${JSON.stringify([...flangeFootprint])}`,
  )
  const flangeArea = flangeTris.reduce((s, tri) => s + triArea(verts, tri), 0)
  const ringArea = W * H - lipArea
  check(
    `${tag} the visible frame ring is ${LW}cm wide on all four sides (area ${ringArea} cm²)`,
    Math.abs(flangeArea - ringArea) / ringArea < 1e-5,
    `${flangeArea.toFixed(4)} vs ${ringArea}`,
  )

  // --- 5. material groups ---------------------------------------------------
  const groups = geometry.groups
  check(`${tag} exactly 2 material groups`, groups.length === 2, `got ${groups.length}`)
  const sorted = [...groups].sort((a, b) => a.start - b.start)
  check(
    `${tag} the groups partition the whole index buffer — no gap, no overlap`,
    sorted.length === 2 &&
      sorted[0].start === 0 &&
      sorted[0].start + sorted[0].count === sorted[1].start &&
      sorted[1].start + sorted[1].count === EXPECT_INDICES,
    JSON.stringify(sorted.map((g) => [g.start, g.count, g.materialIndex])),
  )
  check(
    `${tag} group 0 is the 2 DIFFUSER triangles (material ${DIFFUSER_MATERIAL_INDEX})`,
    sorted[0]?.materialIndex === DIFFUSER_MATERIAL_INDEX && sorted[0]?.count === 6,
    JSON.stringify(sorted[0]),
  )
  check(
    `${tag} group 1 is frame + housing (material ${HOUSING_MATERIAL_INDEX}, 34 triangles)`,
    sorted[1]?.materialIndex === HOUSING_MATERIAL_INDEX &&
      sorted[1]?.count === EXPECT_INDICES - 6,
    JSON.stringify(sorted[1]),
  )
  // …and group 0's index range really IS the diffuser, not merely 6 indices.
  const group0Tris = tris.slice(0, 2)
  check(
    `${tag} group 0's index range holds the +Y diffuser cap at y = −${T}`,
    group0Tris.every((tri) => atY(tri, -T) && isPlusY(faceNormal(verts, tri))),
    JSON.stringify(group0Tris.map((tri) => faceNormal(verts, tri))),
  )
}

// =============================================================================
// The API no longer accepts the panel-designer options
// =============================================================================
{
  let threw = false
  try {
    buildPanelGeometry({ type: 'nope' })
  } catch {
    threw = true
  }
  check('an unknown panel type throws rather than building a NaN mesh', threw)

  const a = buildPanelGeometry({ type: '2x2' })
  const b = buildPanelGeometry({ type: '2x4' })
  check(
    'the two panel types differ only in the Z extent (same profile, same counts)',
    a.getAttribute('position').count === b.getAttribute('position').count &&
      Math.abs(a.boundingBox.max.z - 30) < 1e-4 &&
      Math.abs(b.boundingBox.max.z - 60.5) < 1e-4 &&
      Math.abs(a.boundingBox.max.x - b.boundingBox.max.x) < 1e-6,
    `2x2 z=${a.boundingBox.max.z}, 2x4 z=${b.boundingBox.max.z}`,
  )
}

// =============================================================================
// Summary
// =============================================================================
console.log('\n=== test-geometry ===')
if (failures.length === 0) {
  console.log(`PASS — ${passed}/${passed} checks`)
  process.exit(0)
} else {
  console.log(`FAIL — ${failures.length}/${passed + failures.length} check(s) failed:`)
  for (const f of failures) console.log(`  - ${f}`)
  process.exit(1)
}
