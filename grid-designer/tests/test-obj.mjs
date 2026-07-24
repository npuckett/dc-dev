/**
 * tests/test-obj.mjs — headless checks for utils/exporters.js.
 *
 * Plain node script, no test framework, NO browser. Exits non-zero on any
 * failure.
 *   node tests/test-obj.mjs
 *
 * This is the reason `buildExportGroup` is DOM-free: OBJExporter itself only
 * needs three's math, so the whole bake-and-serialize path can be verified here
 * rather than by eyeballing a downloaded file.
 *
 * What it checks
 *   1. wave preset → one `o <name>` block per layout panel, names unique
 *   2. per-object vertex counts equal buildPanelGeometry's position count for
 *      that panel's type (i.e. nothing was merged, dropped, or duplicated)
 *   3. the bake is in WORLD space: every exported vertex block's bounds contain
 *      the panel's `panelWorldCorners`, and for the flat default layout panel
 *      p0_0's shore corner lands on the absolute expected (0, 3.7, 0)
 *   4. config JSON round-trip: payload → parse → validateConfig ok → solveLayout
 *      deep-equals the original layout
 */

import assert from 'node:assert/strict'
import { DEFAULT_CONFIG, normalizeConfig, validateConfig } from '../src/core/schema.js'
import { buildPreset } from '../src/core/presets.js'
import { panelWorldCorners, solveLayout } from '../src/core/placement.js'
import { buildPanelGeometry } from '../src/geometry/panelGeometry.js'
import {
  buildExportGroup,
  configJSONPayload,
  exportPanelName,
  objPayload,
} from '../src/utils/exporters.js'

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

function checkNoThrow(name, fn) {
  try {
    fn()
    passed++
  } catch (e) {
    failures.push(`${name} — ${e.message}`)
    console.error(`FAIL  ${name} — ${e.message}`)
  }
}

/**
 * Parse an OBJ into `{ name → [[x,y,z], …] }`, preserving object order.
 * Written from scratch here rather than reusing anything from src/.
 */
function parseObjObjects(text) {
  const objects = []
  let current = null
  for (const raw of text.split('\n')) {
    const line = raw.trim()
    if (line.startsWith('o ')) {
      current = { name: line.slice(2).trim(), vertices: [] }
      objects.push(current)
    } else if (line.startsWith('v ') && current) {
      const [x, y, z] = line.slice(2).trim().split(/\s+/).map(Number)
      current.vertices.push([x, y, z])
    }
  }
  return objects
}

/** Expected vertex count for a panel type, straight from the geometry builder. */
const typeVertexCount = (type) =>
  buildPanelGeometry({ type, sidedness: 'single', powerSupplyEdge: 0 }).getAttribute('position')
    .count

const dist3 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])

// =============================================================================
// 1. Wave preset — object count, names, vertex counts
// =============================================================================
{
  const config = normalizeConfig(buildPreset('wave'))
  check('wave: config validates', validateConfig(config).ok)

  const layout = solveLayout(config)
  check(
    'wave: 28 panels (30 cells − 1 vertical plate − 1 horizontal plate)',
    layout.panels.length === 28,
    `got ${layout.panels.length}`,
  )

  const group = buildExportGroup(layout)
  check(
    'wave: export group has one mesh per panel',
    group.children.length === layout.panels.length,
    `${group.children.length} meshes vs ${layout.panels.length} panels`,
  )
  check(
    'wave: every mesh sits at the identity transform (world matrix is baked in)',
    group.children.every(
      (m) =>
        m.position.lengthSq() === 0 &&
        m.scale.x === 1 &&
        m.scale.y === 1 &&
        m.scale.z === 1 &&
        m.quaternion.x === 0 &&
        m.quaternion.y === 0 &&
        m.quaternion.z === 0 &&
        m.quaternion.w === 1,
    ),
  )

  const obj = objPayload(layout)
  const objects = parseObjObjects(obj)

  check(
    'wave OBJ: `o ` object count equals panel count',
    (obj.match(/^o /gm) ?? []).length === layout.panels.length,
    `${(obj.match(/^o /gm) ?? []).length} objects vs ${layout.panels.length} panels`,
  )
  check('wave OBJ: parsed object count matches', objects.length === layout.panels.length)

  const names = objects.map((o) => o.name)
  check('wave OBJ: object names are unique', new Set(names).size === names.length)
  check(
    'wave OBJ: object names/order match exportPanelName over layout.panels',
    JSON.stringify(names) === JSON.stringify(layout.panels.map(exportPanelName)),
    JSON.stringify(names.slice(0, 4)),
  )
  check(
    'wave OBJ: plates are named rect_h_* / rect_v_*, squares panel_*',
    names.includes('rect_v_r2_c0') &&
      names.includes('rect_h_r4_c2') &&
      names.filter((n) => n.startsWith('panel_')).length === 26,
    JSON.stringify(names.filter((n) => !n.startsWith('panel_'))),
  )

  const expectCount = { '2x2': typeVertexCount('2x2'), '2x4': typeVertexCount('2x4') }
  let counts = true
  let countDetail = ''
  objects.forEach((o, i) => {
    const want = expectCount[layout.panels[i].type]
    if (o.vertices.length !== want) {
      counts = false
      countDetail = `${o.name}: ${o.vertices.length} vertices, want ${want}`
    }
  })
  check(
    `wave OBJ: per-object vertex counts match buildPanelGeometry (2x2=${expectCount['2x2']}, 2x4=${expectCount['2x4']})`,
    counts,
    countDetail,
  )

  // Every panel's four lit-face corners must appear among that object's
  // vertices — the strongest available statement that the world matrix was
  // baked into the right object.
  let cornersFound = true
  let cornerDetail = ''
  objects.forEach((o, i) => {
    const panel = layout.panels[i]
    for (const corner of panelWorldCorners(panel, config)) {
      const best = o.vertices.reduce((m, v) => Math.min(m, dist3(v, corner)), Infinity)
      if (best > 1e-4) {
        cornersFound = false
        cornerDetail = `${o.name}: no vertex within 1e-4 of ${JSON.stringify(corner)} (nearest ${best.toExponential(2)})`
      }
    }
  })
  check(
    'wave OBJ: every panel\'s 4 world lit-face corners appear in its own object',
    cornersFound,
    cornerDetail,
  )
}

// =============================================================================
// 2. Flat DEFAULT_CONFIG — absolute spot-checked world vertex
// =============================================================================
{
  const config = normalizeConfig(DEFAULT_CONFIG)
  const layout = solveLayout(config)
  const objects = parseObjObjects(objPayload(layout))

  const p00 = layout.panels.find((p) => p.row === 0 && p.col === 0)
  check('flat: p0_0 exists', !!p00 && p00.id === 'p0_0')

  const obj00 = objects.find((o) => o.name === 'panel_r0_c0')
  check('flat OBJ: contains `o panel_r0_c0`', !!obj00)

  // Derived expectation (corner [0] is the (−X, −Z) lit-face corner) …
  const derived = panelWorldCorners(p00, config)[0]
  check(
    'flat: placement puts p0_0\'s (−X,−Z) lit-face corner at the world origin corner',
    dist3(derived, [0, 3.7, 0]) <= 1e-9,
    `panelWorldCorners → ${JSON.stringify(derived)}`,
  )

  // … and the absolute one: row 0's near edge is anchored on the world X axis
  // at z = 0, a flat centered row starts at x = 0, and the lit face is lifted to
  // y = overallThickness = 3.7. Tolerance 1e-6 covers the float32 round-trip
  // through the geometry's position attribute (3.7 → 3.7000000476837158).
  const nearest = obj00
    ? obj00.vertices.reduce(
        (best, v) => (dist3(v, [0, 3.7, 0]) < best.d ? { v, d: dist3(v, [0, 3.7, 0]) } : best),
        { v: null, d: Infinity },
      )
    : { v: null, d: Infinity }
  check(
    'flat OBJ: panel_r0_c0 has a vertex at exactly (0, 3.7, 0) within 1e-6',
    nearest.d <= 1e-6,
    `nearest ${JSON.stringify(nearest.v)} at ${nearest.d.toExponential(2)}`,
  )
  check(
    'flat OBJ: that vertex also matches the derived corner within 1e-6',
    nearest.v !== null && dist3(nearest.v, derived) <= 1e-6,
  )
}

// =============================================================================
// 3. Config JSON round-trip
// =============================================================================
{
  const config = normalizeConfig(buildPreset('wave'))
  const layout = solveLayout(config)
  const payload = configJSONPayload(config)

  check('round-trip: payload is pretty-printed', payload.includes('\n  "version": 1'))

  let parsed = null
  checkNoThrow('round-trip: payload parses as JSON', () => {
    parsed = JSON.parse(payload)
  })

  const validation = validateConfig(parsed)
  check(
    'round-trip: parsed config validates ok',
    validation.ok,
    JSON.stringify(validation.errors),
  )
  checkNoThrow('round-trip: parsed config deep-equals the original', () => {
    assert.deepStrictEqual(parsed, config)
  })
  checkNoThrow('round-trip: re-solved layout deep-equals the original layout', () => {
    assert.deepStrictEqual(solveLayout(parsed), layout)
  })

  // A minimal hand-written config must survive normalize → validate → solve too:
  // that is the contract the JSON paste panel relies on.
  const minimal = {
    version: 1,
    units: 'cm',
    rows: [
      { zigzagDeg: 0 },
      { zigzagDeg: 25 },
      { zigzagDeg: 45 },
      { zigzagDeg: 10 },
      { zigzagDeg: 5 },
    ],
    rowFoldsDeg: [20, 35, 25, -15],
  }
  const normalizedMinimal = normalizeConfig(minimal)
  check(
    'round-trip: minimal hand-written config validates after normalize',
    validateConfig(normalizedMinimal).ok,
    JSON.stringify(validateConfig(normalizedMinimal).errors),
  )
  const minimalLayout = solveLayout(normalizedMinimal)
  check(
    'round-trip: minimal config solves to 30 panels',
    minimalLayout.panels.length === 30,
    `got ${minimalLayout.panels.length}`,
  )
  checkNoThrow('round-trip: minimal config re-round-trips identically', () => {
    assert.deepStrictEqual(
      JSON.parse(configJSONPayload(normalizedMinimal)),
      normalizedMinimal,
    )
  })
}

// =============================================================================
// Summary
// =============================================================================
console.log('\n=== test-obj ===')
if (failures.length === 0) {
  console.log(`PASS — ${passed}/${passed} checks`)
  process.exit(0)
} else {
  console.log(`FAIL — ${failures.length}/${passed + failures.length} check(s) failed:`)
  for (const f of failures) console.log(`  - ${f}`)
  process.exit(1)
}
