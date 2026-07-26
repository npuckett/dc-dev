/**
 * tests/test-v3-obj.mjs — OBJ export round-trip for the v3 tiled surface.
 *
 * Ported from the v2 suite when the v2 model was removed. `utils/exporters.js`
 * itself is model-agnostic — `buildExportGroup` only reads a panel's type,
 * position and quaternion — so what needs testing here is that v3 tiles reach
 * it correctly through `v3/exportAdapter.js`, and that the baked world
 * transforms actually match the layout.
 */

import assert from 'node:assert/strict'
import { buildPreset } from '../src/core/v3/presets.js'
import { solveLayout } from '../src/core/v3/placement.js'
import { toExportableLayout } from '../src/v3/exportAdapter.js'
import { buildExportGroup, objPayload, exportPanelName, configJSONPayload } from '../src/utils/exporters.js'
import { validateConfig, normalizeConfig } from '../src/core/v3/schema.js'

let passed = 0
let failed = 0
const ok = (c, m) => { if (c) passed++; else { failed++; console.log(`  FAIL: ${m}`) } }

console.log('=== test-v3-obj ===')

const cfg = buildPreset('drift')
const layout = solveLayout(cfg)
const exportable = toExportableLayout(layout)

console.log('1. the adapter presents every placed tile')
{
  const placed = layout.tiles.filter((t) => Array.isArray(t.position))
  ok(exportable.panels.length === placed.length,
    `adapter passes through all ${placed.length} placed tiles (got ${exportable.panels.length})`)
  ok(exportable.panels.every((p) => p.position && p.quaternion),
    'every exported panel carries a position and quaternion')
  // The adapter must not invent v2 semantics beyond naming.
  ok(exportable.panels.every((p) => Number.isInteger(p.row) && Number.isInteger(p.col)),
    'every panel gets integer name indices')
  const plates = exportable.panels.filter((p) => p.type === '2x4')
  ok(plates.length > 0 && plates.every((p) => p.rectOrientation === 'horizontal' || p.rectOrientation === 'vertical'),
    `every plate gets an orientation for its name (${plates.length} plates)`)
  ok(exportable.panels.filter((p) => p.type === '2x2').every((p) => p.rectOrientation === undefined),
    'squares carry no orientation')
}

console.log('2. names are unique — one named object per panel')
{
  const names = exportable.panels.map(exportPanelName)
  ok(new Set(names).size === names.length,
    `all ${names.length} panel names unique (${names.length - new Set(names).size} collisions)`)
}

console.log('3. the OBJ payload parses and matches the layout')
{
  const text = objPayload(exportable)
  ok(typeof text === 'string' && text.length > 0, 'objPayload returns text')

  const lines = text.split('\n')
  const objects = lines.filter((l) => l.startsWith('o '))
  ok(objects.length === exportable.panels.length,
    `one 'o' object per panel (${objects.length} vs ${exportable.panels.length})`)

  const verts = lines.filter((l) => l.startsWith('v '))
  const faces = lines.filter((l) => l.startsWith('f '))
  ok(verts.length > 0 && faces.length > 0, `OBJ carries geometry (${verts.length} v, ${faces.length} f)`)

  // Every face index must be in range and 1-based, per the OBJ format.
  let badIndex = 0
  for (const f of faces) {
    for (const tok of f.slice(2).trim().split(/\s+/)) {
      const idx = parseInt(tok.split('/')[0], 10)
      if (!Number.isFinite(idx) || idx < 1 || idx > verts.length) badIndex++
    }
  }
  ok(badIndex === 0, `all face indices 1-based and in range (${badIndex} bad)`)

  // No NaN escaped into the vertex stream — the classic symptom of a bad
  // quaternion reaching the exporter.
  ok(!/\bNaN\b/.test(text), 'no NaN in the OBJ text')

  // The exported vertex cloud must span the same box as the layout, since the
  // exporter bakes world transforms. Compared against layout.bounds rather than
  // a golden number.
  const pts = verts.map((l) => l.slice(2).trim().split(/\s+/).map(Number))
  const min = [Infinity, Infinity, Infinity]
  const max = [-Infinity, -Infinity, -Infinity]
  for (const p of pts) {
    for (let k = 0; k < 3; k++) {
      if (p[k] < min[k]) min[k] = p[k]
      if (p[k] > max[k]) max[k] = p[k]
    }
  }
  // Tolerance is 1.5cm, not zero, and the reason matters: `layoutBounds`
  // measures each tile's 8-corner SOLID BOX, while the exported mesh is the real
  // profiled shell — recessed diffuser, inward-tapering housing, back plate
  // inset 4cm on all sides. The shell's extreme vertices therefore sit slightly
  // inside the box, most visibly on the normal axis (measured ~0.87cm on the
  // default drift). A zero tolerance here would be asserting that the panel is a
  // box, which it deliberately is not.
  for (let k = 0; k < 3; k++) {
    const size = max[k] - min[k]
    // Upper slack is 0.01cm for OBJ's fixed-precision coordinate rounding, not
    // for geometry: the shell can never exceed the box it is inscribed in.
    ok(size <= layout.bounds.size[k] + 0.01 && size > layout.bounds.size[k] - 1.5,
      `exported extent axis ${k} sits just inside layout bounds (${size.toFixed(2)} vs ${layout.bounds.size[k].toFixed(2)})`)
  }
}

console.log('4. buildExportGroup is headless and consistent')
{
  const g1 = buildExportGroup(exportable)
  ok(g1 && typeof g1 === 'object', 'buildExportGroup returns a group')
  // Determinism: same layout in, identical OBJ text out.
  ok(objPayload(exportable) === objPayload(toExportableLayout(solveLayout(cfg))),
    'OBJ text is byte-identical across repeat solves')
}

console.log('5. the JSON payload round-trips back into a valid config')
{
  const json = configJSONPayload(cfg)
  const reparsed = JSON.parse(json)
  ok(validateConfig(reparsed).valid, 'exported config JSON re-validates')
  ok(JSON.stringify(normalizeConfig(reparsed)) === JSON.stringify(cfg),
    'exported config JSON round-trips to an identical config')
  assert.equal(reparsed.version, 3)
  passed++
}

console.log('6. every preset exports without throwing')
{
  for (const id of ['shelf', 'closed', 'drift', 'dune', 'modular', 'crest']) {
    const l = solveLayout(buildPreset(id))
    const text = objPayload(toExportableLayout(l))
    ok(text.includes('o ') && !/\bNaN\b/.test(text), `${id}: exports clean OBJ`)
  }
}

console.log(`\ntest-v3-obj: ${passed} checks passed, ${failed} failed`)
process.exit(failed ? 1 : 0)
