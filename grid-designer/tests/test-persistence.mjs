/**
 * tests/test-persistence.mjs — headless checks for persistence.js.
 *
 * Plain node script, no test framework. Exits non-zero on any failure.
 *   node tests/test-persistence.mjs
 *
 * persistence.js is deliberately NOT in src/core/ (it uses ordinary browser
 * storage APIs), but it is still exercised here in plain node against a
 * STUBBED `localStorage` installed on `globalThis` — the module reads the bare
 * `localStorage` identifier lazily inside each function call (never at module
 * load), so swapping `globalThis.localStorage` between tests is enough to
 * point it at a different backend with no re-import needed.
 *
 * What this covers: the working-config round trip (including the debounce
 * actually collapsing to the LAST value, not the first or an intermediate
 * one), corrupt / malformed JSON, schema-version mismatches (both the
 * envelope's own tag and the embedded config's `version`), a storage backend
 * that throws on every call, storage being entirely absent, and full slot
 * CRUD. Every one of these must degrade to null / false / a no-op — NEVER
 * throw — because that is the whole point of this module.
 */

import {
  loadWorkingConfig,
  saveWorkingConfig,
  flushWorkingConfig,
  isStorageAvailable,
  listSlots,
  saveSlot,
  loadSlot,
  deleteSlot,
  WORKING_KEY,
  SLOTS_KEY,
} from '../src/persistence.js'

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

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

// -----------------------------------------------------------------------------
// Stubbed storage backends
// -----------------------------------------------------------------------------

/** A working in-memory localStorage, close enough to the real API for this. */
class MockStorage {
  constructor() {
    this.data = new Map()
  }
  getItem(key) {
    return this.data.has(key) ? this.data.get(key) : null
  }
  setItem(key, value) {
    this.data.set(key, String(value))
  }
  removeItem(key) {
    this.data.delete(key)
  }
}

/** Every method throws — private browsing / a hard quota wall, worst case. */
class ThrowingStorage {
  getItem() {
    throw new Error('storage disabled')
  }
  setItem() {
    throw new Error('storage disabled')
  }
  removeItem() {
    throw new Error('storage disabled')
  }
}

function installStorage(storage) {
  globalThis.localStorage = storage
}

function removeStorage() {
  delete globalThis.localStorage
}

// -----------------------------------------------------------------------------
// Fixtures
// -----------------------------------------------------------------------------

/** v3-shaped config (V3_SPEC.md §6) — the CURRENT schema this build accepts. */
function sampleConfig(overrides = {}) {
  return {
    version: 3,
    units: 'cm',
    sheet: { cols: 6, rows: 8 },
    cell: { size: 60, plateLength: 121 },
    gap: 1,
    form: {
      amplitude: 120,
      crestX: 0.62,
      crestZ: 0.55,
      ridgeShear: 0.18,
      toeSharpX: 0.8,
      toeSharpZ: 0.9,
      angularity: 0.5,
      facetCells: 2,
      footprint: { width: 365, depth: 487 },
    },
    tiling: { strategy: 'flat-lie', plateFitToleranceCm: 2.0 },
    placement: { tree: 'bfs-corner', mode: 'surface-fit' },
    gapTolerance: 1.5,
    groundTolerance: 2.0,
    meta: { preset: 'drift', tilePattern: 'flat-lie', notes: '' },
    ...overrides,
  }
}

/** v2-shaped config (per-column fold strips) — the RETIRED schema a v3 build
 *  must discard rather than resurrect (V3_SPEC.md §6 / §7). */
function sampleV2Config(overrides = {}) {
  return {
    version: 2,
    units: 'cm',
    grid: { cols: 6, rows: 5 },
    cell: { size: 60, rectLength: 121 },
    gap: 1,
    gapTolerance: 1.5,
    groundTolerance: 0.5,
    columns: Array.from({ length: 6 }, () => ({
      foldsDeg: [0, 0, 0, 0],
      startPitchDeg: 0,
      endSupport: 'floor',
    })),
    rects: [],
    meta: { preset: 'flat', notes: '' },
    ...overrides,
  }
}

// =============================================================================
// 1. Working config — round trip
// =============================================================================
{
  installStorage(new MockStorage())
  flushWorkingConfig() // clear any state left by a prior run in this process

  check('round trip: nothing saved yet returns null', loadWorkingConfig() === null)

  const cfg = sampleConfig({ name: 'round trip' })
  saveWorkingConfig(cfg, 20)
  check(
    'debounce: nothing written before the delay elapses',
    loadWorkingConfig() === null,
    'expected the write to still be pending',
  )

  await sleep(60)
  const loaded = loadWorkingConfig()
  check('round trip: config comes back after the debounce fires', loaded !== null)
  check(
    'round trip: byte-identical to what was saved',
    JSON.stringify(loaded) === JSON.stringify(cfg),
  )
}

// =============================================================================
// 2. Debounce collapses to the LAST value, never an intermediate one
// =============================================================================
{
  installStorage(new MockStorage())
  flushWorkingConfig()

  saveWorkingConfig(sampleConfig({ name: 'first' }), 30)
  saveWorkingConfig(sampleConfig({ name: 'second' }), 30)
  saveWorkingConfig(sampleConfig({ name: 'third (final)' }), 30)
  await sleep(70)
  const loaded = loadWorkingConfig()
  check(
    'debounce: rapid calls collapse to exactly one write of the LAST value',
    loaded?.name === 'third (final)',
    `got ${JSON.stringify(loaded?.name)}`,
  )
}

// =============================================================================
// 3. flushWorkingConfig writes the pending value immediately (the reload guard)
// =============================================================================
{
  installStorage(new MockStorage())
  flushWorkingConfig()

  saveWorkingConfig(sampleConfig({ name: 'flushed' }), 10_000) // long delay — flush must beat it
  check('flush: nothing written yet (still debounced)', loadWorkingConfig() === null)
  flushWorkingConfig()
  check('flush: value is written immediately on flush', loadWorkingConfig()?.name === 'flushed')

  check('flush: a second flush with nothing pending is a harmless no-op', (() => {
    flushWorkingConfig()
    return true // just must not throw
  })())
}

// =============================================================================
// 4. Corrupt / malformed stored data never throws and always yields null
// =============================================================================
{
  const storage = new MockStorage()
  installStorage(storage)
  flushWorkingConfig()

  storage.setItem(WORKING_KEY, '{not valid json')
  check('corrupt JSON: loadWorkingConfig returns null, does not throw', loadWorkingConfig() === null)

  storage.setItem(WORKING_KEY, JSON.stringify([1, 2, 3]))
  check('malformed envelope (array, not object): returns null', loadWorkingConfig() === null)

  storage.setItem(WORKING_KEY, JSON.stringify({ schemaVersion: 3 })) // no `config`
  check('malformed envelope (missing config): returns null', loadWorkingConfig() === null)

  storage.setItem(WORKING_KEY, JSON.stringify({ schemaVersion: 3, config: 'not an object' }))
  check('malformed envelope (config not an object): returns null', loadWorkingConfig() === null)

  storage.setItem(WORKING_KEY, '')
  check('empty string: returns null', loadWorkingConfig() === null)
}

// =============================================================================
// 5. Schema version mismatch is discarded, not resurrected
// =============================================================================
{
  const storage = new MockStorage()
  installStorage(storage)
  flushWorkingConfig()

  // The envelope's own tag is stale (as if written by a future/older build).
  storage.setItem(
    WORKING_KEY,
    JSON.stringify({ schemaVersion: 999, savedAt: Date.now(), config: sampleConfig() }),
  )
  check(
    'version mismatch: stale envelope schemaVersion is discarded',
    loadWorkingConfig() === null,
  )

  // The envelope tag matches (3), but the embedded config claims a different
  // version (e.g. a v1 or v2 config that slipped in some other way).
  storage.setItem(
    WORKING_KEY,
    JSON.stringify({ schemaVersion: 3, savedAt: Date.now(), config: sampleConfig({ version: 2 }) }),
  )
  check(
    "version mismatch: embedded config's own `version` field is checked independently",
    loadWorkingConfig() === null,
    'a v2-tagged config must not be resurrected into a v3-only build',
  )

  storage.setItem(
    WORKING_KEY,
    JSON.stringify({ schemaVersion: 3, savedAt: Date.now(), config: sampleConfig({ version: 1 }) }),
  )
  check('version mismatch: a v1 config is also discarded', loadWorkingConfig() === null)

  // THE ACTUAL PIVOT CASE (V3_SPEC.md §6 / §7): a real v2 config (per-column
  // fold strips), saved by a pre-pivot build under the OLD envelope tag, must
  // be discarded rather than loaded into a v3 store — the two models describe
  // physically different objects and do not map onto one another.
  storage.setItem(
    WORKING_KEY,
    JSON.stringify({ schemaVersion: 2, savedAt: Date.now(), config: sampleV2Config() }),
  )
  check(
    'v2 pivot: a v2 envelope (schemaVersion 2) saved by a pre-pivot build is discarded, not loaded into v3',
    loadWorkingConfig() === null,
  )
}

// =============================================================================
// 6. A storage backend that throws on every call never breaks the app
// =============================================================================
{
  installStorage(new ThrowingStorage())
  flushWorkingConfig() // must not throw even mid-flush against a throwing backend

  check('throwing storage: loadWorkingConfig returns null, does not throw', (() => {
    try {
      return loadWorkingConfig() === null
    } catch {
      return false
    }
  })())

  check('throwing storage: saveWorkingConfig does not throw', (() => {
    try {
      saveWorkingConfig(sampleConfig(), 5)
      return true
    } catch {
      return false
    }
  })())

  await sleep(20)
  check('throwing storage: the debounced write itself does not throw', (() => {
    try {
      flushWorkingConfig()
      return true
    } catch {
      return false
    }
  })())

  check('throwing storage: isStorageAvailable reports false, does not throw', (() => {
    try {
      return isStorageAvailable() === false
    } catch {
      return false
    }
  })())

  check('throwing storage: slot reads degrade to empty, do not throw', (() => {
    try {
      return Array.isArray(listSlots()) && listSlots().length === 0
    } catch {
      return false
    }
  })())

  check('throwing storage: slot writes degrade to false, do not throw', (() => {
    try {
      return saveSlot('x', sampleConfig()) === false
    } catch {
      return false
    }
  })())
}

// =============================================================================
// 7. No localStorage at all (e.g. a non-browser embedder) is handled the same way
// =============================================================================
{
  removeStorage()
  flushWorkingConfig()

  check('no storage: loadWorkingConfig returns null, does not throw', (() => {
    try {
      return loadWorkingConfig() === null
    } catch {
      return false
    }
  })())
  check('no storage: saveWorkingConfig + flush does not throw', (() => {
    try {
      saveWorkingConfig(sampleConfig(), 5)
      flushWorkingConfig()
      return true
    } catch {
      return false
    }
  })())
  check('no storage: isStorageAvailable reports false', isStorageAvailable() === false)
  check('no storage: listSlots returns []', Array.isArray(listSlots()) && listSlots().length === 0)
}

// =============================================================================
// 8. Named slots — CRUD
// =============================================================================
{
  installStorage(new MockStorage())
  flushWorkingConfig()

  check('slots: empty store lists nothing', listSlots().length === 0)
  check('slots: loading a nonexistent slot returns null', loadSlot('nope') === null)
  check('slots: deleting a nonexistent slot returns false', deleteSlot('nope') === false)

  check('slots: saveSlot rejects an empty name', saveSlot('', sampleConfig()) === false)
  check('slots: saveSlot rejects a non-object config', saveSlot('bad', null) === false)

  const alpha = sampleConfig({ name: 'alpha design' })
  const beta = sampleConfig({ name: 'beta design' })
  check('slots: save "alpha" succeeds', saveSlot('alpha', alpha) === true)
  // Force a distinguishable savedAt ordering (Date.now() alone can tie within a ms).
  await sleep(5)
  check('slots: save "beta" succeeds', saveSlot('beta', beta) === true)

  const listed = listSlots()
  check('slots: list returns both slots', listed.length === 2, `got ${listed.length}`)
  check(
    'slots: list is newest-first',
    listed[0].name === 'beta' && listed[1].name === 'alpha',
    `got order ${listed.map((s) => s.name).join(', ')}`,
  )
  check(
    'slots: each entry carries a savedAt timestamp',
    listed.every((s) => typeof s.savedAt === 'number'),
  )

  const loadedAlpha = loadSlot('alpha')
  check('slots: loadSlot round-trips byte-identically', JSON.stringify(loadedAlpha) === JSON.stringify(alpha))

  // Overwrite: saving "alpha" again under the same name replaces it, not duplicates it.
  const alpha2 = sampleConfig({ name: 'alpha design v2' })
  saveSlot('alpha', alpha2)
  check('slots: re-saving a name overwrites, does not duplicate', listSlots().length === 2)
  check('slots: the overwritten slot loads the new content', loadSlot('alpha')?.name === 'alpha design v2')

  check('slots: deleteSlot removes it', deleteSlot('alpha') === true)
  check('slots: deleted slot no longer lists', listSlots().length === 1 && listSlots()[0].name === 'beta')
  check('slots: deleted slot no longer loads', loadSlot('alpha') === null)

  // Corrupt a slot directly and confirm it is discarded rather than crashing.
  const storage = globalThis.localStorage
  const rawSlots = JSON.parse(storage.getItem(SLOTS_KEY))
  rawSlots.slots.beta.config.version = 999
  storage.setItem(SLOTS_KEY, JSON.stringify(rawSlots))
  check(
    'slots: a slot whose config version no longer matches is discarded on load',
    loadSlot('beta') === null,
  )
}

// =============================================================================
// 9. Working-config autosave and named slots do not share state
// =============================================================================
{
  installStorage(new MockStorage())
  flushWorkingConfig()

  saveWorkingConfig(sampleConfig({ name: 'working' }), 10)
  saveSlot('parked', sampleConfig({ name: 'parked' }))
  await sleep(30)
  check(
    'isolation: the working config and a named slot are independent',
    loadWorkingConfig()?.name === 'working' && loadSlot('parked')?.name === 'parked',
  )
  check('isolation: the working config is not itself listed as a slot', listSlots().length === 1)
}

// =============================================================================
// Summary
// =============================================================================
flushWorkingConfig()
removeStorage()

console.log('')
console.log(`test-persistence: ${passed} checks passed, ${failures.length} failed`)
if (failures.length > 0) {
  console.error('')
  console.error('Failures:')
  for (const f of failures) console.error(`  - ${f}`)
  process.exit(1)
}
