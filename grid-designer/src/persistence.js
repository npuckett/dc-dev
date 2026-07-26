/**
 * grid-designer — localStorage persistence for the working config + named slots.
 *
 * NOT in the headless zone (src/core/): this module is allowed ordinary browser
 * APIs, but it must still work when they are missing, throwing, or lying —
 * private browsing can make `setItem` throw, a full storage quota can make it
 * throw mid-session, and some embedders / test runners have no `localStorage`
 * at all. EVERY access is therefore wrapped in try/catch and degrades to a
 * no-op / `null`. This module must never throw, and never be the reason a
 * render breaks.
 *
 * Deliberately dependency-free and standalone: it does NOT import anything
 * from core/ (or anywhere else in src/), so it can be exercised in plain node
 * against a stubbed `localStorage` with zero setup — see tests/test-persistence.mjs.
 * That also means it re-derives its own tiny "is this a plausible config
 * object" checks rather than borrowing schema.js's; the store is what actually
 * runs `validateConfig` before trusting anything this module hands back.
 *
 * =============================================================================
 * STORAGE KEYS
 * =============================================================================
 *   grid-designer:working   the autosaved in-progress config (one envelope)
 *   grid-designer:slots     named parking slots (one envelope holding a map)
 *
 * Both envelopes carry a `schemaVersion` tag (see EXPECTED_CONFIG_VERSION)
 * independent of `config.version`. Two layers of the same defence: the whole
 * envelope is discarded if ITS tag is stale, and the embedded config's own
 * `version` field is checked again before it is handed back. Bump
 * EXPECTED_CONFIG_VERSION the moment the schema pivots (the v3 rework is
 * imminent — see HANDOFF.md) so a v2 save is discarded rather than resurrected
 * into a v3 store. That is a deliberate, narrow duplication of the `version`
 * check in core/schema.js's `validateConfig` — kept separate on purpose, so
 * this module never has to import core/.
 *
 * =============================================================================
 * THE WORKING-CONFIG DEBOUNCE
 * =============================================================================
 * `saveWorkingConfig(config)` is what the store calls after every commit,
 * including every tick of a dragged slider. Writing on every call would hammer
 * localStorage (synchronous, and not free), so the write is debounced ~300ms
 * trailing: each call remembers the latest config and resets a timer; only the
 * LAST config passed before 300ms of quiet actually gets written. Because the
 * timer is always reset to the newest value (never dropped in favour of an
 * older in-flight one), the final value is never lost as long as the tab stays
 * alive for the quiet window. `flushWorkingConfig()` writes any pending value
 * immediately and is wired to `beforeunload` / `pagehide` below as a second
 * line of defence for the case a reload fires inside that window.
 */

const NS = 'grid-designer'
export const WORKING_KEY = `${NS}:working`
export const SLOTS_KEY = `${NS}:slots`

/** Debounce delay for `saveWorkingConfig`, ms. Exported so tests can be quick. */
export const AUTOSAVE_DELAY_MS = 300

/**
 * The config `version` this build accepts. Mirrors (deliberately, independently
 * of) core/v3/schema.js's `version !== 3` check. Bumped from 2 to 3 for the v3
 * drift-surface pivot (V3_SPEC.md §6 / §7): v2 (per-column fold strip) and v3
 * (one tiled drift surface) configs describe physically different objects and
 * do not map onto one another, so a v2 save must be DISCARDED here, never
 * resurrected into a v3 store — same reasoning as v1→v2's bump before it.
 */
const EXPECTED_CONFIG_VERSION = 3

const isPlainObject = (v) => typeof v === 'object' && v !== null && !Array.isArray(v)

// -----------------------------------------------------------------------------
// Storage access — every call site funnels through here so the try/catch lives
// in exactly one place.
// -----------------------------------------------------------------------------

/** The storage object to use, or null if unavailable for any reason. */
function getStorage() {
  try {
    // eslint-disable-next-line no-undef -- guarded by typeof, safe outside browsers/DOM lib
    if (typeof localStorage === 'undefined' || localStorage === null) return null
    return localStorage
  } catch {
    return null
  }
}

/** `localStorage.getItem`, degrading to null on any failure. */
function safeGet(key) {
  const storage = getStorage()
  if (!storage) return null
  try {
    return storage.getItem(key)
  } catch {
    return null
  }
}

/** `localStorage.setItem`, degrading to false (no-op) on any failure. */
function safeSet(key, value) {
  const storage = getStorage()
  if (!storage) return false
  try {
    storage.setItem(key, value)
    return true
  } catch {
    return false
  }
}

/** `localStorage.removeItem`, degrading to false on any failure. */
function safeRemove(key) {
  const storage = getStorage()
  if (!storage) return false
  try {
    storage.removeItem(key)
    return true
  } catch {
    return false
  }
}

function safeParse(text) {
  try {
    return JSON.parse(text)
  } catch {
    return null
  }
}

/** Read + JSON.parse a key, returning null unless the result is a plain object. */
function readEnvelope(key) {
  const raw = safeGet(key)
  if (typeof raw !== 'string' || raw.length === 0) return null
  const parsed = safeParse(raw)
  return isPlainObject(parsed) ? parsed : null
}

/** JSON.stringify + write; returns whether the write succeeded. */
function writeEnvelope(key, envelope) {
  let text
  try {
    text = JSON.stringify(envelope)
  } catch {
    return false // circular / non-serializable — nothing sane to write
  }
  return safeSet(key, text)
}

/**
 * Is storage actually usable right now? A real probe (set + remove a throwaway
 * key), not just presence — private browsing in some engines exposes
 * `localStorage` but throws on first write. Used by the UI for the "autosave
 * is on" confirmation, so that confirmation tells the truth.
 *
 * @returns {boolean}
 */
export function isStorageAvailable() {
  const storage = getStorage()
  if (!storage) return false
  const probeKey = `${NS}:__probe__`
  try {
    storage.setItem(probeKey, '1')
    storage.removeItem(probeKey)
    return true
  } catch {
    return false
  }
}

// -----------------------------------------------------------------------------
// Working config
// -----------------------------------------------------------------------------

/**
 * The autosaved working config, or null if there isn't one / it's corrupt /
 * unparseable / from a schema version this build no longer accepts.
 *
 * Deliberately permissive about WHAT it returns (any plain object with the
 * right `version`) and strict about discarding anything else — the caller
 * (the store) is the one that runs the real `validateConfig` before trusting
 * this. This function's job is just to never hand back garbage that would
 * throw downstream, and to never resurrect a config from a retired schema.
 *
 * @returns {object|null}
 */
export function loadWorkingConfig() {
  const envelope = readEnvelope(WORKING_KEY)
  if (!envelope) return null
  if (envelope.schemaVersion !== EXPECTED_CONFIG_VERSION) return null
  const config = envelope.config
  if (!isPlainObject(config)) return null
  if (config.version !== EXPECTED_CONFIG_VERSION) return null
  return config
}

/** Immediate (non-debounced) write of the working-config envelope. */
function writeWorkingConfigNow(config) {
  if (!isPlainObject(config)) return false
  return writeEnvelope(WORKING_KEY, {
    schemaVersion: EXPECTED_CONFIG_VERSION,
    savedAt: Date.now(),
    config,
  })
}

let saveTimer = null
let pendingConfig = null

/**
 * Debounced-safe write of the working config: call this on every commit,
 * including every tick of a dragged slider, without worrying about hammering
 * localStorage. Only the LAST config passed before `delayMs` of quiet is
 * actually written — see the module doc comment for why that never loses the
 * final value under normal use, and `flushWorkingConfig` for the reload edge.
 *
 * @param {object} config
 * @param {number} [delayMs] override for tests; defaults to AUTOSAVE_DELAY_MS
 */
export function saveWorkingConfig(config, delayMs = AUTOSAVE_DELAY_MS) {
  pendingConfig = config
  if (typeof setTimeout !== 'function') {
    // No timer API at all (shouldn't happen anywhere this runs) — write now
    // rather than silently dropping the save.
    writeWorkingConfigNow(pendingConfig)
    pendingConfig = null
    return
  }
  if (saveTimer !== null) clearTimeout(saveTimer)
  saveTimer = setTimeout(() => {
    saveTimer = null
    const toWrite = pendingConfig
    pendingConfig = null
    writeWorkingConfigNow(toWrite)
  }, delayMs)
}

/**
 * Write any pending debounced save immediately and cancel the timer. Safe to
 * call with nothing pending (no-op). Wired to `beforeunload` / `pagehide`
 * below so a reload that lands inside the debounce window doesn't lose the
 * last change.
 */
export function flushWorkingConfig() {
  if (saveTimer !== null) {
    clearTimeout(saveTimer)
    saveTimer = null
  }
  if (pendingConfig !== null) {
    const toWrite = pendingConfig
    pendingConfig = null
    writeWorkingConfigNow(toWrite)
  }
}

/** Best-effort: flush before the page goes away. No-ops outside a browser. */
try {
  if (typeof window !== 'undefined' && typeof window.addEventListener === 'function') {
    window.addEventListener('beforeunload', flushWorkingConfig)
    window.addEventListener('pagehide', flushWorkingConfig)
  }
} catch {
  // never let wiring this up break module load
}

// -----------------------------------------------------------------------------
// Named slots
// -----------------------------------------------------------------------------

function readSlots() {
  const envelope = readEnvelope(SLOTS_KEY)
  if (!envelope) return {}
  if (envelope.schemaVersion !== EXPECTED_CONFIG_VERSION) return {}
  return isPlainObject(envelope.slots) ? envelope.slots : {}
}

function writeSlots(slots) {
  return writeEnvelope(SLOTS_KEY, { schemaVersion: EXPECTED_CONFIG_VERSION, slots })
}

/**
 * The saved slots, newest first.
 *
 * @returns {Array<{name: string, savedAt: number|null}>}
 */
export function listSlots() {
  const slots = readSlots()
  return Object.keys(slots)
    .map((name) => {
      const entry = slots[name]
      const savedAt = isPlainObject(entry) && Number.isFinite(entry.savedAt) ? entry.savedAt : null
      return { name, savedAt }
    })
    .sort((a, b) => (b.savedAt ?? 0) - (a.savedAt ?? 0))
}

/**
 * Save `config` under a named slot, overwriting any existing slot of the same
 * name.
 *
 * @param {string} name non-empty slot name
 * @param {object} config
 * @returns {boolean} whether the write succeeded
 */
export function saveSlot(name, config) {
  if (typeof name !== 'string' || name.trim().length === 0) return false
  if (!isPlainObject(config)) return false
  const slots = readSlots()
  slots[name] = { savedAt: Date.now(), config }
  return writeSlots(slots)
}

/**
 * Load a named slot's config, or null if it doesn't exist / is corrupt / is
 * from a schema version this build no longer accepts. Same permissive-shape /
 * strict-version contract as `loadWorkingConfig`.
 *
 * @param {string} name
 * @returns {object|null}
 */
export function loadSlot(name) {
  if (typeof name !== 'string' || name.length === 0) return null
  const slots = readSlots()
  const entry = slots[name]
  if (!isPlainObject(entry)) return null
  const config = entry.config
  if (!isPlainObject(config)) return null
  if (config.version !== EXPECTED_CONFIG_VERSION) return null
  return config
}

/**
 * Delete a named slot.
 *
 * @param {string} name
 * @returns {boolean} whether a slot of that name existed and was removed
 */
export function deleteSlot(name) {
  if (typeof name !== 'string' || name.length === 0) return false
  const slots = readSlots()
  if (!(name in slots)) return false
  delete slots[name]
  return writeSlots(slots)
}
