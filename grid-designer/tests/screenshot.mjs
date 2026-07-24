/**
 * grid-designer — browser smoke test / screenshot harness (WP6, schema v2).
 *
 * Self-contained: it starts its OWN dev server (`npm run dev`, port 5175 via
 * vite.config.js strictPort), waits for it, drives the UI, and tears the server
 * down on every exit path.
 *
 * =============================================================================
 * THE FLOW IT DRIVES (the column-strip design loop, end to end)
 * =============================================================================
 *   01-flat.png    default state — the flat reference surface, 30 panels, no
 *                  flagged joints
 *   02-wave.png    the Wave preset: six phase-shifted fold sequences (asserted
 *                  against core/presets.js itself) plus its two rigid plates
 *   03-fold.png    one hinge dragged: column 3's 0→1 joint to 80°
 *   04-rects.png   the two preset plates removed, "Shift →" phase-shifted the
 *                  whole grid one column, then a HORIZONTAL and a VERTICAL plate
 *                  placed on cells the harness COMPUTES to be legal under the
 *                  current folds
 *   05-reject.png  two illegal placements in a row — an H plate across two
 *                  columns at different pitches (E_CROSSCOL_ANGLE_MISMATCH) and a
 *                  V plate over a folded joint (E_FOLD_ON_REMOVED_JOINT). Nothing
 *                  moved; the map explains why inline.
 *   06-json.png    after pasting a MINIMAL v2 config (columns only — everything
 *                  else defaulted by normalizeConfig) and hitting Apply
 *
 * Placement cells are never hard-coded: legal / illegal candidates are derived
 * from the live config's fold sequences and the solved `columnChains[c].pitchesDeg`
 * at the moment of the click, so the script keeps testing the CONSTRAINTS rather
 * than a lucky coordinate.
 *
 * =============================================================================
 * WHAT IT ASSERTS (any failure ⇒ non-zero exit)
 * =============================================================================
 *   - no page errors and no console errors
 *   - the WebGL canvas is non-blank (pixels sampled through a 2D canvas; the r3f
 *     Canvas runs with preserveDrawingBuffer so the read-back is valid) and the
 *     rendered pixels actually CHANGE at each design step
 *   - the Wave preset's per-column foldsDeg equal buildPreset('wave')'s
 *   - a hinge slider (or setColumnFold) moves one column's fold and the block's
 *     cumulative-pitch readout follows
 *   - "Shift →" rotates every column's fold sequence by one index, wrapping
 *   - grid-map clicks add / remove plates, and reject the two v2 rect
 *     constraints with the right codes surfaced inline next to the map (and NOT
 *     duplicated in the control panel's general error box)
 *   - a vertical plate DISABLES the slider for the joint it removes
 *   - the JSON panel applies a minimal v2 config and REJECTS a v1 one with the
 *     "version must be 2" message
 *   - Export OBJ / Download JSON fire real downloads with the expected names, the
 *     OBJ payload contains `o panel_r0_c0`, one object per panel, and the JSON is
 *     the current fully-defaulted v2 config
 *   - consecutive screenshots differ in bytes (the render responded each time)
 *
 * Usage: node tests/screenshot.mjs
 */

import { chromium } from 'playwright'
import { spawn } from 'node:child_process'
import { existsSync, mkdirSync, readFileSync, readdirSync, rmSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { buildPreset } from '../src/core/presets.js'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const OUT = resolve(ROOT, 'tests/screenshots')
const PORT = 5175
const URL = `http://localhost:${PORT}/`
const VIEWPORT = { width: 1440, height: 900 }
const SETTLE_MS = 800

const results = []
const pass = (name, detail = '') => results.push({ ok: true, name, detail })
const fail = (name, detail = '') => results.push({ ok: false, name, detail })
const check = (ok, name, detail = '') => (ok ? pass(name, detail) : fail(name, detail))

// -----------------------------------------------------------------------------
// dev server
// -----------------------------------------------------------------------------
async function serverUp() {
  try {
    const res = await fetch(URL, { method: 'GET' })
    return res.ok
  } catch {
    return false
  }
}

async function startDevServer() {
  if (await serverUp()) {
    console.log(`! a server is already listening on ${PORT} — reusing it`)
    return { child: null, reused: true }
  }

  console.log('Starting dev server (npm run dev) ...')
  const child = spawn('npm', ['run', 'dev'], {
    cwd: ROOT,
    detached: true,
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  let log = ''
  child.stdout.on('data', (d) => (log += d))
  child.stderr.on('data', (d) => (log += d))

  let exited = false
  child.on('exit', () => (exited = true))

  const deadline = Date.now() + 60_000
  while (Date.now() < deadline) {
    if (await serverUp()) {
      console.log(`Dev server ready at ${URL}`)
      return { child, reused: false }
    }
    if (exited) throw new Error(`dev server exited early:\n${log}`)
    await new Promise((r) => setTimeout(r, 300))
  }
  throw new Error(`dev server did not come up on ${PORT} within 60s:\n${log}`)
}

function stopDevServer(handle) {
  if (!handle?.child || handle.reused) return
  try {
    // Kill the whole process group — npm spawns vite as a child.
    process.kill(-handle.child.pid, 'SIGTERM')
  } catch {
    try {
      handle.child.kill('SIGKILL')
    } catch {
      /* already gone */
    }
  }
}

// -----------------------------------------------------------------------------
// canvas sampling
// -----------------------------------------------------------------------------
/** Sample the WebGL canvas through a 2D canvas and summarize its pixels. */
function canvasStats(page) {
  return page.evaluate(() => {
    const c = document.querySelector('canvas')
    if (!c) return { error: 'no canvas element' }
    const SW = 240
    const SH = Math.max(1, Math.round((SW * c.height) / c.width))
    const off = document.createElement('canvas')
    off.width = SW
    off.height = SH
    const ctx = off.getContext('2d', { willReadFrequently: true })
    ctx.drawImage(c, 0, 0, SW, SH)
    const d = ctx.getImageData(0, 0, SW, SH).data
    const n = SW * SH
    let sum = 0
    let sumSq = 0
    let bright = 0
    const colors = new Set()
    for (let i = 0; i < d.length; i += 4) {
      const l = 0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2]
      sum += l
      sumSq += l * l
      if (l > 60) bright++
      colors.add(((d[i] >> 4) << 8) | ((d[i + 1] >> 4) << 4) | (d[i + 2] >> 4))
    }
    const mean = sum / n
    return {
      width: c.width,
      height: c.height,
      mean,
      std: Math.sqrt(Math.max(0, sumSq / n - mean * mean)),
      brightFraction: bright / n,
      distinctColors: colors.size,
    }
  })
}

/**
 * Assert the canvas is rendering something, and return a fingerprint (pixel
 * summary + a PNG of the canvas element alone) for change detection.
 */
async function assertCanvasRenders(page, label) {
  let stats = null
  const deadline = Date.now() + 15_000
  while (Date.now() < deadline) {
    stats = await canvasStats(page)
    if (!stats.error && stats.distinctColors >= 5 && stats.brightFraction > 0.005) break
    await page.waitForTimeout(400)
  }
  const detail = stats?.error
    ? stats.error
    : `${stats.width}×${stats.height} mean=${stats.mean.toFixed(1)} std=${stats.std.toFixed(1)} ` +
      `bright=${(stats.brightFraction * 100).toFixed(1)}% colors=${stats.distinctColors}`
  check(
    !stats?.error && stats.distinctColors >= 5 && stats.brightFraction > 0.005 && stats.std > 3,
    `canvas non-blank (${label})`,
    detail,
  )
  const buf = await page.locator('canvas').screenshot()
  return { label, stats, buf }
}

/** Did the 3D render actually change between two fingerprints? */
function canvasChanged(a, b) {
  if (!a || !b) return false
  if (!a.buf.equals(b.buf)) return true
  const x = a.stats
  const y = b.stats
  if (!x || !y || x.error || y.error) return false
  return (
    Math.abs(x.mean - y.mean) > 0.05 ||
    Math.abs(x.std - y.std) > 0.05 ||
    x.distinctColors !== y.distinctColors ||
    Math.abs(x.brightFraction - y.brightFraction) > 0.0005
  )
}

// -----------------------------------------------------------------------------
// store snapshot (schema v2)
// -----------------------------------------------------------------------------
const storeState = (page) =>
  page.evaluate(() => {
    const s = window.__gridDesignerStore.getState()
    const { layout, report } = window.__gridDesignerDerived()
    return {
      name: s.config.name,
      preset: s.config.meta?.preset,
      version: s.config.version,
      cols: s.config.grid.cols,
      rows: s.config.grid.rows,
      gap: s.config.gap,
      folds: s.config.columns.map((col) => col.foldsDeg.slice()),
      pitches: layout.columnChains.map((chain) => chain.pitchesDeg.slice()),
      rectObjs: s.config.rects.map((r) => ({ ...r })),
      rects: s.config.rects.map((r) => `${r.orientation[0]}(${r.row},${r.col})`),
      lastErrors: s.lastErrors.map((e) => e.code),
      lastErrorMessages: s.lastErrors.map((e) => e.message),
      layoutWarnings: layout.warnings.map((w) => w.code),
      summary: report.summary,
      panels: layout.panels.length,
    }
  })

// -----------------------------------------------------------------------------
// placement candidate search — derived from the LIVE config, never hard-coded
// -----------------------------------------------------------------------------
const PITCH_EPS = 0.1

/** Every cell any rect already covers, as a Set of "r,c". */
function occupiedCells(rectObjs) {
  const set = new Set()
  for (const r of rectObjs) {
    if (r.orientation === 'horizontal') {
      set.add(`${r.row},${r.col}`)
      set.add(`${r.row},${r.col + 1}`)
    } else {
      set.add(`${r.row},${r.col}`)
      set.add(`${r.row + 1},${r.col}`)
    }
  }
  return set
}

/** A HORIZONTAL plate is legal where the two columns' pitches agree at that row. */
function findLegalHorizontal(state) {
  const busy = occupiedCells(state.rectObjs)
  for (let r = 0; r < state.rows; r++) {
    for (let c = 0; c + 1 < state.cols; c++) {
      if (busy.has(`${r},${c}`) || busy.has(`${r},${c + 1}`)) continue
      if (Math.abs(state.pitches[c][r] - state.pitches[c + 1][r]) <= PITCH_EPS) return { row: r, col: c }
    }
  }
  return null
}

/** An H plate is ILLEGAL where the two columns sit at different pitches. */
function findIllegalHorizontal(state) {
  const busy = occupiedCells(state.rectObjs)
  for (let r = 0; r < state.rows; r++) {
    for (let c = 0; c + 1 < state.cols; c++) {
      if (busy.has(`${r},${c}`) || busy.has(`${r},${c + 1}`)) continue
      if (Math.abs(state.pitches[c][r] - state.pitches[c + 1][r]) > PITCH_EPS) return { row: r, col: c }
    }
  }
  return null
}

/** A VERTICAL plate is legal only over an UNFOLDED joint (it removes that joint). */
function findLegalVertical(state) {
  const busy = occupiedCells(state.rectObjs)
  for (let r = 0; r + 1 < state.rows; r++) {
    for (let c = 0; c < state.cols; c++) {
      if (busy.has(`${r},${c}`) || busy.has(`${r + 1},${c}`)) continue
      if (state.folds[c][r] === 0) return { row: r, col: c }
    }
  }
  return null
}

/** A V plate is ILLEGAL over a folded joint — the rigid plate cannot bend. */
function findIllegalVertical(state) {
  const busy = occupiedCells(state.rectObjs)
  for (let r = 0; r + 1 < state.rows; r++) {
    for (let c = 0; c < state.cols; c++) {
      if (busy.has(`${r},${c}`) || busy.has(`${r + 1},${c}`)) continue
      if (state.folds[c][r] !== 0) return { row: r, col: c }
    }
  }
  return null
}

const shiftedRight = (folds) =>
  folds.map((_, c) => folds[(c - 1 + folds.length) % folds.length].slice())

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------
let server = null
let browser = null

try {
  if (!existsSync(OUT)) mkdirSync(OUT, { recursive: true })
  // Drop stale numbering from earlier work packages so the directory is exactly
  // what this run produced.
  for (const f of readdirSync(OUT)) {
    if (f.endsWith('.png')) rmSync(resolve(OUT, f))
  }
  server = await startDevServer()

  browser = await chromium.launch({ headless: true })
  const page = await browser.newPage({ viewport: VIEWPORT, acceptDownloads: true })

  const consoleErrors = []
  const pageErrors = []
  page.on('console', (msg) => {
    if (msg.type() === 'error') consoleErrors.push(msg.text())
  })
  page.on('pageerror', (err) => pageErrors.push(String(err)))

  console.log(`Loading ${URL} ...`)
  await page.goto(URL, { waitUntil: 'load' })
  await page.waitForSelector('canvas', { state: 'visible', timeout: 30_000 })
  await page.waitForFunction(() => !!window.__gridDesignerStore, null, { timeout: 15_000 })
  await page.waitForTimeout(1200) // let r3f settle

  // --- (a) flat default ----------------------------------------------------
  const flatShot = await assertCanvasRenders(page, 'flat')
  const flatState = await storeState(page)
  check(flatState.preset === 'flat', 'default preset is flat', `preset=${flatState.preset}`)
  check(flatState.version === 2, 'config is schema v2', `version=${flatState.version}`)
  check(
    flatState.panels === 30 && flatState.summary.flagged === 0,
    'flat default: 30 panels, no flagged joints',
    `panels=${flatState.panels} flagged=${flatState.summary.flagged}/${flatState.summary.total}`,
  )
  check(
    flatState.folds.length === 6 && flatState.folds.every((f) => f.length === 4 && f.every((v) => v === 0)),
    'flat default: 6 columns × 4 hinges, all 0°',
    JSON.stringify(flatState.folds),
  )

  // the column UI itself
  check(
    await page.isVisible('[data-testid="column-0"]') && await page.isVisible('[data-testid="column-5"]'),
    'one control block per column (0..5) rendered',
  )
  check(
    await page.isVisible('[data-testid="profile-0"]') && await page.isVisible('[data-testid="profile-5"]'),
    'each column block draws its fold-profile sparkline',
  )
  check(
    await page.isVisible('[data-testid="column-tools"]') &&
      await page.isVisible('[data-testid="tool-shift-right"]') &&
      await page.isVisible('[data-testid="tool-flatten"]') &&
      await page.isVisible('[data-testid="tool-copy-all"]'),
    'cross-column toolbar (copy / shift / flatten) is present',
  )
  check(
    (await page.textContent('[data-testid="column-0"]')).includes('right'),
    'column 0 is labelled as the RIGHT strip (the camera looks in from the shore)',
  )
  await page.screenshot({ path: `${OUT}/01-flat.png` })
  console.log('  → tests/screenshots/01-flat.png')

  // --- (b) wave preset -----------------------------------------------------
  const expectedWaveFolds = buildPreset('wave').columns.map((col) => col.foldsDeg)
  await page.click('[data-testid="preset-wave"]')
  await page.waitForTimeout(SETTLE_MS)
  const waveShot = await assertCanvasRenders(page, 'wave')
  const waveState = await storeState(page)
  check(waveState.preset === 'wave', 'Wave preset applied', `preset=${waveState.preset}`)
  check(
    JSON.stringify(waveState.folds) === JSON.stringify(expectedWaveFolds),
    "wave's per-column fold sequences match core/presets.js",
    JSON.stringify(waveState.folds),
  )
  check(
    JSON.stringify(waveState.rects) === JSON.stringify(['v(1,2)', 'h(1,0)']),
    'wave brought its two rigid plates',
    JSON.stringify(waveState.rects),
  )
  check(
    waveState.summary.flagged > 0,
    'wave flags in-row joints where the phase-shifted columns diverge',
    `flagged=${waveState.summary.flagged}/${waveState.summary.total}`,
  )
  check(canvasChanged(flatShot, waveShot), 'wave render differs from flat')
  check(
    await page.isDisabled('[data-testid="fold-2-1"]'),
    "the wave's vertical plate at (1,2) disabled column 2's 1→2 slider",
  )
  await page.screenshot({ path: `${OUT}/02-wave.png` })
  console.log('  → tests/screenshots/02-wave.png')

  // --- (c) drag one hinge: column 3, joint 0 → 80° -------------------------
  await page.locator('[data-testid="fold-3-0"]').fill('80')
  await page.waitForTimeout(SETTLE_MS)
  let foldState = await storeState(page)
  if (foldState.folds[3][0] !== 80) {
    console.log('! slider fill did not register — falling back to the store action')
    await page.evaluate(() => window.__gridDesignerStore.getState().setColumnFold(3, 0, 80))
    await page.waitForTimeout(SETTLE_MS)
    foldState = await storeState(page)
  }
  const foldShot = await assertCanvasRenders(page, 'fold')
  check(
    foldState.folds[3][0] === 80 && foldState.folds[3][0] !== waveState.folds[3][0],
    "column 3's 0→1 hinge set to 80°",
    `${waveState.folds[3][0]}° → ${foldState.folds[3][0]}°`,
  )
  check(
    JSON.stringify(foldState.folds.filter((_, c) => c !== 3)) ===
      JSON.stringify(waveState.folds.filter((_, c) => c !== 3)),
    'the other five columns were untouched',
  )
  check(
    (await page.textContent('[data-testid="column-3"]')).includes('80°'),
    "column 3's cumulative-pitch readout followed the hinge",
    `pitches=${JSON.stringify(foldState.pitches[3])}`,
  )
  check(canvasChanged(waveShot, foldShot), 'fold render differs from wave')
  check(foldState.lastErrors.length === 0, 'no rejected changes so far', foldState.lastErrors.join(', '))
  await page.screenshot({ path: `${OUT}/03-fold.png` })
  console.log('  → tests/screenshots/03-fold.png')

  // --- (d) remove the preset plates, then "Shift →" ------------------------
  // The plates pin folds (a V plate's joint must stay 0, an H plate's two columns
  // must stay at equal pitch), so a whole-grid phase shift is a rect-free move.
  check(await page.isVisible('[data-testid="grid-map"]'), 'grid map is visible')
  await page.click('[data-testid="cell-1-2"]') // the vertical plate
  await page.click('[data-testid="cell-1-0"]') // the horizontal plate
  await page.waitForTimeout(SETTLE_MS)
  const clearedState = await storeState(page)
  check(
    clearedState.rects.length === 0 && clearedState.panels === 30,
    'grid-map clicks removed both plates',
    `rects=${JSON.stringify(clearedState.rects)} panels=${clearedState.panels}`,
  )

  await page.click('[data-testid="tool-shift-right"]')
  await page.waitForTimeout(SETTLE_MS)
  const shiftState = await storeState(page)
  check(
    JSON.stringify(shiftState.folds) === JSON.stringify(shiftedRight(clearedState.folds)),
    '"Shift →" rotated every column\'s fold sequence one index, wrapping',
    JSON.stringify(shiftState.folds),
  )
  check(
    shiftState.lastErrors.length === 0,
    'the shift committed through validation',
    shiftState.lastErrors.join(', '),
  )

  // --- (e) place a legal H plate, then a legal V plate ---------------------
  const hCell = findLegalHorizontal(shiftState)
  check(!!hCell, 'found a row where two neighbouring columns agree in pitch (H candidate)', JSON.stringify(hCell))
  if (!hCell) throw new Error('no legal horizontal placement under the current folds')
  await page.click('[data-testid="gridmap-mode-horizontal"]')
  await page.click(`[data-testid="cell-${hCell.row}-${hCell.col}"]`)
  await page.waitForTimeout(SETTLE_MS)
  const rectHState = await storeState(page)
  check(
    rectHState.rects.includes(`h(${hCell.row},${hCell.col})`) && rectHState.lastErrors.length === 0,
    `horizontal plate placed at (${hCell.row}, ${hCell.col})`,
    `rects=${JSON.stringify(rectHState.rects)} errors=${JSON.stringify(rectHState.lastErrors)}`,
  )
  check(
    rectHState.panels === shiftState.panels - 1,
    'horizontal plate merged two cells into one panel',
    `${shiftState.panels} → ${rectHState.panels} panels`,
  )

  const vCell = findLegalVertical(rectHState)
  check(!!vCell, 'found an unfolded joint for a V plate', JSON.stringify(vCell))
  if (!vCell) throw new Error('no legal vertical placement under the current folds')
  await page.click('[data-testid="gridmap-mode-vertical"]')
  await page.click(`[data-testid="cell-${vCell.row}-${vCell.col}"]`)
  await page.waitForTimeout(SETTLE_MS)
  const rectVState = await storeState(page)
  check(
    rectVState.rects.includes(`v(${vCell.row},${vCell.col})`) && rectVState.lastErrors.length === 0,
    `vertical plate placed at (${vCell.row}, ${vCell.col}) over an unfolded joint`,
    `rects=${JSON.stringify(rectVState.rects)} errors=${JSON.stringify(rectVState.lastErrors)}`,
  )
  check(
    rectVState.panels === rectHState.panels - 1,
    'vertical plate merged two cells across two rows',
    `${rectHState.panels} → ${rectVState.panels} panels`,
  )
  check(
    await page.isDisabled(`[data-testid="fold-${vCell.col}-${vCell.row}"]`),
    `the new vertical plate disabled column ${vCell.col}'s ${vCell.row}→${vCell.row + 1} slider`,
  )
  const rectsShot = await assertCanvasRenders(page, 'rects')
  check(canvasChanged(foldShot, rectsShot), 'plates + shift render differs from the fold state')
  await page.screenshot({ path: `${OUT}/04-rects.png` })
  console.log('  → tests/screenshots/04-rects.png')

  // --- (f) the two illegal placements --------------------------------------
  const badH = findIllegalHorizontal(rectVState)
  check(!!badH, 'found a row where two columns DISAGREE in pitch (illegal H)', JSON.stringify(badH))
  if (badH) {
    await page.click('[data-testid="gridmap-mode-horizontal"]')
    await page.click(`[data-testid="cell-${badH.row}-${badH.col}"]`)
    await page.waitForTimeout(600)
    const badHState = await storeState(page)
    check(
      JSON.stringify(badHState.rects) === JSON.stringify(rectVState.rects) &&
        badHState.panels === rectVState.panels,
      'illegal horizontal placement left the design untouched',
      `rects=${JSON.stringify(badHState.rects)}`,
    )
    check(
      badHState.lastErrors.includes('E_CROSSCOL_ANGLE_MISMATCH'),
      'illegal horizontal reported E_CROSSCOL_ANGLE_MISMATCH',
      `lastErrors=${JSON.stringify(badHState.lastErrors)}`,
    )
  }

  const badV = findIllegalVertical(rectVState)
  check(!!badV, 'found a folded joint to try a V plate on (illegal V)', JSON.stringify(badV))
  if (badV) {
    await page.click('[data-testid="gridmap-mode-vertical"]')
    await page.click(`[data-testid="cell-${badV.row}-${badV.col}"]`)
    await page.waitForTimeout(600)
    const badVState = await storeState(page)
    check(
      JSON.stringify(badVState.rects) === JSON.stringify(rectVState.rects) &&
        JSON.stringify(badVState.folds) === JSON.stringify(rectVState.folds) &&
        badVState.panels === rectVState.panels,
      'illegal vertical placement left the design untouched',
      `rects=${JSON.stringify(badVState.rects)} panels=${badVState.panels}`,
    )
    check(
      badVState.lastErrors.includes('E_FOLD_ON_REMOVED_JOINT'),
      'illegal vertical reported E_FOLD_ON_REMOVED_JOINT',
      `lastErrors=${JSON.stringify(badVState.lastErrors)}`,
    )
    const inlineText = (await page.textContent('[data-testid="gridmap-error"]')) ?? ''
    check(
      inlineText.includes('E_FOLD_ON_REMOVED_JOINT') && inlineText.includes('rigid plate'),
      'the full rejection message is shown inline under the grid map',
      `${inlineText.length} chars`,
    )
    check(
      !(await page.isVisible('[data-testid="last-errors"]')),
      'the general error box does NOT repeat the plate rejection',
    )
  }
  await page.screenshot({ path: `${OUT}/05-reject.png` })
  console.log('  → tests/screenshots/05-reject.png')

  // --- (g) JSON panel: a minimal v2 config in, a v1 config rejected --------
  // Only `columns` is load-bearing (normalizeConfig will NOT invent it — its
  // length is the grid width); everything else defaults.
  const MINIMAL_V2 = {
    version: 2,
    units: 'cm',
    name: 'pasted study',
    columns: [
      { foldsDeg: [30, 10, -20, -20] },
      { foldsDeg: [20, 25, -25, -20] },
      { foldsDeg: [10, 30, 0, -25] },
      { foldsDeg: [0, 30, 10, -20] },
      { foldsDeg: [0, 20, 25, -15] },
      { foldsDeg: [0, 10, 30, 0] },
    ],
  }
  const V1_CONFIG = {
    version: 1,
    units: 'cm',
    name: 'v1 accordion study',
    rows: [
      { zigzagDeg: 0 },
      { zigzagDeg: 25 },
      { zigzagDeg: 45 },
      { zigzagDeg: 10 },
      { zigzagDeg: 5 },
    ],
    rowFoldsDeg: [20, 35, 25, -15],
  }

  await page.click('[data-testid="json-toggle"]')
  await page.waitForTimeout(250)
  check(await page.isVisible('[data-testid="json-text"]'), 'JSON panel opened (textarea visible)')
  const syncedText = await page.inputValue('[data-testid="json-text"]')
  check(
    syncedText.includes('"version": 2') && syncedText.includes('"foldsDeg"'),
    'textarea was synced to the current v2 config on open',
    `${syncedText.length} chars`,
  )

  await page.fill('[data-testid="json-text"]', JSON.stringify(MINIMAL_V2, null, 2))
  await page.click('[data-testid="json-apply"]')
  await page.waitForTimeout(SETTLE_MS)
  const jsonState = await storeState(page)
  check(jsonState.name === 'pasted study', 'minimal v2 config applied (name took effect)', `name=${jsonState.name}`)
  check(
    JSON.stringify(jsonState.folds) === JSON.stringify(MINIMAL_V2.columns.map((c) => c.foldsDeg)),
    'pasted fold sequences took effect',
    JSON.stringify(jsonState.folds),
  )
  check(
    jsonState.cols === 6 &&
      jsonState.rows === 5 &&
      jsonState.gap === 2 &&
      jsonState.rects.length === 0 &&
      jsonState.panels === 30 &&
      jsonState.lastErrors.length === 0,
    'normalizeConfig filled every omitted default (6×5, gap 2, no rects, 30 panels)',
    `cols=${jsonState.cols} rows=${jsonState.rows} gap=${jsonState.gap} panels=${jsonState.panels}`,
  )
  const jsonShot = await assertCanvasRenders(page, 'json')
  check(canvasChanged(rectsShot, jsonShot), 'pasted config re-rendered the surface')
  await page.screenshot({ path: `${OUT}/06-json.png` })
  console.log('  → tests/screenshots/06-json.png')

  await page.fill('[data-testid="json-text"]', JSON.stringify(V1_CONFIG, null, 2))
  await page.click('[data-testid="json-apply"]')
  await page.waitForTimeout(600)
  const v1State = await storeState(page)
  check(
    v1State.name === 'pasted study' &&
      JSON.stringify(v1State.folds) === JSON.stringify(MINIMAL_V2.columns.map((c) => c.foldsDeg)),
    'v1 config was rejected — the v2 design is untouched',
    `name=${v1State.name}`,
  )
  check(
    v1State.lastErrors.includes('E_SHAPE') &&
      v1State.lastErrorMessages.some((m) => /version must be 2/.test(m)),
    'v1 config rejected with the "version must be 2" message',
    `lastErrors=${JSON.stringify(v1State.lastErrors)}`,
  )
  check(
    (await page.textContent('[data-testid="json-note"]')).includes('rejected'),
    'the JSON panel reports the rejection inline',
  )

  // --- (h) downloads -------------------------------------------------------
  const [objDownload] = await Promise.all([
    page.waitForEvent('download', { timeout: 15_000 }),
    page.click('[data-testid="export-obj"]'),
  ])
  const objName = objDownload.suggestedFilename()
  check(
    /^grid-design_\d{8}_\d{4}\.obj$/.test(objName),
    'Export OBJ downloaded grid-design_<timestamp>.obj',
    objName,
  )
  const objText = readFileSync(await objDownload.path(), 'utf8')
  const objCount = (objText.match(/^o /gm) ?? []).length
  check(
    objText.includes('o panel_r0_c0'),
    'OBJ payload contains `o panel_r0_c0`',
    `${objText.length} bytes, ${objCount} objects`,
  )
  check(
    objCount === jsonState.panels,
    'OBJ object count equals the panel count',
    `${objCount} objects vs ${jsonState.panels} panels`,
  )

  const [jsonDownload] = await Promise.all([
    page.waitForEvent('download', { timeout: 15_000 }),
    page.click('[data-testid="export-json"]'),
  ])
  const jsonName = jsonDownload.suggestedFilename()
  check(
    /^grid-design_\d{8}_\d{4}\.json$/.test(jsonName),
    'Download JSON downloaded grid-design_<timestamp>.json',
    jsonName,
  )
  const jsonText = readFileSync(await jsonDownload.path(), 'utf8')
  let downloadedConfig = null
  try {
    downloadedConfig = JSON.parse(jsonText)
  } catch (e) {
    fail('downloaded JSON parses', e.message)
  }
  check(
    downloadedConfig?.version === 2 &&
      downloadedConfig?.name === 'pasted study' &&
      downloadedConfig?.grid?.cols === 6 &&
      downloadedConfig?.columns?.length === 6 &&
      downloadedConfig?.columns?.[0]?.foldsDeg?.length === 4 &&
      downloadedConfig?.gap === 2,
    'downloaded JSON is the current, fully-defaulted v2 config',
    `${jsonText.length} bytes`,
  )

  // --- byte-level differences ---------------------------------------------
  const shots = ['01-flat.png', '02-wave.png', '03-fold.png', '04-rects.png', '05-reject.png', '06-json.png'].map(
    (name) => ({ name, buf: readFileSync(`${OUT}/${name}`) }),
  )
  for (let i = 0; i + 1 < shots.length; i++) {
    const x = shots[i]
    const y = shots[i + 1]
    check(!x.buf.equals(y.buf), `${x.name} and ${y.name} differ`, `${x.buf.length} vs ${y.buf.length} bytes`)
  }

  // --- error channels ------------------------------------------------------
  check(pageErrors.length === 0, 'no uncaught page errors', pageErrors.join(' | '))
  check(consoleErrors.length === 0, 'no console errors', consoleErrors.join(' | '))
} catch (err) {
  fail('harness', String(err?.stack ?? err))
} finally {
  if (browser) await browser.close().catch(() => {})
  stopDevServer(server)
}

// -----------------------------------------------------------------------------
// summary
// -----------------------------------------------------------------------------
console.log('\n=== SCREENSHOT HARNESS (WP6 — column-strip UI) ===')
for (const r of results) {
  console.log(`${r.ok ? 'PASS' : 'FAIL'}  ${r.name}${r.detail ? `  [${r.detail}]` : ''}`)
}
const failed = results.filter((r) => !r.ok).length
console.log(
  failed === 0
    ? `\nPASS — ${results.length}/${results.length} checks, screenshots in tests/screenshots/`
    : `\nFAIL — ${failed}/${results.length} check(s) failed`,
)
process.exit(failed === 0 ? 0 : 1)
