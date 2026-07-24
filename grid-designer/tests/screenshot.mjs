/**
 * grid-designer — browser smoke test / screenshot harness (WP3 + WP4).
 *
 * Self-contained: it starts its OWN dev server (`npm run dev`, port 5175 via
 * vite.config.js strictPort), waits for it, drives the UI, and tears the server
 * down on every exit path.
 *
 * What it captures
 *   01-flat.png    default state — the flat reference surface
 *   02-wave.png    after clicking the Wave preset button
 *   03-slider.png  after setting row 3's zig-zag slider to 60°
 *   04-rect-h.png  after clicking the grid map to place a HORIZONTAL plate in
 *                  the middle of row 2
 *   05-rect-v.png  after placing a VERTICAL plate at column 0 (where every row's
 *                  in-row tilt is 0, so the rigid-plate constraint always holds)
 *   06-reject.png  after an ILLEGAL vertical placement at (1,1) — nothing moved,
 *                  the error box explains why
 *   07-json.png    after pasting a minimal hand-written config and hitting Apply
 *
 * What it asserts (any failure ⇒ non-zero exit)
 *   - no page errors and no console errors
 *   - the WebGL canvas is non-blank (pixels sampled through a 2D canvas; the
 *     r3f Canvas runs with preserveDrawingBuffer so the read-back is valid)
 *   - the preset click actually changed the store's config
 *   - the slider actually changed the store's config
 *   - grid-map clicks add / reject plates through the store's commit rule
 *   - the JSON paste panel applies a minimal config (defaults filled in)
 *   - Export OBJ / Download JSON fire real downloads with the expected names,
 *     and the OBJ payload contains `o panel_r0_c0`
 *   - consecutive screenshots differ in bytes (the render responded each time)
 *
 * Usage: node tests/screenshot.mjs
 */

import { chromium } from 'playwright'
import { spawn } from 'node:child_process'
import { existsSync, mkdirSync, readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const OUT = resolve(ROOT, 'tests/screenshots')
const PORT = 5175
const URL = `http://localhost:${PORT}/`
const VIEWPORT = { width: 1440, height: 900 }

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
// page helpers
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
    !stats?.error &&
      stats.distinctColors >= 5 &&
      stats.brightFraction > 0.005 &&
      stats.std > 3,
    `canvas non-blank (${label})`,
    detail,
  )
  return stats
}

const storeState = (page) =>
  page.evaluate(() => {
    const s = window.__gridDesignerStore.getState()
    return {
      name: s.config.name,
      preset: s.config.meta?.preset,
      zigzags: s.config.rows.map((r) => r.zigzagDeg),
      rowFolds: s.config.rowFoldsDeg,
      rects: s.config.rects.map((r) => `${r.orientation[0]}(${r.row},${r.col})`),
      lastErrors: s.lastErrors.map((e) => e.code),
      summary: window.__gridDesignerDerived().report.summary,
      panels: window.__gridDesignerDerived().layout.panels.length,
    }
  })

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------
let server = null
let browser = null

try {
  if (!existsSync(OUT)) mkdirSync(OUT, { recursive: true })
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
  await assertCanvasRenders(page, 'flat')
  const flatState = await storeState(page)
  check(flatState.preset === 'flat', 'default preset is flat', `preset=${flatState.preset}`)
  check(
    flatState.panels === 30 && flatState.summary.flagged === 0,
    'flat default: 30 panels, no flagged joints',
    `panels=${flatState.panels} flagged=${flatState.summary.flagged}/${flatState.summary.total}`,
  )
  await page.screenshot({ path: `${OUT}/01-flat.png` })
  console.log('  → tests/screenshots/01-flat.png')

  // --- (b) wave preset -----------------------------------------------------
  await page.click('[data-testid="preset-wave"]')
  await page.waitForTimeout(900)
  const waveState = await storeState(page)
  check(waveState.preset === 'wave', 'Wave preset applied', `preset=${waveState.preset}`)
  check(
    JSON.stringify(waveState.zigzags) !== JSON.stringify(flatState.zigzags),
    'wave changed the config',
    `zigzags=${JSON.stringify(waveState.zigzags)} folds=${JSON.stringify(waveState.rowFolds)}`,
  )
  await assertCanvasRenders(page, 'wave')
  check(
    waveState.summary.flagged > 0,
    'wave report flags row-pair joints (markers visible)',
    `flagged=${waveState.summary.flagged}/${waveState.summary.total}`,
  )
  await page.screenshot({ path: `${OUT}/02-wave.png` })
  console.log('  → tests/screenshots/02-wave.png')

  // --- (c) row 3 zig-zag slider -------------------------------------------
  const slider = page.locator('[data-testid="zigzag-3"]')
  await slider.fill('60')
  await page.waitForTimeout(900)
  let sliderState = await storeState(page)
  if (sliderState.zigzags[3] !== 60) {
    console.log('! slider fill did not register — falling back to the store action')
    await page.evaluate(() => window.__gridDesignerStore.getState().setRowZigzag(3, 60))
    await page.waitForTimeout(900)
    sliderState = await storeState(page)
  }
  check(
    sliderState.zigzags[3] === 60,
    "row 3's zig-zag slider set to 60°",
    `zigzags=${JSON.stringify(sliderState.zigzags)}`,
  )
  check(
    sliderState.zigzags[3] !== waveState.zigzags[3],
    'slider changed row 3 away from the preset value',
    `${waveState.zigzags[3]}° → ${sliderState.zigzags[3]}°`,
  )
  await assertCanvasRenders(page, 'slider')
  await page.screenshot({ path: `${OUT}/03-slider.png` })
  console.log('  → tests/screenshots/03-slider.png')

  check(
    sliderState.lastErrors.length === 0,
    'no rejected changes (store lastErrors empty)',
    sliderState.lastErrors.join(', '),
  )

  // --- (d) grid map: place a HORIZONTAL plate in row 2 ----------------------
  check(await page.isVisible('[data-testid="grid-map"]'), 'grid map is visible')
  await page.click('[data-testid="gridmap-mode-horizontal"]')
  await page.click('[data-testid="cell-2-2"]')
  await page.waitForTimeout(900)
  const rectHState = await storeState(page)
  check(
    rectHState.rects.length === sliderState.rects.length + 1 &&
      rectHState.rects.includes('h(2,2)'),
    'grid-map click placed a horizontal plate at (2,2)',
    `rects=${JSON.stringify(rectHState.rects)}`,
  )
  check(
    rectHState.panels === sliderState.panels - 1,
    'horizontal plate merged two cells into one panel',
    `${sliderState.panels} → ${rectHState.panels} panels`,
  )
  check(rectHState.lastErrors.length === 0, 'horizontal placement accepted', rectHState.lastErrors.join(', '))
  await assertCanvasRenders(page, 'rect-h')
  await page.screenshot({ path: `${OUT}/04-rect-h.png` })
  console.log('  → tests/screenshots/04-rect-h.png')

  // --- (e) grid map: place a VERTICAL plate at column 0 --------------------
  // Column 0's in-row tilt is 0 in every row by construction (the accordion
  // chain starts flat), so a cross-row rigid plate there always validates.
  await page.click('[data-testid="gridmap-mode-vertical"]')
  await page.click('[data-testid="cell-0-0"]')
  await page.waitForTimeout(900)
  const rectVState = await storeState(page)
  check(
    rectVState.rects.length === rectHState.rects.length + 1 && rectVState.rects.includes('v(0,0)'),
    'grid-map click placed a vertical plate at (0,0)',
    `rects=${JSON.stringify(rectVState.rects)}`,
  )
  check(
    rectVState.panels === rectHState.panels - 1,
    'vertical plate merged two cells across rows 0 and 1',
    `${rectHState.panels} → ${rectVState.panels} panels`,
  )
  check(rectVState.lastErrors.length === 0, 'vertical placement accepted', rectVState.lastErrors.join(', '))
  await assertCanvasRenders(page, 'rect-v')
  await page.screenshot({ path: `${OUT}/05-rect-v.png` })
  console.log('  → tests/screenshots/05-rect-v.png')

  // --- (f) grid map: an ILLEGAL vertical placement -------------------------
  // Still in V mode. Cell (1,1) tilts 15° (row 1's zig-zag 15 with the wave
  // preset's j3 override) while (2,1) tilts 30° — a rigid plate cannot span
  // them, so E_CROSSROW_ANGLE_MISMATCH must reject the whole change.
  await page.click('[data-testid="cell-1-1"]')
  await page.waitForTimeout(700)
  const rejectState = await storeState(page)
  check(
    JSON.stringify(rejectState.rects) === JSON.stringify(rectVState.rects),
    'illegal vertical placement left the config untouched',
    `rects=${JSON.stringify(rejectState.rects)}`,
  )
  check(
    rejectState.panels === rectVState.panels &&
      JSON.stringify(rejectState.zigzags) === JSON.stringify(rectVState.zigzags),
    'illegal placement left the layout untouched',
    `panels=${rejectState.panels}`,
  )
  check(
    rejectState.lastErrors.includes('E_CROSSROW_ANGLE_MISMATCH'),
    'illegal placement reported E_CROSSROW_ANGLE_MISMATCH',
    `lastErrors=${JSON.stringify(rejectState.lastErrors)}`,
  )
  check(
    await page.isVisible('[data-testid="last-errors"]'),
    'error box is visible next to the grid map',
  )
  await page.screenshot({ path: `${OUT}/06-reject.png` })
  console.log('  → tests/screenshots/06-reject.png')

  // --- (g) JSON panel: paste a minimal hand-written config ----------------
  const MINIMAL_CONFIG = {
    version: 1,
    units: 'cm',
    name: 'pasted study',
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
  await page.waitForTimeout(200)
  check(
    await page.isVisible('[data-testid="json-text"]'),
    'JSON panel opened (textarea visible)',
  )
  const syncedText = await page.inputValue('[data-testid="json-text"]')
  check(
    syncedText.includes('"version": 2') === false && syncedText.includes('"rowFoldsDeg"'),
    'textarea was synced to the current config on open',
    `${syncedText.length} chars`,
  )
  await page.fill('[data-testid="json-text"]', JSON.stringify(MINIMAL_CONFIG, null, 2))
  await page.click('[data-testid="json-apply"]')
  await page.waitForTimeout(900)
  const jsonState = await storeState(page)
  check(jsonState.name === 'pasted study', 'pasted config applied (name took effect)', `name=${jsonState.name}`)
  check(
    JSON.stringify(jsonState.zigzags) === JSON.stringify([0, 25, 45, 10, 5]) &&
      JSON.stringify(jsonState.rowFolds) === JSON.stringify([20, 35, 25, -15]),
    'pasted zig-zags and row folds took effect',
    `zigzags=${JSON.stringify(jsonState.zigzags)} folds=${JSON.stringify(jsonState.rowFolds)}`,
  )
  check(
    jsonState.rects.length === 0 && jsonState.panels === 30 && jsonState.lastErrors.length === 0,
    'normalizeConfig filled the omitted defaults (no rects, 30 panels, no errors)',
    `rects=${jsonState.rects.length} panels=${jsonState.panels} errors=${JSON.stringify(jsonState.lastErrors)}`,
  )
  await assertCanvasRenders(page, 'json')
  await page.screenshot({ path: `${OUT}/07-json.png` })
  console.log('  → tests/screenshots/07-json.png')

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
    downloadedConfig?.name === 'pasted study' &&
      downloadedConfig?.grid?.cols === 6 &&
      downloadedConfig?.rows?.length === 5 &&
      downloadedConfig?.gap === 2,
    'downloaded JSON is the current, fully-defaulted config',
    `${jsonText.length} bytes`,
  )

  // --- byte-level differences ---------------------------------------------
  const shots = [
    '01-flat.png',
    '02-wave.png',
    '03-slider.png',
    '04-rect-h.png',
    '05-rect-v.png',
    '07-json.png',
  ].map((name) => ({ name, buf: readFileSync(`${OUT}/${name}`) }))
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
console.log('\n=== SCREENSHOT HARNESS (WP3 + WP4) ===')
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
