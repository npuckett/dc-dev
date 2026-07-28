/**
 * tests/screenshot-v4-wave.mjs — drive the REAL v4 UI through the wave and
 * photograph it.
 *
 * A headless test can say the numbers are right. It cannot say the control
 * exists, is reachable, and shows the design changing — which is the only way to
 * find out that a panel renders `undefined` or that a slider is wired to nothing.
 * So this drives the app, asserts what the screen SAYS at each step, and fails on
 * any console error.
 *
 *   node tests/screenshot-v4-wave.mjs            # starts its own dev server
 *   node tests/screenshot-v4-wave.mjs 5175       # attach to one already running
 *
 * Shots land in tests/screenshots/v4-wave-*.png.
 */

import { chromium } from 'playwright'
import { spawn } from 'node:child_process'
import { mkdirSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const OUT = resolve(HERE, 'screenshots')
const attachPort = process.argv[2] ? Number(process.argv[2]) : null
const PORT = attachPort ?? 5179

let failed = 0
const ok = (c, m) => { console.log(`  ${c ? 'ok  ' : 'FAIL'} ${m}`); if (!c) failed++ }

mkdirSync(OUT, { recursive: true })

let server = null
if (!attachPort) {
  server = spawn('npx', ['vite', '--port', String(PORT), '--strictPort'], {
    cwd: resolve(HERE, '..'), stdio: 'ignore', detached: true,
  })
}
const stop = () => { if (server) { try { process.kill(-server.pid) } catch { /* already gone */ } } }
process.on('exit', stop)

async function waitForServer(url, tries = 60) {
  for (let k = 0; k < tries; k++) {
    try {
      const res = await fetch(url)
      if (res.ok) return true
    } catch { /* not up yet */ }
    await new Promise((r) => setTimeout(r, 500))
  }
  return false
}

const url = `http://localhost:${PORT}/`
if (!await waitForServer(url)) {
  console.log(`FAIL: no dev server at ${url}`)
  process.exit(1)
}

const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } })
const consoleErrors = []
page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text()) })
page.on('pageerror', (e) => consoleErrors.push(String(e)))

// A fresh design every run: the store seeds from localStorage, and a stale
// autosave would make the shots depend on whatever was last driven by hand.
await page.goto(url)
await page.evaluate(() => window.localStorage.clear())
await page.reload()
await page.waitForSelector('[data-testid="lattice-map"]')

const text = (sel) => page.locator(sel).innerText()
const shot = (name) => page.screenshot({ path: `${OUT}/v4-wave-${name}.png`, fullPage: false })

console.log('=== screenshot-v4-wave ===')

// --- 1. the checkerboard, untouched ------------------------------------------
console.log('1. trapezoid — the default, and it must not have moved')
{
  const glyphs = await page.locator('[data-testid^="lm-cell-"]').allInnerTexts()
  ok(glyphs.every((g) => g === 'G' || g === 'H'), 'every cell reads G or H')
  ok(await page.locator('[data-testid="pattern-kind-trapezoid"]').getAttribute('class')
    .then((c) => c.includes('seg-btn-on')), 'trapezoid is the selected kind')
  ok(await page.locator('[data-testid="metrics-wave-x"]').count() === 0,
    'and no wave table is drawn')
  await shot('1-trapezoid')
}

// --- 2. the wave at zero scrunch — an egg-crate, not a checkerboard ----------
console.log('2. wave at 0% — uniform, three storeys')
{
  await page.locator('[data-testid="pattern-kind-wave"]').click()
  await page.waitForSelector('[data-testid="wave-readout"]')
  const cells = await page.locator('[data-testid^="lm-cell-"]').allInnerTexts()
  const distinct = new Set(cells)
  ok(distinct.size === 3, `three distinct heights on the plan, not two (got ${[...distinct].join('/')})`)
  ok(cells.every((c) => /^\d+$/.test(c)), 'and every tile shows a number rather than a glyph')
  const readout = await text('[data-testid="wave-readout"]')
  ok(readout.includes('0.0% shorter'), 'the run is uncompressed at 0% scrunch')
  ok(readout.includes('30.0°→30.0°'), 'and every edge sits at the base angle')
  ok((await text('[data-testid="limit-per-joint"]')).includes('1 distinct folds'),
    'one fold over the whole design')
  await shot('2-wave-0')
}

// --- 3. sweep the scrunch ----------------------------------------------------
console.log('3. sweeping the scrunch')
for (const [sx, sz, tag] of [[0.15, 0.1, '3-wave-15'], [0.3, 0.25, '4-wave-30'], [0.6, 0.5, '5-wave-60']]) {
  await page.locator('[data-testid="wave-scrunch-x"]').fill(String(sx))
  await page.locator('[data-testid="wave-scrunch-z"]').fill(String(sz))
  await page.waitForTimeout(120)
  const readout = await text('[data-testid="wave-readout"]')
  const runX = await text('[data-testid="metrics-wave-x-run"]')
  console.log(`     ${(sx * 100).toFixed(0)}%/${(sz * 100).toFixed(0)}% — ${readout.split('\n')[0]}`)
  ok(!readout.includes('0.0% shorter'), `${tag}: the run really did shorten`)
  ok(/θ runs 30\.00°/.test(runX), `${tag}: the base angle is still 30° at the front`)
  await shot(tag)
}

// --- 4. alignment, read off the DOM the user is looking at -------------------
// The core tests assert this on the solve; this asserts it on the thing on
// screen, which is where a rendering bug would live.
console.log('4. alignment, in the rendered plan')
{
  const rows = await page.evaluate(() => {
    const out = []
    for (const b of document.querySelectorAll('[data-testid^="lm-cell-"]')) {
      const [, i, j] = b.dataset.testid.match(/lm-cell-(\d+)-(\d+)/)
      const r = b.getBoundingClientRect()
      out.push({ i: +i, j: +j, x: Math.round(r.x * 100) / 100, y: Math.round(r.y * 100) / 100 })
    }
    return out
  })
  const byCol = new Map()
  const byRow = new Map()
  for (const c of rows) {
    if (!byCol.has(c.i)) byCol.set(c.i, new Set())
    if (!byRow.has(c.j)) byRow.set(c.j, new Set())
    byCol.get(c.i).add(c.x)
    byRow.get(c.j).add(c.y)
  }
  ok([...byCol.values()].every((s) => s.size === 1), 'every column of tiles shares one x on screen')
  ok([...byRow.values()].every((s) => s.size === 1), 'and every row shares one y')
  ok(byCol.size > 1 && new Set([...byCol.values()].map((s) => [...s][0])).size === byCol.size,
    'while the columns are at different x — the check is not passing on a constant')
}

// --- 5. the attractor, and the things the wave declines to do ---------------
console.log('5. the attractor and the declined requests')
{
  await page.locator('[data-testid="wave-attractor-x"]').fill('0.3')
  await page.waitForTimeout(120)
  const tbl = await text('[data-testid="metrics-wave-x"]')
  ok(tbl.includes('60%'), 'past the attractor the table shows full scrunch')
  await page.locator('[data-testid="wall-anchor"] button:nth-child(2)').click()
  await page.waitForTimeout(120)
  const report = await page.locator('body').innerText()
  ok(report.includes('W_WAVE_NO_ANCHOR') || report.includes('builds no anchor ramps'),
    'braced on a wave is reported as declined rather than silently ignored')
  await shot('6-wave-attractor')
}

// --- 6. and back to the checkerboard ----------------------------------------
console.log('6. back to trapezoid')
{
  await page.locator('[data-testid="pattern-kind-trapezoid"]').click()
  await page.waitForTimeout(120)
  const glyphs = await page.locator('[data-testid^="lm-cell-"]').allInnerTexts()
  ok(glyphs.every((g) => g === 'G' || g === 'H'), 'the plan is back to G/H')
  ok(await page.locator('[data-testid="metrics-wave-x"]').count() === 0, 'and the wave tables are gone')
  await shot('7-back-to-trapezoid')
}

ok(consoleErrors.length === 0, `no console errors (${consoleErrors.length}: ${consoleErrors.slice(0, 3).join(' | ')})`)

await browser.close()
stop()
console.log(failed ? `\n${failed} failed` : '\nall ok')
process.exit(failed ? 1 : 0)
