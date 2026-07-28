/**
 * tests/screenshot-v4-export.mjs — click the real export buttons and open what
 * comes out.
 *
 * `test-v4-export.mjs` parses the payloads the writers return. That is not the
 * same claim as "the button downloads a usable file": the UI could stamp the OBJ
 * and the MTL with different names, forget the second download, or hand the GLB
 * ArrayBuffer to a text Blob and produce a corrupt container. None of that is
 * visible from a unit test, and all of it reproduces the reported bug.
 *
 * So this drives the app, catches the ACTUAL downloads, and parses them off
 * disk:
 *
 *   node tests/screenshot-v4-export.mjs            # starts its own dev server
 *   node tests/screenshot-v4-export.mjs 5175       # attach to one already running
 *
 * Shots land in tests/screenshots/v4-export-*.png.
 */

import { chromium } from 'playwright'
import { spawn } from 'node:child_process'
import { mkdirSync, readFileSync, statSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const OUT = resolve(HERE, 'screenshots')
const attachPort = process.argv[2] ? Number(process.argv[2]) : null
const PORT = attachPort ?? 5181

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
    try { if ((await fetch(url)).ok) return true } catch { /* not up yet */ }
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
const page = await browser.newPage({ viewport: { width: 1600, height: 1000 }, acceptDownloads: true })
const consoleErrors = []
page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text()) })
page.on('pageerror', (e) => consoleErrors.push(String(e)))

await page.goto(url)
await page.evaluate(() => window.localStorage.clear())
await page.reload()
await page.waitForSelector('[data-testid="export-obj"]')

console.log('=== screenshot-v4-export ===')

/** Click, and return every file the click actually put on disk. */
async function clickAndCollect(testid, expected) {
  const seen = []
  const done = new Promise((r) => {
    const onDl = async (d) => {
      seen.push({ name: d.suggestedFilename(), path: await d.path() })
      if (seen.length === expected) { page.off('download', onDl); r() }
    }
    page.on('download', onDl)
  })
  await page.click(`[data-testid="${testid}"]`)
  await Promise.race([done, new Promise((r) => setTimeout(r, 15000))])
  return seen
}

// --- 1. the toolbar ----------------------------------------------------------
console.log('1. both export buttons are on the bar, in the house style')
{
  const labels = await page.locator('.export-bar button').allInnerTexts()
  ok(labels.includes('Export OBJ + MTL'), `the OBJ button says so (${labels.join(' | ')})`)
  ok(labels.includes('Export GLB'), 'the GLB button is beside it')
  const classes = await page.locator('.export-bar button').evaluateAll((bs) => bs.map((b) => b.className))
  ok(classes.every((c) => c.includes('preset-btn')), 'every button shares the existing class')
  await page.screenshot({ path: `${OUT}/v4-export-1-toolbar.png` })
}

// --- 2. the OBJ button delivers a RESOLVABLE pair -----------------------------
console.log('2. Export OBJ + MTL puts two files on disk, and they refer to each other')
{
  const files = await clickAndCollect('export-obj', 2)
  ok(files.length === 2, `two downloads (${files.map((f) => f.name).join(', ')})`)
  const objFile = files.find((f) => f.name.endsWith('.obj'))
  const mtlFile = files.find((f) => f.name.endsWith('.mtl'))
  ok(!!objFile && !!mtlFile, 'one .obj and one .mtl')

  const obj = readFileSync(objFile.path, 'utf8')
  const mtl = readFileSync(mtlFile.path, 'utf8')
  ok(statSync(objFile.path).size > 100000, `the OBJ is non-empty (${statSync(objFile.path).size} bytes)`)
  ok(statSync(mtlFile.path).size > 1000, `the MTL is non-empty (${statSync(mtlFile.path).size} bytes)`)

  // THE BUG. The mtllib must name the file that was actually saved beside it.
  const named = obj.match(/^mtllib (.+)$/m)?.[1]?.trim()
  ok(named === mtlFile.name, `the OBJ's mtllib names the downloaded MTL (${named} vs ${mtlFile.name})`)

  const used = [...obj.matchAll(/^usemtl (.+)$/gm)].map((m) => m[1].trim())
  const defined = new Set([...mtl.matchAll(/^newmtl (.+)$/gm)].map((m) => m[1].trim()))
  const objs = [...obj.matchAll(/^o (.+)$/gm)].map((m) => m[1].trim())
  ok(objs.length === 41, `41 objects (${objs.length})`)
  ok(objs.filter((n) => n.startsWith('diffuser_')).length === 37,
    `37 of them individual diffusers (${objs.filter((n) => n.startsWith('diffuser_')).length})`)
  ok(defined.size === objs.length, `${defined.size} materials defined, one per object`)
  ok(used.every((u) => defined.has(u)), 'and every usemtl in the downloaded OBJ resolves in the downloaded MTL')
}

// --- 3. the GLB button delivers a valid container -----------------------------
console.log('3. Export GLB puts one valid binary container on disk')
{
  const files = await clickAndCollect('export-glb', 1)
  ok(files.length === 1 && files[0].name.endsWith('.glb'), `one .glb (${files[0]?.name})`)
  const buf = readFileSync(files[0].path)
  ok(buf.length > 10000, `non-empty (${buf.length} bytes)`)

  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength)
  ok(dv.getUint32(0, true) === 0x46546c67, 'the magic on disk is glTF')
  ok(dv.getUint32(8, true) === buf.length, 'and the declared length matches the downloaded file')
  const jlen = dv.getUint32(12, true)
  const json = JSON.parse(new TextDecoder().decode(buf.subarray(20, 20 + jlen)))
  ok(json.materials.length === 41, `41 materials in the downloaded file, not 1 (${json.materials.length})`)
  ok(json.meshes.length === 41, `41 meshes (${json.meshes.length})`)
  const diffusers = json.nodes.filter((n) => n.name?.startsWith('diffuser_'))
  const mats = diffusers.map((n) => json.meshes[n.mesh].primitives[0].material)
  ok(diffusers.length === 37, `37 named diffuser nodes (${diffusers.length})`)
  ok(new Set(mats).size === 37, `each with its own material (${new Set(mats).size} distinct)`)
  await page.screenshot({ path: `${OUT}/v4-export-2-after.png` })
}

// --- 4. nothing broke on the way ---------------------------------------------
console.log('4. the console is clean')
ok(consoleErrors.length === 0, `no console errors (${consoleErrors.slice(0, 2).join(' | ')})`)

await browser.close()
stop()
console.log(`\nscreenshot-v4-export: ${failed ? `${failed} FAILED` : 'all checks passed'}`)
process.exit(failed ? 1 : 0)
