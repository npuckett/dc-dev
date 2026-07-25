/**
 * grid-designer — browser smoke test / screenshot harness (WP10, schema v2).
 *
 * Self-contained: it starts its OWN dev server (`npm run dev`, port 5175 via
 * vite.config.js strictPort), waits for it, drives the UI, and tears the server
 * down on every exit path.
 *
 * =============================================================================
 * THE FLOW IT DRIVES (the column-strip design loop, end to end)
 * =============================================================================
 *   01-flat.png    default state — the flat reference surface with its 'mirrored
 *                  quad' of four plates (26 panels), no flagged joints
 *   02-wave.png    the Wave preset: six phase-shifted fold sequences (asserted
 *                  against core/presets.js itself) plus its 'crest plates' pattern —
 *                  one rigid plate on every column's crest — and every column
 *                  landing on the floor ("all grounded ✓")
 *   03-floating.png column 3's LAST hinge pitched up to 60°, breaking the
 *                  grounded-end rule: its end panel is now in mid-air, so the
 *                  summary badge reads "1 floating", the column block carries a red
 *                  FLOATING badge with the clearance, its sparkline endpoint turns
 *                  red and the grid map outlines its back cell
 *   04-grounded.png after clicking that column's "Ground end": core/ground.js
 *                  solved the same hinge for floor contact, violations are back to
 *                  zero and only that one angle moved
 *   05-fold.png    one hinge dragged: column 3's 0→1 joint to 80°, which lifts the
 *                  whole strip ~164cm — far beyond what its last hinge can reach,
 *                  so "Ground end" answers E_UNGROUNDABLE and changes nothing
 *   06-rects.png   all six preset plates removed (which drops the design under the
 *                  four-plate pattern rule and raises W_FEW_RECTS), "Shift →"
 *                  phase-shifted the whole grid one column, then a HORIZONTAL and a
 *                  VERTICAL plate placed on cells the harness COMPUTES to be legal
 *                  under the current folds
 *   07-reject.png  two illegal placements in a row — an H plate across two
 *                  columns at different pitches (E_CROSSCOL_ANGLE_MISMATCH) and a
 *                  V plate over a folded joint (E_FOLD_ON_REMOVED_JOINT). Nothing
 *                  moved; the map explains why inline.
 *   08-json.png    after pasting a MINIMAL v2 config (columns only — everything
 *                  else defaulted by normalizeConfig) and hitting Apply
 *   09-swell.png   the STORM family's 'Swell': 7 rows, a PITCHED FRONT in every
 *                  column (42° → 20°, the family's signature) and a 121cm spine
 *                  plate mid-strip in each. Half swell, half snow drift — the
 *                  shoreline BREAKS at joint 0 (a 13–22° counter-bend, different in
 *                  every column) and everything ramps toward the −X WALL beside
 *                  column 0, the RIGHT of the 3D view, from a ~42cm start at the far
 *                  jamb to a ~116cm crest; still every column on the floor
 *   10-wallcrash.png the one design that engages the −X WALL: column 0 — the
 *                  column against it — is `endSupport: 'wall'`, ends ~170cm up, and
 *                  raises NO violation while columns 1–5 land normally
 *   11-rows.png    the ROWS STEPPER driven on the flat preset — 5 → 6 (26 → 32
 *                  panels), up to the 8-row ceiling and back down, with the four
 *                  plates surviving throughout
 *   12-front.png   the FRONT slider dragged on one column: its window panel pitched
 *                  to 35°, the whole strip pitching with it, no hinge touched
 *   13-panel-detail.png the WP11 PANEL CLOSE-UP — the camera parked (through
 *                  `window.__gridDesignerCamera`) beside the 'surge' surface so the
 *                  same frame holds panels showing their LIT FACE (a recessed glowing
 *                  diffuser inside a crisp frame flange) and panels showing their
 *                  BACK (a closed, inward-tapering tray). This is the visual proof
 *                  that the panel solid is no longer inside-out.
 *   14-merge.png   the WP14 PAIR-CLICK MERGE on 'swell': column 4's spine plate split
 *                  back into two squares, then cell (2,4) ARMED — a warm outline —
 *                  with its two mergeable neighbours highlighted in the two styles at
 *                  once: (3,4) FREE (green, solid, "+") across the joint the split
 *                  left flat, and (1,4) COERCE (amber, dashed, "~") across a folded
 *                  joint the merge would have to flatten
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
 *   - THE PLATE-PATTERN RULE end to end: flat and Wave both carry ≥ 4 plates and
 *     name their rule in `meta.rectPattern`, which the grid map shows as a caption
 *     under its title; removing the plates one by one drops the design below four
 *     and raises W_FEW_RECTS in the control panel's warnings box (non-blocking — the
 *     removals still commit)
 *   - THE MEASURING BOX: on by default, its three cm dimension labels present in the
 *     3D view (the height one reading the PEAK), the control panel printing the same
 *     numbers as a `W × H × D cm` line — exactly 365 × 4 × 304 for the flat lattice —
 *     and the toolbar's "bounds" button hiding and restoring all of it without
 *     touching the design
 *   - a hinge slider (or setColumnFold) moves one column's fold and the block's
 *     cumulative-pitch readout follows
 *   - "Shift →" rotates every column's fold sequence by one index, wrapping
 *   - grid-map PAIR CLICKS combine two squares into a plate and split plates back,
 *     and the two v2 rect constraints still reject a plate placed straight at the
 *     store (`addRect`, the path the JSON panel and console use), with the right
 *     codes surfaced inline next to the map (and NOT duplicated in the control
 *     panel's general error box)
 *   - THE MERGE AFFORDANCE end to end (WP14): arming a square highlights every
 *     neighbour it can merge with, in two distinguishable styles — FREE (nothing
 *     else changes) and COERCE (the merge must also flatten the joint it spans or
 *     match the neighbouring column) — with the consequence in the tooltip before
 *     the click and in a calm notice after it; Escape disarms; the merge flattens
 *     exactly one joint and touches nothing else; Undo / Redo (buttons and
 *     Cmd/Ctrl+Z) walk it back and forward; and splitting the new plate again leaves
 *     the joint flat, the documented non-inverse
 *   - a vertical plate DISABLES the slider for the joint it removes
 *   - the GROUNDED-END rule end to end: flat and Wave report "all grounded ✓";
 *     lifting a column's last hinge raises exactly one E_END_FLOATING and surfaces
 *     it in four places (summary badge, column badge, sparkline endpoint, grid-map
 *     tooltip); "Ground end" solves that hinge back to floor contact, moving no
 *     other angle and committing through validation; a column whose chain ends a
 *     whole panel-length too high reports E_UNGROUNDABLE and is left alone; and
 *     "Ground all" lands every column it can reach while flagging the rest
 *   - the JSON panel applies a minimal v2 config and REJECTS a v1 one with the
 *     "version must be 2" message
 *   - PITCHED FRONTS end to end: the storm presets arrive with a non-zero
 *     `startPitchDeg` in every column, the per-column FRONT slider sets one by hand,
 *     and the pitch readout / sparkline follow
 *   - THE WALL end to end: 'wallcrash' marks column 0 wall-supported, its block shows
 *     a WALL-SUPPORTED badge instead of FLOATING with no "Ground end" button, the
 *     design reports zero violations despite that column ending ~170cm up, and the
 *     grid map draws the wall rule on its LEFT (−X) edge. Its end-support TOGGLE
 *     (offered on column 0 only) then proves the exemption is doing the work:
 *     standing the same geometry on the floor raises E_END_FLOATING, bracketing it
 *     back clears it, and no angle moves either way
 *   - ROW RESOLUTION end to end: the stepper reads the preset's row count, walks the
 *     flat grid 5 → 6 → 8 → 6, grows and shrinks the panel count accordingly, appends
 *     hinges at the BACK of every strip and keeps the plates that still fit
 *   - THE PANEL SOLID (WP11): with the camera parked beside the 'surge' surface, the
 *     visible panels genuinely split into ones presenting their LIT FACE to the camera
 *     and ones presenting their BACK — computed from each panel's world normal
 *     (`quaternion · +Y`) against the camera position, which is exactly the test the
 *     old inside-out caps failed silently
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
      rectPattern: s.config.meta?.rectPattern,
      version: s.config.version,
      cols: s.config.grid.cols,
      rows: s.config.grid.rows,
      gap: s.config.gap,
      folds: s.config.columns.map((col) => col.foldsDeg.slice()),
      startPitches: s.config.columns.map((col) => col.startPitchDeg),
      endSupports: s.config.columns.map((col) => col.endSupport),
      chainEndSupports: layout.columnChains.map((chain) => chain.endSupport),
      pitches: layout.columnChains.map((chain) => chain.pitchesDeg.slice()),
      rectObjs: s.config.rects.map((r) => ({ ...r })),
      rects: s.config.rects.map((r) => `${r.orientation[0]}(${r.row},${r.col})`),
      lastErrors: s.lastErrors.map((e) => e.code),
      lastErrorMessages: s.lastErrors.map((e) => e.message),
      validationWarnings: s.lastWarnings.map((w) => w.code),
      layoutWarnings: layout.warnings.map((w) => w.code),
      violations: layout.violations.map((v) => v.code),
      floatingCols: layout.violations.map((v) => v.col),
      clearances: layout.columnChains.map((chain) => chain.endClearanceCm),
      grounded: layout.columnChains.map((chain) => chain.grounded),
      // Per-column crest height, cm — the tallest vertex of each solved chain.
      peaks: layout.columnChains.map((chain) => Math.max(...chain.points.map((p) => p[0]))),
      // The design's overall measuring box (core/placement.js layoutBounds).
      bounds: window.__gridDesignerDerived().bounds,
      showBounds: s.showBounds,
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
  // 30 cells − the 4 plates of the 'mirrored quad' pattern = 26 panels.
  check(
    flatState.panels === 26 && flatState.summary.flagged === 0,
    'flat default: 26 panels (30 cells − the mirrored quad), no flagged joints',
    `panels=${flatState.panels} flagged=${flatState.summary.flagged}/${flatState.summary.total}`,
  )
  check(
    JSON.stringify(flatState.rects) === JSON.stringify(['v(1,1)', 'v(3,1)', 'v(1,4)', 'v(3,4)']),
    "flat default brought its four plates (the 'mirrored quad' in columns 1 and 4)",
    JSON.stringify(flatState.rects),
  )
  check(
    !flatState.validationWarnings.includes('W_FEW_RECTS'),
    'flat default satisfies the plate-pattern rule (no W_FEW_RECTS)',
    JSON.stringify(flatState.validationWarnings),
  )
  check(
    flatState.rectPattern === 'mirrored quad' &&
      (await page.textContent('[data-testid="rect-pattern"]')).includes('mirrored quad'),
    'the grid map captions the plate pattern under its title',
    `meta.rectPattern=${JSON.stringify(flatState.rectPattern)}`,
  )

  // --- (a2) THE MEASURING BOX: cm dimensions in 3D and in the panel ---------
  // The flat lattice is the one design whose box is a closed form: 6·61 − 1 = 365
  // wide, one panel (3.7cm) tall, 5·61 − 1 = 304 deep. It is ON by default, and the
  // toolbar's "bounds" button has to make it go away and come back.
  check(
    flatState.showBounds === true,
    'the measuring box is ON by default',
    `showBounds=${flatState.showBounds}`,
  )
  check(
    JSON.stringify(flatState.bounds.size.map((v) => Math.round(v))) ===
      JSON.stringify([365, 4, 304]),
    'flat: the box is exactly the lattice — 365 × 4 (one panel) × 304 cm',
    JSON.stringify(flatState.bounds),
  )
  const boundsLine = (await page.textContent('[data-testid="bounds-summary"]')) ?? ''
  check(
    boundsLine.replace(/\s+/g, ' ').trim() === '365 × 4 × 304 cm',
    'the control panel prints the overall size as a compact W × H × D line',
    boundsLine.trim(),
  )
  const boundsLabels = async () =>
    Promise.all(
      ['x', 'y', 'z'].map(async (axis) => {
        const sel = `[data-testid="bounds-label-${axis}"]`
        return (await page.locator(sel).count()) > 0 ? (await page.textContent(sel)).trim() : null
      }),
    )
  const flatLabels = await boundsLabels()
  check(
    flatLabels[0] === 'W 365 cm' && flatLabels[1] === 'peak 4 cm' && flatLabels[2] === 'D 304 cm',
    'the 3D view carries all three cm dimension labels, the height one reading the PEAK',
    JSON.stringify(flatLabels),
  )
  await page.click('[data-testid="tool-bounds"]')
  await page.waitForTimeout(300)
  const hiddenLabels = await boundsLabels()
  const boundsOff = await storeState(page)
  check(
    hiddenLabels.every((t) => t === null) && boundsOff.showBounds === false,
    'the "bounds" toggle DISMISSES the box and its labels',
    JSON.stringify(hiddenLabels),
  )
  check(
    JSON.stringify(boundsOff.folds) === JSON.stringify(flatState.folds) &&
      JSON.stringify(boundsOff.bounds) === JSON.stringify(flatState.bounds),
    'hiding the box changes nothing about the design — it is a ruler, not a property',
  )
  await page.click('[data-testid="tool-bounds"]')
  await page.waitForTimeout(300)
  check(
    JSON.stringify(await boundsLabels()) === JSON.stringify(flatLabels) &&
      (await storeState(page)).showBounds === true,
    'and brings it straight back',
  )
  check(
    flatState.folds.length === 6 && flatState.folds.every((f) => f.length === 4 && f.every((v) => v === 0)),
    'flat default: 6 columns × 4 hinges, all 0°',
    JSON.stringify(flatState.folds),
  )
  check(
    flatState.rows === 5 &&
      (await page.textContent('[data-testid="rows-value"]')).includes('5 rows') &&
      (await page.isDisabled('[data-testid="rows-dec"]')),
    'flat default: the rows stepper reads 5 rows and is already at its floor',
    `rows=${flatState.rows}`,
  )
  check(
    flatState.startPitches.every((p) => p === 0) &&
      flatState.endSupports.every((s) => s === 'floor'),
    'flat default (calm family): every front FLAT, every column on the floor',
    `${JSON.stringify(flatState.startPitches)} / ${JSON.stringify(flatState.endSupports)}`,
  )
  check(
    await page.isVisible('[data-testid="front-0"]') && await page.isVisible('[data-testid="front-5"]'),
    'every column block carries a FRONT slider above its hinges',
  )
  check(
    !(await page.isVisible('[data-testid="column-wall-0"]')) &&
      (await page.textContent('[data-testid="column-endsupport-0"]')).includes('bracket to wall'),
    'no WALL-SUPPORTED badge on an all-floor design, just the offer to bracket column 0',
  )
  check(
    await page.isVisible('[data-testid="gridmap-wall"]') &&
      (await page.textContent('[data-testid="grid-map"]')).includes('closes the left edge (col 0)'),
    'the grid map marks the −X wall on its LEFT edge and says so in the hint',
  )
  check(
    (await page.textContent('[data-testid="column-0"]')).includes('at the wall'),
    "column 0's header says it is the strip against the wall",
  )
  check(
    await page.isVisible('[data-testid="preset-swell"]') &&
      await page.isVisible('[data-testid="preset-surge"]') &&
      await page.isVisible('[data-testid="preset-wallcrash"]') &&
      await page.isVisible('[data-testid="preset-divider-storm"]'),
    'the preset bar offers the storm family behind a family divider',
  )
  check(
    flatState.violations.length === 0 &&
      flatState.grounded.every(Boolean) &&
      flatState.clearances.every((v) => v === 0),
    'flat default: every column lies on the floor (clearance 0, no violations)',
    JSON.stringify(flatState.clearances),
  )
  check(
    await page.isVisible('[data-testid="badge-grounded"]') &&
      (await page.textContent('[data-testid="badge-grounded"]')).includes('all grounded'),
    'the summary badge row shows "all grounded ✓"',
  )
  check(
    await page.isVisible('[data-testid="tool-ground-all"]'),
    '"Ground all" is present in the cross-column toolbar',
  )
  check(
    !(await page.isVisible('[data-testid="badge-floating"]')) &&
      !(await page.isVisible('[data-testid="layout-violations"]')),
    'no floating badge and no violation box on a grounded design',
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
    JSON.stringify(waveState.rects) ===
      JSON.stringify(['v(1,0)', 'v(1,1)', 'v(1,2)', 'v(2,3)', 'v(2,4)', 'v(2,5)']),
    "wave brought its 'crest plates' — one rigid plate on every column's crest",
    JSON.stringify(waveState.rects),
  )
  const wavePatternCaption = (await page.textContent('[data-testid="rect-pattern"]')) ?? ''
  check(
    waveState.rectPattern === 'crest plates' &&
      wavePatternCaption.includes('crest plates') &&
      wavePatternCaption.includes('6'),
    'the plate-pattern caption followed the preset (name + plate count)',
    `${JSON.stringify(waveState.rectPattern)} / ${wavePatternCaption.trim()}`,
  )
  check(
    !waveState.validationWarnings.includes('W_FEW_RECTS'),
    'wave satisfies the plate-pattern rule (6 plates ≥ 4, no W_FEW_RECTS)',
    JSON.stringify(waveState.validationWarnings),
  )
  // At the default 1cm gap a 121cm plate is exactly two 60cm cells plus their
  // joint (60 + 1 + 60), so the plates raise no rect-length warning any more.
  check(
    !waveState.validationWarnings.includes('W_RECT_LENGTH'),
    'wave: its six plates raise NO W_RECT_LENGTH (121 = 2·60 + gap 1, an exact fit)',
    JSON.stringify(waveState.validationWarnings),
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
  check(
    waveState.violations.length === 0 && waveState.grounded.every(Boolean),
    'wave preset: every column lands on the floor (no E_END_FLOATING)',
    JSON.stringify(waveState.clearances),
  )
  check(
    await page.isVisible('[data-testid="badge-grounded"]') &&
      (await page.textContent('[data-testid="badge-grounded"]')).includes('all grounded'),
    'wave preset shows "all grounded ✓"',
  )
  await page.screenshot({ path: `${OUT}/02-wave.png` })
  console.log('  → tests/screenshots/02-wave.png')

  // --- (b2) break the grounded-end rule: lift column 3's last panel --------
  // The LAST hinge of the column is the one that decides whether its end panel
  // touches the floor, so pitching it UP is the minimal way to make a column float.
  const FLOAT_COL = 3
  const lastJoint = waveState.folds[FLOAT_COL].length - 1
  await page.locator(`[data-testid="fold-${FLOAT_COL}-${lastJoint}"]`).fill('60')
  await page.waitForTimeout(SETTLE_MS)
  let floatState = await storeState(page)
  if (floatState.folds[FLOAT_COL][lastJoint] !== 60) {
    console.log('! slider fill did not register — falling back to the store action')
    await page.evaluate(
      ([c, k]) => window.__gridDesignerStore.getState().setColumnFold(c, k, 60),
      [FLOAT_COL, lastJoint],
    )
    await page.waitForTimeout(SETTLE_MS)
    floatState = await storeState(page)
  }
  const floatShot = await assertCanvasRenders(page, 'floating')
  check(
    JSON.stringify(floatState.floatingCols) === JSON.stringify([FLOAT_COL]) &&
      floatState.violations.every((code) => code === 'E_END_FLOATING'),
    `pitching column ${FLOAT_COL}'s last hinge up left exactly that column floating`,
    `floating=${JSON.stringify(floatState.floatingCols)} violations=${JSON.stringify(floatState.violations)}`,
  )
  check(
    floatState.clearances[FLOAT_COL] > 1 && floatState.grounded[FLOAT_COL] === false,
    `column ${FLOAT_COL}'s end panel is measurably off the floor`,
    `clearance=${floatState.clearances[FLOAT_COL]}cm`,
  )
  const floatingBadge = (await page.textContent('[data-testid="badge-floating"]')) ?? ''
  check(
    floatingBadge.includes('1') && floatingBadge.includes('floating'),
    'the summary badge row switched to a red "1 floating"',
    floatingBadge.trim(),
  )
  check(!(await page.isVisible('[data-testid="badge-grounded"]')), 'the "all grounded" tick is gone')
  const colBadge = (await page.textContent(`[data-testid="column-floating-${FLOAT_COL}"]`)) ?? ''
  check(
    colBadge.includes('FLOATING') && /\d/.test(colBadge),
    `column ${FLOAT_COL}'s block shows a FLOATING badge with its clearance`,
    colBadge.trim(),
  )
  check(
    (await page.getAttribute(`[data-testid="profile-end-${FLOAT_COL}"]`, 'data-grounded')) === 'false' &&
      (await page.getAttribute('[data-testid="profile-end-0"]', 'data-grounded')) === 'true',
    'the sparkline endpoint marker is red for the floating column, green for the rest',
  )
  const violationText = (await page.textContent('[data-testid="layout-violations"]')) ?? ''
  check(
    violationText.includes('E_END_FLOATING') && violationText.includes('returns to the water'),
    'the violation box explains the rule under the grid map',
    `${violationText.length} chars`,
  )
  // SVG tooltips are <title> CHILD elements, not attributes.
  const backCellTip =
    (await page.textContent(`[data-testid="cell-4-${FLOAT_COL}"] title`)) ?? ''
  const frontCellTip = (await page.textContent('[data-testid="cell-4-0"] title')) ?? ''
  check(
    backCellTip.includes('end floating') &&
      /off the floor/.test(backCellTip) &&
      !frontCellTip.includes('end floating'),
    "the grid map's back cell for that column reports the clearance in its tooltip",
    backCellTip.trim(),
  )
  check(canvasChanged(waveShot, floatShot), 'lifting the last panel changed the render')
  await page.screenshot({ path: `${OUT}/03-floating.png` })
  console.log('  → tests/screenshots/03-floating.png')

  // --- (b3) "Ground end": hand that hinge to the solver --------------------
  await page.click(`[data-testid="column-ground-${FLOAT_COL}"]`)
  await page.waitForTimeout(SETTLE_MS)
  const groundState = await storeState(page)
  const groundShot = await assertCanvasRenders(page, 'grounded')
  check(
    groundState.violations.length === 0 && groundState.grounded.every(Boolean),
    '"Ground end" brought the column back down — zero violations',
    `floating=${JSON.stringify(groundState.floatingCols)} clearance=${groundState.clearances[FLOAT_COL]}`,
  )
  check(
    groundState.clearances[FLOAT_COL] >= -0.25 && groundState.clearances[FLOAT_COL] <= 0.5,
    'the solved end panel rests ON the floor, within groundTolerance',
    `clearance=${groundState.clearances[FLOAT_COL]}cm`,
  )
  check(
    groundState.folds[FLOAT_COL][lastJoint] !== 60 &&
      JSON.stringify(groundState.folds[FLOAT_COL].slice(0, lastJoint)) ===
        JSON.stringify(floatState.folds[FLOAT_COL].slice(0, lastJoint)) &&
      JSON.stringify(groundState.folds.filter((_, c) => c !== FLOAT_COL)) ===
        JSON.stringify(floatState.folds.filter((_, c) => c !== FLOAT_COL)),
    'only that column\'s LAST hinge moved',
    `${floatState.folds[FLOAT_COL][lastJoint]}° → ${groundState.folds[FLOAT_COL][lastJoint]}°`,
  )
  check(
    groundState.lastErrors.length === 0,
    'the grounding change committed through validation',
    groundState.lastErrors.join(', '),
  )
  check(
    await page.isVisible('[data-testid="badge-grounded"]') &&
      !(await page.isVisible(`[data-testid="column-floating-${FLOAT_COL}"]`)),
    'the badges are back to "all grounded ✓" and the column badge is gone',
  )
  check(canvasChanged(floatShot, groundShot), 'grounding the column changed the render')
  await page.screenshot({ path: `${OUT}/04-grounded.png` })
  console.log('  → tests/screenshots/04-grounded.png')

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
  check(canvasChanged(groundShot, foldShot), 'fold render differs from the grounded state')
  check(foldState.lastErrors.length === 0, 'no rejected changes so far', foldState.lastErrors.join(', '))
  await page.screenshot({ path: `${OUT}/05-fold.png` })
  console.log('  → tests/screenshots/05-fold.png')

  // --- (c2) an END THE SOLVER CANNOT REACH ---------------------------------
  // 80° at the front joint puts column 3's chain ~161cm up at the last row. The
  // last panel is 60cm long, so no angle within ±120° brings it to the floor: the
  // profile itself has to arc back down, and the store says so instead of guessing.
  check(
    foldState.floatingCols.includes(FLOAT_COL) && foldState.clearances[FLOAT_COL] > 100,
    `the 80° front fold left column ${FLOAT_COL} far too high to land`,
    `clearance=${foldState.clearances[FLOAT_COL]}cm`,
  )
  await page.click(`[data-testid="column-ground-${FLOAT_COL}"]`)
  await page.waitForTimeout(600)
  const unreachableState = await storeState(page)
  check(
    unreachableState.lastErrors.includes('E_UNGROUNDABLE') &&
      JSON.stringify(unreachableState.folds) === JSON.stringify(foldState.folds),
    '"Ground end" on an unreachable column reports E_UNGROUNDABLE and changes nothing',
    `lastErrors=${JSON.stringify(unreachableState.lastErrors)}`,
  )
  const noticeText = (await page.textContent('[data-testid="last-errors"]')) ?? ''
  check(
    noticeText.includes('E_UNGROUNDABLE') && /arc over and descend|has to arc/.test(noticeText),
    'the message box explains that the fold profile itself has to come back down',
    `${noticeText.length} chars`,
  )

  // --- (d) remove the preset plates, then "Shift →" ------------------------
  // The plates pin folds (a V plate's joint must stay 0, an H plate's two columns
  // must stay at equal pitch), so a whole-grid phase shift is a rect-free move.
  // Taking the crest plates off one at a time also walks the design DOWN through the
  // four-plate pattern rule, which must warn without ever blocking a removal.
  check(await page.isVisible('[data-testid="grid-map"]'), 'grid map is visible')
  const platedCells = foldState.rectObjs.map((r) => `cell-${r.row}-${r.col}`)
  check(platedCells.length === 6, 'six crest plates to remove', JSON.stringify(platedCells))
  let fewRectsSeenAt = null
  for (let i = 0; i < platedCells.length; i++) {
    await page.click(`[data-testid="${platedCells[i]}"]`)
    await page.waitForTimeout(200)
    const step = await storeState(page)
    if (fewRectsSeenAt === null && step.validationWarnings.includes('W_FEW_RECTS')) {
      fewRectsSeenAt = step.rectObjs.length
    }
  }
  await page.waitForTimeout(SETTLE_MS)
  const clearedState = await storeState(page)
  check(
    clearedState.rects.length === 0 && clearedState.panels === 30,
    'grid-map clicks removed all six plates',
    `rects=${JSON.stringify(clearedState.rects)} panels=${clearedState.panels}`,
  )
  check(
    fewRectsSeenAt === 3,
    'W_FEW_RECTS appeared the moment the design dropped to three plates',
    `first seen at ${fewRectsSeenAt} rects`,
  )
  check(
    clearedState.validationWarnings.includes('W_FEW_RECTS') && clearedState.lastErrors.length === 0,
    'a plate-less design warns W_FEW_RECTS but still commits (it is not an error)',
    `warnings=${JSON.stringify(clearedState.validationWarnings)} errors=${JSON.stringify(clearedState.lastErrors)}`,
  )
  const fewRectsText = (await page.textContent('.msg-warnings')) ?? ''
  check(
    fewRectsText.includes('W_FEW_RECTS') && /read in the pattern/.test(fewRectsText),
    "the warnings box explains the plate-pattern rule under the map",
    `${fewRectsText.length} chars`,
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
  // Placed by the PAIR-CLICK MERGE (WP14): arm the anchor cell, then click the
  // neighbour that makes the plate. Both of these pairs are FREE candidates — the
  // pitches already agree / the joint is already flat — so the merge adds the plate
  // and changes no geometry, exactly as the old H/V placement did.
  const hCell = findLegalHorizontal(shiftState)
  check(!!hCell, 'found a row where two neighbouring columns agree in pitch (H candidate)', JSON.stringify(hCell))
  if (!hCell) throw new Error('no legal horizontal placement under the current folds')
  await page.click(`[data-testid="cell-${hCell.row}-${hCell.col}"]`)
  await page.waitForTimeout(150)
  check(
    (await page.getAttribute(`[data-testid="cell-${hCell.row}-${hCell.col + 1}"]`, 'data-merge')) ===
      'free',
    `the pitch-matched neighbour of (${hCell.row}, ${hCell.col}) is highlighted as a FREE merge`,
  )
  await page.click(`[data-testid="cell-${hCell.row}-${hCell.col + 1}"]`)
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
  await page.click(`[data-testid="cell-${vCell.row}-${vCell.col}"]`)
  await page.waitForTimeout(150)
  check(
    (await page.getAttribute(`[data-testid="cell-${vCell.row + 1}-${vCell.col}"]`, 'data-merge')) ===
      'free',
    `the cell behind (${vCell.row}, ${vCell.col}) is highlighted as a FREE merge over the flat joint`,
  )
  await page.click(`[data-testid="cell-${vCell.row + 1}-${vCell.col}"]`)
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
  await page.screenshot({ path: `${OUT}/06-rects.png` })
  console.log('  → tests/screenshots/06-rects.png')

  // --- (f) the two illegal placements --------------------------------------
  // The map's pair-click merge can no longer OFFER these — mergeCandidates only
  // highlights merges core/merge.js has proven, and both of these would need
  // geometry it is not being asked to change. So they are driven straight at
  // `addRect`, the store action the JSON panel and the console still reach, to prove
  // the v2 rect constraints still bite and that the map still explains them inline.
  const placeRaw = (rect) =>
    page.evaluate((r) => window.__gridDesignerStore.getState().addRect(r), rect)

  const badH = findIllegalHorizontal(rectVState)
  check(!!badH, 'found a row where two columns DISAGREE in pitch (illegal H)', JSON.stringify(badH))
  if (badH) {
    await page.click(`[data-testid="cell-${badH.row}-${badH.col}"]`)
    await page.waitForTimeout(200)
    check(
      (await page.getAttribute(`[data-testid="cell-${badH.row}-${badH.col + 1}"]`, 'data-merge')) ===
        'coerce',
      `the pitch-mismatched neighbour of (${badH.row}, ${badH.col}) is offered only as a COERCE merge`,
    )
    await page.keyboard.press('Escape')
    await placeRaw({ row: badH.row, col: badH.col, orientation: 'horizontal' })
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
    await placeRaw({ row: badV.row, col: badV.col, orientation: 'vertical' })
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
  await page.screenshot({ path: `${OUT}/07-reject.png` })
  console.log('  → tests/screenshots/07-reject.png')

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
      jsonState.gap === 1 &&
      jsonState.rects.length === 0 &&
      jsonState.panels === 30 &&
      jsonState.lastErrors.length === 0,
    'normalizeConfig filled every omitted default (6×5, gap 1, no rects, 30 panels)',
    `cols=${jsonState.cols} rows=${jsonState.rows} gap=${jsonState.gap} panels=${jsonState.panels}`,
  )
  const jsonShot = await assertCanvasRenders(page, 'json')
  check(canvasChanged(rectsShot, jsonShot), 'pasted config re-rendered the surface')
  await page.screenshot({ path: `${OUT}/08-json.png` })
  console.log('  → tests/screenshots/08-json.png')

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

  // --- (g2) "Ground all" makes the progress it can --------------------------
  // The pasted profiles only ever climb, so all six columns end in the air and only
  // the shallowest is within one panel-length of the floor. "Ground all" must land
  // that one and leave the other five reported rather than mangling them.
  check(
    v1State.floatingCols.length === 6,
    'the pasted climb-only config leaves all six columns floating',
    JSON.stringify(v1State.clearances.map((v) => Math.round(v))),
  )
  await page.click('[data-testid="tool-ground-all"]')
  await page.waitForTimeout(SETTLE_MS)
  const groundAllState = await storeState(page)
  check(
    groundAllState.floatingCols.length === 5 && !groundAllState.floatingCols.includes(5),
    '"Ground all" landed the one reachable column and left the rest flagged',
    `floating=${JSON.stringify(groundAllState.floatingCols)}`,
  )
  check(
    groundAllState.folds[5][3] !== v1State.folds[5][3] &&
      JSON.stringify(groundAllState.folds.slice(0, 5)) === JSON.stringify(v1State.folds.slice(0, 5)),
    'it only moved the last hinge of the column it could solve',
    `col5: ${JSON.stringify(v1State.folds[5])} → ${JSON.stringify(groundAllState.folds[5])}`,
  )
  check(
    groundAllState.lastErrors.length === 0 &&
      (await page.textContent('[data-testid="badge-floating"]')).includes('5'),
    'the change committed and the summary badge now reads "5 floating"',
    `lastErrors=${JSON.stringify(groundAllState.lastErrors)}`,
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
      downloadedConfig?.gap === 1,
    'downloaded JSON is the current, fully-defaulted v2 config',
    `${jsonText.length} bytes`,
  )

  // --- (i) THE STORM FAMILY: pitched fronts, a deeper grid, the wall --------
  // 'swell' is 7 rows with a STEEP pitched front in EVERY column and a spine plate
  // mid-strip in each — the family's two signatures at once — and everything ramps
  // toward the wall beside column 0, where it peaks.
  await page.click('[data-testid="preset-swell"]')
  await page.waitForTimeout(SETTLE_MS)
  const swellShot = await assertCanvasRenders(page, 'swell')
  const swellState = await storeState(page)
  check(swellState.preset === 'swell', 'Swell preset applied', `preset=${swellState.preset}`)
  check(
    swellState.rows === 7 &&
      (await page.textContent('[data-testid="rows-value"]')).includes('7 rows'),
    'the rows stepper followed the preset to 7 rows',
    `rows=${swellState.rows} stepper="${(await page.textContent('[data-testid="rows-value"]')).trim()}"`,
  )
  check(
    swellState.folds.every((f) => f.length === 6),
    'every column carries grid.rows-1 = 6 hinges',
    JSON.stringify(swellState.folds.map((f) => f.length)),
  )
  check(
    swellState.startPitches.every((p) => p > 0) &&
      JSON.stringify(swellState.startPitches) === JSON.stringify([42, 38, 33, 29, 24, 20]),
    "swell's signature: NO flat front anywhere — the pitches ramp 42° → 20° DOWN from the wall",
    JSON.stringify(swellState.startPitches),
  )
  check(
    swellState.folds.every((f) => Math.abs(f[0]) >= 12 && f[0] < 0),
    'swell: the shoreline BREAKS — joint 0 is a real counter-bend (≥ 12° back) in every column',
    JSON.stringify(swellState.folds.map((f) => f[0])),
  )
  check(
    swellState.panels === 36 && swellState.rectObjs.length === 6,
    'swell: 42 cells − 6 spine plates = 36 panels',
    `panels=${swellState.panels} rects=${swellState.rectObjs.length}`,
  )
  const swellCaption = (await page.textContent('[data-testid="rect-pattern"]')) ?? ''
  check(
    swellState.rectPattern === 'spine plates' && swellCaption.includes('spine plates'),
    'the grid map captions the "spine plates" pattern',
    swellCaption.trim(),
  )
  check(
    swellState.violations.length === 0 &&
      swellState.grounded.every(Boolean) &&
      swellState.chainEndSupports.every((s) => s === 'floor'),
    'swell: every column stands on the FLOOR and lands (no wall support anywhere)',
    `${JSON.stringify(swellState.chainEndSupports)} clearances=${JSON.stringify(swellState.clearances)}`,
  )
  check(
    await page.isVisible('[data-testid="badge-grounded"]'),
    'swell shows "all grounded ✓"',
  )
  check(
    (await page.textContent('[data-testid="column-5"]')).includes('20°'),
    "column 5's readout shows its 20° front pitch — the low end of the ramp",
    `startPitches=${JSON.stringify(swellState.startPitches)}`,
  )
  check(
    (await page.inputValue('[data-testid="front-0"]')) === '42',
    "the FRONT slider of column 0 reads its 42° pitch — the steep end, against the wall",
  )
  // The ramp itself, read off the solved chains the viewport is drawing: the surface
  // rises monotonically toward column 0 (the wall, on the RIGHT of the 3D view), from
  // a LOW start at the far jamb (the LEFT) to a moderate crest — no longer 150cm.
  check(
    swellState.peaks.every((y, c) => c === 0 || y < swellState.peaks[c - 1]) &&
      swellState.peaks[0] >= 110 &&
      swellState.peaks[0] <= 120 &&
      swellState.peaks[5] >= 35 &&
      swellState.peaks[5] <= 45,
    'swell: the surface RAMPS from a ~42cm start at the far jamb to a ~116cm crest at the wall',
    JSON.stringify(swellState.peaks.map((y) => Number(y.toFixed(1)))),
  )
  // The measuring box follows the design: the peak label is the number being tuned.
  check(
    Math.abs(swellState.bounds.max[1] - swellState.peaks[0]) < 6 &&
      swellState.bounds.size[1] < 125,
    "swell: the box's peak agrees with the tallest column and stays under the new ceiling",
    JSON.stringify(swellState.bounds.size.map((v) => Math.round(v))),
  )
  const swellPeakLabel = (await page.textContent('[data-testid="bounds-label-y"]')).trim()
  const swellBoundsLine = (await page.textContent('[data-testid="bounds-summary"]'))
    .replace(/\s+/g, ' ')
    .trim()
  const wantSwellLine = `${swellState.bounds.size.map((v) => Math.round(v)).join(' × ')} cm`
  check(
    swellPeakLabel === `peak ${Math.round(swellState.bounds.max[1])} cm` &&
      swellBoundsLine === wantSwellLine,
    'swell: the 3D peak pill and the panel line both report the retuned dimensions in cm',
    `${swellPeakLabel} / "${swellBoundsLine}" want "${wantSwellLine}"`,
  )
  check(canvasChanged(jsonShot, swellShot), 'the swell render differs from the pasted config')
  await page.screenshot({ path: `${OUT}/09-swell.png` })
  console.log('  → tests/screenshots/09-swell.png')

  // The BOX itself, not just its labels: on a design this tall the wireframe is a
  // large part of the frame, so hiding it has to change the rendered pixels. This is
  // the assertion that the toggle reaches the 3D scene and not only the DOM overlay.
  await page.click('[data-testid="tool-bounds"]')
  await page.waitForTimeout(400)
  const swellNoBox = await assertCanvasRenders(page, 'swell/bounds hidden')
  check(
    canvasChanged(swellShot, swellNoBox) &&
      (await page.locator('[data-testid="bounds-label-y"]').count()) === 0,
    'swell: hiding "bounds" removes the wireframe BOX from the render, not just the pills',
  )
  await page.click('[data-testid="tool-bounds"]')
  await page.waitForTimeout(400)
  const swellBoxBack = await assertCanvasRenders(page, 'swell/bounds shown')
  check(
    canvasChanged(swellNoBox, swellBoxBack) &&
      (await page.locator('[data-testid="bounds-label-y"]').count()) === 1,
    'swell: and showing it again brings both back',
  )

  // 'wallcrash' is the one design that engages the −X wall, beside column 0.
  await page.click('[data-testid="preset-wallcrash"]')
  await page.waitForTimeout(SETTLE_MS)
  const wallShot = await assertCanvasRenders(page, 'wallcrash')
  const wallState = await storeState(page)
  check(wallState.preset === 'wallcrash', 'Wallcrash preset applied', `preset=${wallState.preset}`)
  check(
    JSON.stringify(wallState.endSupports) ===
      JSON.stringify(['wall', 'floor', 'floor', 'floor', 'floor', 'floor']),
    'only the wall-adjacent column 0 is wall-supported',
    JSON.stringify(wallState.endSupports),
  )
  check(
    wallState.startPitches.every((p) => p !== 0) &&
      JSON.stringify(wallState.startPitches) === JSON.stringify([43, 36, 29, 22, 15, 8]),
    'wallcrash: pitched fronts RAMPING toward the wall — steepest at column 0 (43° → 8°)',
    JSON.stringify(wallState.startPitches),
  )
  const wallBadge = (await page.textContent('[data-testid="column-wall-0"]')) ?? ''
  check(
    wallBadge.includes('WALL-SUPPORTED') && /\d+cm up/.test(wallBadge),
    "column 0's block reads WALL-SUPPORTED with the height its end reaches",
    wallBadge.trim(),
  )
  check(
    !(await page.isVisible('[data-testid="column-floating-0"]')) &&
      !(await page.isVisible('[data-testid="column-ground-0"]')),
    'the wall column shows no FLOATING badge and no "Ground end" button',
  )
  check(
    (await page.getAttribute('[data-testid="profile-end-0"]', 'data-end-support')) === 'wall' &&
      (await page.getAttribute('[data-testid="profile-end-5"]', 'data-end-support')) === 'floor',
    'the sparkline endpoint marker is tagged wall-supported for column 0 only',
  )
  check(
    wallState.violations.length === 0 && wallState.lastErrors.length === 0,
    'wallcrash: NO violations at all, even though column 0 ends in mid-air',
    `violations=${JSON.stringify(wallState.violations)} errors=${JSON.stringify(wallState.lastErrors)}`,
  )
  check(
    wallState.grounded[0] === false && wallState.clearances[0] > 30,
    "column 0's end is genuinely elevated — the splash up the wall",
    `clearance=${wallState.clearances[0]}cm`,
  )
  check(
    wallState.grounded.slice(1).every(Boolean),
    'columns 1–5 all land on the floor',
    JSON.stringify(wallState.clearances.slice(1)),
  )
  check(
    await page.isVisible('[data-testid="badge-grounded"]'),
    'the summary badge still reads "all grounded ✓" — the wall column is exempt',
  )
  check(
    await page.isVisible('[data-testid="gridmap-wall"]'),
    'the grid map draws the wall rule along the −X (left) edge',
  )
  check(canvasChanged(swellShot, wallShot), 'the wallcrash render differs from swell')
  await page.screenshot({ path: `${OUT}/10-wallcrash.png` })
  console.log('  → tests/screenshots/10-wallcrash.png')

  // --- (i2) the END-SUPPORT toggle: take the wall away and the rule bites ----
  check(
    (await page.isVisible('[data-testid="column-endsupport-0"]')) &&
      !(await page.isVisible('[data-testid="column-endsupport-1"]')) &&
      !(await page.isVisible('[data-testid="column-endsupport-5"]')),
    'only column 0 — the column against the wall — offers an end-support toggle',
  )
  await page.click('[data-testid="column-endsupport-0"]')
  await page.waitForTimeout(600)
  const unwalledState = await storeState(page)
  check(
    unwalledState.endSupports[0] === 'floor' &&
      JSON.stringify(unwalledState.floatingCols) === JSON.stringify([0]) &&
      unwalledState.violations.every((code) => code === 'E_END_FLOATING'),
    'standing column 0 on the floor makes the SAME geometry break the grounded-end rule',
    `endSupports=${JSON.stringify(unwalledState.endSupports)} floating=${JSON.stringify(unwalledState.floatingCols)}`,
  )
  check(
    JSON.stringify(unwalledState.folds) === JSON.stringify(wallState.folds) &&
      JSON.stringify(unwalledState.startPitches) === JSON.stringify(wallState.startPitches),
    'the toggle changed no angle at all — only how the end is held up',
    JSON.stringify(unwalledState.folds[0]),
  )
  check(
    (await page.isVisible('[data-testid="column-floating-0"]')) &&
      (await page.isVisible('[data-testid="column-ground-0"]')) &&
      !(await page.isVisible('[data-testid="column-wall-0"]')),
    'it now shows the FLOATING badge and a "Ground end" button instead of the wall badge',
  )
  await page.click('[data-testid="column-endsupport-0"]')
  await page.waitForTimeout(600)
  const rewalledState = await storeState(page)
  check(
    rewalledState.endSupports[0] === 'wall' &&
      rewalledState.violations.length === 0 &&
      rewalledState.lastErrors.length === 0,
    'bracketing it back to the wall clears the violation again',
    `endSupports=${JSON.stringify(rewalledState.endSupports)} violations=${JSON.stringify(rewalledState.violations)}`,
  )

  // --- (j) the ROWS STEPPER on a flat design -------------------------------
  await page.click('[data-testid="preset-flat"]')
  await page.waitForTimeout(SETTLE_MS)
  const flatAgain = await storeState(page)
  check(
    flatAgain.rows === 5 && flatAgain.panels === 26,
    'back to the flat preset: 5 rows, 26 panels',
    `rows=${flatAgain.rows} panels=${flatAgain.panels}`,
  )
  check(
    await page.isDisabled('[data-testid="rows-dec"]'),
    'the rows stepper cannot go below 5',
  )
  await page.click('[data-testid="rows-inc"]')
  await page.waitForTimeout(SETTLE_MS)
  const rowsState = await storeState(page)
  const rowsShot = await assertCanvasRenders(page, 'rows')
  check(
    rowsState.rows === 6 &&
      (await page.textContent('[data-testid="rows-value"]')).includes('6 rows'),
    'the stepper took the flat grid from 5 rows to 6',
    `rows=${rowsState.rows}`,
  )
  check(
    rowsState.panels === 32 && rowsState.panels > flatAgain.panels,
    'the surface GREW: 36 cells − the 4 kept plates = 32 panels',
    `${flatAgain.panels} → ${rowsState.panels} panels`,
  )
  check(
    rowsState.folds.every((f) => f.length === 5) &&
      rowsState.folds.every((f, c) =>
        JSON.stringify(f.slice(0, 4)) === JSON.stringify(flatAgain.folds[c]),
      ),
    'every column got a fifth hinge appended at the BACK; the front four are untouched',
    JSON.stringify(rowsState.folds),
  )
  check(
    JSON.stringify(rowsState.rects) === JSON.stringify(flatAgain.rects) &&
      rowsState.lastErrors.length === 0,
    'the four plates all still fit, so none were dropped, and it committed',
    `${JSON.stringify(rowsState.rects)} errors=${JSON.stringify(rowsState.lastErrors)}`,
  )
  check(
    rowsState.violations.length === 0,
    'the extra row is level, so every column still lands',
    JSON.stringify(rowsState.violations),
  )
  // …up to the ceiling, and back down again.
  await page.click('[data-testid="rows-inc"]')
  await page.click('[data-testid="rows-inc"]')
  await page.waitForTimeout(SETTLE_MS)
  const maxRowsState = await storeState(page)
  check(
    maxRowsState.rows === 8 && (await page.isDisabled('[data-testid="rows-inc"]')),
    'the stepper reaches 8 rows and stops there',
    `rows=${maxRowsState.rows}`,
  )
  check(
    maxRowsState.panels === 44 && maxRowsState.folds.every((f) => f.length === 7),
    'at 8 rows: 48 cells − 4 plates = 44 panels, 7 hinges per column',
    `panels=${maxRowsState.panels} folds=${JSON.stringify(maxRowsState.folds.map((f) => f.length))}`,
  )
  await page.click('[data-testid="rows-dec"]')
  await page.click('[data-testid="rows-dec"]')
  await page.waitForTimeout(SETTLE_MS)
  const backRowsState = await storeState(page)
  check(
    backRowsState.rows === 6 && backRowsState.panels === 32,
    'stepping back down to 6 rows restores the 32-panel surface',
    `rows=${backRowsState.rows} panels=${backRowsState.panels}`,
  )
  check(canvasChanged(wallShot, rowsShot), 'the row-resolution change re-rendered the surface')
  await page.screenshot({ path: `${OUT}/11-rows.png` })
  console.log('  → tests/screenshots/11-rows.png')

  // --- (k) the FRONT slider: pitch one column's window panel ---------------
  const FRONT_COL = 2
  await page.locator(`[data-testid="front-${FRONT_COL}"]`).fill('35')
  await page.waitForTimeout(SETTLE_MS)
  let frontState = await storeState(page)
  if (frontState.startPitches[FRONT_COL] !== 35) {
    console.log('! front slider fill did not register — falling back to the store action')
    await page.evaluate(
      (c) => window.__gridDesignerStore.getState().setColumnStartPitch(c, 35),
      FRONT_COL,
    )
    await page.waitForTimeout(SETTLE_MS)
    frontState = await storeState(page)
  }
  const frontShot = await assertCanvasRenders(page, 'front')
  check(
    frontState.startPitches[FRONT_COL] === 35,
    `the FRONT slider set column ${FRONT_COL}'s window panel to 35°`,
    JSON.stringify(frontState.startPitches),
  )
  check(
    JSON.stringify(frontState.startPitches.filter((_, c) => c !== FRONT_COL)) ===
      JSON.stringify([0, 0, 0, 0, 0]),
    'no other column was touched',
    JSON.stringify(frontState.startPitches),
  )
  check(
    JSON.stringify(frontState.folds) === JSON.stringify(backRowsState.folds),
    'it changed the front pitch, not any hinge',
    JSON.stringify(frontState.folds),
  )
  check(
    frontState.pitches[FRONT_COL].every((p) => p === 35),
    'the whole strip pitched with it — every row of that column now reads 35°',
    JSON.stringify(frontState.pitches[FRONT_COL]),
  )
  check(
    frontState.lastErrors.length === 0 &&
      (await page.inputValue(`[data-testid="front-${FRONT_COL}"]`)) === '35',
    'it committed through validation and the slider shows the value',
    JSON.stringify(frontState.lastErrors),
  )
  // A strip that only ever climbs cannot land — the grounded-end rule notices.
  check(
    frontState.floatingCols.includes(FRONT_COL),
    'a 35° front with no descent behind it leaves that column floating',
    `clearance=${frontState.clearances[FRONT_COL]}cm`,
  )
  check(canvasChanged(rowsShot, frontShot), 'the front pitch re-rendered the surface')
  await page.screenshot({ path: `${OUT}/12-front.png` })
  console.log('  → tests/screenshots/12-front.png')

  // --- (l) THE PANEL CLOSE-UP (WP11) ---------------------------------------
  // The panel solid used to be built with its front and back caps wound
  // inside-out: under FrontSide materials both were culled, so panels rendered
  // with no lit face and no housing back. Nothing in this harness noticed,
  // because every whole-surface view happens to look at the panels' EDGES and
  // silhouettes as much as their faces.
  //
  // So: park the camera beside the deepest storm surface, close enough that two
  // or three adjacent panels fill the frame, and at an angle where the SAME
  // frame holds panels facing the camera and panels facing away from it. Then
  // assert, from the layout itself, that both really are in shot.
  const errorsBeforeDetail = consoleErrors.length
  await page.click('[data-testid="preset-surge"]')
  await page.waitForTimeout(SETTLE_MS)
  check(
    await page.evaluate(() => typeof window.__gridDesignerCamera === 'function'),
    'the viewport exposes window.__gridDesignerCamera for close-up framing',
  )
  const DETAIL_CAM = [470, 120, 315]
  const DETAIL_TARGET = [345, 100, 240]
  await page.evaluate(
    ([pos, target]) => window.__gridDesignerCamera(pos, target),
    [DETAIL_CAM, DETAIL_TARGET],
  )
  await page.waitForTimeout(SETTLE_MS)

  // Which panels near the camera show their LIT FACE, and which their BACK?
  // A panel's world lit direction is its stored quaternion applied to +Y; the
  // sign of its dot product with (camera − panel) says which side we can see.
  const facing = await page.evaluate(
    ([cam, radius]) => {
      const { layout } = window.__gridDesignerDerived()
      /**
       * The panel's world LIT direction: its stored quaternion applied to the
       * local +Y axis. Written out longhand (three's Vector3.applyQuaternion
       * specialized to v = (0,1,0)) so nothing from src/ is involved:
       *   t = 2·(q.xyz × v) = (−2qz, 0, 2qx)
       *   v' = v + qw·t + q.xyz × t
       */
      const litDirection = (q) => {
        const norm = Math.hypot(q[0], q[1], q[2], q[3]) || 1
        const qx = q[0] / norm
        const qy = q[1] / norm
        const qz = q[2] / norm
        const qw = q[3] / norm
        const tx = -2 * qz
        const ty = 0
        const tz = 2 * qx
        return [
          0 + qw * tx + (qy * tz - qz * ty),
          1 + qw * ty + (qz * tx - qx * tz),
          0 + qw * tz + (qx * ty - qy * tx),
        ]
      }
      let face = 0
      let back = 0
      let near = 0
      for (const panel of layout.panels) {
        const d = [
          cam[0] - panel.position[0],
          cam[1] - panel.position[1],
          cam[2] - panel.position[2],
        ]
        if (Math.hypot(d[0], d[1], d[2]) > radius) continue
        near++
        const n = litDirection(panel.quaternion)
        const dot = n[0] * d[0] + n[1] * d[1] + n[2] * d[2]
        if (dot > 0) face++
        else back++
      }
      return { near, face, back }
    },
    [DETAIL_CAM, 260],
  )
  check(
    facing.near >= 3,
    'the close-up camera sits within 260cm of at least three panels',
    JSON.stringify(facing),
  )
  check(
    facing.face > 0 && facing.back > 0,
    'the close-up frames BOTH lit faces and backs — the two surfaces the inside-out caps used to hide',
    `${facing.face} facing the camera, ${facing.back} showing their back (of ${facing.near} nearby)`,
  )
  const detailShot = await assertCanvasRenders(page, 'panel-detail')
  check(canvasChanged(frontShot, detailShot), 'the close-up render differs from the wide front view')
  check(
    consoleErrors.length === errorsBeforeDetail,
    'the close-up produced no console errors',
    consoleErrors.slice(errorsBeforeDetail).join(' | '),
  )
  await page.screenshot({ path: `${OUT}/13-panel-detail.png` })
  console.log('  → tests/screenshots/13-panel-detail.png')

  // --- (n) WP14: PAIR-CLICK MERGE — two squares into one plate -------------
  // The inverse of clicking a plate to split it, on the one preset where it costs
  // something: every joint of 'swell' is folded and no two columns agree in pitch,
  // so a merge there has to CHANGE the surface (core/merge.js). Splitting column 4's
  // spine plate first leaves joint 2 of that column flat, which is what puts a FREE
  // candidate and a COERCE candidate in the same frame.
  await page.evaluate(() => window.__gridDesignerCamera([182.5, 220, -320], [182.5, 30, 150]))
  // The control panel may have been scrolled by earlier clicks (the JSON panel is
  // expanded by now) — put the map back in frame before shooting it.
  await page.evaluate(() => document.querySelector('.control-panel').scrollTo(0, 0))
  await page.click('[data-testid="preset-swell"]')
  await page.waitForTimeout(SETTLE_MS)
  const mergeBase = await storeState(page)
  check(
    mergeBase.preset === 'swell' && mergeBase.rects.includes('v(2,4)'),
    'merge stage starts from swell, spine plate on column 4 included',
    JSON.stringify(mergeBase.rects),
  )
  const mergeBaseShot = await assertCanvasRenders(page, 'merge-base')

  await page.click('[data-testid="cell-2-4"]')
  await page.waitForTimeout(400)
  const splitState = await storeState(page)
  check(
    !splitState.rects.includes('v(2,4)') && splitState.rects.length === mergeBase.rects.length - 1,
    'clicking a plate splits it back into two squares',
    JSON.stringify(splitState.rects),
  )
  check(
    splitState.folds[4][2] === 0,
    "the split plate's joint is flat, so re-merging that pair is free",
    `${splitState.folds[4][2]}`,
  )

  await page.click('[data-testid="cell-2-4"]')
  await page.waitForTimeout(300)
  check(await page.isVisible('[data-testid="gridmap-armed-label"]'), 'clicking a square ARMS it')
  const armedLabel = (await page.textContent('[data-testid="gridmap-armed-label"]')) ?? ''
  check(
    armedLabel.includes('armed (2, 4)') && /Esc/.test(armedLabel),
    'the map says which square is armed, and how to let go',
    armedLabel,
  )
  check(
    (await page.getAttribute('[data-testid="cell-3-4"]', 'data-merge')) === 'free',
    'the cell across the flattened joint is highlighted as a FREE merge',
  )
  check(
    (await page.getAttribute('[data-testid="cell-1-4"]', 'data-merge')) === 'coerce',
    'the cell across a FOLDED joint is highlighted as a COERCE merge',
  )
  check(
    (await page.isVisible('[data-testid="candidate-3-4"]')) &&
      (await page.isVisible('[data-testid="candidate-1-4"]')),
    'both candidate styles are drawn on the map at once',
  )
  check(
    (await page.getAttribute('[data-testid="candidate-3-4"]', 'data-kind')) === 'free' &&
      (await page.getAttribute('[data-testid="candidate-1-4"]', 'data-kind')) === 'coerce',
    'the two highlights are drawn in different styles (free vs coerce)',
  )
  const coerceTip = (await page.textContent('[data-testid="cell-1-4"] title')) ?? ''
  check(
    /flattened joint 1 of column 4/.test(coerceTip),
    'the coerce candidate spells its consequence out BEFORE the click',
    coerceTip,
  )
  await page.screenshot({ path: `${OUT}/14-merge.png` })
  console.log('  → tests/screenshots/14-merge.png')

  // Escape lets go.
  await page.keyboard.press('Escape')
  await page.waitForTimeout(250)
  check(
    !(await page.isVisible('[data-testid="gridmap-armed-label"]')) &&
      (await page.getAttribute('[data-testid="cell-1-4"]', 'data-merge')) === null,
    'Escape disarms the square and clears the highlights',
  )

  // Merge the COERCE pair: one plate added, one joint flattened, nothing else.
  await page.click('[data-testid="cell-2-4"]')
  await page.waitForTimeout(200)
  await page.click('[data-testid="cell-1-4"]')
  await page.waitForTimeout(SETTLE_MS)
  const mergedState = await storeState(page)
  check(
    mergedState.rects.includes('v(1,4)') && mergedState.rects.length === splitState.rects.length + 1,
    'clicking a highlighted candidate merged the two squares into a 121cm plate',
    JSON.stringify(mergedState.rects),
  )
  check(
    mergedState.folds[4][1] === 0 && splitState.folds[4][1] !== 0,
    'the merge flattened the joint the rigid plate now spans',
    `${splitState.folds[4][1]}° → ${mergedState.folds[4][1]}°`,
  )
  check(
    mergedState.folds.every((col, c) =>
      col.every((v, k) => v === splitState.folds[c][k] || (c === 4 && k === 1)),
    ),
    'and it changed no other joint in the grid',
    JSON.stringify(mergedState.folds),
  )
  check(mergedState.lastErrors.length === 0, 'the merge committed through validation', JSON.stringify(mergedState.lastErrors))
  const notice = (await page.textContent('[data-testid="gridmap-notice"]')) ?? ''
  check(
    /flattened joint 1 of column 4/.test(notice) && /121cm plate/.test(notice),
    'the notice under the map names the plate and the joint it flattened',
    notice,
  )
  check(
    !(await page.isVisible('[data-testid="gridmap-armed-label"]')),
    'the merge disarmed the square it started from',
  )
  const mergedShot = await assertCanvasRenders(page, 'merged')
  check(canvasChanged(mergeBaseShot, mergedShot), 'the merged surface renders differently from swell')

  // Undo / redo — the way back from a merge's geometry coercion.
  check(!(await page.isDisabled('[data-testid="tool-undo"]')), 'Undo is enabled after the merge')
  await page.click('[data-testid="tool-undo"]')
  await page.waitForTimeout(400)
  const undoneState = await storeState(page)
  check(
    JSON.stringify(undoneState.folds) === JSON.stringify(splitState.folds) &&
      JSON.stringify(undoneState.rects) === JSON.stringify(splitState.rects),
    'Undo restored the config exactly as it was before the merge',
    JSON.stringify(undoneState.rects),
  )
  check(!(await page.isDisabled('[data-testid="tool-redo"]')), 'Redo is enabled after undoing')
  await page.click('[data-testid="tool-redo"]')
  await page.waitForTimeout(400)
  const redoneState = await storeState(page)
  check(
    JSON.stringify(redoneState.folds) === JSON.stringify(mergedState.folds) &&
      JSON.stringify(redoneState.rects) === JSON.stringify(mergedState.rects),
    'Redo re-applied the merge',
    JSON.stringify(redoneState.rects),
  )
  // …and the keyboard does the same thing as the buttons.
  await page.keyboard.press(process.platform === 'darwin' ? 'Meta+z' : 'Control+z')
  await page.waitForTimeout(400)
  const shortcutState = await storeState(page)
  check(
    JSON.stringify(shortcutState.rects) === JSON.stringify(splitState.rects),
    'Cmd/Ctrl+Z undoes as well as the toolbar button',
    JSON.stringify(shortcutState.rects),
  )
  await page.keyboard.press(process.platform === 'darwin' ? 'Shift+Meta+z' : 'Shift+Control+z')
  await page.waitForTimeout(400)
  check(
    JSON.stringify((await storeState(page)).rects) === JSON.stringify(mergedState.rects),
    'Shift+Cmd/Ctrl+Z redoes it',
  )

  // A merge is NOT undone by splitting the plate again: the joint stays flat.
  await page.click('[data-testid="cell-1-4"]')
  await page.waitForTimeout(400)
  const resplitState = await storeState(page)
  check(
    !resplitState.rects.includes('v(1,4)'),
    'the merged plate splits back into two squares',
    JSON.stringify(resplitState.rects),
  )
  check(
    resplitState.folds[4][1] === 0 && resplitState.lastErrors.length === 0,
    'but the flattened joint STAYS flat — a split is not an undo (that is what Undo is for)',
    `${resplitState.folds[4][1]}`,
  )

  // Finally a FREE merge, to show the other half of the pair-click.
  await page.click('[data-testid="cell-2-4"]')
  await page.waitForTimeout(200)
  await page.click('[data-testid="cell-3-4"]')
  await page.waitForTimeout(400)
  const freeMergeState = await storeState(page)
  const freeNotice = (await page.textContent('[data-testid="gridmap-notice"]')) ?? ''
  check(
    freeMergeState.rects.includes('v(2,4)') &&
      JSON.stringify(freeMergeState.folds) === JSON.stringify(resplitState.folds),
    'a FREE merge adds the plate and changes no geometry at all',
    JSON.stringify(freeMergeState.rects),
  )
  check(
    /already flat/.test(freeNotice),
    'and its notice says so',
    freeNotice,
  )

  // --- byte-level differences ---------------------------------------------
  const shots = [
    '01-flat.png',
    '02-wave.png',
    '03-floating.png',
    '04-grounded.png',
    '05-fold.png',
    '06-rects.png',
    '07-reject.png',
    '08-json.png',
    '09-swell.png',
    '10-wallcrash.png',
    '11-rows.png',
    '12-front.png',
    '13-panel-detail.png',
    '14-merge.png',
  ].map(
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
console.log('\n=== SCREENSHOT HARNESS (WP11 — the panel solid) ===')
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
