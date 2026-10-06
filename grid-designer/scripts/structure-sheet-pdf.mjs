#!/usr/bin/env node
/**
 * grid-designer — print the structure sheet to a US Letter PDF.
 *
 *   node scripts/structure-sheet-pdf.mjs [sheetDir]
 *
 * Renders `print.html` (written by structure-sheet.mjs) to
 * `structure-sheet-letter.pdf` beside it, using Playwright driving the
 * installed Google Chrome (present on GitHub's ubuntu runners too), or
 * Playwright's own Chromium when there is none.
 *
 * REFUSES TO WRITE A PDF THAT CLIPS. The print layout is fixed pages with
 * `overflow: hidden`, so a design with a longer kit or an extra warning could
 * silently push content off the bottom of a page — still a valid PDF, just
 * missing a line nobody would know to look for. Every page is measured first.
 */

import path from 'node:path'
import { pathToFileURL } from 'node:url'
import { chromium } from 'playwright'

const sheetDir = path.resolve(process.argv[2] ?? 'sheet')
const src = path.join(sheetDir, 'print.html')
const out = path.join(sheetDir, 'structure-sheet-letter.pdf')

let browser
try {
  browser = await chromium.launch({ channel: 'chrome' })
} catch {
  browser = await chromium.launch()
}

try {
  const page = await browser.newPage()
  await page.goto(pathToFileURL(src).href, { waitUntil: 'networkidle' })
  await page.evaluate(() => document.fonts.ready)
  await page.emulateMedia({ media: 'print' })

  const pages = await page.evaluate(() =>
    [...document.querySelectorAll('.page')].map((el) => {
      // The running foot is pinned to the bottom margin, so measuring to the
      // page edge would always read zero. Measure the content instead: the
      // lowest edge of anything that is not the foot, against the foot's top —
      // and the foot itself against the page's content box.
      const box = el.getBoundingClientRect()
      const pad = parseFloat(getComputedStyle(el).paddingBottom)
      const foot = el.querySelector('.foot')
      // Only things that DRAW count — text-bearing leaves and whole drawings.
      // Layout boxes that grow to fill the page (the plan row is flex: 1) would
      // otherwise always reach the foot and hide the real slack.
      let lowest = 0
      for (const child of el.querySelectorAll('*')) {
        if (foot && (child === foot || foot.contains(child))) continue
        if (child.closest('svg') && child.tagName.toLowerCase() !== 'svg') continue
        const isDrawing = child.tagName.toLowerCase() === 'svg'
        // text counts wherever it sits — a legend line that holds a swatch
        // element AND its words is still a line of words
        const holdsText = [...child.childNodes].some((n) => n.nodeType === 3 && n.textContent.trim())
        if (!isDrawing && !holdsText && child.children.length > 0) continue
        const r = child.getBoundingClientRect()
        if (r.height > 0) lowest = Math.max(lowest, r.bottom)
      }
      const floor = foot ? foot.getBoundingClientRect().top : box.bottom - pad
      const footOverIn = foot ? (foot.getBoundingClientRect().bottom - (box.bottom - pad)) / 96 : 0
      const fonts = [...document.fonts].filter((f) => f.status === 'loaded').map((f) => f.family)
      return { id: el.id, spareIn: (floor - lowest) / 96, footOverIn, fonts: [...new Set(fonts)] }
    }),
  )
  const clipped = pages.filter((p) => p.spareIn < -0.005 || p.footOverIn > 0.005)
  for (const p of pages) {
    console.log(p.footOverIn > 0.005
      ? `  ${p.id}: content pushes the foot ${p.footOverIn.toFixed(2)}in past the bottom margin`
      : `  ${p.id}: ${p.spareIn.toFixed(2)}in free above the foot`)
  }
  if (clipped.length) {
    console.error(`structure-sheet-pdf: ${clipped.map((p) => p.id).join(', ')} overflow the page — no PDF written`)
    process.exitCode = 1
  } else {
    const fonts = pages[0]?.fonts ?? []
    if (!fonts.some((f) => /Archivo/.test(f))) console.warn('  warning: Archivo did not load — the PDF uses the fallback face')
    await page.pdf({ path: out, preferCSSPageSize: true, printBackground: true })
    console.log(`structure sheet PDF → ${path.relative(process.cwd(), out)}`)
  }
} finally {
  await browser.close()
}
