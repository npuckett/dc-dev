#!/usr/bin/env node
/**
 * grid-designer — the STRUCTURE SHEET generator.
 *
 *   node scripts/structure-sheet.mjs [design.json | -] [outDir]
 *
 * Reads a design (the file the editor's "Download JSON" writes; by default the
 * design of record, `design/current.json`) and writes, into `outDir` (default
 * `sheet/`):
 *
 *   structure.json            the headline numbers — panels, angle, connectors
 *   plan.svg                  a flat-shaded plan view from above
 *   connector-exploded.svg    the connector's pieces, exploded along the bolts
 *   connector-sections.svg    both back-half types, assembled, in section
 *   pieces/*.stl              the connector as separate printable pieces (mm)
 *   structure-sheet.html      all of the above as one page (the artifact body:
 *                             no doctype, see the template)
 *   index.html                the same page as a full document, for GitHub
 *                             Pages, with the STLs as download links
 *
 * EVERYTHING IS COMPUTED FROM THE SAME CORE THE EDITOR RUNS. The panel corners,
 * station frames, connector profiles and STL meshes all come from
 * `src/core/` and `src/utils/connectorExport.js`; nothing here restates a
 * dimension. A drawing that disagreed with the model would be worse than none.
 *
 * Bolt and insert POSITIONS are not part of the model yet (only their count
 * and size are), so the exploded view puts them on evenly spaced axes and the
 * sheet says so.
 */

import fs from 'node:fs'
import path from 'node:path'
import { execSync } from 'node:child_process'
import { normalizeConfig, DEFAULT_CONFIG } from '../src/core/v4/schema.js'
import { solveLattice, latticeBounds } from '../src/core/v4/lattice.js'
import { solveConnectorsV4 } from '../src/core/v4/connectors.js'
import { buildReportV4 } from '../src/core/v4/report.js'
import { obstacleExtents } from '../src/core/v4/obstacles.js'
import { getConnectorKit } from '../src/v4/exportAdapter.js'
import { connectorPieces, stlPayload, MM_PER_CM } from '../src/utils/connectorExport.js'
import {
  backHalfProfile,
  frontBarProfile,
  frontBarWidthFor,
  flangeDepthAt,
  CONNECTOR_PROFILE,
} from '../src/core/v3/connectors.js'
import { PANEL_PROFILE } from '../src/config.js'

// THE DESIGN OF RECORD is `design/current.json` — the editor's Download JSON of
// the layout actually being built. `-` (or nothing) means that file, so CI can
// name an output folder without naming a design. The built-in default is only
// a last resort, and the sheet says so in its stamp: it is a different layout,
// and a sheet that silently fell back to it would show the wrong structure.
const RECORD = new URL('../design/current.json', import.meta.url)
const [, , rawDesignArg, outArg] = process.argv
const designPath = rawDesignArg && rawDesignArg !== '-'
  ? path.resolve(rawDesignArg)
  : fs.existsSync(RECORD) ? RECORD.pathname : null
const outDir = path.resolve(outArg ?? 'sheet')
const source = designPath ? JSON.parse(fs.readFileSync(designPath, 'utf8')) : DEFAULT_CONFIG

const config = normalizeConfig(source)
const lattice = solveLattice(config)
const connectors = solveConnectorsV4(config, lattice)
const report = buildReportV4(config, lattice, connectors)
const kit = getConnectorKit(config, lattice, connectors)

const r1 = (v) => Math.round(v * 10) / 10
const fmt = (v, d = 1) => (Math.round(v * 10 ** d) / 10 ** d).toFixed(d)

// =============================================================================
// THE NUMBERS
// =============================================================================
const present = lattice.panels.filter((p) => p.present)
const count = (role) => present.filter((p) => p.role === role).length
const bounds = latticeBounds(lattice.panels)
const bolts = kit.summary.count * CONNECTOR_PROFILE.boltCount

let commit = null
try {
  commit = execSync('git rev-parse --short HEAD', { stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim()
} catch {
  // not a checkout — the sheet still renders, just without the stamp
}

const structure = {
  design: {
    name: config.name ?? null,
    source: designPath
      ? (designPath === RECORD.pathname ? 'design/current.json' : path.basename(designPath))
      : 'built-in default (no design/current.json)',
    pattern: config.pattern.kind,
    cells: `${config.lattice.cols} × ${config.lattice.rows}`,
    commit,
  },
  panels: {
    total: present.length,
    inclined: count('rise') + count('fall'),
    flat: count('ground') + count('high'),
    byRole: { ground: count('ground'), high: count('high'), rise: count('rise'), fall: count('fall') },
    sizeCm: [60, 60],
  },
  angleDeg: config.angleDeg,
  gapCm: config.gap,
  boundsCm: { x: r1(bounds.size[0]), y: r1(bounds.size[1]), z: r1(bounds.size[2]) },
  connectors: {
    stations: kit.summary.count,
    joints: kit.summary.jointCount,
    perJoint: kit.summary.jointCount ? kit.summary.count / kit.summary.jointCount : 0,
    backHalves: kit.kit.map((p) => ({
      id: p.partId,
      count: p.count,
      foldDeg: p.foldDeg,
      sense: p.foldDeg > 0 ? 'convex' : p.foldDeg < 0 ? 'concave' : 'flat',
    })),
    frontBars: kit.bars.map((b) => ({ id: b.barId, count: b.stationIds.length, widthMm: b.widthCm * MM_PER_CM })),
    bolts: { count: bolts, size: CONNECTOR_PROFILE.bolt.name, lengthMm: CONNECTOR_PROFILE.bolt.lengthCm * MM_PER_CM },
    inserts: { count: bolts, size: CONNECTOR_PROFILE.bolt.name },
    lengthMm: (kit.kit[0]?.lengthCm ?? 0) * MM_PER_CM,
  },
  envelope: {
    maxAngleDeg: r1(report.envelope.maxAngleDeg),
    frontBarConcaveLimitDeg: r1(report.envelope.frontBar.concaveLimitDeg),
    frontBarClears: report.envelope.frontBar.clears,
  },
  obstaclesHit: report.obstacles
    .filter((o) => (o.hits?.length ?? 0) > 0)
    .map((o) => ({ id: o.id, panels: o.hits.map((h) => ({ id: h.id, depthCm: r1(h.depthCm) })) })),
}
const hitPanelIds = new Set(structure.obstaclesHit.flatMap((o) => o.panels.map((p) => p.id)))

// =============================================================================
// SHARED DRAWING BITS
// =============================================================================
const hex = (rgb) => `#${rgb.map((c) => Math.max(0, Math.min(255, Math.round(c))).toString(16).padStart(2, '0')).join('')}`
const shade = (base, k) => hex(base.map((c) => c * k))
const norm3 = (v) => {
  const l = Math.hypot(v[0], v[1], v[2]) || 1
  return [v[0] / l, v[1] / l, v[2] / l]
}
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
const pts = (list) => list.map(([x, y]) => `${fmt(x, 2)},${fmt(y, 2)}`).join(' ')
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;')

const mmText = (v) => fmt(v, 1).replace(/\.0$/, '')
/** The real gaps (mm, min and max) at the stations a kit part was binned from. */
function jointSpansMm(part) {
  const ids = new Set(part.stationIds)
  const spans = connectors.stations.filter((st) => ids.has(st.id)).flatMap((st) => [st.spanMinCm * 10, st.spanMaxCm * 10])
  const lo = Math.min(...spans)
  const hi = Math.max(...spans)
  return Math.abs(hi - lo) < 0.05 ? [lo] : [lo, hi]
}

/** Newell normal of a planar polygon — robust for any vertex count. */
function newell(poly) {
  const n = [0, 0, 0]
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i]
    const b = poly[(i + 1) % poly.length]
    n[0] += (a[1] - b[1]) * (a[2] + b[2])
    n[1] += (a[2] - b[2]) * (a[0] + b[0])
    n[2] += (a[0] - b[0]) * (a[1] + b[1])
  }
  return norm3(n)
}

function signedArea2(poly) {
  let a = 0
  for (let i = 0; i < poly.length; i++) {
    const [x0, y0] = poly[i]
    const [x1, y1] = poly[(i + 1) % poly.length]
    a += x0 * y1 - x1 * y0
  }
  return a / 2
}

/**
 * Each SVG carries its own styles so the files stand alone, written against
 * CSS custom properties WITH fallbacks: inlined into the sheet page they pick
 * up its light/dark tokens, opened on their own they use the fallbacks.
 */
const SVG_STYLE = `<style>
.plan-ground{fill:var(--sheet-ground,#f3f1ec)}
.hatch{stroke:var(--sheet-faint,#b9b4aa);stroke-width:1.2}
.ctx-wall{fill:url(#plan-hatch);stroke:var(--sheet-faint,#b9b4aa)}
.ctx-solid{fill:var(--sheet-ctx,#d9d5cc);stroke:var(--sheet-faint,#b9b4aa)}
.ctx-column{fill:var(--sheet-ctx,#d9d5cc);stroke:var(--sheet-warn,#b4562f);stroke-width:1.5;stroke-dasharray:4 3}
.ctx-zone{fill:none;stroke:var(--sheet-faint,#b9b4aa);stroke-dasharray:3 3}
.ctx-glass{fill:var(--sheet-glass,#b9d3dc)}
.ctx-label{font:500 11px var(--sheet-mono,ui-monospace,monospace);fill:var(--sheet-warn,#b4562f)}
.panel{stroke:#5d574d;stroke-width:0.8;stroke-linejoin:round}
.panel-hit{stroke:var(--sheet-warn,#b4562f);stroke-width:2}
.clash{fill:var(--sheet-warn,#b4562f);fill-opacity:.14;stroke:var(--sheet-warn,#b4562f);stroke-width:1.6;stroke-dasharray:5 3}
.spot{font:500 10px var(--sheet-mono,ui-monospace,monospace);fill:#5d574d;text-anchor:middle}
.station{stroke:var(--sheet-accent,#c2531c);stroke-width:3.2;stroke-linecap:butt}
.origin circle{fill:var(--sheet-ink,#24211c)}
.origin text,.dim text{font:500 11px var(--sheet-mono,ui-monospace,monospace);fill:var(--sheet-ink,#24211c)}
.dim line{stroke:var(--sheet-ink,#24211c);stroke-width:1}
.edge-label{font:600 10px var(--sheet-mono,ui-monospace,monospace);letter-spacing:.18em;fill:var(--sheet-muted,#6f695f)}
.exploded polygon,.sections polygon{stroke-width:.7;stroke-linejoin:round}
.axis{stroke:var(--sheet-muted,#6f695f);stroke-width:1;stroke-dasharray:6 3 1.5 3}
.leader{fill:none;stroke:var(--sheet-muted,#6f695f);stroke-width:.9}
.leader-dot{fill:var(--sheet-ink,#24211c)}
.callout-title,.sec-title{font:600 13px var(--sheet-sans,system-ui,sans-serif);fill:var(--sheet-ink,#24211c)}
.callout-sub,.sec-sub{font:400 12px var(--sheet-mono,ui-monospace,monospace);fill:var(--sheet-muted,#6f695f)}
.sec-panel{fill:var(--sheet-ctx,#d9d5cc);stroke:var(--sheet-faint,#9c968b)}
.sec-back{fill:#4680b0;stroke:#284a66}
.sec-bar{fill:#de7840;stroke:#7a3d1c}
.scale line{stroke:var(--sheet-ink,#24211c);stroke-width:2}
.scale text{font:500 15px var(--sheet-mono,ui-monospace,monospace);fill:var(--sheet-ink,#24211c)}
.sections .sec-title{font-size:21px}
.sections .sec-sub{font-size:17px}
</style>`

// =============================================================================
// PLAN VIEW — from above, window at the bottom, wall on the right
// =============================================================================
// The editor's convention: looking down −y with the window (low z) at the
// bottom puts +z up the page and +x to the LEFT, which is why the wall at
// x = 0 lands on the right. A proper rotation, not a mirror — the plan reads
// the same way the room does from above.
function planSVG() {
  const margin = 45
  const x0 = Math.min(-20, bounds.min[0] - margin)
  const x1 = bounds.max[0] + margin
  const z0 = Math.min(-70, bounds.min[2] - margin)
  const z1 = bounds.max[2] + margin
  const k = 1.6 // px per cm
  const pad = { l: 64, r: 70, t: 56, b: 50 }
  const W = (x1 - x0) * k + pad.l + pad.r
  const H = (z1 - z0) * k + pad.t + pad.b
  const sx = (x) => pad.l + (x1 - x) * k
  const sy = (z) => pad.t + (z1 - z) * k

  const out = []
  out.push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${fmt(W, 0)} ${fmt(H, 0)}" class="plan" role="img" aria-label="Plan view of the panel network from above">`, SVG_STYLE)
  out.push(`<defs><clipPath id="plan-clip"><rect x="${pad.l}" y="${pad.t}" width="${fmt((x1 - x0) * k)}" height="${fmt((z1 - z0) * k)}"/></clipPath>`)
  out.push(`<pattern id="plan-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="6" class="hatch"/></pattern></defs>`)
  out.push(`<rect x="${pad.l}" y="${pad.t}" width="${fmt((x1 - x0) * k)}" height="${fmt((z1 - z0) * k)}" class="plan-ground"/>`)

  // --- room context: floor-level things only. The stair passes OVER the
  // network (its landing is ~2m up), so drawing it here would read as a clash.
  out.push('<g clip-path="url(#plan-clip)">')
  const want = (id) =>
    /^(heating|heating-return|column|glass|sill)$/.test(id) || /^mullion-\d$/.test(id)
  for (const o of config.obstacles.filter((o) => want(o.id))) {
    const e = obstacleExtents(o)
    const cls = o.kind === 'zone' ? 'ctx-zone' : o.kind === 'glass' ? 'ctx-glass' : o.id === 'column' ? 'ctx-column' : 'ctx-solid'
    out.push(`<rect x="${fmt(sx(e.max[0]))}" y="${fmt(sy(e.max[2]))}" width="${fmt(e.size[0] * k)}" height="${fmt(e.size[2] * k)}" class="${cls}"/>`)
  }
  // the wall, as a hatched band behind x = 0
  const wallT = config.room?.wallThicknessCm ?? 8.9
  out.push(`<rect x="${fmt(sx(0))}" y="${pad.t}" width="${fmt(wallT * k)}" height="${fmt((z1 - z0) * k)}" class="ctx-wall"/>`)
  out.push('</g>')

  // --- the panels: lit faces, lowest first, flat-shaded by their normal ------
  const light = norm3([0.55, 1.25, 0.7]) // from up-left of the page: +x, up, +z
  const base = [236, 230, 218]
  const panels = [...present].sort((a, b) => a.position[1] - b.position[1])
  for (const p of panels) {
    const face = p.corners.slice(0, 4)
    const n = p.flipped ? p.normal.map((v) => -v) : p.normal
    const k2 = 0.5 + 0.5 * Math.max(0, dot3(n, light))
    const fill = shade(base, k2)
    out.push(`<polygon points="${pts(face.map((c) => [sx(c[0]), sy(c[2])]))}" fill="${fill}" class="${hitPanelIds.has(p.id) ? 'panel panel-hit' : 'panel'}"/>`)
  }
  // obstacles a panel runs into, outlined OVER the panels — drawn underneath
  // they would be hidden by exactly the panels that hit them
  for (const hit of structure.obstaclesHit) {
    const o = config.obstacles.find((x) => x.id === hit.id)
    if (!o) continue
    const e = obstacleExtents(o)
    out.push(`<rect x="${fmt(sx(e.max[0]))}" y="${fmt(sy(e.max[2]))}" width="${fmt(e.size[0] * k)}" height="${fmt(e.size[2] * k)}" class="clash"/>`)
  }
  // spot heights on the raised flat cells — a plan cannot otherwise tell a
  // high cell from a ground one, they face the same way
  for (const p of panels.filter((p) => p.role === 'high')) {
    out.push(`<text x="${fmt(sx(p.position[0]))}" y="${fmt(sy(p.position[2]) + 4)}" class="spot">+${Math.round(p.position[1])}</text>`)
  }

  // --- connectors: one short bar per station, along its joint ---------------
  for (const st of connectors.stations) {
    const h = st.lengthCm / 2
    const a = [st.mid[0] - st.frame.r[0] * h, st.mid[2] - st.frame.r[2] * h]
    const b = [st.mid[0] + st.frame.r[0] * h, st.mid[2] + st.frame.r[2] * h]
    out.push(`<line x1="${fmt(sx(a[0]))}" y1="${fmt(sy(a[1]))}" x2="${fmt(sx(b[0]))}" y2="${fmt(sy(b[1]))}" class="station"/>`)
  }

  // --- origin and dimensions ------------------------------------------------
  out.push(`<g class="origin"><circle cx="${fmt(sx(0))}" cy="${fmt(sy(0))}" r="3.5"/><text x="${fmt(sx(0) - 7)}" y="${fmt(sy(0) - 7)}" text-anchor="end">0,0</text></g>`)
  const dimX = pad.t - 22
  const xa = sx(bounds.max[0])
  const xb = sx(bounds.min[0])
  out.push(`<g class="dim"><line x1="${fmt(xa)}" y1="${dimX}" x2="${fmt(xb)}" y2="${dimX}"/><line x1="${fmt(xa)}" y1="${dimX - 5}" x2="${fmt(xa)}" y2="${dimX + 5}"/><line x1="${fmt(xb)}" y1="${dimX - 5}" x2="${fmt(xb)}" y2="${dimX + 5}"/>`)
  out.push(`<text x="${fmt((xa + xb) / 2)}" y="${dimX - 7}" text-anchor="middle">${fmt(bounds.size[0])} cm</text></g>`)
  const dimZ = W - pad.r + 26
  const za = sy(bounds.max[2])
  const zb = sy(bounds.min[2])
  out.push(`<g class="dim"><line x1="${dimZ}" y1="${fmt(za)}" x2="${dimZ}" y2="${fmt(zb)}"/><line x1="${dimZ - 5}" y1="${fmt(za)}" x2="${dimZ + 5}" y2="${fmt(za)}"/><line x1="${dimZ - 5}" y1="${fmt(zb)}" x2="${dimZ + 5}" y2="${fmt(zb)}"/>`)
  out.push(`<text x="${dimZ + 10}" y="${fmt((za + zb) / 2)}" text-anchor="middle" transform="rotate(90 ${dimZ + 10} ${fmt((za + zb) / 2)})">${fmt(bounds.size[2])} cm</text></g>`)

  // --- edge labels ------------------------------------------------------------
  out.push(`<text x="${fmt(pad.l + (x1 - x0) * k / 2)}" y="${fmt(H - 16)}" class="edge-label" text-anchor="middle">WINDOW</text>`)
  const col = config.obstacles.find((o) => o.id === 'column')
  if (col) {
    const e = obstacleExtents(col)
    out.push(`<text x="${fmt(sx(e.max[0]) - 4)}" y="${fmt(sy(e.centre[2]) + 4)}" class="ctx-label" text-anchor="end">column</text>`)
  }
  out.push('</svg>')
  return out.join('\n')
}

// =============================================================================
// THE CONNECTOR, IN ITS OWN FRAME
// =============================================================================
// (p, q, r): p across the gap (rim A → rim B), q toward the lit face, r along
// the joint. Origin at the station midpoint on the panels' front plane. This
// is the frame `connectorProfile` and `connectorGeometry.js` use.

/** A point on a panel's section, `i` inboard of rim `side`, `d` below the front plane. */
function sectionPoint(side, spanCm, foldDeg, i, d) {
  const phi = (foldDeg * Math.PI) / 180 / 2
  const cs = Math.cos(phi)
  const sn = Math.sin(phi)
  // identical to backHalfProfile's `sides` — the same rim, inward and deeper axes
  const s = side === 0
    ? { rim: [-spanCm / 2, 0], inward: [-cs, -sn], deeper: [sn, -cs] }
    : { rim: [spanCm / 2, 0], inward: [cs, -sn], deeper: [-sn, -cs] }
  return [s.rim[0] + i * s.inward[0] + d * s.deeper[0], s.rim[1] + i * s.inward[1] + d * s.deeper[1]]
}

/** A short stub of one panel's section, cut `stubCm` inboard of its rim. */
function panelStub(side, spanCm, foldDeg, stubCm) {
  const p = PANEL_PROFILE
  if (stubCm <= p.flangeWidth + p.taperWidth) {
    throw new Error(`panelStub: ${stubCm}cm does not reach the back plate — the outline would self-intersect`)
  }
  const ring = [
    [stubCm, p.diffuserDepth],
    [p.bezelWidth, p.diffuserDepth],
    [p.bezelWidth, 0],
    [0, p.bezelDrop],
    [0, p.outerWallDepth],
    [p.flangeWidth, flangeDepthAt(p.flangeWidth)],
    [p.flangeWidth + p.taperWidth, p.overallThickness],
    [stubCm, p.overallThickness],
  ]
  const poly = ring.map(([i, d]) => sectionPoint(side, spanCm, foldDeg, i, d))
  return signedArea2(poly) < 0 ? poly.reverse() : poly
}

/** A profile in (p, q) swept along r from −len/2 to +len/2, offset by `dq` in q. */
function prism(profile, len, dq = 0) {
  const prof = signedArea2(profile) < 0 ? [...profile].reverse() : profile
  const h = len / 2
  const S = prof.map(([p, q]) => [p, q + dq, -h])
  const E = prof.map(([p, q]) => [p, q + dq, h])
  const faces = []
  for (let k = 0; k < prof.length; k++) {
    const k1 = (k + 1) % prof.length
    faces.push({ poly: [S[k], S[k1], E[k1], E[k]], cap: false })
  }
  faces.push({ poly: [...E], cap: true })
  faces.push({ poly: [...S].reverse(), cap: true })
  return faces
}

/** A frustum along q (a bolt head, a shank, an insert) at (p, r), from q0 to q1. */
function frustum(pc, rc, q0, rad0, q1, rad1, seg = 20) {
  const ring = (q, rad) =>
    Array.from({ length: seg }, (_, k) => {
      const t = (k / seg) * Math.PI * 2
      return [pc + rad * Math.cos(t), q, rc + rad * Math.sin(t)]
    })
  const A = ring(q0, rad0)
  const B = ring(q1, rad1)
  const faces = []
  for (let k = 0; k < seg; k++) {
    const k1 = (k + 1) % seg
    faces.push({ poly: [A[k], A[k1], B[k1], B[k]], cap: false })
  }
  faces.push({ poly: [...A], cap: true })
  faces.push({ poly: [...B], cap: true })
  // convex, so orient every face away from the axis midpoint
  const mid = [pc, (q0 + q1) / 2, rc]
  for (const f of faces) {
    const n = newell(f.poly)
    const c = f.poly.reduce((s, v) => [s[0] + v[0] / f.poly.length, s[1] + v[1] / f.poly.length, s[2] + v[2] / f.poly.length], [0, 0, 0])
    if (dot3(n, sub3(c, mid)) < 0) f.poly.reverse()
  }
  return faces
}

/** Camera for the axonometric: yaw about q, pitch down toward it. */
function camera(yawDeg, pitchDeg, scale) {
  const y = (yawDeg * Math.PI) / 180
  const p = (pitchDeg * Math.PI) / 180
  const toward = [Math.sin(y) * Math.cos(p), Math.sin(p), Math.cos(y) * Math.cos(p)] // scene → eye
  const right = norm3(cross3([0, 1, 0], toward))
  const up = cross3(toward, right)
  return {
    toward,
    project: (v) => [dot3(v, right) * scale, -dot3(v, up) * scale],
    depth: (v) => dot3(v, toward),
  }
}

/**
 * Painter's algorithm over a list of solids. Back faces are culled; within a
 * solid, side faces go far-to-near and the near cap last (a ray that reaches a
 * prism's near cap cannot hit one of its own sides first); solids are ordered
 * far-to-near by centroid. Adequate for a handful of separated convex-ish
 * parts, which is exactly what an exploded view is.
 */
function renderSolids(solids, cam, light) {
  const order = solids
    .map((s) => {
      const vs = s.faces.flatMap((f) => f.poly)
      const c = vs.reduce((a, v) => [a[0] + v[0] / vs.length, a[1] + v[1] / vs.length, a[2] + v[2] / vs.length], [0, 0, 0])
      return { s, d: cam.depth(c) }
    })
    .sort((a, b) => a.d - b.d)
  const out = []
  for (const { s } of order) {
    const vis = s.faces
      .map((f) => ({ f, n: newell(f.poly) }))
      .filter(({ n }) => dot3(n, cam.toward) > 1e-6)
      .map(({ f, n }) => ({
        f,
        n,
        d: f.poly.reduce((a, v) => a + cam.depth(v), 0) / f.poly.length,
      }))
      .sort((a, b) => (a.f.cap === b.f.cap ? a.d - b.d : a.f.cap ? 1 : -1))
    out.push(`<g class="${s.cls}">`)
    for (const { f, n } of vis) {
      const k = 0.55 + 0.45 * Math.max(0, dot3(n, light))
      out.push(`<polygon points="${pts(f.poly.map(cam.project))}" fill="${shade(s.base, k)}" stroke="${shade(s.base, 0.42)}"/>`)
    }
    out.push('</g>')
  }
  return out.join('\n')
}

// =============================================================================
// EXPLODED VIEW
// =============================================================================
function explodedSVG() {
  const C = CONNECTOR_PROFILE
  const B = C.bolt
  // Show the CONVEX type when there is one — it is the ridge joint, the one the
  // eye reads as a fold — but either type explodes the same way.
  const part = kit.kit.find((p) => p.foldDeg > 0) ?? kit.kit[0]
  const bar = kit.bars[0]
  const span = (part.spanStartCm + part.spanEndCm) / 2
  const fold = part.foldDeg
  const len = part.lengthCm

  // How far each piece moves out along the bolt axis (q), in cm. The panels
  // are left out here — the assembled sections below show what it clamps.
  const BAR_UP = 1.8
  const BOLT_UP = 3.9
  const INSERT_DOWN = 0.2
  const BACK_DOWN = -2.4

  const barProf = frontBarProfile(bar.widthCm).points
  const backProf = backHalfProfile({ spanCm: span, foldDeg: fold }).points
  const barTop = Math.max(...barProf.map((v) => v[1]))
  const backTop = Math.max(...backProf.map((v) => v[1]))
  const backBot = Math.min(...backProf.map((v) => v[1]))

  // three bolts on evenly spaced axes — POSITIONS NOT MODELLED, see header
  const boltR = [-len / 3, 0, len / 3]
  const solids = [
    { cls: 'bar', base: [222, 120, 64], faces: prism(barProf, len, BAR_UP) },
    { cls: 'back', base: [70, 128, 176], faces: prism(backProf, len, BACK_DOWN) },
  ]
  for (const rc of boltR) {
    const top = barTop + BOLT_UP
    solids.push({ cls: 'bolt', base: [120, 124, 132], faces: frustum(0, rc, top - B.headDepthCm, B.shankCm / 2, top, B.headCm / 2) })
    solids.push({ cls: 'bolt', base: [120, 124, 132], faces: frustum(0, rc, top - B.lengthCm, B.shankCm / 2, top - B.headDepthCm, B.shankCm / 2) })
    const it = backTop + BACK_DOWN + INSERT_DOWN + 1.1
    solids.push({ cls: 'insert', base: [196, 160, 72], faces: frustum(0, rc, it - B.insertLenCm, B.insertOdCm / 2, it, B.insertOdCm / 2) })
  }

  const cam = camera(-38, 24, 34)
  const light = norm3([-0.35, 1, 0.55])
  const body = renderSolids(solids, cam, light)

  // axes through the bolts, top of the bolt to the bottom of the back half
  const axes = boltR.map((rc) => {
    const a = cam.project([0, barTop + BOLT_UP + 0.6, rc])
    const b = cam.project([0, backBot + BACK_DOWN - 0.6, rc])
    return `<line x1="${fmt(a[0])}" y1="${fmt(a[1])}" x2="${fmt(b[0])}" y2="${fmt(b[1])}" class="axis"/>`
  })

  // callouts: an anchor on the part, a label column on the right
  const anchors = [
    { at: [bar.widthCm / 2, barTop + BAR_UP, len / 2], lines: [`Front bar ${bar.barId}`, `${fmt(bar.widthCm * 10, 0)} × ${fmt(len * 10, 0)} × ${fmt(C.frontThicknessCm * 10, 0)} mm · ${bar.stationIds.length} off`] },
    { at: [0, barTop + BOLT_UP, boltR[2]], lines: [`${B.name} countersunk bolt`, `${fmt(B.lengthCm * 10, 0)} mm · ${CONNECTOR_PROFILE.boltCount} per connector`] },
    { at: [B.insertOdCm / 2, backTop + BACK_DOWN + INSERT_DOWN + 1.1 - 0.3, boltR[2]], lines: [`${B.name} heat-set insert`, `${fmt(B.insertOdCm * 10, 0)} mm OD · ${CONNECTOR_PROFILE.boltCount} per connector`] },
    { at: [Math.max(...backProf.map((v) => v[0])), backBot + BACK_DOWN + 0.2, len / 2], lines: [`Back half ${part.partId}`, `${fold > 0 ? 'convex' : 'concave'} ${fold > 0 ? '+' : '−'}${Math.abs(fold)}° · ${part.count} off`] },
  ]

  // fit the viewBox to the drawing
  const all = solids.flatMap((s) => s.faces.flatMap((f) => f.poly.map(cam.project)))
  const minX = Math.min(...all.map((v) => v[0]))
  const maxX = Math.max(...all.map((v) => v[0]))
  const minY = Math.min(...all.map((v) => v[1]))
  const maxY = Math.max(...all.map((v) => v[1]))
  const labelX = maxX + 46
  const vbX = minX - 20
  const vbY = minY - 24
  const vbW = labelX + 210 - vbX
  const vbH = maxY - minY + 48

  const used = []
  const callouts = anchors
    .map((a) => ({ ...a, p: cam.project(a.at) }))
    .sort((a, b) => a.p[1] - b.p[1])
    .map((a) => {
      let y = a.p[1]
      for (const u of used) if (Math.abs(y - u) < 40) y = u + 40
      used.push(y)
      return [
        `<polyline points="${pts([a.p, [labelX - 14, y], [labelX - 4, y]])}" class="leader"/>`,
        `<circle cx="${fmt(a.p[0])}" cy="${fmt(a.p[1])}" r="2.5" class="leader-dot"/>`,
        `<text x="${fmt(labelX)}" y="${fmt(y - 2)}" class="callout-title">${esc(a.lines[0])}</text>`,
        `<text x="${fmt(labelX)}" y="${fmt(y + 14)}" class="callout-sub">${esc(a.lines[1])}</text>`,
      ].join('')
    })

  return [
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${fmt(vbX)} ${fmt(vbY)} ${fmt(vbW)} ${fmt(vbH)}" class="exploded" role="img" aria-label="Exploded view of one connector: front bar, bolts, inserts and back half">`,
    SVG_STYLE,
    ...axes,
    body,
    ...callouts,
    '</svg>',
  ].join('\n')
}

// =============================================================================
// ASSEMBLED SECTIONS — every back-half type, looking along the joint
// =============================================================================
function sectionsSVG() {
  const k = 34 // px per cm
  // The stub must reach past the back plate's inset (flange + taper, 4.62cm) or
  // the outline folds back across itself.
  const stub = PANEL_PROFILE.flangeWidth + PANEL_PROFILE.taperWidth + 1.4
  const bar = kit.bars[0]
  const gutter = 30
  const labelH = 70
  // Build every section first, in (p, q), so the cells can be sized to what is
  // actually drawn — a tilted panel reaches further than its stub length.
  const sections = kit.kit.map((part) => {
    const span = (part.spanStartCm + part.spanEndCm) / 2
    const fold = part.foldDeg
    const back = backHalfProfile({ spanCm: span, foldDeg: fold }).points
    const shapes = [
      { cls: 'sec-panel', poly: panelStub(0, span, fold, stub) },
      { cls: 'sec-panel', poly: panelStub(1, span, fold, stub) },
      { cls: 'sec-back', poly: back },
      { cls: 'sec-bar', poly: frontBarProfile(bar.widthCm).points },
    ]
    const all = shapes.flatMap((sh) => sh.poly)
    const box = {
      pMin: Math.min(...all.map((v) => v[0])),
      pMax: Math.max(...all.map((v) => v[0])),
      qMin: Math.min(...all.map((v) => v[1])) - 0.4,
      qMax: Math.max(...all.map((v) => v[1])) + 0.5,
    }
    return { part, span, fold, back, shapes, box }
  })
  const cellW = Math.max(...sections.map((s) => (s.box.pMax - s.box.pMin) * k)) + gutter
  const drawH = Math.max(...sections.map((s) => (s.box.qMax - s.box.qMin) * k))
  const qTop = Math.max(...sections.map((s) => s.box.qMax))
  const W = cellW * sections.length
  const H = drawH + labelH + 10
  const out = []
  out.push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${fmt(W, 0)} ${fmt(H, 0)}" class="sections" role="img" aria-label="Assembled sections of each back-half type">`, SVG_STYLE)
  sections.forEach((sec, idx) => {
    const ox = idx * cellW + cellW / 2 - ((sec.box.pMax + sec.box.pMin) / 2) * k
    const oy = 6 + qTop * k
    const P = ([p, q]) => [ox + p * k, oy - q * k]
    out.push('<g class="section">')
    for (const sh of sec.shapes) out.push(`<polygon points="${pts(sh.poly.map(P))}" class="${sh.cls}"/>`)
    const top = P([0, 0.9])
    const bot = P([0, Math.min(...sec.back.map((v) => v[1])) - 0.35])
    out.push(`<line x1="${fmt(top[0])}" y1="${fmt(top[1])}" x2="${fmt(bot[0])}" y2="${fmt(bot[1])}" class="axis"/>`)
    const cx = idx * cellW + cellW / 2
    out.push(`<text x="${fmt(cx)}" y="${fmt(drawH + 36)}" class="sec-title" text-anchor="middle">${sec.part.partId} · ${sec.fold > 0 ? 'convex' : 'concave'} ${sec.fold > 0 ? '+' : '−'}${Math.abs(sec.fold)}°</text>`)
    const real = jointSpansMm(sec.part)
    const built = mmText(sec.span * 10)
    const differs = real.some((v) => Math.abs(v - sec.span * 10) > 0.05)
    const sub = differs
      ? `${sec.part.count} off · built for ${built} mm · joints ${real.map(mmText).join('–')} mm`
      : `${sec.part.count} off · ${built} mm gap`
    out.push(`<text x="${fmt(cx)}" y="${fmt(drawH + 60)}" class="sec-sub" text-anchor="middle">${sub}</text>`)
    out.push('</g>')
  })
  // 10 mm scale bar, bottom left
  out.push(`<g class="scale"><line x1="8" y1="${fmt(H - 8)}" x2="${fmt(8 + k)}" y2="${fmt(H - 8)}"/><text x="${fmt(8 + k / 2)}" y="${fmt(H - 16)}" text-anchor="middle">10 mm</text></g>`)
  out.push('</svg>')
  return out.join('\n')
}

// =============================================================================
// WRITE
// =============================================================================
fs.mkdirSync(path.join(outDir, 'pieces'), { recursive: true })
for (const f of fs.readdirSync(path.join(outDir, 'pieces'))) {
  if (f.endsWith('.stl')) fs.unlinkSync(path.join(outDir, 'pieces', f))
}
const pieces = connectorPieces(kit.kit, kit.bars)
structure.connectors.pieces = pieces.map((p) => ({ id: p.partId, kind: p.kind, file: `pieces/${p.filename}`, sizeMm: p.sizeMm, count: p.quantity }))
for (const piece of pieces) {
  const buf = stlPayload(piece.mesh, `grid-designer ${piece.partId} ${piece.kind} (mm)`)
  fs.writeFileSync(path.join(outDir, 'pieces', piece.filename), Buffer.from(buf))
}
fs.writeFileSync(path.join(outDir, 'structure.json'), `${JSON.stringify(structure, null, 2)}\n`)
fs.writeFileSync(path.join(outDir, 'plan.svg'), `${planSVG()}\n`)
fs.writeFileSync(path.join(outDir, 'connector-exploded.svg'), `${explodedSVG()}\n`)
fs.writeFileSync(path.join(outDir, 'connector-sections.svg'), `${sectionsSVG()}\n`)

// =============================================================================
// THE PAGE — the template filled from `structure` and the drawings above
// =============================================================================
/**
 * Two renderings of one template:
 *   'artifact'    the page BODY (no doctype) for a claude.ai artifact, which
 *                 cannot offer downloads, so the files are named, not linked
 *   'standalone'  a full document for GitHub Pages, served next to `pieces/`,
 *                 so every STL is a real download link
 */
const MODEL_URL = 'https://npuckett.github.io/dc-dev/grid-designer/'

function sheetHTML(drawings, mode) {
  const tpl = fs.readFileSync(new URL('./structure-sheet.template.html', import.meta.url), 'utf8')
  const c = structure.connectors
  const mm = (v) => fmt(v, 1).replace(/\.0$/, '')
  const size = (s) => s.map(mm).join(' × ')
  const signed = (d) => `${d > 0 ? '+' : d < 0 ? '−' : ''}${Math.abs(d)}°`
  const pieceOf = (id) => c.pieces.find((p) => p.id === id)
  const highY = present.find((p) => p.role === 'high')?.position[1] ?? 0
  const printed = c.backHalves.reduce((n, b) => n + b.count, 0) + c.frontBars.reduce((n, b) => n + b.count, 0)

  const kitRows = [
    ...c.backHalves.map((b) =>
      `<tr><td><i class="s-swatch" style="background:var(--back)"></i>Back half ${b.id}<br><span style="color:var(--muted)">${b.sense} ${signed(b.foldDeg)}</span></td><td>${size(pieceOf(b.id).sizeMm)}</td><td class="n">${b.count}</td></tr>`),
    ...c.frontBars.map((b) =>
      `<tr><td><i class="s-swatch" style="background:var(--accent)"></i>Front bar ${b.id}</td><td>${size(pieceOf(b.id).sizeMm)}</td><td class="n">${b.count}</td></tr>`),
    `<tr><td>${c.bolts.size} countersunk bolt</td><td>${mm(c.bolts.lengthMm)} long</td><td class="n">${c.bolts.count}</td></tr>`,
    `<tr><td>${c.inserts.size} heat-set insert</td><td>${mm(CONNECTOR_PROFILE.bolt.insertOdCm * MM_PER_CM)} OD</td><td class="n">${c.inserts.count}</td></tr>`,
    `<tr class="s-total"><td>Printed pieces</td><td></td><td class="n">${printed}</td></tr>`,
  ].join('\n              ')

  const standalone = mode === 'standalone'
  const fileRows = c.pieces
    .map((p) => {
      const name = esc(p.file.replace('pieces/', ''))
      const label = standalone ? `<a href="${esc(p.file)}" download>${name}</a>` : name
      return `<li>${label}<span>${p.kind.replace('-', ' ')} · ${size(p.sizeMm)} mm · print ${p.count}</span></li>`
    })
    .join('\n            ')
  const filesNote = standalone
    ? 'Millimetres, laid flat for printing. The editor\'s <b>Connector pieces</b> button exports the same files.'
    : 'In <code>grid-designer/sheet/pieces/</code>, millimetres, laid flat for printing. The editor\'s <b>Connector pieces</b> button exports the same files.'

  const hits = structure.obstaclesHit
  const describe = (h) => {
    const kinds = h.panels.map((p) => {
      const panel = lattice.panels.find((x) => x.id === p.id)
      return `${panel?.kind === 'ramp' ? 'an inclined panel' : 'a flat panel'} by ${mm(p.depthCm)} cm`
    })
    return kinds.length > 1 ? `${kinds.slice(0, -1).join(', ')} and ${kinds.at(-1)}` : kinds[0]
  }
  const clashNote = hits.length
    ? hits.map((h) => `<div class="s-note s-note--warn"><strong>${esc(h.id)} clash</strong>The ${esc(h.id)} overlaps ${h.panels.length} panel${h.panels.length > 1 ? 's' : ''}: ${describe(h)}. The check uses each panel's full bounding box, so it leans toward reporting a clash.</div>`).join('\n')
    : ''
  const clashLegend = hits.length
    ? `<span><i class="s-key s-key--clash"></i>${esc(hits.map((h) => h.id).join(', '))}, overlapping ${hits.reduce((n, h) => n + h.panels.length, 0)} panels</span>`
    : ''

  const concave = c.backHalves.filter((b) => b.foldDeg < 0)
  const worstConcave = Math.max(0, ...concave.map((b) => -b.foldDeg))
  const foulNote = !structure.envelope.frontBarClears && concave.length
    ? `<div class="s-note s-note--warn"><strong>Front bar fouls on concave joints</strong>A flat bar clears the bezels up to ${structure.envelope.frontBarConcaveLimitDeg}° of concave fold. This design folds ${worstConcave}°, so the ${concave.reduce((n, b) => n + b.count, 0)} concave connectors (${concave.map((b) => b.id).join(', ')}) need a different bar section. The overlap shows in the ${concave[0].id} section below.</div>`
    : ''

  // THE BIN. Kit parts are built at their bin's value, not the joints' own, so
  // a coarse bin prints pieces for a gap the design does not have. Said on the
  // sheet rather than left for the printer to discover.
  const binErrMm = kit.summary.worstBinSpanErrorCm * MM_PER_CM
  const gapMm = structure.gapCm * MM_PER_CM
  const builtMm = (kit.kit[0]?.spanStartCm ?? structure.gapCm) * MM_PER_CM
  const idealBarMm = frontBarWidthFor(structure.gapCm) * MM_PER_CM
  const barMm = (kit.bars[0]?.widthCm ?? 0) * MM_PER_CM
  const binNote = binErrMm > 0.05
    ? `<div class="s-note s-note--warn"><strong>Pieces rounded to the connector bin</strong>Every joint here is ${mm(gapMm)} mm, but the kit rounds gaps to ${mm(kit.summary.binSpanCm * MM_PER_CM)} mm steps. The back halves are built for ${mm(builtMm)} mm and the bar is ${mm(barMm)} mm wide instead of ${mm(idealBarMm)}. Set the editor's <b>span bin</b> to ${Math.round(gapMm) % 2 === 0 ? '0.20' : '0.10'} cm or finer to build them at exactly ${mm(gapMm)} mm.</div>`
    : ''

  const shown = c.backHalves.find((b) => b.foldDeg > 0) ?? c.backHalves[0]
  const values = {
    DESIGN_LINE: `Design “${esc(structure.design.name ?? 'unnamed')}”`,
    DATE: new Date().toISOString().slice(0, 10),
    COMMIT: esc(structure.design.commit ?? 'n/a'),
    SOURCE: esc(structure.design.source),
    PATTERN: `${esc(structure.design.pattern)} · ${structure.design.cells} cells`,
    FOOTPRINT: `${fmt(structure.boundsCm.x)} × ${fmt(structure.boundsCm.z)} cm`,
    HEIGHT: `${fmt(structure.boundsCm.y)} cm`,
    GAP: `${mm(structure.gapCm * MM_PER_CM)} mm`,
    PANELS: structure.panels.total,
    INCLINED: structure.panels.inclined,
    FLAT: structure.panels.flat,
    GROUND: structure.panels.byRole.ground,
    HIGH: structure.panels.byRole.high,
    RISE: structure.panels.byRole.rise,
    FALL: structure.panels.byRole.fall,
    HIGH_Y: Math.round(highY),
    ANGLE: structure.angleDeg,
    MAX_ANGLE: structure.envelope.maxAngleDeg,
    STATIONS: c.stations,
    PER_JOINT: c.perJoint,
    JOINTS: c.joints,
    PIECES_TOTAL: printed,
    BOLTS: c.bolts.count,
    BOLT: c.bolts.size,
    KIT_ROWS: kitRows,
    MODEL_HREF: standalone ? '../' : MODEL_URL,
    FILES_NOTE: filesNote,
    FILE_ROWS: fileRows,
    CLASH_NOTE: clashNote,
    CLASH_LEGEND: clashLegend,
    FOUL_NOTE: foulNote,
    BIN_NOTE: binNote,
    SHOWN_PART: `back half ${shown.id} (${shown.sense} ${signed(shown.foldDeg)}) with front bar ${c.frontBars[0]?.id ?? ''}`,
    PLAN_SVG: drawings.plan,
    EXPLODED_SVG: drawings.exploded,
    SECTIONS_SVG: drawings.sections,
  }
  const html = tpl.replace(/\{\{([A-Z_]+)\}\}/g, (m, key) => {
    if (!(key in values)) throw new Error(`structure-sheet template: no value for {{${key}}}`)
    return String(values[key])
  })
  if (!standalone) return html
  // A full document for Pages. The artifact host supplies this skeleton itself;
  // a plain web server does not.
  return [
    '<!doctype html>',
    '<html lang="en">',
    '<head>',
    '<meta charset="utf-8">',
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">',
    '</head>',
    '<body>',
    html,
    '</body>',
    '</html>',
    '',
  ].join('\n')
}

const drawings = {
  plan: fs.readFileSync(path.join(outDir, 'plan.svg'), 'utf8').trim(),
  exploded: fs.readFileSync(path.join(outDir, 'connector-exploded.svg'), 'utf8').trim(),
  sections: fs.readFileSync(path.join(outDir, 'connector-sections.svg'), 'utf8').trim(),
}
fs.writeFileSync(path.join(outDir, 'structure-sheet.html'), sheetHTML(drawings, 'artifact'))
fs.writeFileSync(path.join(outDir, 'index.html'), sheetHTML(drawings, 'standalone'))

console.log(`structure sheet → ${path.relative(process.cwd(), outDir) || '.'}`)
console.log(`  design      ${structure.design.name ?? '(unnamed)'} (${structure.design.source})`)
console.log(`  panels      ${structure.panels.total} — ${structure.panels.inclined} inclined, ${structure.panels.flat} flat`)
console.log(`  angle       ${structure.angleDeg}°`)
console.log(`  connectors  ${structure.connectors.stations} at ${structure.connectors.joints} joints`)
console.log(`  pieces      ${pieces.map((p) => p.filename).join(', ')}`)
