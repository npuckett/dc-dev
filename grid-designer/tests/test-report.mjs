/**
 * tests/test-report.mjs — headless checks for core/report.js (v2).
 *
 * Plain node script, no test framework. Exits non-zero on any failure.
 *   node tests/test-report.mjs
 *
 * These tests encode the EXPECTED PHYSICS of the column-strip model:
 *   - in-column joints (the fold hinges) are exact by construction, and their
 *     signed dihedral reproduces the configured fold,
 *   - in-row joints (the simple side-by-side connections) are exact only while
 *     two adjacent columns share a fold profile, and deviate as the profiles
 *     diverge — that is where all of the slack is deliberately parked,
 *   - a rigid plate adds NO slack of its own at the default geometry, because
 *     cell.rectLength 121 = 2·cell.size + gap = 60 + 1 + 60 exactly: a plate is a
 *     drop-in for the two squares and the joint it replaces.
 * Joint counts are hand-derived and asserted exactly, and every gap expectation is
 * derived from `config.gap` so a change to the default moves them all at once.
 */

import { DEFAULT_CONFIG, normalizeConfig, validateConfig } from '../src/core/schema.js'
import { buildPreset } from '../src/core/presets.js'
import { solveLayout } from '../src/core/placement.js'
import { SKEW_TOLERANCE_DEG, formatReport, jointReport } from '../src/core/report.js'

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

const near = (a, b, eps = 1e-6) => Math.abs(a - b) <= eps
/** The gap every "exact joint" expectation below is measured against (1cm). */
const GAP = DEFAULT_CONFIG.gap
const SIZE = DEFAULT_CONFIG.cell.size
const cfgOf = (over) => normalizeConfig({ ...DEFAULT_CONFIG, ...over })
const columnsOf = (perColumn) => perColumn.map((foldsDeg) => ({ foldsDeg: foldsDeg.slice() }))
const flatColumns = () => columnsOf(Array.from({ length: 6 }, () => [0, 0, 0, 0]))
const run = (cfg) => jointReport(solveLayout(cfg), cfg)
const inRow = (rep) => rep.joints.filter((j) => j.class === 'in-row')
const inCol = (rep) => rep.joints.filter((j) => j.class === 'in-column')
const ids = (list) => list.map((j) => j.id).join(', ')

// =============================================================================
// 1. Flat config — everything exact
// =============================================================================
{
  const cfg = DEFAULT_CONFIG
  const rep = run(cfg)

  // Hand-counted: in-row = rows·(cols−1) = 5·5 = 25 (column-to-column);
  //               in-column = cols·(rows−1) = 6·4 = 24 (the fold hinges).
  check('flat: 25 in-row joints', inRow(rep).length === 25, `got ${inRow(rep).length}`)
  check('flat: 24 in-column joints', inCol(rep).length === 24, `got ${inCol(rep).length}`)
  check('flat: total 49 joints', rep.summary.total === 49, `got ${rep.summary.total}`)
  check('flat: every joint OK', rep.summary.flagged === 0 && rep.summary.ok === 49, JSON.stringify(rep.summary))
  check('flat: ok + flagged = total', rep.summary.ok + rep.summary.flagged === rep.summary.total)

  check('flat: the report expects the configured 1.0cm gap', GAP === 1 && rep.summary.gap === 1, String(rep.summary.gap))
  check(
    'flat: in-row gapMid / gapMin / gapMax all exactly the gap (1.0)',
    inRow(rep).every((j) => near(j.gapMid, GAP) && near(j.gapMin, GAP) && near(j.gapMax, GAP)),
    ids(inRow(rep).filter((j) => !near(j.gapMid, GAP))),
  )
  check(
    'flat: in-column gapMid / gapMin / gapMax all exactly the gap (1.0)',
    inCol(rep).every((j) => near(j.gapMid, GAP) && near(j.gapMin, GAP) && near(j.gapMax, GAP)),
    ids(inCol(rep).filter((j) => !near(j.gapMid, GAP))),
  )
  check('flat: zero skew everywhere', rep.joints.every((j) => near(j.skewDeg, 0)))
  check('flat: dihedral ≈ 0 everywhere', rep.joints.every((j) => near(j.dihedralDeg, 0)))
  check(
    'flat: summary worst values ≈ 0',
    near(rep.summary.worstGapDeviation, 0) && near(rep.summary.worstSkew, 0),
    JSON.stringify(rep.summary),
  )
  check(
    'flat: joint ids and panel refs are populated',
    rep.joints.every((j) => j.id && j.panelA && j.panelB && Array.isArray(j.flags) && j.ok === true),
  )
  check(
    'flat: joint ids follow the v2 naming (in-row:r#:j# / in-column:c#:k#)',
    rep.joints.every((j) =>
      j.class === 'in-row'
        ? j.id === `in-row:r${j.row}:j${j.col}`
        : j.id === `in-column:c${j.col}:k${j.row}`,
    ),
    ids(rep.joints.slice(0, 3)),
  )
  check(
    'flat: no joint spans two cells of one panel',
    rep.joints.every((j) => j.panelA !== j.panelB),
  )
  check('flat: formatReport reports all clear', formatReport(rep).includes('ALL JOINTS WITHIN TOLERANCE'))
}

// =============================================================================
// 2. Every column folded identically ([30,30,30,30]) — a cylinder, so exact
// =============================================================================
{
  const cfg = cfgOf({ columns: columnsOf(Array.from({ length: 6 }, () => [30, 30, 30, 30])) })
  const rep = run(cfg)

  check('cylinder: total 49 joints', rep.summary.total === 49, `got ${rep.summary.total}`)
  check(
    'cylinder: every joint OK (in-row joints exact when profiles match)',
    rep.summary.flagged === 0,
    ids(rep.joints.filter((j) => !j.ok)),
  )
  check(
    'cylinder: all gaps exactly the configured gap',
    rep.joints.every((j) => near(j.gapMid, GAP) && near(j.gapMin, GAP) && near(j.gapMax, GAP)),
    ids(rep.joints.filter((j) => !near(j.gapMid, GAP))),
  )
  check('cylinder: zero skew everywhere', rep.joints.every((j) => near(j.skewDeg, 0, 1e-4)))

  // Sign convention: an in-column joint's dihedral reproduces foldsDeg[k], and a
  // positive fold pitches the next panel up.
  check(
    'cylinder: every in-column dihedral = +30 (positive fold pitches the next panel up)',
    inCol(rep).every((j) => near(j.dihedralDeg, 30, 1e-5)),
    JSON.stringify(inCol(rep).map((j) => j.dihedralDeg)),
  )
  check(
    'cylinder: in-column dihedralDeg === expectedFoldDeg',
    inCol(rep).every((j) => near(j.dihedralDeg, j.expectedFoldDeg, 1e-5)),
  )
  check('cylinder: in-row dihedrals ≈ 0', inRow(rep).every((j) => near(j.dihedralDeg, 0, 1e-5)))
  check(
    'cylinder: in-row joints expect no fold at all (simple connections)',
    inRow(rep).every((j) => j.expectedFoldDeg === 0),
  )

  // Negative folds must read as negative dihedrals.
  const down = run(cfgOf({ columns: columnsOf(Array.from({ length: 6 }, () => [-20, -20, -20, -20])) }))
  check(
    'cylinder: negative folds give negative dihedrals and still exact joints',
    down.summary.flagged === 0 && inCol(down).every((j) => near(j.dihedralDeg, -20, 1e-5)),
    JSON.stringify(down.summary),
  )
}

// =============================================================================
// 3. ONE folded column ([30,-60,60,-30] on column 2) — the divergence case
// =============================================================================
{
  const columns = flatColumns()
  columns[2] = { foldsDeg: [30, -60, 60, -30] }
  const cfg = cfgOf({ columns })
  check('one folded column: config validates', validateConfig(cfg).ok)
  const rep = run(cfg)

  check('one folded column: total 49 joints', rep.summary.total === 49, `got ${rep.summary.total}`)

  // The fold hinges themselves stay exact — that is what the bisector advance buys.
  check(
    'one folded column: ALL in-column joints OK',
    inCol(rep).every((j) => j.ok),
    ids(inCol(rep).filter((j) => !j.ok)),
  )
  check(
    'one folded column: in-column gaps exactly the configured gap (column 2 included)',
    inCol(rep).every((j) => near(j.gapMid, GAP) && near(j.gapMin, GAP) && near(j.gapMax, GAP)),
    ids(inCol(rep).filter((j) => !near(j.gapMid, GAP))),
  )
  check(
    'one folded column: in-column skew 0 (facing edges stay parallel)',
    inCol(rep).every((j) => near(j.skewDeg, 0)),
  )
  check(
    "one folded column: column 2's dihedrals reproduce foldsDeg = +30/−60/+60/−30",
    inCol(rep)
      .filter((j) => j.col === 2)
      .every((j, k) => near(j.dihedralDeg, [30, -60, 60, -30][k], 1e-6)),
    JSON.stringify(inCol(rep).filter((j) => j.col === 2).map((j) => j.dihedralDeg)),
  )
  check(
    'one folded column: in-column dihedralDeg === expectedFoldDeg everywhere',
    inCol(rep).every((j) => near(j.dihedralDeg, j.expectedFoldDeg, 1e-6)),
  )

  // In-row joints: column 2 diverges from BOTH neighbours in every row but the
  // shore, so exactly the 8 joints (rows 1..4) × (boundaries j1 and j2) fail.
  const flaggedRow = inRow(rep).filter((j) => !j.ok)
  check(
    'one folded column: exactly 8 in-row joints flagged — rows 1..4 on both sides of column 2',
    flaggedRow.length === 8 &&
      flaggedRow.every((j) => j.row >= 1 && (j.col === 1 || j.col === 2)),
    ids(flaggedRow),
  )
  check(
    'one folded column: the shore row stays exact (all columns at pitch 0)',
    inRow(rep).filter((j) => j.row === 0).every((j) => j.ok && near(j.gapMid, GAP)),
    ids(inRow(rep).filter((j) => j.row === 0 && !j.ok)),
  )
  check(
    'one folded column: in-row joints away from column 2 stay exact',
    inRow(rep).filter((j) => j.col === 0 || j.col === 3 || j.col === 4).every((j) => j.ok),
    ids(inRow(rep).filter((j) => (j.col === 0 || j.col === 3 || j.col === 4) && !j.ok)),
  )
  check(
    'one folded column: summary.flagged = 8, ok = 41',
    rep.summary.flagged === 8 && rep.summary.ok === 41,
    JSON.stringify(rep.summary),
  )
  check(
    'one folded column: worst skew is the 30° pitch difference',
    near(rep.summary.worstSkew, 30, 1e-5),
    String(rep.summary.worstSkew),
  )
  check(
    'one folded column: rows 1-3 flag SKEW + GAP (pitch differs), row 4 GAP only (pitch back to 0)',
    flaggedRow
      .filter((j) => j.row <= 3)
      .every((j) => j.flags.includes('SKEW') && j.flags.includes('GAP_OUT_OF_TOL')) &&
      flaggedRow
        .filter((j) => j.row === 4)
        .every((j) => j.flags.includes('GAP_OUT_OF_TOL') && !j.flags.includes('SKEW')),
    JSON.stringify(flaggedRow.map((j) => [j.id, j.flags])),
  )
  check(
    "one folded column: an in-row joint's dihedral is the unsigned pitch difference",
    flaggedRow.filter((j) => j.row <= 3).every((j) => near(Math.abs(j.dihedralDeg), 30, 1e-5)),
    JSON.stringify(flaggedRow.map((j) => j.dihedralDeg)),
  )
  check('one folded column: skew tolerance constant is 5°', SKEW_TOLERANCE_DEG === 5)
}

// =============================================================================
// 4. Vertical plate — interior boundary gone, and NO slack anywhere: the 121cm
//    plate is exactly the two squares plus the joint it replaces (2·60 + 1)
// =============================================================================
{
  const columns = flatColumns()
  columns[2] = { foldsDeg: [0, 0, 40, 0] }
  const cfg = cfgOf({ columns, rects: [{ row: 1, col: 2, orientation: 'vertical' }] })
  check('v-plate: config validates', validateConfig(cfg).ok, JSON.stringify(validateConfig(cfg).errors))
  const rep = run(cfg)

  // 25 in-row + (24 − 1 interior to the plate) in-column = 48
  check('v-plate: total 48 joints', rep.summary.total === 48, `got ${rep.summary.total}`)
  check(
    "v-plate: the plate's interior boundary (1,2)-(2,2) is absent",
    !rep.joints.some((j) => j.id === 'in-column:c2:k1'),
  )
  check(
    'v-plate: the plate is one panel on both of its in-column joints',
    rep.joints.filter((j) => j.panelA === 'p1_2' || j.panelB === 'p1_2').length > 0,
  )
  // Its own hinges stay exact, and so do the in-row joints beside BOTH of its
  // cells: the plate's two 60cm cell windows land precisely on the lattice
  // positions the two squares it replaced occupied.
  const behind = rep.joints.find((j) => j.id === 'in-column:c2:k2')
  check(
    'v-plate: the hinge behind the plate still measures exactly the gap at 40°',
    behind && near(behind.gapMid, GAP) && near(behind.dihedralDeg, 40, 1e-5) && behind.ok,
    JSON.stringify(behind),
  )
  check(
    'v-plate: the plate length is exactly 2·size + gap, so it adds no slack',
    cfg.cell.rectLength === 2 * SIZE + GAP,
    `${cfg.cell.rectLength} vs ${2 * SIZE + GAP}`,
  )
  for (const [row, id] of [[1, 'in-row:r1:j1'], [2, 'in-row:r2:j1']]) {
    const beside = rep.joints.find((j) => j.id === id)
    check(
      `v-plate: the in-row joint beside its row-${row} cell measures exactly the gap, zero skew`,
      beside &&
        near(beside.gapMid, GAP) &&
        near(beside.gapMin, GAP) &&
        near(beside.gapMax, GAP) &&
        near(beside.skewDeg, 0, 1e-5) &&
        beside.ok,
      JSON.stringify(beside),
    )
  }
  // The rows BEHIND the plate are still flagged in-row — but now only because
  // column 2 pitches 40° away from its flat neighbours, not because a plate pulled
  // them forward. The near ends of those edges are still gap-apart.
  const behindRow = rep.joints.find((j) => j.id === 'in-row:r3:j1')
  check(
    'v-plate: the row behind it is flagged for the 40° pitch divergence (SKEW + GAP)',
    behindRow &&
      !behindRow.ok &&
      behindRow.flags.includes('SKEW') &&
      behindRow.flags.includes('GAP_OUT_OF_TOL') &&
      near(behindRow.dihedralDeg, 40, 1e-5),
    JSON.stringify(behindRow),
  )
  // …and with column 2 flat, a vertical plate leaves the WHOLE report exact.
  {
    const flatPlate = cfgOf({ columns: flatColumns(), rects: [{ row: 1, col: 2, orientation: 'vertical' }] })
    const flatRep = run(flatPlate)
    check(
      'v-plate on a flat column: every remaining joint is exact — a true drop-in',
      flatRep.summary.flagged === 0 &&
        flatRep.joints.every((j) => near(j.gapMid, GAP) && near(j.skewDeg, 0, 1e-5)),
      ids(flatRep.joints.filter((j) => !j.ok || !near(j.gapMid, GAP))),
    )
  }
}

// =============================================================================
// 5. Horizontal plate spanning two matched columns — the rigid in-row joint
// =============================================================================
{
  const columns = flatColumns()
  // Columns 0-2 share one profile, so the plate meets column 2 cleanly too.
  columns[0] = { foldsDeg: [30, 0, 0, 0] }
  columns[1] = { foldsDeg: [30, 0, 0, 0] }
  columns[2] = { foldsDeg: [30, 0, 0, 0] }
  const cfg = cfgOf({ columns, rects: [{ row: 1, col: 0, orientation: 'horizontal' }] })
  const v = validateConfig(cfg)
  check('h-plate: config validates (both columns at pitch 30 by row 1)', v.ok, JSON.stringify(v.errors))
  check(
    'h-plate: no W_CROSSCOL_POSITION (the two chains are identical in front of it)',
    !v.warnings.some((w) => w.code === 'W_CROSSCOL_POSITION'),
    JSON.stringify(v.warnings.map((w) => w.code)),
  )

  const rep = run(cfg)
  // (25 − 1 interior to the plate) in-row + 24 in-column = 48
  check('h-plate: total 48 joints', rep.summary.total === 48, `got ${rep.summary.total}`)
  check(
    "h-plate: the plate's interior boundary (1,0)-(1,1) is absent",
    !rep.joints.some((j) => j.id === 'in-row:r1:j0'),
  )
  // The 121cm plate fills its 2·60 + gap = 121cm slot exactly, so there is no
  // per-side slack left to measure: its in-column joints are as exact as a square's.
  check(
    'h-plate: its two in-column joints measure exactly the gap at 30° (zero plate slack)',
    ['in-column:c0:k0', 'in-column:c1:k0'].every((id) => {
      const j = rep.joints.find((x) => x.id === id)
      return j && j.ok && near(j.gapMid, GAP) && near(j.dihedralDeg, 30, 1e-5)
    }),
    JSON.stringify(rep.joints.filter((j) => j.id.startsWith('in-column:c0:k0') || j.id.startsWith('in-column:c1:k0'))),
  )
  check(
    "h-plate: the plate's right edge sits exactly the gap from column 2 — no slack",
    (() => {
      const j = rep.joints.find((x) => x.id === 'in-row:r1:j1')
      return j && near(j.gapMid, GAP) && near(j.skewDeg, 0, 1e-5) && j.ok
    })(),
    JSON.stringify(rep.joints.find((x) => x.id === 'in-row:r1:j1')),
  )
}

// =============================================================================
// 6. Wave preset — hand-counted joints, sane summary, eyeball output
// =============================================================================
{
  const cfg = buildPreset('wave')
  const layout = solveLayout(cfg)
  const rep = jointReport(layout, cfg)

  // Hand count for the 'crest plates' pattern (one VERTICAL plate per column, on the
  // crest plateau — rows 1–2 in columns 0–2, rows 2–3 in columns 3–5):
  //   25 in-row   − 0 (no horizontal plate spans an in-row boundary)      = 25
  //   24 in-column − 6 (one joint interior to each column's crest plate)  = 18
  check('wave: 25 in-row joints (no horizontal plate to remove one)', inRow(rep).length === 25, `got ${inRow(rep).length}`)
  check('wave: 18 in-column joints (six interior to the six crest plates)', inCol(rep).length === 18, `got ${inCol(rep).length}`)
  check('wave: total 43 joints', rep.summary.total === 43, `got ${rep.summary.total}`)
  check('wave: ok + flagged = total', rep.summary.ok + rep.summary.flagged === rep.summary.total)
  check(
    "wave: every crest plate's interior boundary is absent from the report",
    cfg.rects.every((r) => !rep.joints.some((j) => j.id === `in-column:c${r.col}:k${r.row}`)),
    ids(rep.joints.filter((j) => cfg.rects.some((r) => j.id === `in-column:c${r.col}:k${r.row}`))),
  )
  check(
    'wave: no in-row boundary is removed (its plates all run along the columns)',
    inRow(rep).length === 25,
    `got ${inRow(rep).length}`,
  )
  check(
    'wave: EVERY in-column joint (fold hinge) is within tolerance',
    inCol(rep).every((j) => j.ok),
    ids(inCol(rep).filter((j) => !j.ok)),
  )
  check(
    'wave: every in-column dihedral reproduces its configured fold',
    inCol(rep).every((j) => near(j.dihedralDeg, j.expectedFoldDeg, 1e-5)),
    JSON.stringify(inCol(rep).filter((j) => !near(j.dihedralDeg, j.expectedFoldDeg, 1e-5)).map((j) => [j.id, j.dihedralDeg, j.expectedFoldDeg])),
  )
  check(
    'wave: the flagged joints are all in-row (the price of a travelling wave)',
    rep.joints.filter((j) => !j.ok).every((j) => j.class === 'in-row') && rep.summary.flagged > 0,
    ids(rep.joints.filter((j) => !j.ok)),
  )
  check(
    'wave: summary numbers are finite and non-negative',
    Number.isFinite(rep.summary.worstGapDeviation) &&
      Number.isFinite(rep.summary.worstSkew) &&
      rep.summary.worstGapDeviation >= 0 &&
      rep.summary.worstSkew >= 0,
    JSON.stringify(rep.summary),
  )
  check(
    'wave: every metric is finite',
    rep.joints.every(
      (j) =>
        Number.isFinite(j.gapMid) &&
        Number.isFinite(j.gapMin) &&
        Number.isFinite(j.gapMax) &&
        Number.isFinite(j.skewDeg) &&
        Number.isFinite(j.dihedralDeg),
    ),
  )
  check(
    'wave: skew stays inside [0, 90]',
    rep.joints.every((j) => j.skewDeg >= 0 && j.skewDeg <= 90 + 1e-9),
  )
  check(
    'wave: flags only ever GAP_OUT_OF_TOL / SKEW',
    rep.joints.every((j) => j.flags.every((f) => f === 'GAP_OUT_OF_TOL' || f === 'SKEW')),
  )
  check('wave: jointReport is deterministic', JSON.stringify(rep) === JSON.stringify(jointReport(solveLayout(cfg), cfg)))

  console.log('')
  console.log('--- wave preset joint report (eyeball) ---')
  console.log(formatReport(rep))
  console.log('')
}

// =============================================================================
// 6b. ROW RESOLUTION — the joint enumeration follows grid.rows, not a literal
//
// Hand counts for a cols × rows grid with no plates:
//   in-row    = rows·(cols−1)      (column-to-column, one per row)
//   in-column = cols·(rows−1)      (the fold hinges)
// =============================================================================
{
  for (const rows of [5, 6, 7, 8]) {
    const cfg = cfgOf({
      grid: { cols: 6, rows },
      columns: columnsOf(Array.from({ length: 6 }, () => new Array(rows - 1).fill(0))),
    })
    check(`rows ${rows}: config validates`, validateConfig(cfg).ok, JSON.stringify(validateConfig(cfg).errors))
    const rep = run(cfg)
    check(
      `rows ${rows}: ${rows * 5} in-row + ${6 * (rows - 1)} in-column = ${rows * 5 + 6 * (rows - 1)} joints`,
      inRow(rep).length === rows * 5 &&
        inCol(rep).length === 6 * (rows - 1) &&
        rep.summary.total === rows * 5 + 6 * (rows - 1),
      `${inRow(rep).length} / ${inCol(rep).length} / ${rep.summary.total}`,
    )
    check(
      `rows ${rows}: a flat grid is exact everywhere`,
      rep.summary.flagged === 0 &&
        rep.joints.every((j) => near(j.gapMid, GAP) && near(j.skewDeg, 0)),
      ids(rep.joints.filter((j) => !j.ok)),
    )
    check(
      `rows ${rows}: the deepest in-column joint is k = ${rows - 2}`,
      Math.max(...inCol(rep).map((j) => j.row)) === rows - 2,
      String(Math.max(...inCol(rep).map((j) => j.row))),
    )
  }
}
{
  // A folded 7-row config: the fold hinges stay exact at any depth, and a column
  // whose neighbours differ flags the in-row joints beside it — nothing about the
  // measurement depends on the grid being 5 deep.
  const columns = columnsOf(Array.from({ length: 6 }, () => new Array(6).fill(0)))
  columns[3] = { foldsDeg: [25, 15, -20, -20, 0, 0] }
  const cfg = cfgOf({ grid: { cols: 6, rows: 7 }, columns })
  check('rows 7 folded: config validates', validateConfig(cfg).ok, JSON.stringify(validateConfig(cfg).errors))
  const rep = run(cfg)
  check('rows 7 folded: total 35 + 36 = 71 joints', rep.summary.total === 71, String(rep.summary.total))
  check(
    'rows 7 folded: every in-column joint is exact and reproduces its fold',
    inCol(rep).every((j) => j.ok && near(j.dihedralDeg, j.expectedFoldDeg, 1e-6)),
    ids(inCol(rep).filter((j) => !j.ok)),
  )
  check(
    'rows 7 folded: the flagged joints are all in-row, beside column 3',
    rep.joints.filter((j) => !j.ok).every((j) => j.class === 'in-row' && (j.col === 2 || j.col === 3)) &&
      rep.summary.flagged > 0,
    ids(rep.joints.filter((j) => !j.ok)),
  )
  check(
    'rows 7 folded: the shore row is still exact (every column at pitch 0 there)',
    inRow(rep).filter((j) => j.row === 0).every((j) => j.ok),
    ids(inRow(rep).filter((j) => j.row === 0 && !j.ok)),
  )
}

// =============================================================================
// 6c. A STORM preset runs through jointReport — pitched fronts and a wall column
//     change the surface, not the measurement.
// =============================================================================
{
  const cfg = buildPreset('wallcrash')
  const layout = solveLayout(cfg)
  const rep = jointReport(layout, cfg)

  // 'wall splash' is six VERTICAL plates on a 6×6 grid, so:
  //   in-row    = 6 rows · 5 boundaries                                = 30
  //   in-column = 6 cols · 5 boundaries − 6 (one interior to each plate) = 24
  check('wallcrash: 30 in-row joints', inRow(rep).length === 30, `got ${inRow(rep).length}`)
  check('wallcrash: 24 in-column joints (six interior to the six plates)', inCol(rep).length === 24, `got ${inCol(rep).length}`)
  check('wallcrash: total 54 joints', rep.summary.total === 54, String(rep.summary.total))
  check(
    "wallcrash: every plate's interior boundary is absent from the report",
    cfg.rects.every((r) => !rep.joints.some((j) => j.id === `in-column:c${r.col}:k${r.row}`)),
    ids(rep.joints.filter((j) => cfg.rects.some((r) => j.id === `in-column:c${r.col}:k${r.row}`))),
  )
  check(
    'wallcrash: EVERY in-column joint (fold hinge) is still exact at the gap',
    inCol(rep).every((j) => j.ok && near(j.gapMid, GAP)),
    ids(inCol(rep).filter((j) => !j.ok)),
  )
  check(
    'wallcrash: every in-column dihedral reproduces its configured fold',
    inCol(rep).every((j) => near(j.dihedralDeg, j.expectedFoldDeg, 1e-5)),
    JSON.stringify(
      inCol(rep)
        .filter((j) => !near(j.dihedralDeg, j.expectedFoldDeg, 1e-5))
        .map((j) => [j.id, j.dihedralDeg, j.expectedFoldDeg]),
    ),
  )
  check(
    'wallcrash: the flagged joints are all in-row — the price of the ramp',
    rep.joints.filter((j) => !j.ok).every((j) => j.class === 'in-row') && rep.summary.flagged > 0,
    ids(rep.joints.filter((j) => !j.ok)),
  )
  check(
    'wallcrash: the SHORE row is flagged too — the fronts are pitched differently',
    inRow(rep).some((j) => j.row === 0 && !j.ok),
    ids(inRow(rep).filter((j) => j.row === 0)),
  )
  check(
    'wallcrash: every metric is finite and skew stays inside [0, 90]',
    rep.joints.every(
      (j) =>
        Number.isFinite(j.gapMid) &&
        Number.isFinite(j.gapMin) &&
        Number.isFinite(j.gapMax) &&
        Number.isFinite(j.skewDeg) &&
        Number.isFinite(j.dihedralDeg) &&
        j.skewDeg >= 0 &&
        j.skewDeg <= 90 + 1e-9,
    ),
  )
  check(
    'wallcrash: jointReport is deterministic',
    JSON.stringify(rep) === JSON.stringify(jointReport(solveLayout(cfg), cfg)),
  )
  check('wallcrash: formatReport returns text', formatReport(rep).length > 100)
}

// =============================================================================
// 7. All presets produce a sane report
// =============================================================================
for (const id of ['flat', 'calm', 'wave', 'crash', 'swell', 'surge', 'wallcrash']) {
  const cfg = buildPreset(id)
  const rep = run(cfg)
  check(
    `${id} preset: report sane (ok + flagged = total, all metrics finite)`,
    rep.summary.ok + rep.summary.flagged === rep.summary.total &&
      rep.joints.every((j) => Number.isFinite(j.gapMid) && Number.isFinite(j.dihedralDeg)),
    JSON.stringify(rep.summary),
  )
  check(
    `${id} preset: all fold hinges within tolerance`,
    inCol(rep).every((j) => j.ok),
    ids(inCol(rep).filter((j) => !j.ok)),
  )
  check(`${id} preset: formatReport returns text`, formatReport(rep).length > 100)
}
for (const seed of [0, 3, 42]) {
  const cfg = buildPreset('random', seed)
  const rep = run(cfg)
  check(
    `random(${seed}) preset: report sane`,
    rep.summary.ok + rep.summary.flagged === rep.summary.total && rep.summary.total > 0,
    JSON.stringify(rep.summary),
  )
}

// =============================================================================
// Summary
// =============================================================================
console.log('')
console.log(`test-report: ${passed} checks passed, ${failures.length} failed`)
if (failures.length > 0) {
  console.error('')
  console.error('Failures:')
  for (const f of failures) console.error(`  - ${f}`)
  process.exit(1)
}
