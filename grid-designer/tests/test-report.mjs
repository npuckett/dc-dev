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
 *   - a rigid plate's far side carries the 121-vs-122 hardware slack.
 * Joint counts are hand-derived and asserted exactly.
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

  check(
    'flat: in-row gapMid / gapMin / gapMax all exactly 2.0',
    inRow(rep).every((j) => near(j.gapMid, 2) && near(j.gapMin, 2) && near(j.gapMax, 2)),
    ids(inRow(rep).filter((j) => !near(j.gapMid, 2))),
  )
  check(
    'flat: in-column gapMid / gapMin / gapMax all exactly 2.0',
    inCol(rep).every((j) => near(j.gapMid, 2) && near(j.gapMin, 2) && near(j.gapMax, 2)),
    ids(inCol(rep).filter((j) => !near(j.gapMid, 2))),
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
    'cylinder: all gaps exactly 2.0',
    rep.joints.every((j) => near(j.gapMid, 2) && near(j.gapMin, 2) && near(j.gapMax, 2)),
    ids(rep.joints.filter((j) => !near(j.gapMid, 2))),
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
    'one folded column: in-column gaps exactly 2.0 (column 2 included)',
    inCol(rep).every((j) => near(j.gapMid, 2) && near(j.gapMin, 2) && near(j.gapMax, 2)),
    ids(inCol(rep).filter((j) => !near(j.gapMid, 2))),
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
    inRow(rep).filter((j) => j.row === 0).every((j) => j.ok && near(j.gapMid, 2)),
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
// 4. Vertical plate — interior boundary gone, 1cm slack on the far side
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
  // Its own hinges stay exact; the 1cm of plate slack lands on the in-row joints
  // beside it and on the joint behind it.
  const behind = rep.joints.find((j) => j.id === 'in-column:c2:k2')
  check(
    'v-plate: the hinge behind the plate still measures exactly 2.0 at 40°',
    behind && near(behind.gapMid, 2) && near(behind.dihedralDeg, 40, 1e-5) && behind.ok,
    JSON.stringify(behind),
  )
  const beside = rep.joints.find((j) => j.id === 'in-row:r2:j1')
  check(
    "v-plate: the plate's second cell sits 1cm short in z beside its neighbour → gap hypot(2, 1)",
    beside && near(beside.gapMid, Math.hypot(2, 1), 1e-6) && beside.ok,
    JSON.stringify(beside),
  )
  const behindRow = rep.joints.find((j) => j.id === 'in-row:r3:j1')
  check(
    'v-plate: the rows behind it are pulled 1cm forward, flagged in-row',
    behindRow && !behindRow.ok && behindRow.flags.includes('GAP_OUT_OF_TOL'),
    JSON.stringify(behindRow),
  )
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
  check(
    'h-plate: its two in-column joints carry only the 0.5cm-per-side plate slack',
    ['in-column:c0:k0', 'in-column:c1:k0'].every((id) => {
      const j = rep.joints.find((x) => x.id === id)
      return j && j.ok && near(j.gapMid, Math.hypot(2, 0.5), 1e-6) && near(j.dihedralDeg, 30, 1e-5)
    }),
    JSON.stringify(rep.joints.filter((j) => j.id.startsWith('in-column:c0:k0') || j.id.startsWith('in-column:c1:k0'))),
  )
  check(
    "h-plate: the plate's right edge is 2.5cm from column 2 (0.5cm of slack)",
    (() => {
      const j = rep.joints.find((x) => x.id === 'in-row:r1:j1')
      return j && near(j.gapMid, 2.5, 1e-6) && j.ok
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

  // Hand count: 25 in-row   − 1 (interior to the horizontal plate at (1,0)) = 24
  //             24 in-column − 1 (interior to the vertical plate at (1,2))  = 23
  check('wave: 24 in-row joints (one removed by the horizontal plate)', inRow(rep).length === 24, `got ${inRow(rep).length}`)
  check('wave: 23 in-column joints (one interior to the vertical plate)', inCol(rep).length === 23, `got ${inCol(rep).length}`)
  check('wave: total 47 joints', rep.summary.total === 47, `got ${rep.summary.total}`)
  check('wave: ok + flagged = total', rep.summary.ok + rep.summary.flagged === rep.summary.total)
  check(
    'wave: the plates\' interior boundaries are absent',
    !rep.joints.some((j) => j.id === 'in-row:r1:j0') &&
      !rep.joints.some((j) => j.id === 'in-column:c2:k1'),
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
// 7. All presets produce a sane report
// =============================================================================
for (const id of ['flat', 'calm', 'wave', 'crash']) {
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
