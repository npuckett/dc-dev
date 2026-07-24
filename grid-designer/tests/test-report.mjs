/**
 * tests/test-report.mjs — headless checks for core/report.js.
 *
 * Plain node script, no test framework. Exits non-zero on any failure.
 *   node tests/test-report.mjs
 *
 * These tests encode the EXPECTED PHYSICS of the placement model:
 *   - in-row joints are exact by construction (bisector gap advance),
 *   - row-pair joints are exact only while adjacent rows share a zig-zag
 *     profile, and deviate as the profiles diverge,
 *   - a rigid vertical rect deviates from its far-side neighbours as soon as the
 *     row boundary it spans folds.
 * Joint counts are hand-derived and asserted exactly.
 */

import { DEFAULT_CONFIG, normalizeConfig, validateConfig } from '../src/core/schema.js'
import { buildPreset, presetCalm } from '../src/core/presets.js'
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
const rowsOf = (zigzags) => zigzags.map((z) => ({ zigzagDeg: z, jointOverridesDeg: {} }))
const run = (cfg) => jointReport(solveLayout(cfg), cfg)
const inRow = (rep) => rep.joints.filter((j) => j.class === 'in-row')
const rowPair = (rep) => rep.joints.filter((j) => j.class === 'row-pair')
const ids = (list) => list.map((j) => j.id).join(', ')

// =============================================================================
// 1. Flat config — everything exact
// =============================================================================
{
  const cfg = DEFAULT_CONFIG
  const rep = run(cfg)

  // Hand-counted: in-row = rows·(cols−1) = 5·5 = 25; row-pair = (rows−1)·cols = 4·6 = 24.
  check('flat: 25 in-row joints', inRow(rep).length === 25, `got ${inRow(rep).length}`)
  check('flat: 24 row-pair joints', rowPair(rep).length === 24, `got ${rowPair(rep).length}`)
  check('flat: total 49 joints', rep.summary.total === 49, `got ${rep.summary.total}`)
  check('flat: every joint OK', rep.summary.flagged === 0 && rep.summary.ok === 49, JSON.stringify(rep.summary))
  check('flat: ok + flagged = total', rep.summary.ok + rep.summary.flagged === rep.summary.total)

  check(
    'flat: in-row gapMid / gapMin / gapMax all exactly 2.0',
    inRow(rep).every((j) => near(j.gapMid, 2) && near(j.gapMin, 2) && near(j.gapMax, 2)),
    ids(inRow(rep).filter((j) => !near(j.gapMid, 2))),
  )
  check(
    'flat: row-pair gapMid / gapMin / gapMax all exactly 2.0',
    rowPair(rep).every((j) => near(j.gapMid, 2) && near(j.gapMin, 2) && near(j.gapMax, 2)),
    ids(rowPair(rep).filter((j) => !near(j.gapMid, 2))),
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
    'flat: no joint spans two cells of one panel',
    rep.joints.every((j) => j.panelA !== j.panelB),
  )
  check('flat: formatReport reports all clear', formatReport(rep).includes('ALL JOINTS WITHIN TOLERANCE'))
}

// =============================================================================
// 2. Zig-zag row 2 (θ = 30) with everything else flat
// =============================================================================
{
  const cfg = cfgOf({ rows: rowsOf([0, 0, 30, 0, 0]) })
  check('zigzag row 2: config validates', validateConfig(cfg).ok)
  const rep = run(cfg)

  check('zigzag row 2: total 49 joints', rep.summary.total === 49, `got ${rep.summary.total}`)

  // In-row joints are exact by construction — including row 2's own, which is
  // the whole point of advancing the chain along each joint's bisector.
  check(
    'zigzag row 2: ALL in-row joints OK',
    inRow(rep).every((j) => j.ok),
    ids(inRow(rep).filter((j) => !j.ok)),
  )
  check(
    'zigzag row 2: in-row gaps exactly 2.0 (row 2 included)',
    inRow(rep).every((j) => near(j.gapMid, 2) && near(j.gapMin, 2) && near(j.gapMax, 2)),
    ids(inRow(rep).filter((j) => !near(j.gapMid, 2))),
  )
  check(
    'zigzag row 2: in-row skew 0 (facing edges stay parallel)',
    inRow(rep).every((j) => near(j.skewDeg, 0)),
  )

  // Sign convention: an in-row joint's dihedral reproduces φ(r,j).
  check(
    "zigzag row 2: row 2's in-row dihedrals = +30/−30/+30/−30/+30 (φ sign convention)",
    inRow(rep)
      .filter((j) => j.row === 2)
      .every((j, i) => near(j.dihedralDeg, [30, -30, 30, -30, 30][i], 1e-6)),
    JSON.stringify(inRow(rep).filter((j) => j.row === 2).map((j) => j.dihedralDeg)),
  )
  check(
    'zigzag row 2: in-row dihedralDeg === expectedFoldDeg everywhere',
    inRow(rep).every((j) => near(j.dihedralDeg, j.expectedFoldDeg, 1e-6)),
  )

  // Row-pair joints: rows 0/1 and 3/4 share flat profiles → exact. Rows 1/2 and
  // 2/3 straddle the zig-zag → all 6 columns of each boundary deviate.
  const flaggedPairs = rowPair(rep).filter((j) => !j.ok)
  check(
    'zigzag row 2: exactly the 12 row-pair joints at boundaries 1/2 and 2/3 are flagged',
    flaggedPairs.length === 12 && flaggedPairs.every((j) => j.row === 1 || j.row === 2),
    ids(flaggedPairs),
  )
  check(
    'zigzag row 2: row-pair boundaries 0/1 and 3/4 stay exact',
    rowPair(rep)
      .filter((j) => j.row === 0 || j.row === 3)
      .every((j) => j.ok && near(j.gapMid, 2)),
    ids(rowPair(rep).filter((j) => (j.row === 0 || j.row === 3) && !j.ok)),
  )
  check(
    'zigzag row 2: summary.flagged = 12, ok = 37',
    rep.summary.flagged === 12 && rep.summary.ok === 37,
    JSON.stringify(rep.summary),
  )
  check(
    'zigzag row 2: worst skew is the 30° tilt itself',
    near(rep.summary.worstSkew, 30, 1e-5),
    String(rep.summary.worstSkew),
  )
  check(
    'zigzag row 2: flat-tilt columns 0/2/4 flagged on gap, tilted 1/3/5 also on skew',
    flaggedPairs
      .filter((j) => j.col % 2 === 1)
      .every((j) => j.flags.includes('SKEW') && j.flags.includes('GAP_OUT_OF_TOL')) &&
      flaggedPairs
        .filter((j) => j.col % 2 === 0)
        .every((j) => j.flags.includes('GAP_OUT_OF_TOL') && !j.flags.includes('SKEW')),
    JSON.stringify(flaggedPairs.map((j) => [j.id, j.flags])),
  )
}

// =============================================================================
// 3. Row folds only ([30, 30, 30, 30]) — identical profiles, so exact
// =============================================================================
{
  const cfg = cfgOf({ rowFoldsDeg: [30, 30, 30, 30] })
  const rep = run(cfg)

  check('row folds: total 49 joints', rep.summary.total === 49, `got ${rep.summary.total}`)
  check(
    'row folds: every joint OK (row-pair joints exact when profiles match)',
    rep.summary.flagged === 0,
    ids(rep.joints.filter((j) => !j.ok)),
  )
  check(
    'row folds: all gaps exactly 2.0',
    rep.joints.every((j) => near(j.gapMid, 2) && near(j.gapMin, 2) && near(j.gapMax, 2)),
    ids(rep.joints.filter((j) => !near(j.gapMid, 2))),
  )
  check('row folds: zero skew everywhere', rep.joints.every((j) => near(j.skewDeg, 0, 1e-4)))

  // Sign convention: a row-pair joint's dihedral reproduces rowFoldsDeg[r], and
  // a positive rowFold pitches the far row up.
  check(
    'row folds: every row-pair dihedral = +30 (positive fold pitches next row up)',
    rowPair(rep).every((j) => near(j.dihedralDeg, 30, 1e-5)),
    JSON.stringify(rowPair(rep).map((j) => j.dihedralDeg)),
  )
  check(
    'row folds: row-pair dihedralDeg === expectedFoldDeg',
    rowPair(rep).every((j) => near(j.dihedralDeg, j.expectedFoldDeg, 1e-5)),
  )
  check('row folds: in-row dihedrals ≈ 0', inRow(rep).every((j) => near(j.dihedralDeg, 0, 1e-5)))

  // Negative folds must read as negative dihedrals.
  const down = run(cfgOf({ rowFoldsDeg: [-20, -20, -20, -20] }))
  check(
    'row folds: negative folds give negative dihedrals and still exact joints',
    down.summary.flagged === 0 && rowPair(down).every((j) => near(j.dihedralDeg, -20, 1e-5)),
    JSON.stringify(down.summary),
  )
}

// =============================================================================
// 4. Wave preset — sane summary, eyeball output
// =============================================================================
{
  const cfg = buildPreset('wave')
  const layout = solveLayout(cfg)
  const rep = jointReport(layout, cfg)

  // Hand count: 25 in-row − 1 (interior to the horizontal rect at (4,2))  = 24
  //             24 row-pair − 1 (interior to the vertical rect at (2,0))  = 23
  check('wave: 24 in-row joints (one removed by the horizontal rect)', inRow(rep).length === 24, `got ${inRow(rep).length}`)
  check('wave: 23 row-pair joints (one interior to the vertical rect)', rowPair(rep).length === 23, `got ${rowPair(rep).length}`)
  check('wave: total 47 joints', rep.summary.total === 47, `got ${rep.summary.total}`)
  check('wave: ok + flagged = total', rep.summary.ok + rep.summary.flagged === rep.summary.total)
  check(
    'wave: summary numbers are finite and non-negative',
    Number.isFinite(rep.summary.worstGapDeviation) &&
      Number.isFinite(rep.summary.worstSkew) &&
      rep.summary.worstGapDeviation >= 0 &&
      rep.summary.worstSkew >= 0,
    JSON.stringify(rep.summary),
  )
  check(
    'wave: the removed joint (4,2)-(4,3) is absent',
    !rep.joints.some((j) => j.id === 'in-row:r4:j2'),
  )
  check(
    "wave: the vertical rect's interior boundary (2,0)-(3,0) is absent",
    !rep.joints.some((j) => j.id === 'row-pair:r2:c0'),
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
  check(
    'wave: only the vertical rect\'s far-side in-row joint breaks in-row exactness',
    inRow(rep)
      .filter((j) => !j.ok)
      .every((j) => j.id === 'in-row:r3:j0'),
    ids(inRow(rep).filter((j) => !j.ok)),
  )
  check('wave: jointReport is deterministic', JSON.stringify(rep) === JSON.stringify(jointReport(solveLayout(cfg), cfg)))

  console.log('')
  console.log('--- wave preset joint report (eyeball) ---')
  console.log(formatReport(rep))
  console.log('')
}

// =============================================================================
// 5. Vertical rect spanning a folding row boundary (calm + rect at (1,0))
// =============================================================================
{
  const cfg = normalizeConfig({
    ...presetCalm(),
    name: 'calm + cross-row plate',
    rects: [{ row: 1, col: 0, orientation: 'vertical' }],
  })
  const v = validateConfig(cfg)
  check('v-rect fold: config validates', v.ok, JSON.stringify(v.errors))
  check(
    'v-rect fold: W_CROSSROW_FOLD raised by WP1 validation (spanned fold = 14°)',
    v.warnings.some((w) => w.code === 'W_CROSSROW_FOLD'),
    JSON.stringify(v.warnings.map((w) => w.code)),
  )

  const rep = run(cfg)
  // 25 in-row + (24 − 1 interior) row-pair = 48
  check('v-rect fold: total 48 joints', rep.summary.total === 48, `got ${rep.summary.total}`)
  check(
    "v-rect fold: the plate's interior boundary (1,0)-(2,0) is absent",
    !rep.joints.some((j) => j.id === 'row-pair:r1:c0'),
  )

  // The plate stays in row 1's plane; row 2's panels pitch 14° away from it.
  const farInRow = rep.joints.find((j) => j.id === 'in-row:r2:j0')
  check('v-rect fold: joint (2,0)-(2,1) exists', !!farInRow)
  check(
    "v-rect fold: the plate's far-side in-row joint to row 2 is flagged",
    farInRow && !farInRow.ok,
    JSON.stringify(farInRow),
  )
  check(
    'v-rect fold: that joint skews by the spanned rowFold (14°)',
    farInRow && near(farInRow.skewDeg, 14, 1e-5) && farInRow.flags.includes('SKEW'),
    farInRow && String(farInRow.skewDeg),
  )
  check(
    'v-rect fold: that joint also breaks the gap tolerance',
    farInRow && farInRow.flags.includes('GAP_OUT_OF_TOL') && farInRow.gapMax > 2 + cfg.gapTolerance,
    farInRow && JSON.stringify([farInRow.gapMid, farInRow.gapMin, farInRow.gapMax]),
  )

  // And across the next row boundary: the plate's far edge vs row 3.
  const farPair = rep.joints.find((j) => j.id === 'row-pair:r2:c0')
  check('v-rect fold: joint (2,0)-(3,0) exists', !!farPair)
  check(
    "v-rect fold: the plate's row-pair joint to row 3 is flagged on gap",
    farPair && !farPair.ok && farPair.flags.includes('GAP_OUT_OF_TOL'),
    JSON.stringify(farPair),
  )
  check(
    'v-rect fold: the plate reports a dihedral of ψ_3 − ψ_1 = 32° (it never bent)',
    farPair && near(farPair.dihedralDeg, 32, 1e-5),
    farPair && String(farPair.dihedralDeg),
  )
  check(
    'v-rect fold: the plate is anchored on its near side (its (1,0) in-row joint is exact)',
    (() => {
      const nearJoint = rep.joints.find((j) => j.id === 'in-row:r1:j0')
      return nearJoint && nearJoint.ok && near(nearJoint.gapMid, 2)
    })(),
    JSON.stringify(rep.joints.find((j) => j.id === 'in-row:r1:j0')),
  )
  check('v-rect fold: skew tolerance constant is 5°', SKEW_TOLERANCE_DEG === 5)

  // Same plate with the spanned boundary flattened: the far side comes back into
  // tolerance, which is the physical claim the W_CROSSROW_FOLD warning encodes.
  {
    const flatSpan = normalizeConfig({
      ...DEFAULT_CONFIG,
      rects: [{ row: 1, col: 0, orientation: 'vertical' }],
    })
    check('v-rect no fold: config validates', validateConfig(flatSpan).ok)
    check(
      'v-rect no fold: no W_CROSSROW_FOLD warning',
      !validateConfig(flatSpan).warnings.some((w) => w.code === 'W_CROSSROW_FOLD'),
    )
    const flatRep = run(flatSpan)
    check(
      'v-rect no fold: every joint back inside tolerance (only 121-vs-122 slack remains)',
      flatRep.summary.flagged === 0,
      ids(flatRep.joints.filter((j) => !j.ok)),
    )
    check(
      'v-rect no fold: worst deviation is the ~1cm rect-length slack',
      flatRep.summary.worstGapDeviation > 0.2 && flatRep.summary.worstGapDeviation < 1.5,
      String(flatRep.summary.worstGapDeviation),
    )
  }
}

// =============================================================================
// 6. All presets produce a sane report
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
