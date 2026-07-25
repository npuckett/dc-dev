/**
 * tests/test-merge.mjs — headless checks for core/merge.js.
 *
 * Plain node script, no test framework. Exits non-zero on any failure.
 *   node tests/test-merge.mjs
 *
 * The model under test: two adjacent 60cm squares can be combined into one rigid
 * 60×121 plate, and because the plate CANNOT BEND the merge carries the geometric
 * consequence with it — a vertical merge flattens the joint it spans, a horizontal
 * merge matches the neighbouring column's profile in front of the plate. The
 * headline invariant is the SWEEP at the bottom: every candidate `mergeCandidates`
 * offers must actually merge, so the plan view can never highlight a promise the
 * model then refuses.
 */

import { mergeCandidates, mergeCells } from '../src/core/merge.js'
import { normalizeConfig, validateConfig } from '../src/core/schema.js'
import { solveLayout } from '../src/core/placement.js'
import { buildPreset } from '../src/core/presets.js'

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

const flat = () => normalizeConfig(buildPreset('flat'))
const swell = () => normalizeConfig(buildPreset('swell'))

const cellKey = (cell) => `${cell.row},${cell.col}`
const rectKey = (r) => `${r.orientation[0]}(${r.row},${r.col})`
const rectKeys = (cfg) => cfg.rects.map(rectKey).sort()

/** Every cell any plate covers, as a Set of "r,c". */
function platedCells(cfg) {
  const set = new Set()
  for (const rect of cfg.rects) {
    if (rect.orientation === 'horizontal') {
      set.add(`${rect.row},${rect.col}`)
      set.add(`${rect.row},${rect.col + 1}`)
    } else {
      set.add(`${rect.row},${rect.col}`)
      set.add(`${rect.row + 1},${rect.col}`)
    }
  }
  return set
}

/** The config minus its rects, for "nothing else moved" diffs. */
const geometryOf = (cfg) =>
  JSON.stringify({
    grid: cfg.grid,
    cell: cfg.cell,
    gap: cfg.gap,
    columns: cfg.columns,
  })

/**
 * Which columns / joints differ between two configs, as flat strings.
 * ['col 2 startPitch 42 → 20', 'col 2 fold 1 18 → 0', …]
 */
function geometryDiff(a, b) {
  const out = []
  a.columns.forEach((col, c) => {
    const other = b.columns[c]
    if (col.startPitchDeg !== other.startPitchDeg) {
      out.push(`col ${c} startPitch ${col.startPitchDeg} → ${other.startPitchDeg}`)
    }
    if (col.endSupport !== other.endSupport) {
      out.push(`col ${c} endSupport ${col.endSupport} → ${other.endSupport}`)
    }
    col.foldsDeg.forEach((v, k) => {
      if (v !== other.foldsDeg[k]) out.push(`col ${c} fold ${k} ${v} → ${other.foldsDeg[k]}`)
    })
  })
  return out
}

// =============================================================================
// 1. mergeCandidates on `flat` — everything is free
// =============================================================================
// The flat reference surface has every joint at 0° and six identical columns, so a
// plate can be dropped anywhere without touching the geometry.
{
  const cfg = flat()
  const busy = platedCells(cfg)
  let freeCount = 0
  let coerceCount = 0
  let plateNeighbourOffered = 0
  let outOfBoundsOffered = 0

  for (let r = 0; r < cfg.grid.rows; r++) {
    for (let c = 0; c < cfg.grid.cols; c++) {
      if (busy.has(`${r},${c}`)) continue
      for (const cand of mergeCandidates(cfg, { row: r, col: c })) {
        if (cand.kind === 'free') freeCount++
        else coerceCount++
        if (busy.has(cellKey(cand))) plateNeighbourOffered++
        if (
          cand.row < 0 ||
          cand.row > cfg.grid.rows - 1 ||
          cand.col < 0 ||
          cand.col > cfg.grid.cols - 1
        ) {
          outOfBoundsOffered++
        }
      }
    }
  }
  check('flat: every merge candidate is FREE', coerceCount === 0 && freeCount > 0, `free=${freeCount} coerce=${coerceCount}`)
  check('flat: no candidate is a cell an existing plate covers', plateNeighbourOffered === 0)
  check('flat: no candidate is outside the grid', outOfBoundsOffered === 0)

  // A free candidate in each orientation, and the anchor is the LOWER row / LEFT col.
  const cands = mergeCandidates(cfg, { row: 0, col: 2 })
  const up = cands.find((k) => k.row === 1 && k.col === 2)
  const right = cands.find((k) => k.row === 0 && k.col === 3)
  const left = cands.find((k) => k.row === 0 && k.col === 1)
  check('flat (0,2): the neighbour behind it is a free VERTICAL candidate', up?.orientation === 'vertical' && up?.kind === 'free', JSON.stringify(up))
  check('flat (0,2): vertical rect is anchored at the lower row', up && up.rect.row === 0 && up.rect.col === 2 && up.rect.orientation === 'vertical', JSON.stringify(up?.rect))
  check('flat (0,2): the neighbour to its right is a free HORIZONTAL candidate', right?.orientation === 'horizontal' && right?.kind === 'free', JSON.stringify(right))
  check('flat (0,2): horizontal rect is anchored at the left column', right && right.rect.row === 0 && right.rect.col === 2, JSON.stringify(right?.rect))
  check('flat (0,2): the neighbour to its LEFT anchors at col 1', left && left.rect.col === 1 && left.rect.orientation === 'horizontal', JSON.stringify(left?.rect))
  check('flat (0,2): a free candidate carries no changes', up.changes.length === 0 && right.changes.length === 0)

  // Corner and edge cells only ever get in-bounds neighbours.
  const corner = mergeCandidates(cfg, { row: 0, col: 0 })
  check(
    'flat (0,0): a corner offers only its two in-bounds neighbours',
    corner.length === 2 && corner.every((k) => k.row >= 0 && k.col >= 0),
    JSON.stringify(corner.map(cellKey)),
  )
  // The opposite corner (4,5) has only two in-bounds neighbours too, and one of
  // them — (4,4) — is the back half of flat's plate at rows 3–4 of column 4.
  const farCorner = mergeCandidates(cfg, { row: cfg.grid.rows - 1, col: cfg.grid.cols - 1 })
  check(
    'flat (last, last): the opposite corner offers its one free neighbour, not the plated one',
    farCorner.length === 1 && cellKey(farCorner[0]) === `${cfg.grid.rows - 2},${cfg.grid.cols - 1}`,
    JSON.stringify(farCorner.map(cellKey)),
  )
  const edge = mergeCandidates(cfg, { row: 0, col: 2 })
  check('flat (0,2): a shore-edge cell offers three neighbours', edge.length === 3, JSON.stringify(edge.map(cellKey)))
}

// =============================================================================
// 2. mergeCandidates on `swell` — everything costs geometry
// =============================================================================
// A designed profile has no flat joints and no two columns at the same pitch, so
// every merge is a COERCE. (This is exactly why plates could not be added before.)
{
  const cfg = swell()
  const busy = platedCells(cfg)
  let free = 0
  let flattenJoint = 0
  let matchColumns = 0
  let mixedChanges = 0
  let plateNeighbourOffered = 0

  for (let r = 0; r < cfg.grid.rows; r++) {
    for (let c = 0; c < cfg.grid.cols; c++) {
      if (busy.has(`${r},${c}`)) continue
      for (const cand of mergeCandidates(cfg, { row: r, col: c })) {
        if (busy.has(cellKey(cand))) plateNeighbourOffered++
        if (cand.kind === 'free') {
          free++
          continue
        }
        if (cand.changes.length !== 1) mixedChanges++
        const kinds = cand.changes.map((ch) => ch.kind)
        if (cand.orientation === 'vertical') {
          if (kinds.includes('flatten-joint')) flattenJoint++
        } else if (kinds.includes('match-columns')) {
          matchColumns++
        }
      }
    }
  }
  check('swell: no merge is free — every joint is folded and no two columns agree', free === 0, `${free} free candidates`)
  check('swell: vertical candidates coerce with flatten-joint', flattenJoint > 0, `${flattenJoint}`)
  check('swell: horizontal candidates coerce with match-columns', matchColumns > 0, `${matchColumns}`)
  check('swell: each coercion is exactly one change', mixedChanges === 0)
  check('swell: no candidate is a cell the spine plates cover', plateNeighbourOffered === 0)

  // The spine plate occupies rows 2–3 of every column, so cell (1,c) can only merge
  // sideways and forward — never with the plated cell behind it.
  const behindPlate = mergeCandidates(cfg, { row: 1, col: 2 })
  check(
    'swell (1,2): the plated cell (2,2) behind it is NOT offered',
    behindPlate.every((k) => !(k.row === 2 && k.col === 2)),
    JSON.stringify(behindPlate.map(cellKey)),
  )
  check(
    'swell (1,2): its three free neighbours (0,2) (1,1) (1,3) are',
    ['0,2', '1,1', '1,3'].every((key) => behindPlate.some((k) => cellKey(k) === key)),
    JSON.stringify(behindPlate.map(cellKey)),
  )

  // A cell INSIDE a plate arms nothing — a plate is exactly two cells.
  check('swell: a cell inside a plate offers no candidates', mergeCandidates(cfg, { row: 2, col: 0 }).length === 0)

  // The description is the tooltip text, so it has to name the consequence.
  const vertical = behindPlate.find((k) => k.row === 0 && k.col === 2)
  check(
    'swell: a vertical candidate names the joint it will flatten',
    /flattened joint 0 of column 2/.test(vertical.description),
    vertical.description,
  )
  const horizontal = behindPlate.find((k) => k.row === 1 && k.col === 3)
  check(
    'swell: a horizontal candidate names the column it will match',
    /matched column 3 to column 2 through row 1/.test(horizontal.description),
    horizontal.description,
  )
  check('swell: descriptions carry the plate length', /121cm plate/.test(vertical.description), vertical.description)
}

// =============================================================================
// 3. mergeCells on `swell` — VERTICAL
// =============================================================================
{
  const before = swell()
  const A = { row: 0, col: 2 }
  const B = { row: 1, col: 2 }
  const foldBefore = before.columns[2].foldsDeg[0]
  const result = mergeCells(before, A, B)
  check('swell vertical merge: ok', result.ok, result.ok ? '' : `${result.code} ${result.message}`)
  if (result.ok) {
    const after = result.config
    check('swell vertical merge: the result validates', validateConfig(after).ok)
    check('swell vertical merge: the rect exists', rectKeys(after).includes('v(0,2)'), JSON.stringify(rectKeys(after)))
    check('swell vertical merge: it is the only new rect', after.rects.length === before.rects.length + 1)
    check('swell vertical merge: the spanned joint is exactly 0', after.columns[2].foldsDeg[0] === 0, `${after.columns[2].foldsDeg[0]}`)
    check('swell vertical merge: that joint really was folded before', foldBefore !== 0, `${foldBefore}`)
    const diff = geometryDiff(before, after)
    check(
      'swell vertical merge: NOTHING else changed in the geometry',
      diff.length === 1 && diff[0] === `col 2 fold 0 ${foldBefore} → 0`,
      JSON.stringify(diff),
    )
    check(
      'swell vertical merge: every other rect survives unchanged',
      before.rects.every((r) => rectKeys(after).includes(rectKey(r))),
    )
    check(
      'swell vertical merge: the change record says what it did',
      result.changes.length === 1 &&
        result.changes[0].kind === 'flatten-joint' &&
        result.changes[0].col === 2 &&
        result.changes[0].joint === 0 &&
        result.changes[0].fromDeg === foldBefore &&
        result.changes[0].toDeg === 0,
      JSON.stringify(result.changes),
    )
    check('swell vertical merge: kind is coerce', result.kind === 'coerce')
  }
}

// =============================================================================
// 4. mergeCells on `swell` — HORIZONTAL
// =============================================================================
// Asserted through the SOLVED chains, not just the config: the point of matching
// the two columns is that the plate's two cells end up in the same plane at the
// same place, which only `solveLayout` can confirm.
{
  const before = swell()
  const A = { row: 1, col: 2 } // clicked first — column 2's profile survives
  const B = { row: 1, col: 3 }
  const result = mergeCells(before, A, B)
  check('swell horizontal merge: ok', result.ok, result.ok ? '' : `${result.code} ${result.message}`)
  if (result.ok) {
    const after = result.config
    check('swell horizontal merge: the result validates', validateConfig(after).ok)
    check('swell horizontal merge: the rect exists', rectKeys(after).includes('h(1,2)'), JSON.stringify(rectKeys(after)))

    const chainsBefore = solveLayout(before).columnChains
    const chainsAfter = solveLayout(after).columnChains
    const dPitchBefore = Math.abs(chainsBefore[2].pitchesDeg[1] - chainsBefore[3].pitchesDeg[1])
    const dPitch = Math.abs(chainsAfter[2].pitchesDeg[1] - chainsAfter[3].pitchesDeg[1])
    check('swell horizontal merge: the two columns disagreed in pitch before', dPitchBefore > 0.1, `${dPitchBefore.toFixed(3)}°`)
    check('swell horizontal merge: they agree in cumulative pitch at row 1 after', dPitch <= 1e-9, `${dPitch}`)

    // Chain POSITION at that row: the second point of each chain's polyline is the
    // start of row 1 (row 0's segment end + the gap step), so comparing the whole
    // polyline up to it covers the plate's footing.
    const pA = chainsAfter[2].points.slice(0, 3)
    const pB = chainsAfter[3].points.slice(0, 3)
    check(
      'swell horizontal merge: the two chains are in the same place through row 1',
      JSON.stringify(pA) === JSON.stringify(pB),
      `${JSON.stringify(pA)} vs ${JSON.stringify(pB)}`,
    )

    const diff = geometryDiff(before, after)
    const expected = new Set(['col 3 startPitch', 'col 3 fold 0'])
    check(
      'swell horizontal merge: only column 3 changed, and only in front of row 1',
      diff.length > 0 && diff.every((d) => [...expected].some((p) => d.startsWith(p))),
      JSON.stringify(diff),
    )
    check(
      'swell horizontal merge: column 3 keeps its own folds from row 1 back',
      before.columns[3].foldsDeg.slice(1).join(',') === after.columns[3].foldsDeg.slice(1).join(','),
      after.columns[3].foldsDeg.join(','),
    )
    check(
      'swell horizontal merge: column 2 (clicked first) is untouched',
      JSON.stringify(before.columns[2]) === JSON.stringify(after.columns[2]),
    )
    check(
      'swell horizontal merge: the change record names source, target and row',
      result.changes.length === 1 &&
        result.changes[0].kind === 'match-columns' &&
        result.changes[0].from === 2 &&
        result.changes[0].to === 3 &&
        result.changes[0].throughRow === 1,
      JSON.stringify(result.changes),
    )

    // Clicking the other cell first coerces the OTHER column — the pair is not
    // symmetric, and that is the whole point of "clicked-first wins".
    const mirrored = mergeCells(before, B, A)
    check('swell horizontal merge: the mirrored click also merges', mirrored.ok, mirrored.ok ? '' : mirrored.message)
    check(
      'swell horizontal merge: clicking (1,3) first coerces column 2 instead',
      mirrored.ok &&
        mirrored.changes[0].from === 3 &&
        mirrored.changes[0].to === 2 &&
        JSON.stringify(mirrored.config.columns[3]) === JSON.stringify(before.columns[3]),
      mirrored.ok ? JSON.stringify(mirrored.changes) : '',
    )
    check(
      'swell horizontal merge: both directions produce the SAME rect',
      mirrored.ok && rectKey(mirrored.rect) === rectKey(result.rect),
      mirrored.ok ? `${rectKey(mirrored.rect)} vs ${rectKey(result.rect)}` : '',
    )
  }
}

// A horizontal merge at row 0 copies the front pitch only (no folds in front of it).
{
  const before = swell()
  const result = mergeCells(before, { row: 0, col: 4 }, { row: 0, col: 5 })
  check('swell row-0 horizontal merge: ok', result.ok, result.ok ? '' : result.message)
  if (result.ok) {
    check(
      'swell row-0 horizontal merge: only the front pitch was copied',
      result.config.columns[5].startPitchDeg === before.columns[4].startPitchDeg &&
        result.config.columns[5].foldsDeg.join(',') === before.columns[5].foldsDeg.join(','),
      JSON.stringify(geometryDiff(before, result.config)),
    )
    check(
      'swell row-0 horizontal merge: its description mentions the front pitch, not joints',
      /front pitch/.test(result.description) && !/joints/.test(result.description),
      result.description,
    )
  }
}

// =============================================================================
// 5. Rejections
// =============================================================================
{
  const cfg = swell()
  const rows = cfg.grid.rows
  const cols = cfg.grid.cols

  const expectReject = (name, a, b, code) => {
    const r = mergeCells(cfg, a, b)
    check(`reject ${name}: not ok`, !r.ok, r.ok ? 'it merged' : '')
    check(`reject ${name}: ${code}`, !r.ok && r.code === code, r.ok ? '' : `${r.code} ${r.message}`)
    check(`reject ${name}: has a message`, !r.ok && typeof r.message === 'string' && r.message.length > 20)
  }

  expectReject('the same cell twice', { row: 1, col: 1 }, { row: 1, col: 1 }, 'E_MERGE_SAME_CELL')
  expectReject('a diagonal pair', { row: 0, col: 0 }, { row: 1, col: 1 }, 'E_MERGE_NOT_ADJACENT')
  expectReject('a non-adjacent pair', { row: 0, col: 0 }, { row: 0, col: 3 }, 'E_MERGE_NOT_ADJACENT')
  expectReject('a cell off the shore edge', { row: 0, col: 0 }, { row: -1, col: 0 }, 'E_MERGE_BOUNDS')
  expectReject('a cell past the last row', { row: rows - 1, col: 0 }, { row: rows, col: 0 }, 'E_MERGE_BOUNDS')
  expectReject('a cell past the last column', { row: 0, col: cols - 1 }, { row: 0, col: cols }, 'E_MERGE_BOUNDS')
  // (1,c) + (2,c): the spine plate already owns rows 2–3 of every column.
  expectReject('a cell an existing plate covers', { row: 1, col: 2 }, { row: 2, col: 2 }, 'E_MERGE_OCCUPIED')
  expectReject('arming from inside a plate', { row: 3, col: 2 }, { row: 4, col: 2 }, 'E_MERGE_OCCUPIED')
  expectReject('junk cells', { row: 1.5, col: 0 }, { row: 2, col: 0 }, 'E_MERGE_SHAPE')
  expectReject('a missing cell', null, { row: 2, col: 0 }, 'E_MERGE_SHAPE')

  check('rejections leave the input config untouched', JSON.stringify(cfg) === JSON.stringify(swell()))
}

// A coercion that would break ANOTHER plate is refused rather than committed, and
// never offered. Built by hand: column 1 has a folded joint 0, column 2 carries a
// vertical plate over rows 0–1 (so ITS joint 0 must stay 0). A horizontal merge at
// row 2 would copy column 1's joints 0..1 into column 2 and refold that plated
// joint — the validator's E_FOLD_ON_REMOVED_JOINT, surfaced as E_MERGE_REJECTED.
{
  const cfg = flat()
  cfg.rects = [{ row: 0, col: 2, orientation: 'vertical' }]
  cfg.columns[1].foldsDeg[0] = 30
  check('hand-built conflict case is itself valid', validateConfig(cfg).ok, JSON.stringify(validateConfig(cfg).errors))

  const refused = mergeCells(cfg, { row: 2, col: 1 }, { row: 2, col: 2 })
  check('a coercion that would refold a plated joint is refused', !refused.ok && refused.code === 'E_MERGE_REJECTED', refused.ok ? 'it merged' : refused.code)
  check(
    'and it explains itself with the validator\'s own code',
    !refused.ok && /E_FOLD_ON_REMOVED_JOINT/.test(refused.message),
    refused.ok ? '' : refused.message,
  )
  check(
    'mergeCandidates does not offer that pair',
    mergeCandidates(cfg, { row: 2, col: 1 }).every((k) => !(k.row === 2 && k.col === 2)),
    JSON.stringify(mergeCandidates(cfg, { row: 2, col: 1 }).map(cellKey)),
  )
  // The other direction is fine: column 2's flat profile copied into column 1
  // breaks nothing, so it IS offered.
  const allowed = mergeCells(cfg, { row: 2, col: 2 }, { row: 2, col: 1 })
  check('the same pair clicked the other way round merges', allowed.ok, allowed.ok ? '' : allowed.message)
  check(
    'and it is offered from (2,2)',
    mergeCandidates(cfg, { row: 2, col: 2 }).some((k) => k.row === 2 && k.col === 1),
    JSON.stringify(mergeCandidates(cfg, { row: 2, col: 2 }).map(cellKey)),
  )
}

// =============================================================================
// 6. Merge → split is NOT a round trip (the documented non-inverse property)
// =============================================================================
{
  const before = swell()
  const foldBefore = before.columns[1].foldsDeg[4]
  const merged = mergeCells(before, { row: 4, col: 1 }, { row: 5, col: 1 })
  check('non-inverse: the vertical merge went through', merged.ok, merged.ok ? '' : merged.message)
  if (merged.ok) {
    const split = structuredClone(merged.config)
    split.rects = split.rects.filter((r) => !(r.orientation === 'vertical' && r.row === 4 && r.col === 1))
    const result = validateConfig(split)
    check('non-inverse: the config still validates after splitting the plate again', result.ok, JSON.stringify(result.errors))
    check(
      'non-inverse: the flattened joint STAYS flat — a split is not an undo',
      split.columns[1].foldsDeg[4] === 0 && foldBefore !== 0,
      `was ${foldBefore}°, is ${split.columns[1].foldsDeg[4]}°`,
    )
    check(
      'non-inverse: everything else came back to where it started',
      geometryDiff(before, split).length === 1 && rectKeys(split).join() === rectKeys(before).join(),
      JSON.stringify(geometryDiff(before, split)),
    )
  }
}

// =============================================================================
// 7. Determinism, and no input mutation
// =============================================================================
{
  const cfg = swell()
  const snapshot = JSON.stringify(cfg)

  const a = mergeCells(cfg, { row: 0, col: 1 }, { row: 1, col: 1 })
  const b = mergeCells(cfg, { row: 0, col: 1 }, { row: 1, col: 1 })
  check('deterministic: the same merge twice gives the same config', JSON.stringify(a.config) === JSON.stringify(b.config))
  check('deterministic: and the same description', a.description === b.description, a.description)
  check('mergeCells does not mutate its input', JSON.stringify(cfg) === snapshot)

  const c1 = mergeCandidates(cfg, { row: 1, col: 1 })
  const c2 = mergeCandidates(cfg, { row: 1, col: 1 })
  check('deterministic: mergeCandidates is stable', JSON.stringify(c1) === JSON.stringify(c2))
  check('mergeCandidates does not mutate its input', JSON.stringify(cfg) === snapshot)
  check(
    'mergeCandidates order is fixed (row−1, row+1, col−1, col+1)',
    JSON.stringify(c1.map(cellKey)) === JSON.stringify(['0,1', '1,0', '1,2']),
    JSON.stringify(c1.map(cellKey)),
  )

  // Junk input is tolerated rather than thrown.
  let threw = false
  try {
    mergeCandidates(undefined, { row: 0, col: 0 })
    mergeCandidates({}, null)
    mergeCells(null, { row: 0, col: 0 }, { row: 0, col: 1 })
    mergeCells({ grid: { cols: 'x', rows: null } }, { row: 0, col: 0 }, { row: 0, col: 1 })
  } catch (e) {
    threw = true
    console.error(e)
  }
  check('merge functions tolerate junk input', !threw)
}

// =============================================================================
// 8. THE SWEEP — highlighting never promises a merge the model refuses
// =============================================================================
for (const [id, build] of [
  ['flat', flat],
  ['swell', swell],
]) {
  const cfg = build()
  let offered = 0
  let broken = 0
  let mismatched = 0
  for (let r = 0; r < cfg.grid.rows; r++) {
    for (let c = 0; c < cfg.grid.cols; c++) {
      const cell = { row: r, col: c }
      for (const cand of mergeCandidates(cfg, cell)) {
        offered++
        const result = mergeCells(cfg, cell, { row: cand.row, col: cand.col })
        if (!result.ok) {
          broken++
          console.error(`  ${id}: (${r},${c}) → (${cand.row},${cand.col}) offered but ${result.code}`)
          continue
        }
        if (
          result.kind !== cand.kind ||
          result.description !== cand.description ||
          rectKey(result.rect) !== rectKey(cand.rect) ||
          !validateConfig(result.config).ok
        ) {
          mismatched++
        }
      }
    }
  }
  check(`sweep ${id}: candidates were offered at all`, offered > 0, `${offered}`)
  check(`sweep ${id}: EVERY offered candidate merges`, broken === 0, `${broken} of ${offered} failed`)
  check(`sweep ${id}: and reports the same kind / rect / description`, mismatched === 0, `${mismatched} mismatched`)
}

// =============================================================================
// Summary
// =============================================================================
console.log('')
console.log(`test-merge: ${passed} checks passed, ${failures.length} failed`)
if (failures.length > 0) {
  console.error('')
  console.error('Failures:')
  for (const f of failures) console.error(`  - ${f}`)
  process.exit(1)
}
