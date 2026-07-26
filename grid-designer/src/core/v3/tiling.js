/**
 * grid-designer v3 — the domino tiling of the material cell grid.
 *
 * HEADLESS ZONE (src/core/): pure functions, importable from plain node.
 *   - explicit `.js` extensions on ALL relative imports
 *   - imports `three`'s math classes only (none needed here — this file is
 *     plain scalar/2-vector arithmetic); never a scene graph, DOM, or store
 *   - pure and deterministic: same config in, byte-identical output
 *
 * =============================================================================
 * WHAT THIS IS — V3_SPEC.md §3
 * =============================================================================
 * Covers the `sheet.cols × sheet.rows` MATERIAL cell grid with 1×1 squares
 * ('2x2', 60×60) and 1×2 plates ('2x4', 60×121 — two cells PLUS the 1cm joint
 * between them, see schema.js's "PLATE-LENGTH EXACTNESS"). This is a domino
 * tiling of the cell grid: every candidate domino (a cell and one orthogonal
 * neighbour) is scored, placed highest-score-first (skipping anything that
 * conflicts with an already-placed piece), and whatever cells are left over
 * become squares. This is where "all choices about square or rectangular
 * should be made by the tiling algorithm" (the brief) actually happens — no
 * randomness, no manual authoring, just a scored, deterministic sweep.
 *
 * `i`/`j` are MATERIAL cell indices, never "row"/"column" — see schema.js's
 * header and V3_SPEC.md §1. A cell (i, j) is a candidate's ANCHOR (its
 * lower-index corner); a domino is `{ i, j, axis }` where `axis: 'u'` means it
 * spans cells (i, j) and (i+1, j), and `axis: 'v'` means it spans (i, j) and
 * (i, j+1).
 *
 * =============================================================================
 * WHICH SURFACE THE SCORING READS
 * =============================================================================
 * Scoring and the fit gate both read the TARGET (src/core/v3/target.js), via an
 * injected `target` argument to `solveTiling` — the same object `placement.js`
 * is about to seat the panels on. That is not an optimisation; it is what stops
 * the tiler and the placer disagreeing about what surface the panels lie on.
 *
 * They used to disagree. This file measured `driftHeight` (the SMOOTH form)
 * through `materialToPlanApprox`, a deliberately-approximate linear material→
 * plan scale, while placement seated the panels on the FACETED target via the
 * real arc-length unroll. The tiler was therefore blind to `angularity` and
 * `facetCells` and reported an identical sagitta for every faceting setting —
 * placing plates where they physically could not fit (measured 5.26cm against a
 * 2cm tolerance) while refusing plates that would have fitted almost perfectly
 * (0.17cm). Both failure modes came from the same root cause.
 *
 * `materialToPlanApprox` is still exported, and is still used by the
 * `ridge-aligned` / `toe-bands` strategies for PROXIMITY weighting — a
 * comparative "how near the crest / an edge is this" question where a linear
 * scale is fine and no fit claim is made. It must NEVER be used to decide
 * whether a plate fits, or to compute an actual panel position.
 *
 * =============================================================================
 * THE THREE STRATEGIES — V3_SPEC.md §3.1
 * =============================================================================
 *   flat-lie (default)  score = -|discrete second difference of H along the
 *                       domino's long axis, sampled at its two cell centres
 *                       and their midpoint|. "A rigid plate needs a flat place
 *                       to lie." Per spec: sample at exactly those three
 *                       points and take the discrete second difference — NOT
 *                       an analytic curvature. See `scoreFlatLie`.
 *   ridge-aligned       score = |alignment of the domino's long axis with the
 *                       ridge tangent| × a proximity weight that falls off
 *                       with distance from the crest line. The "ridge tangent"
 *                       here is the CONCEPTUAL straight diagonal line V3_SPEC
 *                       §2.3 describes (`tc(s) = crestZ + ridgeShear·s`,
 *                       UNCLAMPED — the clamp in form.js is a numerical guard
 *                       on the profile's crest parameter, not a claim about
 *                       where the ridge "really" runs), so its tangent
 *                       direction is a single constant vector for the whole
 *                       sheet. See `scoreRidgeAligned`.
 *   toe-bands           score = proximity to the edge the domino's long axis
 *                       runs PARALLEL to: a 'v'-axis domino (long axis along
 *                       j, i.e. running parallel to the i = 0 wall edge) is
 *                       scored by closeness to i = 0; a 'u'-axis domino
 *                       (parallel to the j = 0 window edge) by closeness to
 *                       j = 0. A domino near an edge it runs PERPENDICULAR to
 *                       gets no bonus from that edge — it isn't banding it.
 *                       See `scoreToeBands`.
 *
 * Ties break on `(i, j, axis)` ascending ('u' before 'v') so the result is
 * stable regardless of sort algorithm or float noise in the scores.
 *
 * `MIN_PLATES = 4` is enforced exactly as v2's `MIN_RECTS`: fewer raises the
 * non-blocking `W_FEW_PLATES`. Before P2b this warning was effectively
 * unreachable through any valid config (see PLATE FIT below for why); it is
 * reachable now, for real, whenever the form is too curved for `plateFitTol-
 * eranceCm` to admit at least 4 plates.
 *
 * =============================================================================
 * PLATE FIT (P2b) — a plate is only placed where a plate actually fits
 * =============================================================================
 * A 60×121 plate is RIGID. Laid over a curved region it cannot follow the
 * surface: it bridges as a flat chord, and the surface bows away from it. The
 * maximum deviation between that chord and the actual surface over the
 * plate's 121cm span — the SAGITTA — is a real, physical, measurable
 * quantity, and it is the thing that decides whether a plate can go there:
 *
 *   A plate is only placed where the target surface is flat enough that a
 *   rigid 121cm plate stays within `tiling.plateFitToleranceCm` of it.
 *   Everywhere else, two squares — which have a joint between them and can
 *   follow the bend.
 *
 * This runs BEFORE the per-strategy scoring below, as a hard filter: every
 * candidate domino has its sagitta computed (`computeSagittaCm`), and any
 * candidate whose sagitta exceeds the tolerance is dropped from contention
 * entirely — it can never become a plate, no matter how well a strategy would
 * otherwise score it. The strategies still choose, highest-score-first, among
 * whichever candidates SURVIVE that filter; they just no longer get to place
 * ones that do not fit. Rejected cells fall through to the "fill remainder
 * with squares" step exactly like cells nobody ever proposed a domino for.
 *
 * This is why plate count is now a CONSEQUENCE of the form rather than an
 * artefact of the algorithm: a shallow drift is flat almost everywhere, so
 * nearly every candidate survives and the sheet tiles almost entirely in
 * plates; a steep one only has flat-enough real estate on the crest plateau
 * and the feathered toes, so plates concentrate there and `MIN_PLATES` can
 * genuinely fail to be reached (`W_FEW_PLATES`, see `buildFewPlatesWarning`).
 * A `plateFitToleranceCm` at the top of its 0.1–20 range relaxes the filter
 * until it stops rejecting anything a legal domino could occupy — that is
 * exactly this package's pre-P2b maximal-packing behaviour, kept reachable on
 * purpose as the permissive end of the knob.
 *
 * See `computeSagittaCm` and `SAGITTA_SAMPLE_COUNT` below for the sampling
 * method and why the midpoint alone is not always the worst point.
 *
 * =============================================================================
 * MANUAL OVERRIDES (P8) — `config.tiling.overrides`, hard commitments
 * =============================================================================
 * The fit gate above went slack for a reason worth restating: a faceted
 * target (form.angularity > 0) is locally PLANAR by construction, so a
 * candidate's sagitta collapses to near zero almost everywhere and nearly
 * every candidate survives the filter — the strategies stop having anything
 * to choose between, and all three collapse onto the same tiling. Manual
 * override is the lever that puts choice back in the user's hands: pin a
 * cell to a square or a plate directly, by hand, from the plan view
 * (src/v3/TilingMap.jsx) — see schema.js's "MANUAL TILE OVERRIDES" for the
 * config shape (`{ i, j, type, axis }`, keyed on the candidate's anchor cell,
 * exactly as this file already identifies one).
 *
 * `solveTiling` applies every override BEFORE the scored sweep runs, as a
 * hard commitment: an override's cells are marked covered first, so the
 * strategy sweep (which already skips any candidate touching a covered cell)
 * never sees them as available, and the final square-fill pass never
 * revisits them either. The partition property — every cell covered exactly
 * once — holds the same way it always has: `normalizeConfig`'s
 * `sanitizeOverrides` has already guaranteed the overrides array itself is
 * in-bounds and cell-disjoint before this file ever sees it, so "place
 * overrides first" cannot double-cover a cell.
 *
 * A MANUALLY-PLACED PLATE IS NEVER SUBJECT TO THE SAGITTA GATE. This is the
 * crux of the package, and it inverts the automatic behaviour on purpose: an
 * auto-candidate whose sagitta exceeds `plateFitToleranceCm` is dropped
 * silently (two squares instead); a manual override with the SAME sagitta is
 * PLACED — the user is overruling the algorithm intentionally — and instead
 * raises the non-blocking `W_PLATE_OVERRIDE_MISFIT` warning, naming the tile
 * and its measured sagitta against the tolerance. v2's `core/merge.js`
 * (commit 565af13, HANDOFF.md §3.6) hit this exact question and recorded the
 * answer: a merge tool that refuses every physically-awkward request the
 * user actually wants is not a usable tool. Sagitta is still measured
 * IDENTICALLY for an override as for an auto-placed plate — against the
 * TARGET, in 3D, via the same `computeSagittaCm` (exported below so callers,
 * including the plan view, can preview the cost of a candidate combination
 * before committing to it).
 *
 * =============================================================================
 * ADJACENCY — computed in MATERIAL space, exactly, never inferred from world
 * positions (V3_SPEC.md §3.2)
 * =============================================================================
 * Two tiles are neighbours iff their material footprints — each tile's own
 * `uv: { u0, v0, uLen, vLen }` rectangle — are separated by EXACTLY `gap` on
 * one axis and overlap with positive length on the other. That overlap IS the
 * shared boundary segment; its length is exact because the lattice is exact
 * (V3_SPEC §1: "Gaps are exactly `gap` here by construction"). This is done
 * with full rectangle geometry, not by walking each tile's constituent CELLS
 * and asking whether a neighbouring cell belongs to a different tile —
 * because a plate's long (121cm) edge can face TWO different neighbouring
 * tiles at once (e.g. two squares on the far side of a plate), and the
 * cell-pair approach would either double-count that edge or merge it into a
 * fictitious single 121cm joint. The rectangle-overlap approach used here
 * (`axisAdjacency`) finds each real, distinct shared segment on its own.
 */

import { buildTarget } from './target.js'
import { normalizeConfig } from './schema.js'

export const MIN_PLATES = 4

/**
 * Code for the non-blocking warning a manually-overridden plate raises when
 * its sagitta exceeds `tiling.plateFitToleranceCm` — see this file's header,
 * "MANUAL OVERRIDES". Placed anyway; this is the reported cost, not a veto.
 */
export const OVERRIDE_MISFIT_CODE = 'W_PLATE_OVERRIDE_MISFIT'

/**
 * Builds the `W_FEW_PLATES` warning, or returns null when there's nothing to
 * warn about. Factored out of `solveTiling` and exported specifically so it
 * is directly unit-testable. Before P2b's sagitta-based plate-fit rejection
 * (see the "PLATE FIT" section in this file's header), highest-score-first
 * domino placement paired every cell it legally could, so this warning was
 * effectively unreachable through any valid public config. It is reachable
 * now: whenever `tiling.plateFitToleranceCm` is tight enough (or the form
 * curved enough) that fewer than `MIN_PLATES` candidates survive the sagitta
 * filter, `solveTiling` calls this with the real `toleranceCm` it used, and
 * the returned message explains *why* — pointing at the knob — rather than
 * reading like a bug. When `toleranceCm` is omitted (e.g. calling this helper
 * directly, as the tests below do) that clause is simply left out.
 */
export function buildFewPlatesWarning(plateCount, strategy, cols, rows, toleranceCm) {
  if (plateCount >= MIN_PLATES) return null
  const toleranceNote =
    typeof toleranceCm === 'number'
      ? ` — the surface is too curved for a rigid 121cm plate to stay within ` +
        `tiling.plateFitToleranceCm (currently ${toleranceCm}cm) of it in enough places; raise ` +
        `tiling.plateFitToleranceCm to allow more (or less flat) plates`
      : ''
  return {
    code: 'W_FEW_PLATES',
    message:
      `tiling placed ${plateCount} plate${plateCount === 1 ? '' : 's'}, fewer than MIN_PLATES ` +
      `(${MIN_PLATES}) — the "${strategy}" strategy found too few candidate spots on this ` +
      `${cols}×${rows} sheet for the panel kit's modularity to read as a system${toleranceNote}`,
    path: 'tiling',
  }
}

// =============================================================================
// materialToPlanApprox — see file header. Exported exactly as named.
// =============================================================================

function materialWidthOf(cfg) {
  const pitch = cfg.cell.size + cfg.gap
  return cfg.sheet.cols * pitch - cfg.gap
}

function materialDepthOf(cfg) {
  const pitch = cfg.cell.size + cfg.gap
  return cfg.sheet.rows * pitch - cfg.gap
}

/** Internal: `cfg` must already be normalized (avoids re-normalizing inside
 *  the scoring hot loop). Use the exported `materialToPlanApprox` from
 *  outside this module. */
function mapMaterialToPlan(cfg, u, v) {
  return {
    x: (u / materialWidthOf(cfg)) * cfg.form.footprint.width,
    z: (v / materialDepthOf(cfg)) * cfg.form.footprint.depth,
  }
}

/**
 * materialToPlanApprox(config, u, v) → { x, z }
 *
 * DELIBERATELY APPROXIMATE — see the file header. Scoring only; never use
 * this to place a panel.
 *
 * @param {object} config raw or normalized config (normalized internally)
 * @param {number} u material u, cm
 * @param {number} v material v, cm
 * @returns {{x:number, z:number}} plan-space point
 */
export function materialToPlanApprox(config, u, v) {
  return mapMaterialToPlan(normalizeConfig(config), u, v)
}

// =============================================================================
// Candidate enumeration and scoring
// =============================================================================

/** The two cell-centre points (material space) a candidate domino spans. */
function candidateCellCenters(cand, pitch, size) {
  const half = size / 2
  const a = [cand.i * pitch + half, cand.j * pitch + half]
  const b =
    cand.axis === 'u' ? [(cand.i + 1) * pitch + half, cand.j * pitch + half] : [cand.i * pitch + half, (cand.j + 1) * pitch + half]
  return [a, b]
}

/**
 * flat-lie: -|discrete second difference of H along the domino's long axis|,
 * sampled at the two cell centres and their midpoint (V3_SPEC §3.1 — an
 * explicit discrete estimate, not an analytic curvature).
 */
function scoreFlatLie(cfg, cand, target) {
  const pitch = cfg.cell.size + cfg.gap
  const [cA, cB] = candidateCellCenters(cand, pitch, cfg.cell.size)
  const mid = [(cA[0] + cB[0]) / 2, (cA[1] + cB[1]) / 2]

  // Read the TARGET, for the same reason computeSagittaCm does — scoring
  // against the smooth form while the panels sit on the faceted target ranks
  // candidates by a surface that is not the one they have to lie on.
  const pA = target.frameAtMaterial(cA[0], cA[1]).point
  const pM = target.frameAtMaterial(mid[0], mid[1]).point
  const pB = target.frameAtMaterial(cB[0], cB[1]).point

  // Spacing between consecutive samples along the actual surface, so the second
  // difference is a curvature and not an index difference. The arc-length unroll
  // makes A->mid and mid->B unequal in general, so use their mean.
  const dA = Math.hypot(pM[0] - pA[0], pM[1] - pA[1], pM[2] - pA[2])
  const dB = Math.hypot(pB[0] - pM[0], pB[1] - pM[1], pB[2] - pM[2])
  const h = (dA + dB) / 2 || 1e-9
  const secondDiff = (pB[1] - 2 * pM[1] + pA[1]) / (h * h)
  return -Math.abs(secondDiff)
}

/**
 * ridge-aligned: |alignment of the domino's long axis with the ridge tangent|
 * × a weight that falls off with distance from the (unclamped) crest line.
 */
function scoreRidgeAligned(cfg, cand) {
  const { form } = cfg
  const pitch = cfg.cell.size + cfg.gap
  const [cA, cB] = candidateCellCenters(cand, pitch, cfg.cell.size)
  const mid = [(cA[0] + cB[0]) / 2, (cA[1] + cB[1]) / 2]
  const pMid = mapMaterialToPlan(cfg, mid[0], mid[1])

  // The ridge line z_ridge(x) = footprint.depth · (crestZ + ridgeShear·x/width)
  // (V3_SPEC §2.3's tc(s), UNCLAMPED: we want the conceptual straight diagonal
  // the brief describes, not form.js's numerically-guarded profile parameter).
  // Its slope is a single constant — the ridge tangent direction for the
  // whole sheet.
  const slope = (form.ridgeShear * form.footprint.depth) / form.footprint.width
  const tLen = Math.hypot(1, slope) || 1
  const tUnit = { x: 1 / tLen, z: slope / tLen }

  const axisUnit = cand.axis === 'u' ? { x: 1, z: 0 } : { x: 0, z: 1 }
  const alignment = Math.abs(axisUnit.x * tUnit.x + axisUnit.z * tUnit.z)

  const zRidge = form.footprint.depth * (form.crestZ + (form.ridgeShear * pMid.x) / form.footprint.width)
  const dist = Math.abs(pMid.z - zRidge)
  const weight = 1 / (1 + (dist / form.footprint.depth) ** 2)

  return alignment * weight
}

/**
 * toe-bands: proximity to the grounded edge the domino's long axis runs
 * PARALLEL to. A 'v'-axis domino (constant i) bands the i = 0 wall edge; a
 * 'u'-axis domino (constant j) bands the j = 0 window edge.
 */
function scoreToeBands(cand) {
  return cand.axis === 'v' ? 1 / (1 + cand.i) : 1 / (1 + cand.j)
}

function scoreCandidate(strategy, cfg, cand, target) {
  if (strategy === 'ridge-aligned') return scoreRidgeAligned(cfg, cand)
  if (strategy === 'toe-bands') return scoreToeBands(cand)
  return scoreFlatLie(cfg, cand, target)
}

// =============================================================================
// PLATE FIT — sagitta computation. See "PLATE FIT" in the file header for the
// physical reasoning; this is the mechanism.
// =============================================================================

/**
 * Samples per candidate when estimating its sagitta (`computeSagittaCm`).
 * Exported so tests can mirror the exact sampling grid when independently
 * recomputing sagitta from `form.js`, rather than trusting the value this
 * file reports on a tile.
 *
 * WHY 11: the target surface's per-axis profile `B(τ; p, a)` (form.js) is a
 * single smooth hump — one interior extremum, no wiggles — so its deviation
 * from a chord over one 121cm plate span is itself close to unimodal. The
 * only thing that pulls the worst point away from the geometric midpoint is
 * `ridgeShear` making a u-axis plate's effective crest parameter `tc(s)` vary
 * across the span (see "THE GRADIENT" in form.js), which shifts, but does not
 * multiply, that single extremum. 11 evenly-spaced samples (9 interior + the
 * 2 chord endpoints) resolve a single smooth bump on a span this short to
 * well under a hundredth of a centimetre of error against a much finer scan —
 * comfortably inside the coarsest `plateFitToleranceCm` step (0.1cm) the
 * schema allows.
 */
export const SAGITTA_SAMPLE_COUNT = 11

/** Guards the `sagittaCm <= tolerance` comparison against float noise. */
const SAGITTA_EPS = 1e-9

/** Material-space footprint `{u0, v0, uLen, vLen}` of a candidate domino if
 *  it became a plate. Shared by `computeSagittaCm` (which needs it to know
 *  the span to sample) and `makePlateTile` (which needs it as the tile's
 *  actual `uv`), so the two can never disagree about where the plate is. */
function plateFootprint(cfg, cand) {
  const pitch = cfg.cell.size + cfg.gap
  const { i, j, axis } = cand
  const u0 = i * pitch
  const v0 = j * pitch
  const uLen = axis === 'u' ? cfg.cell.plateLength : cfg.cell.size
  const vLen = axis === 'v' ? cfg.cell.plateLength : cfg.cell.size
  return { u0, v0, uLen, vLen }
}

/**
 * Sagitta (cm) of a candidate plate against **the surface the panels actually
 * sit on** — see "PLATE FIT" in the file header.
 *
 * MEASURED AGAINST THE TARGET, NOT THE SMOOTH FORM. This used to read
 * `driftHeight` through `materialToPlanApprox`, and that was wrong in both
 * directions once faceting existed: the tiler reported an identical sagitta for
 * every `angularity` / `facetCells` setting, because it could not see faceting
 * at all, while `placement.js` was seating the panels on the faceted target. So
 * the tiler and the placer disagreed about what surface the panels lie on.
 * Measured at amplitude 100: the tiler claimed 1.70cm everywhere, while the true
 * sagitta ranged 0.17cm (broad facets — plates fit far better than the tiler
 * believed, so good plates were being refused) to 5.26cm (2.6x over tolerance —
 * bad plates were being placed). Both failure modes are gone now.
 *
 * Samples `SAGITTA_SAMPLE_COUNT` points evenly along the plate's long MATERIAL
 * axis with the cross-axis held at the plate's own centreline, and returns the
 * greatest perpendicular distance in 3D from the straight chord joining the two
 * end samples. 3D, not (distance, height): a rigid plate is a straight chord in
 * space, and the arc-length unroll means the plan spacing between samples is not
 * uniform, so the flattened 2D version quietly understated the bow.
 */
export function computeSagittaCm(cfg, cand, target) {
  const uv = plateFootprint(cfg, cand)
  const along = cand.axis === 'u' ? { start: uv.u0, len: uv.uLen } : { start: uv.v0, len: uv.vLen }
  const crossPos = cand.axis === 'u' ? uv.v0 + uv.vLen / 2 : uv.u0 + uv.uLen / 2

  const pts = new Array(SAGITTA_SAMPLE_COUNT)
  for (let k = 0; k < SAGITTA_SAMPLE_COUNT; k++) {
    const t = k / (SAGITTA_SAMPLE_COUNT - 1)
    const pos = along.start + t * along.len
    const u = cand.axis === 'u' ? pos : crossPos
    const v = cand.axis === 'u' ? crossPos : pos
    pts[k] = target.frameAtMaterial(u, v).point
  }

  const a = pts[0]
  const b = pts[pts.length - 1]
  const d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]]
  const dLen = Math.hypot(d[0], d[1], d[2]) || 1e-9

  let maxSagitta = 0
  for (let k = 1; k < pts.length - 1; k++) {
    const p = pts[k]
    const w = [p[0] - a[0], p[1] - a[1], p[2] - a[2]]
    // |w x d| / |d| — perpendicular distance from the point to the chord line.
    const cx = w[1] * d[2] - w[2] * d[1]
    const cy = w[2] * d[0] - w[0] * d[2]
    const cz = w[0] * d[1] - w[1] * d[0]
    maxSagitta = Math.max(maxSagitta, Math.hypot(cx, cy, cz) / dLen)
  }
  return maxSagitta
}

// =============================================================================
// Tile construction
// =============================================================================

function makeSquareTile(cfg, i, j) {
  const pitch = cfg.cell.size + cfg.gap
  return {
    id: `sq_${i}_${j}`,
    cells: [[i, j]],
    type: '2x2',
    // `axis` is meaningless for a 1×1 square (both its sides are `cell.size`);
    // kept at 'u' by convention so every tile carries the field the spec
    // shape declares.
    axis: 'u',
    uv: { u0: i * pitch, v0: j * pitch, uLen: cfg.cell.size, vLen: cfg.cell.size },
    // A square has no long rigid axis to bow away from the surface — its
    // joint with its neighbours is exactly where the surface can bend. Fit
    // quality only means something for a rigid plate; report 0 here so every
    // tile carries the field uniformly (P2b, see "PLATE FIT").
    sagittaCm: 0,
    // Overwritten to `true` at the override call site (P8) — see "MANUAL
    // OVERRIDES". `false` here means "the algorithm chose this", the default
    // for every tile the scored sweep or the remainder fill produces.
    pinned: false,
  }
}

function makePlateTile(cfg, cand) {
  const { i, j, axis } = cand
  const uv = plateFootprint(cfg, cand)
  const cells = axis === 'u' ? [[i, j], [i + 1, j]] : [[i, j], [i, j + 1]]
  return {
    id: `pl_${i}_${j}_${axis}`,
    cells,
    type: '2x4',
    axis,
    uv,
    // Already computed during candidate scoring/filtering — see solveTiling.
    sagittaCm: cand.sagittaCm,
    // See makeSquareTile's identical field: `false` unless overwritten at the
    // override call site.
    pinned: false,
  }
}

// =============================================================================
// Adjacency — see file header for why this is rectangle-overlap, not
// cell-pair-based.
// =============================================================================

const ADJ_EPS = 1e-6

/**
 * Checks whether tiles A and B are separated by exactly `gap` along
 * `sepAxis` ('u' or 'v'), and if so, whether their footprints overlap with
 * positive length on the OTHER axis (the axis the shared boundary segment
 * runs along — reported as the adjacency record's `axis`). Returns an
 * adjacency record or null. Symmetric in A/B.
 */
function axisAdjacency(A, B, sepAxis, gap) {
  const runAxis = sepAxis === 'u' ? 'v' : 'u'
  const key0 = sepAxis === 'u' ? 'u0' : 'v0'
  const keyLen = sepAxis === 'u' ? 'uLen' : 'vLen'

  const a0 = A.uv[key0]
  const aEnd = a0 + A.uv[keyLen]
  const b0 = B.uv[key0]
  const bEnd = b0 + B.uv[keyLen]

  let aBoundary
  let bBoundary
  if (Math.abs(aEnd + gap - b0) < ADJ_EPS) {
    aBoundary = aEnd
    bBoundary = b0
  } else if (Math.abs(bEnd + gap - a0) < ADJ_EPS) {
    aBoundary = a0
    bBoundary = bEnd
  } else {
    return null
  }

  const rKey0 = runAxis === 'u' ? 'u0' : 'v0'
  const rKeyLen = runAxis === 'u' ? 'uLen' : 'vLen'
  const from = Math.max(A.uv[rKey0], B.uv[rKey0])
  const to = Math.min(A.uv[rKey0] + A.uv[rKeyLen], B.uv[rKey0] + B.uv[rKeyLen])
  const materialLength = to - from
  if (materialLength <= ADJ_EPS) return null

  return {
    a: A.id,
    b: B.id,
    axis: runAxis,
    materialLength,
    edge: { a: aBoundary, b: bBoundary, from, to },
  }
}

function computeAdjacency(tiles, gap) {
  const adjacency = []
  for (let a = 0; a < tiles.length; a++) {
    for (let b = a + 1; b < tiles.length; b++) {
      const A = tiles[a]
      const B = tiles[b]
      const uEdge = axisAdjacency(A, B, 'u', gap)
      if (uEdge) adjacency.push(uEdge)
      const vEdge = axisAdjacency(A, B, 'v', gap)
      if (vEdge) adjacency.push(vEdge)
    }
  }
  return adjacency
}

// =============================================================================
// solveTiling — V3_SPEC.md §3.2
// =============================================================================

/**
 * Deterministic, form-driven domino tiling of `config.sheet.cols ×
 * config.sheet.rows`. Safe to call on raw (un-normalized) input.
 *
 * Each tile carries `pinned: boolean` — `true` for a tile placed from
 * `config.tiling.overrides` (P8, "MANUAL OVERRIDES"), `false` for one the
 * scored sweep or the remainder fill chose. `warnings` may include
 * `W_PLATE_OVERRIDE_MISFIT` (see `OVERRIDE_MISFIT_CODE`) for any override
 * whose sagitta exceeds `tiling.plateFitToleranceCm` — the plate is placed
 * regardless; this only reports the cost.
 *
 * @param {object} config raw or normalized v3 config
 * @returns {{ tiles: Array, adjacency: Array, pattern: string, warnings: Array }}
 */
export function solveTiling(config, injectedTarget = null) {
  const cfg = normalizeConfig(config)
  const { cols, rows } = cfg.sheet
  const strategy = cfg.tiling.strategy
  const tolerance = cfg.tiling.plateFitToleranceCm

  // The tiler and the placer MUST measure the same surface, or the tiler decides
  // square-vs-plate against a shape the panels never lie on. `placement.js`
  // injects the target it is about to place on; a standalone caller gets one
  // built here. Injection is not an optimisation detail — it is what guarantees
  // the two agree, and it also avoids solving the arc-length unroll twice.
  const target = injectedTarget ?? buildTarget(cfg)

  const covered = new Set()
  const cellKey = (i, j) => `${i},${j}`
  const tiles = []
  let plateCount = 0
  const overrideWarnings = []

  // --- 0. MANUAL OVERRIDES — hard commitments, placed BEFORE the scored
  //        sweep. See the file header's "MANUAL OVERRIDES" section: an
  //        override always wins over the algorithm, and — critically — a
  //        manually-placed plate is placed and reported even when its
  //        sagitta exceeds `plateFitToleranceCm`; it is never subject to the
  //        PLATE FIT gate below, which only ever filters the algorithm's own
  //        candidates. `cfg.tiling.overrides` has already been sanitized by
  //        `normalizeConfig` (schema.js's `sanitizeOverrides`) into an
  //        in-bounds, cell-disjoint array, so the bounds/overlap guards here
  //        are defensive, not load-bearing.
  for (const ov of cfg.tiling.overrides) {
    if (ov.type === '2x4') {
      const cand = { i: ov.i, j: ov.j, axis: ov.axis }
      const cells = cand.axis === 'u' ? [[cand.i, cand.j], [cand.i + 1, cand.j]] : [[cand.i, cand.j], [cand.i, cand.j + 1]]
      if (cells.some(([i, j]) => i < 0 || i >= cols || j < 0 || j >= rows)) continue
      if (cells.some(([i, j]) => covered.has(cellKey(i, j)))) continue
      cand.sagittaCm = computeSagittaCm(cfg, cand, target)
      for (const [i, j] of cells) covered.add(cellKey(i, j))
      const plateTile = makePlateTile(cfg, cand)
      plateTile.pinned = true
      tiles.push(plateTile)
      plateCount++
      if (cand.sagittaCm > tolerance + SAGITTA_EPS) {
        const tileId = `pl_${cand.i}_${cand.j}_${cand.axis}`
        const overCm = cand.sagittaCm - tolerance
        overrideWarnings.push({
          code: OVERRIDE_MISFIT_CODE,
          tile: tileId,
          sagittaCm: cand.sagittaCm,
          toleranceCm: tolerance,
          message:
            `manually-placed plate ${tileId} bows ${cand.sagittaCm.toFixed(2)}cm away from the target ` +
            `over its 121cm span — ${overCm.toFixed(2)}cm past tiling.plateFitToleranceCm ` +
            `(${tolerance}cm). Placed anyway: this was a manual override, not the algorithm's choice, ` +
            `so it is reported rather than refused.`,
        })
      }
    } else {
      if (ov.i < 0 || ov.i >= cols || ov.j < 0 || ov.j >= rows) continue
      if (covered.has(cellKey(ov.i, ov.j))) continue
      covered.add(cellKey(ov.i, ov.j))
      const squareTile = makeSquareTile(cfg, ov.i, ov.j)
      squareTile.pinned = true
      tiles.push(squareTile)
    }
  }

  // --- 1. Enumerate every candidate domino, score it, and measure its fit ---
  const candidates = []
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) {
      if (i + 1 < cols) candidates.push({ i, j, axis: 'u' })
      if (j + 1 < rows) candidates.push({ i, j, axis: 'v' })
    }
  }
  for (const c of candidates) {
    c.score = scoreCandidate(strategy, cfg, c, target)
    c.sagittaCm = computeSagittaCm(cfg, c, target)
  }

  // --- 2. PLATE FIT — drop any candidate whose sagitta exceeds the tolerance
  //        BEFORE ranking. See the file header's "PLATE FIT" section: a
  //        strategy chooses among candidates that fit; it never gets to place
  //        one that does not, no matter how well it would otherwise score.
  //        (Overrides above are exempt from this filter entirely — see
  //        "MANUAL OVERRIDES".)
  const eligible = candidates.filter((c) => c.sagittaCm <= tolerance + SAGITTA_EPS)

  // --- 3. Place highest-score-first among the survivors; ties break on
  //        (i, j, axis). Anything touching a cell an override already
  //        committed is skipped exactly like a cell an earlier-placed
  //        candidate already covered — overrides are seeded into `covered`
  //        before this loop runs, so this is the SAME mechanism, not a
  //        special case. ---------------------------------------------------
  eligible.sort((p, q) => {
    if (q.score !== p.score) return q.score - p.score
    if (p.i !== q.i) return p.i - q.i
    if (p.j !== q.j) return p.j - q.j
    if (p.axis !== q.axis) return p.axis < q.axis ? -1 : 1
    return 0
  })

  for (const cand of eligible) {
    const cells = cand.axis === 'u' ? [[cand.i, cand.j], [cand.i + 1, cand.j]] : [[cand.i, cand.j], [cand.i, cand.j + 1]]
    if (cells.some(([i, j]) => covered.has(cellKey(i, j)))) continue
    for (const [i, j] of cells) covered.add(cellKey(i, j))
    tiles.push(makePlateTile(cfg, cand))
    plateCount++
  }

  // --- 4. Fill the remainder with squares — this also catches cells whose
  //        only candidates were rejected for exceeding the sagitta tolerance,
  //        exactly like cells nobody ever proposed a domino for. ------------
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) {
      if (!covered.has(cellKey(i, j))) {
        tiles.push(makeSquareTile(cfg, i, j))
        covered.add(cellKey(i, j))
      }
    }
  }

  // --- 5. Stable output order: by anchor cell (j, i), independent of the
  //        (score-dependent) insertion order above. ---------------------------
  tiles.sort((p, q) => {
    const [pi, pj] = p.cells[0]
    const [qi, qj] = q.cells[0]
    if (pj !== qj) return pj - qj
    if (pi !== qi) return pi - qi
    return p.id < q.id ? -1 : p.id > q.id ? 1 : 0
  })

  // --- 6. MIN_PLATES — non-blocking, see file header and buildFewPlatesWarning.
  //        Now genuinely reachable (P2b): a form too curved for
  //        `plateFitToleranceCm` to admit MIN_PLATES plates fires this for
  //        real, and the message says why (points at the tolerance knob).
  //        Override misfit warnings (P8, "MANUAL OVERRIDES") are collected
  //        first, so they read before the aggregate plate-count warning.
  const warnings = [...overrideWarnings]
  const fewPlatesWarning = buildFewPlatesWarning(plateCount, strategy, cols, rows, tolerance)
  if (fewPlatesWarning) warnings.push(fewPlatesWarning)

  // --- 7. Adjacency, in material space, exact -------------------------------
  const adjacency = computeAdjacency(tiles, cfg.gap)

  return { tiles, adjacency, pattern: strategy, warnings }
}
