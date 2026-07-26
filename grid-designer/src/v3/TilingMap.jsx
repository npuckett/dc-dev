/**
 * grid-designer v3 — plan view of the material tiling (`cols × rows`).
 *
 * The MATERIAL lattice (V3_SPEC.md §1): `i` runs across the window (+X, wall
 * side first — `sheet.cols` values), `j` runs window → back (+Z —
 * `sheet.rows` values). This maps onto v2's GridMap.jsx conventions exactly —
 * i is v2's "column" axis (wall at i = 0), j is v2's "row" axis (window at
 * j = 0) — so this view keeps GridMap's layout: ROW (j) 0 AT THE BOTTOM, WALL
 * (i = 0) ON THE LEFT. THE MIRROR (same note as GridMap.jsx and
 * DriftViewport.jsx): the 3D camera looks in from the window, so wall (x = 0)
 * reads as the RIGHT side there and the LEFT side here.
 *
 * Squares are single cells; plates span two cells — `axis: 'u'` (spans i,
 * i+1) draws HORIZONTAL here, `axis: 'v'` (spans j, j+1) draws VERTICAL.
 *
 * Tint mode: 'height' (cumulative material height at the tile's centre, from
 * the target surface — reads as a literal drift-shaped wash) or 'facet'
 * (facet plane the tile belongs to, matching the viewport's facet colour
 * mode via the same hash-to-hue scheme).
 *
 * Hover shows the tile's id / type / sagitta / clearance in a small readout,
 * and is written to the shared store (`hoveredTileId`) so DriftViewport's 3D
 * scene highlights the SAME tile — a hover here is legible in both views.
 *
 * =============================================================================
 * INTERACTION — MANUAL OVERRIDES (P8), reproducing v2's pair-click merge
 * =============================================================================
 * The hit grid also drives `config.tiling.overrides` (schema.js's "MANUAL TILE
 * OVERRIDES"), mirroring v2's GridMap.jsx (commit 565af13):
 *   - click a SQUARE           → it is ARMED (a warm outline), and every
 *                                orthogonal neighbour it could combine with
 *                                lights up
 *   - click a HIGHLIGHTED      → the two squares become one 60×121 plate —
 *     neighbour                  green solid "+" where the plate fits within
 *                                `tiling.plateFitToleranceCm`, amber dashed "~"
 *                                where it would not (placed anyway and
 *                                flagged — see tiling.js's "MANUAL OVERRIDES";
 *                                a manual override is never refused for not
 *                                fitting, only reported)
 *   - click the armed cell again, or elsewhere → disarm. So does Escape.
 *   - click a PLATE            → split it back into two squares
 * The candidate's `<title>` tooltip spells out the consequence (sagitta vs.
 * tolerance) BEFORE the click, and `lastActionNotice` (the calm line under the
 * map) repeats what actually happened after it.
 *
 * A pinned tile (`tile.pinned`, set by `solveTiling` for anything that came
 * from an override rather than the algorithm) gets a small dot so "why is
 * this a plate/square here" is legible without hunting through the JSON.
 */

import { useEffect, useMemo, useState } from 'react'
import * as THREE from 'three'
import useStoreV3, { getDerived } from './store.js'
import { buildTarget } from '../core/v3/target.js'
import { computeSagittaCm } from '../core/v3/tiling.js'
import { normalizeConfig } from '../core/v3/schema.js'

const CELL = 26
const GAP = 3
const PITCH = CELL + GAP
const PAD_L = 16
const PAD_R = 10
const PAD_T = 10
const PAD_B = 15

const SQUARE_STROKE = '#3c3c56'
const RECT_STROKE = '#4a7fc4'
const WALL = '#3ad0c0'
const SHORE = '#8fcaff'
const COLLISION = '#ff2d2d'

/** The square being held, waiting for a partner to combine with (P8). */
const ARMED = '#ffd479'
/** A combine that fits within tiling.plateFitToleranceCm. */
const FREE = '#5fd08a'
/** A combine whose plate would bow past tiling.plateFitToleranceCm — placed anyway, flagged. */
const MISFIT = '#ffa94d'
/** Guards the `sagittaCm <= tolerance` comparison against float noise (mirrors tiling.js's SAGITTA_EPS). */
const OVERRIDE_FIT_EPS = 1e-9

function hashHue(str) {
  let h = 0
  for (let i = 0; i < str.length; i++) h = (h * 31 + str.charCodeAt(i)) >>> 0
  return (h % 360) / 360
}

function heightFill(h, maxH) {
  const t = THREE.MathUtils.clamp(h / (maxH > 0 ? maxH : 1), 0, 1)
  const c = new THREE.Color().lerpColors(new THREE.Color('#1f3350'), new THREE.Color('#ffb27f'), t)
  return `#${c.getHexString()}`
}

function facetFill(key) {
  return `#${new THREE.Color().setHSL(hashHue(key), 0.5, 0.4).getHexString()}`
}

export default function TilingMap() {
  const config = useStoreV3((s) => s.config)
  const hoveredTileId = useStoreV3((s) => s.hoveredTileId)
  const setHoveredTile = useStoreV3((s) => s.setHoveredTile)
  const armedCell = useStoreV3((s) => s.armedCell)
  const armCell = useStoreV3((s) => s.armCell)
  const clearArmed = useStoreV3((s) => s.clearArmed)
  const combineCells = useStoreV3((s) => s.combineCells)
  const splitTileAt = useStoreV3((s) => s.splitTileAt)
  const clearOverrides = useStoreV3((s) => s.clearOverrides)
  const lastActionNotice = useStoreV3((s) => s.lastActionNotice)
  const { layout, report } = getDerived(config)
  const [tintMode, setTintMode] = useState('height')

  const cols = config.sheet.cols
  const rows = config.sheet.rows

  const normCfg = useMemo(() => normalizeConfig(config), [config])
  const target = useMemo(() => buildTarget(normCfg), [normCfg])
  const tolerance = normCfg.tiling.plateFitToleranceCm
  const overrideCount = normCfg.tiling.overrides.length

  // Escape disarms — the standard way out of a held selection (mirrors v2's
  // GridMap.jsx).
  useEffect(() => {
    if (!armedCell) return undefined
    const onKeyDown = (e) => {
      if (e.key === 'Escape') clearArmed()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [armedCell, clearArmed])

  const collisionSet = useMemo(() => {
    const s = new Set()
    for (const c of report.collisions) {
      s.add(c.a)
      s.add(c.b)
    }
    return s
  }, [report])

  const worstDevByTile = useMemo(() => {
    const m = new Map()
    for (const j of report.joints) {
      m.set(j.a, Math.max(m.get(j.a) ?? 0, j.deviationCm))
      m.set(j.b, Math.max(m.get(j.b) ?? 0, j.deviationCm))
    }
    return m
  }, [report])

  const tileHeights = useMemo(() => {
    const m = new Map()
    for (const t of layout.tiles) {
      const u = t.uv.u0 + t.uv.uLen / 2
      const v = t.uv.v0 + t.uv.vLen / 2
      m.set(t.id, target.heightAtMaterial(u, v))
    }
    return m
  }, [layout, target])
  const maxHeight = useMemo(() => Math.max(1, ...Array.from(tileHeights.values())), [tileHeights])

  const width = PAD_L + cols * CELL + (cols - 1) * GAP + PAD_R
  const height = PAD_T + rows * CELL + (rows - 1) * GAP + PAD_B
  const x = (i) => PAD_L + i * PITCH
  const y = (j) => PAD_T + (rows - 1 - j) * PITCH

  const ownerOf = useMemo(() => {
    const m = new Map()
    for (const tile of layout.tiles) {
      for (const [i, j] of tile.cells) m.set(`${i},${j}`, tile)
    }
    return m
  }, [layout])

  // The armed square's combinable neighbours, keyed "i,j". Each candidate's
  // sagitta is computed the SAME way tiling.js measures an override's own —
  // `computeSagittaCm` against the TARGET, in 3D — so the tooltip and the
  // free/misfit colouring are never guessing at a consequence the store's
  // `combineCells` would then measure differently.
  const candidates = useMemo(() => {
    const m = new Map()
    if (!armedCell) return m
    const owner = ownerOf.get(`${armedCell.i},${armedCell.j}`)
    if (!owner || owner.type !== '2x2') return m
    const neighbours = [
      [armedCell.i - 1, armedCell.j],
      [armedCell.i + 1, armedCell.j],
      [armedCell.i, armedCell.j - 1],
      [armedCell.i, armedCell.j + 1],
    ]
    for (const [ni, nj] of neighbours) {
      if (ni < 0 || ni >= cols || nj < 0 || nj >= rows) continue
      const nOwner = ownerOf.get(`${ni},${nj}`)
      if (!nOwner || nOwner.type !== '2x2') continue
      const axis = nj === armedCell.j ? 'u' : 'v'
      const i = Math.min(armedCell.i, ni)
      const j = Math.min(armedCell.j, nj)
      const sagittaCm = computeSagittaCm(normCfg, { i, j, axis }, target)
      const fits = sagittaCm <= tolerance + OVERRIDE_FIT_EPS
      // `ni`/`nj` (the neighbour's OWN cell, where the marker is drawn) are
      // distinct from `i`/`j` (the merged plate's ANCHOR, its lower-index
      // corner) — the two coincide only when the neighbour is already the
      // lower-index cell.
      m.set(`${ni},${nj}`, { ni, nj, i, j, axis, sagittaCm, fits })
    }
    return m
  }, [armedCell, ownerOf, normCfg, target, tolerance, cols, rows])

  /** Click on a plan-view cell: arm / combine / disarm / split — see the file
   *  header's "INTERACTION" section. */
  const handleCellClick = (i, j) => {
    const owner = ownerOf.get(`${i},${j}`)
    if (!owner) return
    if (owner.type === '2x4') {
      splitTileAt(owner.cells[0][0], owner.cells[0][1])
      clearArmed()
      return
    }
    if (armedCell && armedCell.i === i && armedCell.j === j) {
      clearArmed()
      return
    }
    if (armedCell && candidates.has(`${i},${j}`)) {
      combineCells(armedCell, { i, j })
      clearArmed()
      return
    }
    // Either nothing was armed, or the clicked square isn't a valid partner
    // for the one that was — arm this one instead.
    armCell({ i, j })
  }

  const hovered = hoveredTileId ? layout.tiles.find((t) => t.id === hoveredTileId) : null

  const fillFor = (tile) => {
    if (collisionSet.has(tile.id)) return COLLISION
    if (tintMode === 'facet') {
      const u = tile.uv.u0 + tile.uv.uLen / 2
      const v = tile.uv.v0 + tile.uv.vLen / 2
      return facetFill(target.facetIndexAt(u, v))
    }
    return heightFill(tileHeights.get(tile.id) ?? 0, maxHeight)
  }

  return (
    <section className="tiling-map" data-testid="tiling-map">
      <header className="grid-map-head">
        <span className="grid-map-title">tiling — {config.tiling.strategy}</span>
        {armedCell && (
          <span className="grid-map-armed-label" data-testid="tilingmap-armed-label">
            armed ({armedCell.i}, {armedCell.j}) · {candidates.size} to combine with · Esc
          </span>
        )}
        <div className="tiling-tint-group">
          <button
            type="button"
            className={`preset-btn${tintMode === 'height' ? ' preset-active' : ''}`}
            data-testid="tiling-tint-height"
            onClick={() => setTintMode('height')}
          >
            height
          </button>
          <button
            type="button"
            className={`preset-btn${tintMode === 'facet' ? ' preset-active' : ''}`}
            data-testid="tiling-tint-facet"
            onClick={() => setTintMode('facet')}
          >
            facet
          </button>
          {overrideCount > 0 && (
            <button
              type="button"
              className="preset-btn"
              data-testid="tilingmap-clear-overrides"
              title={`clear ${overrideCount} manual override${overrideCount === 1 ? '' : 's'} — let the algorithm decide every cell again`}
              onClick={() => clearOverrides()}
            >
              clear pins ({overrideCount})
            </button>
          )}
        </div>
      </header>

      <svg
        className="grid-map-svg"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`${cols} by ${rows} material tiling, row (j) 0 at the bottom, wall (i = 0) at the left`}
        onMouseLeave={() => setHoveredTile(null)}
        onClick={() => clearArmed()}
      >
        <g style={{ pointerEvents: 'none' }}>
          {Array.from({ length: rows }, (_, j) => (
            <text key={`r-${j}`} className="grid-map-rowlabel" x={x(cols - 1) + CELL + 4} y={y(j) + CELL / 2 + 3}>
              {j}
            </text>
          ))}
          {Array.from({ length: cols }, (_, i) => (
            <text key={`c-${i}`} className="grid-map-collabel" x={x(i) + CELL / 2} y={PAD_T - 3}>
              {i}
            </text>
          ))}

          {layout.tiles.map((tile) => {
            const isPlate = tile.type === '2x4'
            const horiz = isPlate && tile.axis === 'u'
            const vert = isPlate && tile.axis === 'v'
            const [i0, j0] = tile.cells[0]
            const tx = x(i0)
            const ty = vert ? y(j0 + 1) : y(j0)
            const tw = horiz ? 2 * CELL + GAP : CELL
            const th = vert ? 2 * CELL + GAP : CELL
            return (
              <g key={tile.id}>
                <rect
                  x={tx}
                  y={ty}
                  width={tw}
                  height={th}
                  rx={3}
                  fill={fillFor(tile)}
                  stroke={isPlate ? RECT_STROKE : SQUARE_STROKE}
                  strokeWidth={isPlate ? 1.3 : 0.8}
                  opacity={hoveredTileId && hoveredTileId !== tile.id ? 0.55 : 1}
                />
                {/* a manually-pinned tile (P8) gets a small dot — "why is this
                    a plate/square here" legible without the JSON panel */}
                {tile.pinned && (
                  <circle
                    cx={tx + tw - 4.5}
                    cy={ty + 4.5}
                    r={2.1}
                    fill={ARMED}
                    data-testid={`tile-pinned-${tile.id}`}
                  />
                )}
              </g>
            )
          })}

          {hovered && (
            <rect
              x={x(hovered.cells[0][0]) + 0.5}
              y={
                hovered.type === '2x4' && hovered.axis === 'v'
                  ? y(hovered.cells[0][1] + 1) + 0.5
                  : y(hovered.cells[0][1]) + 0.5
              }
              width={(hovered.type === '2x4' && hovered.axis === 'u' ? 2 * CELL + GAP : CELL) - 1}
              height={(hovered.type === '2x4' && hovered.axis === 'v' ? 2 * CELL + GAP : CELL) - 1}
              rx={3}
              fill="none"
              stroke="#ffffff"
              strokeWidth={2}
            />
          )}

          {/* the armed square, and the neighbours it could combine with (P8).
              FREE = solid green "+", MISFIT = dashed amber "~": whether the
              resulting plate would bow past tiling.plateFitToleranceCm — see
              the file header's "INTERACTION" section. A misfit candidate is
              still offered and still placeable; the styling only tells you
              the consequence before you click it. */}
          {armedCell && ownerOf.get(`${armedCell.i},${armedCell.j}`) && (
            <g data-testid="tilingmap-armed">
              <rect
                x={x(armedCell.i) + 1}
                y={y(armedCell.j) + 1}
                width={CELL - 2}
                height={CELL - 2}
                rx={2.5}
                fill={ARMED}
                fillOpacity={0.16}
                stroke={ARMED}
                strokeWidth={2.4}
              />
            </g>
          )}

          {[...candidates.values()].map((cand) => {
            const color = cand.fits ? FREE : MISFIT
            return (
              <g
                key={`cand-${cand.ni}-${cand.nj}`}
                data-testid={`tilingmap-candidate-${cand.ni}-${cand.nj}`}
                data-fits={cand.fits}
              >
                <rect
                  x={x(cand.ni) + 1}
                  y={y(cand.nj) + 1}
                  width={CELL - 2}
                  height={CELL - 2}
                  rx={2.5}
                  fill={color}
                  fillOpacity={cand.fits ? 0.2 : 0.13}
                  stroke={color}
                  strokeWidth={2}
                  strokeDasharray={cand.fits ? undefined : '4 2.5'}
                />
                <text className="grid-map-candmark" x={x(cand.ni) + CELL / 2} y={y(cand.nj) + CELL / 2 + 4.5} fill={color}>
                  {cand.fits ? '+' : '~'}
                </text>
              </g>
            )
          })}

          <line
            x1={PAD_L}
            y1={y(0) + CELL + 2}
            x2={PAD_L + cols * CELL + (cols - 1) * GAP}
            y2={y(0) + CELL + 2}
            stroke={SHORE}
            strokeWidth={1.2}
          />
          <text
            className="grid-map-shore"
            x={PAD_L + (cols * CELL + (cols - 1) * GAP) / 2}
            y={height - 3}
            textAnchor="middle"
          >
            window / shore
          </text>

          <g data-testid="tilingmap-wall">
            <line x1={PAD_L - 3} y1={PAD_T} x2={PAD_L - 3} y2={y(0) + CELL} stroke={WALL} strokeWidth={2.2} />
            <text
              className="grid-map-wall"
              transform={`translate(${PAD_L - 9} ${(PAD_T + y(0) + CELL) / 2}) rotate(-90)`}
              textAnchor="middle"
            >
              wall
            </text>
          </g>
        </g>

        {/* hit grid on top, cell-owner aware so a hover targets the whole tile;
            also the click surface for the P8 arm/combine/split interaction */}
        <g>
          {Array.from({ length: rows }, (_, j) =>
            Array.from({ length: cols }, (_, i) => {
              const owner = ownerOf.get(`${i},${j}`)
              const isArmed = Boolean(armedCell) && armedCell.i === i && armedCell.j === j
              const cand = candidates.get(`${i},${j}`)
              const action = !owner
                ? undefined
                : owner.type === '2x4'
                  ? `click to split ${owner.id} back into two squares`
                  : isArmed
                    ? 'armed — click a highlighted neighbour to combine, or click here again to let go'
                    : cand
                      ? `click to combine into a plate — ${cand.fits ? `fits within ${tolerance.toFixed(1)}cm tolerance` : `bows past tolerance (sagitta ${cand.sagittaCm.toFixed(2)}cm vs ${tolerance.toFixed(1)}cm) — placed anyway and flagged`}`
                      : armedCell
                        ? 'not combinable with the armed square — click to arm this one instead'
                        : 'click to arm this square, then click a neighbour to combine into a 121cm plate'
              return (
                <rect
                  key={`hit-${i}-${j}`}
                  data-testid={`tile-cell-${i}-${j}`}
                  className="grid-map-hit"
                  data-armed={isArmed || undefined}
                  data-candidate={cand ? (cand.fits ? 'free' : 'misfit') : undefined}
                  x={x(i)}
                  y={y(j)}
                  width={CELL}
                  height={CELL}
                  fill="#ffffff"
                  fillOpacity={0}
                  onMouseEnter={() => owner && setHoveredTile(owner.id)}
                  onClick={(e) => {
                    e.stopPropagation()
                    handleCellClick(i, j)
                  }}
                >
                  {owner && (
                    <title>
                      {`${owner.id} · ${owner.type} · sagitta ${owner.sagittaCm.toFixed(2)}cm · clearance ${
                        Number.isFinite(owner.minY) ? owner.minY.toFixed(2) : '—'
                      }cm · worst joint dev ${(worstDevByTile.get(owner.id) ?? 0).toFixed(2)}cm` +
                        (owner.pinned ? ' · manually pinned' : '') +
                        (action ? ` — ${action}` : '')}
                    </title>
                  )}
                </rect>
              )
            }),
          )}
        </g>
      </svg>

      {hovered && (
        <p className="grid-map-hint" data-testid="tiling-hover-detail">
          <b>{hovered.id}</b> · {hovered.type} · sagitta {hovered.sagittaCm.toFixed(2)}cm · clearance{' '}
          {Number.isFinite(hovered.minY) ? hovered.minY.toFixed(2) : '—'}cm · worst joint dev{' '}
          {(worstDevByTile.get(hovered.id) ?? 0).toFixed(2)}cm{hovered.pinned ? ' · manually pinned' : ''}
        </p>
      )}
      <p className="grid-map-hint">
        click a square, then a highlighted neighbour, to <b>combine</b> them into one 60×121 plate —{' '}
        <span className="cand-free">green +</span> fits, <span className="cand-coerce">amber ~</span> bows past
        tolerance (placed anyway, flagged) · click a plate to <b>split</b> it (Esc lets go) ·{' '}
        row (j) 0 at bottom (window) · <b>wall</b> (i = 0) at the left here, <b>right</b> in 3D · plates:{' '}
        {layout.tiles.filter((t) => t.type === '2x4').length} · squares:{' '}
        {layout.tiles.filter((t) => t.type === '2x2').length} · pinned:{' '}
        {layout.tiles.filter((t) => t.pinned).length}
      </p>

      {lastActionNotice && (
        <p className="grid-map-notice" data-testid="tilingmap-notice" title="what the last override change did">
          {lastActionNotice}
        </p>
      )}
    </section>
  )
}
