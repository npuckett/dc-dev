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
 */

import { useMemo, useState } from 'react'
import * as THREE from 'three'
import useStoreV3, { getDerived } from './store.js'
import { buildTarget } from '../core/v3/target.js'
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
  const { layout, report } = getDerived(config)
  const [tintMode, setTintMode] = useState('height')

  const cols = config.sheet.cols
  const rows = config.sheet.rows

  const target = useMemo(() => buildTarget(normalizeConfig(config)), [config])

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

  const ownerOf = new Map()
  for (const tile of layout.tiles) {
    for (const [i, j] of tile.cells) ownerOf.set(`${i},${j}`, tile)
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
        </div>
      </header>

      <svg
        className="grid-map-svg"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`${cols} by ${rows} material tiling, row (j) 0 at the bottom, wall (i = 0) at the left`}
        onMouseLeave={() => setHoveredTile(null)}
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
            return (
              <rect
                key={tile.id}
                x={x(i0)}
                y={vert ? y(j0 + 1) : y(j0)}
                width={horiz ? 2 * CELL + GAP : CELL}
                height={vert ? 2 * CELL + GAP : CELL}
                rx={3}
                fill={fillFor(tile)}
                stroke={isPlate ? RECT_STROKE : SQUARE_STROKE}
                strokeWidth={isPlate ? 1.3 : 0.8}
                opacity={hoveredTileId && hoveredTileId !== tile.id ? 0.55 : 1}
              />
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

        {/* hit grid on top, cell-owner aware so a hover targets the whole tile */}
        <g>
          {Array.from({ length: rows }, (_, j) =>
            Array.from({ length: cols }, (_, i) => {
              const owner = ownerOf.get(`${i},${j}`)
              return (
                <rect
                  key={`hit-${i}-${j}`}
                  data-testid={`tile-cell-${i}-${j}`}
                  x={x(i)}
                  y={y(j)}
                  width={CELL}
                  height={CELL}
                  fill="#ffffff"
                  fillOpacity={0}
                  onMouseEnter={() => owner && setHoveredTile(owner.id)}
                >
                  {owner && (
                    <title>
                      {`${owner.id} · ${owner.type} · sagitta ${owner.sagittaCm.toFixed(2)}cm · clearance ${
                        Number.isFinite(owner.minY) ? owner.minY.toFixed(2) : '—'
                      }cm · worst joint dev ${(worstDevByTile.get(owner.id) ?? 0).toFixed(2)}cm`}
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
          {(worstDevByTile.get(hovered.id) ?? 0).toFixed(2)}cm
        </p>
      )}
      <p className="grid-map-hint">
        row (j) 0 at bottom (window) · <b>wall</b> (i = 0) at the left here, <b>right</b> in 3D · plates:{' '}
        {layout.tiles.filter((t) => t.type === '2x4').length} · squares:{' '}
        {layout.tiles.filter((t) => t.type === '2x2').length}
      </p>
    </section>
  )
}
