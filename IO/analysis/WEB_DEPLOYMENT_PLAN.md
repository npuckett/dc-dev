# Drop Ceiling — Plan: Run Data Online (Interactive Diagram Engine)

*Plan of record for moving the run database online to power an interactive, web-based
diagram engine. Written to be picked up later. The guiding principle: **serve the data at
the tier each use actually needs** — don't ship a 479 MB database to do 260 KB of work.*

---

## 1. The core insight (why this isn't "host the .db")

The data splits into three tiers of wildly different size and access pattern:

| Tier | Size | Contents | Who needs it |
|---|---|---|---|
| **Aggregates** | `hourly_stats_filled` ≈ **260 KB** JSON (~40 KB gzipped); `daily_stats_v2` ≈ 10 KB | per-hour / per-day rollups | **~90% of diagrams** (D1/D2, E4, F1–F3, rhythms, totals) |
| **Behaviour / audit** | `behavior_adjustments` 217K rows + `light_behavior` 172K rows (the bulk of the 479 MB, mostly `adjustments_json` blobs) | per-cycle tuning decisions, light-state log | deep / interactive drill-downs only |
| **Raw archive** | full `merged_run.db` ≈ **479 MB** (raw `tracking_events` already pruned) | everything | download / provenance, **not** live querying |

**Vercel is right for the app, wrong for the .db.** Serverless has no persistent disk;
loading a 479 MB SQLite per invocation to answer "give me hourly counts" is pathological.
So: static JSON on the CDN for aggregates, a hosted SQL service for the behaviour tables,
object storage for the full-file archive.

---

## 2. Target architecture

```
                         ┌─────────────────────────────┐
   Browser (interactive  │  Vercel app (frontend + thin │
   diagram engine) ──────│  API routes if needed)      │
                         └──────────┬─────────┬─────────┘
                                    │         │
                fetch() static      │         │  query (only for drill-downs)
                ▼                    ▼         ▼
   ┌────────────────────┐  ┌──────────────────────┐  ┌───────────────────┐
   │ /public/data/*.json│  │  Turso (hosted libSQL │  │ Cloudflare R2 / B2 │
   │ aggregates (260 KB)│  │  / SQLite): behaviour  │  │ full merged_run.db │
   │ — CDN, free, fast  │  │  + audit tables        │  │ (archive download) │
   └────────────────────┘  └──────────────────────┘  └───────────────────┘
        TIER 1 (build first)     TIER 2 (add if needed)     TIER 3 (backup)
```

**Build order: Tier 1 first, alone.** It already covers every figure made so far. Add
Tier 2 only when a diagram genuinely needs to query the 200K+ behaviour rows live. Tier 3 is
just the Dropbox/R2 archive copy.

---

## 3. Tier 1 — static JSON aggregates (the first deliverable)

### What to export (from `IO/analysis/merged_run.db`)
- **`hourly_stats_filled`** → `hourly.json` — the authoritative continuous, provenance-tagged,
  artifact-corrected hourly series. **Carry the flags** (`src`, `estimated`, `active_orig`,
  `passive_orig`) so the front-end can be honest about data quality.
- **`daily_stats_v2`** (+ a daily roll-up of the filled table) → `daily.json`.
- A small **`meta.json`**: run span, total events, day count, the software-version timeline
  (dates + labels, as in F1), the gap/artifact notes, the gesture/mode "first appeared"
  dates. This drives annotations and honesty toggles.

### Shape (keep it flat and small)
```jsonc
// hourly.json
[ { "date":"2026-03-10","hour":16,"events":156176,"people":3171,
    "active":1504,"passive":76005,"mode":"aware","brightness":203,
    "ltr":52887,"rtl":14701,"src":"late","est":0 }, ... ]
```
Gzip/Brotli handles the rest (~40 KB on the wire). No backend needed.

### Where it lives
- Simplest: a `data/` folder in the Vercel app's `/public` (or GitHub Pages — the viewer is
  already deployed there). Static, cached at the edge, free, works offline.

### Front-end
- Plain `fetch()` + a charting lib: **D3** (max control for bespoke/interactive diagrams),
  **Observable Plot** (fast, declarative, great for exploratory), or **Chart.js** (simplest).
  Given the goal is a *bespoke diagram engine*, D3 or Plot.
- Re-implement the existing figures (D/E/F) as interactive versions reading `hourly.json`.

### Honesty as a feature (differentiator)
Bake the provenance flags into the UI: toggles for "show estimated days," "highlight
report-recovered week," annotations for when modes/gestures first appeared (§18). An
interactive engine that **shows its own data quality** is more credible than one that hides
it — and it's a natural story for the project.

---

## 4. Tier 2 — hosted SQL (only if/when drill-downs need it)

Trigger: a diagram needs to query individual rows live (e.g. "every adjustment to `energy`
on Mar 10," "gesture frequency by hour from `light_behavior`").

- **First choice: Turso** (hosted libSQL / SQLite) — the data *is* SQLite, generous free
  tier, edge-replicated, clean Vercel integration. Push only the tables/columns you query
  (strip `adjustments_json` to the few fields needed → well under free-tier size).
- Alternatives: **Neon** or **Supabase** (Postgres) if Postgres is preferred; both have
  good free tiers + Vercel integrations.
- Access via thin Vercel API routes (so the DB creds stay server-side) or Turso's edge SDK.

**Pre-process before upload:** explode the useful keys out of `adjustments_json` into real
columns (param deltas, budget before/after/cost, gradients) so they're queryable without
JSON parsing per row. ~50–100 MB once trimmed.

---

## 5. Tier 3 — full-file archive (backup + provenance)

- Keep the full **`merged_run.db` (479 MB)** and the two source DBs (628 MB, 487 MB) in
  **Dropbox** (already planned) and optionally **Cloudflare R2 / Backblaze B2 / S3** as a
  public/downloadable "source of truth."
- **Critical reminder:** the DBs are **gitignored**, so GitHub does **not** back them up.
  The off-machine copy (Dropbox/R2) is the *only* backup of the run data. Keep it current.

---

## 6. Concrete next steps (when we return)

1. **Write the export script** (`IO/analysis/export_web_json.py`): reads `merged_run.db`,
   emits `hourly.json`, `daily.json`, `meta.json` with flags. Idempotent; re-runnable when
   data updates.
2. **Stand up the Vercel app skeleton** (or reuse the existing GitHub Pages viewer):
   `/public/data/*.json` + a minimal page that fetches and renders one diagram (port D2 or
   E4 first as the proof).
3. **Re-implement 1–2 figures interactively** against the JSON to validate the pipeline.
4. **Decide on Tier 2** only after hitting a real drill-down need.
5. **Confirm the Tier 3 archive** (Dropbox copy of all DBs) is in place before relying on it.

### Open decisions to make then
- Charting lib: **D3 vs Observable Plot** (vs Chart.js for the simple ones).
- Host: **Vercel app** vs extend the **existing GitHub Pages** viewer (the live viewer
  already lives there — could co-locate).
- Whether the interactive engine is **public** or gated.

---

## 7. Data references (for the export script)

- Authoritative table: **`IO/analysis/merged_run.db` → `hourly_stats_filled`** (continuous
  Jan 29 – Mar 17, 48 days, ~23.3 M events, provenance-tagged).
- Provenance/quality notes & the software-version timeline: **`DATA_TIMELINE_AND_MERGE.md`**
  (this folder) and **§18** of `ACADIA_2026_DIAGRAM_GUIDE.md`.
- Existing static figures to port: **D1/D2, E4, F1/F2/F3** (`IO/diagrams/`).
