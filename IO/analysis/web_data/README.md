# web_data/

Static-JSON projection of `IO/analysis/merged_run.db` for the web viewer
(Tier 1 of `IO/analysis/WEB_DEPLOYMENT_PLAN.md`).

The full 479 MB SQLite database is never shipped to the browser. These
files are the *only* data the viewer needs to render every figure that
has been published so far (~90% of the diagram set).

## Files

| File | Shape | Size on disk (gzip) |
|---|---|---|
| `hourly.json` | flat array, one row per hour, ordered by date+hour | ~260 KB (~40 KB) |
| `daily.json`  | flat array, one row per day, computed from `hourly_stats_filled` | ~10 KB (~2 KB) |
| `meta.json`   | single object: run totals, source breakdown, software timeline, gap/artifact notes, vocabulary first-seen dates | ~3 KB (~1 KB) |

### `hourly.json` fields (per row)

| key | type | source column | meaning |
|---|---|---|---|
| `date` | string | `date` | YYYY-MM-DD |
| `hour` | int 0-23 | `hour` | local hour |
| `events` | int | `total_events` | tracking events that hour |
| `people` | int | `unique_people` | unique detections that hour (sum overcounts repeats) |
| `active` | int | `active_count` | active-zone visits |
| `passive` | int | `passive_count` | passive-zone visits |
| `ltr` | int | `left_to_right` | flow direction count |
| `rtl` | int | `right_to_left` | flow direction count |
| `blooms` | int | `bloom_count` | bloom gestures that hour |
| `mode` | string | `dominant_mode` | light's dominant mode that hour (`idle`/`flow`/`aware`/`engaged`/`crowd`/`unknown`) |
| `brightness` | float | `avg_brightness` | mean brightness that hour |
| `speed` | float | `avg_speed` | mean pedestrian speed that hour |
| `src` | string | `src` | `early` / `late` / `report` (provenance) |
| `est` | 0\|1 | `estimated` | 1 = the active/passive split is modelled, not measured |
| `active_orig` | int | `active_orig` | as-logged active count (pre-correction) |
| `passive_orig` | int | `passive_orig` | as-logged passive count (pre-correction) |

### `daily.json` fields (per row)

`date`, `events`, `active`, `passive`, `blooms`, `ltr`, `rtl`, `hours`,
`est_hours`, `report_hours`, `early_hours`, `late_hours`, `brightness`,
`speed`, `flow_balance`, `dominant_flow`, `people_sum` (ceiling, see note).

### `meta.json` keys

`run`, `src_breakdown`, `estimated_hours_total`, `gap`, `calibration_artifact`,
`software_timeline`, `vocabulary_first_seen`.

## Regenerate

```sh
.venv/bin/python IO/analysis/export_web_json.py
```

The script is idempotent. It writes the three files and (by default)
auto-commits them with a generated message. Pass `--no-commit` to write
without committing, or `--pretty` for human-readable output.

Useful flags:

```sh
python IO/analysis/export_web_json.py --pretty --no-commit   # inspect
python IO/analysis/export_web_json.py                       # ship it
```

## Hosting

Served as static files from GitHub Pages (the same site the existing
viewer lives on). Nothing here is sensitive — these are aggregate
counts. The full `merged_run.db` stays off the network.

## When to regenerate

After any change to `IO/analysis/merged_run.db` (new daily-report
ingest, schema change, calibration-fix update, etc.). The
auto-generated commit message records the date, hour count, and event
total so the history is self-describing.

## What this does not include

- `behavior_adjustments` and `light_behavior` (the 217K + 172K row
  audit tables) — Tier 2 in the deployment plan, only added if a
  diagram needs to query individual rows live.
- The full 479 MB `merged_run.db` — Tier 3, kept off-machine as the
  archive / provenance copy.
