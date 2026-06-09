# Drop Ceiling — Run Data: Merge, Coverage & Change Timeline

*Reference for all longitudinal analysis. Built from two intact database copies found on
the Desktop dc-dev repo. The installation's software changed continuously over the run, so
**what was captured changed too** — this document records when, so figures and claims can
be framed honestly.*

---

## 1. Source databases (both integrity-checked: clean, no corruption)

| | Path | Size | hourly_stats span | Role |
|---|---|---|---|---|
| **late** | `~/Desktop/dc-dev/tracking_history.db` | 628 MB | Jan 29–Feb 2, **Feb 25–Mar 17** | late run |
| **early** | `~/Desktop/dc-dev/IO/tracking_history.db` | 487 MB | **Feb 12–25** | early run |

Both pass `PRAGMA integrity_check` = `ok`. **Schemas are identical** across all tables, so
a clean union was safe. They are **complementary, not duplicates**: the early DB fills the
Feb 12–24 window the late DB is missing.

## 2. Merged database

`IO/analysis/merged_run.db` (read-only analysis copy; originals untouched). Built by
`ATTACH`-ing both and unioning, deduplicating overlapping days/timestamps (preferring the
row with more events). Carries a `src` column (`early`/`late`) on every row for provenance.

| Table | Merged rows | Notes |
|---|---|---|
| `hourly_stats` | 895 | **41 distinct days**, Jan 29 → Mar 17 |
| `daily_stats_v2` | 40 | per-day rollups |
| `behavior_adjustments` | 217,285 | the tuner's audit trail |
| `light_behavior` | 172,125 | light's own state log (sparse — see §5) |

**True run totals (from merged `hourly_stats`):**
- **Tracking events: ~20.3 million** (was estimated ~15M — the real number is higher).
- Total blooms: 1,523.
- *Unique people:* summing per-hour uniques gives 536,693, but that **over-counts** anyone
  seen across multiple hours, so it is a ceiling, not the headline figure. The "~1 million
  visitors" claim should be sourced/qualified separately, not from this sum.

## 3. Day-by-day coverage (41 days present of a ~48-day span)

```
Jan 29 ████  (15h)        ─┐ early days (V2-era)
Jan 30–Feb 01 ████ (24h)   │
Feb 02 ██ (16h)           ─┘
Feb 03–09  ✗ MISSING (7-day gap)
Feb 10–11  ▏ (2h each, trace only)
Feb 12 ▌ (10h)  ← both DBs; DB "fixed corrupt database" commit this day
Feb 13–24 ████ (24h)      ─ early-run continuous block (V4/V5-era)
Feb 25 ████ (24h)  ← handoff: both DBs cover this day
Feb 26–Mar 16 ████ (24h)  ─ late-run continuous block (V5→V6→V6.5-era)
Mar 17 ████ (10h)  ← run ends
```

**Two real discontinuities:**
- **Feb 3–9 gap (7 days):** no aggregates. Cause not yet confirmed (downtime, or data lost
  in the pre-Feb-12 corruption). Treat the run as **two continuous blocks** (late-Jan +
  Feb-12-onward) rather than one unbroken 54-day line, unless downtime is confirmed benign.
- **Feb 10–12 trace:** 2-hour days, then the **"fixed corrupt database" fix on Feb 12**
  (git), after which capture is continuous. Your recollection of a corruption issue is
  correct — it sits right here, and the early DB starts clean from Feb 12.

## 4. What was captured changed over the run (the capture-timeline)

The data records the software's own evolution. Key transitions, cross-checked against git:

| Date | Data shows | Git / software change |
|---|---|---|
| **~Jan 27–28** | run begins; "active/passive" fixes | `massive V2 update`, `fixes to active passive` |
| **Feb 11–12** | trace days, then capture stabilises | `v4 update`, **`fixed corrupt database`** (Feb 12) |
| **Feb 23** | **`light_behavior` logging begins**; the full **mode** vocabulary (idle/flow/engaged/crowd) and **13 gestures** first appear in the DB | V5-era behaviour system with gestures/dwell |
| **Feb 25** | `sweep` gesture first appears | `v5 update` (anisotropic falloff + SWEEP/FOCUS) |
| **Mar 02** | brief `light_behavior` rows (227) — redeploy churn | `the big v6 update`, `V6.1*` stabilisation |
| **Mar 03** | three-tier passive behaviour lands | `V6.5: Passive-flow-driven behavior with three tiers` |
| **Mar 04** | data-driven movement | `V6.5c` |
| **Mar 15** | **`aware` mode first appears** in data; `playful` gesture first logged | V6.5 three-tier (IDLE/FLOW/**AWARE**) reaching steady logging |
| **Mar 16** | `focus` gesture first logged | late V6.5 gesture set complete |

**Implications for figures/claims:**
- **Mode vocabulary is not constant.** `aware` only exists from **Mar 15**; before that the
  light had four modes, not five. Any mode-distribution figure must either span only the
  V6.5 window or annotate the change.
- **Gesture vocabulary grew over the run** (13 gestures Feb 23 → +sweep Feb 25 → +playful
  Mar 15 → +focus Mar 16). "16 gestures" describes the *final* system, not the whole run.
- **`light_behavior` (the light's own state log) is sparse and intermittent** — only ~8
  days have substantial rows (Feb 23–25, Mar 15–17). Self-analysis / light-state figures
  (position entropy, gesture frequency over time) can only be built for those windows, not
  the full run.

## 5. What is and isn't buildable from this data

| Figure / claim | Buildable? | From |
|---|---|---|
| **D1/D2 refresh** (real run totals) | ✅ | merged `hourly_stats` (~20.3M events) |
| **E4 24-hour rhythm** | ✅ | `hourly_stats` by hour — pick a representative full V6.5 day |
| **E5 week-1 vs week-7** (activity/flow/brightness) | ✅ | `hourly_stats`/`daily_stats_v2`, early block vs late block |
| **E5 via gesture variety / position entropy** | ⚠️ partial | only Feb 23–25 vs Mar 15–17 (the two `light_behavior` windows) |
| Personality drift (E3) | ✅ already built | slider/override JSON (not DB-dependent) |
| Per-person dwell sessions | ❌ | `person_sessions` empty in both DBs |
| "162 meta-reviews" count | ❌ | `meta_tuning_reviews` empty (0 / 1 row) — **don't cite this number** |
| Raw position heatmaps | ⚠️ | `tracking_events` only last ~3 days each DB (48h prune) |

## 5b. CALIBRATION ARTIFACT — Jan 29–Feb 2 active counts are invalid

An uncorrected tracking offset early in the run mapped nearly every pedestrian into the
**active** zone. The active-zone share confirms it unambiguously:

| Window | Active share | Active / Passive |
|---|---|---|
| **Jan 29 – Feb 2** (offset uncorrected) | **97.1%** | 1,812,666 / 53,186 |
| **Feb 13 – Mar 17** (after Feb-12 fix) | **3.6%** | 338,411 / 9,043,709 |

A 97% active share is physically impossible for this site (the alcove is a minority
destination off a busy sidewalk). The offset shifted tracked positions into the active band
(Z 78–283); after the **Feb-12 "fixed corrupt database" / calibration fix**, the split
becomes realistic (~1–12% active). Corroboration: late-run raw `z` averages ~405–427 cm —
correctly *out on the sidewalk*, well beyond the active zone — exactly what the early offset
would have pulled inward.

**Consequences:**
- **Exclude Jan 29–Feb 2 from all active-engagement metrics.** They look like a flood of
  engagement but are a calibration error; including them overstates active-zone entries and
  would corrupt any before/after learning comparison.
- The valid active-engagement window is **Feb 12/13 onward**.
- Passive counts and total-event counts from the early days are less affected (people were
  still detected, just mis-zoned), but the **zone label** for Jan 29–Feb 2 is unreliable.

**Separate bug (Feb 23–24):** raw `z` max reaches 2,667,803 cm and 392,804 cm — impossible
projection blow-ups from pre-V5/V6 calibration/fusion. Zone logic filters them, but they
poison raw-position averages for those days. Prefer aggregated active/passive *ratios* over
raw coordinate stats for any pre-V6 date.

## 5c. Estimating the early active/passive split (and the alignment figure)

The Jan 29–Feb 12 active/passive labels are unrecoverable (raw positions long pruned), so
they are **re-estimated** and clearly flagged, not left as the bad 97%:

- **Method.** For each early hour, take its total detections and re-split them using the
  measured active share of the **same weekday + same hour** drawn from the valid post-fix
  window (Feb 13 – Mar 16). Fall back to same-hour, then global, when a cell is thin
  (<500 events). The measured global post-fix active share is **3.62%**.
- **Result.** Jan 29–Feb 2 active share drops from ~97% to a plausible **2.4–3.9%**,
  varying by weekday/hour. Stored in `merged_run.db` as **`hourly_stats_corrected`**, which
  keeps `active_orig` / `passive_orig` columns and an **`estimated`** flag (1 for
  date ≤ Feb 12, 0 otherwise). 117 hours flagged.
- **Status.** This is a model for chart continuity, **never a measurement** — always label
  it "estimated." Total event/people counts are unchanged; only the zone split is modelled.

**Figure `F1_software_data_timeline`** renders the key talking point: daily capture
(events/day, colour-coded measured / partial / estimated, with the Feb 3–9 gap) on a shared
date axis above the **software-version milestones** (V2 → V4 → DB-fix Feb 12 → V5 → V6 →
V6.5 → AWARE-in-data Mar 15). It makes visible that **what the database captured tracked
what the software could do** — modes/gestures enter the data as versions add them.

## 5d. Feb 3–9 gap RECOVERED from the daily-report archive

The 7-day gap (§3) existed only in the **database** — those days were pruned (48h raw
retention), but the **daily-report JSONs** generated at the time (`reports/daily/`, present
in the Desktop repo) preserve full **24-hour breakdowns** for every one of them. The
reports are a frozen snapshot of data the DB no longer holds — an unintended benefit of the
tiered-retention design (raw → hourly → daily report).

- **Ingested** into `merged_run.db` as table **`hourly_stats_filled`** (= `hourly_stats_corrected`
  + 168 report-derived hours, tagged `src='report'`, `estimated=1`). Active/passive split
  re-estimated by the §5c matched-share method, since the reports show `active=0`
  (early zone bug). Totals/flow come straight from the reports.
- **Result: continuous coverage Jan 29 → Mar 17, no gaps.** The recovered week adds
  **~3.0 M events** (~33K people). New run total: **~23.3 M tracking events.**

**Provenance of `hourly_stats_filled`:**

| src | days | hours | events |
|---|---|---|---|
| `early` (DB, Feb 12–25) | 14 | 310 | 2.47 M |
| `report` (Feb 3–9, recovered) | 7 | 168 | 3.04 M |
| `late` (DB, late-Jan + Feb 25–Mar 17) | 29 | 585 | 17.80 M |
| **total** | **48** | **1,063** | **23.3 M** |

This is itself a talking point: the system's own reporting pipeline became the backup that
recovered data its database had discarded. Figure `F1_software_data_timeline` now shows the
recovered week hatched, with no gap.

**Table to use for run-wide figures:** `hourly_stats_filled` (continuous, provenance-tagged,
artifact-corrected). Use `hourly_stats` for DB-only/measured-only views.

## 6. Honest framing going forward

- Lead scale claim: **"~20 million tracking events across the run"** is solid and now exact.
- The run is best described as **continuous from Feb 12 to Mar 17** (~34 days of dense data)
  plus a late-January opening block — not a single unbroken 54-day capture. The *install*
  ran ~54 days; the *data* has a documented gap. State whichever is true to the claim being
  made.
- The "before/after" story is real and well-supported: **early block (Feb 13–24, V4/V5)**
  vs **late block (Mar 5–16, V6.5)** — but note the *software differed* between them, so the
  change reflects **both** learning *and* version changes. That's not a flaw to hide; it is
  the actual story of an evolving live system, and worth stating plainly.

---

*Merged DB: `IO/analysis/merged_run.db`. Rebuild: see the merge script in session history
or re-`ATTACH` both source DBs. Originals on Desktop are never modified.*
