#!/usr/bin/env python3
"""Export merged_run.db -> static JSON for the web viewer (Tier 1).

Reads IO/analysis/merged_run.db and writes three files into
IO/analysis/web_data/:

  hourly.json  - continuous hourly series (provenance flags preserved)
  daily.json   - per-day rollup recomputed from hourly_stats_filled
  meta.json    - run totals, src breakdown, software timeline, gap/artifact notes

Idempotent. stdlib only. Run with the project venv:
    .venv/bin/python IO/analysis/export_web_json.py

See IO/analysis/WEB_DEPLOYMENT_PLAN.md for the full hosting plan.
"""
import argparse
import gzip
import json
import sqlite3
import subprocess
import sys
from datetime import date
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DB = SCRIPT_DIR / "merged_run.db"
DEFAULT_OUT = SCRIPT_DIR / "web_data"
REPO_ROOT = SCRIPT_DIR.parent.parent

STATIC_META = {
    "gap": {
        "feb3_feb9_recovered": True,
        "method": "daily-report archive (see DATA_TIMELINE_AND_MERGE.md \u00a75d)",
    },
    "calibration_artifact": {
        "pre_fix_window": "2026-01-29..2026-02-02",
        "fix_date": "2026-02-12",
        "method": "matched weekday+hour from post-fix window (\u00a75c)",
    },
    "software_timeline": [
        {"date": "2026-01-27", "label": "V2", "kind": "version"},
        {"date": "2026-02-11", "label": "V4", "kind": "version"},
        {"date": "2026-02-12", "label": "DB fix (corruption)", "kind": "milestone"},
        {"date": "2026-02-25", "label": "V5 (gestures, falloff)", "kind": "version"},
        {"date": "2026-03-02", "label": "V6", "kind": "version"},
        {"date": "2026-03-03", "label": "V6.5 (3 tiers)", "kind": "version"},
        {"date": "2026-03-04", "label": "V6.5c", "kind": "version"},
        {"date": "2026-03-15", "label": "AWARE mode in data", "kind": "milestone"},
    ],
    "vocabulary_first_seen": {
        "modes": {
            "idle": "2026-02-23",
            "flow": "2026-02-23",
            "engaged": "2026-02-23",
            "crowd": "2026-02-23",
            "aware": "2026-03-15",
        },
        "gestures": {
            "_earlier_gestures": "2026-02-23",
            "sweep": "2026-02-25",
            "playful": "2026-03-15",
            "focus": "2026-03-16",
        },
    },
}

HOURLY_COLS = [
    "date", "hour", "total_events", "unique_people", "active_count",
    "passive_count", "left_to_right", "right_to_left", "bloom_count",
    "dominant_mode", "avg_brightness", "avg_speed", "src", "estimated",
    "active_orig", "passive_orig",
]

HOURLY_KEY_MAP = {
    "total_events": "events",
    "unique_people": "people",
    "active_count": "active",
    "passive_count": "passive",
    "left_to_right": "ltr",
    "right_to_left": "rtl",
    "bloom_count": "blooms",
    "dominant_mode": "mode",
    "avg_brightness": "brightness",
    "avg_speed": "speed",
    "estimated": "est",
}


def round2(x):
    if x is None:
        return None
    return round(float(x), 2)


def export_hourly(con):
    cur = con.execute(
        "SELECT date, hour, total_events, unique_people, active_count, "
        "passive_count, left_to_right, right_to_left, bloom_count, "
        "dominant_mode, avg_brightness, avg_speed, src, estimated, "
        "active_orig, passive_orig "
        "FROM hourly_stats_filled ORDER BY date, hour"
    )
    rows = []
    for raw in cur.fetchall():
        rec = dict(zip(HOURLY_COLS, raw))
        out = {}
        for k, v in rec.items():
            out[HOURLY_KEY_MAP.get(k, k)] = v
        out["brightness"] = round2(out.get("brightness"))
        out["speed"] = round2(out.get("speed"))
        rows.append(out)
    return rows


def export_daily(con):
    sql = """
        WITH per_day AS (
            SELECT date,
                   SUM(total_events)  AS events,
                   SUM(unique_people) AS people_sum,
                   SUM(active_count)  AS active,
                   SUM(passive_count) AS passive,
                   SUM(bloom_count)   AS blooms,
                   SUM(left_to_right) AS ltr,
                   SUM(right_to_left) AS rtl,
                   COUNT(*)           AS hours,
                   SUM(estimated)     AS est_hours,
                   SUM(CASE WHEN src='report' THEN 1 ELSE 0 END) AS report_hours,
                   SUM(CASE WHEN src='early'  THEN 1 ELSE 0 END) AS early_hours,
                   SUM(CASE WHEN src='late'   THEN 1 ELSE 0 END) AS late_hours,
                   AVG(avg_brightness) AS brightness,
                   AVG(avg_speed)      AS speed
            FROM hourly_stats_filled GROUP BY date
        )
        SELECT date, events, people_sum, active, passive, blooms, ltr, rtl,
               hours, est_hours, report_hours, early_hours, late_hours,
               brightness, speed
        FROM per_day ORDER BY date
    """
    cur = con.execute(sql)
    rows = []
    for (d, events, people_sum, active, passive, blooms, ltr, rtl,
         hours, est_hours, report_hours, early_hours, late_hours,
         brightness, speed) in cur.fetchall():
        denom = (ltr or 0) + (rtl or 0)
        flow_balance = ((ltr - rtl) / denom) if denom else 0.0
        if ltr >= rtl:
            dominant_flow = "left_to_right" if ltr > rtl else "balanced"
        else:
            dominant_flow = "right_to_left"
        rows.append({
            "date": d,
            "events": events,
            "people_sum": people_sum,
            "active": active,
            "passive": passive,
            "blooms": blooms,
            "ltr": ltr,
            "rtl": rtl,
            "hours": hours,
            "est_hours": est_hours,
            "report_hours": report_hours,
            "early_hours": early_hours,
            "late_hours": late_hours,
            "brightness": round2(brightness),
            "speed": round2(speed),
            "flow_balance": round(flow_balance, 3),
            "dominant_flow": dominant_flow,
        })
    return rows


def export_meta(con, hourly, daily):
    (total_events,) = con.execute(
        "SELECT COALESCE(SUM(total_events),0) FROM hourly_stats_filled"
    ).fetchone()
    (total_blooms,) = con.execute(
        "SELECT COALESCE(SUM(bloom_count),0) FROM hourly_stats_filled"
    ).fetchone()
    (people_ceiling,) = con.execute(
        "SELECT COALESCE(SUM(unique_people),0) FROM hourly_stats_filled"
    ).fetchone()
    (est_hours_total,) = con.execute(
        "SELECT COALESCE(SUM(estimated),0) FROM hourly_stats_filled"
    ).fetchone()

    cur = con.execute(
        "SELECT src, COUNT(*), COUNT(DISTINCT date), COALESCE(SUM(total_events),0) "
        "FROM hourly_stats_filled GROUP BY src"
    )
    src_breakdown = {
        s: {"hours": h, "days": d, "events": e}
        for s, h, d, e in cur.fetchall()
    }

    (start,) = con.execute(
        "SELECT MIN(date) FROM hourly_stats_filled"
    ).fetchone()
    (end,) = con.execute(
        "SELECT MAX(date) FROM hourly_stats_filled"
    ).fetchone()
    (days_present,) = con.execute(
        "SELECT COUNT(DISTINCT date) FROM hourly_stats_filled"
    ).fetchone()
    (hours_present,) = con.execute(
        "SELECT COUNT(*) FROM hourly_stats_filled"
    ).fetchone()

    meta = {
        "run": {
            "start": start,
            "end": end,
            "days_present": days_present,
            "hours_present": hours_present,
            "total_events": total_events,
            "total_blooms": total_blooms,
            "unique_people_ceiling": people_ceiling,
            "unique_people_note": (
                "sum of per-hour unique counts over-counts repeat visitors; "
                "this is a ceiling, not the headline visitor figure "
                "(see DATA_TIMELINE_AND_MERGE.md \u00a72)"
            ),
        },
        "src_breakdown": src_breakdown,
        "estimated_hours_total": est_hours_total,
    }
    meta.update(STATIC_META)
    return meta


def write_json(path, obj, pretty=False):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            obj, f,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            ensure_ascii=False,
            sort_keys=True,
        )


def gzipped_size(path):
    with open(path, "rb") as f:
        return len(gzip.compress(f.read()))


def fmt_bytes(n):
    if n >= 1024:
        return f"{n/1024:.1f} KB"
    return f"{n} B"


def self_test(hourly_path, daily_path, meta_path, hourly, daily, meta):
    failures = []
    h_text = hourly_path.read_text(encoding="utf-8")
    d_text = daily_path.read_text(encoding="utf-8")
    m_text = meta_path.read_text(encoding="utf-8")

    h_parsed = json.loads(h_text)
    d_parsed = json.loads(d_text)
    m_parsed = json.loads(m_text)

    if len(h_parsed) != 1063:
        failures.append(f"hourly.json row count {len(h_parsed)} != 1063")
    if len(d_parsed) != 48:
        failures.append(f"daily.json row count {len(d_parsed)} != 48")
    total = sum(r["events"] for r in h_parsed)
    if not (23_000_000 <= total <= 24_000_000):
        failures.append(f"hourly total events {total} outside expected range")
    for key in ("run", "src_breakdown", "gap", "calibration_artifact",
                "software_timeline", "vocabulary_first_seen"):
        if key not in m_parsed:
            failures.append(f"meta.json missing key: {key}")
    for r in h_parsed[:5] + h_parsed[-5:]:
        for k in ("date", "hour", "events", "people", "active", "passive",
                  "ltr", "rtl", "mode", "src", "est", "active_orig",
                  "passive_orig"):
            if k not in r:
                failures.append(f"hourly row missing key: {k}")
                break
    return failures


def auto_commit(out_dir, message):
    try:
        subprocess.run(
            ["git", "add", str(out_dir.relative_to(REPO_ROOT))],
            check=True, cwd=str(REPO_ROOT),
        )
    except subprocess.CalledProcessError as e:
        print(f"git add failed: {e}", file=sys.stderr)
        return False
    diff = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=str(REPO_ROOT),
    )
    if diff.returncode == 0:
        print("no changes to commit")
        return True
    try:
        subprocess.run(
            ["git", "commit", "-m", message],
            check=True, cwd=str(REPO_ROOT),
        )
    except subprocess.CalledProcessError as e:
        print(f"git commit failed: {e}", file=sys.stderr)
        return False
    print(f"committed: {message}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--pretty", action="store_true",
                    help="write human-readable JSON (larger files)")
    ap.add_argument("--no-commit", action="store_true",
                    help="write JSON but skip auto git commit")
    ap.add_argument("--skip-test", action="store_true",
                    help="skip the post-write self-test")
    args = ap.parse_args()

    if not args.db.exists():
        print(f"DB not found: {args.db}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(str(args.db))
    con.row_factory = None
    try:
        hourly = export_hourly(con)
        daily = export_daily(con)
        meta = export_meta(con, hourly, daily)
    finally:
        con.close()

    hourly_path = args.out / "hourly.json"
    daily_path = args.out / "daily.json"
    meta_path = args.out / "meta.json"
    write_json(hourly_path, hourly, pretty=args.pretty)
    write_json(daily_path, daily, pretty=args.pretty)
    write_json(meta_path, meta, pretty=args.pretty)

    h_size = hourly_path.stat().st_size
    d_size = daily_path.stat().st_size
    m_size = meta_path.stat().st_size
    print(f"hourly.json  {len(hourly):>5} rows  {fmt_bytes(h_size):>9}  "
          f"(gzip est {fmt_bytes(gzipped_size(hourly_path))})")
    print(f"daily.json   {len(daily):>5} rows  {fmt_bytes(d_size):>9}  "
          f"(gzip est {fmt_bytes(gzipped_size(daily_path))})")
    print(f"meta.json    {1:>5} obj   {fmt_bytes(m_size):>9}  "
          f"(gzip est {fmt_bytes(gzipped_size(meta_path))})")
    print(f"total on disk: {fmt_bytes(h_size + d_size + m_size)}")

    failures = []
    if not args.skip_test:
        failures = self_test(hourly_path, daily_path, meta_path,
                             hourly, daily, meta)
    if failures:
        print("SELF-TEST FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    if not args.skip_test:
        print("self-test: ok")

    if not args.no_commit:
        today = date.today().isoformat()
        events = meta["run"]["total_events"]
        hours = meta["run"]["hours_present"]
        msg = f"web_data: regenerate from merged_run.db ({today}, {hours} hours, {events} events)"
        if not auto_commit(args.out, msg):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
