#!/usr/bin/env python3
"""
H-series — fresh analysis of the DropCeiling run data.

Builds on what g_data_prep.py extracted for the G-series (the "fresh findings"
figures) but goes well beyond it: pulls from the daily-report JSONs (not just
the merged DB), the per-cycle behavior_adjustments JSON, the surviving
meta_tuning_reviews row, and the hourly_stats_filled table (48 days of
continuous coverage, not just the sparse light_behavior windows).

The 10 H-series figures (H1–H10) are designed to answer questions the
DROPCEILING_FINDINGS.md did not. See IO/DROPCEILING_FINDINGS_H.md for the
narrative.

Output: IO/analysis/h_data.json (all extracted data, one row per figure).
       IO/diagrams/H1..H10.{svg,png} (the figures themselves).

Run from repo root:  .venv/bin/python IO/analysis/h_data_prep.py
"""
import os, sys, json, math, datetime as dt, sqlite3, glob, statistics
from collections import defaultdict, Counter, OrderedDict
import numpy as np

# Paths
HERE   = os.path.dirname(os.path.abspath(__file__))         # IO/analysis/
IO     = os.path.dirname(HERE)                              # IO/
REPO   = os.path.dirname(IO)                                # repo root
MERGED = os.path.join(IO, "analysis", "merged_run.db")
EARLY  = os.path.join(REPO, "IO", "tracking_history.db")
LATE   = os.path.join(REPO, "tracking_history.db")
REPORTS= os.path.join(IO, "reports", "daily")
OUT    = os.path.join(HERE, "h_data.json")

# -------- helpers --------
def q(sql, db_path, params=()):
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.execute(sql, params)
    rows = cur.fetchall()
    con.close()
    return [dict(r) for r in rows]

def load_reports():
    out = OrderedDict()
    for p in sorted(glob.glob(os.path.join(REPORTS, "*.json"))):
        if p.endswith("_index.json"): continue
        d = json.load(open(p))
        if "date" not in d: continue
        out[d["date"]] = d
    return out

# -------- H1: tuner cycle-level trajectory of the 12 meta-params --------
# Use the per-cycle old_values/new_values from behavior_adjustments.adjustments_json
# across the full 217K-row span (Feb 11 – Mar 2).  G1 only used one value/day.
def h1_tuner_cycles():
    rows = q("SELECT timestamp, datetime, adjustments_json, aggression_level, "
             "short_activity, medium_activity, long_activity, energy_level "
             "FROM behavior_adjustments ORDER BY timestamp", MERGED)
    out = []
    for r in rows:
        try:
            aj = json.loads(r["adjustments_json"])
        except Exception:
            continue
        nv = aj.get("new_values")
        if not nv or not isinstance(nv, dict):
            continue
        out.append({
            "ts": r["timestamp"],
            "dt": r["datetime"],
            "aggression": r["aggression_level"],
            "energy": r["energy_level"],
            "short_activity": r["short_activity"],
            "medium_activity": r["medium_activity"],
            "long_activity": r["long_activity"],
            "values": nv,                         # all 12 params
            "budget_after": aj.get("budget_after"),
        })
    return out

# -------- H2: per-param "regime-conditional" deltas --------
# When aggression is high / activity is rush, which params move, and in which
# direction?  This is the empirical reading of the friction stack + the
# autotune's regime logic.
def h2_regime_deltas(cycles):
    # 4 regimes, 12 params, mean delta = (new - old) per cycle
    def regime(c):
        sa = c.get("short_activity") or 0
        ma = c.get("medium_activity") or 0
        if sa < 0.02 and ma < 0.05: return "dead"
        if sa < 0.10: return "trickle"
        if sa < 0.30: return "steady"
        return "rush"
    out = {reg: defaultdict(list) for reg in ["dead","trickle","steady","rush"]}
    # We don't have old_values here, so we need to compute delta between
    # consecutive cycles.  Re-read for old_values.
    rows = q("SELECT timestamp, adjustments_json, short_activity, medium_activity "
             "FROM behavior_adjustments ORDER BY timestamp", MERGED)
    prev = None
    for r in rows:
        try:
            aj = json.loads(r["adjustments_json"])
            ov = aj.get("old_values")
            nv = aj.get("new_values")
        except Exception:
            prev = None; continue
        if not (isinstance(ov, dict) and isinstance(nv, dict)):
            prev = None; continue
        if prev is not None:
            reg = regime(r)
            deltas = {p: nv.get(p, 0) - prev.get(p, 0) for p in ov}
            for p, d in deltas.items():
                out[reg][p].append(d)
        prev = nv
    # summarise
    summary = {}
    for reg, params in out.items():
        summary[reg] = {p: {"n": len(v), "mean": float(np.mean(v)) if v else 0,
                            "std": float(np.std(v)) if v else 0}
                        for p, v in params.items()}
    return summary

# -------- H3: gesture → target spatial bias (V5 and V6.5 windows) --------
# For each gesture, compute the mean target_x / target_z — where does the
# light "go" when it performs that gesture?  The light follows people
# during engagement, so this tells us how each gesture spatially relates
# to the tracked people.
def h3_gesture_spatial():
    out = {}
    for label, sql in [("v5", "SELECT gesture_type, position_x, position_y, position_z, "
                                  "target_x, target_y, target_z, mode "
                                  "FROM light_behavior WHERE datetime LIKE '2026-02-23%' OR "
                                  "datetime LIKE '2026-02-24%' OR datetime LIKE '2026-02-25%'"),
                       ("v65", "SELECT gesture_type, position_x, position_y, position_z, "
                                "target_x, target_y, target_z, mode "
                                "FROM light_behavior WHERE datetime LIKE '2026-03-15%' OR "
                                "datetime LIKE '2026-03-16%' OR datetime LIKE '2026-03-17%'")]:
        rows = q(sql, MERGED)
        by_gest = defaultdict(list)
        for r in rows:
            if r["gesture_type"] in (None, "", "none", "NONE"): continue
            by_gest[r["gesture_type"]].append({
                "px": r["position_x"], "py": r["position_y"], "pz": r["position_z"],
                "tx": r["target_x"],   "ty": r["target_y"],   "tz": r["target_z"],
                "mode": r["mode"],
            })
        for g, recs in list(by_gest.items()):
            if len(recs) < 50: del by_gest[g]
        out[label] = {g: {"n": len(recs),
                          "px_mean": float(np.mean([r["px"] for r in recs])),
                          "pz_mean": float(np.mean([r["pz"] for r in recs])),
                          "tx_mean": float(np.mean([r["tx"] for r in recs])),
                          "tz_mean": float(np.mean([r["tz"] for r in recs])),
                          "tx_std":  float(np.std([r["tx"] for r in recs])),
                          "tz_std":  float(np.std([r["tz"] for r in recs]))}
                     for g, recs in by_gest.items()}
    return out

# -------- H4: engagement episodes by hour-of-day + by day-of-week --------
# The G3 figure is a single histogram.  We add two cuts.
def h4_episodes_by_time():
    rows = q("SELECT timestamp, datetime, active_count FROM light_behavior "
             "WHERE date(timestamp,'unixepoch','localtime')>='2026-03-15' "
             "ORDER BY timestamp", MERGED)
    episodes = []
    start = None
    start_dt = None
    last = None
    for r in rows:
        ts = r["timestamp"]
        a = r["active_count"] or 0
        try:
            d = dt.datetime.fromisoformat(r["datetime"])
        except Exception:
            d = None
        if a > 0 and start is None:
            start = ts
            start_dt = d
        elif a == 0 and start is not None:
            dur = round(last - start, 1)
            if 0 <= dur < 600 and start_dt is not None:
                episodes.append({"start_dt": start_dt.isoformat(),
                                 "hour": start_dt.hour,
                                 "weekday": start_dt.weekday(),
                                 "duration": dur,
                                 "phase": ("walk" if dur < 0.5 else
                                           "notice" if dur < 3 else
                                           "greet" if dur < 10 else
                                           "engage" if dur < 30 else "bond")})
            start = None
            start_dt = None
        last = ts
    return episodes

# -------- H5: meta-review self-diagnosis raw data (the one surviving row) --------
def h5_meta_review():
    rows = q("SELECT * FROM meta_tuning_reviews", EARLY)
    if not rows: return {}
    r = rows[0]
    out = {
        "timestamp": r["timestamp"], "datetime": r["datetime"],
        "review_window_hours": r["review_window_hours"],
        "total_adjustments": r["total_adjustments"],
        "total_tracking_events": r["total_tracking_events"],
        "unique_people": r["unique_people"],
        "avg_short_activity": r["avg_short_activity"],
        "median_short_activity": r["median_short_activity"],
        "avg_medium_activity": r["avg_medium_activity"],
        "avg_energy_level": r["avg_energy_level"],
        "pct_at_floor": json.loads(r["pct_at_floor_json"] or "{}"),
        "pct_at_ceiling": json.loads(r["pct_at_ceiling_json"] or "{}"),
        "mode_distribution": json.loads(r["mode_distribution_json"] or "{}"),
        "param_stats": json.loads(r["param_stats_json"] or "{}"),
        "old_config": json.loads(r["old_config_json"] or "{}"),
        "new_config": json.loads(r["new_config_json"] or "{}"),
        "changes_summary": r["changes_summary"],
        "diagnosis": r["diagnosis"],
        "recommendations": json.loads(r["recommendations_json"] or "[]"),
    }
    return out

# -------- H6: 48-day mode-time signature from the daily reports --------
# Each daily report has a 24-hour breakdown; the prior G6 just lumped mode
# counts.  This is the full 48-day picture of when the light is engaged /
# aware / idle — and the "aware" mode only exists from Mar 15, so the
# picture has a hard cut.
def h6_mode_signature(reports):
    out = []
    for date, r in reports.items():
        hourly = r.get("hourly_trends") or []
        # mode counts: by looking up dominant_mode per hour
        counts = Counter()
        for h in hourly:
            m = h.get("dominant_mode") or "unknown"
            if m == "unknown": continue
            counts[m] += 1
        if not counts: continue
        out.append({"date": date, "n_hours": len(hourly),
                    "mode_hours": dict(counts)})
    return out

# -------- H7: 48-day brightness ladder, by date and by hour --------
# The daily report has summary.avg_brightness; the hourly_stats_filled table
# has hourly avg_brightness.  This is the "how bright was the light across
# the whole run" figure.
def h7_brightness_run():
    rows = q("SELECT date, hour, avg_brightness, total_events, unique_people, dominant_mode "
             "FROM hourly_stats_filled ORDER BY date, hour", MERGED)
    return rows

# -------- H8: position wander pattern per mode (V6.5 only) --------
# Plot the light's (x, z) per mode — a topographic map of where the light
# "lives" in each mode.  This is the spatial signature of the modes.
def h8_wander_per_mode():
    rows = q("SELECT mode, position_x, position_y, position_z "
             "FROM light_behavior WHERE datetime >= '2026-03-15' "
             "AND position_x IS NOT NULL AND position_z IS NOT NULL", MERGED)
    out = defaultdict(list)
    for r in rows:
        if r["mode"] in (None, "", "unknown"): continue
        out[r["mode"]].append([r["position_x"], r["position_z"]])
    return {k: v for k, v in out.items() if len(v) > 100}

# -------- H9: param correlations (autotune_daily_learnings.param_journeys) --------
# 33 days, 12 params, each with net_change.  Which params tend to move
# together across days?  This is the inter-parameter correlation of
# "personality drift."
def h9_param_correlations():
    rows = q("SELECT date, param_journeys_json FROM autotune_daily_learnings "
             "WHERE param_journeys_json IS NOT NULL", EARLY) + \
           q("SELECT date, param_journeys_json FROM autotune_daily_learnings "
             "WHERE param_journeys_json IS NOT NULL", LATE)
    series = OrderedDict()  # param -> [net_change_day1, day2, ...]
    for r in rows:
        try:
            pj = json.loads(r["param_journeys_json"])
        except Exception:
            continue
        for p, info in pj.items():
            series.setdefault(p, []).append(info.get("net_change", 0))
    if not series: return {}
    params = list(series.keys())
    corr = {}
    for a in params:
        corr[a] = {}
        for b in params:
            x = series[a]; y = series[b]
            if len(x) != len(y) or len(x) < 5: continue
            xm = np.mean(x); ym = np.mean(y)
            num = sum((xi-xm)*(yi-ym) for xi, yi in zip(x, y))
            den = math.sqrt(sum((xi-xm)**2 for xi in x) * sum((yi-ym)**2 for yi in y))
            corr[a][b] = float(num/den) if den else 0
    return {"days": len(next(iter(series.values()))), "params": params, "corr": corr,
            "series": series}

# -------- H10: per-day mode entropy across the 48-day run --------
# For each daily report, the `light_behavior.mode_distribution` field gives
# fractions per mode.  Shannon entropy of that distribution tells us the
# "behavioural richness" each day.  Days with all-idle → 0 entropy; days
# mixing all 5 modes → high entropy.
def h10_mode_entropy(reports):
    out = []
    for date, r in reports.items():
        mb = (r.get("light_behavior") or {}).get("mode_distribution") or {}
        # Filter out "unknown" and non-positive
        probs = [v for k, v in mb.items() if k not in ("unknown",) and v > 0]
        if not probs:
            continue
        s = sum(probs)
        if s <= 0: continue
        probs = [p/s for p in probs]
        H = -sum(p * math.log(p) for p in probs)
        # normalize by log(5) (5 modes)
        Hn = H / math.log(5) if math.log(5) > 0 else 0
        out.append({"date": date, "H": H, "Hn": Hn,
                    "n_modes": len(probs), "modes": mb})
    return out

# -------- run --------
def main():
    print("loading reports...")
    reports = load_reports()
    print(f"  {len(reports)} daily reports")

    print("H1 tuner cycles...")
    cycles = h1_tuner_cycles()

    print("H2 regime deltas...")
    h2 = h2_regime_deltas(cycles)

    print("H3 gesture spatial...")
    h3 = h3_gesture_spatial()

    print("H4 episodes by time...")
    h4 = h4_episodes_by_time()

    print("H5 meta-review...")
    h5 = h5_meta_review()

    print("H6 mode signature...")
    h6 = h6_mode_signature(reports)

    print("H7 brightness run...")
    h7 = h7_brightness_run()

    print("H8 wander per mode...")
    h8 = h8_wander_per_mode()

    print("H9 param correlations...")
    h9 = h9_param_correlations()

    print("H10 mode entropy...")
    h10 = h10_mode_entropy(reports)

    out = {
        "_meta": {
            "generated": dt.datetime.now().isoformat(),
            "n_reports": len(reports),
            "n_cycles": len(cycles),
            "n_episodes": len(h4),
        },
        "h1_cycles": cycles,
        "h2_regime_deltas": h2,
        "h3_gesture_spatial": h3,
        "h4_episodes": h4,
        "h5_meta_review": h5,
        "h6_mode_signature": h6,
        "h7_brightness": h7,
        "h8_wander": h8,
        "h9_param_corr": h9,
        "h10_mode_entropy": h10,
    }
    json.dump(out, open(OUT, "w"), indent=2, default=str)
    print(f"wrote {OUT}")
    print(f"  H1 cycles: {len(cycles)}")
    print(f"  H2 regimes: {list(h2.keys())}")
    print(f"  H3 v5 gestures: {list(h3['v5'].keys())[:6]}")
    print(f"  H3 v65 gestures: {list(h3['v65'].keys())[:6]}")
    print(f"  H4 episodes: {len(h4)}")
    print(f"  H5 meta-review: {'yes' if h5 else 'no'}")
    print(f"  H6 mode days: {len(h6)}")
    print(f"  H7 brightness rows: {len(h7)}")
    print(f"  H8 modes: {list(h8.keys())}")
    print(f"  H9 params: {len(h9.get('params', []))}")
    print(f"  H10 entropy days: {len(h10)}")

if __name__ == "__main__":
    main()
