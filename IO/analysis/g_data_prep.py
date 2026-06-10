#!/usr/bin/env python3
"""
Extract all data for the G-series findings figures into analysis/g_data.json.
Sources: analysis/merged_run.db + the two source DBs (raw events, daily learnings).
Re-runnable. Run: python3 analysis/g_data_prep.py  (from IO/)
"""
import sqlite3, json, datetime as dt, statistics as st
from collections import defaultdict

OUT = {}
MERGED = "analysis/merged_run.db"
EARLY  = "/Users/npmac/Desktop/dc-dev/IO/tracking_history.db"
LATE   = "/Users/npmac/Desktop/dc-dev/tracking_history.db"

con = sqlite3.connect(MERGED)
q = lambda s, p=(): con.execute(s, p).fetchall()

# ---------------- G1 + G10: daily personality trajectories + diary ----------------
traj = {}
diary = {}
for db in (EARLY, LATE):
    try:
        c = sqlite3.connect(db)
        for date, ov, ss in c.execute(
                "SELECT date, optimal_values_json, strategy_summary FROM autotune_daily_learnings ORDER BY date"):
            if ov:
                v = json.loads(ov)
                if any(v.get(k, 0) for k in ('responsiveness', 'energy')):  # skip the V6 zero rows
                    traj[date] = v
            if ss:
                diary[date] = ss
        c.close()
    except Exception as e:
        print("learnings:", db, e)
OUT["g1_trajectory"] = {d: traj[d] for d in sorted(traj)}
OUT["g10_diary"] = {d: diary[d] for d in sorted(diary)}

# ---------------- G2: flow direction by hour (measured days) ----------------
rows = q("""SELECT hour, SUM(left_to_right), SUM(right_to_left)
            FROM hourly_stats_filled WHERE estimated=0 GROUP BY hour ORDER BY hour""")
OUT["g2_flow"] = [{"hour": h, "ltr": l or 0, "rtl": r or 0} for h, l, r in rows]

# ---------------- G3: engagement episodes (V6.5 window) ----------------
rows = q("""SELECT timestamp, active_count FROM light_behavior
            WHERE date(timestamp,'unixepoch','localtime')>='2026-03-15' ORDER BY timestamp""")
eps, start, last = [], None, None
for ts, a in rows:
    a = a or 0
    if a > 0 and start is None:
        start = ts
    elif a == 0 and start is not None:
        eps.append(round(last - start, 1)); start = None
    last = ts
OUT["g3_episodes"] = [e for e in eps if 0 <= e < 600]

# ---------------- G4: aggression by hour + people by hour ----------------
rows = q("""SELECT CAST(strftime('%H',timestamp,'unixepoch','localtime') AS INT) h,
                   AVG(aggression_level) FROM behavior_adjustments
            WHERE aggression_level IS NOT NULL GROUP BY h ORDER BY h""")
OUT["g4_aggression"] = [{"hour": h, "agg": round(a, 4)} for h, a in rows]
rows = q("""SELECT hour, AVG(unique_people) FROM hourly_stats_filled
            WHERE estimated=0 GROUP BY hour ORDER BY hour""")
OUT["g4_people"] = [{"hour": h, "people": round(p or 0, 1)} for h, p in rows]

# ---------------- G5: gesture economy V5 vs V6.5 ----------------
def gestures(lo, hi):
    rows = q(f"""SELECT gesture_type, COUNT(*) FROM light_behavior
                WHERE gesture_type IS NOT NULL AND gesture_type!=''
                AND date(timestamp,'unixepoch','localtime') BETWEEN '{lo}' AND '{hi}'
                GROUP BY gesture_type ORDER BY 2 DESC""")
    return {g: c for g, c in rows}
OUT["g5_gestures"] = {"v5": gestures('2026-02-23', '2026-02-25'),
                      "v65": gestures('2026-03-15', '2026-03-17')}

# ---------------- G6: mode time budget V5 vs V6.5 ----------------
def modes(lo, hi):
    rows = q(f"""SELECT mode, COUNT(*) FROM light_behavior
                WHERE date(timestamp,'unixepoch','localtime') BETWEEN '{lo}' AND '{hi}'
                GROUP BY mode""")
    return {m or "unknown": c for m, c in rows}
OUT["g6_modes"] = {"v5": modes('2026-02-23', '2026-02-25'),
                   "v65": modes('2026-03-15', '2026-03-17')}

# ---------------- G7: brightness ladder by mode (V6.5) ----------------
rows = q("""SELECT mode, AVG(brightness), MAX(brightness), COUNT(*) FROM light_behavior
            WHERE date(timestamp,'unixepoch','localtime')>='2026-03-15'
            AND mode IS NOT NULL GROUP BY mode""")
OUT["g7_brightness"] = [{"mode": m, "avg": round(a, 1), "max": round(x, 1), "n": n}
                        for m, a, x, n in rows]

# ---------------- G8: Z-occupancy from raw positions (late DB) ----------------
c = sqlite3.connect(LATE)
rows = c.execute("""SELECT CAST(z/25 AS INT)*25, COUNT(*) FROM tracking_events
                    WHERE z BETWEEN 0 AND 900 GROUP BY 1 ORDER BY 1""").fetchall()
c.close()
OUT["g8_zhist"] = [{"z": zb, "n": n} for zb, n in rows]

# ---------------- G9: budget regimes (sampled JSON parse) ----------------
rows = q("""SELECT timestamp, adjustments_json FROM behavior_adjustments
            WHERE adjustments_json IS NOT NULL AND adjustments_json!='' AND (rowid%20)=0""")
daily = defaultdict(lambda: dict(n=0, b=0.0, dep=0))
for ts, js in rows:
    try:
        j = json.loads(js)
    except Exception:
        continue
    d = dt.datetime.fromtimestamp(ts).date().isoformat()
    a = daily[d]; a["n"] += 1
    a["b"] += j.get("budget_before", 0) or 0
    if (j.get("budget_after", 0) or 0) < 1:
        a["dep"] += 1
OUT["g9_budget"] = [{"date": d, "avg_budget": round(v["b"]/v["n"], 1),
                     "pct_depleted": round(100*v["dep"]/v["n"], 1), "n": v["n"]}
                    for d, v in sorted(daily.items()) if v["n"] >= 10]

con.close()
with open("analysis/g_data.json", "w") as f:
    json.dump(OUT, f, indent=1)
print("wrote analysis/g_data.json")
for k, v in OUT.items():
    n = len(v) if hasattr(v, "__len__") else "?"
    print(f"  {k}: {n} entries")
