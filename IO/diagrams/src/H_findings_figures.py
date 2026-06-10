#!/usr/bin/env python3
"""
H-series — figures for DROPCEILING_FINDINGS_H.md, all from h_data.json.
House style: greyscale + one warm accent reserved for the light.
Run from diagrams/:  ../../.venv/bin/python src/H_findings_figures.py
"""
import os, json, datetime as dt, math
from collections import defaultdict, Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # diagrams/
IO   = os.path.dirname(HERE)                                            # IO/
D    = json.load(open(os.path.join(IO, "analysis", "h_data.json")))

plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
INK="#1a1a1a"; GREY="#9a9a9a"; MID="#6e6e6e"; FAINT="#d9d9d9"; WARM="#e8a23d"; DARKW="#c47f1d"

def save(fig, name):
    fig.savefig(os.path.join(HERE, f"{name}.svg"), bbox_inches="tight", transparent=True)
    fig.savefig(os.path.join(HERE, "png", f"{name}.png"), bbox_inches="tight", dpi=200, facecolor="white")
    plt.close(fig); print("rendered", name)

def foot(fig, text, x=0.07, y=0.015):
    fig.text(x, y, text, fontsize=8, color="#555555")

def warm_diverging(v, vmax):
    """Map v in [-vmax, vmax] to a warm-cold diverging colour."""
    if v >= 0:
        t = min(v / vmax, 1.0)
        return "#f4d4a0" if t < 0.5 else WARM if t < 0.85 else DARKW
    else:
        t = min(-v / vmax, 1.0)
        return "#cfcfcf" if t < 0.5 else GREY if t < 0.85 else "#5e5e5e"

# ============== H1 — regime-conditional param deltas (heatmap) ==============
def h1():
    h2 = D["h2_regime_deltas"]
    params = ["responsiveness","energy","attention_span","sociability","exploration","memory",
              "brightness_global","speed_global","pulse_global","follow_speed_global",
              "dwell_influence","idle_trend_weight"]
    regimes = ["rush","steady","trickle","dead"]  # top to bottom in plot
    M = np.array([[h2[r][p]["mean"] for p in params] for r in regimes])
    vmax = np.nanmax(np.abs(M))
    fig,ax = plt.subplots(figsize=(12.5,3.8)); fig.subplots_adjust(left=.07,right=.90,top=.80,bottom=.20)
    # Symmetric colour map centred on zero
    im = ax.imshow(M, aspect="auto", cmap="RdGy_r", vmin=-vmax*0.7, vmax=vmax*0.7)
    ax.set_xticks(range(len(params)))
    ax.set_xticklabels([p.replace("_global","").replace("_","\n") for p in params],
                       fontsize=8.5, rotation=0)
    ax.set_yticks(range(len(regimes)))
    ax.set_yticklabels([r.upper() for r in regimes], fontsize=10)
    for i in range(len(regimes)):
        for j in range(len(params)):
            v = M[i, j]
            ax.text(j, i, f"{v*1000:+.1f}", ha="center", va="center",
                    fontsize=7.5,
                    color=("white" if abs(v) > vmax*0.40 else INK))
    ax.set_title("H1   Tuner behaviour depends on regime — empirical param-delta heatmap (× 10⁻³)",
                 fontsize=13,fontweight="bold",color=INK,loc="left",pad=10)
    # divider between personality and output groups
    ax.axvline(5.5, color=INK, lw=1.2, zorder=5)
    ax.text(2.5, 4.35, "personality", ha="center", fontsize=9, fontweight="bold", color=MID)
    ax.text(8.5, 4.35, "output", ha="center", fontsize=9, fontweight="bold", color=MID)
    # Side panel: RUSH explainer
    ax.text(12.05, 3.0, "RUSH",
            ha="left", va="center", fontsize=12, fontweight="bold", color=DARKW)
    ax.text(12.05, 2.5, "when crowds arrive:",
            ha="left", va="center", fontsize=8.5, color=INK)
    ax.text(12.05, 2.2, "+ responsiveness, energy,",
            ha="left", va="center", fontsize=8.5, color=DARKW, fontweight="bold")
    ax.text(12.05, 1.95, "  sociability, follow_speed",
            ha="left", va="center", fontsize=8.5, color=DARKW, fontweight="bold")
    ax.text(12.05, 1.65, "− brightness, speed, pulse,",
            ha="left", va="center", fontsize=8.5, color=INK)
    ax.text(12.05, 1.40, "  idle_trend_weight",
            ha="left", va="center", fontsize=8.5, color=INK)
    ax.set_ylim(-0.5, 3.8)
    ax.set_xlim(-0.5, 11.5)
    foot(fig, "Mean per-cycle Δparam, conditioned on activity regime, all 217K tuner cycles (Feb 11 – Mar 2). "
              "Each cell is a real (new − previous) per cycle, in thousandths. Red = pushed up, blue = pushed down. "
              "The signs are the story: rush regimes push responsiveness up and idle_trend_weight down; "
              "dead regimes drift around zero; trickle reverses most directions toward home.")
    save(fig, "H1_regime_deltas")

# ============== H2 — meta-review self-diagnosis Feb 13 ==============
def h2():
    m = D["h5_meta_review"]
    pf = m["pct_at_floor"]; pc = m["pct_at_ceiling"]
    params = sorted(set(list(pf.keys()) + list(pc.keys())))
    floors = [pf.get(p, 0) for p in params]
    ceilings = [pc.get(p, 0) for p in params]
    fig, ax = plt.subplots(figsize=(9.0, 3.6)); fig.subplots_adjust(left=.16,right=.97,top=.88,bottom=.18)
    y = np.arange(len(params))
    for i, (f, c) in enumerate(zip(floors, ceilings)):
        # Plot as a horizontal line from -c to f at row i
        if f > 0 or c > 0:
            ax.plot([0, f], [i, i], color=WARM, lw=6, solid_capstyle="butt", zorder=3)
        if c > 0:
            ax.plot([-c, 0], [i, i], color=GREY, lw=6, solid_capstyle="butt", zorder=3)
        if f > 5: ax.text(f+2, i, f"{f:.0f}%", va="center", fontsize=8.5, color=DARKW, fontweight="bold")
        if c > 5: ax.text(-c-2, i, f"{c:.0f}%", va="center", fontsize=8.5, color=INK, ha="right", fontweight="bold")
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y); ax.set_yticklabels(params, fontsize=9)
    ax.set_xlim(-110, 110)
    ax.set_xticks([-100,-50,0,50,100])
    ax.set_xticklabels(["100% ceiling","50%","0%","50% floor","100%"], fontsize=8.5, color=MID)
    ax.set_title("H2   The first self-diagnosis — Feb 13, 10:23, 8 h into the run",
                 fontsize=13,fontweight="bold",color=INK,loc="left",pad=10)
    ax.grid(axis="x", color="#eeeeee", lw=0.7); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    # legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0],[0], color=WARM, lw=6, label="% at floor (clamped below)"),
                       Line2D([0],[0], color=GREY, lw=6, label="% at ceiling (clamped above)")]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8.5, frameon=False)
    foot(fig, "The single surviving meta-tuning review (Feb 13, 8 h after the V4 tuner went live): every parameter was "
              "pinned against a wall. Memory was floor-clamped 100% of the time; 7 of 12 parameters were at their floor "
              "or ceiling over 96% of the time. The review's response: raise 4 homes, drop 3 ceilings, raise 5 floors, "
              "and tighten the budget. The fixes are visible in the same-week trajectory (G1).")
    save(fig, "H2_meta_review")

# ============== H3 — aggression per cycle, hour-of-day heatmap ==============
def h3():
    cycles = D["h1_cycles"]
    # aggression by hour, day
    grid = defaultdict(list)
    for c in cycles:
        try:
            d = dt.datetime.fromisoformat(c["dt"])
        except Exception: continue
        if c.get("aggression") is None: continue
        grid[(d.date().isoformat(), d.hour)].append(c["aggression"])
    days = sorted({k[0] for k in grid})
    hours = list(range(24))
    M = np.full((len(days), 24), np.nan)
    for i, day in enumerate(days):
        for h in hours:
            v = grid.get((day, h))
            if v: M[i, h] = float(np.mean(v))
    fig, ax = plt.subplots(figsize=(11.5, 4.4)); fig.subplots_adjust(left=.07,right=.97,top=.88,bottom=.20)
    im = ax.imshow(M, aspect="auto", cmap="YlOrBr", vmin=0, vmax=0.25, interpolation="nearest")
    ax.set_yticks(range(len(days)))
    ax.set_yticklabels([d[5:] for d in days], fontsize=8)
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=8)
    ax.set_xlabel("hour of day", fontsize=10)
    ax.set_title("H3   Aggression per tuner cycle — 217K rows, by day and hour",
                 fontsize=13,fontweight="bold",color=INK,loc="left",pad=10)
    # Day separators
    for d in ("2026-02-13", "2026-02-25", "2026-03-02"):
        if d in days: ax.axhline(days.index(d)-0.5, color=INK, lw=0.8, ls=(0,(3,2)))
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("mean aggression (0–1)", fontsize=8.5, color=MID)
    cbar.ax.tick_params(labelsize=8, colors=MID)
    foot(fig, "Per-cycle aggression from behavior_adjustments (the tuner's own decision log), 20 days of "
              "continuous tuner activity (Feb 11 – Mar 2). The 4 AM peak is the dark band on the left; "
              "the 12-16h valley is the light centre. The V5 step (Feb 25) is visible as a subtle cooling.")
    save(fig, "H3_aggression_heatmap")

# ============== H4 — 12-param net_change heatmap (factor structure) ==============
def h4():
    import sqlite3
    REPO = os.path.dirname(IO)
    # Build a per-day net_change matrix from the autotune_daily_learnings tables
    early = os.path.join(IO, "tracking_history.db")
    late  = os.path.join(REPO, "tracking_history.db")
    rows = []
    for p in (early, late):
        if not os.path.exists(p): continue
        con = sqlite3.connect(p)
        con.row_factory = sqlite3.Row
        rows += con.execute("SELECT date, param_journeys_json FROM autotune_daily_learnings "
                            "WHERE param_journeys_json IS NOT NULL").fetchall()
        con.close()
    by_day = {}
    for r in rows:
        try:
            pj = json.loads(r["param_journeys_json"])
        except Exception: continue
        for p, info in pj.items():
            by_day.setdefault(p, {})[r["date"]] = info.get("net_change", 0)
    order = ["brightness_global","speed_global","pulse_global",
             "responsiveness","energy","sociability","follow_speed_global","attention_span",
             "exploration","memory","dwell_influence","idle_trend_weight"]
    days = sorted({d for p in by_day for d in by_day[p]})
    M = np.zeros((len(order), len(days)))
    for i, p in enumerate(order):
        for j, d in enumerate(days):
            M[i, j] = by_day.get(p, {}).get(d, 0)
    vmax = max(abs(np.nanmin(M)), abs(np.nanmax(M)))
    fig, ax = plt.subplots(figsize=(11.5, 4.6)); fig.subplots_adjust(left=.10,right=.97,top=.88,bottom=.18)
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels([d[5:] for d in days], fontsize=8, rotation=0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([p.replace("_global","") for p in order], fontsize=9.5)
    # Cluster boxes
    ax.add_patch(plt.Rectangle((-0.5, -0.5), len(days), 3, fill=False, edgecolor=WARM, lw=2, zorder=4))
    ax.add_patch(plt.Rectangle((-0.5, 2.5), len(days), 5, fill=False, edgecolor=GREY, lw=2, zorder=4))
    ax.add_patch(plt.Rectangle((-0.5, 7.5), len(days), 5, fill=False, edgecolor=INK, lw=2, zorder=4))
    ax.text(len(days)-0.5, 1.5, "outputs", color=DARKW, fontsize=9, fontweight="bold",
            ha="right", va="center")
    ax.text(len(days)-0.5, 5.0, "personality", color=INK, fontsize=9, fontweight="bold",
            ha="right", va="center")
    ax.text(len(days)-0.5, 10.0, "trade-offs", color=INK, fontsize=9, fontweight="bold",
            ha="right", va="center")
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("net change that day", fontsize=8.5, color=MID)
    cbar.ax.tick_params(labelsize=8, colors=MID)
    ax.set_title("H4   Three clusters of meta-parameters drift together (and one trades off)",
                 fontsize=13,fontweight="bold",color=INK,loc="left",pad=10)
    foot(fig, "Per-day net change for the 12 meta-parameters, 33 daily snapshots. The warm box and grey box highlight the "
              "two positive-correlated clusters (outputs r=0.99, personality r=0.99) found by H9; the dark box highlights "
              "the trade-off axis (dwell_influence ↔ exploration r=−0.94). The 12-dial personality is a 3-factor system.")
    save(fig, "H4_param_clusters")

# ============== H5 — 48-day mode entropy timeline ==============
def h5():
    rows = D["h10_mode_entropy"]
    rows = [r for r in rows if r["date"] >= "2026-02-13"]  # exclude <Feb 13 where data is mostly empty
    days = [dt.date.fromisoformat(r["date"]) for r in rows]
    H = [r["Hn"] for r in rows]
    fig, ax = plt.subplots(figsize=(11.5, 4.0)); fig.subplots_adjust(left=.07,right=.97,top=.88,bottom=.18)
    ax.plot(days, H, color=INK, lw=1.6, marker="o", ms=4, zorder=3)
    ax.fill_between(days, H, color=WARM, alpha=0.18, zorder=2)
    # Mark V6 deployment (Mar 3) and the mode-5 appearance
    for d, lab in [(dt.date(2026,3,3),"V6 (Mar 3)"), (dt.date(2026,3,15),"AWARE in data (Mar 15)")]:
        ax.axvline(d, color=DARKW, lw=1.2, ls=(0,(4,2)), zorder=2)
        ax.annotate(lab, xy=(d, 0.95), xytext=(d, 0.95), fontsize=9, color=DARKW,
                    fontweight="bold", ha="center", annotation_clip=False)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("normalised mode entropy (0–1)", fontsize=10)
    ax.set_title("H5   Behavioural richness tripled when the vocabulary did (Mar 3)",
                 fontsize=13,fontweight="bold",color=INK,loc="left",pad=10)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.grid(axis="y", color="#eeeeee", lw=0.7); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    foot(fig, "Shannon entropy of the light's per-day mode distribution, normalised by log(5)=1.61. "
              "A day with only idle = 0. A day mixing all 5 modes evenly = 1.0. The jump on Mar 3 is the day "
              "the V6 software deployed with the AWARE mode in the code; the data starts to record it from Mar 15.")
    save(fig, "H5_mode_entropy")

# ============== H6 — engagement episodes, 3-day × 24-hour grid ==============
def h6():
    eps = D["h4_episodes"]
    days = ["2026-03-15","2026-03-16","2026-03-17"]
    # for each (day, hour, phase) count
    grid = defaultdict(lambda: defaultdict(int))
    for e in eps:
        d = e["start_dt"][:10]
        h = e["hour"]
        if d not in days: continue
        grid[d][h] += 1
    hours = list(range(24))
    M = np.zeros((len(days), 24))
    for i, d in enumerate(days):
        for h in hours:
            M[i, h] = grid[d].get(h, 0)
    fig, ax = plt.subplots(figsize=(11.5, 2.8)); fig.subplots_adjust(left=.08,right=.97,top=.80,bottom=.22)
    im = ax.imshow(M, aspect="auto", cmap="YlOrBr", vmin=0, interpolation="nearest")
    ax.set_xticks(range(24)); ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=8)
    ax.set_yticks(range(len(days))); ax.set_yticklabels([d[5:] for d in days], fontsize=9.5)
    for i in range(len(days)):
        for j in range(24):
            v = int(M[i, j])
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=7.5,
                        color=("white" if v > M.max()*0.4 else INK))
    ax.set_xlabel("hour of day", fontsize=10)
    ax.set_title("H6   Where the 758 episodes happened — 3 days, 24 hours",
                 fontsize=13,fontweight="bold",color=INK,loc="left",pad=10)
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("episodes started this hour", fontsize=8.5, color=MID)
    cbar.ax.tick_params(labelsize=8, colors=MID)
    foot(fig, "758 engagement episodes reconstructed from the light's own state log, distributed by start hour "
              "and date. The afternoon band (12:00–17:00) is where most episodes start. The 12 bond episodes "
              "(durations > 30 s) are concentrated in this band on the warmer days.")
    save(fig, "H6_episode_grid")

# ============== H7 — spatial footprint of each mode (V6.5) ==============
def h7():
    w = D["h8_wander"]
    modes = ["idle","flow","aware","engaged","crowd"]
    fig, ax = plt.subplots(figsize=(10.5, 5.6)); fig.subplots_adjust(left=.10,right=.97,top=.88,bottom=.14)
    # Panel zone: x in [-290, -30], z in [0, 28] roughly. Mark panels at z=0.
    ax.axvline(-290, color=INK, lw=1.5); ax.text(-292, 30, "panels", fontsize=9, color=INK, ha="right", fontweight="bold")
    # Draw a vertical line at z=0 (panel face) and z=+28 (active zone boundary)
    ax.axhline(0, color=INK, lw=0.5, ls=(0,(2,2)))
    # Place mode labels at the centroids with offsets; use a side panel for the legend
    cols = {"idle":GREY, "flow":"#bdbdbd", "aware":WARM, "engaged":DARKW, "crowd":"#5e5e5e"}
    # Manual label positions (axes-fractional) — push them apart, then leader to centroid
    label_positions_axes = {
        "idle":    (0.62, 0.55),  # mid-right, upper
        "flow":    (0.62, 0.40),  # mid-right, lower
        "aware":   (0.85, 0.65),  # far right, upper
        "engaged": (0.85, 0.30),  # far right, lower
        "crowd":   (0.40, 0.85),  # upper, left
    }
    for m in modes:
        if m not in w: continue
        arr = np.array(w[m])
        h2d, xedges, yedges = np.histogram2d(arr[:,0], arr[:,1], bins=[30, 24],
                                              range=[[-330, -20], [-30, 30]])
        X, Y = np.meshgrid((xedges[:-1]+xedges[1:])/2, (yedges[:-1]+yedges[1:])/2)
        h2d = h2d / h2d.max() if h2d.max() else 0
        ax.contourf(X, Y, h2d.T, levels=[0.05, 0.30, 0.65, 1.0],
                    colors=[cols[m]]*3, alpha=0.35, zorder=2)
        cx, cy = arr[:,0].mean(), arr[:,1].mean()
        ax.plot(cx, cy, "o", color=cols[m], ms=10, mec=INK, mew=0.8, zorder=5)
        fx, fy = label_positions_axes.get(m, (0.5, 0.5))
        ax.annotate(f"{m}\nμ=({cx:+.0f}, {cy:+.0f}) cm  n={len(arr):,}",
                    xy=(cx, cy), xytext=(fx, fy), textcoords="axes fraction",
                    fontsize=9, color=INK, fontweight="bold", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GREY, lw=0.5, alpha=0.9),
                    arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6))
    ax.set_xlim(-330, -20); ax.set_ylim(-32, 32)
    ax.set_xlabel("light position x (cm) — panels run along z=0 from x=−290 (left) to x=−30 (right)",
                  fontsize=9, color=MID)
    ax.set_ylabel("light position z (cm) — z=0 panel face, z>0 away from window",
                  fontsize=9, color=MID)
    ax.set_title("H7   Where the light lives — spatial footprint of each mode (V6.5)",
                 fontsize=13,fontweight="bold",color=INK,loc="left",pad=10)
    ax.set_aspect("equal")
    ax.grid(True, color="#eeeeee", lw=0.5); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    foot(fig, "Density of the light's recorded (x, z) position per behaviour mode across the V6.5 window "
              "(82K ticks, Mar 15–17). Idle lives at the panel centre (x=−159, z=−1); engaged and crowd pull forward "
              "(z=+8 and +14) and outward — the light physically approaches the street when it's with people.")
    save(fig, "H7_mode_footprints")

# ============== H8 — gesture spatial map ==============
def h8():
    h3 = D["h3_gesture_spatial"]["v65"]
    items = sorted(h3.items(), key=lambda x: -x[1]["n"])
    items = items[:8]
    fig, ax = plt.subplots(figsize=(10.5, 5.6)); fig.subplots_adjust(left=.10,right=.97,top=.88,bottom=.14)
    ax.axvline(-290, color=INK, lw=1.5); ax.text(-292, 30, "panels", fontsize=9, color=INK, ha="right", fontweight="bold")
    ax.axhline(0, color=INK, lw=0.5, ls=(0,(2,2)))
    colours = [WARM if g == "welcome" else (FAINT if g in ("thinking","curious","bored") else GREY)
               for g, _ in items]
    # Axes-fractional label positions (data coords → axes-fraction), hand-placed to avoid overlap
    # We map by sorted index (most-frequent first → top item)
    name_to_pos = {
        "sway":     (0.50, 0.88),
        "orbit":    (0.20, 0.80),
        "sweep":    (0.40, 0.92),
        "welcome":  (0.95, 0.50),
        "playful":  (0.30, 0.30),
        "thinking": (0.80, 0.20),
        "curious":  (0.55, 0.18),
        "bored":    (0.85, 0.08),
        "bloom":    (0.65, 0.05),
    }
    for (g, s), col in zip(items, colours):
        ax.plot(s["tx_mean"], s["tz_mean"], "o", color=col, ms=9, mec=INK, mew=0.6, zorder=4)
        fx, fy = name_to_pos.get(g, (0.5, 0.5))
        ax.annotate(f"{g}  n={s['n']}",
                    xy=(s["tx_mean"], s["tz_mean"]),
                    xytext=(fx, fy), textcoords="axes fraction",
                    fontsize=8.5, color=INK,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=GREY, lw=0.4, alpha=0.85),
                    arrowprops=dict(arrowstyle="-", color=GREY, lw=0.5))
        # Disc of std as amplitude
        circ = plt.Circle((s["tx_mean"], s["tz_mean"]), max(s["tx_std"], 5),
                          color=col, alpha=0.20, zorder=2)
        ax.add_patch(circ)
    ax.set_xlim(-330, -20); ax.set_ylim(-32, 32)
    ax.set_xlabel("gesture target x (cm) — panels run along z=0 from x=−290 (left) to x=−30 (right)",
                  fontsize=9, color=MID)
    ax.set_ylabel("gesture target z (cm) — z=0 panel face, z>0 away from window",
                  fontsize=9, color=MID)
    ax.set_title("H8   Where each gesture goes — 8 motion glyphs, V6.5 window",
                 fontsize=13,fontweight="bold",color=INK,loc="left",pad=10)
    ax.set_aspect("equal"); ax.grid(True, color="#eeeeee", lw=0.5); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    foot(fig, "Mean (x, z) target of the light during each gesture, V6.5 window. Sway, orbit, sweep pull the body "
              "outward (z=+18 to +22 cm); the others are postures performed near the panel face. "
              "The disc radius = 1 std in target_x (a rough gesture amplitude).")
    save(fig, "H8_gesture_spatial")

# ============== H9 — 48-day brightness timeline ==============
def h9():
    rows = D["h7_brightness"]
    from collections import defaultdict
    day_b = defaultdict(list)
    for r in rows:
        if r["avg_brightness"] is not None: day_b[r["date"]].append(r["avg_brightness"])
    days = sorted(day_b)
    means = [float(np.mean(day_b[d])) for d in days]
    x = [dt.date.fromisoformat(d) for d in days]
    fig, ax = plt.subplots(figsize=(11.5, 3.8)); fig.subplots_adjust(left=.07,right=.97,top=.88,bottom=.20)
    ax.fill_between(x, means, color=WARM, alpha=0.30, zorder=2, label="daily mean")
    ax.plot(x, means, color=INK, lw=1.4, zorder=3)
    # Annotate key moments with vertical lines and text ABOVE the data
    events = [(dt.date(2026,1,29), "light off\n(V2)"),
              (dt.date(2026,2,10), "first\ntraces"),
              (dt.date(2026,2,12), "DB fix"),
              (dt.date(2026,2,25), "V5"),
              (dt.date(2026,3,2), "V6"),
              (dt.date(2026,3,15), "AWARE\nin data")]
    day_dates = [dt.date.fromisoformat(x_) for x_ in days]
    for d, lab in events:
        if d in day_dates:
            ax.axvline(d, color=MID, lw=0.6, ls=(0,(3,2)), zorder=1)
            ax.annotate(lab, xy=(d, 130), xytext=(d, 165), fontsize=8, color=MID,
                        ha="center", arrowprops=dict(arrowstyle="-", color=GREY, lw=0.5))
    ax.set_ylim(0, 200)
    ax.set_ylabel("daily mean brightness (DMX units, 0–600 cap)", fontsize=10)
    ax.set_title("H9   The light's 48-day brightness curve — it took a week to come on",
                 fontsize=13,fontweight="bold",color=INK,loc="left",pad=10)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.grid(axis="y", color="#eeeeee", lw=0.7); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    foot(fig, "Daily mean of hourly avg_brightness from hourly_stats_filled, the 48-day continuous table. "
              "First 5 days (Jan 29 – Feb 2) = 0 DMX: the panels were off while the V2 code was being brought up. "
              "After the Feb 12 DB fix the curve settles into the 50–90 range. The shape: the light came on, "
              "stayed on, and never went off again. (One bright spike on Mar 2 = the day the V6 was being smoke-tested.)")
    save(fig, "H9_brightness_timeline")

# ============== H10 — per-param step distribution, by regime ==============
def h10():
    h2 = D["h2_regime_deltas"]
    params = ["responsiveness","energy","brightness_global","idle_trend_weight",
              "dwell_influence","exploration","follow_speed_global","sociability"]
    regimes = ["dead","trickle","steady","rush"]
    fig, axs = plt.subplots(2, 4, figsize=(11.5, 5.0), sharey=True)
    fig.subplots_adjust(left=.07, right=.97, top=.88, bottom=.12, wspace=0.30, hspace=0.55)
    for ax, p in zip(axs.flatten(), params):
        # We need raw step values, not just summary.  Re-derive from the data.
        # Skip if no data
        stds = [h2[r][p]["std"] for r in regimes]
        means = [h2[r][p]["mean"] for r in regimes]
        # Display as a bar of std with mean marker
        cols = [GREY, FAINT, "#c8c8c8", WARM]
        bars = ax.bar(regimes, stds, color=cols, edgecolor=INK, lw=0.5)
        ax.plot(regimes, means, "_", color=DARKW, ms=12, mew=2.5)
        ax.set_title(p.replace("_global",""), fontsize=9.5, fontweight="bold", color=INK, loc="left")
        ax.set_xticks(range(len(regimes)))
        ax.set_xticklabels([r[:3] for r in regimes], fontsize=8)
        ax.tick_params(axis="y", labelsize=8, colors=MID)
        if p == params[0]:
            ax.set_ylabel("|Δparam| std (per cycle)", fontsize=9, color=MID)
    fig.suptitle("H10   Step-size distribution per parameter and regime",
                 fontsize=13,fontweight="bold",color=INK,x=0.07,ha="left",y=0.98)
    # legend for mean
    fig.text(0.5, 0.01, "Bars = standard deviation of per-cycle Δparam (the spread).  Markers = mean Δparam. "
                          "Dead = no one; trickle = a few; steady = regular; rush = crowd.",
             ha="center", fontsize=8.5, color="#555555")
    save(fig, "H10_step_distribution")

for f in (h1, h2, h3, h4, h5, h6, h7, h8, h9, h10):
    try: f()
    except Exception as e:
        import traceback
        print(f"FAILED {f.__name__}: {e}")
        traceback.print_exc()
print("done.")
