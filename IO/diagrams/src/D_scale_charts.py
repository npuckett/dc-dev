#!/usr/bin/env python3
"""
Drop Ceiling — scale figures (D1, D2) to accompany the §15-0 estimate table.

Renders two SVGs + PNGs into diagrams/.  All values are the ESTIMATES from the
guide's "Run at a glance" table (extrapolated from 34 surviving daily reports).

Run with the project venv:  ../.venv/bin/python src/D_scale_charts.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # diagrams/
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": "#555555",
    "axes.linewidth": 0.8,
    "svg.fonttype": "none",
})

INK = "#2a2030"      # dark plum (matches RESIST nodes)
ACCENT = "#9db2ff"   # blue (process)
WARM = "#ffcf9d"     # warm (io)
GREEN = "#1f3a2a"    # result green
RED = "#3a1f1f"      # meta/force


def human(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M".replace(".0M", "M")
    if n >= 1_000:
        return f"{n/1_000:.0f}K"
    return str(int(n))


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.svg"), bbox_inches="tight", transparent=True)
    fig.savefig(os.path.join(OUT, "png", f"{name}.png"), bbox_inches="tight", dpi=200,
                facecolor="white")
    plt.close(fig)
    print("rendered", name)


# ---------------------------------------------------------------------------
# D1 — Run totals over 54 days (the headline counts), log-x horizontal bars
# ---------------------------------------------------------------------------
labels = [
    "Flow-tracker updates",
    "Tracking events",
    "Short-term evaluations",
    "Unique visitors",
    "Long-term meta-reviews",
    "Daily-learning cycles",
]
values = [3_100_000, 15_000_000, 580_000, 1_000_000, 162, 54]
colors = [ACCENT, WARM, ACCENT, "#fff39a", RED, RED]

# sort ascending for a clean ladder
order = sorted(range(len(values)), key=lambda i: values[i])
labels = [labels[i] for i in order]
values = [values[i] for i in order]
colors = [colors[i] for i in order]

fig, ax = plt.subplots(figsize=(8.2, 3.6))
bars = ax.barh(labels, values, color=colors, edgecolor=INK, linewidth=0.6)
ax.set_xscale("log")
ax.set_xlim(10, 5e7)
ax.set_xlabel("count over the 54-day run  (log scale)")
ax.set_title("D1  Drop Ceiling — run totals over 54 days, 24/7  (estimated)",
             fontsize=12, fontweight="bold", loc="left", color=INK)
for b, v in zip(bars, values):
    ax.text(v * 1.15, b.get_y() + b.get_height() / 2, human(v),
            va="center", ha="left", fontsize=10, fontweight="bold", color=INK)
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: human(x) if x >= 1000 else str(int(x))))
ax.grid(axis="x", color="#dddddd", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.text(0.01, -0.04,
         "Estimated from 34 surviving daily reports (stable, repeatable day/week patterns), "
         "scaled to 54 days. Rounded.",
         fontsize=8, color="#666666")
save(fig, "D1_run_totals")


# ---------------------------------------------------------------------------
# D2 — Fast vs slow: the adaptation cadence contrast (per day), log-y
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.6, 4.0))
names = ["Flow update\n(~1.5 s)", "Short-term eval\n(~8 s)",
         "Meta-review\n(3×/day)", "Daily learning\n(midnight)"]
perday = [57_600, 10_800, 3, 1]
bar_colors = [ACCENT, ACCENT, RED, RED]
bars = ax.bar(names, perday, color=bar_colors, edgecolor=INK, linewidth=0.6, width=0.62)
ax.set_yscale("log")
ax.set_ylim(0.5, 1.2e6)
ax.set_ylabel("evaluations per day  (log scale)")
ax.set_title("D2  Many fast adjustments, few slow reviews  (per day)",
             fontsize=12, fontweight="bold", loc="left", color=INK, pad=24)
for b, v in zip(bars, perday):
    ax.text(b.get_x() + b.get_width() / 2, v * 1.3, f"{v:,}",
            ha="center", va="bottom", fontsize=10, fontweight="bold", color=INK)
# annotate the two regimes (placed in the headroom above all bars)
ax.axvspan(-0.5, 1.5, color=ACCENT, alpha=0.07)
ax.axvspan(1.5, 3.5, color=RED, alpha=0.06)
ax.text(0.5, 6e5, "FAST  ·  reaction + anticipation", ha="center", fontsize=9,
        color="#3b4a7a", fontweight="bold")
ax.text(2.5, 6e5, "SLOW  ·  self-tuning", ha="center", fontsize=9,
        color="#7a3b3b", fontweight="bold")
ax.grid(axis="y", color="#dddddd", linewidth=0.6)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.text(0.01, -0.06,
         "The friction argument visualised: ~10^4–10^5 fast adjustments a day are governed "
         "by a handful of slow, deliberate reviews. Estimated.",
         fontsize=8, color="#666666")
save(fig, "D2_eval_cadence")

print("done.")
