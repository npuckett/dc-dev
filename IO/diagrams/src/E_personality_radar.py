#!/usr/bin/env python3
"""
Figure 3 — Personality radar: neutral home values vs as-deployed (54-day drift).
Greyscale structure + single warm accent reserved for the light's evolved state.

Values are real:
  home  -> autotune_overrides.json  (the resting/neutral starting personality)
  final -> slider_settings.json     (the as-deployed values after the run)

Run:  ../.venv/bin/python src/E_personality_radar.py
"""
import os, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # diagrams/
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "svg.fonttype": "none",
})

INK   = "#000000"
GREY  = "#9a9a9a"
LIGHT = "#d9d9d9"
WARM  = "#e8a23d"      # the single warm accent = the light's evolved personality
WARMF = "#e8a23d"

# 6 personality axes (0-1).  (label, home, deployed)
axes = [
    ("Responsiveness", 0.53, 0.83),
    ("Energy",         0.48, 0.77),
    ("Attention\nspan",0.50, 0.51),
    ("Sociability",    0.49, 0.44),
    ("Exploration",    0.40, 0.71),
    ("Memory",         0.35, 0.49),
]
labels = [a[0] for a in axes]
home   = [a[1] for a in axes]
final  = [a[2] for a in axes]

N = len(axes)
ang = [n / float(N) * 2 * math.pi for n in range(N)]
ang += ang[:1]
home_c  = home  + home[:1]
final_c = final + final[:1]

fig, ax = plt.subplots(figsize=(7.2, 7.2), subplot_kw=dict(polar=True))
ax.set_theta_offset(math.pi / 2)
ax.set_theta_direction(-1)

# rings / grid
ax.set_ylim(0, 1.0)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], color=GREY, fontsize=8)
ax.set_xticks(ang[:-1])
ax.set_xticklabels(labels, fontsize=12, color=INK)
ax.tick_params(axis="x", pad=14)
ax.grid(color="#cfcfcf", linewidth=0.8)
ax.spines["polar"].set_color("#cfcfcf")

# neutral / home: thin grey outline
ax.plot(ang, home_c, color=GREY, linewidth=1.6, linestyle=(0, (5, 3)), zorder=3)
ax.fill(ang, home_c, color=GREY, alpha=0.06, zorder=2)

# as-deployed: filled warm accent (the light's evolved body)
ax.plot(ang, final_c, color=WARM, linewidth=2.6, zorder=5)
ax.fill(ang, final_c, color=WARMF, alpha=0.22, zorder=4)

# dots + delta callouts on the big movers
for i, (lab, h, f) in enumerate(axes):
    a = ang[i]
    ax.plot(a, f, "o", color=WARM, markersize=6, zorder=6)
    ax.plot(a, h, "o", color=GREY, markersize=4, zorder=6)
    d = f - h
    if abs(d) >= 0.10:  # annotate only meaningful drift
        ax.annotate(f"+{d:.2f}" if d > 0 else f"{d:.2f}",
                    xy=(a, f), xytext=(a, min(f + 0.13, 1.02)),
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    color=INK, zorder=7)

# legend
from matplotlib.lines import Line2D
leg = [
    Line2D([0], [0], color=GREY, lw=1.6, linestyle=(0, (5, 3)), label="Neutral start (home values)"),
    Line2D([0], [0], color=WARM, lw=2.6, label="As-deployed after 54 days"),
]
ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.06),
          frameon=False, fontsize=10, ncol=2)

fig.suptitle("E3  Personality drift over the run",
             fontsize=14, fontweight="bold", color=INK, y=1.00)
fig.text(0.5, 0.955,
         "the six personality meta-parameters: where the light started vs where it settled",
         ha="center", fontsize=10, color="#555555")

fig.savefig(os.path.join(OUT, "E3_personality_radar.svg"), bbox_inches="tight", transparent=True)
fig.savefig(os.path.join(OUT, "png", "E3_personality_radar.png"), bbox_inches="tight", dpi=200, facecolor="white")
plt.close(fig)
print("rendered E3_personality_radar")
