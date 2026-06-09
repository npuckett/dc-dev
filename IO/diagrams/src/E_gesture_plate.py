#!/usr/bin/env python3
"""
Figure 1 — Gesture-notation plate.
The light's movement vocabulary as a grid of motion glyphs.
Greyscale structure; the warm accent = the luminous body / its motion.

Amplitudes are real (light_behavior.py ENGAGED_GESTURE_AMPLITUDES).

Run:  ../.venv/bin/python src/E_gesture_plate.py
"""
import os, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Circle, Ellipse

OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})

INK="#1a1a1a"; GREY="#9a9a9a"; FAINT="#d4d4d4"; WARM="#e8a23d"

# Grid: 4 cols x 2 rows = 8 engaged gestures
gestures = [
    ("NOD",     "12 cm vertical bob",        "greet"),
    ("LEAN",    "15 cm forward + stretch",   "greet"),
    ("SWAY",    "18 cm lateral oscillation", "engage"),
    ("ORBIT",   "15/8 cm circle + rotate",   "engage"),
    ("SETTLE",  "draw in, tighten beam",     "bond"),
    ("BREATHE", "brightness +/-12%, radius", "all"),
    ("SWEEP",   "25 cm wide scan + stretch", "engage"),
    ("FOCUS",   "narrow beam, +brightness",  "bond"),
]

fig, axs = plt.subplots(2, 4, figsize=(11.2, 6.2))
fig.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.10, wspace=0.18, hspace=0.78)

def base_dot(ax):
    ax.plot(0, 0, "o", color=FAINT, markersize=9, zorder=1)  # rest position (ghost)

def arrow(ax, x0, y0, x1, y1, **kw):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                 mutation_scale=14, lw=2.4, color=WARM, zorder=5, **kw))

def draw(ax, name):
    ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.15, 1.15); ax.set_aspect("equal"); ax.axis("off")
    base_dot(ax)
    if name == "NOD":            # vertical bob
        arrow(ax, 0, -0.7, 0, 0.7); arrow(ax, 0, 0.7, 0, -0.7)
        ax.plot(0, 0, "o", color=WARM, markersize=11, zorder=6)
    elif name == "LEAN":         # forward (toward viewer) + depth stretch
        ax.add_patch(Ellipse((0,0), 1.7, 0.9, fc=WARM, ec="none", alpha=0.18, zorder=2))
        arrow(ax, -0.55, -0.35, 0.55, 0.35)
        ax.plot(0.55, 0.35, "o", color=WARM, markersize=11, zorder=6)
    elif name == "SWAY":         # lateral oscillation + horizontal stretch
        ax.add_patch(Ellipse((0,0), 2.0, 0.7, fc=WARM, ec="none", alpha=0.16, zorder=2))
        th=np.linspace(-math.pi,math.pi,120); ax.plot(0.85*np.sin(th), 0.18*np.cos(th)-0.0, color=WARM, lw=1.2, alpha=0.5, zorder=3)
        arrow(ax, -0.85, 0, 0.85, 0)
        ax.plot(0.85, 0, "o", color=WARM, markersize=11, zorder=6)
    elif name == "ORBIT":        # XY circle + rotation tick
        th=np.linspace(0, 1.85*math.pi, 100)
        ax.plot(0.7*np.cos(th), 0.55*np.sin(th), color=WARM, lw=2.4, zorder=5)
        ex,ey=0.7*math.cos(1.85*math.pi), 0.55*math.sin(1.85*math.pi)
        arrow(ax, 0.7*math.cos(1.7*math.pi), 0.55*math.sin(1.7*math.pi), ex, ey)
        ax.plot(0.7, 0, "o", color=WARM, markersize=10, zorder=6)
    elif name == "SETTLE":       # inward contracting rings
        for r,a in [(1.0,0.25),(0.66,0.45),(0.34,0.8)]:
            ax.add_patch(Circle((0,0), r, fill=False, ec=WARM, lw=2.0, alpha=a, zorder=4))
        for a in range(4):
            an=a*math.pi/2+math.pi/4
            arrow(ax, 0.9*math.cos(an),0.9*math.sin(an), 0.42*math.cos(an),0.42*math.sin(an))
        ax.plot(0,0,"o",color=WARM, markersize=12, zorder=6)
    elif name == "BREATHE":      # pulsing rings + waveform
        for r,a in [(1.05,0.2),(0.78,0.4),(0.5,0.7)]:
            ax.add_patch(Circle((0,0), r, fill=False, ec=WARM, lw=2.0, alpha=a, zorder=4))
        ax.plot(0,0,"o",color=WARM, markersize=12, zorder=6)
    elif name == "SWEEP":        # wide horizontal scan, stretched
        ax.add_patch(Ellipse((0,0), 2.3, 0.6, fc=WARM, ec="none", alpha=0.16, zorder=2))
        arrow(ax, -1.0, 0, 1.0, 0)
        th=np.linspace(-1,1,60); ax.plot(th, 0.28*(1-th**2), color=WARM, lw=1.0, alpha=0.5, zorder=3)
        ax.plot(1.0,0,"o",color=WARM, markersize=11, zorder=6)
    elif name == "FOCUS":        # narrowing crosshair + bright core
        ax.add_patch(Circle((0,0), 0.9, fill=False, ec=FAINT, lw=1.6, ls=(0,(3,3)), zorder=2))
        for a in range(4):
            an=a*math.pi/2
            arrow(ax, 0.85*math.cos(an),0.85*math.sin(an), 0.3*math.cos(an),0.3*math.sin(an))
        ax.add_patch(Circle((0,0), 0.2, fc=WARM, ec="none", zorder=6))
        ax.plot(0,0,"o",color="#fff4e0", markersize=6, zorder=7)

for ax, (name, desc, phase) in zip(axs.flat, gestures):
    draw(ax, name)
    ax.set_title(name, fontsize=13, fontweight="bold", color=INK, pad=6)
    ax.text(0.5, -0.16, desc, transform=ax.transAxes, ha="center", va="top",
            fontsize=8.5, color="#555555")
    ax.text(0.5, -0.30, f"dwell: {phase}", transform=ax.transAxes, ha="center", va="top",
            fontsize=7.5, color=GREY, style="italic")

fig.suptitle("E1  Gesture vocabulary of the luminous body",
             fontsize=15, fontweight="bold", color=INK, x=0.5, y=0.95)
fig.text(0.5, 0.885,
         "eight ongoing gestures layered on the light while a person is present; the grey dot marks rest, the warm mark the moving body",
         ha="center", fontsize=9.5, color="#555555")

fig.savefig(os.path.join(OUT, "E1_gesture_plate.svg"), bbox_inches="tight", transparent=True)
fig.savefig(os.path.join(OUT, "png", "E1_gesture_plate.png"), bbox_inches="tight", dpi=200, facecolor="white")
plt.close(fig)
print("rendered E1_gesture_plate")
