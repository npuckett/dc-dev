#!/usr/bin/env python3
"""
F2 — The calibration artifact: active-zone share over the run.
Shows the Jan 29-Feb 2 offset (~97% active, impossible) collapsing to a
realistic ~3% after the Feb 12 fix, with the re-estimated values overlaid.
Greyscale + warm accent. Real data from hourly_stats_filled.

Run: ../.venv/bin/python src/F2_calibration_artifact.py
"""
import os, pickle, datetime as dt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

OUT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams.update({"font.family":"DejaVu Sans","svg.fonttype":"none"})
INK="#1a1a1a"; GREY="#9a9a9a"; FAINT="#cfcfcf"; WARM="#e8a23d"; BAD="#c0392b"

data=pickle.load(open("/tmp/f2data.pkl","rb"))   # (date, orig%, corr%, est, src)
dates=[dt.date.fromisoformat(d) for d,_,_,_,_ in data]
orig=[o for _,o,_,_,_ in data]
corr=[c for _,_,c,_,_ in data]
est =[e for _,_,_,e,_ in data]

fig,ax=plt.subplots(figsize=(12.0,4.6))
fig.subplots_adjust(left=0.07,right=0.985,top=0.84,bottom=0.16)

# original (as-logged) active share — the artifact, drawn as a stark line where it exists
ox=[d for d,o in zip(dates,orig) if o is not None]
oy=[o for o in orig if o is not None]
ax.plot(ox,oy,color=FAINT,lw=1.0,zorder=2)
# highlight the impossible early spike
early=[(d,o) for d,o,e,s in zip(dates,orig,est,[x[4] for x in data]) if o is not None and d<=dt.date(2026,2,2)]
ax.plot([d for d,_ in early],[o for _,o in early],"o-",color=BAD,lw=2.2,ms=5,zorder=5,
        label="as-logged (offset uncorrected)")

# corrected / estimated active share
cx=[d for d,c in zip(dates,corr) if c is not None]
cy=[c for c in corr if c is not None]
ax.plot(cx,cy,color=GREY,lw=1.4,zorder=3)
ax.plot([d for d,c,e in zip(dates,corr,est) if c is not None and e],
        [c for c,e in zip(corr,est) if c is not None and e],
        "o",color=WARM,ms=5,zorder=6,label="corrected / estimated split")
ax.plot([d for d,c,e in zip(dates,corr,est) if c is not None and not e],
        [c for c,e in zip(corr,est) if c is not None and not e],
        "o",color=INK,ms=3.5,zorder=6,label="measured split")

# the fix marker
ax.axvline(dt.date(2026,2,12),color=WARM,lw=1.6,zorder=4)
ax.annotate("Feb 12 — calibration / DB fix",xy=(dt.date(2026,2,12),85),
            xytext=(dt.date(2026,2,15),88),fontsize=9,color="#8a6320",fontweight="bold",va="center")

# realistic band
ax.axhspan(0,15,color=GREY,alpha=0.06,zorder=0)
ax.text(dates[-1],8,"realistic range\n(~1–12% active)",ha="right",va="center",fontsize=8,color=GREY,style="italic")

ax.set_ylim(0,100); ax.set_ylabel("active-zone share of detections (%)",fontsize=10,color=INK)
ax.set_title("F2   The calibration artifact: a 97% “active” spike that wasn’t real",
             fontsize=13.5,fontweight="bold",color=INK,loc="left",pad=10)
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.set_xlim(dt.date(2026,1,26),dt.date(2026,3,19))
ax.grid(axis="y",color="#eeeeee",lw=0.7); ax.set_axisbelow(True)
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.legend(loc="upper right",fontsize=8.5,frameon=False)

fig.text(0.07,0.02,
 "Before Feb 12 an uncorrected tracking offset mapped nearly every pedestrian into the active zone (red, ~95–98% — physically impossible "
 "for a sidewalk alcove). After the fix the split is realistic (~1–12% active). Pre-fix values are re-estimated by matched weekday+hour (warm); "
 "later values are measured (black).",
 fontsize=8,color="#555555")

fig.savefig(os.path.join(OUT,"F2_calibration_artifact.svg"),bbox_inches="tight",transparent=True)
fig.savefig(os.path.join(OUT,"png","F2_calibration_artifact.png"),bbox_inches="tight",dpi=200,facecolor="white")
plt.close(fig)
print("rendered F2_calibration_artifact")
