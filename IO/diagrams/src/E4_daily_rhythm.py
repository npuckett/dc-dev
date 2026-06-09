#!/usr/bin/env python3
"""
E4 — 24-hour rhythm strip: the body in time.
A single representative full V6.5 day (Tue Mar 10), real data from hourly_stats_filled.
Top: pedestrian activity (people/hr area) with the rush-hour peak marked.
Bottom: the behaviour MODE the light spent each hour in (idle/flow/aware/engaged).
Greyscale + warm accent for the light's most-engaged states.

Run: ../.venv/bin/python src/E4_daily_rhythm.py
"""
import os, pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams.update({"font.family":"DejaVu Sans","svg.fonttype":"none"})
INK="#1a1a1a"; GREY="#9a9a9a"; FAINT="#d9d9d9"; WARM="#e8a23d"; DARKW="#c47f1d"

d=pickle.load(open("/tmp/e4.pkl","rb"))
rows=d["rows"]
hrs=[r["hour"] for r in rows]
ppl=[r["people"] for r in rows]
modes=[r["mode"] for r in rows]

# mode -> greyscale ramp, with the active-engagement modes in warm
MODE_FILL={"idle":"#ededed","flow":"#bdbdbd","aware":WARM,"engaged":DARKW,
           "crowd":"#9a4f00","unknown":"#f4f4f4","?":"#f4f4f4"}
MODE_LABEL={"idle":"idle","flow":"flow","aware":"aware","engaged":"engaged",
            "crowd":"crowd","unknown":"—","?":"—"}

fig,(axA,axM)=plt.subplots(2,1,figsize=(12.2,5.8),height_ratios=[3.0,0.8],sharex=True)
fig.subplots_adjust(left=0.07,right=0.985,top=0.85,bottom=0.22,hspace=0.10)

# ---- TOP: people/hour area (the rhythm) ----
x=np.array(hrs); y=np.array(ppl)
axA.fill_between(x,y,color=FAINT,zorder=2)
axA.plot(x,y,color=GREY,lw=1.6,zorder=3)
# mark the rush-hour peak
pk=int(np.argmax(y))
axA.plot(x[pk],y[pk],"o",color=WARM,ms=8,zorder=5)
axA.annotate(f"rush-hour peak\n{y[pk]:,} people · {x[pk]:02d}:00",
             xy=(x[pk],y[pk]),xytext=(x[pk]-3.2,y[pk]*0.9),fontsize=9,fontweight="bold",
             color=DARKW,ha="right",va="top",
             arrowprops=dict(arrowstyle="-",color=WARM,lw=1.2))
# label the morning commute bump
mc=7
axA.annotate("morning\ncommute",xy=(mc,y[mc]),xytext=(mc-0.3,y[mc]+520),fontsize=8.5,
             color=GREY,ha="center",va="bottom",style="italic")
# overnight lull
axA.annotate("overnight lull",xy=(3,y[3]),xytext=(3,1400),fontsize=8.5,color=GREY,
             ha="center",style="italic")
axA.set_ylabel("people tracked / hour",fontsize=10,color=INK)
axA.set_ylim(0,max(y)*1.18)
axA.set_title("E4   One day in the life — Tuesday, 24 hours",
              fontsize=14,fontweight="bold",color=INK,loc="left",pad=10)
axA.grid(axis="y",color="#eeeeee",lw=0.7); axA.set_axisbelow(True)
for s in ("top","right"): axA.spines[s].set_visible(False)

# ---- BOTTOM: mode lane ----
for h,m in zip(hrs,modes):
    axM.bar(h,1,width=1.0,color=MODE_FILL.get(m,"#f0f0f0"),edgecolor="white",linewidth=0.6,zorder=2)
axM.set_ylim(0,1); axM.set_yticks([])
axM.set_xlim(-0.5,23.5)
axM.set_xticks(range(0,24,2))
axM.set_xticklabels([f"{h:02d}:00" for h in range(0,24,2)],fontsize=9,color=INK)
axM.set_xlabel("hour of day",fontsize=10,color=INK)
for s in ("top","right","left"): axM.spines[s].set_visible(False)
axM.text(-0.4,1.5,"light's dominant mode each hour",fontsize=8.5,color=GREY,style="italic",
         transform=axM.transData)

# mode legend
from matplotlib.patches import Patch
present=[m for m in ["idle","flow","aware","engaged"] if m in modes]
leg=[Patch(fc=MODE_FILL[m],ec="white",label=MODE_LABEL[m]) for m in present]
axM.legend(handles=leg,loc="upper center",bbox_to_anchor=(0.5,-1.15),ncol=len(leg),
           frameon=False,fontsize=9)

fig.text(0.07,0.02,
 "Real data, one representative day. The light idles through the night, follows the morning commute, drifts with midday flow, and rises to its "
 "busiest 'aware' and 'engaged' states across the afternoon rush — its behaviour breathing with the rhythm of the street.",
 fontsize=8,color="#555555")

fig.savefig(os.path.join(OUT,"E4_daily_rhythm.svg"),bbox_inches="tight",transparent=True)
fig.savefig(os.path.join(OUT,"png","E4_daily_rhythm.png"),bbox_inches="tight",dpi=200,facecolor="white")
plt.close(fig)
print("rendered E4_daily_rhythm")
