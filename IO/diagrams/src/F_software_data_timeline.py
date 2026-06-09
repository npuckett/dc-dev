#!/usr/bin/env python3
"""
F1 — Software-update timeline aligned against database capture.
Top lane: daily data coverage + quality (events/day, coverage state).
Bottom lane: software version milestones (from git).
Shared date axis. Greyscale + warm accent for the light/version events.

Run: ../.venv/bin/python src/F_software_data_timeline.py
"""
import os, sqlite3, datetime as dt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

OUT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams.update({"font.family":"DejaVu Sans","svg.fonttype":"none"})
INK="#1a1a1a"; GREY="#9a9a9a"; FAINT="#d9d9d9"; WARM="#e8a23d"; BAD="#b0b0b0"

con=sqlite3.connect(os.path.join(OUT,"..","analysis","merged_run.db"))
rows=con.execute("""SELECT date,SUM(total_events),COUNT(*),MAX(estimated),
                    GROUP_CONCAT(DISTINCT src)
                    FROM hourly_stats_filled GROUP BY date ORDER BY date""").fetchall()
con.close()

dates=[dt.date.fromisoformat(d) for d,_,_,_,_ in rows]
events=[e or 0 for _,e,_,_,_ in rows]
hours=[h for _,_,h,_,_ in rows]
est=[bool(x) for _,_,_,x,_ in rows]
srcs=[s or "" for _,_,_,_,s in rows]

fig,(axD,axS)=plt.subplots(2,1,figsize=(12.5,7.0),height_ratios=[2.6,1.5],sharex=True)
fig.subplots_adjust(left=0.07,right=0.985,top=0.82,bottom=0.13,hspace=0.12)

# ---- TOP: events/day bars, coloured by quality/provenance ----
for d,e,h,is_est,src in zip(dates,events,hours,est,srcs):
    if src=="report":    c,ec,hatch=FAINT,WARM,"///"   # recovered from report archive (DB-pruned)
    elif is_est:         c,ec,hatch=FAINT,WARM,None     # estimated (calibration artifact corrected)
    elif h<20:           c,ec,hatch=FAINT,GREY,None     # partial day
    else:                c,ec,hatch=GREY,INK,None       # full measured day
    axD.bar(d,e,width=0.9,color=c,edgecolor=ec,linewidth=1.1 if (is_est or src=="report") else 0.5,
            hatch=hatch,zorder=3)
axD.text(dt.date(2026,2,6),axD.get_ylim()[1]*0.80 if axD.get_ylim()[1]>0 else 5e5,
         "Feb 3–9\nrecovered from\nreport archive\n(pruned from DB)",
         ha="center",va="center",fontsize=7.5,color="#8a6320",style="italic")
axD.set_ylabel("tracking events / day",fontsize=10,color=INK)
axD.set_title("F1   Software updates vs database capture, over the run",
              fontsize=14,fontweight="bold",color=INK,loc="left",pad=42)
axD.yaxis.set_major_formatter(lambda x,_:(f"{x/1e6:.1f}M".replace(".0M","M") if x>=1e6 else (f"{x/1e3:.0f}K" if x>=1e3 else "0")))
axD.grid(axis="y",color="#eeeeee",lw=0.7); axD.set_axisbelow(True)
for s in ("top","right"): axD.spines[s].set_visible(False)
leg=[Patch(fc=GREY,ec=INK,label="measured (full day)"),
     Patch(fc=FAINT,ec=GREY,label="partial day"),
     Patch(fc=FAINT,ec=WARM,label="estimated split (calibration corrected)"),
     Patch(fc=FAINT,ec=WARM,hatch="///",label="recovered from report archive")]
axD.legend(handles=leg,loc="upper left",fontsize=8,frameon=False,ncol=4,bbox_to_anchor=(0,1.16))

# ---- BOTTOM: software version milestones ----
axS.set_ylim(0,1); axS.set_yticks([])
for s in ("top","right","left"): axS.spines[s].set_visible(False)
milestones=[
    ("2026-01-27","V2",1),
    ("2026-02-11","V4",1),
    ("2026-02-12","DB fix\n(corruption)",0),
    ("2026-02-25","V5\n(gestures, falloff)",1),
    ("2026-03-02","V6",0),
    ("2026-03-03","V6.5\n(3 tiers)",1),
    ("2026-03-04","V6.5c",2),
    ("2026-03-15","AWARE mode\nin data",1),
]
lane_y={0:0.24, 1:0.55, 2:0.04}
for ds,label,lane in milestones:
    d=dt.date.fromisoformat(ds)
    y=lane_y[lane]
    big = ("DB fix" in label) or ("AWARE" in label)
    axS.axvline(d,color=WARM if big else GREY,lw=1.6 if big else 1.0,
                ls="-" if big else (0,(2,2)),zorder=2,
                ymin=0,ymax=1.9,clip_on=False)
    axS.plot(d,y,"o",color=WARM if big else INK,markersize=7 if big else 5,zorder=4)
    axS.annotate(label,xy=(d,y),xytext=(d,y+0.12),ha="center",va="bottom",
                 fontsize=8.5,fontweight="bold" if big else "normal",
                 color=INK,zorder=5)
axS.text(dates[0],0.93,"software version",fontsize=9,color=GREY,style="italic")

# shared x axis
axS.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
axS.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
axS.set_xlim(dt.date(2026,1,26),dt.date(2026,3,19))
plt.setp(axS.get_xticklabels(),fontsize=9,color=INK)

fig.text(0.07,0.025,
 "Capture begins with V2 (Jan 29–Feb 2 active-zone counts invalid from an uncorrected tracking offset, re-estimated here). "
 "Feb 3–9 were pruned from the database but survive in the daily-report archive (hatched). A calibration fix on Feb 12 restores clean capture. "
 "Behaviour modes and gestures enter the data as the software adds them — the five-mode vocabulary (AWARE) only appears from Mar 15.",
 fontsize=8,color="#555555")

fig.savefig(os.path.join(OUT,"F1_software_data_timeline.svg"),bbox_inches="tight",transparent=True)
fig.savefig(os.path.join(OUT,"png","F1_software_data_timeline.png"),bbox_inches="tight",dpi=200,facecolor="white")
plt.close(fig)
print("rendered F1_software_data_timeline")
