#!/usr/bin/env python3
"""
G-series — figures for DROPCEILING_FINDINGS.md, all from real run data
(analysis/g_data.json, produced by analysis/g_data_prep.py).
House style: greyscale + one warm accent reserved for the light/highlight.

Run from diagrams/:  ../../.venv/bin/python src/G_findings_figures.py
"""
import os, json, datetime as dt, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # diagrams/
IO   = os.path.dirname(HERE)                                            # IO/
D    = json.load(open(os.path.join(IO, "analysis", "g_data.json")))

plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
INK="#1a1a1a"; GREY="#9a9a9a"; MID="#6e6e6e"; FAINT="#d9d9d9"; WARM="#e8a23d"; DARKW="#c47f1d"

def save(fig, name):
    fig.savefig(os.path.join(HERE, f"{name}.svg"), bbox_inches="tight", transparent=True)
    fig.savefig(os.path.join(HERE, "png", f"{name}.png"), bbox_inches="tight", dpi=200, facecolor="white")
    plt.close(fig); print("rendered", name)

def foot(fig, text, x=0.07, y=0.015):
    fig.text(x, y, text, fontsize=8, color="#555555")

VERSIONS = [("2026-02-13","first\nself-review"), ("2026-02-25","V5"), ("2026-03-02","V6")]

# ============ G1 — personality trajectories, punctuated not gradual ============
def g1():
    tr = D["g1_trajectory"]; dates=[dt.date.fromisoformat(d) for d in tr]
    params=[("responsiveness",INK,"-"),("energy",MID,"-"),("exploration",GREY,"-"),
            ("attention_span",MID,(0,(5,3))),("memory",GREY,(0,(2,2)))]
    fig,ax=plt.subplots(figsize=(11.5,4.8)); fig.subplots_adjust(left=.07,right=.84,top=.86,bottom=.18)
    finals=[]
    for p,c,ls in params:
        ys=[tr[d.isoformat()].get(p,0) for d in dates]
        ax.plot(dates,ys,color=c,ls=ls,lw=1.9,zorder=4)
        finals.append((p,c,ys[-1]))
    # staggered end labels with leader lines (sorted by final value, fixed slots)
    finals.sort(key=lambda t:-t[2])
    top=max(f[2] for f in finals)+0.16
    for i,(p,c,v) in enumerate(finals):
        slot=top-i*0.09
        ax.annotate(p.replace("_"," "),xy=(dates[-1],v),
                    xytext=(dates[-1]+dt.timedelta(days=0.7),slot),
                    fontsize=9,color=c,va="center",annotation_clip=False,
                    arrowprops=dict(arrowstyle="-",color=c,lw=.7,
                                    connectionstyle="arc3,rad=0.15"))
    for ds,lab in VERSIONS:
        d=dt.date.fromisoformat(ds)
        ax.axvline(d,color=WARM,lw=1.5,zorder=2)
        ax.annotate(lab,xy=(d,1.04),fontsize=8.5,fontweight="bold",color=DARKW,ha="center",
                    annotation_clip=False)
    ax.annotate("day-one chaos:\nparams pinned at limits",xy=(dates[0],0.99),
                xytext=(dt.date(2026,2,13)+dt.timedelta(hours=12),0.88),
                fontsize=8.5,color=MID,ha="left",
                arrowprops=dict(arrowstyle="-",color=GREY,lw=1))
    ax.annotate("the V4/V5 equilibrium\n(11 days, barely moving)",
                xy=(dt.date(2026,2,19),0.52),xytext=(dt.date(2026,2,17),0.27),fontsize=8.5,color=MID,
                arrowprops=dict(arrowstyle="-",color=GREY,lw=1))
    ax.annotate("V5 steps the\ncharacter calmer",xy=(dt.date(2026,2,26),0.38),
                xytext=(dt.date(2026,2,26),0.06),fontsize=8.5,color=MID,ha="left",
                arrowprops=dict(arrowstyle="-",color=GREY,lw=1))
    ax.set_ylim(0,1.05); ax.set_ylabel("parameter value (0–1)",fontsize=10,color=INK)
    ax.set_title("G1   Personality change was punctuated, not gradual — 19 daily snapshots",
                 fontsize=13.5,fontweight="bold",color=INK,loc="left",pad=18)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.grid(axis="y",color="#eeeeee",lw=.7); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    foot(fig,"Daily optimal-value snapshots recorded by the system itself (autotune_daily_learnings). "
             "Record ends Mar 2: the V6 tuner stopped writing this table — the instrument changed with the software.")
    save(fig,"G1_personality_trajectories")

# ============ G2 — the breathing street ============
def g2():
    rows=D["g2_flow"]
    hrs=[r["hour"] for r in rows]; ltr=np.array([r["ltr"] for r in rows]); rtl=np.array([r["rtl"] for r in rows])
    fig,ax=plt.subplots(figsize=(9.5,6.4)); fig.subplots_adjust(left=.12,right=.95,top=.88,bottom=.12)
    ax.barh(hrs,-rtl/1000,color=GREY,edgecolor=INK,lw=.4,zorder=3,label="into the district (R→L)")
    ax.barh(hrs, ltr/1000,color=FAINT,edgecolor=INK,lw=.4,zorder=3,label="homeward (L→R)")
    for h in (8,16):  # the two tide peaks, warm
        i=hrs.index(h)
        ax.barh(h,-rtl[i]/1000,color=WARM if h==8 else GREY,edgecolor=INK,lw=.6,zorder=4)
        ax.barh(h, ltr[i]/1000,color=WARM if h==16 else FAINT,edgecolor=INK,lw=.6,zorder=4)
    ax.annotate("08:00 — the street inhales\nbalance −0.33",xy=(-rtl[8]/1000,8),xytext=(-rtl[8]/1000-90,5.6),
                fontsize=9,color=DARKW,fontweight="bold",ha="left")
    ax.annotate("16:00 — the street exhales\nbalance +0.48",xy=(ltr[16]/1000,16),xytext=(ltr[16]/1000-260,19.0),
                fontsize=9,color=DARKW,fontweight="bold")
    ax.axvline(0,color=INK,lw=1)
    ax.set_ylim(23.8,-0.8); ax.set_yticks(range(0,24,2))
    ax.set_yticklabels([f"{h:02d}:00" for h in range(0,24,2)],fontsize=9)
    ax.set_xlabel("directional movement events per hour-of-day, whole run (thousands)",fontsize=10)
    ax.set_title("G2   The street breathes — commute tide by hour",
                 fontsize=13.5,fontweight="bold",color=INK,loc="left",pad=12)
    ax.legend(loc="lower right",fontsize=9,frameon=False)
    ax.grid(axis="x",color="#eeeeee",lw=.7); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    foot(fig,"10.4M directional events across all measured days. Mornings flow right-to-left into the financial "
             "district; afternoons reverse, at twice the volume. The light's flow-following idle drifted with this tide.")
    save(fig,"G2_breathing_street")

# ============ G3 — engagement episodes + the twelve bonds ============
def g3():
    eps=np.array(D["g3_episodes"]); eps_pos=eps[eps>0]
    fig,ax=plt.subplots(figsize=(10.5,4.6)); fig.subplots_adjust(left=.08,right=.97,top=.84,bottom=.2)
    bins=np.logspace(np.log10(0.4),np.log10(300),28)
    ax.hist(np.clip(eps_pos,0.4,300),bins=bins,color=FAINT,edgecolor=INK,lw=.5,zorder=3)
    walk=int((eps<=0.5).sum())
    ax.set_xscale("log")
    for x,lab in [(3,"notice→greet"),(10,"greet→engage"),(30,"engage→bond")]:
        ax.axvline(x,color=MID,lw=1.2,ls=(0,(4,3)),zorder=2)
        ax.annotate(lab,xy=(x,ax.get_ylim()[1]),xytext=(x*1.05,ax.get_ylim()[1]*0.93),fontsize=8,color=MID)
    bonds=sorted(eps[eps>=30])
    for i,b in enumerate(bonds):
        ax.plot(b,6+ (i%3)*4,"o",color=WARM,ms=7,zorder=6,mec=INK,mew=.5)
    ax.annotate(f"the twelve bonds (30 s+)\nlongest: {max(bonds):.0f} s",
                xy=(bonds[-1],8),xytext=(60,38),fontsize=9.5,fontweight="bold",color=DARKW,
                arrowprops=dict(arrowstyle="-",color=WARM,lw=1.2))
    ax.set_xlabel("engagement episode duration (seconds, log scale)",fontsize=10)
    ax.set_ylabel("episodes",fontsize=10)
    ax.set_title("G3   Engagement is fleeting — 758 episodes, twelve bonds",
                 fontsize=13.5,fontweight="bold",color=INK,loc="left",pad=10)
    ax.grid(axis="y",color="#eeeeee",lw=.7); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    foot(fig,f"V6.5 window (Mar 15–17), episodes reconstructed from the light's own state log. {walk} of 758 are "
             "momentary walk-throughs (<0.5 s, clipped at left edge); 10.7% pass the 3-second notice phase; "
             "1.6% reach the bond phase the dwell architecture was built for.")
    save(fig,"G3_engagement_episodes")

# ============ G4 — loneliest at 4 AM ============
def g4():
    ag={r["hour"]:r["agg"] for r in D["g4_aggression"]}
    pp={r["hour"]:r["people"] for r in D["g4_people"]}
    hrs=list(range(24))
    fig,ax=plt.subplots(figsize=(10.5,4.4)); fig.subplots_adjust(left=.08,right=.9,top=.84,bottom=.2)
    ax2=ax.twinx()
    ax2.fill_between(hrs,[pp.get(h,0) for h in hrs],color=FAINT,zorder=1)
    ax2.set_ylabel("people tracked / hour (avg)",fontsize=9.5,color=GREY)
    ax2.tick_params(colors=GREY)
    # twinx renders above the primary axis — put ax back on top so the warm line stays visible
    ax.set_zorder(ax2.get_zorder()+1); ax.patch.set_visible(False)
    ax.plot(hrs,[ag.get(h,0) for h in hrs],color=WARM,lw=2.6,zorder=5)
    ax.plot(4,ag.get(4,0),"o",color=DARKW,ms=8,zorder=6)
    ax.annotate("loneliest hour:\naggression peaks ~04:00\nwhen no one is there",
                xy=(4,ag.get(4,0)),xytext=(6.2,0.145),fontsize=9.5,fontweight="bold",color=DARKW,
                arrowprops=dict(arrowstyle="-",color=WARM,lw=1.2))
    ax.annotate("held low all afternoon\nby constant company",xy=(14,ag.get(14,0)),
                xytext=(12.5,0.07),fontsize=9,color=MID,
                arrowprops=dict(arrowstyle="-",color=GREY,lw=1))
    ax.set_xlim(0,23); ax.set_xticks(range(0,24,2))
    ax.set_xticklabels([f"{h:02d}" for h in range(0,24,2)],fontsize=9)
    ax.set_xlabel("hour of day",fontsize=10)
    ax.set_ylabel("attention-seeking (aggression, 0–1)",fontsize=10,color=DARKW)
    ax.set_ylim(0,0.18)
    ax.set_title("G4   The light is loneliest at 4 AM — attention-seeking mirrors the traffic",
                 fontsize=13.5,fontweight="bold",color=INK,loc="left",pad=10)
    for s in ("top",): ax.spines[s].set_visible(False); ax2.spines[s].set_visible(False)
    foot(fig,"Aggression averaged per hour across 217K tuning cycles (Feb 11 – Mar 2); grey area = average pedestrians "
             "per hour. The two curves are in opposite phase: the urge to attract peaks exactly when the street is empty.")
    save(fig,"G4_loneliest_hour")

# ============ G5 — gesture economy inverted ============
def g5():
    v5=D["g5_gestures"]["v5"]; v65=D["g5_gestures"]["v65"]
    def shares(d):
        t=sum(d.values()); return [(g,100*c/t) for g,c in sorted(d.items(),key=lambda x:-x[1])[:8]]
    s5,s65=shares(v5),shares(v65)
    fig,axs=plt.subplots(1,2,figsize=(11,4.4),sharex=True); fig.subplots_adjust(left=.12,right=.97,top=.82,bottom=.16,wspace=.42)
    for ax,data,title in [(axs[0],s5,"V5 era (Feb 23–25)"),(axs[1],s65,"V6.5 era (Mar 15–17)")]:
        gs=[g for g,_ in data][::-1]; vs=[v for _,v in data][::-1]
        cols=[WARM if g=="welcome" else (FAINT if g in("thinking","curious","bored") else GREY) for g in gs]
        ax.barh(gs,vs,color=cols,edgecolor=INK,lw=.5)
        for g,v in zip(gs,vs): ax.text(v+0.6,g,f"{v:.0f}%",va="center",fontsize=8.5,color=INK)
        ax.set_title(title,fontsize=11,fontweight="bold",color=INK,loc="left")
        ax.set_xlim(0,36); ax.grid(axis="x",color="#eeeeee",lw=.7); ax.set_axisbelow(True)
        for s in ("top","right"): ax.spines[s].set_visible(False)
    fig.suptitle("G5   The gesture economy inverted — from solitary to social",
                 fontsize=13.5,fontweight="bold",color=INK,x=0.12,ha="left")
    foot(fig,"Share of all gesture ticks. Pale grey = solitary gestures (thinking, curious, bored); warm = welcome. "
             "V5's light mostly gestured to itself; by V6.5 its most frequent gesture greeted someone. "
             "(Repertoire also grew between eras — behaviour and vocabulary changed together.)")
    save(fig,"G5_gesture_economy")

# ============ G6 — a life of waiting, restructured ============
def g6():
    order=["idle","flow","aware","engaged","crowd"]
    cols={"idle":"#ededed","flow":"#bdbdbd","aware":WARM,"engaged":DARKW,"crowd":"#8a5a00"}
    fig,ax=plt.subplots(figsize=(10.5,3.3)); fig.subplots_adjust(left=.16,right=.96,top=.8,bottom=.3)
    for y,(key,label) in enumerate([("v65","V6.5 era\n(Mar 15–17)"),("v5","V5 era\n(Feb 23–25)")]):
        d=D["g6_modes"][key]; tot=sum(v for k,v in d.items() if k in order)
        x=0
        for m in order:
            w=100*d.get(m,0)/tot
            if w<=0: continue
            ax.barh(y,w,left=x,color=cols[m],edgecolor="white",lw=1,height=.55)
            if w>4: ax.text(x+w/2,y,f"{m}\n{w:.0f}%",ha="center",va="center",fontsize=8.5,
                            color=INK if m in("idle","flow") else "white")
            x+=w
        ax.text(-1.5,y,label,ha="right",va="center",fontsize=9.5,color=INK)
    ax.set_xlim(0,100); ax.set_ylim(-.5,1.6); ax.axis("off")
    ax.set_title("G6   A life of waiting, restructured — engagement constant, idle redistributed",
                 fontsize=13.5,fontweight="bold",color=INK,loc="left",pad=14)
    foot(fig,"Share of the light's recorded ticks by mode. Engaged+crowd stays ~9–10% across eras; "
             "V6.5 turned twenty points of idle into flow/aware — attending to passing traffic instead of wandering alone.",y=0.04)
    save(fig,"G6_mode_budget")

# ============ G7 — the brightness ladder ============
def g7():
    order=["idle","flow","aware","engaged","crowd"]
    rows={r["mode"]:r for r in D["g7_brightness"]}
    vals=[rows[m]["avg"] for m in order]
    shades=["#d9d9d9","#bdbdbd",WARM,DARKW,"#8a5a00"]
    fig,ax=plt.subplots(figsize=(8.6,4.4)); fig.subplots_adjust(left=.1,right=.96,top=.84,bottom=.18)
    bars=ax.bar(order,vals,color=shades,edgecolor=INK,lw=.6,width=.62)
    for b,v in zip(bars,vals):
        ax.text(b.get_x()+b.get_width()/2,v+8,f"{v:.0f}",ha="center",fontsize=10,fontweight="bold",color=INK)
    ax.axhline(600,color=GREY,lw=1,ls=(0,(4,3)))
    ax.text(4.4,608,"cap 600",fontsize=8,color=GREY,ha="right")
    ax.set_ylim(0,640); ax.set_ylabel("average brightness (DMX-scale units)",fontsize=10)
    ax.set_title("G7   The brightness ladder — the panels display attention",
                 fontsize=13.5,fontweight="bold",color=INK,loc="left",pad=10)
    ax.grid(axis="y",color="#eeeeee",lw=.7); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    foot(fig,"Average brightness by behaviour mode, V6.5 era. Monotonic from idle (30) to crowd (388): "
             "the window's glow encodes how much attention is present in the alcove.")
    save(fig,"G7_brightness_ladder")

# ============ G8 — the desire line misses the alcove ============
def g8():
    rows=D["g8_zhist"]; z=[r["z"] for r in rows]; n=np.array([r["n"] for r in rows],dtype=float)
    pct=100*n/n.sum()
    fig,ax=plt.subplots(figsize=(10.5,4.4)); fig.subplots_adjust(left=.08,right=.97,top=.84,bottom=.2)
    ax.axvspan(78,283,color=WARM,alpha=.14,zorder=1)
    ax.axvspan(283,633,color=GREY,alpha=.10,zorder=1)
    cols=[WARM if 78<=zz<283 else GREY for zz in z]
    ax.bar(z,pct,width=24,align="edge",color=cols,edgecolor=INK,lw=.4,zorder=3)
    ax.axvline(0,color=INK,lw=2.5); ax.text(6,max(pct)*.97,"panels\n(z=0)",fontsize=8.5,color=INK)
    ax.text(180,max(pct)*.8,"ACTIVE ZONE\n~6% of positions",fontsize=9,fontweight="bold",color=DARKW,ha="center")
    ax.text(455,max(pct)*.93,"PASSIVE ZONE — the sidewalk's desire line",fontsize=9,fontweight="bold",color=MID,ha="center")
    pk=z[int(np.argmax(pct))]
    ax.annotate(f"peak occupancy z {pk}–{pk+25} cm",xy=(pk+12,max(pct)),xytext=(pk+120,max(pct)*1.0),
                fontsize=8.5,color=MID,arrowprops=dict(arrowstyle="-",color=GREY,lw=1))
    ax.set_xlim(0,900); ax.set_xlabel("distance from the panels, z (cm)",fontsize=10)
    ax.set_ylabel("% of all recorded positions",fontsize=10)
    ax.set_title("G8   The desire line misses the alcove — where people actually walked",
                 fontsize=13.5,fontweight="bold",color=INK,loc="left",pad=10)
    ax.grid(axis="y",color="#eeeeee",lw=.7); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    foot(fig,"1.27M raw positions (Mar 15–17). Pedestrians travel the band z 350–450 cm; entering the active zone "
             "means stepping out of the river. The geometry beneath the 3.6% active share.")
    save(fig,"G8_desire_line")

# ============ G9 — friction had two regimes ============
def g9():
    rows=D["g9_budget"]; dates=[dt.date.fromisoformat(r["date"]) for r in rows]
    bud=[r["avg_budget"] for r in rows]; dep=[r["pct_depleted"] for r in rows]
    fig,ax=plt.subplots(figsize=(10.8,4.4)); fig.subplots_adjust(left=.08,right=.9,top=.84,bottom=.2)
    ax2=ax.twinx()
    ax2.bar(dates,dep,color=FAINT,edgecolor=GREY,lw=.5,width=.8,zorder=2)
    ax2.set_ylabel("% of cycles budget-depleted",fontsize=9.5,color=GREY); ax2.set_ylim(0,100)
    ax2.tick_params(colors=GREY)
    ax.plot(dates,bud,color=WARM,lw=2.6,marker="o",ms=4.5,zorder=5)
    ax.set_ylabel("average budget level (max 200)",fontsize=10,color=DARKW); ax.set_ylim(0,210)
    d25=dt.date(2026,2,25)
    ax.axvline(d25,color=INK,lw=1.3,ls=(0,(4,3)))
    ax.annotate("V5 update",xy=(d25,205),fontsize=9,fontweight="bold",color=INK,ha="center",annotation_clip=False)
    ax.annotate("starved: avg ~1.3 of 200\nthrottled 84–88% of cycles",xy=(dt.date(2026,2,18),12),
                xytext=(dt.date(2026,2,13),95),fontsize=9.5,color=DARKW,fontweight="bold",
                arrowprops=dict(arrowstyle="-",color=WARM,lw=1.2))
    ax.annotate("never binds again:\npegged at max, 0% depleted",xy=(dt.date(2026,2,27),200),
                xytext=(dt.date(2026,2,25),140),fontsize=9.5,color=DARKW,fontweight="bold",
                arrowprops=dict(arrowstyle="-",color=WARM,lw=1.2))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_title("G9   Friction had two opposite regimes — the same budget, starved then untouched",
                 fontsize=13.5,fontweight="bold",color=INK,loc="left",pad=10)
    for s in ("top",): ax.spines[s].set_visible(False); ax2.spines[s].set_visible(False)
    foot(fig,"Budget telemetry sampled from 217K tuning cycles. A fixed friction constant suffocated the V4 tuner and "
             "never touched the V5 one — the empirical case for V6 letting the meta-review retune the budget itself.")
    save(fig,"G9_friction_regimes")

# ============ G10 — the diary ============
def g10():
    tr=D["g1_trajectory"]; diary=D["g10_diary"]
    dates=[dt.date.fromisoformat(d) for d in sorted(tr)]
    energy=[tr[d.isoformat()]["energy"] for d in dates]
    def first_clause(s,n=2):
        parts=[p.strip() for p in s.split(";")][:n]
        return ";\n".join(parts)+";…"
    # four diary entries in a fixed 2x2 layout (axes-fraction coords), leaders to data points
    quotes=[("2026-02-12","Day one",            (0.015,0.97),"left"),
            ("2026-02-13","After the first self-review",(0.015,0.30),"left"),
            ("2026-02-25","V5 arrives",         (0.985,0.97),"right"),
            ("2026-03-02","V6 arrives",         (0.985,0.30),"right")]
    fig,ax=plt.subplots(figsize=(11.5,5.8)); fig.subplots_adjust(left=.06,right=.97,top=.87,bottom=.28)
    ax.plot(dates,energy,color=GREY,lw=2,zorder=3)
    ax.fill_between(dates,energy,color=FAINT,alpha=.5,zorder=2)
    ax.set_ylim(0,1.0); ax.set_ylabel("energy parameter (backdrop)",fontsize=9,color=GREY)
    ax.set_yticks([0,.5,1])
    for ds,label,(fx,fy),ha in quotes:
        d=dt.date.fromisoformat(ds)
        txt=first_clause(diary[ds])
        ax.plot(d,tr[ds]["energy"],"o",color=WARM,ms=8,zorder=6,mec=INK,mew=.6)
        ax.annotate(f"{label} — {ds}\n“{txt}”",xy=(d,tr[ds]["energy"]),
                    xycoords="data",xytext=(fx,fy),textcoords="axes fraction",
                    fontsize=7.6,color=INK,ha=ha,va="top",
                    bbox=dict(boxstyle="round,pad=0.45",fc="white",ec=GREY,lw=.8),
                    arrowprops=dict(arrowstyle="-",color=WARM,lw=1.1,
                                    connectionstyle="arc3,rad=0.12"),zorder=7)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_title("G10   The diary — the system narrating its own becoming",
                 fontsize=13.5,fontweight="bold",color=INK,loc="left",pad=10)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    foot(fig,"Verbatim excerpts from the strategy_summary the system wrote each midnight (33 entries survive), "
             "set on the energy parameter's trajectory. The machine kept a plain-English log of its own evolution.",y=0.03)
    save(fig,"G10_diary")

for f in (g1,g2,g3,g4,g5,g6,g7,g8,g9,g10):
    try: f()
    except Exception as e:
        print(f"FAILED {f.__name__}: {e}")
print("done.")
