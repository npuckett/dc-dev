# Drop Ceiling — Renderable Diagrams (Mermaid + Graphviz)

*Renderable counterparts to the ◧ DRAW blocks in
[ACADIA_2026_DIAGRAM_GUIDE.md](ACADIA_2026_DIAGRAM_GUIDE.md). Prose lives there; this
file is figures only.*

**Framework choice:** **Mermaid** is used for all flow/state figures (§A) — it renders
natively in GitHub/VS Code/Notion, exports to SVG/PNG, and its `stateDiagram-v2` is
ideal for the mode machine. **Graphviz** is used (§B) only for the three figures where it
is objectively better: the tiered DB funnel (rank control), the concentric nested loops
(filled clusters), and the to-scale spatial plan (`neato` with real cm coordinates).

> Rendering quick-start: paste Mermaid blocks into <https://mermaid.live> or any Markdown
> preview that supports Mermaid; render Graphviz with `dot -Tsvg f.dot -o f.svg` (or
> `neato` where noted), or at <https://dreampuf.github.io/GraphvizOnline>.

---

# §A — Mermaid figures

## A1. Top-level pipeline (§1)

```mermaid
flowchart LR
  cam["2× Reolink cameras"] -->|"RTSP 25 fps"| tracker["camera_tracker_osc.py<br/>YOLO → floor x,y"]
  tracker -->|"OSC/UDP :7000<br/>/tracker/count · /tracker/person"| ctrl["lightController_osc.py<br/>behaviour + tuning + DB + WS"]
  ctrl -->|"Art-Net/UDP :6454"| panels["LED panels"]
  ctrl -->|"WebSocket :8765"| web["Three.js web viewer"]
  ctrl -->|"writes"| db[("SQLite DB")]
  db -->|"reads 8 h history"| meta["autotune_meta_review.py<br/>scheduled 3×/day"]
  meta -->|"autotune_overrides.json<br/>hot-reload"| ctrl

  classDef proc fill:#1f2a44,stroke:#9db2ff,color:#fff;
  classDef io fill:#33291f,stroke:#ffcf9d,color:#fff;
  class tracker,ctrl,meta proc;
  class cam,panels,web,db io;
```

## A2. Three nested loops (§2)

*Mermaid can show containment via nested subgraphs (below). For true concentric rings
see the Graphviz version in §B2.*

```mermaid
flowchart TB
  subgraph L3["LOOP 3 · SELF-TUNING · hours → days"]
    d3["meta-review 3×/day + daily learning reshape the PERSONALITY<br/>friction = interaction budget"]
    subgraph L2["LOOP 2 · ANTICIPATION · seconds → hours"]
      d2["trend analysis 1 / 5 / 30 / 60 min + 30 s flow tracker<br/>bias idle/flow · pre-position the light"]
      subgraph L1["LOOP 1 · REACTION · ~33 ms"]
        d1["behaviour mode + gestures follow who is present NOW<br/>30 Hz render → Art-Net"]
      end
    end
  end
```

## A3. Mode state machine (§4) — *the strongest Mermaid case*

```mermaid
stateDiagram-v2
  [*] --> IDLE
  IDLE --> FLOW: ≥2 ppl/min (5s)
  FLOW --> IDLE: <2/min (10s)
  FLOW --> AWARE: ≥10 ppl/min (10s)
  AWARE --> FLOW: <10/min (8s)
  IDLE --> AWARE: ≥10/min (8s)
  AWARE --> IDLE: quiet (12s)
  IDLE --> ENGAGED: active entry (0s · fast)
  FLOW --> ENGAGED: active entry (0s · fast)
  AWARE --> ENGAGED: active entry (0s · fast)
  ENGAGED --> CROWD: 2+ active (3s)
  CROWD --> ENGAGED: crowd thins (5s)
  ENGAGED --> IDLE: empty (5s · slow fade)
  CROWD --> IDLE: empty (5s · slow fade)
  note right of ENGAGED
    Min mode duration 8 s.
    Engage = fast in (0.4 s),
    disengage = slow out (3 s).
    Dwell phases while engaged:
    notice→greet→engage→bond
  end note
```

## A4. Modulation chain — how a meta-parameter becomes light (§3c)

```mermaid
flowchart LR
  base["MODE base value<br/>(MODE_PARAMS)"] --> p["× personality<br/>(meta-param)"]
  p --> t["× time-of-day<br/>(TIME_CONFIGS)"]
  t --> z["× proximity<br/>(Z distance)"]
  z --> i["interpolate"] --> light["LIGHT<br/>position · brightness · pulse · falloff · speed"]

  ex1["e.g. pulse 2500 ms"] -.-> p
  ex2["× energy 0.77 → ×0.86"] -.-> p
  ex3["× evening ×1.3"] -.-> t
  ex4["near ×1.4 brightness"] -.-> z
```

## A5. Trend windows → influence signals (§5)

```mermaid
flowchart LR
  subgraph Q["DB queries · background thread"]
    r["Recent · 1 min"]
    s["Short · 5 min"]
    m["Medium · 30 min"]
    l["Long · 60 min"]
  end
  r --> fold
  s --> fold
  m --> fold
  l --> fold
  fold{{"fold + weight by<br/>idle_trend_weight"}}
  fold --> aa["activity_anticipation 0–1"]
  fold --> fm["flow_momentum −1…+1"]
  fold --> en["energy_level 0–1"]
  aa --> o1["idle brightness · wander interval"]
  fm --> o2["wander-box X shift"]
  en --> o3["idle pulse / speed"]
```

## A6. Aggression as a regulated tank (§5)

```mermaid
flowchart TB
  inflow["ignored time + passers-by who don't stop"] -->|"raise"| tank([aggression level 0–1])
  engage["someone engages"] -->|"lower"| tank
  cap["time-of-day cap (hourly curve)<br/>≈0 at night · peak at lunch"] -.->|"ceiling valve"| tank
  tank --> out["wider/faster wander · more BORED gestures · brighter pulses"]
```

## A7. Self-tuning feedback with the interaction budget (§6c) — *key figure*

```mermaid
flowchart TB
  push["live presence + short-term trends"] -->|"PUSH"| tuner
  tuner["per-frame tuner · SmartAutoTuner ~8 s<br/>gradient ascent on engagement score"]
  resist["RESIST<br/>budget + mean-reversion to home + caps"] -.->|"throttle Δ"| tuner
  tuner -->|"writes Δ"| metap["meta-parameters<br/>(light personality)"]
  tuner -->|"logs every change"| dba[("behavior_adjustments")]
  metap --> light["light output"]
  dba -->|"reads 8 h history"| review["meta-review 3×/day<br/>diagnose: stuck · static · starved<br/>budget too tight / loose"]
  review -->|"META: rewrite home/floors/caps/budget<br/>autotune_overrides.json hot-reload"| resist

  classDef force fill:#3a1f1f,stroke:#ff9d9d,color:#fff;
  class push,resist,review force;
```

## A8. Interaction-budget mechanism (§6b)

```mermaid
flowchart LR
  refill["refill = max / restore_seconds (600 s)"] --> budget([budget · max 200])
  engaged["+ bonus when engaged"] --> budget
  budget --> check{"total cost &gt; budget?"}
  spend["spend = Σ abs(Δ) × cost_scale (30)"] --> check
  check -->|"yes"| throttle["scale all Δ down (throttle)"]
  check -->|"no"| apply["apply Δ · subtract cost"]
  throttle --> budget
  apply --> budget
```

## A9. Self-analysis mirror loop (§7)

```mermaid
flowchart LR
  light["light output"] --> rec["record own state<br/>(light_behavior table)"]
  rec --> read["read it back"]
  read --> metrics["position entropy 1 h · response similarity 24 h<br/>mode distribution 24 h · position cooldown 30 s"]
  metrics -->|"bias to unexplored space · force more variety"| light
```

## A10. Real-time ingestion pipeline (§9)

```mermaid
flowchart LR
  rtsp["2× Reolink RTSP"] --> yolo["batched YOLO · CUDA · 25 fps"]
  yolo --> proj["ArUco-calibrated foot → floor projection (cm)"]
  proj --> fuse["cross-camera fusion<br/>EMA smoothing · velocity coasting"]
  fuse -->|"OSC :7000<br/>/tracker/count · /tracker/person"| ctrl["controller<br/>zone-classify (active/passive) · velocity · callbacks"]
  ctrl --> beh["behaviour system"]
  ctrl --> dbw[("DB write")]
  ctrl -->|"Art-Net :6454 · per-panel DMX"| panels["LED panels (distance-falloff)"]
```

## A11. Web interface — two data planes (§10)

```mermaid
flowchart LR
  ctrl["controller"]
  ctrl -->|"WebSocket :8765 · ~15 fps"| imm["IMMEDIATE plane · Three.js live<br/>light · people · mode · gesture · status"]
  ctrl --> gen["generate_reports.py · nightly<br/>reports/daily/*.json + _index"]
  gen --> pages["GitHub Pages"]
  pages -->|"reports.js fetch + charts"| trends["TRENDS plane · charts page<br/>hourly curves · peak/quiet · flow balance · mode mix"]
  funnel["transport: Tailscale Funnel exposes :8765 as wss:// for HTTPS"] -.-> ctrl

  classDef live fill:#1f3a2a,stroke:#9dffc0,color:#fff;
  classDef slow fill:#2a2a3a,stroke:#b0b0ff,color:#fff;
  class imm live;
  class trends slow;
```

---

# §B — Graphviz figures (where it beats Mermaid)

## B1. Tiered database funnel (§8) — *rank control gives clean columns*

```dot
digraph db {
  rankdir=LR;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fillcolor="#eef2ff"];
  edge [fontname="Helvetica", fontsize=10];

  subgraph cluster_ingest {
    label="INGEST · per OSC msg · 48 h"; style=filled; color="#f5f5f5";
    te  [label="tracking_events\nx,z,vel,zone,flow"];
    lb  [label="light_behavior\nmode,pos,bright,gesture"];
    ba  [label="behavior_adjustments\n+budget before/after/cost"];
    ps  [label="person_sessions"];
  }
  subgraph cluster_agg {
    label="AGGREGATE · hourly / midnight"; style=filled; color="#f5f5f5";
    hs  [label="hourly_stats  (∞)", fillcolor="#e6ffe6"];
    adl [label="autotune_daily_learnings  (∞)", fillcolor="#e6ffe6"];
  }
  subgraph cluster_forever {
    label="KEPT FOREVER"; style=filled; color="#f5f5f5";
    ds  [label="daily_stats_v2  (∞)", fillcolor="#e6ffe6"];
    mtr [label="meta_tuning_reviews  (∞)", fillcolor="#e6ffe6"];
  }

  te -> hs -> ds;
  ba -> adl;
  ba -> mtr;
  lb -> sa [label="self-analysis (§7)", style=dashed];
  sa [label="anti-repetition\nmetrics", shape=note, fillcolor="#fff6e6"];

  { rank=same; te; lb; ba; ps; }
  { rank=same; hs; adl; }
  { rank=same; ds; mtr; }
}
```

## B2. Concentric nested loops (§2) — *filled clusters = true containment*

```dot
digraph loops {
  compound=true; rankdir=TB; fontname="Helvetica";
  node [shape=point, width=0.01, style=invis];

  subgraph cluster_L3 {
    label="LOOP 3 · SELF-TUNING · hours → days\n(meta-review 3×/day + daily learning · friction = interaction budget)";
    style=filled; fillcolor="#e8ecff"; fontname="Helvetica-Bold";
    subgraph cluster_L2 {
      label="LOOP 2 · ANTICIPATION · seconds → hours\n(trends 1/5/30/60 min + 30 s flow · pre-position light)";
      style=filled; fillcolor="#dff0e6";
      subgraph cluster_L1 {
        label="LOOP 1 · REACTION · ~33 ms\n(mode + gestures follow who is present NOW · 30 Hz)";
        style=filled; fillcolor="#ffe9d6";
        core [label="LIGHT", shape=circle, style=filled, fillcolor="#fff3b0", width=0.6, fixedsize=false];
      }
    }
  }
}
```

## B3. Spatial plan, to scale (§9) — *`neato` pins real cm coordinates*

*Coordinates from the source (cm; X along the storefront 0 → −300, Z out toward the
street). Node positions are `X·0.02, Z·0.02` inches so the plan is geometrically true.
Render with **neato**: `neato -Tsvg plan.dot -o plan.svg`. Overlay the active/passive
zone rectangles in your drawing tool (bounds noted below).*

```dot
graph plan {
  layout=neato;
  node [shape=box, fontname="Helvetica", fontsize=9, style=filled];
  edge [style=invis];

  // --- LED panel units (Z = 0) ---
  U0 [label="U0", pos="-0.6,0!",  fillcolor="#c9d4ff"];
  U1 [label="U1", pos="-2.2,0!",  fillcolor="#c9d4ff"];
  U2 [label="U2", pos="-3.8,0!",  fillcolor="#c9d4ff"];
  U3 [label="U3", pos="-5.4,0!",  fillcolor="#c9d4ff"];

  // --- Cameras (Z = 78) ---
  C1 [label="Cam 1\n(-30, 78)",  pos="-0.6,1.56!", shape=diamond, fillcolor="#ffd0d0"];
  C2 [label="Cam 2\n(-270, 78)", pos="-5.4,1.56!", shape=diamond, fillcolor="#d0d0ff"];

  // --- ArUco markers (front row Z=168, back row Z=219) ---
  M0 [label="0", pos="-0.6,3.36!", shape=circle, fillcolor="#fff0c0"];
  M1 [label="1", pos="-3.0,3.36!", shape=circle, fillcolor="#fff0c0"];
  M2 [label="2", pos="-5.4,3.36!", shape=circle, fillcolor="#fff0c0"];
  M3 [label="3", pos="-0.6,4.38!", shape=circle, fillcolor="#fff0c0"];
  M6 [label="6", pos="-3.0,4.38!", shape=circle, fillcolor="#fff0c0"];
  M4 [label="4", pos="-5.4,4.38!", shape=circle, fillcolor="#fff0c0"];
  M5 [label="5 · subway wall\n(-150, 628)", pos="-3.0,12.56!", shape=circle, fillcolor="#fff0c0"];

  // --- Light home position (approx panel field centre) ---
  L  [label="light\n(home)", pos="-3.0,0.4!", shape=octagon, fillcolor="#fff39a"];

  // --- Zone reference labels (rectangles to be drawn in illustration) ---
  AZ [label="ACTIVE ZONE\nX −350…50 · Z 78…283", pos="0.9,2.0!", shape=plaintext, fillcolor=none];
  PZ [label="PASSIVE ZONE\nX −475…175 · Z 283…633", pos="1.2,5.5!", shape=plaintext, fillcolor=none];
}
```

---

## Notes / fidelity

- **State-machine timings** (A3) are the V6.5c stickiness values from `MODE_STICKINESS`
  in `light_behavior.py`; mode thresholds (≥2 / ≥10 ppl·min⁻¹, ≥1 / ≥2 active) from
  `determine_mode()`.
- **Budget constants** (A7/A8) — `max 200`, `restore_seconds 600`, `cost_scale 30` — are
  the production values from `autotune_overrides.json`.
- The **spatial plan** (B3) uses authoritative `TRACKZONE` / `PASSIVE_TRACKZONE` and
  camera/marker coordinates from `lightController_osc.py`. Mermaid cannot pin geometry,
  which is why this one is Graphviz `neato`; for a publication figure, treat it as a
  layout reference and redraw to scale in CAD/Illustrator with the zone rectangles
  filled.
- If you prefer a single framework end-to-end, B1 and B2 also have the Mermaid
  equivalents above (A2, A8-area); only B3 truly requires Graphviz.
```
