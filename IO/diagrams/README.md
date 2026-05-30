# Drop Ceiling — Rendered Diagrams

SVG (vector, for layout/print) + PNG (raster preview) renders of the figures described
in [../ACADIA_2026_DIAGRAMS_RENDERABLE.md](../ACADIA_2026_DIAGRAMS_RENDERABLE.md).
Editable sources are in [`src/`](src). Re-render everything with `./build.sh`.

| ID | File | Figure | Engine |
|----|------|--------|--------|
| A1 | `A1_pipeline` | Top-level pipeline (3 processes, 2 channels) | Mermaid flowchart |
| A2 | `A2_nested_loops` | Three nested loops (subgraph form) | Mermaid flowchart |
| A3 | `A3_mode_state_machine` | Behaviour mode state machine | Mermaid state diagram |
| A4 | `A4_modulation_chain` | Meta-param → light modulation chain | Mermaid flowchart |
| A5 | `A5_trend_windows` | Trend windows → influence signals | Mermaid flowchart |
| A6 | `A6_aggression_tank` | Aggression as a regulated tank | Mermaid flowchart |
| A7 | `A7_self_tuning_feedback` | Self-tuning feedback (PUSH/RESIST/META) | Mermaid flowchart |
| A8 | `A8_budget_mechanism` | Interaction-budget mechanism | Mermaid flowchart |
| A9 | `A9_self_analysis_mirror` | Self-analysis mirror loop | Mermaid flowchart |
| A10 | `A10_ingestion` | Real-time ingestion pipeline | Mermaid flowchart |
| A11 | `A11_web_two_planes` | Web interface — two data planes | Mermaid flowchart |
| B1 | `B1_db_funnel` | Tiered database funnel | Graphviz `dot` |
| B2 | `B2_nested_loops` | Concentric nested loops (filled rings) | Graphviz `dot` |
| B3 | `B3_spatial_plan` | Installation plan, to scale (cm) | Graphviz `neato` |

## Layout
```
diagrams/
├── *.svg            ← vector renders (use these for the submission)
├── png/*.png        ← raster previews (A* at 2×, B* at 150 dpi)
├── src/             ← editable .mmd / .dot sources
├── puppeteer.json   ← headless-chrome args for mmdc
└── build.sh         ← re-render all
```

## Notes
- **SVG is authoritative** for the proceedings/poster — infinitely scalable, editable in
  Illustrator/Inkscape. PNGs are convenience previews.
- **B3 is a to-scale layout reference**, not a finished figure: `neato` pins panels,
  cameras and ArUco markers to their real centimetre coordinates from
  `lightController_osc.py`. For the poster, open it and fill the active/passive zone
  rectangles (bounds are printed in the figure).
- All numeric values trace to the V6.5c source (see the parent guide for line refs).
- The Mermaid figures use plain ASCII in labels (no `×`, `→`, `≥`, `±`) so they parse
  cleanly across renderers; the parent guide keeps the typographic versions.
