#!/usr/bin/env bash
# Render all Drop Ceiling diagrams from src/ to SVG (+ PNG).
# Requires: mermaid-cli (mmdc) and graphviz (dot, neato).
#   npm i -g @mermaid-js/mermaid-cli   #  → mmdc
#   brew install graphviz              #  → dot / neato
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p png

# --- Mermaid (§A) → SVG + PNG@2x ---
for f in src/A*.mmd; do
  n=$(basename "$f" .mmd)
  mmdc -i "$f" -o "$n.svg"     -b transparent -p puppeteer.json
  mmdc -i "$f" -o "png/$n.png" -b white -s 2   -p puppeteer.json
  echo "rendered $n"
done

# --- Graphviz (§B) → SVG + PNG@150dpi ---
dot   -Tsvg src/B1_db_funnel.dot    -o B1_db_funnel.svg
dot   -Tsvg src/B2_nested_loops.dot -o B2_nested_loops.svg
neato -Tsvg src/B3_spatial_plan.dot -o B3_spatial_plan.svg
dot   -Tpng -Gdpi=150 src/B1_db_funnel.dot    -o png/B1_db_funnel.png
dot   -Tpng -Gdpi=150 src/B2_nested_loops.dot -o png/B2_nested_loops.png
neato -Tpng -Gdpi=150 src/B3_spatial_plan.dot -o png/B3_spatial_plan.png
echo "rendered B1, B2, B3"
echo "done."
