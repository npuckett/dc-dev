#!/usr/bin/env bash
# Render all Drop Ceiling diagrams from src/ to SVG (+ PNG).
# Requires: mermaid-cli (mmdc) and graphviz (dot, neato).
#   npm i -g @mermaid-js/mermaid-cli   #  -> mmdc
#   brew install graphviz              #  -> dot / neato
# Graphviz files set their engine via an internal `layout=` attribute,
# so `dot` dispatches neato/etc. automatically.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p png

# --- Mermaid (.mmd) -> SVG + PNG@2x ---
for f in src/*.mmd; do
  n=$(basename "$f" .mmd)
  mmdc -i "$f" -o "$n.svg"     -b transparent -p puppeteer.json
  mmdc -i "$f" -o "png/$n.png" -b white -s 2   -p puppeteer.json
  echo "rendered $n"
done

# --- Graphviz (.dot) -> SVG + PNG@150dpi (respects internal layout=) ---
for f in src/*.dot; do
  n=$(basename "$f" .dot)
  dot -Tsvg "$f" -o "$n.svg"
  dot -Tpng -Gdpi=150 "$f" -o "png/$n.png"
  echo "rendered $n"
done

echo "done."
