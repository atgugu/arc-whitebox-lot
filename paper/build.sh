#!/usr/bin/env bash
# Build writeup.pdf from writeup.md (needs pandoc >= 2.9 and a TeX Live with pdflatex, mathpazo, tgpagella, tgheros, tcolorbox).
set -euo pipefail
cd "$(dirname "$0")"
pandoc writeup.md -o writeup.pdf \
  --pdf-engine=pdflatex \
  -H preamble.tex \
  --lua-filter callouts.lua \
  --resource-path=.:../figures
echo "built $(pwd)/writeup.pdf"
