#!/usr/bin/env bash
# Build the six Figure 1 panels and the vector composite.
# Run from the repository root. Requires pdflatex.

set -euo pipefail
cd "$(dirname "$0")/.."

PANELS=figure1/panels
OUT=results/figure1
mkdir -p "$OUT"

echo "A  passive latent dynamics"
TITLE='Passive Latent Dynamics  ($u(t)=0$)' SUBTITLE="Observations" \
MODE=samples CONNECT=1 STRIP=1 N_SAMPLES=14 OBS_NOISE=0.08 SAMPLE_SIGMA=0.20 \
COMPACT=1 \
  uv run python $PANELS/cell1_minimal.py $OUT/Figure_1A_passive.png

echo "B  latent dynamics under perturbation"
COMPACT=1 \
  uv run python $PANELS/cell2_minimal.py $OUT/Figure_1B_perturbation.png

echo "C  shared dynamical model"
COMPACT=1 POPULATION=1 \
  uv run python $PANELS/cell2_minimal.py $OUT/Figure_1C_shared.png

echo "D-F  the three evaluation layers"
uv run python $PANELS/cell4_individual.py $OUT

echo "composite"
uv run python figure1/assemble_figure1.py "$OUT" "$OUT/Figure_1_concept.png"
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=/tmp \
  figure1/assemble_figure1_vector.tex >/dev/null
cp /tmp/assemble_figure1_vector.pdf "$OUT/Figure_1_concept.pdf"
echo "wrote $OUT/Figure_1_concept.pdf"
