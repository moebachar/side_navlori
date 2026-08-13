#!/usr/bin/env bash
# B3 NetVLAD retrieval via official cvg/Hierarchical-Localization (Apache-2.0).
# First run downloads the 529 MB VGG16-NetVLAD-Pitts30K .mat. ~10-20 min total on T4.
set -e
CM="$(cd "$(dirname "$0")" && pwd)"
MREPOS=${MREPOS:-/content/mrepos}
RUNS=${RUNS:-/content/runs}
mkdir -p "$MREPOS"
if [ ! -d "$MREPOS/hloc" ]; then
  git clone --recursive https://github.com/cvg/Hierarchical-Localization.git "$MREPOS/hloc"
  (cd "$MREPOS/hloc" && git checkout c13273bd0ecc2917a35910fd843712a1c6243193 \
     && git submodule update --init --recursive)
fi
pip install -q -e "$MREPOS/hloc"
python "$CM/netvlad_helper.py" stage
python -m hloc.extract_features --conf netvlad \
  --image_dir "$RUNS/netvlad/images" --export_dir "$RUNS/netvlad" --as_half
python "$CM/netvlad_helper.py" post
python "$CM/smoke_eval.py" netvlad
