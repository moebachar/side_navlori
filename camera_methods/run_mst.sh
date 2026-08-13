#!/usr/bin/env bash
# B2 APR: MS-Transformer (ICCV 2021, authors' official repo; NOTE: no LICENSE file).
# Trains from scratch on our scene: ~30-60 min on T4 (30 epochs, batch 8).
set -e
CM="$(cd "$(dirname "$0")" && pwd)"
MREPOS=${MREPOS:-/content/mrepos}
RUNS=${RUNS:-/content/runs}
DATA=${DATA:-/content/data}
mkdir -p "$MREPOS"
if [ ! -d "$MREPOS/mst" ]; then
  git clone https://github.com/yolish/multi-scene-pose-transformer.git "$MREPOS/mst"
  (cd "$MREPOS/mst" && git checkout 56ca699fa61ad689f3631231c03608e46c9c7938)
fi
pip install -q efficientnet-pytorch==0.7.1
python "$CM/prep_mst.py" "$MREPOS/mst"
cd "$MREPOS/mst"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
python "$CM/mst_wrap.py" main.py ems-transposenet train \
  ./models/backbones/efficient-net-b0.pth "$DATA/camera/images" \
  "$RUNS/mst/myroom_train.csv" "$RUNS/mst/myroom_config.json" --experiment myroom
CKPT=$(ls -t out/run_*_final.pth | head -1)
echo "using checkpoint: $CKPT"
python "$CM/mst_dump.py" "$MREPOS/mst" "$MREPOS/mst/$CKPT"
python "$CM/smoke_eval.py" mst
