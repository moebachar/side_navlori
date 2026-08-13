#!/usr/bin/env bash
# R2: Reloc3r-512 (CVPR 2025, ffrivera0/reloc3r — CC BY-NC-SA 4.0).
# Pretrained, no scene training. Downloads 1.7 GB weights + NetVLAD .mat. ~20-40 min on T4.
set -e
CM="$(cd "$(dirname "$0")" && pwd)"
MREPOS=${MREPOS:-/content/mrepos}
RUNS=${RUNS:-/content/runs}
mkdir -p "$MREPOS"
if [ ! -d "$MREPOS/reloc3r" ]; then
  git clone --recursive https://github.com/ffrivera0/reloc3r.git "$MREPOS/reloc3r"
  (cd "$MREPOS/reloc3r" && git checkout 761fac648e9c21fd7dcda01ab2ccd4fc20058102 \
     && git submodule update --init --recursive)
fi
pip install -q roma matplotlib tqdm opencv-python scipy einops tensorboard \
  'huggingface-hub[torch]>=0.22' imagesize
python "$CM/prep_reloc3r.py"
mkdir -p "$MREPOS/reloc3r/data"
ln -sfn "$RUNS/reloc3r/7scenes" "$MREPOS/reloc3r/data/7scenes"
cd "$MREPOS/reloc3r"
CUDA_VISIBLE_DEVICES=0 python eval_visloc.py \
  --model "Reloc3rRelpose(img_size=512)" \
  --dataset_db "SevenScenesRetrieval(scene='{}', split='train')" \
  --dataset_q "SevenScenesRetrieval(scene='{}', split='test')" \
  --dataset_relpose "SevenScenesRelpose(scene='{}', pair_id={}, resolution={})" \
  --scene "heads" --topk 10 --batch_size 8 --num_workers 2 --amp 1
python "$CM/reloc3r_post.py" "$MREPOS/reloc3r" 10
python "$CM/smoke_eval.py" reloc3r
