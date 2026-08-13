#!/usr/bin/env bash
# R3: DPVO / DPV-SLAM (NeurIPS 2023, princeton-vl — MIT). Visual odometry row.
# Own venv (torch 2.3.1 pin — its CUDA extensions don't compile on torch>=2.5).
# First run ~15-25 min install (CUDA compile), then ~5-10 min per pass on T4.
set -e
CM="$(cd "$(dirname "$0")" && pwd)"
MREPOS=${MREPOS:-/content/mrepos}
RUNS=${RUNS:-/content/runs}
VENV=${VENV:-/content/venvs/dpvo}
mkdir -p "$MREPOS"
if [ ! -d "$MREPOS/DPVO" ]; then
  git clone --recursive https://github.com/princeton-vl/DPVO.git "$MREPOS/DPVO"
  (cd "$MREPOS/DPVO" && git checkout 859bbbfdac6c6185f345003b3c473901fcd13ace)
fi
if [ ! -f "$VENV/ok" ]; then
  python -m venv "$VENV"
  "$VENV/bin/pip" -q install -U pip setuptools wheel
  "$VENV/bin/pip" -q install torch==2.3.1 torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cu121
  "$VENV/bin/pip" -q install numpy==1.26.4 'opencv-python<4.11' evo plyfile yacs \
    tqdm einops numba pypose kornia scipy tensorboard gdown
  "$VENV/bin/pip" -q install torch-scatter -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
  if [ ! -d "$MREPOS/DPVO/thirdparty/eigen-3.4.0" ]; then
    (cd "$MREPOS/DPVO" && wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip \
       && unzip -qo eigen-3.4.0.zip -d thirdparty)
  fi
  (cd "$MREPOS/DPVO" && TORCH_CUDA_ARCH_LIST="7.5" "$VENV/bin/pip" install --no-build-isolation .)
  if [ ! -f "$MREPOS/DPVO/dpvo.pth" ]; then
    (cd "$MREPOS/DPVO" && "$VENV/bin/gdown" 1dRqftpImtHbbIPNBIseCv9EvrlHEnjhX -O models.zip \
       && unzip -o models.zip)
  fi
  "$VENV/bin/python" -c "
import os
assert os.path.getsize('$MREPOS/DPVO/dpvo.pth') == 14167743"
  "$VENV/bin/python" -c "from dpvo.dpvo import DPVO; print('DPVO import OK')"
  touch "$VENV/ok"
fi
python "$CM/prep_dpvo.py"
cd "$MREPOS/DPVO"
"$VENV/bin/python" demo.py --imagedir="$RUNS/dpvo/frames" --calib="$RUNS/dpvo/calib.txt" \
  --network=dpvo.pth --config=config/default.yaml --stride=1 --name=navlori --save_trajectory
"$VENV/bin/python" demo.py --imagedir="$RUNS/dpvo/frames" --calib="$RUNS/dpvo/calib.txt" \
  --network=dpvo.pth --config=config/default.yaml --stride=1 --name=navlori_lc \
  --save_trajectory --opts LOOP_CLOSURE True
RUNS="$RUNS" "$VENV/bin/python" "$CM/dpvo_eval.py" saved_trajectories/navlori.txt dpvo
RUNS="$RUNS" "$VENV/bin/python" "$CM/dpvo_eval.py" saved_trajectories/navlori_lc.txt dpvo_lc
