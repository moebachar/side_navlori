#!/usr/bin/env bash
# Smoke-test cells 0-12 (data load -> EDA -> split -> harness) on the fresh venv. No GPU compute.
set -x
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST=6.1
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
source /root/navlori/venv/bin/activate
IN=/mnt/c/Users/FabLab/AppData/Local/Temp/claude/x--side-navlori/f73304e9-d6f1-4430-b44e-0bfcf16646b7/scratchpad/smoke_0_12.ipynb
OUT=/root/navlori/runs/smoke_0_12.out.ipynb
mkdir -p /root/navlori/runs
cd /root/navlori
papermill "$IN" "$OUT" -k python3 --log-output --no-progress-bar
echo "SMOKE_EXIT rc=$?"
