#!/usr/bin/env bash
# Durable overnight run of the camera notebook, FRESH from data + notebook.
# - waits until the GPU is free (respects other work already using it)
# - runs the notebook headless via papermill on the fresh venv
# - resumable: papermill re-runs every cell, the notebook's own done()/have()
#   guards skip finished methods, so a re-launch continues where it stopped
# Launched detached (hidden Windows wsl.exe) so it survives a disconnect.
set -uo pipefail

LOG=/root/navlori/logs/camera_run.log
exec > >(tee -a "$LOG") 2>&1
echo "############################################################"
echo "# camera run start $(date '+%F %T')"
echo "############################################################"

# --- environment ---------------------------------------------------------
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_HOME=/root/navlori/torch_cache          # weights cached locally + durably
export HF_HOME=/root/navlori/hf_cache
export MPLBACKEND=Agg                                # headless matplotlib
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "$TORCH_HOME" "$HF_HOME"

source /root/navlori/venv/bin/activate
echo "python: $(which python)  |  $(python --version 2>&1)"
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"

# --- wait for a free GPU -------------------------------------------------
FREE_MB=1500          # consider the GPU free below this used-memory
NEED=5                # consecutive free polls required (~2.5 min)
MAXWAIT=$((8*3600))   # give up after 8 h
ok=0; waited=0
echo "waiting for GPU (used < ${FREE_MB} MB for ${NEED} checks)..."
while :; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [ "${used:-99999}" -lt "$FREE_MB" ]; then ok=$((ok+1)); else ok=0; fi
  if [ $((waited % 300)) -eq 0 ]; then echo "  [$(date +%T)] GPU used=${used}MB  free-streak=${ok}/${NEED}"; fi
  if [ "$ok" -ge "$NEED" ]; then echo "GPU is free (used=${used}MB) — starting run"; break; fi
  if [ "$waited" -ge "$MAXWAIT" ]; then echo "TIMEOUT waiting for GPU after ${MAXWAIT}s — aborting"; exit 2; fi
  sleep 30; waited=$((waited+30))
done

# --- run the notebook ----------------------------------------------------
IN=/mnt/x/side_navlori/side_navlori_camera.ipynb     # repo notebook = source of truth
OUT=/root/navlori/side_navlori_camera.executed.ipynb
echo "papermill: $IN -> $OUT"
papermill "$IN" "$OUT" --kernel python3 --log-output --no-progress-bar
rc=$?
echo "############################################################"
if [ $rc -eq 0 ]; then
  echo "# camera run OK $(date '+%F %T')"
else
  echo "# camera run FAILED rc=$rc $(date '+%F %T')  (re-launch to resume)"
fi
echo "############################################################"
echo "RUN_EXIT rc=$rc"
exit $rc
