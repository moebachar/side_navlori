#!/bin/bash
# Headless full run of side_navlori.ipynb on the GTX 1080 (fablab WSL2), driven by papermill.
# - Output notebook is saved after EVERY cell to /root/navlori/runs/side_navlori.run.ipynb
#   (open it in Jupyter Lab under runs/ to watch progress; the repo notebook is not modified).
# - Cell stdout/stderr is streamed to /root/navlori/logs/notebook.log.
# - Everything in the notebook checkpoints to /content/ckpt and resumes, so re-running is safe.
ROOT=/root/navlori
mkdir -p "$ROOT/logs" "$ROOT/runs"
source "$ROOT/venv/bin/activate"
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST=6.1
python -c "import papermill" 2>/dev/null || pip install -q papermill
cd "$ROOT/side_navlori"
LOG="$ROOT/logs/notebook.log"
echo "=== notebook run started $(date) ===" >> "$LOG"
papermill side_navlori.ipynb "$ROOT/runs/side_navlori.run.ipynb" \
  --log-output --cwd "$ROOT/side_navlori" -k python3 >> "$LOG" 2>&1
RC=$?
echo "=== notebook run finished $(date) exit=$RC ===" >> "$LOG"
exit $RC
