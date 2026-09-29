#!/usr/bin/env bash
# Durable overnight run of the camera notebook, FRESH from data + notebook.
# Launch (survives disconnect):
#   Start-Process -WindowStyle Hidden wsl.exe -ArgumentList '-d','Ubuntu-WSL2','-u','root','--','bash','/root/navlori/scripts_local/run_camera.sh'
# Waits for a free GPU (never fights other work), then runs the notebook headless with
# retry + resume (the notebook checkpoints, so a killed attempt continues), plus a stall watchdog.
ROOT=/root/navlori
LOG=$ROOT/logs/camera_run.log
STATUS=$ROOT/runs/STATUS
exec >> "$LOG" 2>&1
echo ""
echo "################ camera run launched $(date '+%F %T') pid=$$ ################"

# --- environment ---
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_HOME=$ROOT/torch_cache          # weights cached locally + durably
export HF_HOME=$ROOT/hf_cache
export TORCH_CUDA_ARCH_LIST=6.1              # GTX 1080 (matches prior deploy); DPVO cell adds 7.5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLBACKEND=Agg
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "$TORCH_HOME" "$HF_HOME" "$ROOT/runs"
source "$ROOT/venv/bin/activate"
python -c "import torch;print('torch',torch.__version__,'cuda avail',torch.cuda.is_available())"

# --- wait for a free GPU (respect other work already using it) ---
FREE_MB=1500; NEED=5; MAXWAIT=$((10*3600)); ok=0; waited=0
echo "waiting for GPU (used < ${FREE_MB} MB x ${NEED} checks; max ${MAXWAIT}s)..."
echo "waiting for free GPU since $(date)" > "$STATUS"
while :; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [ "${used:-99999}" -lt "$FREE_MB" ]; then ok=$((ok+1)); else ok=0; fi
  if [ $((waited % 300)) -eq 0 ]; then echo "  [$(date +%T)] GPU used=${used}MB free-streak=${ok}/${NEED}"; fi
  [ "$ok" -ge "$NEED" ] && { echo "GPU free (used=${used}MB) — starting run $(date)"; break; }
  [ "$waited" -ge "$MAXWAIT" ] && { echo "TIMEOUT waiting for GPU"; echo "TIMEOUT waiting for GPU $(date)" > "$STATUS"; exit 2; }
  sleep 30; waited=$((waited+30))
done

# --- papermill with retry + resume + stall watchdog (pattern from prior deploy) ---
IN=/mnt/x/side_navlori/side_navlori_camera.ipynb     # repo notebook = source of truth
OUT=$ROOT/runs/side_navlori_camera.run.ipynb         # watch progress here
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}; STALL_MIN=${STALL_MIN:-300}
failed_cell() { tac "$LOG" | grep -m1 -oE 'Exception encountered at "In \[[0-9]+\]"'; }
prev_fail=""
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "=== attempt $attempt/$MAX_ATTEMPTS start $(date) ==="
  echo "running attempt $attempt/$MAX_ATTEMPTS since $(date)" > "$STATUS"
  papermill "$IN" "$OUT" -k python3 --log-output --no-progress-bar &
  PM=$!
  ( while kill -0 $PM 2>/dev/null; do
      sleep 300
      age=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
      if [ "$age" -gt $((STALL_MIN*60)) ]; then
        echo "=== WATCHDOG: silent ${age}s — killing attempt $attempt $(date) ==="
        pkill -TERM -P $PM 2>/dev/null; kill -TERM $PM 2>/dev/null; sleep 20
        pkill -KILL -P $PM 2>/dev/null; kill -KILL $PM 2>/dev/null; break
      fi
    done ) &
  WD=$!
  wait $PM; RC=$?
  kill $WD 2>/dev/null
  echo "=== attempt $attempt finished $(date) exit=$RC ==="
  if [ "$RC" -eq 0 ]; then echo "FINISHED OK on attempt $attempt at $(date)" > "$STATUS"; echo "RUN_DONE rc=0"; exit 0; fi
  cell=$(failed_cell)
  echo "attempt $attempt failed (exit $RC) at ${cell:-unknown} $(date)" > "$STATUS"
  if [ -n "$cell" ] && [ "$cell" = "$prev_fail" ]; then
    echo "=== same cell failed twice — stopping ==="; echo "STOPPED: $cell failed twice ($(date))" > "$STATUS"; echo "RUN_DONE rc=$RC"; exit "$RC"
  fi
  prev_fail="$cell"; sleep 30
done
echo "STOPPED: $MAX_ATTEMPTS attempts exhausted ($(date))" > "$STATUS"; echo "RUN_DONE rc=1"; exit 1
