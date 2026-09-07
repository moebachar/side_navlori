#!/bin/bash
# Jupyter Lab for the navlori notebook (fablab WSL2). Idempotent: exits if already running.
# Reach it from the laptop with:  ssh -L 8888:localhost:8888 fablab   then http://localhost:8888/?token=<token>
# The token lives in /root/navlori/jupyter_token. Log: /root/navlori/logs/jupyter.log
ROOT=/root/navlori
mkdir -p "$ROOT/logs"
if pgrep -f "jupyter-lab" > /dev/null; then echo "jupyter lab already running"; exit 0; fi
TOKEN_FILE="$ROOT/jupyter_token"
[ -s "$TOKEN_FILE" ] || python3 -c "import secrets; print(secrets.token_hex(12))" > "$TOKEN_FILE"
TOKEN=$(cat "$TOKEN_FILE")
source "$ROOT/venv/bin/activate"
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST=6.1
cd "$ROOT"
echo "starting jupyter lab $(date) token=$TOKEN" >> "$ROOT/logs/jupyter.log"
exec jupyter lab --no-browser --ip 127.0.0.1 --port 8888 --allow-root \
  --ServerApp.token="$TOKEN" --ServerApp.root_dir="$ROOT" --ServerApp.open_browser=False \
  >> "$ROOT/logs/jupyter.log" 2>&1
