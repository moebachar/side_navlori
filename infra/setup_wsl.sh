#!/bin/bash
# navlori GPU workstation bootstrap — Ubuntu 22.04 WSL2 on fablab, GTX 1080 (sm_61).
# One venv for everything: torch 2.3.1+cu121 (the pin already validated for DPVO on Colab;
# its wheels ship sm_61 kernels). CUDA toolkit 12.1 so ACE (dsacstar) and DPVO extensions compile.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
ROOT=/root/navlori
mkdir -p "$ROOT"

echo "=== [1/5] system packages ==="
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
  build-essential ninja-build cmake git curl unzip wget ca-certificates \
  python3.10-venv python3-pip python3-dev \
  libgl1 libglib2.0-0 ffmpeg > /dev/null

echo "=== [2/5] CUDA toolkit 12.1 (WSL flavour, no driver) ==="
if [ ! -d /usr/local/cuda-12.1 ]; then
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
  dpkg -i /tmp/cuda-keyring.deb > /dev/null
  apt-get update -qq
  apt-get install -y -qq cuda-toolkit-12-1 > /dev/null
fi
grep -q cuda-12.1 /root/.bashrc || {
  echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> /root/.bashrc
  echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH:-}' >> /root/.bashrc
  echo 'export TORCH_CUDA_ARCH_LIST=6.1' >> /root/.bashrc
}
export PATH=/usr/local/cuda-12.1/bin:$PATH
nvcc --version | tail -2

echo "=== [3/5] the one venv ==="
[ -d "$ROOT/venv" ] || python3.10 -m venv "$ROOT/venv"
source "$ROOT/venv/bin/activate"
pip install -q -U pip wheel setuptools
pip install -q torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -q "numpy==1.26.4" pandas matplotlib pillow scipy jupyterlab ipywidgets opencv-python-headless

echo "=== [4/5] git identity ==="
git config --global user.name "Mohamed BACHAR"
git config --global user.email "j.elfirqi@gmail.com"
git config --global --add safe.directory "*"

echo "=== [5/5] GPU smoke test ==="
python - <<'EOF'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda)
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
x = torch.randn(2048, 2048, device="cuda")
torch.cuda.synchronize()
print("matmul on GPU ok:", float((x @ x).abs().sum()) > 0)
print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
EOF
echo SETUP_OK
