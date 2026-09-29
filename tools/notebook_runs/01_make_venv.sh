#!/usr/bin/env bash
# Fresh main kernel venv (GTX 1080 / sm_61, CUDA 12.1). Linear + loud.
# Invoke line-buffered:  stdbuf -oL -eL bash 01_make_venv.sh
set -x
echo "venv build start $(date '+%F %T') pid=$$"
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
V=/root/navlori/venv
SENT=/root/navlori/venv/.READY

rm -f "$SENT"
rm -rf "$V"
python3 -m venv "$V"; echo "rc-venv=$?"
source "$V/bin/activate"
python -m pip install --upgrade pip wheel setuptools; echo "rc-pip=$?"

echo ">>> torch @ $(date +%T)"
pip install --retries 5 --timeout 180 torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121; echo "rc-torch=$?"
sync

echo ">>> base @ $(date +%T)"
pip install --retries 5 numpy pandas scipy matplotlib scikit-learn \
    opencv-python h5py pyyaml tqdm Pillow einops; echo "rc-base=$?"
sync

echo ">>> hub/jup @ $(date +%T)"
pip install --retries 5 "huggingface-hub[torch]>=0.22" gdown \
    jupyter ipykernel papermill nbclient nbformat; echo "rc-jup=$?"
sync

echo ">>> kernel @ $(date +%T)"
python -m ipykernel install --user --name python3 --display-name "navlori (venv)"; echo "rc-kernel=$?"

python - <<'PY'
import torch, torchvision, numpy, pandas, cv2, sys
print("python", sys.version.split()[0])
print("torch", torch.__version__, "tv", torchvision.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
print("numpy", numpy.__version__, "pandas", pandas.__version__, "cv2", cv2.__version__)
PY
rc=$?
if [ $rc -eq 0 ]; then touch "$SENT"; echo "venv ready at $V @ $(date '+%F %T')"; else echo "VERIFY FAILED rc=$rc"; fi
