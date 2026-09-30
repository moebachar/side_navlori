#!/usr/bin/env bash
exec > /root/navlori/logs/rerun_figs.log 2>&1
source /root/navlori/venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.1; export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_HOME=/root/navlori/torch_cache; export HF_HOME=/root/navlori/hf_cache
export TORCH_CUDA_ARCH_LIST=6.1; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# deliberately NOT setting MPLBACKEND=Agg -> ipykernel inline backend embeds PNG figures.
cd /root/navlori
LOCK="/root/navlori/venv/bin/python /mnt/x/navlori-fusion/scripts/gpu_lock.py"
$LOCK acquire --wait --who side_navlori --what "camera notebook figures (rerun_figs.sh)" || exit 2
trap '$LOCK release --who side_navlori' EXIT
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=python3 \
  --ExecutePreprocessor.timeout=1800 \
  --output /root/navlori/runs/side_navlori_camera.figs.ipynb \
  /mnt/x/side_navlori/side_navlori_camera.ipynb
echo "NBEXEC rc=$?"
python3 - <<PY
import json
nb=json.load(open("/root/navlori/runs/side_navlori_camera.figs.ipynb"))
imgs=sum(1 for c in nb["cells"] if c["cell_type"]=="code" for o in c.get("outputs",[]) if "image/png" in o.get("data",{}))
err=sum(1 for c in nb["cells"] if c["cell_type"]=="code" for o in c.get("outputs",[]) if o.get("output_type")=="error")
print("FIGS_EMBEDDED", imgs, "ERRORS", err, "CELLS", len(nb["cells"]))
PY
