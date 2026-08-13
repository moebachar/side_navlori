#!/usr/bin/env bash
# R1: ACE scene coordinate regression (CVPR 2023, nianticlabs/ace — NON-COMMERCIAL license).
# dsacstar C++ extension compiles against apt OpenCV. Train ~15-30 min on T4 (CPU-bound buffer fill).
set -e
CM="$(cd "$(dirname "$0")" && pwd)"
MREPOS=${MREPOS:-/content/mrepos}
RUNS=${RUNS:-/content/runs}
mkdir -p "$MREPOS"
if [ ! -d "$MREPOS/ace" ]; then
  git clone https://github.com/nianticlabs/ace.git "$MREPOS/ace"
  (cd "$MREPOS/ace" && git checkout e9e90f2d02ee92c348bf411a5a60e230af6c315e)
fi
python -c "
import pathlib, os
s = pathlib.Path(os.environ.get('MREPOS', '/content/mrepos'), 'ace', 'ace_encoder_pretrained.pt').stat().st_size
assert s == 22278371, f'encoder looks like an LFS pointer: {s} bytes'"
apt-get -qq update && apt-get -qq install -y libopencv-dev > /dev/null
pip install -q pyrender trimesh
python -c "import dsacstar" 2>/dev/null || \
  (cd "$MREPOS/ace/dsacstar" && CONDA_PREFIX=/usr pip install -q --no-build-isolation .)
python -c "import dsacstar; print('dsacstar OK')"
python "$CM/prep_ace.py"
cd "$MREPOS/ace"
python train_ace.py "$RUNS/ace/datasets/navlori" "$RUNS/ace/navlori.pt" \
  --training_buffer_size 4000000
python test_ace.py "$RUNS/ace/datasets/navlori" "$RUNS/ace/navlori.pt" --session run1
python "$CM/ace_post.py" "$RUNS/ace/poses_navlori_run1.txt"
python "$CM/smoke_eval.py" ace
