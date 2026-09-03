#!/bin/bash
# navlori deploy: unpack code + data staged on X:, wire /content, validate the init path.
set -euo pipefail
ROOT=/root/navlori
[ -d /mnt/x/navlori_staging ] || { echo "X: not mounted at /mnt/x"; exit 1; }

echo "=== [1/6] code + data from X:\\navlori_staging ==="
rm -rf "$ROOT/side_navlori"
tar -xf /mnt/x/navlori_staging/side_navlori_repo.tar -C "$ROOT"
cp -f /mnt/x/navlori_staging/data.zip "$ROOT/data.zip"
mkdir -p "$ROOT/ckpt" "$ROOT/runs"
[ -L /content ] || ln -s "$ROOT" /content
git -C "$ROOT/side_navlori" log --oneline -1

echo "=== [2/6] uv (the DPVO cell provisions its own 3.11 venv with it) ==="
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null
[ -x /usr/local/bin/uv ] || ln -sf /root/.local/bin/uv /usr/local/bin/uv

echo "=== [3/6] deps Colab preinstalls but a bare venv lacks ==="
source "$ROOT/venv/bin/activate"
pip install -q scikit-learn einops tqdm pyyaml h5py kornia

echo "=== [4/6] unpack dataset ==="
[ -d "$ROOT/data/camera/images" ] || python -c "
import zipfile; zipfile.ZipFile('$ROOT/data.zip').extractall('$ROOT')"
echo "images: $(ls "$ROOT/data/camera/images" | wc -l)"

echo "=== [5/6] init-cell dry run (exact code the notebook executes) ==="
cd "$ROOT/side_navlori"
python -c "
import json
src = ''.join(next(c for c in json.load(open('side_navlori.ipynb'))['cells'] if c.get('id') == 'init')['source'])
exec(src)"

echo "=== [6/6] Claude Code ==="
command -v claude > /dev/null || curl -fsSL https://claude.ai/install.sh | bash > /dev/null 2>&1 || echo "claude install failed (non-fatal)"
command -v claude > /dev/null && claude --version || true
echo DEPLOY_OK
