#!/usr/bin/env bash
exec > /root/navlori/logs/cnnloc_venv.log 2>&1
set -x
export MAMBA_ROOT_PREFIX=/root/navlori/mamba
MM=/root/navlori/mamba/bin/micromamba
mkdir -p /root/navlori/mamba/bin

# 1) fetch micromamba static binary (once)
if [ ! -x "$MM" ]; then
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /root/navlori/mamba bin/micromamba || { echo "FAIL micromamba dl"; exit 1; }
fi
"$MM" --version || { echo "FAIL micromamba run"; exit 1; }

# 2) create python 3.7 env from conda-forge
ENV=/root/navlori/venvs_wifi/cnnloc
rm -rf "$ENV"
"$MM" create -y -p "$ENV" -c conda-forge python=3.7 pip || { echo "FAIL env create"; exit 2; }
PY="$ENV/bin/python"
$PY --version || { echo "FAIL py"; exit 2; }

# 3) legacy stack via pip (cp37 wheels)
$PY -m pip install -q --upgrade "pip<24" "setuptools<60" wheel || { echo "FAIL pip up"; exit 3; }
$PY -m pip install -q "numpy==1.18.5" "h5py==2.10.0" "protobuf==3.19.6" || { echo "FAIL base"; exit 3; }
$PY -m pip install -q "tensorflow==1.15.5" "keras==2.2.5" "scikit-learn==0.22.2" "pandas==1.1.5" "matplotlib==3.3.4" || { echo "FAIL tf"; exit 4; }

# 4) verify
$PY -c "import tensorflow as tf, keras, pandas, sklearn, h5py; print('TF', tf.__version__, 'Keras', keras.__version__, 'np', __import__('numpy').__version__)" || { echo "FAIL import"; exit 5; }
echo "CNNLOC_VENV_READY at $ENV"
