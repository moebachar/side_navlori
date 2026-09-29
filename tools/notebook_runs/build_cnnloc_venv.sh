#!/usr/bin/env bash
exec > /root/navlori/logs/cnnloc_venv.log 2>&1
set -x
export PATH=/root/.local/bin:$PATH
V=/root/navlori/venvs_wifi/cnnloc
rm -rf "$V"; mkdir -p /root/navlori/venvs_wifi
uv venv -q --python 3.7 "$V" || { echo "FAIL venv"; exit 1; }
UVP="uv pip install -q --python $V/bin/python"
$UVP "numpy==1.18.5" "h5py==2.10.0" "protobuf==3.19.6" || { echo "FAIL base"; exit 2; }
$UVP "tensorflow==1.15.5" "keras==2.2.5" "scikit-learn==0.22.2" "pandas==1.1.5" "matplotlib==3.3.4" || { echo "FAIL tf"; exit 3; }
"$V/bin/python" -c "import tensorflow as tf, keras, pandas, sklearn; print('TF', tf.__version__, 'Keras', keras.__version__, 'pandas', pandas.__version__)" || { echo "FAIL import"; exit 4; }
echo "CNNLOC_VENV_READY at $V"
