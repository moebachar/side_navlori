#!/usr/bin/env bash
exec > /root/navlori/logs/wifi_nb_exec.log 2>&1
cd /root/navlori
PY=/root/navlori/venv/bin/python
$PY -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=900 --ExecutePreprocessor.kernel_name=python3 \
  /mnt/x/side_navlori/side_navlori_wifi.ipynb
echo "NBEXEC_EXIT=$?"
