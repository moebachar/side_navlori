#!/usr/bin/env bash
# Execute the WiFi notebook in place, headless (CPU only: no GPU lock needed).
# The deep-competitor cell takes ~50 min on a cold cache (/content/runs/wifi_golden), so no cell timeout.
# Launch detached:
#   Start-Process -WindowStyle Hidden wsl.exe -ArgumentList '-d','Ubuntu-WSL2','-u','root','--','bash','/mnt/x/side_navlori/tools/notebook_runs/run_wifi_nb.sh'
exec > /root/navlori/logs/wifi_nb_exec.log 2>&1
echo "wifi notebook start $(date '+%F %T')"
cd /root/navlori
PY=/root/navlori/venv/bin/python
$PY -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 \
  /mnt/x/side_navlori/side_navlori_wifi.ipynb
echo "NBEXEC_EXIT=$? $(date '+%F %T')"
