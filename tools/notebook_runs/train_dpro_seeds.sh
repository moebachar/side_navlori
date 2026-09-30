#!/usr/bin/env bash
# Extra training seeds of the synthetic-only DPRO model (seed 0 = runs/dpro/sim from train_dpro.sh). CPU only.
#   Start-Process -WindowStyle Hidden wsl.exe -ArgumentList '-d','Ubuntu-WSL2','-u','root','--','bash','/mnt/x/side_navlori/tools/notebook_runs/train_dpro_seeds.sh'
exec > /root/navlori/logs/dpro_train_seeds.log 2>&1
cd /mnt/x/side_navlori
PY=/root/navlori/venv/bin/python
OUT=/root/navlori/runs/dpro
export PYTHONWARNINGS=ignore
echo "start $(date '+%F %T')"
for s in 1 2; do
  $PY -m dpro.train --out $OUT/sim_s$s --sim_steps 4000 --so_steps 300 --seed $s || exit 1
done
echo "ALL_DONE $(date '+%F %T')"
