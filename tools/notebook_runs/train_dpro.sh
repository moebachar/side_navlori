#!/usr/bin/env bash
# DPRO training, CPU only (no GPU lock). Launch detached:
#   Start-Process -WindowStyle Hidden wsl.exe -ArgumentList '-d','Ubuntu-WSL2','-u','root','--','bash','/mnt/x/side_navlori/tools/notebook_runs/train_dpro.sh'
exec > /root/navlori/logs/dpro_train.log 2>&1
cd /mnt/x/side_navlori
PY=/root/navlori/venv/bin/python
OUT=/root/navlori/runs/dpro
export PYTHONWARNINGS=ignore
echo "start $(date '+%F %T')"
$PY -m dpro.train --out $OUT/sim --sim_steps 4000 --so_steps 300 --seed 0 || exit 1
$PY -m dpro.train --out $OUT/east --ckpt $OUT/sim/final.pth --real_steps 600 --real_zone east --lr 2e-4 --seed 1 || exit 1
$PY -m dpro.train --out $OUT/west --ckpt $OUT/sim/final.pth --real_steps 600 --real_zone west --lr 2e-4 --seed 2 || exit 1
echo "ALL_DONE $(date '+%F %T')"
