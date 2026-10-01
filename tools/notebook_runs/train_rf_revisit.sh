#!/usr/bin/env bash
# Radio Flow SLAM phase 2: train the learned revisit check (CPU, no GPU lock).
exec > /root/navlori/logs/rf_revisit.log 2>&1
cd /mnt/x/side_navlori
export PYTHONWARNINGS=ignore
echo "start $(date '+%F %T')"
/root/navlori/venv/bin/python -m dpro.rf.revisit --out /root/navlori/runs/dpro/revisit --threads 2
echo "ALL_DONE $(date '+%F %T')"
