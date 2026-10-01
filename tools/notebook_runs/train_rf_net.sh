#!/usr/bin/env bash
# Radio Flow SLAM phase 3: train the learned update operator, 3 seeds in parallel (CPU, 1 thread each, no GPU lock).
cd /mnt/x/side_navlori
export PYTHONWARNINGS=ignore
TUNE=/root/navlori/runs/dpro/rf_tune.json
for s in 0 1 2; do
  /root/navlori/venv/bin/python -m dpro.rf.train_net --tune $TUNE --out /root/navlori/runs/dpro/rf_net_s$s --seed $s --threads 1 --steps 1500 \
    > /root/navlori/logs/rf_net_s$s.log 2>&1 &
done
wait
echo "ALL_DONE $(date '+%F %T')" >> /root/navlori/logs/rf_net_s0.log
