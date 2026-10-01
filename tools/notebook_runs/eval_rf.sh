#!/usr/bin/env bash
# Radio Flow SLAM: evaluate methods on the 12 golden runs (CPU). Usage: eval_rf.sh <only> [extra args]
ONLY=${1:-all}; shift
exec >> /root/navlori/logs/rf_eval.log 2>&1
cd /mnt/x/side_navlori
export PYTHONWARNINGS=ignore
echo "start $ONLY $(date '+%F %T')"
/root/navlori/venv/bin/python -m dpro.rf.evaluate_golden --tune /root/navlori/runs/dpro/rf_tune.json \
  --out /root/navlori/runs/dpro/rf_results.pkl --only $ONLY --threads 1 \
  --revisit_scores /root/navlori/runs/dpro/revisit/revisit_eval_scores.npz "$@"
echo "EVAL_STAGE_DONE $ONLY $(date '+%F %T')"
