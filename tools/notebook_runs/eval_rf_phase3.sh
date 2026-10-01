#!/usr/bin/env bash
# Radio Flow SLAM: phase 3 evaluations (learned update) + phase 4 follow-up (WiFi gyro-bias check), CPU.
# Learned rows go to one pickle per training (3 processes), merged into rf_results.pkl at the end.
cd /mnt/x/side_navlori
export PYTHONWARNINGS=ignore
PY=/root/navlori/venv/bin/python
RUNS=/root/navlori/runs/dpro
LOG=/root/navlori/logs
TUNE=$RUNS/rf_tune.json
RES=/mnt/x/side_navlori/dpro/rf/results
for s in 0 1 2; do
  $PY -m dpro.rf.evaluate_golden --tune $TUNE --out $RUNS/rf_results_net$s.pkl --only net --threads 1 \
      --nets $RUNS/rf_net_s$s --tseed0 $s > $LOG/rf_eval_net$s.log 2>&1 &
done
$PY -m dpro.rf.evaluate_golden --tune $TUNE --out $RUNS/rf_results.pkl --only gyro --threads 1 --gyro $RES/gyro_bias.json > $LOG/rf_eval_gyro.log 2>&1
$PY -m dpro.rf.eval_synth --tune $TUNE --nets $RUNS/rf_net_s0,$RUNS/rf_net_s1,$RUNS/rf_net_s2 --out $RES/synth_eval.json > $LOG/rf_eval_synth.log 2>&1
$PY -m dpro.rf.analyze_net --tune $TUNE --net $RUNS/rf_net_s0 --out $RES/net_behaviour.json > $LOG/rf_analyze_net.log 2>&1
wait
$PY - <<'EOF'
import pickle
R = "/root/navlori/runs/dpro"
rows = pickle.load(open(f"{R}/rf_results.pkl", "rb"))
done = {(r["method"], r["run"], r["seed"], r.get("tseed", 0)) for r in rows}
for s in range(3):
    for r in pickle.load(open(f"{R}/rf_results_net{s}.pkl", "rb")):
        if (r["method"], r["run"], r["seed"], r.get("tseed", 0)) not in done:
            rows.append(r)
pickle.dump(rows, open(f"{R}/rf_results.pkl", "wb"))
print("MERGED", len(rows))
EOF
echo "PHASE3_EVAL_DONE $(date '+%F %T')"
