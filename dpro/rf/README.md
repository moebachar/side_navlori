# Radio Flow SLAM (dpro/rf)

DROID-SLAM's structure rebuilt for WiFi: a graph of scan pairs (radio-flow links up to 8 scans back, absolute
RSSI factors, revisit links), one joint solve for the robot path and the AP map (Schur complement on the AP
blocks), and optional learned parts. Results report: <https://claude.ai/artifact/1KxQD1ZS2tK7Xi5EwXDhT5>.

## Results on the 12 golden runs (2026-10-01)

Mean over runs of the per-run median error, aligned (whole-run rigid fit). Weights chosen on synthetic sites only.

| Method | Aligned (m) |
| --- | --- |
| Best hand-weighted (flow + absolute + revisits) | 1.47 |
| Learned update, also online (3 trainings) | 1.46 |
| DPRO v1, network off, tuned (best earlier) | 1.47 |
| Ceiling: absolute RSSI with the true AP map | 1.20 |
| Wheel odometry (/odom, gyro heading) | 0.48 |
| /odom after the WiFi gyro-bias check | 0.30 |
| Wheel encoders only, no gyro | 0.13 |

None of the four phase gates passed. Runs 4 and 9 have a constant IMU gyro bias (-21.9 and +4.6 deg/s) that
breaks /odom; WiFi can estimate it (`gyro_bias.py`), but the wheel encoders do not share the fault and fix it
better (`encoder_odometry.py`).

## Modules

| File | What it does |
| --- | --- |
| `slam.py` | `RFConfig`, `RadioFlowSLAM`: streaming frontend, revisit backend, learned refinement |
| `ba.py` | Gauss-Newton solve: pair and single factors, motion priors, odometry, Schur on AP blocks |
| `graph.py` | 3-scan averaging, fingerprint distance, revisit proposals with NMS, noise model |
| `metrics.py` | aligned, online (30 s fit), drift per 5 m; `summarize.py` adds causal |
| `tune.py`, `tune_extra.py` | weight sweeps on 24 synthetic sites |
| `revisit.py`, `eval_revisit.py` | phase 2: learned revisit check |
| `net.py`, `train_net.py`, `eval_synth.py`, `analyze_net.py` | phase 3: learned update operator |
| `odometry.py`, `odo_diag.py` | phase 4: odometry factor tuning (synthetic), weight diagnostic (real) |
| `gyro_bias.py` | WiFi estimate of a constant gyro bias; margin set on 48 synthetic runs |
| `encoder_odometry.py` | control: dead reckoning from the wheel encoders alone |
| `evaluate_golden.py`, `summarize.py`, `export_report.py` | all methods on the golden runs, tables, report data |
| `report/` | the results page (`build.py` assembles `template.html` + `report.js` + report data) |

Small results are in `results/`; checkpoints and `rf_results.pkl` stay in `/root/navlori/runs/dpro` (WSL).
Commands: see the "Reproduce" section of the report. The algorithm, with every equation and constant, is in `ALGORITHM.tex` (paste into Overleaf).
