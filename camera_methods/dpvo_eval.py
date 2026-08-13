"""Sim(3)-align a DPVO trajectory to camera GT; report translation ATE.

Monocular VO has arbitrary scale and its own world origin, so Umeyama alignment
with scale correction (-as) is mandatory. Rotation metrics are NOT reported
(camera-frame trajectory vs base-frame GT differ by the extrinsic lever arm).
Runs inside the DPVO venv (needs evo, numpy only — no pandas/common.py).

Usage: python dpvo_eval.py <trajectory.txt> [variant_name]
"""
import csv
import json
import os
import sys
from evo.core import metrics, sync
from evo.tools import file_interface

RUNS = os.environ.get("RUNS", "/content/runs")
traj_file = sys.argv[1]
variant = sys.argv[2] if len(sys.argv) > 2 else "dpvo"
gt = file_interface.read_tum_trajectory_file(os.path.join(RUNS, "dpvo", "gt_tum.txt"))
est = file_interface.read_tum_trajectory_file(traj_file)
gt_s, est_s = sync.associate_trajectories(gt, est)
est_s.align(gt_s, correct_scale=True)
ape = metrics.APE(metrics.PoseRelation.translation_part)
ape.process_data((gt_s, est_s))
stats = {k: float(v) for k, v in ape.get_all_statistics().items()}

names = [row["filename"] for row in
         csv.DictReader(open(os.path.join(RUNS, "cam_index.csv")))]
outd = os.path.join(RUNS, "dpvo")
suffix = "" if variant == "dpvo" else f"_{variant}"
with open(os.path.join(outd, f"pred{suffix}.csv"), "w") as f:
    f.write("frame,x,y,yaw\n")
    for t, p in zip(est_s.timestamps.astype(int), est_s.positions_xyz):
        f.write(f"{names[t]},{p[0]},{p[1]},nan\n")
json.dump(stats, open(os.path.join(outd, f"ate{suffix}.json"), "w"), indent=1)
print(f"[SMOKE] {variant}: Sim(3)-aligned translation ATE over {len(est_s.timestamps)} frames — "
      f"rmse {stats['rmse']*100:.1f} cm, median {stats['median']*100:.1f} cm, "
      f"max {stats['max']*100:.1f} cm")
