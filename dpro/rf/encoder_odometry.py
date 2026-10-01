"""Phase 4 control: odometry from the wheel encoders alone (joint_states), no gyro.

The robot's /odom heading comes from the IMU gyro. In runs 4 and 9 that gyro has a constant bias (-21.9 and +4.6
deg/s against the truth) while the encoders do not (-0.14 and -0.11 deg/s). So the fault the WiFi gyro-bias check
finds is also visible from the encoders alone; this module measures how far plain encoder dead reckoning gets.

  python -m dpro.rf.encoder_odometry --out encoder_odometry.json
"""
import argparse, json, os, sys
import numpy as np

from ..data import load_golden
from .metrics import all_metrics

R_WHEEL, SEPARATION = 0.033, 0.287          # ROBOTIS spec, as in the vault's calibration file


def dead_reckon(j, t_ns):
    """Differential-drive integration of the wheel angles, sampled at t_ns."""
    l, r = j.left_pos_rad.values, j.right_pos_rad.values
    dl, dr = np.diff(l) * R_WHEEL, np.diff(r) * R_WHEEL
    ds, dth = (dl + dr) / 2, (dr - dl) / SEPARATION
    th = np.concatenate([[0], np.cumsum(dth)])
    mid = th[:-1] + dth / 2
    x = np.concatenate([[0], np.cumsum(ds * np.cos(mid))]); y = np.concatenate([[0], np.cumsum(ds * np.sin(mid))])
    return np.c_[np.interp(t_ns, j.t_ns.values, x), np.interp(t_ns, j.t_ns.values, y)], th, j.t_ns.values


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); args = ap.parse_args()
    repo = os.environ.get("SIDE_NAVLORI", "/mnt/x/side_navlori")
    sys.path.insert(0, f"{repo}/dataset_pipeline/export")
    from load_dataset import Dataset
    res = {}
    for s in load_golden():
        d = Dataset(f"{repo}/data/{s['name']}")
        X, th, tj = dead_reckon(d.joints, s["t_ns"])
        m = all_metrics(X, X, s["G"], s["t"])
        # heading drift of encoders and of /odom against the truth (deg/s)
        tt = np.arange(tj[0] + 5e9, tj[-1] - 1e9, 0.5e9)
        gt = np.unwrap(np.asarray(d.gt_at(tt)[2], float)); sec = (tt - tt[0]) / 1e9
        enc = np.interp(tt, tj, th); od = np.interp(tt, d.odom.t_ns.values, np.unwrap(d.odom.yaw.values))
        m["enc_drift_deg_s"] = float(np.rad2deg(np.polyfit(sec, enc - gt, 1)[0]))
        m["odom_drift_deg_s"] = float(np.rad2deg(np.polyfit(sec, od - gt, 1)[0]))
        res[s["name"]] = m
        print(f"{s['name']:14s} encoders only: aligned {m['aligned']:.2f} m, drift5 {m['drift5']:.2f} m | heading drift: encoders {m['enc_drift_deg_s']:+.2f}, /odom {m['odom_drift_deg_s']:+.2f} deg/s", flush=True)
    res["mean"] = {k: float(np.mean([v[k] for v in res.values()])) for k in ["aligned", "online", "drift5"]}
    print("mean", res["mean"])
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
