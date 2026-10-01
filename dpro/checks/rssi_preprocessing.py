"""Which RSSI preprocessing could help? Measurement checks on the 12 golden runs, against the truth.

1. The ~5 BSSIDs a radio broadcasts: are their readings in one scan independent samples (averaging them would
   cut noise, like the repeated samples of a static robot) or copies of one measurement?
2. Aggregating those BSSIDs: max (current) vs mean vs median, and timing each reading at its own instant
   (scan end - last_seen_ms) instead of the scan's mid-time. Scored by the residual against a log-distance
   model fitted with the true positions (an oracle AP map, pooled over all runs).
3. Stale readings: residual by reading age.
4. Censoring: weak APs are reported only when noise lifts them over the detection threshold, so the
   readings that exist near the threshold are biased high. Residual by predicted RSSI.

  python -m dpro.checks.rssi_preprocessing
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

REPO = os.environ.get("SIDE_NAVLORI", "/mnt/x/side_navlori")
RUNS = [f"golden_run_{i}" for i in range(1, 13)]
h = lambda xy, th: th[2] - 10 * th[3] * np.log10(np.hypot(xy[:, 0] - th[0], xy[:, 1] - th[1]) + 0.5)


def load():
    sys.path.append(f"{REPO}/dataset_pipeline/export")
    from load_dataset import Dataset
    out = []
    for r in RUNS:
        d = Dataset(f"{REPO}/data/{r}")
        w = d.wifi[d.wifi.ssid.str.contains("CESI", case=False, na=False) & (d.wifi.last_seen_ms <= 4000)].copy()
        w["radio"] = w.bssid.str[:-1] + np.where(w.freq_mhz < 3000, "/2G", "/5G")
        w["t_mid"] = (w.t_start_ns + w.t_end_ns) // 2
        w["t_own"] = w.t_end_ns - (w.last_seen_ms * 1e6).astype("int64")
        for c in ("mid", "own"):
            x, y, _ = d.gt_at(w[f"t_{c}"].values)
            w[f"x_{c}"], w[f"y_{c}"] = x, y
        w["run"] = r
        out.append(w)
    return pd.concat(out, ignore_index=True)


def fit(xy, y):
    i = int(np.argmax(y)); best = None
    for dx, dy in [(0, 0), (3, 0), (-3, 0), (0, 3), (0, -3)]:
        r = least_squares(lambda t: h(xy, t) - y, [xy[i, 0] + dx, xy[i, 1] + dy, min(y[i] + 5, -1), 2.5],
                          loss="soft_l1", f_scale=4.0, bounds=([-20, -20, -90, 1.0], [90, 50, 0, 6.0]))
        if best is None or r.cost < best.cost:
            best = r
    return best.x


def residuals(df, value, when):
    """Fit one oracle model per radio on the aggregated readings, return the residuals (dB)."""
    res = []
    for radio, g in df.groupby("radio"):
        if len(g) < 15:
            continue
        xy = g[[f"x_{when}", f"y_{when}"]].values; y = g[value].values
        th = fit(xy, y)
        res.append(pd.DataFrame(dict(radio=radio, r=y - h(xy, th), pred=h(xy, th), age=g.age.values)))
    return pd.concat(res, ignore_index=True)


def rstd(r):
    return 1.4826 * np.median(np.abs(r - np.median(r)))


def main():
    w = load()
    print(f"{len(w)} readings, {w.radio.nunique()} radios, {w.bssid.nunique()} BSSIDs")

    # 1. virtual BSSIDs of one radio within one scan
    g = w.groupby(["run", "scan_idx", "radio"])
    agg = g.agg(n=("rssi_dbm", "size"), mx=("rssi_dbm", "max"), mn=("rssi_dbm", "min"), mean=("rssi_dbm", "mean"),
                med=("rssi_dbm", "median"), age=("last_seen_ms", "min"), nage=("last_seen_ms", "nunique"),
                x_mid=("x_mid", "mean"), y_mid=("y_mid", "mean"), x_own=("x_own", "mean"), y_own=("y_own", "mean"),
                t_mid=("t_mid", "first")).reset_index()
    multi = agg[agg.n >= 2]
    spread = multi.mx - multi.mn
    print(f"\n1. BSSIDs per radio per scan: median {agg.n.median():.0f}, max {agg.n.max()}")
    print(f"   radio-scans with >= 2 BSSIDs: {len(multi)}; all BSSIDs read the same value: {np.mean(spread == 0):.0%}")
    print(f"   spread max-min within a radio-scan: median {spread.median():.1f} dB, 90th pct {spread.quantile(0.9):.1f} dB")
    print(f"   BSSIDs of one radio share the same age (same measurement instant): {np.mean(multi.nage == 1):.0%}")

    # 2. aggregation and timing, scored by the oracle residual
    print("\n2. residual vs an oracle log-distance model (true positions), robust std in dB (lower = cleaner):")
    for value, when, label in [("mx", "mid", "max over BSSIDs, scan mid-time (current)"),
                               ("mean", "mid", "mean over BSSIDs, scan mid-time"),
                               ("med", "mid", "median over BSSIDs, scan mid-time"),
                               ("mx", "own", "max over BSSIDs, each reading at its own time"),
                               ("mean", "own", "mean over BSSIDs, each reading at its own time")]:
        R = residuals(agg, value, when)
        print(f"   {label:50s} {rstd(R.r):.2f} dB  (std {R.r.std():.2f})")

    # 3. staleness and 4. censoring, on the current aggregation
    R = residuals(agg, "mx", "mid")
    print("\n3. residual by reading age (current aggregation):")
    for lo, hi in [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4001)]:
        s = R[(R.age >= lo) & (R.age < hi)].r
        print(f"   age {lo / 1000:.0f}-{hi / 1000:.0f} s: n {len(s):5d}  mean {s.mean():+.2f}  robust std {rstd(s):.2f} dB")
    print("\n4. residual by predicted RSSI (censoring check; detection floor ~ -90 dBm):")
    for lo, hi in [(-120, -85), (-85, -80), (-80, -75), (-75, -70), (-70, -60), (-60, 0)]:
        s = R[(R.pred >= lo) & (R.pred < hi)].r
        if len(s):
            print(f"   predicted {lo:4d} to {hi:4d} dBm: n {len(s):5d}  mean residual {s.mean():+.2f} dB  robust std {rstd(s):.2f}")

    # 5. is the model error a property of the place? residual agreement between readings of the same radio
    # taken close together, in different runs (other day/time) and in the same run (a revisit)
    R2 = residuals_with_meta(agg)
    print("\n5. same radio, readings within d metres: correlation of their residuals (1 = the error repeats exactly)")
    for d_max in (0.5, 1.0, 2.0, 4.0, 8.0):
        cross, same = [], []
        for radio, g in R2.groupby("radio"):
            P = g[["x", "y"]].values; D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
            iu = np.triu_indices(len(g), 1); near = D[iu] < d_max
            a, b = iu[0][near], iu[1][near]
            run, sc, r = g.run.values, g.scan_idx.values, g.r.values
            other = run[a] != run[b]; revisit = (run[a] == run[b]) & (np.abs(sc[a] - sc[b]) >= 7)
            cross += list(zip(r[a][other], r[b][other])); same += list(zip(r[a][revisit], r[b][revisit]))
        c = np.array(cross); s = np.array(same)
        cc = np.corrcoef(c.T)[0, 1] if len(c) > 10 else float("nan"); cs = np.corrcoef(s.T)[0, 1] if len(s) > 10 else float("nan")
        print(f"   d < {d_max:3.1f} m: other runs r = {cc:.2f} ({len(c)} pairs) | same run, >= 7 scans apart r = {cs:.2f} ({len(s)} pairs)")
    print("   (r = share of the residual variance that is a property of the place)")

    # 6. robot body / antenna: residual by the AP's bearing relative to the robot heading
    print("\n6. residual by the AP's direction relative to the robot heading (0 = AP straight ahead):")
    b = (np.rad2deg(R2.rel) + 360) % 360
    for lo in range(0, 360, 45):
        s = R2.r[(b >= lo) & (b < lo + 45)]
        print(f"   {lo:3d}-{lo + 45:3d} deg: n {len(s):5d}  mean residual {s.mean():+.2f} dB")
    near = R2.dist < 3
    print(f"   (APs closer than 3 m only: ahead +-45 deg {R2.r[near & ((b < 45) | (b >= 315))].mean():+.2f} dB, "
          f"behind +-45 deg {R2.r[near & (b >= 135) & (b < 225)].mean():+.2f} dB)")


def residuals_with_meta(df):
    sys.path.append(f"{REPO}/dataset_pipeline/export")
    from load_dataset import Dataset
    yaw = {}
    for r in RUNS:
        d = Dataset(f"{REPO}/data/{r}")
        sub = df[df.run == r]
        yaw.update({(r, s): v for s, v in zip(sub.scan_idx.values, d.gt_at(sub.t_mid.values)[2])} if "t_mid" in sub else {})
    out = []
    for radio, g in df.groupby("radio"):
        if len(g) < 15:
            continue
        xy = g[["x_mid", "y_mid"]].values; y = g["mx"].values; th = fit(xy, y)
        hd = np.array([yaw.get((a, s), np.nan) for a, s in zip(g.run.values, g.scan_idx.values)])
        bearing = np.arctan2(th[1] - xy[:, 1], th[0] - xy[:, 0])
        out.append(pd.DataFrame(dict(radio=radio, run=g.run.values, scan_idx=g.scan_idx.values, x=xy[:, 0], y=xy[:, 1],
                                     r=y - h(xy, th), rel=bearing - hd, dist=np.hypot(xy[:, 0] - th[0], xy[:, 1] - th[1]))))
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    main()
