"""Scan sequences for DPRO: the golden runs (real) and training segments.

A sequence is a dict: name, t (N,) seconds, G (N, 2) ground-truth position at mid-scan (m, building frame),
R (N, L) RSSI in dBm with nan = not heard, landmarks (L,) names, odom (N, 2) wheel-odometry position (real only).
Landmark = one physical radio on one band: BSSIDs that differ only in their last hex digit are merged (max RSSI).
"""
import os
import sys
import numpy as np
import pandas as pd

REPO = os.environ.get("SIDE_NAVLORI", "/mnt/x/side_navlori")
RUNS = [f"golden_run_{i}" for i in range(1, 13)]
ZONE = {r: ("west" if int(r.split("_")[-1]) <= 6 else "east") for r in RUNS}


def load_golden(runs=RUNS, repo=REPO, max_age_ms=4000):
    sys.path.append(f"{repo}/dataset_pipeline/export")
    from load_dataset import Dataset
    rows = []
    for r in runs:
        d = Dataset(f"{repo}/data/{r}")
        w = d.wifi[d.wifi.ssid.str.contains("CESI", case=False, na=False) & (d.wifi.last_seen_ms <= max_age_ms)].copy()
        w["lm"] = w.bssid.str[:-1] + np.where(w.freq_mhz < 3000, "/2G", "/5G")
        sc = w.groupby("scan_idx").agg(t0=("t_start_ns", "min"), t1=("t_end_ns", "max")).sort_index()
        rows.append((r, d, w, sc))
    lms = sorted(set().union(*[set(w.lm) for _, _, w, _ in rows]))
    col = {l: i for i, l in enumerate(lms)}
    seqs = []
    for r, d, w, sc in rows:
        tm = ((sc.t0 + sc.t1) // 2).values
        x, y, _ = d.gt_at(tm)
        R = np.full((len(sc), len(lms)), np.nan)
        row = {s: i for i, s in enumerate(sc.index)}
        g = w.groupby(["scan_idx", "lm"]).rssi_dbm.max()
        for (s, l), v in g.items():
            R[row[s], col[l]] = v
        o = d.odom
        ox = np.interp(tm, o.t_ns.values, o.x.values); oy = np.interp(tm, o.t_ns.values, o.y.values)
        seqs.append(dict(name=r, zone=ZONE[r], t=(tm - tm[0]) / 1e9, G=np.c_[x, y], R=R, landmarks=lms,
                         odom=np.c_[ox, oy], t_ns=tm))
    return seqs


def segments(seq, length=15, stride=4):
    """Training segments of `length` consecutive scans (DPVO trains on 15-frame clips)."""
    N = len(seq["t"]); out = []
    if N < 10:
        return out
    for s in range(0, max(N - length, 0) + 1, stride):
        e = min(s + length, N)
        seg = {k: (v[s:e] if isinstance(v, np.ndarray) and len(v) == N else v) for k, v in seq.items()}
        seg["t"] = seg["t"] - seg["t"][0]
        keep = ~np.all(np.isnan(seg["R"]), 0)
        seg["R"] = seg["R"][:, keep]
        if "Rtrue" in seg:
            seg["Rtrue"] = seg["Rtrue"][:, keep]
        out.append(seg)
    return out
