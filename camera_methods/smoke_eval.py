"""Smoke metrics for one method: pred.csv vs GT on the PROVISIONAL split.

These numbers validate plumbing only — the real evaluation protocol is frozen
later, in the notebook's eval section.
"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from common import RUNS, load_index

method = sys.argv[1]
pred = pd.read_csv(os.path.join(RUNS, method, "pred.csv"))
idx = load_index().set_index("filename")
j = pred.join(idx, on="frame", rsuffix="_gt")
dp = np.hypot(j.x - j.x_gt, j.y - j.y_gt)
line = (f"[SMOKE | provisional split] {method}: n={len(j)}  "
        f"pos err median {dp.median()*100:.1f} / mean {dp.mean()*100:.1f} / "
        f"p90 {dp.quantile(0.9)*100:.1f} cm")
if j.yaw.notna().any():
    dy = np.abs((j.yaw - j.yaw_gt + np.pi) % (2 * np.pi) - np.pi)
    line += f"  |  yaw median {np.degrees(dy.median()):.2f} deg"
print(line)
for c in ["good", "blurred", "degenerate", "stationary-dup"]:
    m = (j.cls == c).values
    if m.any():
        print(f"    {c:15s} n={m.sum():4d}  median {dp[m].median()*100:7.1f} cm")
