"""Recover Reloc3r absolute query poses via the official motion averaging.

eval_visloc.py caches per-pair poses but never writes final absolute poses;
this re-runs Reloc3rVisloc.motion_averaging (official API, code unchanged)
over the cached files and converts to base-frame pred.csv.

Usage: python reloc3r_post.py <repo_root> [topk]
"""
import os
import sys
import numpy as np
import pandas as pd
CM = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CM)
from common import RUNS, base_from_cam, write_pred

repo = os.path.abspath(sys.argv[1])
topk = int(sys.argv[2]) if len(sys.argv) > 2 else 10
sys.path.insert(0, repo)
from reloc3r.reloc3r_visloc import Reloc3rVisloc  # noqa: E402

cache = os.path.join(repo, "_db-q_pair_info")
mapping = pd.read_csv(os.path.join(RUNS, "reloc3r", "7scenes", "heads",
                                   "map_seq-01.csv"), index_col=0)
va = Reloc3rVisloc()
frames, xs, ys, yaws = [], [], [], []
for fid in range(1000):
    pdb, pq2d = [], []
    for k in range(topk):
        d = os.path.join(cache, f"poses_heads_pair-id={k}")
        pdb.append(np.loadtxt(os.path.join(d, f"seq-01_{fid:06d}_pose-db.txt")))
        pq2d.append(np.loadtxt(os.path.join(d, f"seq-01_{fid:06d}_pose-q2d.txt")))
    x, y, yaw = base_from_cam(va.motion_averaging(pdb, pq2d))
    frames.append(mapping.loc[fid, "frame"])
    xs.append(x)
    ys.append(y)
    yaws.append(yaw)
df = pd.DataFrame(dict(frame=frames, x=xs, y=ys, yaw=yaws)).drop_duplicates("frame")
write_pred("reloc3r", df.frame, df.x.values, df.y.values, df.yaw.values)
print("reloc3r: pred written for", len(df), "queries")
