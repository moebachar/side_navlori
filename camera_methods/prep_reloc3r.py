"""R2 adapter: fake 7Scenes 'heads' tree for Reloc3r.

CRITICAL: images are undistorted then resized to 640x480 by US — the repo's
loader crops a 640x480 window centered on the HARDCODED principal point (320,240),
which would silently discard the right/bottom bands of a raw 820x616 frame.
'heads' expects exactly 1000 db frames (seq-02) and 1000 queries (seq-01).
"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RUNS, cam_to_world, load_index, undistort_into

root = os.path.join(RUNS, "reloc3r", "7scenes", "heads")
idx = load_index()


def pick(rows, n):
    k = np.linspace(0, len(rows) - 1, n).round().astype(int)
    return rows.reset_index(drop=True).iloc[k].reset_index(drop=True)


db = pick(idx[(idx.split == "train") & idx.db_keep], 1000)
q = pick(idx[idx.split == "test"], 1000)
for seq, rows in [("seq-02", db), ("seq-01", q)]:
    d = os.path.join(root, seq)
    names = [f"frame-{i:06d}.color.png" for i in range(len(rows))]
    undistort_into(list(rows.filename), d, resize=(640, 480), names=names)
    for i, r in enumerate(rows.itertuples()):
        np.savetxt(os.path.join(d, f"frame-{i:06d}.pose.txt"),
                   cam_to_world(r.x, r.y, r.yaw))
    rows.filename.to_frame("frame").to_csv(os.path.join(root, f"map_{seq}.csv"))
    print("reloc3r", seq, len(rows), "frames")
