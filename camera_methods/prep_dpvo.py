"""R3 adapter: frames dir + calib line + GT (TUM) for DPVO.

DPVO undistorts internally from the calib line (full plumb_bob supported) and
crops to /16 divisibility. GT timestamps must be DPVO's OUTPUT INDICES
(0..N-1 over the strided list), not frame numbers — evo associates on timestamp.
"""
import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CX, CY, DIST, FX, FY, IMAGES, RUNS, cam_to_world, load_index

out = os.path.join(RUNS, "dpvo")
frames_d = os.path.join(out, "frames")
os.makedirs(frames_d, exist_ok=True)
idx = load_index().reset_index(drop=True)
for i, fn in enumerate(idx.filename):
    dst = os.path.join(frames_d, f"frame_{i:06d}.jpg")
    if not os.path.lexists(dst):
        os.symlink(os.path.join(IMAGES, fn), dst)
with open(os.path.join(out, "calib.txt"), "w") as f:
    f.write(f"{FX} {FY} {CX} {CY} {DIST[0]} {DIST[1]} {DIST[2]} {DIST[3]} {DIST[4]}\n")
with open(os.path.join(out, "gt_tum.txt"), "w") as f:      # stride=1, skip=0
    for t, r in enumerate(idx.itertuples()):
        T = cam_to_world(r.x, r.y, r.yaw)                  # camera GT (DPVO outputs camera poses)
        q = Rotation.from_matrix(T[:3, :3]).as_quat()
        f.write(f"{t} {T[0, 3]} {T[1, 3]} {T[2, 3]} {q[0]} {q[1]} {q[2]} {q[3]}\n")
print("dpvo inputs ready:", len(idx), "frames")
