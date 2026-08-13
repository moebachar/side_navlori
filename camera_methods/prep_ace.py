"""R1 adapter: our frames + GT -> ACE dataset layout.

Undistorted images (nominal calibration), 4x4 CAM-TO-WORLD poses in OpenCV
camera axes, full 3x3 K files with f_mean in both slots (ACE asserts fx == fy;
0.2%% averaging error is far below the nominal-intrinsics uncertainty).
"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CX, CY, FX, FY, RUNS, cam_to_world, load_index, undistort_into

scene = os.path.join(RUNS, "ace", "datasets", "navlori")
idx = load_index()
f_mean = (FX + FY) / 2.0
for split, sel in [("train", (idx.split == "train") & idx.db_keep),
                   ("test", idx.split == "test")]:
    rows = idx[sel].reset_index(drop=True)
    rgb = os.path.join(scene, split, "rgb")
    poses = os.path.join(scene, split, "poses")
    calib = os.path.join(scene, split, "calibration")
    names = [f"frame-{i:06d}.color.jpg" for i in range(len(rows))]
    undistort_into(list(rows.filename), rgb, names=names)
    os.makedirs(poses, exist_ok=True)
    os.makedirs(calib, exist_ok=True)
    lines = ["ace_name,frame"]
    for i, r in enumerate(rows.itertuples()):
        np.savetxt(os.path.join(poses, f"frame-{i:06d}.pose.txt"),
                   cam_to_world(r.x, r.y, r.yaw))
        with open(os.path.join(calib, f"frame-{i:06d}.calibration.txt"), "w") as f:
            f.write(f"{f_mean} 0 {CX}\n0 {f_mean} {CY}\n0 0 1\n")
        lines.append(f"frame-{i:06d}.color.jpg,{r.filename}")
    with open(os.path.join(scene, f"map_{split}.csv"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("ace", split, len(rows), "frames")
