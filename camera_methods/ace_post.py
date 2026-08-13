"""Convert ACE's poses file (WORLD-TO-CAM, qw-first) to base-frame pred.csv."""
import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RUNS, base_from_cam, write_pred

poses_file = sys.argv[1]
scene = os.path.join(RUNS, "ace", "datasets", "navlori")
m = pd.read_csv(os.path.join(scene, "map_test.csv")).set_index("ace_name")
frames, xs, ys, yaws, errs = [], [], [], [], []
for line in open(poses_file):
    p = line.split()
    name = p[0]
    qw, qx, qy, qz, tx, ty, tz = map(float, p[1:8])
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    T[:3, 3] = [tx, ty, tz]
    x, y, yaw = base_from_cam(np.linalg.inv(T))     # file is world-to-cam: invert
    frames.append(m.loc[name, "frame"])
    xs.append(x)
    ys.append(y)
    yaws.append(yaw)
    errs.append(float(p[9]))                        # ACE's own tr_err, meters
write_pred("ace", frames, xs, ys, yaws, extra=dict(ace_tr_err_m=errs))
print("ace: pred written for", len(frames), "test frames")
