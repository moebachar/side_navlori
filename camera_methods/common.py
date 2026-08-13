"""Shared contract for all camera-method adapters.

Environment: DATA (default /content/data), RUNS (default /content/runs),
INDEX (default $RUNS/cam_index.csv — written by the notebook from the Section-2b
`cam` substrate table). Every runner ends by writing $RUNS/<method>/pred.csv
with columns: frame (original image filename), x, y, yaw (base_footprint pose
in the map frame; yaw may be NaN when a method cannot estimate it).
"""
import os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

DATA = os.environ.get("DATA", "/content/data")
RUNS = os.environ.get("RUNS", "/content/runs")
INDEX = os.environ.get("INDEX", os.path.join(RUNS, "cam_index.csv"))
IMAGES = os.path.join(DATA, "camera", "images")

# Nominal Pi Cam v2 intrinsics @ 820x616 (data/calib/camera_intrinsics_nominal.yaml)
FX, FY = 644.1408245617476, 641.7347972317088
CX, CY = 398.5361240843924, 310.506616520141
K = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])
DIST = np.array([0.1639958233797625, -0.271840030972792,
                 0.001055841660100477, -0.00166555973740089, 0.0])

# base_footprint -> camera_rgb_optical_frame (data/calib/extrinsics.yaml, exact)
T_BASE_CAM = np.eye(4)
T_BASE_CAM[:3, :3] = Rotation.from_quat(
    [-0.4999998415, 0.4996018366, -0.4999998415, 0.5003981634]).as_matrix()
T_BASE_CAM[:3, 3] = [0.076, 0.0, 0.103]
T_CAM_BASE = np.linalg.inv(T_BASE_CAM)


def load_index():
    df = pd.read_csv(INDEX)
    df["db_keep"] = df["db_keep"].astype(bool)
    return df


def T_world_base(x, y, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[0, 0], T[0, 1], T[1, 0], T[1, 1] = c, -s, s, c
    T[0, 3], T[1, 3] = x, y
    return T


def cam_to_world(x, y, yaw):
    """4x4 cam-to-world of the optical frame (OpenCV axes) for a planar base pose."""
    return T_world_base(x, y, yaw) @ T_BASE_CAM


def base_from_cam(T_wc):
    """(x, y, yaw) of base_footprint from a camera cam-to-world matrix."""
    T_wb = T_wc @ T_CAM_BASE
    return T_wb[0, 3], T_wb[1, 3], float(np.arctan2(T_wb[1, 0], T_wb[0, 0]))


def undistort_into(filenames, dst_dir, resize=None, names=None):
    """cv2.undistort with the nominal calibration (K preserved), optional resize."""
    import cv2
    os.makedirs(dst_dir, exist_ok=True)
    for i, fn in enumerate(filenames):
        name = names[i] if names else fn
        dst = os.path.join(dst_dir, name)
        if os.path.exists(dst):
            continue
        img = cv2.undistort(cv2.imread(os.path.join(IMAGES, fn)), K, DIST)
        if resize:
            img = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)
        params = [cv2.IMWRITE_JPEG_QUALITY, 95] if dst.lower().endswith((".jpg", ".jpeg")) else []
        cv2.imwrite(dst, img, params)


def write_pred(method, frames, x, y, yaw, extra=None):
    d = os.path.join(RUNS, method)
    os.makedirs(d, exist_ok=True)
    df = pd.DataFrame(dict(frame=list(frames), x=x, y=y, yaw=yaw))
    for k, v in (extra or {}).items():
        df[k] = v
    p = os.path.join(d, "pred.csv")
    df.to_csv(p, index=False)
    return p
