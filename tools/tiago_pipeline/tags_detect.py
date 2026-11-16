"""TIAGo++ Strasbourg runs: detect AprilTags (tag36h11, 15 cm) in the head camera and place them in each run's map.

Input  : <processed>/<run>/images/*.jpg, camera.csv, camera_path.csv (camera optical-frame pose in the run's
         SLAM map for every image, from process_run.py), camera_intrinsics.txt (factory K, D, plumb_bob).
Output : <out>/<run>_tags.csv, one row per detection:
         t_ns, filename, tag, u, v (centre px), side_px, cx, cy, cz (tag centre in the camera optical frame, m),
         mx, my, mz (tag centre in the run's map frame, m), dist_m, view_deg (angle between the camera ray and
         the tag normal), reproj_px.

  python tags_detect.py --processed /root/navlori/tiago_raw/navlori/processed --out /root/navlori/tiago_work/tags [runs...]
"""
import argparse, glob, os
import numpy as np, pandas as pd, cv2
from scipy.spatial.transform import Rotation as Rot

SIZE = 0.15
OBJ = SIZE * np.array([[-0.5, 0.5, 0], [0.5, 0.5, 0], [0.5, -0.5, 0], [-0.5, -0.5, 0]], dtype=np.float64)   # TL TR BR BL


def intrinsics(path):
    K = D = None
    for line in open(path):
        if line.startswith("K "):
            K = np.array([float(v) for v in line[2:].strip(" []\n").split(",")]).reshape(3, 3)
        elif line.startswith("D "):
            D = np.array([float(v) for v in line[2:].strip(" []\n").split(",")])
    return K, D


def detector():
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), p)


def run_one(proc, run, out, stride, valid):
    d = os.path.join(proc, run)
    K, D = intrinsics(os.path.join(d, "camera_intrinsics.txt"))
    cp = pd.read_csv(os.path.join(d, "camera_path.csv"))
    det, rows = detector(), []
    for k in range(0, len(cp), stride):
        r = cp.iloc[k]
        g = cv2.imread(os.path.join(d, "images", r.filename), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        corners, ids, _ = det.detectMarkers(g)
        if ids is None:
            continue
        Rmc = Rot.from_quat([r.qx, r.qy, r.qz, r.qw])          # camera optical frame -> map
        tmc = np.array([r.x, r.y, r.z])
        for c, i in zip(corners, ids.ravel()):
            if int(i) not in valid:
                continue
            c = c.reshape(4, 2).astype(np.float64)
            ok, rv, tv = cv2.solvePnP(OBJ, c, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                continue
            proj, _ = cv2.projectPoints(OBJ, rv, tv, K, D)
            rep = float(np.sqrt(((proj.reshape(4, 2) - c) ** 2).sum(1)).mean())
            tv = tv.ravel(); n = cv2.Rodrigues(rv)[0][:, 2]
            view = float(np.degrees(np.arccos(abs(n @ (-tv / np.linalg.norm(tv))))))
            m = Rmc.apply(tv) + tmc
            rows.append(dict(t_ns=int(r.t_ns), filename=r.filename, tag=int(i), u=c[:, 0].mean(), v=c[:, 1].mean(),
                             side_px=cv2.arcLength(c.astype(np.float32), True) / 4, cx=tv[0], cy=tv[1], cz=tv[2],
                             mx=m[0], my=m[1], mz=m[2], dist_m=float(np.linalg.norm(tv)), view_deg=view, reproj_px=rep))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, f"{run}_tags.csv"), index=False)
    if len(df):
        s = df.groupby("tag").agg(n=("tag", "size"), mx=("mx", "median"), my=("my", "median"), mz=("mz", "median"),
                                  dist=("dist_m", "median"))
        s["spread"] = 0.0
        for t in s.index:            # within-run consistency: median distance of each detection to the tag's median
            g = df[df.tag == t]
            s.loc[t, "spread"] = float(np.median(np.hypot(g.mx - g.mx.median(), g.my - g.my.median())))
        print(f"{run}: {len(df)} detections, tags " + ", ".join(f"{t}(n={int(s.loc[t,'n'])}, d={s.loc[t,'dist']:.1f} m, "
              f"spread {100*s.loc[t,'spread']:.0f} cm, z {s.loc[t,'mz']:.2f})" for t in s.index), flush=True)
    else:
        print(f"{run}: no detections", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=2); ap.add_argument("runs", nargs="*")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    runs = a.runs or sorted(r for r in os.listdir(a.processed) if os.path.isdir(os.path.join(a.processed, r)))
    for run in runs:
        run_one(a.processed, run, a.out, a.stride, valid=set(range(1, 30)))


if __name__ == "__main__":
    main()
