"""One video per packaged TIAGo run: head camera with AprilTag detections | merged map with the SLAM path growing.

Also writes ground_truth/gt_on_map.png (the run's path on the merged map). Video at 2x speed (every 2nd camera frame,
30 fps out), 1120x480. Written to <run>/ground_truth/video_tags_slam.mp4.

  python make_videos.py --robot .../package/robot --merged .../tiago_work/merged [runs...]
"""
import argparse, os
import numpy as np, pandas as pd, cv2

PANEL = 480


def detector():
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), p)


def map_base(merged, tags):
    P = np.load(os.path.join(merged, "merged_points.npz"))["xy"]
    lo, hi = P.min(0) - 1.0, P.max(0) + 1.0
    s = (PANEL - 20) / max(hi - lo)
    off = np.array([10, 10]) + ((PANEL - 20) - (hi - lo) * s) / 2
    to_px = lambda xy: np.stack([off[0] + (np.asarray(xy)[..., 0] - lo[0]) * s, PANEL - (off[1] + (np.asarray(xy)[..., 1] - lo[1]) * s)], -1)
    img = np.full((PANEL, PANEL, 3), 255, np.uint8)
    for x, y in to_px(P).astype(int):
        if 0 <= x < PANEL and 0 <= y < PANEL:
            img[y, x] = (170, 170, 170)
    return img, to_px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", required=True); ap.add_argument("--merged", required=True); ap.add_argument("runs", nargs="*")
    a = ap.parse_args()
    runs = a.runs or sorted(r for r in os.listdir(a.robot) if r.startswith("tiago_run") or r.startswith("tiago_demo"))
    det = detector()
    for run in runs:
        d = os.path.join(a.robot, run)
        tags = {}
        for line in open(os.path.join(d, "calib", "apriltags.yaml")):
            if "estimated:" in line:
                k = int(line.split(":")[0]); xy = line.split("[")[1].split("]")[0].split(",")
                tags[k] = (float(xy[0]), float(xy[1]))
        base, to_px = map_base(a.merged, tags)
        for k, xy in tags.items():
            c = tuple(to_px(xy).astype(int))
            cv2.rectangle(base, (c[0] - 3, c[1] - 3), (c[0] + 3, c[1] + 3), (60, 60, 60), -1)
            cv2.putText(base, str(k), (c[0] + 4, c[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60, 60, 60), 1, cv2.LINE_AA)
        gt = pd.read_csv(os.path.join(d, "ground_truth", "gt_pose.csv"))
        gpx = to_px(gt[["x", "y"]].values)
        # still image of the whole path
        still = base.copy()
        cv2.polylines(still, [gpx.astype(np.int32)], False, (200, 80, 20), 2, cv2.LINE_AA)
        cv2.circle(still, tuple(gpx[0].astype(int)), 5, (0, 160, 0), -1)
        cv2.putText(still, run, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(d, "ground_truth", "gt_on_map.png"), still)
        cam = pd.read_csv(os.path.join(d, "camera", "camera.csv"))
        out = cv2.VideoWriter(os.path.join(d, "ground_truth", "video_tags_slam.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640 + PANEL, 480))
        t0 = gt.t_ns.iloc[0]
        for k in range(0, len(cam), 2):
            r = cam.iloc[k]
            img = cv2.imread(os.path.join(d, "camera", "images", r.filename))
            if img is None:
                continue
            corners, ids, _ = det.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            seen = []
            if ids is not None:
                for c, i in zip(corners, ids.ravel()):
                    c = c.reshape(4, 2).astype(np.int32)
                    cv2.polylines(img, [c], True, (0, 220, 0), 2, cv2.LINE_AA)
                    cv2.putText(img, f"tag {i}", tuple(c[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2, cv2.LINE_AA)
                    seen.append(int(i))
            j = int(np.searchsorted(gt.t_ns.values, r.t_ns)); j = min(max(j, 1), len(gt) - 1)
            m = base.copy()
            cv2.polylines(m, [gpx[:j + 1].astype(np.int32)], False, (200, 80, 20), 2, cv2.LINE_AA)
            p = gpx[j]; yaw = gt.yaw.values[j]
            tip = p + 12 * np.array([np.cos(yaw), -np.sin(yaw)])
            cv2.circle(m, tuple(p.astype(int)), 5, (0, 0, 220), -1)
            cv2.arrowedLine(m, tuple(p.astype(int)), tuple(tip.astype(int)), (0, 0, 220), 2, tipLength=0.4)
            for i in seen:
                if i in tags:
                    c = to_px(tags[i]).astype(int)
                    cv2.line(m, tuple(p.astype(int)), tuple(c), (0, 200, 0), 1, cv2.LINE_AA)
                    cv2.rectangle(m, (c[0] - 5, c[1] - 5), (c[0] + 5, c[1] + 5), (0, 200, 0), 2)
            cv2.putText(m, f"{run}  t = {(r.t_ns - t0) / 1e9:5.1f} s", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img, "head camera (2x speed)", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            out.write(np.hstack([img, m]))
        out.release()
        print(f"{run}: video + gt_on_map.png", flush=True)
    print("VIDEOS_DONE")


if __name__ == "__main__":
    main()
