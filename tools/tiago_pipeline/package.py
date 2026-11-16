"""Package the TIAGo++ Strasbourg runs in the golden-run layout (same folders as the TurtleBot golden runs).

For each run: camera/, imu/, wheel_odom/, wifi/, lidar/, ground_truth/, calib/, README.md.
Sources: the ROS 1 bag (lidar, joint states, static TF), process_run.py outputs (CSVs, images, SLAM path),
tags_detect.py, tag_network.py and merge_map.py outputs (common frame, tag positions, agreement).

  python package.py --bags .../bags --processed .../processed --work /root/navlori/tiago_work --out .../package/robot
"""
import argparse, json, os, shutil
import numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as Rot
from rosbags.highlevel import AnyReader
from pathlib import Path

NAMES = {**{f"run{k:02d}": f"tiago_run_{k:02d}" for k in range(1, 13)}, "run_final": "tiago_demo_1", "run_final4": "tiago_demo_2"}
WHEEL_R, HALF_TRACK = 0.0985, 0.2022
PAL_CAL = dict(left_wheel_radius_multiplier=1.004832487, right_wheel_radius_multiplier=1.011583756, wheel_separation_multiplier=1.025037129)


def rot2(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def read_bag(bag):
    scans, joints, static = [], [], {}
    with AnyReader([Path(bag)]) as r:
        cons = [c for c in r.connections if c.topic in ("/scan_raw", "/joint_states", "/tf_static")]
        for c, t, raw in r.messages(connections=cons):
            m = r.deserialize(raw, c.msgtype)
            if c.topic == "/scan_raw":
                st = m.header.stamp.sec * 10**9 + m.header.stamp.nanosec
                scans.append((st, int(t), m.angle_min, m.angle_max, m.angle_increment, m.time_increment, m.range_min, m.range_max,
                              np.asarray(m.ranges, np.float32), m.header.frame_id))
            elif c.topic == "/joint_states":
                st = m.header.stamp.sec * 10**9 + m.header.stamp.nanosec
                idx = {n: k for k, n in enumerate(m.name)}
                if "wheel_left_joint" in idx and "wheel_right_joint" in idx:
                    li, ri = idx["wheel_left_joint"], idx["wheel_right_joint"]
                    v = m.velocity if len(m.velocity) else [np.nan] * len(m.name)
                    joints.append((st, int(t), m.position[li], m.position[ri], v[li], v[ri]))
            else:
                for tr in m.transforms:
                    tl, q = tr.transform.translation, tr.transform.rotation
                    static[(tr.header.frame_id, tr.child_frame_id)] = ([tl.x, tl.y, tl.z], [q.x, q.y, q.z, q.w])
    return scans, joints, static


def chain_to(static, child):
    """Pose of `child` in base_footprint by walking parents in the static tree (None if it is not static)."""
    parent = {c: (p, v) for (p, c), v in static.items()}
    p_, r_ = np.zeros(3), Rot.identity(); node = child; path = []
    while node != "base_footprint":
        if node not in parent:
            return None
        p, (xyz, q) = parent[node]; path.append((xyz, q)); node = p
    for xyz, q in reversed(path):
        p_ = p_ + r_.apply(xyz); r_ = r_ * Rot.from_quat(q)
    return p_, r_


def write_lidar(scans, d):
    os.makedirs(d, exist_ok=True)
    offs = np.cumsum([0] + [len(s[8]) for s in scans])
    rng = np.concatenate([s[8] for s in scans]); ang = []
    for s in scans:
        n = len(s[8]); ang.append(s[2] + np.arange(n) * (s[3] - s[2]) / max(n - 1, 1))
    np.savez_compressed(os.path.join(d, "scans.npz"), offsets=offs, ranges=rng, angles=np.concatenate(ang).astype(np.float32),
                        t_ns=np.array([s[0] for s in scans], np.int64),
                        beam_time_offset_ns=np.concatenate([np.arange(len(s[8])) * int(round(s[5] * 1e9)) for s in scans]).astype(np.int64))
    pd.DataFrame(dict(t_ns=[s[0] for s in scans], t_bag_ns=[s[1] for s in scans], angle_min=[s[2] for s in scans],
                      angle_max=[s[3] for s in scans], angle_increment_raw=[s[4] for s in scans], time_increment=[s[5] for s in scans],
                      range_min=[s[6] for s in scans], range_max=[s[7] for s in scans], n_beams=[len(s[8]) for s in scans],
                      frame_id=[s[9] for s in scans])).to_csv(os.path.join(d, "scans_meta.csv"), index=False)


def yaml_pose(name, p, r):
    q = r.as_quat(); rpy = r.as_euler("xyz", degrees=True)
    return (f"  {name}:\n    xyz: [{p[0]:.5f}, {p[1]:.5f}, {p[2]:.5f}]\n    quat_xyzw: [{q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f}]\n"
            f"    rpy_deg: [{rpy[0]:.3f}, {rpy[1]:.3f}, {rpy[2]:.3f}]\n")


def main():
    ap = argparse.ArgumentParser()
    for k in ("bags", "processed", "work", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("runs", nargs="*")
    a = ap.parse_args()
    net = json.load(open(os.path.join(a.work, "network", "network.json")))
    agree = json.load(open(os.path.join(a.work, "merged", "agreement.json")))
    tags = {int(k): v for k, v in net["tags"].items()}; tag_z = {int(k): v for k, v in net["tag_z"].items()}
    runs = a.runs or [r for r in NAMES if r in net["runs"]]
    summary = []
    for run in runs:
        name = NAMES[run]; src = os.path.join(a.processed, run); dst = os.path.join(a.out, name)
        print(f"== {run} -> {name}", flush=True)
        v = net["runs"][run]; th = np.radians(v["theta_deg"]); R = rot2(th); t = np.array([v["tx"], v["ty"]])
        for sub in ("camera", "imu", "wheel_odom", "wifi", "lidar", "ground_truth", "calib"):
            os.makedirs(os.path.join(dst, sub), exist_ok=True)
        scans, joints, static = read_bag(os.path.join(a.bags, f"{run}.bag"))
        # camera: images + timestamps + camera pose in the common frame
        if not os.path.exists(os.path.join(dst, "camera", "images")):
            shutil.copytree(os.path.join(src, "images"), os.path.join(dst, "camera", "images"))
        pd.read_csv(os.path.join(src, "camera.csv"))[["t_ns", "t_bag_ns", "filename"]].to_csv(os.path.join(dst, "camera", "camera.csv"), index=False)
        cp = pd.read_csv(os.path.join(src, "camera_path.csv"))
        xy = cp[["x", "y"]].values @ R.T + t
        q = (Rot.from_euler("z", th) * Rot.from_quat(cp[["qx", "qy", "qz", "qw"]].values)).as_quat()
        pd.DataFrame(dict(t_ns=cp.t_ns, filename=cp.filename, x=xy[:, 0], y=xy[:, 1], z=cp.z, qx=q[:, 0], qy=q[:, 1], qz=q[:, 2], qw=q[:, 3])
                     ).to_csv(os.path.join(dst, "camera", "camera_pose_gt.csv"), index=False)
        # imu, odometry, joints, wifi (raw columns only; ground truth lives in ground_truth/)
        imu = pd.read_csv(os.path.join(src, "imu.csv"))
        imu[["t_ns", "t_bag_ns", "wx", "wy", "wz", "ax", "ay", "az", "qx", "qy", "qz", "qw"]].to_csv(os.path.join(dst, "imu", "imu.csv"), index=False)
        od = pd.read_csv(os.path.join(src, "odom.csv"))
        od[["t_ns", "t_bag_ns", "x", "y", "yaw", "v_lin", "w_ang"]].to_csv(os.path.join(dst, "wheel_odom", "odom.csv"), index=False)
        pd.DataFrame(joints, columns=["t_ns", "t_bag_ns", "left_pos_rad", "right_pos_rad", "left_vel_radps", "right_vel_radps"]
                     ).to_csv(os.path.join(dst, "wheel_odom", "joint_states.csv"), index=False)
        wf = pd.read_csv(os.path.join(src, "wifi.csv"))
        wf[["t_start_ns", "t_end_ns", "t_bag_ns", "scan_idx", "bssid", "ssid", "rssi_dbm", "freq_mhz", "last_seen_ms"]].to_csv(
            os.path.join(dst, "wifi", "wifi.csv"), index=False)
        shutil.copy(os.path.join(src, "wifi_raw.jsonl"), os.path.join(dst, "wifi", "wifi_raw.jsonl"))
        write_lidar(scans, os.path.join(dst, "lidar"))
        # ground truth: common frame + the raw per-run SLAM path
        g = pd.read_csv(os.path.join(src, "gt_slam.csv"))
        gx = g[["x", "y"]].values @ R.T + t
        pd.DataFrame(dict(t_ns=g.t_ns, x=gx[:, 0], y=gx[:, 1], yaw=np.angle(np.exp(1j * (g.yaw.values + th))), quality=g.quality,
                          inlier_frac=g.inlier_frac, slam_rmse_m=g.rmse_m)).to_csv(os.path.join(dst, "ground_truth", "gt_pose.csv"), index=False)
        g.to_csv(os.path.join(dst, "ground_truth", "gt_pose_slam_raw.csv"), index=False)
        td = pd.read_csv(os.path.join(a.work, "tags", f"{run}_tags.csv"))
        txy = td[["mx", "my"]].values @ R.T + t
        td["gx"], td["gy"] = txy[:, 0], txy[:, 1]
        td.to_csv(os.path.join(dst, "ground_truth", "tag_detections.csv"), index=False)
        mp = np.load(os.path.join(src, "map_points.npz"))["xy"].astype(float) @ R.T + t
        np.savez_compressed(os.path.join(dst, "ground_truth", "map_points.npz"), xy=mp.astype(np.float32))
        for f in ("path_map.png", "slam_quality.png", "odom_plots.png", "imu_plots.png", "wifi_map.png", "wifi_scans.png"):
            if os.path.exists(os.path.join(src, f)):
                shutil.copy(os.path.join(src, f), os.path.join(dst, "ground_truth", "slamframe_" + f))
        # calibration
        K = D = None
        for line in open(os.path.join(src, "camera_intrinsics.txt")):
            if line.startswith("K "): K = [float(x) for x in line[2:].strip(" []\n").split(",")]
            if line.startswith("D "): D = [float(x) for x in line[2:].strip(" []\n").split(",")]
        with open(os.path.join(dst, "calib", "camera_intrinsics.yaml"), "w") as f:
            f.write("# Head camera (Xtion RGB), 640x480, plumb_bob. FACTORY calibration from /xtion/rgb/camera_info;\n"
                    "# our own calibration is pending.\nimage_width: 640\nimage_height: 480\ncamera_name: xtion_rgb\n"
                    f"camera_matrix: {K}\ndistortion_model: plumb_bob\ndistortion_coefficients: {D}\n")
        # camera in base_footprint: median over the run (head and torso did not move)
        gt_t = g.t_ns.values; ext = []
        for _, r in cp.iloc[::30].iterrows():
            k = np.searchsorted(gt_t, r.t_ns); k = min(max(k, 1), len(gt_t) - 1)
            bx, by, byaw = g.x.values[k], g.y.values[k], g.yaw.values[k]
            Rb = Rot.from_euler("z", byaw)
            ext.append(np.concatenate([Rb.inv().apply([r.x - bx, r.y - by, r.z]), (Rb.inv() * Rot.from_quat([r.qx, r.qy, r.qz, r.qw])).as_rotvec()]))
        ext = np.median(np.array(ext), 0)
        with open(os.path.join(dst, "calib", "extrinsics.yaml"), "w") as f:
            f.write("# Sensor poses in base_footprint. Laser and IMU from /tf_static; camera = median over the run of\n"
                    "# (camera pose from TF) relative to (base pose), head and torso were not moved during the runs.\nframes:\n")
            f.write(yaml_pose("xtion_rgb_optical_frame", ext[:3], Rot.from_rotvec(ext[3:])))
            for fr in ("base_laser_link", "base_imu_link"):
                c = chain_to(static, fr)
                if c:
                    f.write(yaml_pose(fr, *c))
        with open(os.path.join(dst, "calib", "robot.yaml"), "w") as f:
            f.write("robot: PAL Robotics TIAGo++ (tiago-201c), base PMB2, ROS 1 Noetic\n"
                    f"wheel_radius_m: {WHEEL_R}\nwheel_separation_m: {2 * HALF_TRACK}\n"
                    f"pal_wheel_calibration: {json.dumps(PAL_CAL)}\n"
                    "laser: base laser, /scan_raw ~10 Hz\nimu: /base_imu ~10 Hz, z axis points DOWN (yaw rate up = -wz)\n"
                    "camera: head Xtion RGB, /xtion/rgb/image_raw/compressed 30 Hz, 640x480\n"
                    "odometry: /mobile_base_controller/odom 50 Hz; joint_states 50 Hz (wheel angles rad, velocities rad/s)\n"
                    "wifi: /wifi/rssi (iw scan on the robot, CESI_Recherche link), ~1 scan / 2.3 s\n")
        with open(os.path.join(dst, "calib", "apriltags.yaml"), "w") as f:
            f.write("# AprilTags tag36h11, 15.0 cm. Positions ESTIMATED by the tag network (no survey yet), common frame.\n"
                    "family: tag36h11\nsize_m: 0.150\ntags:\n")
            for k in sorted(tags):
                f.write(f"  {k}: {{estimated: [{tags[k][0]:.4f}, {tags[k][1]:.4f}], z_m: {tag_z.get(k, float('nan')):.3f}}}\n")
        # dataset card
        dur = (g.t_ns.iloc[-1] - g.t_ns.iloc[0]) / 1e9; length = float(np.hypot(*np.diff(gx, axis=0).T).sum())
        ag = agree.get(run, {}); ho = v.get("heldout_cm_median")
        nscan = wf.scan_idx.nunique()
        acc = (f"held-out tag error {ho:.0f} cm (median)" if ho is not None else "frame reference (held-out not applicable)") \
            if not run.startswith("run_final") else "placed on the tags solved from the 12 main runs (not part of the network)"
        with open(os.path.join(dst, "README.md"), "w") as f:
            f.write(f"# {name}\n\nPAL TIAGo++ (tiago-201c), CESI Strasbourg campus, recorded {'2026-11-12' if not run.startswith('run_final') else '2026-11-13'}"
                    f" (source bag `{run}.bag`). **{dur:.0f} s, {length:.1f} m**, tags seen: {v['tags']}.\n\n"
                    "| folder | what | rate |\n|---|---|---|\n"
                    f"| `camera/` | `images/<t_ns>.jpg` 640x480 + `camera.csv`; `camera_pose_gt.csv` (camera pose in the common frame) | {len(cp) / dur:.1f} Hz |\n"
                    f"| `imu/` | `imu.csv` (base IMU; z axis points down) | {len(imu) / dur:.1f} Hz |\n"
                    f"| `wheel_odom/` | `odom.csv`, `joint_states.csv` (wheel angles rad, velocities rad/s) | {len(od) / dur:.1f} Hz |\n"
                    f"| `wifi/` | `wifi.csv`, `wifi_raw.jsonl` | {nscan} scans (~1 per {dur / max(nscan, 1):.1f} s) |\n"
                    f"| `lidar/` | `scans.npz` (+ `scans_meta.csv`), base laser | {len(scans) / dur:.1f} Hz |\n"
                    "| `ground_truth/` | `gt_pose.csv` (common frame) + raw SLAM path, tag detections, map, plots | per laser scan |\n"
                    "| `calib/` | camera intrinsics (factory), extrinsics, robot, AprilTags (estimated) | |\n\n"
                    "## Ground truth\nLidar scan-to-map SLAM (point-to-line ICP, `process_run.py`), placed rigidly in the **common TIAGo frame** "
                    "(the start of tiago_run_01) by a joint adjustment over all AprilTag sightings of the 12 main runs. "
                    f"{acc}; walls agree with the other runs to {ag.get('median_cm', float('nan')):.1f} cm (median). "
                    "Tag positions are estimated, not surveyed: the frame will be re-registered on the floor plan later.\n")
        summary.append(dict(run=name, source_bag=f"{run}.bag", duration_s=round(dur, 1), path_m=round(length, 1), tags=" ".join(map(str, v["tags"])),
                            heldout_tag_cm=ho, walls_agree_cm=ag.get("median_cm"), camera_frames=len(cp), imu_hz=round(len(imu) / dur, 1),
                            odom_hz=round(len(od) / dur, 1), lidar_scans=len(scans), wifi_scans=nscan))
    pd.DataFrame(summary).to_csv(os.path.join(a.out, "tiago_common", "tiago_runs_summary.csv") if os.path.isdir(os.path.join(a.out, "tiago_common"))
                                 else os.path.join(a.out, "tiago_runs_summary.csv"), index=False)
    print("PACKAGE_DONE")


if __name__ == "__main__":
    main()
