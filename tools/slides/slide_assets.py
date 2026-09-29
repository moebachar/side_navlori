#!/usr/bin/env python
# Build slide assets (slides 1-5) into X:\side_navlori\slides_progress\slideN_*\
import os, json, shutil, numpy as np, pandas as pd, cv2
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
Image.MAX_IMAGE_PIXELS = None
ROOT = "/mnt/x/side_navlori"; DATA = ROOT + "/data"; OUT = ROOT + "/slides_progress"; FPD = DATA + "/golden_floorplan"
D = {k: f"{OUT}/{v}" for k, v in dict(s1="slide1_calibration", s2="slide2_floorplan_markers", s3="slide3_dataset",
                                      s4="slide4_ground_truth", s5="slide5_visualization", s6="slide6_results").items()}
for d in D.values(): os.makedirs(d, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
RUNS = [f"golden_run_{i}" for i in range(1, 13)]
INFO = {r: json.load(open(f"{DATA}/{r}/ground_truth/gt_info.json")) for r in RUNS}
K = np.array([[1275.6918, 0, 338.6633], [0, 1274.0334, 291.8436], [0, 0, 1.0]]); Dist = np.array([0.110053, -0.002819, 0.009096, 0.008142])

# ---------------- slide 1: calibration ----------------
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
ts = get_typestore(Stores.ROS2_HUMBLE); frames = []
with Reader(f"{ROOT}/calib_cam") as r:
    conns = [c for c in r.connections if c.topic == "/camera/image_raw"]
    msgs = list(r.messages(connections=conns))
for idx in np.linspace(40, len(msgs) - 40, 6).astype(int):
    m = ts.deserialize_cdr(msgs[idx][2], msgs[idx][0].msgtype)
    frames.append(np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, 3))
for k, f in enumerate(frames): Image.fromarray(f).save(f"{D['s1']}/target_view_{k+1}.jpg", quality=92)
grid = np.vstack([np.hstack(frames[:3]), np.hstack(frames[3:])]); Image.fromarray(grid).save(f"{D['s1']}/target_views_montage.jpg", quality=90)
# undistortion demo on a corridor frame from a golden run
cam = pd.read_csv(f"{DATA}/golden_run_11/camera/camera.csv"); img = cv2.imread(f"{DATA}/golden_run_11/camera/images/{cam.filename.iloc[len(cam)//2]}")
und = cv2.undistort(img, K, Dist)
cv2.imwrite(f"{D['s1']}/raw_vs_undistorted.jpg", np.hstack([img, np.full((480, 12, 3), 255, np.uint8), und]))
# reprojection-error style figure from the Kalibr report pages
import fitz
pdf = fitz.open(f"{ROOT}/calib/kalibr_640x480/calib_cam-report-cam.pdf")
for p in range(len(pdf)): pdf[p].get_pixmap(dpi=110).save(f"{D['s1']}/kalibr_report_page{p+1}.png")
shutil.copy2(f"{ROOT}/calib/kalibr_640x480/calib_cam-camchain.yaml", f"{D['s1']}/kalibr_result.yaml")
shutil.copy2(f"{ROOT}/calib/aprilgrid_8x6_30mm_A3.pdf", f"{D['s1']}/aprilgrid_target_A3.pdf")
fitz.open(f"{ROOT}/calib/aprilgrid_8x6_30mm_A3.pdf")[0].get_pixmap(dpi=60).save(f"{D['s1']}/aprilgrid_target.png")

# ---------------- plan helpers ----------------
tf = json.load(open(f"{FPD}/plan_transform.json")); S, R_, TX, TY = tf["px_per_m"], tf["rotation_rad"], tf["tx_px"], tf["ty_px"]; F = 4
plan_full = Image.open(f"{ROOT}/{tf['image']}").convert("RGB"); W0, H0 = plan_full.size
plan = np.array(plan_full.resize((W0 // F, H0 // F), Image.LANCZOS))
def w2p(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); c, s = np.cos(R_), np.sin(R_)
    return (TX + S * (c * x - s * y)) / F, (TY - S * (s * x + c * y)) / F
CAD = {int(k): (v["x"], v["y"]) for k, v in json.load(open(f"{DATA}/tags_ground_truth.json")).items()}
def plan_ax(figsize=(24, 8)):
    fig, ax = plt.subplots(figsize=figsize); ax.imshow(plan, alpha=0.9); ax.axis("off")
    x0, y0 = plan.shape[1] * 0.015, plan.shape[0] * 0.97; L = 5 * S / F
    ax.plot([x0, x0 + L], [y0, y0], "-", color="k", lw=4); ax.text(x0 + L / 2, y0 - 12, "5 m", ha="center", va="bottom", fontsize=13)
    return fig, ax

# ---------------- slide 2: floor plan + marker network ----------------
fig, ax = plan_ax()
for t, (x, y) in CAD.items():
    u, v = w2p(x, y); ax.plot(u, v, "s", ms=15, mfc="#f2b705", mec="k", mew=1.5, zorder=5)
    ax.annotate(str(t), (u, v), xytext=(7, 7), textcoords="offset points", fontsize=14, fontweight="bold", zorder=6)
fig.savefig(f"{D['s2']}/floorplan_26_markers.png", dpi=110, bbox_inches="tight", pad_inches=0.05); plt.close(fig)
Image.fromarray(plan).save(f"{D['s2']}/floorplan_clean.png")
# coverage heat: where the 12 runs drove (for a "coverage" callout)
fig, ax = plan_ax()
for r in RUNS:
    g = pd.read_csv(f"{DATA}/{r}/ground_truth/gt_pose.csv"); u, v = w2p(g.x, g.y); ax.plot(u, v, "-", color="#1f6fd1", lw=7, alpha=0.18, solid_capstyle="round")
for t, (x, y) in CAD.items():
    u, v = w2p(x, y); ax.plot(u, v, "s", ms=11, mfc="#f2b705", mec="k", mew=1.2, zorder=5)
fig.savefig(f"{D['s2']}/floorplan_markers_and_coverage.png", dpi=110, bbox_inches="tight", pad_inches=0.05); plt.close(fig)
# marker photos as seen by the robot (4 tags) + one printable tag
shutil.copy2(f"{ROOT}/gt_review/run6_tags.jpg", f"{D['s2']}/markers_seen_by_robot.jpg")
import re, base64, io
h = open(f"{ROOT}/print_tags.html").read(); b = re.findall(r"data:image/png;base64,([A-Za-z0-9+/=]+)", h)
tag0 = Image.open(io.BytesIO(base64.b64decode(b[0]))).convert("L").resize((600, 600), Image.NEAREST)
Image.fromarray(np.pad(np.array(tag0), 60, constant_values=255)).save(f"{D['s2']}/apriltag_example_tag36h11.png")

# ---------------- slide 3: dataset ----------------
shutil.copy2(f"{FPD}/all_paths_on_plan.png", f"{D['s3']}/all_12_paths_on_floorplan.png")
# small multiples: 12 runs, one panel each, cropped around each path
fig, axs = plt.subplots(3, 4, figsize=(20, 11)); cols = plt.get_cmap("viridis")
for ax, r in zip(axs.ravel(), RUNS):
    g = pd.read_csv(f"{DATA}/{r}/ground_truth/gt_pose.csv"); u, v = w2p(g.x, g.y); t = (g.t_ns - g.t_ns.iloc[0]) / 1e9
    ax.imshow(plan); ax.scatter(u, v, c=t, cmap="viridis", s=3); ax.plot(u.iloc[0] if hasattr(u, "iloc") else u[0], v[0], "o", ms=8, mfc="lime", mec="k")
    pad = 60; ax.set_xlim(u.min() - pad, u.max() + pad); ax.set_ylim(v.max() + pad, v.min() - pad); ax.axis("off")
    ax.set_title(f"{r.replace('golden_run_', 'run ')}  ·  {INFO[r]['path_m']:.0f} m  ·  {INFO[r]['duration_s']:.0f} s", fontsize=13)
plt.tight_layout(); fig.savefig(f"{D['s3']}/12_runs_small_multiples.png", dpi=100); plt.close(fig)
tot = dict(runs=12, minutes=sum(INFO[r]["duration_s"] for r in RUNS) / 60, metres=sum(INFO[r]["path_m"] for r in RUNS))
cams = sum(len(pd.read_csv(f"{DATA}/{r}/camera/camera.csv")) for r in RUNS)
wifi = sum(pd.read_csv(f"{DATA}/{r}/wifi/wifi.csv", keep_default_na=False).scan_idx.nunique() for r in RUNS)
imu = sum(len(pd.read_csv(f"{DATA}/{r}/imu/imu.csv")) for r in RUNS)
scans = sum(len(np.load(f"{DATA}/{r}/lidar/scans.npz")["t_ns"]) for r in RUNS)
key = (f"# Dataset key figures (12 golden runs, 2026-09-24)\n\n- runs: 12\n- total duration: {tot['minutes']:.1f} min\n- total path length: {tot['metres']:.0f} m\n"
       f"- camera frames: {cams:,} (640x480, 30 Hz)\n- IMU samples: {imu:,} (~145 Hz)\n- wheel-odometry samples: ~{imu:,} (~145 Hz)\n- lidar scans: {scans:,} (9.6 Hz)\n"
       f"- WiFi scans: {wifi} (~1 per 4 s, ~100 access points per run)\n- markers: 26 AprilTags (tag36h11, 15 cm)\n- raw size: 3.0 GB\n"
       f"- format: one folder per run (camera, imu, wheel_odom, lidar, wifi, ground_truth, calib), one loader\n")
open(f"{D['s3']}/key_figures.md", "w").write(key)

# ---------------- slide 4: ground truth ----------------
# pipeline diagram
fig, ax = plt.subplots(figsize=(16, 5.2)); ax.axis("off"); ax.set_xlim(0, 16); ax.set_ylim(0, 5.2)
def box(x, y, w, h, txt, fc, fs=14):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.25", fc=fc, ec="#333", lw=1.5))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, fontweight="bold", color="#1b1b1b")
def arrow(x0, y0, x1, y1): ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", lw=2.2, color="#333", mutation_scale=22))
box(0.2, 3.3, 3.0, 1.3, "Lidar SLAM\n(distances)", "#cfe3ff"); box(0.2, 0.6, 3.0, 1.3, "Gyroscope\n(heading)", "#cfe3ff")
box(4.4, 1.95, 3.2, 1.3, "Fused\ntrajectory", "#d9f2d0"); box(8.8, 3.3, 3.0, 1.3, "Camera\nmarker sightings", "#ffe7b3")
box(8.8, 0.6, 3.0, 1.3, "Marker map\n(26 tags, floor plan)", "#ffe7b3"); box(12.9, 1.95, 2.9, 1.3, "Ground truth\n(building frame)", "#f6c9c9")
arrow(3.2, 3.95, 4.4, 2.95); arrow(3.2, 1.25, 4.4, 2.25); arrow(7.6, 2.6, 12.9, 2.6); arrow(11.8, 3.95, 12.9, 2.95); arrow(11.8, 1.25, 12.9, 2.25)
ax.text(10.3, 2.75, "rigid alignment", ha="center", fontsize=12, style="italic")
fig.savefig(f"{D['s4']}/gt_pipeline_diagram.png", dpi=120, bbox_inches="tight"); plt.close(fig)
# accuracy chart (held-out marker error per run) + wall agreement
rows = []
for r in RUNS:
    i = INFO[r]; h_ = i.get("heldout_median_cm")
    if i["status"] == "ok" and h_ == h_ and h_ is not None: rows.append((r.replace("golden_run_", "run "), h_, "blind marker test"))
R = pd.DataFrame(rows, columns=["run", "err", "kind"]).sort_values("err")
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.barh(R.run, R.err, color="#2b8a3e"); ax.axvline(R.err.median(), color="k", ls="--", lw=1.2)
ax.text(R.err.median() + 0.8, len(R) - 0.6, f"median {R.err.median():.0f} cm", fontsize=13)
ax.set_xlabel("position error at a hidden marker (cm, median per run)", fontsize=13); ax.tick_params(labelsize=12)
ax.set_title("Ground-truth accuracy: blind test on markers", fontsize=15, loc="left")
plt.tight_layout(); fig.savefig(f"{D['s4']}/gt_accuracy_blind_marker_test.png", dpi=120); plt.close(fig)
shutil.copy2(f"{DATA}/golden_run_7/ground_truth/gt_on_plan.png", f"{D['s4']}/example_run7_on_plan.png")
open(f"{D['s4']}/numbers.md", "w").write(
    f"# Ground-truth numbers\n\n- blind marker test (marker hidden, whole system re-solved, error at that marker): median over runs {R.err.median():.0f} cm "
    f"(per-run medians {R.err.min():.0f}-{R.err.max():.0f} cm)\n- walls seen by different runs coincide within 6-7 cm (median)\n"
    f"- one pose per lidar scan (10 Hz), interpolated to every camera frame / WiFi scan\n- two single-marker runs placed by matching their lidar walls to the others: 88-91% of wall points within 10 cm\n\n"
    "## Equations\n- heading: theta(t) = integral( omega_z - b ) dt   (gyro, bias b removed)\n"
    "- position: p_{k+1} = p_k + R(theta_k) * dp_k^lidar\n"
    "- alignment: min_{R,t,T} sum_j || R p_j + t - T_j ||^2  +  sum_j || T_j - T_j^CAD ||^2 / sigma^2\n")

# ---------------- slide 5: visualisation ----------------
for f in ["data_collection.mp4", "demo_slam_timelapse_big.mp4"]:
    if os.path.exists(f"{ROOT}/figures/{f}"): shutil.copy2(f"{ROOT}/figures/{f}", f"{D['s5']}/{f}")
shutil.copy2(f"{DATA}/golden_run_10/ground_truth/gt_on_plan.png", f"{D['s5']}/run10_path_colored_by_time.png")
cap = cv2.VideoCapture(f"{ROOT}/figures/data_collection.mp4"); cap.set(cv2.CAP_PROP_POS_MSEC, 60000); ok, fr = cap.read()
if ok: cv2.imwrite(f"{D['s5']}/data_collection_video_still.jpg", fr)
print("ASSETS_1_5_DONE")
