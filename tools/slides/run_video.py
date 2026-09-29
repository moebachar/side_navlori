#!/usr/bin/env python
# Slide 5 video: one golden run replayed. Left = the 4 sensor streams (camera with tag boxes, IMU, wheel odometry, WiFi);
# right = lidar SLAM map building up on the CAD floor plan, robot path, markers lighting up when the camera sees them.
import json, re, sys, numpy as np, pandas as pd, yaml
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Polygon
Image.MAX_IMAGE_PIXELS = None

RUN = sys.argv[1] if len(sys.argv) > 1 else "golden_run_10"
SPEED, FPS = 3.0, 30
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else None           # frames (for a quick test)
SN = "/mnt/x/side_navlori"; R = f"{SN}/data/{RUN}"
OUT = sys.argv[3] if len(sys.argv) > 3 else f"{SN}/slides_progress/slide5_visualization/{RUN}_replay.mp4"
C_WIFI, C_IMU, C_ODOM, C_CAM, C_LIDAR, C_TAG = "#7b2cbf", "#f77f00", "#2a9d3f", "#1d4ed8", "#374151", "#f2b705"
WIN = 10.0                                                          # seconds shown in the strip charts
F = 4                                                               # floor plan downsampling

# ---------------- data ----------------
gt = pd.read_csv(f"{R}/ground_truth/gt_pose.csv"); T0 = int(gt.t_ns.iloc[0])
sec = lambda a: (np.asarray(a, np.int64) - T0) / 1e9
tg, GX, GY, GYAW = sec(gt.t_ns), gt.x.values, gt.y.values, gt.yaw.values
DUR = tg[-1]
cam = pd.read_csv(f"{R}/camera/camera.csv"); tc = sec(cam.t_ns)
det = pd.read_csv(f"{R}/ground_truth/tag_detections.csv")
det = det[(det.reproj_px / det.side_px < 0.015) & (det.side_px > 18)].copy(); det["t"] = sec(det.t_ns)
imu = pd.read_csv(f"{R}/imu/imu.csv"); ti = sec(imu.t_ns); wz = np.degrees(imu.wz.values); ax_ = imu.ax.values
js = pd.read_csv(f"{R}/wheel_odom/joint_states.csv"); tj = sec(js.t_ns)
vl, vr = js.left_vel_radps.values, js.right_vel_radps.values          # joint_states velocity is wheel speed in m/s (column name says radps)
wf = pd.read_csv(f"{R}/wifi/wifi.csv"); wf = wf[wf.last_seen_ms <= 4000]
scans = []
for _, s in wf.groupby("scan_idx"):
    s = s.sort_values("rssi_dbm", ascending=False)
    scans.append((float(sec(s.t_end_ns.iloc[0])), len(s), s.head(8)[["ssid", "bssid", "rssi_dbm"]].values))
scans.sort(key=lambda x: x[0]); tw = np.array([s[0] for s in scans])
z = np.load(f"{R}/lidar/scans.npz"); off = z["offsets"]
tags_y = yaml.safe_load(open(f"{R}/calib/apriltags.yaml"))["tags"]
TAGW = {int(k): np.array(v["estimated"] or v["cad"]) for k, v in tags_y.items()}
info = json.load(open(f"{R}/ground_truth/gt_info.json")); USED = info["used_tags"]

# ---------------- floor plan crop ----------------
tf = json.load(open(f"{SN}/data/golden_floorplan/plan_transform.json")); S, RR, TX, TY = tf["px_per_m"], tf["rotation_rad"], tf["tx_px"], tf["ty_px"]
def w2p_full(x, y):
    c, s = np.cos(RR), np.sin(RR); return TX + S * (c * x - s * y), TY - S * (s * x + c * y)
pts = np.c_[np.r_[GX, [TAGW[t][0] for t in USED]], np.r_[GY, [TAGW[t][1] for t in USED]]]
u, v = w2p_full(pts[:, 0], pts[:, 1]); M = 2.5 * S
plan = Image.open(f"{SN}/{tf['image']}").convert("L")
x0, x1 = int(max(0, u.min() - M)), int(min(plan.size[0], u.max() + M)); y0, y1 = int(max(0, v.min() - M)), int(min(plan.size[1], v.max() + M))
base = np.array(plan.crop((x0, y0, x1, y1)).resize(((x1 - x0) // F, (y1 - y0) // F), Image.LANCZOS)).astype(np.float32)
base = 255 - (255 - base) * 0.45                                    # lighten the plan
Hm, Wm = base.shape
def w2p(x, y):
    a, b = w2p_full(np.asarray(x, float), np.asarray(y, float)); return (a - x0) / F, (b - y0) / F
count = np.zeros((Hm, Wm), np.float32)

# ---------------- figure ----------------
fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="white")
gs = fig.add_gridspec(4, 2, width_ratios=[0.40, 0.60], height_ratios=[0.62, 0.14, 0.14, 0.17],
                      left=0.035, right=0.985, top=0.905, bottom=0.05, wspace=0.07, hspace=0.36)
axc = fig.add_subplot(gs[0, 0]); axi = fig.add_subplot(gs[1, 0]); axo = fig.add_subplot(gs[2, 0]); axw = fig.add_subplot(gs[3, 0])
axm = fig.add_subplot(gs[:, 1])
fig.text(0.03, 0.955, f"{RUN.replace('_', ' ')}  ·  {DUR/60:.1f} min  ·  {info['path_m']:.0f} m", fontsize=20, weight="bold")
ttxt = fig.text(0.985, 0.955, "", fontsize=18, ha="right", family="monospace")
fig.text(0.985, 0.925, f"played at {SPEED:.0f}× speed", fontsize=12, ha="right", color="0.4")

def head(ax, name, rate, col):
    ax.set_title(f"{name}", loc="left", fontsize=13, color=col, weight="bold", pad=4)
    ax.text(1.0, 1.02, rate, transform=ax.transAxes, ha="right", va="bottom", fontsize=10, color="0.45")
# camera
im_cam = axc.imshow(np.zeros((480, 640, 3), np.uint8)); axc.set_xticks([]); axc.set_yticks([])
for s_ in axc.spines.values(): s_.set_edgecolor(C_CAM); s_.set_linewidth(2.5)
head(axc, "Camera", "640×480 · 30 Hz", C_CAM)
box, = axc.plot([], [], "-", color=C_TAG, lw=3.5); boxlab = axc.text(0, 0, "", color="k", fontsize=13, weight="bold",
                                                                   bbox=dict(fc=C_TAG, ec="none", pad=2), visible=False)
# IMU
head(axi, "IMU", "145 Hz", C_IMU)
li_g, = axi.plot([], [], color=C_IMU, lw=1.6, label="turn rate (°/s)"); axi.set_ylim(-80, 80); axi.set_ylabel("°/s", fontsize=9)
axi2 = axi.twinx(); li_a, = axi2.plot([], [], color="#fdb863", lw=0.7, alpha=0.6, label="forward accel (m/s²)"); axi2.set_ylim(-2.5, 2.5)
axi2.tick_params(labelsize=8); axi.tick_params(labelsize=8)
axi.legend(handles=[li_g, li_a], loc="upper left", fontsize=8.5, frameon=False, ncol=2)
# wheel odometry
head(axo, "Wheel odometry", "145 Hz", C_ODOM)
lo_l, = axo.plot([], [], color=C_ODOM, lw=1.6, label="left wheel"); lo_r, = axo.plot([], [], color="#95d5a3", lw=1.6, label="right wheel")
axo.set_ylim(-0.05, 0.40); axo.set_ylabel("m/s", fontsize=9); axo.tick_params(labelsize=8)
axo.legend(loc="upper left", fontsize=8.5, frameon=False, ncol=2)
for a in (axi, axo):
    a.grid(alpha=0.25); a.set_xlim(-WIN, 0); a.set_xticks([-10, -5, 0]); a.set_xticklabels(["-10 s", "-5 s", "now"])
    for s_ in ("top",): a.spines[s_].set_visible(False)
# WiFi
head(axw, "WiFi", "1 scan / ~4 s", C_WIFI)
bars = axw.barh(np.arange(8), np.zeros(8), left=-100, color=C_WIFI, height=0.7)
axw.set_xlim(-100, -30); axw.set_ylim(7.6, -0.6); axw.set_yticks([]); axw.tick_params(labelsize=8)
axw.set_xlabel("signal strength (dBm), 8 strongest access points", fontsize=9)
wlabels = [axw.text(-99, i, "", va="center", fontsize=9, color="white", weight="bold") for i in range(8)]
wtxt = axw.text(1.0, -0.62, "", transform=axw.transAxes, ha="right", va="top", fontsize=9.5, color=C_WIFI)
for s_ in ("top", "right"): axw.spines[s_].set_visible(False)
# map
axm.imshow(base, cmap="gray", vmin=0, vmax=255)
ov_rgba = np.zeros((Hm, Wm, 4), np.float32); ov_rgba[..., :3] = matplotlib.colors.to_rgb(C_LIDAR)
im_ov = axm.imshow(ov_rgba, interpolation="nearest")
axm.set_xlim(0, Wm); axm.set_ylim(Hm, 0); axm.set_xticks([]); axm.set_yticks([])
for s_ in axm.spines.values(): s_.set_edgecolor("0.75")
axm.set_title("Lidar SLAM on the floor plan", loc="left", fontsize=15, weight="bold", pad=6)
path, = axm.plot([], [], "-", color="k", lw=2.4, zorder=5)
scan_now = axm.scatter([], [], s=5, color=C_LIDAR, zorder=4)
robot = Polygon(np.zeros((3, 2)), closed=True, fc="#d62828", ec="k", lw=1.2, zorder=8); axm.add_patch(robot)
tag_ids = [t for t in TAGW if 0 <= w2p(*TAGW[t])[0] < Wm and 0 <= w2p(*TAGW[t])[1] < Hm]
tp = np.array([w2p(*TAGW[t]) for t in tag_ids])
tag_sc = axm.scatter(tp[:, 0], tp[:, 1], s=190, marker="s", c=["white"] * len(tag_ids), edgecolors="k", linewidths=1.3, zorder=7)
for t, (a, b) in zip(tag_ids, tp):
    axm.annotate(f"{t}", (a, b), xytext=(9, 7), textcoords="offset points", fontsize=12, weight="bold", zorder=9)
ring = axm.scatter([], [], s=900, marker="o", facecolors="none", edgecolors=C_TAG, linewidths=3, zorder=6)
sight, = axm.plot([], [], "--", color=C_TAG, lw=2.2, zorder=6)
L = 2 * S / F; bx, by = Wm * 0.03, Hm * 0.965
axm.plot([bx, bx + L], [by, by], "-", color="k", lw=4); axm.text(bx + L / 2, by - 8, "2 m", ha="center", va="bottom", fontsize=12)
from matplotlib.lines import Line2D
axm.legend(handles=[Line2D([], [], color=C_LIDAR, marker="o", ls="", ms=5, label="lidar map (built live)"),
                    Line2D([], [], color="k", lw=2.4, label="robot path (ground truth)"),
                    Line2D([], [], color="#d62828", marker=">", ls="", ms=11, mec="k", label="robot"),
                    Line2D([], [], color="white", marker="s", ls="", ms=11, mec="k", label="marker"),
                    Line2D([], [], color=C_TAG, marker="s", ls="", ms=11, mec="k", label="marker seen by the camera")],
           loc="lower right", fontsize=11, framealpha=0.9)

# ---------------- animation ----------------
seen = set(); k_scan = 0; last_wifi = -1
def scan_xy(i, beam=1):
    a, r = z["angles"][off[i]:off[i + 1]:beam], z["ranges"][off[i]:off[i + 1]:beam]
    ok = np.isfinite(r) & (r > 0.1) & (r < 8.0); a, r = a[ok], r[ok]
    xb, yb = -0.064 + r * np.cos(a), r * np.sin(a); c, s = np.cos(GYAW[i]), np.sin(GYAW[i])
    return GX[i] + c * xb - s * yb, GY[i] + s * xb + c * yb
def update(t):
    global k_scan, last_wifi
    ttxt.set_text(f"t = {t:5.1f} s")
    j = max(0, int(np.searchsorted(tc, t, side="right")) - 1)
    im_cam.set_data(np.array(Image.open(f"{R}/camera/images/{cam.filename.iloc[j]}").convert("RGB")))
    d = det[(det.t - tc[j]).abs() < 0.07]
    if len(d):
        d = d.sort_values("side_px").iloc[-1]
        bx_ = [d.c0x, d.c1x, d.c2x, d.c3x, d.c0x]; by_ = [d.c0y, d.c1y, d.c2y, d.c3y, d.c0y]
        box.set_data(bx_, by_); boxlab.set_text(f" marker {int(d.tag_id)} "); boxlab.set_position((min(bx_[:4]), min(by_[:4]) - 12))
        boxlab.set_visible(True); tid = int(d.tag_id); seen.add(tid)
    else:
        box.set_data([], []); boxlab.set_visible(False); tid = None
    m = (ti > t - WIN) & (ti <= t); li_g.set_data(ti[m] - t, wz[m]); li_a.set_data(ti[m] - t, ax_[m])
    m = (tj > t - WIN) & (tj <= t); lo_l.set_data(tj[m] - t, vl[m]); lo_r.set_data(tj[m] - t, vr[m])
    w = int(np.searchsorted(tw, t, side="right")) - 1
    if w >= 0 and w != last_wifi:
        last_wifi = w; top = scans[w][2]
        for b, lab, row in zip(bars, wlabels, list(top) + [None] * (8 - len(top))):
            if row is None: b.set_width(0); lab.set_text(""); continue
            ssid, bssid, rssi = row; b.set_width(rssi + 100)
            ssid = str(ssid) if isinstance(ssid, str) else ""
            name = ssid[:14] if re.match(r"(?i)^(cesi|eduspot|eduroam)", ssid) else "other network"   # no private hotspot names
            lab.set_text(f"{name} …{bssid[-5:]}  {rssi:.0f}")
    if w >= 0:
        age = t - tw[w]; [b.set_alpha(1.0 if age < 0.6 else 0.78) for b in bars]
        wtxt.set_text(f"scan {w+1}/{len(scans)}  ·  {scans[w][1]} access points heard")
    n = int(np.searchsorted(tg, t, side="right"))
    while k_scan < n:
        x, y = scan_xy(k_scan); pu, pv = w2p(x, y); pu, pv = pu.astype(int), pv.astype(int)
        ok = (pu >= 0) & (pu < Wm) & (pv >= 0) & (pv < Hm); np.add.at(count, (pv[ok], pu[ok]), 1.0); k_scan += 1
    ov_rgba[..., 3] = np.clip(count / 3.0, 0, 1) * 0.85; im_ov.set_data(ov_rgba)
    i = max(0, n - 1); pu, pv = w2p(GX[:n], GY[:n]); path.set_data(pu, pv)
    x, y = scan_xy(i, beam=2); a, b = w2p(x, y); scan_now.set_offsets(np.c_[a, b])
    c, s = np.cos(GYAW[i]), np.sin(GYAW[i]); tri = np.array([[0.30, 0], [-0.18, 0.16], [-0.18, -0.16]])
    wx_, wy_ = GX[i] + c * tri[:, 0] - s * tri[:, 1], GY[i] + s * tri[:, 0] + c * tri[:, 1]; robot.set_xy(np.c_[w2p(wx_, wy_)])
    tag_sc.set_facecolors([C_TAG if t_ in seen else "white" for t_ in tag_ids])
    if tid is not None and tid in TAGW:
        cx_, cy_ = GX[i] + c * 0.076, GY[i] + s * 0.076; a0, b0 = w2p(cx_, cy_); a1, b1 = w2p(*TAGW[tid])
        sight.set_data([a0, a1], [b0, b1]); ring.set_offsets([[a1, b1]])
    else:
        sight.set_data([], []); ring.set_offsets(np.zeros((0, 2)))

times = np.arange(0, DUR, SPEED / FPS)
if LIMIT: times = times[:LIMIT]
wr = FFMpegWriter(fps=FPS, codec="libx264", bitrate=6000, extra_args=["-pix_fmt", "yuv420p", "-preset", "medium"])
with wr.saving(fig, OUT, dpi=100):
    for k, t in enumerate(times):
        update(t); wr.grab_frame()
        if k % 150 == 0: print(f"frame {k}/{len(times)}  t={t:.1f}s", flush=True)
fig.savefig(OUT.replace(".mp4", "_still.jpg"), dpi=100)
print(f"VIDEO_DONE {OUT} frames={len(times)} markers_seen={sorted(seen)}", flush=True)
