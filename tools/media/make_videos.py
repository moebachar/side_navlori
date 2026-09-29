import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from PIL import Image

RUN = "/mnt/x/side_navlori/data/staging/big"
CAM = "/root/navlori/_bigcam/camera"
OUT = "/mnt/x/side_navlori/figures"; os.makedirs(OUT, exist_ok=True)

gt = pd.read_csv(f"{RUN}/ground_truth/gt_pose.csv")
X, Y, YAW, T = gt.x.values, gt.y.values, gt.yaw.values, gt.t_ns.values.astype(np.int64)
ns = len(gt)
z = np.load(f"{RUN}/lidar/scans.npz"); off = z["offsets"]; rng = z["ranges"]; ang = z["angles"]

# transform every lidar return into the map frame (scan i uses SLAM pose i)
WX, WY, counts = [], [], []
for i in range(ns):
    a = ang[off[i]:off[i+1]]; r = rng[off[i]:off[i+1]]
    m = np.isfinite(r) & (r > 0.05) & (r < 8.0)
    a, r = a[m], r[m]
    WX.append(X[i] + r*np.cos(a + YAW[i])); WY.append(Y[i] + r*np.sin(a + YAW[i])); counts.append(m.sum())
counts = np.array(counts); WX = np.concatenate(WX); WY = np.concatenate(WY)
cum = np.concatenate([[0], np.cumsum(counts)])
point_scan = np.repeat(np.arange(ns), counts)
pad = 1.0; xlim = (WX.min()-pad, WX.max()+pad); ylim = (WY.min()-pad, WY.max()+pad)

cam = pd.read_csv(f"{CAM}/camera.csv"); ct = cam.t_ns.values.astype(np.int64); cf = cam.filename.values

NF = 340
scan_of = np.round(np.linspace(0, ns-1, NF)).astype(int)

def tri(x, y, yaw, s=0.32):
    p = np.array([[s, 0], [-s*0.6, s*0.55], [-s*0.6, -s*0.55]])
    c, si = np.cos(yaw), np.sin(yaw)
    return p @ np.array([[c, -si], [si, c]]).T + [x, y]

# ================= VIDEO 2 — SLAM map-building timelapse =================
sub = slice(None, None, 2)
WXs, WYs, PSs = WX[sub], WY[sub], point_scan[sub]
fig, ax = plt.subplots(figsize=(8, 8)); ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal"); ax.axis("off")
mappts = ax.scatter([], [], s=1.3, c="#3b6fb0", alpha=0.55, zorder=1, edgecolors="none")
traj, = ax.plot([], [], "-", color="#e8590c", lw=2.2, zorder=3)
rob = ax.fill(*tri(X[0], Y[0], YAW[0]).T, color="k", zorder=4)[0]
ttl = ax.set_title("", fontsize=13)
def up2(f):
    k = scan_of[f]; msk = PSs <= k
    mappts.set_offsets(np.c_[WXs[msk], WYs[msk]])
    traj.set_data(X[:k+1], Y[:k+1]); rob.set_xy(tri(X[k], Y[k], YAW[k]))
    ttl.set_text(f"Lidar SLAM — mapping while driving    t = {(T[k]-T[0])/1e9:4.0f} s    {k+1}/{ns} scans")
    return mappts, traj, rob, ttl
FuncAnimation(fig, up2, frames=NF, blit=False).save(
    f"{OUT}/demo_slam_timelapse_big.mp4", writer=FFMpegWriter(fps=20, bitrate=4500))
plt.close(fig); print("video2 done", flush=True)

# ================= VIDEO 1 — data-collection cockpit =================
fig = plt.figure(figsize=(14, 6.2))
axc = fig.add_axes([0.02, 0.06, 0.46, 0.86]); axc.axis("off"); axc.set_title("onboard camera", fontsize=12)
axm = fig.add_axes([0.52, 0.05, 0.46, 0.90]); axm.set_aspect("equal"); axm.axis("off"); axm.set_title("map + trajectory (lidar)", fontsize=12)
axm.set_xlim(*xlim); axm.set_ylim(*ylim)
axm.scatter(WX[::3], WY[::3], s=0.5, c="0.83", zorder=0, edgecolors="none")   # faint full map context
imobj = axc.imshow(np.array(Image.open(f"{CAM}/images/{cf[0]}")))
traj1, = axm.plot([], [], "-", color="#e8590c", lw=2.2, zorder=3)
live = axm.scatter([], [], s=4, c="#1264a3", alpha=0.75, zorder=2, edgecolors="none")
rob1 = axm.fill(*tri(X[0], Y[0], YAW[0]).T, color="k", zorder=4)[0]
sup = fig.suptitle("", fontsize=14)
cam_idx = np.clip(np.searchsorted(ct, T[scan_of]), 0, len(cf)-1)
def up1(f):
    k = scan_of[f]
    imobj.set_data(np.array(Image.open(f"{CAM}/images/{cf[cam_idx[f]]}")))
    traj1.set_data(X[:k+1], Y[:k+1])
    live.set_offsets(np.c_[WX[cum[k]:cum[k+1]], WY[cum[k]:cum[k+1]]])
    rob1.set_xy(tri(X[k], Y[k], YAW[k]))
    sup.set_text(f"TurtleBot3 — driving a trajectory while all sensors stream    ·    t = {(T[k]-T[0])/1e9:4.0f} s")
    return imobj, traj1, live, rob1, sup
FuncAnimation(fig, up1, frames=NF, blit=False).save(
    f"{OUT}/demo_cockpit_big.mp4", writer=FFMpegWriter(fps=20, bitrate=6000))
plt.close(fig); print("video1 done", flush=True)
print("ALL_VIDEOS_DONE", flush=True)
