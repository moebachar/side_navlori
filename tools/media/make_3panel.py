import os, numpy as np, pandas as pd, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from PIL import Image

RUN = "/mnt/x/side_navlori/data/staging/big"
CAM = "/root/navlori/_bigcam/camera"
PHONE = "/mnt/x/side_navlori/IMG_1757_1925.MOV"
OUT = "/mnt/x/side_navlori/figures/data_collection_big.mp4"
FPS, DUR = 24, 100                      # 100 s, high frame rate

# ---- run data (onboard + SLAM share the bag timeline) ----
gt = pd.read_csv(f"{RUN}/ground_truth/gt_pose.csv")
X, Y, YAW, T = gt.x.values, gt.y.values, gt.yaw.values, gt.t_ns.values.astype(np.int64)
ns = len(gt)
z = np.load(f"{RUN}/lidar/scans.npz"); off = z["offsets"]; rng = z["ranges"]; ang = z["angles"]
WX, WY, counts = [], [], []
for i in range(ns):
    a = ang[off[i]:off[i+1]]; r = rng[off[i]:off[i+1]]
    m = np.isfinite(r) & (r > 0.05) & (r < 8.0); a, r = a[m], r[m]
    WX.append(X[i] + r*np.cos(a + YAW[i])); WY.append(Y[i] + r*np.sin(a + YAW[i])); counts.append(int(m.sum()))
counts = np.array(counts); WX = np.concatenate(WX); WY = np.concatenate(WY)
cum = np.concatenate([[0], np.cumsum(counts)])
point_scan = np.repeat(np.arange(ns), counts)
WXs, WYs, PSs = WX[::2], WY[::2], point_scan[::2]        # subsampled for the accumulating map
pad = 1.0; xlim = (WX.min()-pad, WX.max()+pad); ylim = (WY.min()-pad, WY.max()+pad)
cam = pd.read_csv(f"{CAM}/camera.csv"); ct = cam.t_ns.values.astype(np.int64); cf = cam.filename.values

# ---- phone reader (sequential grab, rotation-corrected) ----
capp = cv2.VideoCapture(PHONE)
try: capp.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
except Exception: pass
nph = int(capp.get(cv2.CAP_PROP_FRAME_COUNT))
_cur = [-1]; _frame = [None]
def phone_at(idx):
    idx = int(idx)
    while _cur[0] < idx:
        capp.grab(); _cur[0] += 1
        if _cur[0] == idx:
            ok, fr = capp.retrieve()
            if ok:
                _frame[0] = np.rot90(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), -1)
    return _frame[0]

# ---- timeline (all three proportionally synced across 100 s) ----
NF = FPS * DUR; frac = np.linspace(0, 1, NF)
scan_of = np.round(frac*(ns-1)).astype(int)
cam_of = np.clip(np.searchsorted(ct, T[scan_of]), 0, len(cf)-1)
phone_of = np.round(frac*(nph-1)).astype(int)

def tri(x, y, yaw, s=0.32):
    p = np.array([[s, 0], [-s*0.6, s*0.55], [-s*0.6, -s*0.55]])
    c, si = np.cos(yaw), np.sin(yaw)
    return p @ np.array([[c, -si], [si, c]]).T + [x, y]

# ---- layout: col1 = onboard (top) + user (bottom), col2 = SLAM ----
fig = plt.figure(figsize=(19.2, 10.8), dpi=100); fig.patch.set_facecolor("white")
fig.text(0.5, 0.955, "Data collection", ha="center", va="center", fontsize=30, fontweight="bold")
ax_on = fig.add_axes([0.035, 0.55, 0.35, 0.36]); ax_on.axis("off"); ax_on.set_title("onboard camera", fontsize=17)
ax_us = fig.add_axes([0.035, 0.05, 0.35, 0.44]); ax_us.axis("off"); ax_us.set_title("user camera", fontsize=17)
ax_sl = fig.add_axes([0.42, 0.05, 0.55, 0.86]); ax_sl.set_aspect("equal"); ax_sl.axis("off"); ax_sl.set_title("SLAM", fontsize=17)
ax_sl.set_xlim(*xlim); ax_sl.set_ylim(*ylim); ax_sl.set_anchor("N")   # hug the top, near the title

im_on = ax_on.imshow(np.array(Image.open(f"{CAM}/images/{cf[0]}")))
im_us = ax_us.imshow(phone_at(0))
mappts = ax_sl.scatter([], [], s=1.3, c="#3b6fb0", alpha=0.55, zorder=1, edgecolors="none")  # map builds live
traj, = ax_sl.plot([], [], "-", color="#e8590c", lw=2.4, zorder=3)
rob = ax_sl.fill(*tri(X[0], Y[0], YAW[0]).T, color="k", zorder=4)[0]

def up(f):
    k = scan_of[f]
    im_on.set_data(np.array(Image.open(f"{CAM}/images/{cf[cam_of[f]]}")))
    ph = phone_at(phone_of[f])
    if ph is not None: im_us.set_data(ph)
    traj.set_data(X[:k+1], Y[:k+1])
    msk = PSs <= k; mappts.set_offsets(np.c_[WXs[msk], WYs[msk]])
    rob.set_xy(tri(X[k], Y[k], YAW[k]))
    if f % 200 == 0: print("frame", f, "/", NF, flush=True)
    return im_on, im_us, traj, mappts, rob

if os.environ.get("PREVIEW"):
    up(int(NF*0.55)); fig.savefig("/mnt/x/side_navlori/figures/_panel_preview.png", dpi=100)
    print("PREVIEW saved", flush=True); raise SystemExit

FuncAnimation(fig, up, frames=NF, blit=False).save(OUT, writer=FFMpegWriter(fps=FPS, bitrate=10000))
print("DONE", OUT, flush=True)
