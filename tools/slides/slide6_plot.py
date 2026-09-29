#!/usr/bin/env python
# Slide 6: test runs of side_golden on the CAD floor plan, ground truth vs each method's prediction.
import json, os, sys, numpy as np, pandas as pd
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
Image.MAX_IMAGE_PIXELS = None
SN = "/mnt/x/side_navlori"; NF = "/mnt/x/navlori-fusion"; R = NF + "/runs/side_golden"; D = NF + "/data/side_golden"
OUT = SN + "/slides_progress/slide6_results"; EXTRA = SN + "/slides_progress/_extra/slide6_results"
os.makedirs(OUT, exist_ok=True); os.makedirs(EXTRA, exist_ok=True)
tf = json.load(open(SN + "/data/golden_floorplan/plan_transform.json")); S, RR, TX, TY = tf["px_per_m"], tf["rotation_rad"], tf["tx_px"], tf["ty_px"]
F = 3
plan = Image.open(f"{SN}/{tf['image']}").convert("L"); W0, H0 = plan.size
plan = np.array(plan.resize((W0 // F, H0 // F), Image.LANCZOS))
def w2p(xy):
    x, y = xy[:, 0], xy[:, 1]; c, s = np.cos(RR), np.sin(RR)
    return (TX + S * (c * x - s * y)) / F, (TY - S * (s * x + c * y)) / F

TEST = [1, 5, 11]
odo = np.load(f"{R}/odom_dr/trajectories.npz"); ron = np.load(f"{R}/imu_ronin/trajectories.npz")
wifi = np.load(f"{R}/wifi_wlanloc/pred_test.npz")
fus = {k: np.load(f"{R}/{k}/pred_test.npz") for k in ["fusion_3mod", "fusion_4mod"] if os.path.exists(f"{R}/{k}/pred_test.npz")}
STY = {  # label, colour, kind
    "wifi": ("WiFi (wlan_localization)", "#7b2cbf", "dots"),
    "imu": ("IMU (RoNIN)", "#f77f00", "line"),
    "odom": ("Wheel odometry", "#2a9d3f", "line"),
    "fusion_3mod": ("Fusion WiFi+IMU+odom", "#d62828", "fine"),
    "fusion_4mod": ("Fusion + camera (4 sensors)", "#1d4ed8", "fine"),
}
def series(pid):
    g = pd.read_csv(f"{D}/path_{pid:02d}/ground_truth.csv"); out = {"gt": g[["gt_x", "gt_y"]].values}
    m = wifi["path_id"] == pid; out["wifi"] = wifi["pred"][m]
    out["imu"] = ron[f"ronin_path_{pid:02d}"]; out["odom"] = odo[f"wheel_odom_path_{pid:02d}"]
    for k, f in fus.items():
        m = f["path_id"] == pid; o = np.argsort(f["sim_time"][m]); out[k] = f["pred"][m][o]
    return out

def draw(ax, pid, legend=True, title=None):
    s = series(pid); u, v = w2p(s["gt"])
    pu, pv = w2p(s["gt"]); m = 6.0 * S / F                       # crop: ground truth + 6 m
    x0, x1 = max(0, pu.min() - m), min(plan.shape[1], pu.max() + m); y0, y1 = max(0, pv.min() - m), min(plan.shape[0], pv.max() + m)
    ax.imshow(plan, cmap="gray", vmin=0, vmax=255, alpha=0.55); ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.axis("off")
    ax.plot(u, v, "-", color="k", lw=3.2, label="Ground truth", zorder=5)
    ax.plot(u[0], v[0], "o", ms=11, mfc="white", mec="k", mew=2, zorder=9)
    for k in ["wifi", "imu", "odom"] + list(fus):
        lab, col, kind = STY[k]; pu_, pv_ = w2p(s[k])
        if kind == "dots": ax.plot(pu_, pv_, "o", ms=7, mfc=col, mec="white", mew=0.8, alpha=0.9, label=lab, zorder=6)
        elif kind == "fine": ax.plot(pu_[::3], pv_[::3], ".", ms=4, color=col, alpha=0.45, label=lab, zorder=6)
        else: ax.plot(pu_, pv_, "-", color=col, lw=2.0, alpha=0.9, label=lab, zorder=7)
    L = 5 * S / F; bx, by = x0 + 0.03 * (x1 - x0), y1 - 0.05 * (y1 - y0)
    ax.plot([bx, bx + L], [by, by], "-", color="k", lw=4); ax.text(bx + L / 2, by - 8, "5 m", ha="center", va="bottom", fontsize=12)
    if title: ax.set_title(title, fontsize=13)
    if legend: ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3, fontsize=12, frameon=False, markerscale=2.5)

res = json.load(open(f"{R}/fusion_3mod/results.json"))["per_path_test_mae"]
hero = int(sys.argv[1]) if len(sys.argv) > 1 else sorted(TEST, key=lambda p: res[str(p)])[1]   # median run for the fusion
meta = json.load(open(f"{D}/path_{hero:02d}/metadata.json"))
fig, ax = plt.subplots(figsize=(12, 7)); draw(ax, hero)
fig.savefig(f"{OUT}/test_run_on_floorplan.png", dpi=130, bbox_inches="tight"); plt.close(fig)
fig, axs = plt.subplots(1, 3, figsize=(24, 7))
for a, p in zip(axs, TEST):
    mt = json.load(open(f"{D}/path_{p:02d}/metadata.json"))
    draw(a, p, legend=(p == TEST[1]), title=f"test run: golden_run_{p+1}  ({mt['duration_s']:.0f} s, {mt['path_m']:.0f} m)")
fig.savefig(f"{EXTRA}/all_3_test_runs_on_floorplan.png", dpi=110, bbox_inches="tight"); plt.close(fig)
print("hero = path", hero, "golden_run", hero + 1, meta["duration_s"], "s", meta["path_m"], "m"); print("SLIDE6_PLOT_DONE")
