#!/usr/bin/env python
# Real-time GIF: robot drives the full route; each method's estimate pops as its WiFi scan fires.
# One panel per method (protocol A, in-map). Runs in the notebook venv; deep methods via sub-venvs.
import sys, os, subprocess, tempfile, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
sys.path.append("/mnt/x/side_navlori/dataset_pipeline/export")
from load_dataset import Dataset

DATA = "/mnt/x/side_navlori/data/run2"
ds = Dataset(DATA)
M, bssids, t_ns = ds.wifi_matrix()
gx, gy, _ = ds.gt_at(t_ns); XY = np.c_[gx, gy]; N = len(t_ns)
walls = np.load(f"{DATA}/ground_truth/map_points.npz")["xy"]
wsub = walls[::12]                                   # subsample walls for a light background
gt_t = ds.gt.t_ns.values.astype(float); gt_x = ds.gt.x.values; gt_y = ds.gt.y.values

# ---- protocol A predictions for all five methods ----
K = 4; foldA = np.arange(N) % K
def by_protocol_A(predict):
    p = np.zeros((N, 2))
    for f in range(K):
        tr, te = foldA != f, foldA == f
        p[te] = predict(M[tr], XY[tr], M[te])
    return p
def knn(Mtr, Ptr, Mte, k=1, weighted=False, rep=lambda m: m):
    A, B = rep(Mtr), rep(Mte)
    D = np.sqrt(((B[:, None, :] - A[None, :, :]) ** 2).sum(-1))
    nn = np.argsort(D, 1)[:, :k]; dn = np.take_along_axis(D, nn, 1)
    if weighted:
        w = 1.0 / (dn + 1e-9); w /= w.sum(1, keepdims=True)
        return (w[:, :, None] * Ptr[nn]).sum(1)
    return Ptr[nn].mean(1)
powed = lambda m: ((np.clip(m, -100, -20) + 100) / 80.0) ** np.e
def horus(Mtr, Ptr, Mte, cell=1.5, sigma=6.0):
    key = np.floor(Ptr / cell).astype(int)
    _, inv = np.unique(key, axis=0, return_inverse=True); Cn = inv.max() + 1
    mu = np.stack([Mtr[inv == c].mean(0) for c in range(Cn)])
    cen = np.stack([Ptr[inv == c].mean(0) for c in range(Cn)])
    ll = -((Mte[:, None, :] - mu[None, :, :]) ** 2).sum(-1) / (2 * sigma ** 2)
    return cen[ll.argmax(1)]
SOTA = {"CNNLoc": ("/root/navlori/venvs_wifi/cnnloc/bin/python", "/root/navlori/mrepos_wifi/cnnloc/cnnloc_runner.py", "/root/navlori/mrepos_wifi/cnnloc"),
        "Kim":    ("/root/navlori/venvs_wifi/kim/bin/python",    "/root/navlori/mrepos_wifi/kimdnn/kim_runner.py", None)}
def pad520(m):
    z = np.full((m.shape[0], 520), -100.0); z[:, :m.shape[1]] = m; return z
def sota(name):
    py, runner, cwd = SOTA[name]
    def predict(Mtr, Ptr, Mte):
        with tempfile.TemporaryDirectory() as td:
            ip, op = td + "/in.npz", td + "/out.npz"
            np.savez(ip, Xtr=pad520(Mtr), ytr=np.c_[Ptr, np.zeros((len(Ptr), 2))], Xte=pad520(Mte))
            subprocess.run([py, runner, ip, op], cwd=cwd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return np.load(op)["pred"]
    return predict

print("computing predictions...")
PRED = {
    "RADAR": by_protocol_A(lambda a, b, c: knn(a, b, c, k=1)),
    "WkNN":  by_protocol_A(lambda a, b, c: knn(a, b, c, k=5, weighted=True, rep=powed)),
    "Horus": by_protocol_A(horus),
    "CNNLoc": by_protocol_A(sota("CNNLoc")),
    "Kim":    by_protocol_A(sota("Kim")),
}
methods = list(PRED)

# ---- animation ----
FRAMES = 96
ft = np.linspace(gt_t[0], gt_t[-1], FRAMES)          # frame times across the run
xlim = (walls[:, 0].min() - 1, walls[:, 0].max() + 1)
ylim = (walls[:, 1].min() - 1, walls[:, 1].max() + 1)
plt.rcParams["figure.dpi"] = 74
fig, axes = plt.subplots(1, len(methods), figsize=(3.3 * len(methods), 4.0), sharex=True, sharey=True)
COL = "tab:blue"
art = []
for ax, name in zip(axes, methods):
    ax.scatter(wsub[:, 0], wsub[:, 1], s=0.2, c="0.86", zorder=0)
    ax.plot(gt_x, gt_y, "-", color="0.9", lw=0.7, zorder=1)
    route, = ax.plot([], [], "-", color="0.55", lw=1.2, zorder=2)         # traveled path
    errs = [ax.plot([], [], "-", color="crimson", lw=0.8, alpha=0.55, zorder=3)[0] for _ in range(N)]
    past = ax.scatter([], [], s=12, c=COL, alpha=0.35, zorder=4)           # fired estimates
    cur, = ax.plot([], [], marker="o", ms=8, color=COL, mec="w", mew=0.6, zorder=6)  # current estimate
    robot, = ax.plot([], [], marker="*", ms=15, color="k", zorder=7)      # the robot
    ttl = ax.set_title(f"{name}", fontsize=10)
    ax.set_aspect("equal"); ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_xlabel("x (m)")
    art.append(dict(name=name, route=route, errs=errs, past=past, cur=cur, robot=robot, ttl=ttl))
axes[0].set_ylabel("y (m)")
sup = fig.suptitle("", fontsize=11)

def update(fi):
    tau = ft[fi]
    fired = int((t_ns <= tau).sum())                 # scans that have happened
    rx = np.interp(tau, gt_t, gt_x); ry = np.interp(tau, gt_t, gt_y)
    gmask = gt_t <= tau
    out = []
    for a in art:
        P = PRED[a["name"]]
        a["route"].set_data(gt_x[gmask], gt_y[gmask])
        a["robot"].set_data([rx], [ry])
        if fired > 0:
            a["past"].set_offsets(P[:fired])
            j = fired - 1
            a["cur"].set_data([P[j, 0]], [P[j, 1]])
            for k in range(N):
                if k < fired:
                    a["errs"][k].set_data([XY[k, 0], P[k, 0]], [XY[k, 1], P[k, 1]])
                else:
                    a["errs"][k].set_data([], [])
            e = np.hypot(P[:fired, 0] - XY[:fired, 0], P[:fired, 1] - XY[:fired, 1])
            a["ttl"].set_text(f"{a['name']}   med {np.median(e):.2f} m")
        out += [a["route"], a["robot"], a["past"], a["cur"], a["ttl"], *a["errs"]]
    sup.set_text(f"WiFi localization in real time  ·  t = {(tau - gt_t[0]) / 1e9:5.1f} s  ·  scans fired {fired}/{N}")
    return out

print("rendering", FRAMES, "frames...")
anim = FuncAnimation(fig, update, frames=FRAMES, blit=False)
out = "/mnt/x/side_navlori/figures/wifi_methods_realtime.gif"
anim.save(out, writer=PillowWriter(fps=10))
print("SAVED", out, os.path.getsize(out) // 1024, "KB")
