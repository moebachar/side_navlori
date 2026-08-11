"""Build the lidar-based ground-truth trajectory for the side_navlori dataset.

Method (documented in data/ground_truth/method.md):
  1. Load ragged scans from data/lidar/scans.npz and raw wheel odometry.
  2. Deskew every scan with odometry motion over the ~0.12 s sweep
     (per-beam timestamps t_ns + beam_time_offset_ns).
  3. Pass 1: odometry-seeded incremental point-to-line ICP (PLICP) against a
     growing voxel-deduplicated point map with scan-derived normals.
  4. Refinement: 6 rounds (0.30 m gate / 0.05 m Huber) + 4 polish rounds
     (0.15 m / 0.03 m). Each round freezes the map built from current poses,
     re-matches every scan with a weak odometry motion-model prior, rebuilds.
  5. Validation: per-scan residuals, an end-vs-start-map loop-closure check,
     GT-vs-odom twist consistency, rendered map + trajectory figures.

Frame: 'map' is defined by one final rigid alignment that maps the MEAN GT
pose of the initial stationary segment (scans 0-24, robot immobile) onto the
mean odometry pose at the same stamps. GT poses are base_footprint in this
frame, at raw scan header stamps — no resampling.
"""
import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = r"C:\Users\mbachar\side_navlori\data"
GT_DIR = os.path.join(ROOT, "ground_truth")
os.makedirs(GT_DIR, exist_ok=True)

LIDAR_X = -0.064          # base_scan origin in base_footprint (yaw = 0)
R_MIN, R_MAX = 0.05, 12.0  # valid beam range gate [m]
VOXEL = 0.01               # map dedup voxel [m]
GATE = 0.30                # ICP correspondence gate [m]
HUBER = 0.05               # robust kernel scale [m]
INLIER_D = 0.10            # inlier definition for quality stats [m]
MAX_ITER = 40
ROUNDS = 4

# ---------------------------------------------------------------- data loading
z = np.load(os.path.join(ROOT, "lidar", "scans.npz"))
t_scan = z["t_ns"].astype(np.int64)
offsets = z["offsets"]
angles_f = z["angles"]
ranges_f = z["ranges"].astype(np.float64)
bto_f = z["beam_time_offset_ns"].astype(np.int64)
N = len(t_scan)

od = pd.read_csv(os.path.join(ROOT, "wheel_odom", "odom.csv"))
T0 = int(t_scan[0])
ot = (od.t_ns.values.astype(np.int64) - T0) / 1e9
ox, oy = od.x.values, od.y.values
oyaw = np.unwrap(od.yaw.values)
ts_rel = (t_scan - T0) / 1e9

def odom_pose(t):
    """Interpolated odometry pose (x, y, yaw) at relative time(s) t."""
    return np.interp(t, ot, ox), np.interp(t, ot, oy), np.interp(t, ot, oyaw)

# ------------------------------------------------------------- SE(2) helpers
def compose(a, b):
    ax, ay, ath = a
    bx, by, bth = b
    c, s = np.cos(ath), np.sin(ath)
    return (ax + c * bx - s * by, ay + s * bx + c * by, ath + bth)

def inverse(a):
    ax, ay, ath = a
    c, s = np.cos(ath), np.sin(ath)
    return (-c * ax - s * ay, s * ax - c * ay, -ath)

def transform(pose, pts):
    x, y, th = pose
    c, s = np.cos(th), np.sin(th)
    return np.c_[c * pts[:, 0] - s * pts[:, 1] + x,
                 s * pts[:, 0] + c * pts[:, 1] + y]

# ------------------------------------------------- deskew + normals per scan
def prep_scan(k):
    """Deskewed points (base_footprint @ scan stamp) + unit normals (+validity)."""
    sl = slice(offsets[k], offsets[k + 1])
    r, th, bt = ranges_f[sl], angles_f[sl], bto_f[sl]
    ok = np.isfinite(r) & (r > R_MIN) & (r < R_MAX)
    idx = np.where(ok)[0]
    if len(idx) < 30:
        return None, None
    # beam endpoints in base_footprint (before motion correction)
    px = r[idx] * np.cos(th[idx]) + LIDAR_X
    py = r[idx] * np.sin(th[idx])
    # motion correction: pose(t_beam) relative to pose(t_stamp), via odometry
    tb = (t_scan[k] - T0) / 1e9 + bt[idx] / 1e9
    x0, y0, th0 = odom_pose(ts_rel[k])
    xb, yb, thb = odom_pose(tb)
    dth = thb - th0
    c0, s0 = np.cos(th0), np.sin(th0)
    dxw, dyw = xb - x0, yb - y0                    # world-frame displacement
    dx = c0 * dxw + s0 * dyw                       # -> base frame at stamp
    dy = -s0 * dxw + c0 * dyw
    cd, sd = np.cos(dth), np.sin(dth)
    qx = cd * px - sd * py + dx
    qy = sd * px + cd * py + dy
    pts = np.c_[qx, qy]
    # normals from angular neighbors (index-adjacent beams)
    nrm = np.full((len(idx), 2), np.nan)
    pos_in_idx = {j: i for i, j in enumerate(idx)}
    for i, j in enumerate(idx):
        a, b = pos_in_idx.get(j - 1), pos_in_idx.get(j + 1)
        if a is None or b is None:
            continue
        tvec = pts[b] - pts[a]
        L = np.hypot(*tvec)
        if 1e-4 < L < 0.30:
            n = np.array([-tvec[1], tvec[0]]) / L
            # orient toward the sensor
            if np.dot(n, pts[i] - np.array([LIDAR_X, 0.0])) > 0:
                n = -n
            nrm[i] = n
    return pts, nrm

print("prepping scans (deskew + normals)...")
SCANS = [prep_scan(k) for k in range(N)]

# --------------------------------------------------------------------- PLICP
def icp(pts, seed, tree, mpts, mnrm, max_iter=MAX_ITER, gate=GATE, huber=HUBER,
        prior=None, lam=2.0):
    """Point-to-line ICP of scan pts (base frame) against map. Returns pose, stats.

    prior: optional (x, y, th) pulled toward with weight lam (unit residual per
    meter / radian) — a weak motion-model anchor that constrains the solution
    only along directions the scan geometry leaves free.
    """
    x, y, th = seed
    for it in range(max_iter):
        c, s = np.cos(th), np.sin(th)
        q = np.c_[c * pts[:, 0] - s * pts[:, 1] + x,
                  s * pts[:, 0] + c * pts[:, 1] + y]
        d, j = tree.query(q, k=1, distance_upper_bound=gate)
        m = np.isfinite(d)
        if m.sum() < 30:
            return (x, y, th), dict(n_matched=int(m.sum()), inlier_frac=0.0,
                                    rmse=np.nan, ok=False)
        qi, ji = q[m], j[m]
        n = mnrm[ji]
        res = np.einsum("ij,ij->i", n, qi - mpts[ji])
        ares = np.maximum(np.abs(res), 1e-12)
        w = np.where(ares < huber, 1.0, huber / ares)
        # jacobian: d(residual)/d(x,y,th);  dq/dth = R'(th) p
        px, py = pts[m, 0], pts[m, 1]
        dqx = -s * px - c * py
        dqy = c * px - s * py
        J = np.c_[n[:, 0], n[:, 1], n[:, 0] * dqx + n[:, 1] * dqy]
        A = (J * w[:, None]).T @ J
        b = (J * w[:, None]).T @ res
        if prior is not None:
            px_, py_, pth_ = prior
            dth_p = np.arctan2(np.sin(th - pth_), np.cos(th - pth_))
            A[0, 0] += lam; A[1, 1] += lam; A[2, 2] += lam
            b += lam * np.array([x - px_, y - py_, dth_p])
        try:
            delta = np.linalg.solve(A, -b)
        except np.linalg.LinAlgError:
            break
        x += delta[0]; y += delta[1]; th += delta[2]
        if np.abs(delta[:2]).max() < 1e-5 and abs(delta[2]) < 1e-6:
            break
    inl = np.abs(res) < INLIER_D
    rmse = float(np.sqrt(np.mean(res[inl] ** 2))) if inl.any() else np.nan
    return (x, y, th), dict(n_matched=int(m.sum()),
                            inlier_frac=float(inl.mean()), rmse=rmse, ok=True)

def build_map(poses, ks):
    """Voxel-deduplicated map (points + normals) from scans ks at given poses."""
    P, Nn = [], []
    for k in ks:
        pts, nrm = SCANS[k]
        if pts is None:
            continue
        ok = np.isfinite(nrm[:, 0])
        w = transform(poses[k], pts[ok])
        x, y, th = poses[k]
        c, s = np.cos(th), np.sin(th)
        nw = np.c_[c * nrm[ok, 0] - s * nrm[ok, 1], s * nrm[ok, 0] + c * nrm[ok, 1]]
        P.append(w); Nn.append(nw)
    P = np.vstack(P); Nn = np.vstack(Nn)
    # voxel MEAN (not keep-first): centroids move smoothly as poses update,
    # which keeps the map stable across refinement rounds.
    key = np.floor(P / VOXEL).astype(np.int64)
    _, inv_idx, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    nv = len(counts)
    cp = np.zeros((nv, 2)); cn = np.zeros((nv, 2))
    np.add.at(cp, inv_idx, P)
    np.add.at(cn, inv_idx, Nn)
    cp /= counts[:, None]
    L = np.hypot(cn[:, 0], cn[:, 1])
    ok = L > 0.3 * counts        # drop voxels with inconsistent normal directions
    cn[ok] /= L[ok, None]
    return cp[ok], cn[ok]

# ------------------------------------------------------------------- pass 1
print("pass 1: incremental scan-to-map...")
poses = [None] * N
stats = [None] * N
poses[0] = tuple(np.array(odom_pose(ts_rel[0]), dtype=float))
stats[0] = dict(n_matched=len(SCANS[0][0]), inlier_frac=1.0, rmse=0.0, ok=True)
mpts, mnrm = build_map(poses[:1], [0])
tree = cKDTree(mpts)
odo_prev = odom_pose(ts_rel[0])
fallbacks = []
for k in range(1, N):
    odo_k = odom_pose(ts_rel[k])
    seed = compose(poses[k - 1], compose(inverse(odo_prev), odo_k))
    if SCANS[k][0] is None:
        poses[k], stats[k] = seed, dict(n_matched=0, inlier_frac=0.0, rmse=np.nan, ok=False)
        fallbacks.append(k)
    else:
        poses[k], stats[k] = icp(SCANS[k][0], seed, tree, mpts, mnrm)
        if not stats[k]["ok"] or stats[k]["inlier_frac"] < 0.5:
            poses[k] = seed
            fallbacks.append(k)
    odo_prev = odo_k
    if k % 5 == 0 or k == N - 1:          # map update (dedup keeps it bounded)
        mpts, mnrm = build_map(poses[:k + 1], range(0, k + 1, 2))
        tree = cKDTree(mpts)
print(f"  pass1 done, map pts={len(mpts)}, fallbacks={len(fallbacks)}: {fallbacks[:10]}")

# -------------------------------------------------------- refinement rounds
# Scan 0 is PINNED for all rounds: its pose defines the map frame (gauge fix).
# Re-anchoring the whole trajectory each round would rotate about the distant
# frame origin and inject centimeter-level gauge oscillation — never do that.
def refine(rounds, gate, huber, label, lam=40.0):
    global poses, stats
    for rnd in range(rounds):
        mpts, mnrm = build_map(poses, range(N))
        tree = cKDTree(mpts)
        new_poses, new_stats = [None] * N, [None] * N
        corr = np.zeros(N)
        for k in range(N):
            if SCANS[k][0] is None:
                new_poses[k], new_stats[k] = poses[k], stats[k]
                continue
            if k == 0:
                pr = poses[0]      # chain start: previous-round pose
            else:
                # motion-model prior: this round's previous pose + odometry delta
                odoD = compose(inverse(odom_pose(ts_rel[k - 1])), odom_pose(ts_rel[k]))
                pr = compose(new_poses[k - 1], odoD)
            p, st = icp(SCANS[k][0], poses[k], tree, mpts, mnrm,
                        gate=gate, huber=huber, prior=pr, lam=lam)
            if not st["ok"] or st["inlier_frac"] < 0.5:
                p, st = poses[k], dict(**stats[k])
            corr[k] = np.hypot(p[0] - poses[k][0], p[1] - poses[k][1])
            new_poses[k], new_stats[k] = p, st
        poses, stats = new_poses, new_stats
        print(f"  {label} round {rnd + 1}: map={len(mpts)} pts, corr mm "
              f"p50={np.percentile(corr,50)*1000:.2f} p90={np.percentile(corr,90)*1000:.2f} "
              f"p99={np.percentile(corr,99)*1000:.2f} max={corr.max()*1000:.1f} "
              f"(worst k={int(np.argmax(corr))})")
        if corr.max() < 0.002:
            break

refine(6, GATE, HUBER, "refine")
refine(4, 0.15, 0.03, "polish")

# Final gauge fix (applied ONCE): align the mean pose of the stationary start
# segment (scans 0..24, robot immobile) to the mean odometry pose at the same
# stamps. This defines the map frame as "odom frame during the initial
# stationary segment" without pinning any single noisy scan.
gx = np.mean([poses[k][0] for k in range(25)])
gy = np.mean([poses[k][1] for k in range(25)])
gth = np.arctan2(np.mean([np.sin(poses[k][2]) for k in range(25)]),
                 np.mean([np.cos(poses[k][2]) for k in range(25)]))
ax_, ay_, ath_ = (np.mean(v) for v in odom_pose(ts_rel[:25]))
anchor = compose((ax_, ay_, ath_), inverse((gx, gy, gth)))
poses = [compose(anchor, p) for p in poses]
print(f"  final gauge shift: {np.hypot(*anchor[:2])*1000:.1f} mm rotation {np.degrees(anchor[2]):.4f} deg (one-time frame alignment)")

# ------------------------------------------------------------------ outputs
X = np.array([p[0] for p in poses])
Y = np.array([p[1] for p in poses])
TH = np.array([p[2] for p in poses])
TH_wrapped = np.arctan2(np.sin(TH), np.cos(TH))
qual = pd.DataFrame(stats)
gt = pd.DataFrame(dict(
    t_ns=t_scan, x=X, y=Y, yaw=TH_wrapped,
    n_matched=qual.n_matched.astype(int), inlier_frac=qual.inlier_frac.round(4),
    rmse_m=qual.rmse.round(5),
    quality=np.where(qual.ok & (qual.inlier_frac >= 0.5), "good", "fallback")))
gt.to_csv(os.path.join(GT_DIR, "gt_pose.csv"), index=False)

mpts, mnrm = build_map(poses, range(N))
np.savez_compressed(os.path.join(GT_DIR, "map_points.npz"),
                    xy=mpts, normals=mnrm.astype(np.float32))

# ------------------------------------------------------------------ validation
print("\n=== VALIDATION ===")
good = gt.quality == "good"
print(f"good scans: {good.sum()}/{N}  (fallbacks: {(~good).sum()})")
print(f"rmse: median {gt.rmse_m[good].median()*1000:.1f} mm  p95 {gt.rmse_m[good].quantile(0.95)*1000:.1f} mm  max {gt.rmse_m[good].max()*1000:.1f} mm")
print(f"inlier_frac: median {gt.inlier_frac[good].median():.3f}  min {gt.inlier_frac[good].min():.3f}")

# loop closure: match last stationary scans against a start-only map
m0, n0 = build_map(poses, range(25))
tree0 = cKDTree(m0)
lc = []
for k in range(N - 40, N):
    if SCANS[k][0] is None:
        continue
    p, st = icp(SCANS[k][0], poses[k], tree0, m0, n0)
    if st["ok"] and st["inlier_frac"] > 0.3:
        lc.append((np.hypot(p[0] - poses[k][0], p[1] - poses[k][1]),
                   abs(np.arctan2(np.sin(p[2] - poses[k][2]), np.cos(p[2] - poses[k][2])))))
lc = np.array(lc)
if len(lc):
    print(f"loop closure (end scans vs start-only map, n={len(lc)}): "
          f"median {np.median(lc[:,0])*1000:.1f} mm / {np.degrees(np.median(lc[:,1])):.3f} deg, "
          f"max {lc[:,0].max()*1000:.1f} mm / {np.degrees(lc[:,1].max()):.3f} deg")

# pose repeatability from stationary segments (direct precision estimate)
for name, ks in [("start (scans 0-24)", range(25)), ("end (last 40)", range(N - 40, N))]:
    sx = X[list(ks)] * 1000
    sy = Y[list(ks)] * 1000
    sth = np.degrees(np.unwrap(TH[list(ks)]))
    print(f"stationary scatter {name}: std x {sx.std():.1f} mm, y {sy.std():.1f} mm, "
          f"yaw {sth.std():.3f} deg;  range x {np.ptp(sx):.1f} mm, y {np.ptp(sy):.1f} mm")

# GT twist vs odom twist
dt = np.diff(ts_rel)
vx = np.hypot(np.diff(X), np.diff(Y)) / dt * np.sign(
    np.cos(TH[:-1]) * np.diff(X) + np.sin(TH[:-1]) * np.diff(Y))
wz = np.diff(np.unwrap(TH)) / dt
ov = np.interp(ts_rel[:-1] + dt / 2, ot, od.v_lin.values)
ow = np.interp(ts_rel[:-1] + dt / 2, ot, od.w_ang.values)
k5 = np.ones(5) / 5
vs, ws = np.convolve(vx, k5, "same"), np.convolve(wz, k5, "same")
print(f"GT-vs-odom twist (raw finite diff): v corr {np.corrcoef(vx, ov)[0,1]:.4f} rmse {np.sqrt(np.mean((vx-ov)**2)):.4f} m/s; "
      f"w corr {np.corrcoef(wz, ow)[0,1]:.4f} rmse {np.sqrt(np.mean((wz-ow)**2)):.4f} rad/s")
print(f"GT-vs-odom twist (5-tap smoothed):  v corr {np.corrcoef(vs, ov)[0,1]:.4f} rmse {np.sqrt(np.mean((vs-ov)**2)):.4f} m/s; "
      f"w corr {np.corrcoef(ws, ow)[0,1]:.4f} rmse {np.sqrt(np.mean((ws-ow)**2)):.4f} rad/s")

# odom drift vs GT
oxx, oyy, oth = odom_pose(ts_rel)
drift = np.hypot(oxx - X, oyy - Y)
dyaw = np.degrees(np.abs(np.arctan2(np.sin(oth - TH), np.cos(oth - TH))))
print(f"odom drift vs GT: final {drift[-1]*100:.1f} cm / {dyaw[-1]:.2f} deg;  max {drift.max()*100:.1f} cm / {dyaw.max():.2f} deg")

# ------------------------------------------------------------------- figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(15, 6.5), width_ratios=[1.35, 1])
ax[0].scatter(mpts[:, 0], mpts[:, 1], s=0.3, c="k", alpha=0.5, lw=0)
ax[0].plot(X, Y, "b-", lw=1.2, label="ground truth (lidar)")
ax[0].plot(oxx, oyy, "r--", lw=1.0, label="wheel odometry")
ax[0].plot(X[0], Y[0], "go", ms=9, label="start")
ax[0].plot(X[-1], Y[-1], "r*", ms=13, label="end")
ax[0].set_aspect("equal"); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[0].set_xlabel("x [m]"); ax[0].set_ylabel("y [m]")
ax[0].set_title(f"Map ({len(mpts)} pts) + trajectories")
ax[1].plot(ts_rel, drift * 100, "r-", label="position drift [cm]")
ax[1].set_xlabel("time [s]"); ax[1].set_ylabel("odom position error vs GT [cm]", color="r")
ax2 = ax[1].twinx()
ax2.plot(ts_rel, dyaw, "b-", alpha=0.7)
ax2.set_ylabel("odom yaw error [deg]", color="b")
ax[1].set_title("Wheel-odometry drift vs lidar GT"); ax[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(GT_DIR, "gt_overview.png"), dpi=130)

# occupancy grid (hits only, 1 cm)
res = 0.01
allp = []
for k in range(0, N, 1):
    pts, _ = SCANS[k]
    if pts is not None:
        allp.append(transform(poses[k], pts))
allp = np.vstack(allp)
x0m, y0m = allp.min(0) - 0.2
x1m, y1m = allp.max(0) + 0.2
W, H = int((x1m - x0m) / res) + 1, int((y1m - y0m) / res) + 1
ij = ((allp - [x0m, y0m]) / res).astype(int)
grid = np.zeros((H, W), dtype=np.int32)
np.add.at(grid, (ij[:, 1], ij[:, 0]), 1)

# map.png = the RAW georeferenced grid, exactly W x H pixels, 1 px = 1 cm.
# Row 0 of the PNG is the TOP of the map (max y), map_server-style.
from PIL import Image
img8 = (255 - np.clip(grid, 0, 5) * 51).astype(np.uint8)
Image.fromarray(np.flipud(img8), mode="L").save(os.path.join(GT_DIR, "map.png"))

# decorated human-readable preview (separate file; NOT georeferenced)
img = np.clip(grid, 0, 5) / 5.0
plt.figure(figsize=(12, 7))
plt.imshow(1 - img, origin="lower", cmap="gray",
           extent=[x0m, x1m, y0m, y1m])
plt.plot(X, Y, "b-", lw=0.8)
plt.title("Occupancy render (1 cm) + GT trajectory")
plt.xlabel("x [m]"); plt.ylabel("y [m]")
plt.savefig(os.path.join(GT_DIR, "map_preview.png"), dpi=150, bbox_inches="tight")

with open(os.path.join(GT_DIR, "map.yaml"), "w") as f:
    f.write(
        f"# Georeferencing for map.png (raw lidar hit-count grid, grayscale).\n"
        f"# Pixel values: 255 = no returns observed; 204/153/102/51/0 = 1..5+ hits.\n"
        f"# PNG row 0 is the TOP of the map: world_y = origin_y + (height_px - 1 - row) * resolution\n"
        f"#                                  world_x = origin_x + col * resolution\n"
        f"# origin_xy is the lower-left corner of the bottom-left pixel.\n"
        f"# For a decorated preview with axes/trajectory see map_preview.png (not georeferenced).\n"
        f"image: map.png\nresolution: {res}\norigin_xy: [{x0m:.4f}, {y0m:.4f}]\n"
        f"width_px: {W}\nheight_px: {H}\n"
        f"frame: map (= odom frame aligned over the initial stationary segment; see method.md)\n")
print("\nwrote gt_pose.csv, map_points.npz, gt_overview.png, map.png, map_preview.png, map.yaml")
