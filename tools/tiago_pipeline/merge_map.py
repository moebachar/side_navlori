"""Merge all TIAGo runs into one map with the tag network (network.json) and check how well they agree.

Outputs in --out:
  merged_map.png         lidar maps of all runs (grey), every path (one colour per run), tags with IDs
  merged_map_runs.png    one small panel per run on the merged map
  merged_points.npz      xy of every run's map points in the common frame + run index
  occupancy.png/.yaml    occupancy image of the merged map (5 cm/px), for registration on a floor plan later
  gt_common/<run>.csv    the SLAM path of each run in the common frame (t_ns, x, y, yaw)
  agreement.json         wall agreement: for each run, median distance from its map points to the other runs'
                         points (only where another run is within 0.5 m, i.e. where they see the same walls)

  python merge_map.py --processed .../processed --network .../network/network.json --out .../merged
"""
import argparse, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

RES = 0.05


def rot(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True); ap.add_argument("--network", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(os.path.join(a.out, "gt_common"), exist_ok=True)
    net = json.load(open(a.network))
    runs = sorted(net["runs"], key=lambda r: (r.startswith("run_final"), r))
    tags = {int(k): np.array(v) for k, v in net["tags"].items()}
    pts, owner, paths = [], [], {}
    for k, r in enumerate(runs):
        v = net["runs"][r]; R, t = rot(np.radians(v["theta_deg"])), np.array([v["tx"], v["ty"]])
        P = np.load(os.path.join(a.processed, r, "map_points.npz"))["xy"].astype(float) @ R.T + t
        pts.append(P); owner.append(np.full(len(P), k))
        g = pd.read_csv(os.path.join(a.processed, r, "gt_slam.csv"))
        xy = g[["x", "y"]].values @ R.T + t
        yaw = np.angle(np.exp(1j * (g.yaw.values + np.radians(v["theta_deg"]))))
        out = pd.DataFrame(dict(t_ns=g.t_ns, x=xy[:, 0], y=xy[:, 1], yaw=yaw, quality=g.quality))
        out.to_csv(os.path.join(a.out, "gt_common", f"{r}.csv"), index=False)
        paths[r] = xy
    P = np.vstack(pts); O = np.concatenate(owner)
    np.savez_compressed(os.path.join(a.out, "merged_points.npz"), xy=P.astype(np.float32), run=O.astype(np.int16), runs=np.array(runs))

    # wall agreement between runs (main runs only: they are what the dataset is)
    agree = {}
    main_runs = [r for r in runs if not r.startswith("run_final")]
    for k, r in enumerate(runs):
        mine = P[O == k]; others = P[(O != k) & np.isin(O, [runs.index(m) for m in main_runs])]
        if len(others) == 0:
            continue
        dist, _ = cKDTree(others).query(mine, distance_upper_bound=0.5)
        dist = dist[np.isfinite(dist)]
        agree[r] = dict(overlap_frac=float(len(dist) / len(mine)), median_cm=float(100 * np.median(dist)) if len(dist) else None,
                        p90_cm=float(100 * np.percentile(dist, 90)) if len(dist) else None)
    json.dump(agree, open(os.path.join(a.out, "agreement.json"), "w"), indent=1)

    # figures
    cmap = plt.get_cmap("tab20")
    lo, hi = P.min(0) - 1, P.max(0) + 1
    fig, ax = plt.subplots(figsize=(14, 14 * (hi[1] - lo[1]) / (hi[0] - lo[0]) + 1))
    ax.scatter(P[:, 0], P[:, 1], s=0.3, c="0.6", lw=0)
    for k, r in enumerate(runs):
        ls = "--" if r.startswith("run_final") else "-"
        ax.plot(*paths[r].T, ls, color=cmap(k % 20), lw=1.4, label=r)
        ax.plot(*paths[r][0], "o", color=cmap(k % 20), ms=4)
    for i, p in tags.items():
        ax.plot(*p, "s", color="k", ms=6); ax.annotate(str(i), p, xytext=(4, 4), textcoords="offset points", fontsize=9, weight="bold")
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"TIAGo++ Strasbourg: {len(main_runs)} runs + {len(runs) - len(main_runs)} demo runs (dashed) in one frame "
                 f"(frame = {net['ref']} start), {len(tags)} AprilTags (estimated)")
    ax.legend(loc="upper left", fontsize=8, ncol=2, markerscale=2)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "merged_map.png"), dpi=130); plt.close(fig)

    n = len(runs); cols = 4; rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows * (hi[1] - lo[1]) / (hi[0] - lo[0]) + 0.6))
    for k, (ax, r) in enumerate(zip(axs.ravel(), runs)):
        ax.scatter(P[:, 0], P[:, 1], s=0.1, c="0.75", lw=0)
        ax.scatter(*pts[k].T, s=0.2, color=cmap(k % 20), lw=0)
        ax.plot(*paths[r].T, color="k", lw=1)
        for i in net["runs"][r]["tags"]:
            if i in tags:
                ax.plot(*tags[i], "s", color="r", ms=3)
        ag = agree.get(r, {})
        ax.set_title(f"{r}: walls agree {ag.get('median_cm', float('nan')):.1f} cm", fontsize=9)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "merged_map_runs.png"), dpi=110); plt.close(fig)

    # occupancy image (white = free/unknown, black = hit), origin at the lower-left corner
    W, H = (np.ceil((hi - lo) / RES)).astype(int)
    img = np.full((H, W), 255, np.uint8)
    ij = ((P - lo) / RES).astype(int)
    cnt = np.zeros((H, W), int); np.add.at(cnt, (H - 1 - ij[:, 1], ij[:, 0]), 1)
    img[cnt >= 2] = 0
    plt.imsave(os.path.join(a.out, "occupancy.png"), img, cmap="gray", vmin=0, vmax=255)
    with open(os.path.join(a.out, "occupancy.yaml"), "w") as f:
        f.write(f"image: occupancy.png\nresolution: {RES}\norigin: [{lo[0]:.3f}, {lo[1]:.3f}, 0.0]\n"
                f"frame: common TIAGo frame = {net['ref']} SLAM map (odometry frame at its start)\n")
    print("agreement (median distance of a run's walls to the other runs' walls):")
    for r, v in agree.items():
        print(f"  {r:11s} {v['median_cm']:.1f} cm (p90 {v['p90_cm']:.1f}), overlap {100 * v['overlap_frac']:.0f}%")
    print("MERGE_DONE")


if __name__ == "__main__":
    main()
