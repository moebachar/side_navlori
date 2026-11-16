"""Place every TIAGo run in one common frame with the AprilTag network (no surveyed tag coordinates yet).

Unknowns: one rigid 2D transform (theta, tx, ty) per run, and the 2D position of every tag.
Each detection d of tag i in run r gives the tag centre m_d in the run's own SLAM map frame (tags_detect.py):
    residual = R(theta_r) m_d + t_r - T_i            (metres, x and y)
weighted by sigma_d = 2 cm + 2 % of the camera-tag distance, with a Cauchy loss (scale 10 cm). The reference run's
transform is fixed (it defines the frame: its SLAM map, i.e. its odometry frame at the start of the run).
Like the Reims golden runs, tags only PLACE each run (rigid), they never bend it: lidar SLAM is cm-accurate locally.

Accuracy check (held-out): remove all detections of one tag in one run, re-solve, and measure how far that run's
own (median) observation of the tag lands from the tag position given by the other runs.

  python tag_network.py --tags /root/navlori/tiago_work/tags --out /root/navlori/tiago_work/network --ref run01
"""
import argparse, glob, json, os
import numpy as np, pandas as pd
from scipy.optimize import least_squares

SIG0, SIG_REL, LOSS_SCALE = 0.02, 0.02, 0.10


def load(tags_dir, max_view=70.0, max_reproj=2.0):
    dfs = []
    for f in sorted(glob.glob(os.path.join(tags_dir, "*_tags.csv"))):
        d = pd.read_csv(f)
        if len(d):
            d["run"] = os.path.basename(f)[:-9]
            dfs.append(d)
    d = pd.concat(dfs, ignore_index=True)
    return d[(d.view_deg < max_view) & (d.reproj_px < max_reproj)].reset_index(drop=True)


def rot(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def kabsch2(A, B):
    """theta, t with R A + t ~ B (rows are points)."""
    ma, mb = A.mean(0), B.mean(0)
    H = (A - ma).T @ (B - mb)
    th = np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
    return th, mb - rot(th) @ ma


def _chain(med, runs, seed):
    """Greedy chain inside one group: place runs one by one on the tags already placed (>= 2 shared tags)."""
    pose = {seed: (0.0, np.zeros(2))}
    tagpos = {t: med.loc[(seed, t)].values for t in med.loc[seed].index}
    while True:
        best = None
        for r in runs:
            if r in pose:
                continue
            shared = [t for t in med.loc[r].index if t in tagpos]
            if len(shared) >= 2 and (best is None or len(shared) > best[1]):
                best = (r, len(shared), shared)
        if best is None:
            return pose, tagpos
        r, _, shared = best
        A = np.array([med.loc[(r, t)].values for t in shared]); B = np.array([tagpos[t] for t in shared])
        th, t = kabsch2(A, B); pose[r] = (th, t)
        for tg in med.loc[r].index:
            if tg not in tagpos:
                tagpos[tg] = rot(th) @ med.loc[(r, tg)].values + t


def init(d, runs, ref):
    """Build groups of runs linked by >= 2 shared tags, then join whole groups that share >= 2 tags between them
    (a group can be linked through tags that no single run shares twice)."""
    med = d.groupby(["run", "tag"])[["mx", "my"]].median()
    left = [r for r in runs]; groups = []
    for seed in [ref] + [r for r in runs if r != ref]:
        if seed not in left:
            continue
        p, tp = _chain(med, left, seed)
        groups.append((p, tp)); left = [r for r in left if r not in p]
    pose, tagpos = groups[0]
    rest = groups[1:]
    while rest:
        joined = False
        for g in list(rest):
            gp, gt = g
            shared = [t for t in gt if t in tagpos]
            if len(shared) >= 2:
                th, t = kabsch2(np.array([gt[s] for s in shared]), np.array([tagpos[s] for s in shared]))
                for r, (a, b) in gp.items():
                    pose[r] = (a + th, rot(th) @ b + t)
                for tg, v in gt.items():
                    tagpos.setdefault(tg, rot(th) @ v + t)
                rest.remove(g); joined = True
        if not joined:
            break
    return pose, tagpos


def solve(d, runs, ref, pose0, tag0):
    free = [r for r in runs if r != ref and r in pose0]
    tags = sorted(tag0)
    ri = {r: k for k, r in enumerate(free)}; ti = {t: k for k, t in enumerate(tags)}
    x0 = np.concatenate([np.concatenate([[pose0[r][0]], pose0[r][1]]) for r in free] + [tag0[t] for t in tags])
    use = d[d.run.isin([ref] + free) & d.tag.isin(tags)]
    R_ = use.run.values; T_ = use.tag.values; M = use[["mx", "my"]].values
    w = 1.0 / (SIG0 + SIG_REL * use.dist_m.values)
    nr = 3 * len(free)

    def fun(x):
        th = np.zeros(len(use)); tx = np.zeros(len(use)); ty = np.zeros(len(use))
        for r, k in ri.items():
            m = R_ == r
            th[m], tx[m], ty[m] = x[3 * k], x[3 * k + 1], x[3 * k + 2]
        c, s = np.cos(th), np.sin(th)
        px = c * M[:, 0] - s * M[:, 1] + tx; py = s * M[:, 0] + c * M[:, 1] + ty
        tg = np.array([ti[t] for t in T_])
        ex = (px - x[nr + 2 * tg]) * w; ey = (py - x[nr + 2 * tg + 1]) * w
        return np.concatenate([ex, ey])

    sol = least_squares(fun, x0, loss="cauchy", f_scale=LOSS_SCALE * 25, x_scale="jac")
    x = sol.x
    pose = {ref: (0.0, np.zeros(2))}
    for r, k in ri.items():
        pose[r] = (x[3 * k], x[3 * k + 1:3 * k + 3])
    tagpos = {t: x[nr + 2 * k: nr + 2 * k + 2] for t, k in ti.items()}
    return pose, tagpos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True); ap.add_argument("--out", required=True); ap.add_argument("--ref", default="run01")
    ap.add_argument("--exclude", default="", help="comma list of runs placed afterwards on the solved tags (not in the network)")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    d = load(a.tags)
    excl = [r for r in a.exclude.split(",") if r]
    runs = sorted(r for r in d.run.unique() if r not in excl)
    pose0, tag0 = init(d[d.run.isin(runs)], runs, a.ref)
    unplaced = [r for r in runs if r not in pose0]
    pose, tagpos = solve(d, runs, a.ref, pose0, tag0)
    # runs kept out of the network (demo runs): placed on the solved tags, they do not move them
    for r in excl:
        med = d[d.run == r].groupby("tag")[["mx", "my"]].median()
        shared = [t for t in med.index if t in tagpos]
        if len(shared) >= 2:
            th, t = kabsch2(med.loc[shared].values, np.array([tagpos[s] for s in shared]))
            pose[r] = (th, t)
        else:
            unplaced.append(r)
    # residuals per run and held-out check
    med = d.groupby(["run", "tag"])[["mx", "my"]].median()
    report = {}
    for r in sorted(pose):
        th, t = pose[r]
        rr = [float(np.linalg.norm(rot(th) @ med.loc[(r, tg)].values + t - tagpos[tg])) for tg in med.loc[r].index if tg in tagpos]
        report[r] = dict(theta_deg=float(np.degrees(th)), tx=float(t[0]), ty=float(t[1]), tags=[int(x) for x in med.loc[r].index],
                         fit_resid_cm_median=100 * float(np.median(rr)), heldout_cm=[])
    for r in runs:
        if r == a.ref or r not in pose:
            continue
        for tg in med.loc[r].index:
            others = d[(d.tag == tg) & (d.run != r)].run.unique()
            others = [o for o in others if o in pose and o in runs]      # only network runs are independent witnesses
            if not others:
                continue
            dd = d[~((d.run == r) & (d.tag == tg))]
            p0 = {k: pose[k] for k in pose if k in runs}
            # warm start from the full solution, without the held-out sighting
            try:
                ph, tp = solve(dd, runs, a.ref, p0, {k: v for k, v in tagpos.items()})
            except Exception:
                continue
            th, t = ph[r]
            err = float(np.linalg.norm(rot(th) @ med.loc[(r, tg)].values + t - tp[tg]))
            report[r]["heldout_cm"].append(round(100 * err, 1))
    for r in report:
        h = report[r]["heldout_cm"]
        report[r]["heldout_cm_median"] = float(np.median(h)) if h else None
    json.dump(dict(ref=a.ref, unplaced=unplaced, excluded=excl, runs=report,
                   tags={int(k): [float(v[0]), float(v[1])] for k, v in tagpos.items()},
                   tag_z=d.groupby("tag").mz.median().round(3).to_dict()), open(os.path.join(a.out, "network.json"), "w"), indent=1)
    print(f"frame = {a.ref}; unplaced runs: {unplaced or 'none'}")
    for r, v in report.items():
        print(f"  {r:11s} tags {v['tags']}  fit {v['fit_resid_cm_median']:.1f} cm  held-out {v['heldout_cm_median']} cm  {v['heldout_cm']}")
    print("NETWORK_DONE")


if __name__ == "__main__":
    main()
