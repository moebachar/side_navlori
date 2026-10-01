"""Phase 4 follow-up: estimate a constant heading-rate bias of the wheel odometry from the WiFi alone.

In runs 4 and 9 the odometry heading turns at a constant wrong rate (-21.9 and +4.6 deg/s against the ground
truth; the other 10 runs stay under 0.1 deg/s): the step lengths are right, the turning is not, so the path
curls into loops. Found by comparing with the truth; the fix itself uses no truth:

  for each candidate bias b: rotate the odometry steps by -b * t, fit every AP's log-distance model to that
  path (robust least squares), and score how well the path explains the RSSI. Keep the best b only if it
  beats b = 0 by a margin (chosen on synthetic runs, not on the real ones).

  python -m dpro.rf.gyro_bias --out gyro_bias.json
"""
import argparse, json, time
import numpy as np

from ..data import load_golden


def correct(odom, t, b):
    """Odometry with a heading-rate bias b (rad/s) removed: step k is rotated by -b * (mid time of step k)."""
    d = np.diff(odom, axis=0); tm = (t[1:] + t[:-1]) / 2 - t[0]
    c, s = np.cos(-b * tm), np.sin(-b * tm)
    d2 = np.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]], 1)
    return np.vstack([odom[:1], odom[:1] + np.cumsum(d2, 0)])


def radio_cost(X, R, min_n=5, margin=10.0, cell=0.75, f=4.0):
    """How well a path explains the RSSI: per AP, the best log-distance fit over a grid of AP positions
    (P0 and n in closed form, n kept in [1, 6]), robust (soft-L1, 4 dB) cost; mean over all readings."""
    lo, hi = X.min(0) - margin, X.max(0) + margin
    gx, gy = np.meshgrid(np.arange(lo[0], hi[0], cell), np.arange(lo[1], hi[1], cell))
    cells = np.stack([gx.ravel(), gy.ravel()], 1)
    cost, n = 0.0, 0
    for a in range(R.shape[1]):
        m = ~np.isnan(R[:, a])
        if m.sum() < min_n:
            continue
        xy, y = X[m], R[m, a]
        L = np.log10(np.hypot(cells[:, None, 0] - xy[None, :, 0], cells[:, None, 1] - xy[None, :, 1]) + 0.5)
        Lc, yc = L - L.mean(1, keepdims=True), y - y.mean()
        nn = np.clip(-(Lc @ yc) / (10 * (Lc ** 2).sum(1) + 1e-12), 1.0, 6.0)
        P0 = (y[None] + 10 * nn[:, None] * L).mean(1)
        r = y[None] - (P0[:, None] - 10 * nn[:, None] * L)
        cost += float((f ** 2 * (np.sqrt(1 + (r / f) ** 2) - 1)).sum(1).min()); n += int(m.sum())
    return cost / max(n, 1)


def scan(odom, t, R, grid_deg):
    return np.array([radio_cost(correct(odom, t, np.deg2rad(b)), R) for b in grid_deg])


def estimate(odom, t, R, coarse=np.arange(-30, 30.01, 1.0), fine_step=0.1):
    cc = scan(odom, t, R, coarse); b0 = coarse[int(np.argmin(cc))]
    fine = np.arange(b0 - 1, b0 + 1.0001, fine_step); cf = scan(odom, t, R, fine)
    b = fine[int(np.argmin(cf))]
    c0 = radio_cost(odom, R)
    return dict(b_deg_s=float(b), cost=float(cf.min()), cost0=float(c0), gain=float(c0 / max(cf.min(), 1e-9)),
                coarse=coarse.tolist(), coarse_cost=cc.tolist())


def causal(odom, t, R, tau, min_scans=8):
    """Online version: at each scan, re-estimate the bias from the scans so far and correct the latest position."""
    N = len(t); Xc = np.asarray(odom, float).copy(); b_now = 0.0
    for j in range(min_scans - 1, N):
        e = estimate(odom[:j + 1], t[:j + 1], R[:j + 1])
        b_now = np.deg2rad(e["b_deg_s"]) if e["gain"] > tau else 0.0
        Xc[j] = correct(odom[:j + 1], t[:j + 1], b_now)[j]
    return Xc


def synthetic_runs(n_per=12, lengths=(12, 24, 36, 60), seed=4242, b_range=(2.0, 25.0)):
    """Synthetic runs with simulated odometry; half of them get a constant heading-rate bias (deg/s)."""
    from .. import sim as dsim
    from .odometry import sim_odometry
    rng = np.random.default_rng(seed); out = []
    for L in lengths:
        for k in range(n_per):
            run = dsim.make_run(dsim.make_site(rng), rng, L)
            run["R"] = run["R"][:, ~np.all(np.isnan(run["R"]), 0)]
            o = sim_odometry(run["G"], rng)
            b = float(rng.choice([-1, 1]) * rng.uniform(*b_range)) if k % 2 else 0.0
            run["odom"] = correct(o, run["t"], -np.deg2rad(b)); run["b_true"] = b
            out.append(run)
    return out


def calibrate(taus=(1.0, 1.02, 1.04, 1.06, 1.08, 1.1, 1.15, 1.2, 1.3, 1.5)):
    """Choose the acceptance margin on synthetic runs: lowest mean error of the corrected odometry."""
    from ..evaluate import ate
    runs = synthetic_runs(); rows = []
    for r in runs:
        e = estimate(r["odom"], r["t"], r["R"])
        err_raw = float(np.median(ate(r["odom"], r["G"])))
        err_cor = float(np.median(ate(correct(r["odom"], r["t"], np.deg2rad(e["b_deg_s"])), r["G"])))
        rows.append(dict(n=len(r["t"]), b_true=r["b_true"], b_hat=e["b_deg_s"], gain=e["gain"], err_raw=err_raw, err_cor=err_cor))
    score = {str(tau): float(np.mean([x["err_cor"] if x["gain"] > tau else x["err_raw"] for x in rows])) for tau in taus}
    best = min(taus, key=lambda tau: score[str(tau)])
    return dict(rows=rows, score=score, tau=best,
                no_correction=float(np.mean([x["err_raw"] for x in rows])), always=float(np.mean([x["err_cor"] for x in rows])))


def gt_slope(s):
    G, O, t = s["G"], s["odom"], s["t"]
    dg, do = np.diff(G, axis=0), np.diff(O, axis=0)
    mv = (np.hypot(*dg.T) > 0.3) & (np.hypot(*do.T) > 0.3)
    off = np.unwrap(np.arctan2(do[mv, 1], do[mv, 0]) - np.arctan2(dg[mv, 1], dg[mv, 0]))
    tm = ((t[1:] + t[:-1]) / 2 - t[0])[mv]
    return float(np.rad2deg(np.polyfit(tm, off, 1)[0])) if mv.sum() > 3 else 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    args = ap.parse_args(); t0 = time.time()
    cal = calibrate()
    print("synthetic: margin -> mean error " + " ".join(f"{k}:{v:.2f}" for k, v in cal["score"].items())
          + f" | chosen {cal['tau']} | never correct {cal['no_correction']:.2f}, always {cal['always']:.2f} ({time.time() - t0:.0f}s)", flush=True)
    res = dict(calibration=cal, runs={})
    for s in load_golden():
        e = estimate(np.asarray(s["odom"], float), s["t"], s["R"])
        e["truth_deg_s"] = gt_slope(s); e["accepted"] = bool(e["gain"] > cal["tau"])
        res["runs"][s["name"]] = e
        print(f"{s['name']:14s} estimated {e['b_deg_s']:6.2f} deg/s  truth {e['truth_deg_s']:6.2f}  cost gain over b=0 {e['gain']:.3f}"
              f"  {'CORRECTED' if e['accepted'] else 'kept'}  ({time.time() - t0:.0f}s)", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("GYRO_DONE", flush=True)


if __name__ == "__main__":
    main()
