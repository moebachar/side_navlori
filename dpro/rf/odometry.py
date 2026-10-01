"""Phase 4 helpers: simulated wheel odometry for synthetic runs, and tuning of the odometry variant on them.

Simulated odometry = true steps with per-step noise, a scale error and a slowly drifting heading bias,
integrated from the start (its own frame, rotated by a random angle).

  python -m dpro.rf.odometry --tune tune.json --out tune_odo.json
"""
import argparse, json, time
import numpy as np
import torch

from .. import sim as dsim
from ..evaluate import ate
from .slam import RadioFlowSLAM, RFConfig


def sim_odometry(G, rng, sig0=0.02, sig_rel=0.03, scale_sd=0.02, drift_deg=0.4):
    steps = np.diff(G, axis=0)
    scale = 1 + rng.normal(0, scale_sd)
    bias = np.cumsum(rng.normal(0, np.deg2rad(drift_deg), len(steps)))
    rot0 = rng.uniform(0, 2 * np.pi)
    out = [np.zeros(2)]
    for s, b in zip(steps, bias):
        n = np.hypot(*s)
        s = s * scale + rng.normal(0, sig0 + sig_rel * n, 2)
        c, si = np.cos(b + rot0), np.sin(b + rot0)
        out.append(out[-1] + np.array([c * s[0] - si * s[1], si * s[0] + c * s[1]]))
    return np.array(out)


def sites_with_odometry(n=24, scans=36, seed=2026):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        run = dsim.make_run(dsim.make_site(rng), rng, scans)
        run["R"] = run["R"][:, ~np.all(np.isnan(run["R"]), 0)]
        run["odom"] = sim_odometry(run["G"], rng)
        out.append(run)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tune", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args(); torch.set_num_threads(2)
    best = json.load(open(args.tune))["best"]
    runs = sites_with_odometry()
    score = lambda cfg: float(np.mean([np.median(ate(RadioFlowSLAM(cfg, 0).run(r)["X"], r["G"])) for r in runs]))
    t0 = time.time(); res = {}
    res["odometry only"] = float(np.mean([np.median(ate(r["odom"], r["G"])) for r in runs]))
    res["odometry in the solver, no WiFi"] = score(RFConfig(W=3, use_flow=False, odometry=True, fixedp=1))
    print(f"odometry only {res['odometry only']:.2f} | in the solver without WiFi {res['odometry in the solver, no WiFi']:.2f}", flush=True)
    sweep = {}
    for c in [1.0, 0.3, 0.1, 0.03, 0.01]:
        sweep[c] = score(RFConfig(W=3, c_flow=c, odometry=True, fixedp=1))
        print(f"odometry + radio flow, c_flow {c}: {sweep[c]:.2f} ({time.time() - t0:.0f}s)", flush=True)
    bc = min(sweep, key=sweep.get)
    res["sweep"] = {str(k): v for k, v in sweep.items()}; res["best_c_flow"] = bc
    rv = {c: score(RFConfig(W=3, c_flow=bc, odometry=True, revisits="raw", c_rev=c, fixedp=1)) for c in [1.0, 0.3, 0.1, 0.03]}
    res["rev_sweep"] = {str(k): v for k, v in rv.items()}; res["best_c_rev"] = min(rv, key=rv.get)
    print("with raw revisits: " + " ".join(f"{k}:{v:.2f}" for k, v in rv.items()), flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("TUNE_ODO_DONE", flush=True)


if __name__ == "__main__":
    main()
