"""Choose the weight multipliers of the hand-weighted radio-flow solver on synthetic sites only
(the real golden runs are never used for tuning).

  python -m dpro.rf.tune --out tune.json
"""
import argparse, json, time
import numpy as np
import torch

from .. import sim as dsim
from ..evaluate import ate
from .slam import RadioFlowSLAM, RFConfig


def sites(n=24, scans=36, seed=2026):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        run = dsim.make_run(dsim.make_site(rng), rng, scans)
        run["R"] = run["R"][:, ~np.all(np.isnan(run["R"]), 0)]
        out.append(run)
    return out


def score(cfg, runs, seed=0):
    return float(np.mean([np.median(ate(RadioFlowSLAM(cfg, seed).run(r)["X"], r["G"])) for r in runs]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args(); torch.set_num_threads(args.threads)
    runs = sites()
    C = [1.0, 0.3, 0.1, 0.03, 0.01]
    res, best = {}, {}
    t0 = time.time()
    for name, mk in [("abs_W1", lambda c: RFConfig(W=1, use_flow=False, use_abs=True, c_abs=c)),
                     ("flow_W1", lambda c: RFConfig(W=1, c_flow=c)),
                     ("flow_W3", lambda c: RFConfig(W=3, c_flow=c))]:
        res[name] = {c: score(mk(c), runs) for c in C}
        best[name] = min(res[name], key=res[name].get)
        print(f"{name}: " + " ".join(f"{c}:{v:.2f}" for c, v in res[name].items()) + f" -> {best[name]}  ({time.time() - t0:.0f}s)", flush=True)
    cf, ca = best["flow_W3"], best["abs_W1"]
    grid = {}
    for f in sorted({cf, cf * 0.3, cf * 3}):
        for a in sorted({ca, ca * 0.3, ca * 0.1}):
            grid[(f, a)] = score(RFConfig(W=3, use_abs=True, c_flow=f, c_abs=a), runs)
    bfa = min(grid, key=grid.get)
    res["flow_abs_W3"] = {f"{f:g},{a:g}": v for (f, a), v in grid.items()}; best["flow_abs_W3"] = list(bfa)
    print("flow_abs_W3: " + " ".join(f"{k}:{v:.2f}" for k, v in res["flow_abs_W3"].items()) + f" -> {bfa}", flush=True)
    res["flow_W3_rev"] = {c: score(RFConfig(W=3, c_flow=cf, revisits="raw", c_rev=c), runs) for c in [1.0, 0.3, 0.1, 0.03]}
    best["flow_W3_rev"] = min(res["flow_W3_rev"], key=res["flow_W3_rev"].get)
    print("flow_W3_rev: " + " ".join(f"{c}:{v:.2f}" for c, v in res["flow_W3_rev"].items()) + f" -> {best['flow_W3_rev']}", flush=True)
    res["K"] = {K: score(RFConfig(W=3, c_flow=cf, K=K), runs) for K in [2, 4, 8, 12]}
    print("K (flow_W3): " + " ".join(f"{k}:{v:.2f}" for k, v in res["K"].items()), flush=True)
    json.dump(dict(results={k: {str(kk): vv for kk, vv in v.items()} for k, v in res.items()}, best=best,
                   n_sites=len(runs), seconds=time.time() - t0), open(args.out, "w"), indent=1)
    print("TUNE_DONE", flush=True)


if __name__ == "__main__":
    main()
