"""Phase 4 diagnostic, swept on the real runs (so a look at the trade-off, not a fair method): how much WiFi
weight would it take to fix the two runs where wheel odometry fails (4 and 9), and what does that weight cost
on the runs where odometry is good?

  python -m dpro.rf.odo_diag --out odo_diag.json
"""
import argparse, json, time
import torch

from ..data import load_golden
from .metrics import all_metrics
from .slam import RadioFlowSLAM, RFConfig

GRID = [("radio flow", dict(c_flow=c)) for c in [0.1, 0.3, 1.0, 3.0]] + \
       [("radio flow + abs", dict(c_flow=0.1, use_abs=True, c_abs=c)) for c in [0.1, 0.3, 1.0]]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    args = ap.parse_args(); torch.set_num_threads(1)
    seqs = load_golden(); res = {}; t0 = time.time()
    for name, kw in GRID:
        key = f"{name}, " + ", ".join(f"{k} {v}" for k, v in kw.items() if k != "use_abs")
        res[key] = {}
        for s in seqs:
            out = RadioFlowSLAM(RFConfig(W=3, odometry=True, fixedp=1, **kw), 0).run(s)
            res[key][s["name"]] = all_metrics(out["X"], out["Xc"], s["G"], s["t"])
        print(f"{key}: " + " ".join(f"{v['aligned']:.2f}" for v in res[key].values()) + f" ({time.time() - t0:.0f}s)", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("ODO_DIAG_DONE", flush=True)


if __name__ == "__main__":
    main()
