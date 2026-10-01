"""Phase 3 in-domain check: hand-weighted solver vs learned update on synthetic sites never used in training
or tuning (different seed from both).

  python -m dpro.rf.eval_synth --tune tune.json --nets d1,d2,d3 --out synth_eval.json
"""
import argparse, json
from dataclasses import replace
import numpy as np
import torch

from .. import sim as dsim
from ..evaluate import ate
from .net import RFUpdate
from .slam import RadioFlowSLAM
from .train_net import hand_cfg


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tune", required=True); ap.add_argument("--nets", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=30); ap.add_argument("--scans", type=int, default=36)
    args = ap.parse_args(); torch.set_num_threads(1)
    best = json.load(open(args.tune))["best"]; base = hand_cfg(best)
    nets = []
    for d in args.nets.split(","):
        m = RFUpdate(); m.load_state_dict(torch.load(f"{d}/final.pth", weights_only=True)); nets.append(m.eval())
    rng = np.random.default_rng(777); res = {"hand": []}
    for k in range(len(nets)):
        res[f"learned_s{k}"] = []
    for i in range(args.n):
        run = dsim.make_run(dsim.make_site(rng), rng, args.scans)
        run["R"] = run["R"][:, ~np.all(np.isnan(run["R"]), 0)]
        res["hand"].append(float(np.median(ate(RadioFlowSLAM(base, 0).run(run)["X"], run["G"]))))
        for k, net in enumerate(nets):
            out = RadioFlowSLAM(replace(base, learned_iters=12), 0, net=net).run(run)
            res[f"learned_s{k}"].append(float(np.median(ate(out["X"], run["G"]))))
    summ = {k: dict(mean=float(np.mean(v)), median=float(np.median(v))) for k, v in res.items()}
    for k in res:
        if k != "hand":
            summ[k]["better_than_hand"] = int(np.sum(np.array(res[k]) < np.array(res["hand"])))
    for k, v in summ.items():
        print(k, v, flush=True)
    json.dump(dict(per_site=res, summary=summ, n=args.n), open(args.out, "w"), indent=1)
    print("SYNTH_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
