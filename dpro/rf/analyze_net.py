"""Phase 3: what the learned update does on the real runs.

For each golden run: the phase-1 solver, then 12 learned iterations; record for every radio-flow factor the
final weight multiplier exp(z) and the target shift delta (dB), with its baseline (scans apart) and kind
(temporal or revisit), and whether a revisit link is truly within 3 m.

  python -m dpro.rf.analyze_net --tune tune.json --net dir --out net_behaviour.json
"""
import argparse, json
import numpy as np
import torch

from ..data import load_golden
from .net import RFUpdate
from .slam import RadioFlowSLAM
from .train_net import hand_cfg


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tune", required=True); ap.add_argument("--net", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args(); torch.set_num_threads(1)
    best = json.load(open(args.tune))["best"]; cfg = hand_cfg(best)
    net = RFUpdate(); net.load_state_dict(torch.load(f"{args.net}/final.pth", weights_only=True)); net.eval()
    rec = []
    for s in load_golden():
        sl = RadioFlowSLAM(cfg, 0); out = sl.run(s)
        f = sl.factor_dict(); N = len(s["G"])
        free = torch.zeros(N, dtype=torch.bool); free[cfg.fixedp:] = True
        _, _, traj = sl.learned_refine(torch.tensor(out["X"]), torch.tensor(out["TH"]), net, 12, free, N, f=f)
        mult = (traj[-1]["w"] / f["w_hand"]).numpy(); shift = (traj[-1]["target"] - f["dy"]).numpy()
        G = s["G"]; pi, pj = f["pi"].numpy(), f["pj"].numpy()
        gd = np.hypot(*(G[pj] - G[pi]).T)
        for k in range(len(mult)):
            rec.append((int(min(pj[k] - pi[k], 99)), int(f["kind"][k]), float(mult[k]), float(shift[k]), float(gd[k])))
    R = np.array(rec)
    out = {"by_k": {}, "revisit": {}}
    for k in range(1, 9):
        m = (R[:, 1] == 0) & (R[:, 0] == k)
        out["by_k"][k] = dict(n=int(m.sum()), mult_median=float(np.median(R[m, 2])), mult_mean=float(R[m, 2].mean()), shift_abs_median=float(np.median(np.abs(R[m, 3]))))
    rv = R[:, 1] == 1
    for lab, mm in [("true (< 3 m)", rv & (R[:, 4] < 3)), ("wrong (>= 3 m)", rv & (R[:, 4] >= 3))]:
        out["revisit"][lab] = dict(n=int(mm.sum()), mult_median=float(np.median(R[mm, 2])) if mm.any() else None, mult_mean=float(R[mm, 2].mean()) if mm.any() else None)
    print(json.dumps(out, indent=1))
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
