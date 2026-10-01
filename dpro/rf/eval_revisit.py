"""Phase 2 evaluation: are the revisit proposals right? Raw fingerprint distance vs the learned check.

For every golden run, propose revisit pairs (>= 7 scans apart, best first, near-duplicates suppressed, N // 6
pairs) and count how many are truly within 2 m / 3 m. Learned scores come from a model that never saw the run's
zone (trained on synthetic sites + the other zone's real runs), or from synthetic sites only.
Also the ranking quality (AUC) over all within-run pairs >= 7 apart.

  python -m dpro.rf.eval_revisit --models /root/navlori/runs/dpro/revisit --out revisit_eval.json
"""
import argparse, json
import numpy as np
import torch

from ..data import load_golden
from .graph import fingerprint_distance, propose_pairs
from .revisit import PairNet, score_run


def auc(scores, labels):
    s, l = np.asarray(scores), np.asarray(labels, bool)
    if l.all() or (~l).all():
        return float("nan")
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[l].sum() - l.sum() * (l.sum() + 1) / 2) / (l.sum() * (~l).sum()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args(); torch.set_num_threads(2)
    nets = {}
    for k in ["syn", "syn_east", "syn_west"]:
        n = PairNet(); n.load_state_dict(torch.load(f"{args.models}/{k}.pth", weights_only=True)); nets[k] = n.eval()
    seqs = load_golden()
    res = {m: dict(gd=[], auc_s=[], auc_l=[]) for m in ["raw", "learned (synthetic only)", "learned (synthetic + other zone)"]}
    per_run, scores = {}, {}
    for s in seqs:
        G = s["G"]; N = len(G)
        GD = np.hypot(G[:, None, 0] - G[None, :, 0], G[:, None, 1] - G[None, :, 1])
        FD = fingerprint_distance(s["R"])
        other = "syn_west" if s["zone"] == "east" else "syn_east"
        P_syn, P_oth = score_run(nets["syn"], s), score_run(nets[other], s)
        scores[s["name"]] = P_oth
        iu = [(i, j) for i in range(N) for j in range(i + 7, N)]
        lab = [GD[i, j] < 3 for i, j in iu]
        per_run[s["name"]] = {}
        for m, M, hib in [("raw", FD, False), ("learned (synthetic only)", P_syn, True), ("learned (synthetic + other zone)", P_oth, True)]:
            prop = propose_pairs(M, higher_is_better=hib)
            gd = [GD[i, j] for i, j, _ in prop]
            res[m]["gd"] += gd
            sc = [(-M[i, j] if not hib else M[i, j]) for i, j in iu]
            per_run[s["name"]][m] = dict(n=len(gd), lt2=float(np.mean(np.array(gd) < 2)), lt3=float(np.mean(np.array(gd) < 3)),
                                         auc=auc(sc, lab), pairs=[(int(i), int(j), round(float(GD[i, j]), 2)) for i, j, _ in prop])
    summary = {}
    for m, r in res.items():
        gd = np.array(r["gd"])
        aucs = [per_run[k][m]["auc"] for k in per_run if per_run[k][m]["auc"] == per_run[k][m]["auc"]]
        summary[m] = dict(n=int(len(gd)), lt2=float(np.mean(gd < 2)), lt3=float(np.mean(gd < 3)), median=float(np.median(gd)), auc_mean=float(np.mean(aucs)))
        print(f"{m:34s} proposals {len(gd):3d} | within 2 m {np.mean(gd < 2):.0%} | within 3 m {np.mean(gd < 3):.0%} | median {np.median(gd):.1f} m | mean AUC {np.mean(aucs):.2f}", flush=True)
    json.dump(dict(summary=summary, per_run=per_run), open(args.out, "w"), indent=1)
    np.savez_compressed(args.out.replace(".json", "_scores.npz"), **scores)
    print("REVISIT_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
