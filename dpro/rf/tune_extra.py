"""Extend the synthetic tuning grid where the best value sat at its edge (abs weight, abs part of flow+abs,
revisit weight). Updates the 'best' entries in place.   python -m dpro.rf.tune_extra --tune tune.json"""
import argparse, json
import torch
from .tune import sites, score
from .slam import RFConfig


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tune", required=True); args = ap.parse_args()
    torch.set_num_threads(2)
    T = json.load(open(args.tune)); R, B = T["results"], T["best"]
    runs = sites()
    for c in [3.0, 10.0]:
        R["abs_W1"][str(c)] = score(RFConfig(W=1, use_flow=False, use_abs=True, c_abs=c), runs)
    B["abs_W1"] = float(min(R["abs_W1"], key=lambda k: R["abs_W1"][k]))
    print("abs_W1: " + " ".join(f"{k}:{v:.2f}" for k, v in R["abs_W1"].items()) + f" -> {B['abs_W1']}", flush=True)
    for f, a in [(0.09, 3.0), (0.03, 3.0), (0.03, 1.0), (0.09, 10.0)]:
        R["flow_abs_W3"][f"{f:g},{a:g}"] = score(RFConfig(W=3, use_abs=True, c_flow=f, c_abs=a), runs)
    k = min(R["flow_abs_W3"], key=lambda k: R["flow_abs_W3"][k]); B["flow_abs_W3"] = [float(x) for x in k.split(",")]
    print("flow_abs_W3: " + " ".join(f"{k}:{v:.2f}" for k, v in R["flow_abs_W3"].items()) + f" -> {B['flow_abs_W3']}", flush=True)
    for c in [3.0, 10.0]:
        R["flow_W3_rev"][str(c)] = score(RFConfig(W=3, c_flow=float(B["flow_W3"]), revisits="raw", c_rev=c), runs)
    B["flow_W3_rev"] = float(min(R["flow_W3_rev"], key=lambda k: R["flow_W3_rev"][k]))
    print("flow_W3_rev: " + " ".join(f"{k}:{v:.2f}" for k, v in R["flow_W3_rev"].items()) + f" -> {B['flow_W3_rev']}", flush=True)
    B["flow_W1"] = float(B["flow_W1"]); B["flow_W3"] = float(B["flow_W3"])
    json.dump(T, open(args.tune, "w"), indent=1)
    print("TUNE_EXTRA_DONE", flush=True)


if __name__ == "__main__":
    main()
