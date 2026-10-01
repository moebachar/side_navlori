"""Evaluate every method on the 12 golden runs with the three metrics (aligned, online, drift over 5 m).

  python -m dpro.rf.evaluate_golden --tune tune.json --out results.pkl [--only rf]

Seeds: 3 evaluation seeds (random patch picks in v1, random AP initial offsets in both); v1 DPRO also over its
3 trainings. Oracle rows use AP parameters fitted with the true positions of all 12 runs (a ceiling, not a method).
"""
import argparse, json, os, pickle, time
import numpy as np
import torch
from scipy.optimize import least_squares

from ..data import load_golden
from ..dpro import DPRO, Config
from ..net import RONet
from .slam import RadioFlowSLAM, RFConfig
from .metrics import all_metrics

CK = os.environ.get("DPRO_CK", "/root/navlori/runs/dpro")


def oracle_params(seqs):
    h = lambda x, th: th[2] - 10 * th[3] * np.log10(np.hypot(x[..., 0] - th[0], x[..., 1] - th[1]) + 0.5)
    G = np.vstack([s["G"] for s in seqs]); R = np.vstack([s["R"] for s in seqs]); TH = {}
    for a in range(R.shape[1]):
        m = ~np.isnan(R[:, a])
        if m.sum() < 15:
            continue
        xy, y = G[m], R[m, a]; i = int(np.argmax(y)); best = None
        for dx, dy in [(0, 0), (3, 0), (-3, 0), (0, 3), (0, -3)]:
            r = least_squares(lambda t: h(xy, t) - y, [xy[i, 0] + dx, xy[i, 1] + dy, min(y[i] + 5, -1), 2.5],
                              loss="soft_l1", f_scale=4.0, bounds=([-10, -10, -80, 1.0], [80, 40, 0, 6.0]))
            if best is None or r.cost < best.cost:
                best = r
        TH[a] = tuple(float(v) for v in best.x)
    return TH


def wknn_p1(seqs, k=5):
    """Fingerprinting with a radio map from the other 11 runs (radio-level APs), weighted kNN, powed RSS."""
    powed = lambda m: ((np.clip(m, -100, -20) + 100) / 80.0) ** np.e
    out = {}
    for s in seqs:
        tr = [q for q in seqs if q["name"] != s["name"]]
        A = powed(np.where(np.isnan(np.vstack([q["R"] for q in tr])), -100, np.vstack([q["R"] for q in tr])))
        P = np.vstack([q["G"] for q in tr]); B = powed(np.where(np.isnan(s["R"]), -100, s["R"]))
        D = np.sqrt(((B[:, None] - A[None]) ** 2).sum(-1)); nn = np.argsort(D, 1)[:, :k]
        w = 1 / (np.take_along_axis(D, nn, 1) + 1e-9); w /= w.sum(1, keepdims=True)
        out[s["name"]] = (w[:, :, None] * P[nn]).sum(1)
    return out


def v1_run(net, mode, seed, s, off_w=1.0):
    d = DPRO(net, Config(OFF_WEIGHT=off_w), mode, seed)
    X = d.run(s, record=True)
    Xc = np.array([d.snaps[j][1][j] for j in range(len(d.snaps))])
    return X, Xc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--only", default="all")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--tune_odo", default=None); ap.add_argument("--revisit_scores", default=None)
    ap.add_argument("--nets", default="")          # comma list of phase-3 checkpoint dirs
    ap.add_argument("--tseed0", type=int, default=0)   # training index of the first net (to split trainings over processes)
    ap.add_argument("--gyro", default=None)        # gyro_bias.json (phase 4 follow-up)
    args = ap.parse_args(); torch.set_num_threads(args.threads)
    best = json.load(open(args.tune))["best"]
    seqs = load_golden()
    from .revisit import seq_smooth
    rscores = dict(np.load(args.revisit_scores)) if args.revisit_scores else {}
    todo = json.load(open(args.tune_odo)) if args.tune_odo else None
    gyro = json.load(open(args.gyro)) if args.gyro else None
    from .net import RFUpdate
    nets = []
    for d in [x for x in args.nets.split(",") if x]:
        m = RFUpdate(); m.load_state_dict(torch.load(f"{d}/final.pth", weights_only=True)); nets.append(m.eval())
    rows, t0 = [], time.time()
    if os.path.exists(args.out):
        rows = pickle.load(open(args.out, "rb"))
    done = {(r["method"], r["run"], r["seed"], r.get("tseed", 0)) for r in rows}

    def add(method, s, seed, X, Xc, tseed=0, **extra):
        rows.append(dict(method=method, run=s["name"], zone=s["zone"], seed=seed, tseed=tseed, X=X, Xc=Xc,
                         **all_metrics(X, Xc, s["G"], s["t"]), **extra))

    ora = oracle_params(seqs)
    fa = best["flow_abs_W3"]
    RF = {
        "abs, all APs (v1 solver, no patches)": RFConfig(W=1, use_flow=False, use_abs=True, c_abs=float(best["abs_W1"])),
        "radio flow, single scans": RFConfig(W=1, c_flow=float(best["flow_W1"])),
        "radio flow, 3-scan average": RFConfig(W=3, c_flow=float(best["flow_W3"])),
        "radio flow + abs, 3-scan average": RFConfig(W=3, use_abs=True, c_flow=float(fa[0]), c_abs=float(fa[1])),
        "radio flow, 3-scan average + raw revisits": RFConfig(W=3, c_flow=float(best["flow_W3"]), revisits="raw", c_rev=float(best["flow_W3_rev"])),
        "radio flow + abs, 3-scan average + raw revisits (phase 3 start)": RFConfig(W=3, use_abs=True, c_flow=float(fa[0]), c_abs=float(fa[1]), revisits="raw", c_rev=float(best["flow_W3_rev"])),
    }
    ORACLE = {
        "ceiling: abs with true APs": RFConfig(W=1, use_flow=False, use_abs=True, c_abs=float(best["abs_W1"])),
        "ceiling: radio flow with true APs": RFConfig(W=3, c_flow=float(best["flow_W3"])),
    }
    c3, crev = float(best["flow_W3"]), float(best["flow_W3_rev"])
    for s in seqs:
        for seed in range(args.seeds):
            if args.only in ("all", "rev") and rscores:
                m = "radio flow, 3-scan average + learned revisits"
                if (m, s["name"], seed, 0) not in done:
                    sl = RadioFlowSLAM(RFConfig(W=3, c_flow=c3, revisits="scores", c_rev=crev), seed, revisit_score=seq_smooth(rscores[s["name"]], 3))
                    out = sl.run(s); add(m, s, seed, out["X"], out["Xc"], rev=sl.rev_pairs, rev_w=sl.rev_weight)
            if args.only in ("all", "odo") and todo:
                for m, cfg in [("odometry in the solver, no WiFi", RFConfig(W=3, use_flow=False, odometry=True, fixedp=1)),
                               ("odometry + radio flow", RFConfig(W=3, c_flow=float(todo["best_c_flow"]), odometry=True, fixedp=1)),
                               ("odometry + radio flow + raw revisits", RFConfig(W=3, c_flow=float(todo["best_c_flow"]), odometry=True, revisits="raw", c_rev=float(todo["best_c_rev"]), fixedp=1))]:
                    if (m, s["name"], seed, 0) in done: continue
                    out = RadioFlowSLAM(cfg, seed).run(s); add(m, s, seed, out["X"], out["Xc"])
            if args.only in ("all", "net") and nets:
                from dataclasses import replace
                from .train_net import hand_cfg
                base = hand_cfg(best)
                for ti, net in enumerate(nets, start=args.tseed0):
                    for m, cfg in [("learned update (final refinement)", replace(base, learned_iters=12)),
                                   ("learned update (also while streaming)", replace(base, learned_iters=12, learned_stream=2))]:
                        if (m, s["name"], seed, ti) in done: continue
                        sl = RadioFlowSLAM(cfg, seed, net=net); out = sl.run(s)
                        add(m, s, seed, out["X"], out["Xc"], tseed=ti, X_hand=sl.X_hand.numpy())
            if args.only in ("all", "gyro") and gyro and seed == 0:      # deterministic: one seed
                from .gyro_bias import correct, causal
                e, o = gyro["runs"][s["name"]], np.asarray(s["odom"], float)
                m = "odometry + WiFi gyro-bias check"
                if (m, s["name"], 0, 0) not in done:
                    X = correct(o, s["t"], np.deg2rad(e["b_deg_s"])) if e["accepted"] else o
                    add(m, s, 0, X, causal(o, s["t"], s["R"], gyro["calibration"]["tau"]), b_hat=e["b_deg_s"], accepted=e["accepted"])
                m = "ceiling: odometry with the true gyro bias removed"
                if (m, s["name"], 0, 0) not in done:
                    X = correct(o, s["t"], np.deg2rad(e["truth_deg_s"])); add(m, s, 0, X, X)
            if args.only in ("all", "rf"):
                for m, cfg in RF.items():
                    if (m, s["name"], seed, 0) in done: continue
                    sl = RadioFlowSLAM(cfg, seed); out = sl.run(s)
                    add(m, s, seed, out["X"], out["Xc"], rev=getattr(sl, "rev_pairs", None), rev_w=getattr(sl, "rev_weight", None))
                for m, cfg in ORACLE.items():
                    if (m, s["name"], seed, 0) in done: continue
                    out = RadioFlowSLAM(cfg, seed, oracle_th=ora).run(s); add(m, s, seed, out["X"], out["Xc"])
            if args.only in ("all", "v1"):
                for m, mode, w in [("motion rules only", "priors", 1.0), ("v1 network off", "off", 1.0), ("v1 network off, tuned", "off", 0.1)]:
                    if (m, s["name"], seed, 0) in done: continue
                    X, Xc = v1_run(None, mode, seed, s, w); add(m, s, seed, X, Xc)
                for ti, kn in enumerate(["sim", "sim_s1", "sim_s2"]):
                    if ("v1 DPRO", s["name"], seed, ti) in done: continue
                    net = RONet(); net.load_state_dict(torch.load(f"{CK}/{kn}/final.pth", weights_only=True)); net.eval()
                    X, Xc = v1_run(net, "net", seed, s); add("v1 DPRO", s, seed, X, Xc, tseed=ti)
        print(f"{s['name']} done ({time.time() - t0:.0f}s)", flush=True)
        pickle.dump(rows, open(args.out, "wb"))
    if args.only in ("all", "ctx"):
        wk = wknn_p1(seqs)
        for s in seqs:
            add("wheel odometry", s, 0, s["odom"], s["odom"])
            add("WkNN + radio map", s, 0, wk[s["name"]], wk[s["name"]])
    pickle.dump(rows, open(args.out, "wb"))
    print("EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
