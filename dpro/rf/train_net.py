"""Phase 3 training: the learned update operator on synthetic sites (DROID-SLAM's recipe, radio version).

Each step: a new synthetic site and a 32-scan run; the hand-weighted phase-1 solver gives the starting point
(no gradient); S learned iterations (network + 2 solver steps) with gradients through the solver;
loss = sum_s gamma^(S-1-s) * (pose error after rigid alignment + 0.1 * |target change - noise-free change| / 5).

  python -m dpro.rf.train_net --tune tune.json --out /root/navlori/runs/dpro/rf_net_s0 --seed 0
"""
import argparse, json, os, time
import numpy as np
import torch

from .. import sim as dsim
from ..evaluate import kabsch_loss
from .graph import windowed
from .net import RFUpdate
from .slam import RadioFlowSLAM, RFConfig


def hand_cfg(best, revisits=True):
    """The best hand-weighted solver from phase 1 (radio flow + absolute factors, 3-scan averages, raw revisits)."""
    cf, ca = best["flow_abs_W3"]
    return RFConfig(W=3, use_abs=True, c_flow=float(cf), c_abs=float(ca), revisits="raw" if revisits else "none",
                    c_rev=float(best.get("flow_W3_rev", 0.1)), learned_iters=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--S", type=int, default=6)
    ap.add_argument("--scans", type=int, default=32); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args(); torch.set_num_threads(args.threads); torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    best = json.load(open(args.tune))["best"]
    cfg = hand_cfg(best)
    rng = np.random.default_rng(1000 + args.seed)
    net = RFUpdate(); net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, args.steps, pct_start=0.05, cycle_momentum=False, anneal_strategy="linear")
    hist, t0 = [], time.time()
    for step in range(args.steps):
        run = dsim.make_run(dsim.make_site(rng), rng, args.scans)
        keep = ~np.all(np.isnan(run["R"]), 0)
        run["R"], run["Rtrue"] = run["R"][:, keep], run["Rtrue"][:, keep]
        sl = RadioFlowSLAM(cfg, seed=int(rng.integers(1 << 30)))
        out = sl.run(run)
        f = sl.factor_dict()
        if len(f["pi"]) == 0:
            continue
        Yt, _ = windowed(run["Rtrue"], cfg.W)
        true_dy = torch.tensor(Yt[f["pj"].numpy(), f["pa"].numpy()] - Yt[f["pi"].numpy(), f["pa"].numpy()])
        G = torch.tensor(run["G"])
        X0, TH0 = torch.tensor(out["X"]), torch.tensor(out["TH"])
        N = len(G); free = torch.zeros(N, dtype=torch.bool); free[cfg.fixedp:] = True
        _, _, traj = sl.learned_refine(X0, TH0, net, args.S, free, N, grad=True, f=f)
        loss, last = 0.0, {}
        for s, it in enumerate(traj):
            g = 0.9 ** (len(traj) - 1 - s)
            pose = kabsch_loss(it["X"], G)
            flow = (it["target"] - true_dy).abs().mean() / 5.0
            loss = loss + g * (pose + 0.1 * flow)
            last = dict(pose=float(pose), flow_db=float(flow) * 5)
        start = float(kabsch_loss(X0, G))
        if not torch.isfinite(loss):
            continue
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step(); sched.step()
        hist.append(dict(step=step, loss=float(loss), start=start, **last))
        if step % 50 == 0 or step == args.steps - 1:
            h = hist[-50:]
            print(f"step {step:5d}/{args.steps}  pose {np.mean([x['pose'] for x in h]):.2f} m (hand start {np.mean([x['start'] for x in h]):.2f})  "
                  f"target err {np.mean([x['flow_db'] for x in h]):.2f} dB  {(time.time() - t0) / (step + 1):.2f} s/step", flush=True)
        if (step + 1) % 500 == 0:
            torch.save(net.state_dict(), f"{args.out}/step_{step + 1:05d}.pth")
    torch.save(net.state_dict(), f"{args.out}/final.pth")
    json.dump(hist, open(f"{args.out}/history.json", "w"))
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
