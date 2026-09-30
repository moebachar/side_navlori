"""DPRO training: DPVO's train.py. Synthetic pretraining (the first steps structure-only, positions fixed to the
ground truth, as DPVO fixes poses for its first 1k steps), then a mix of synthetic and real clips.

Losses after every update-BA iteration, later iterations weighted more (gamma):
  pose loss  mean position error after rigid alignment (DPVO: relative pose error after scale alignment)
  RSSI loss  |target - noise-free RSSI| on synthetic clips (DPVO's flow loss in radio form)

  python -m dpro.train --out /root/navlori/runs/dpro/sim --sim_steps 3000
  python -m dpro.train --out /root/navlori/runs/dpro/east --ckpt .../sim/final.pth --real_steps 800 --real_zone east
"""
import argparse, json, os, time
import numpy as np
import torch

from .net import RONet
from .sim import sample_segment
from .data import load_golden, segments
from .evaluate import kabsch_loss
from .ba import SIG


def loss_fn(traj, seg, so, pose_w=1.0, rssi_w=0.5, gamma=0.9):
    S = len(traj); loss = 0.0; last = {}
    Rtrue = torch.tensor(seg["Rtrue"]) if "Rtrue" in seg else None
    for i, s in enumerate(traj):
        g = gamma ** (S - 1 - i); li = 0.0
        if Rtrue is not None:
            e = (s["target"] - Rtrue[s["jj"], s["aa"]]).abs() / SIG
            li = li + rssi_w * e.mean(); last["rssi_db"] = float(e.mean()) * SIG
        if not so and i >= 2:
            pe = kabsch_loss(s["X"], s["G"])
            li = li + pose_w * pe; last["pose_m"] = float(pe)
        loss = loss + g * li
    return loss, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt")
    ap.add_argument("--sim_steps", type=int, default=0)
    ap.add_argument("--so_steps", type=int, default=300)
    ap.add_argument("--real_steps", type=int, default=0)
    ap.add_argument("--real_zone", default="east")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--steps_per_clip", type=int, default=18)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    net = RONet()
    if args.ckpt:
        net.load_state_dict(torch.load(args.ckpt, weights_only=True))
    net.train()
    total = args.sim_steps + args.real_steps
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total, pct_start=0.02, cycle_momentum=False, anneal_strategy="linear")

    pool = []
    if args.real_steps:
        seqs = [s for s in load_golden() if s["zone"] == args.real_zone]
        pool = [c for s in seqs for c in segments(s)]
        print(f"real clips from the {args.real_zone} zone: {len(pool)}", flush=True)

    log, t0, hist = [], time.time(), []
    for step in range(total):
        real = step >= args.sim_steps and rng.random() < 0.5
        seg = pool[int(rng.integers(len(pool)))] if real else sample_segment(rng)
        so = (not args.ckpt) and step < args.so_steps
        traj = net(seg, STEPS=args.steps_per_clip, structure_only=so, rng=rng)
        loss, last = loss_fn(traj, seg, so)
        if not torch.is_tensor(loss) or not torch.isfinite(loss):
            continue
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step(); sched.step()
        hist.append(dict(step=step, real=bool(real), so=bool(so), loss=float(loss), **last))
        if step % 25 == 0 or step == total - 1:
            recent = [h for h in hist[-25:] if "pose_m" in h]
            pm = np.mean([h["pose_m"] for h in recent]) if recent else float("nan")
            rd = np.mean([h["rssi_db"] for h in hist[-25:] if "rssi_db" in h] or [float("nan")])
            print(f"step {step:5d}/{total}  loss {np.mean([h['loss'] for h in hist[-25:]]):.3f}  pose {pm:.2f} m  "
                  f"rssi {rd:.2f} dB  {'SO ' if so else ''}{(time.time() - t0) / (step + 1):.2f} s/step", flush=True)
        if (step + 1) % 500 == 0:
            torch.save(net.state_dict(), f"{args.out}/step_{step + 1:06d}.pth")
    torch.save(net.state_dict(), f"{args.out}/final.pth")
    json.dump(hist, open(f"{args.out}/history.json", "w"))
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
