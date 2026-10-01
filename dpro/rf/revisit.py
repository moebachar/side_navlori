"""Learned revisit check (phase 2): probability that two scans were taken within 3 m of each other.

The WiFi counterpart of the learned matching features in RAFT / DROID-SLAM: raw fingerprint similarity finds
candidates (16 % within 2 m on the golden runs), a small pairwise network decides which are real.
It never sees AP identities: each pair is the set of APs heard in either scan, described by their signals in
both scans (raw and 3-scan averages), pooled, plus a few whole-pair features. So it can run in a new building.

  python -m dpro.rf.revisit --out /root/navlori/runs/dpro/revisit      # trains synthetic-only, east+syn, west+syn
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn

from .. import sim as dsim
from .graph import windowed, fingerprint_distance

FA, FG = 8, 6           # per-AP and global feature sizes
NEAR = 3.0              # metres: "same place"


def pair_features(Ri, Rj, Yi, Yj, fd):
    """Features of one pair from raw readings Ri, Rj and 3-scan averages Yi, Yj (dBm, nan = not heard)."""
    hi, hj = ~np.isnan(Ri), ~np.isnan(Rj)
    u = np.where(hi | hj)[0]
    nz = lambda v: np.where(np.isnan(v), 0.0, (v + 70.0) / 15.0)
    both = hi[u] & hj[u]
    d = np.where(both, np.abs(np.nan_to_num(Ri[u]) - np.nan_to_num(Rj[u])) / 10.0, 0.0)
    mx = np.where(hi[u] | hj[u], (np.fmax(np.nan_to_num(Ri[u], nan=-100), np.nan_to_num(Rj[u], nan=-100)) + 70) / 15.0, 0.0)
    per = np.stack([nz(Ri[u]), nz(Rj[u]), hi[u].astype(float), hj[u].astype(float), d, mx, nz(Yi[u]), nz(Yj[u])], -1)
    ns, nu = both.sum(), len(u)
    md = d[both].mean() if ns else 3.0
    glob = np.array([ns / 10.0, nu / 10.0, ns / max(nu, 1), md, fd / 10.0, abs(hi.sum() - hj.sum()) / 10.0])
    return per, glob


class PairNet(nn.Module):
    def __init__(self, dim=48):
        super().__init__()
        self.ap = nn.Sequential(nn.Linear(FA, dim), nn.ReLU(), nn.Linear(dim, dim), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2 * dim + FG, dim), nn.ReLU(), nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))

    def forward(self, per, mask, glob):
        f = self.ap(per)                                              # (B, U, dim)
        m = mask[..., None].float()
        mean = (f * m).sum(1) / m.sum(1).clamp(min=1)
        mx = torch.where(m > 0, f, torch.full_like(f, -1e4)).max(1).values
        return self.head(torch.cat([mean, mx, glob], -1))[:, 0]


def batch(items):
    U = max(len(p) for p, _, _ in items)
    per = np.zeros((len(items), U, FA)); mask = np.zeros((len(items), U), bool); glob = np.zeros((len(items), FG)); y = np.zeros(len(items))
    for b, (p, g, lab) in enumerate(items):
        per[b, :len(p)] = p; mask[b, :len(p)] = True; glob[b] = g; y[b] = lab
    return torch.tensor(per, dtype=torch.float32), torch.tensor(mask), torch.tensor(glob, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def pool_pairs(scans, rng, max_pairs=None, min_gap=7):
    """scans: list of (run_id, index, G, R, Y) for one site; pairs across runs, or within a run >= min_gap apart."""
    out = []
    n = len(scans)
    R = np.stack([s[3] for s in scans]); FD = fingerprint_distance(R)
    idx = [(a, b) for a in range(n) for b in range(a + 1, n) if scans[a][0] != scans[b][0] or abs(scans[a][1] - scans[b][1]) >= min_gap]
    if max_pairs and len(idx) > max_pairs:
        # keep every near pair, fill the rest with random far pairs and the most look-alike far pairs
        near = [p for p in idx if np.hypot(*(scans[p[0]][2] - scans[p[1]][2])) < NEAR]
        far = [p for p in idx if np.hypot(*(scans[p[0]][2] - scans[p[1]][2])) >= NEAR]
        far_sorted = sorted(far, key=lambda p: FD[p])
        k = max(0, max_pairs - len(near))
        hard = far_sorted[:k // 2]
        rest = [far[i] for i in rng.choice(len(far), min(len(far), k - len(hard)), replace=False)] if far else []
        idx = near + hard + rest
    for a, b in idx:
        per, glob = pair_features(scans[a][3], scans[b][3], scans[a][4], scans[b][4], FD[a, b])
        out.append((per, glob, float(np.hypot(*(scans[a][2] - scans[b][2])) < NEAR)))
    return out


def synthetic_pairs(n_sites, rng, runs_per_site=3, scans=30, per_site=400):
    items = []
    for _ in range(n_sites):
        site = dsim.make_site(rng); sc = []
        for r in range(runs_per_site):
            run = dsim.make_run(site, rng, scans); Y, _ = windowed(run["R"], 3)
            sc += [(r, i, run["G"][i], run["R"][i], Y[i]) for i in range(scans)]
        items += pool_pairs(sc, rng, per_site)
    return items


def real_pairs(seqs, rng, max_pairs=None):
    sc = []
    for s in seqs:
        Y, _ = windowed(s["R"], 3)
        sc += [(s["name"], i, s["G"][i], s["R"][i], Y[i]) for i in range(len(s["t"]))]
    return pool_pairs(sc, rng, max_pairs)


def train(items, steps=3000, bs=256, lr=1e-3, seed=0, log=None):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = PairNet(); opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    pos = np.mean([it[2] for it in items]); pw = torch.tensor((1 - pos) / max(pos, 1e-3))
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    for st in range(steps):
        b = [items[i] for i in rng.integers(len(items), size=bs)]
        per, mask, glob, y = batch(b)
        loss = lossf(net(per, mask, glob), y)
        opt.zero_grad(); loss.backward(); opt.step()
        if log and st % 500 == 0:
            print(f"  step {st} loss {loss.item():.3f}", flush=True)
    return net.eval()


@torch.no_grad()
def score_run(net, seq, min_gap=7):
    """(N, N) probability that scans i, j are within 3 m (only pairs >= min_gap apart are scored)."""
    R = seq["R"]; Y, _ = windowed(R, 3); FD = fingerprint_distance(R); N = len(R)
    P = np.zeros((N, N)); pairs = [(i, j) for i in range(N) for j in range(i + min_gap, N)]
    for k in range(0, len(pairs), 512):
        chunk = pairs[k:k + 512]
        per, mask, glob, _ = batch([(*pair_features(R[i], R[j], Y[i], Y[j], FD[i, j]), 0.0) for i, j in chunk])
        p = torch.sigmoid(net(per, mask, glob)).numpy()
        for (i, j), v in zip(chunk, p):
            P[i, j] = P[j, i] = v
    return P


def seq_smooth(M, L=3):
    """Sequence matching (SeqSLAM-style): average a pair score over L consecutive pairs, either in the same
    direction (i+o, j+o) or the opposite one (i+o, j-o), and keep the better of the two."""
    N = len(M); best = np.full((N, N), -np.inf)
    for sgn in (1, -1):
        acc = np.zeros((N, N)); cnt = np.zeros((N, N))
        for o in range(-(L // 2), L // 2 + 1):
            for i in range(N):
                ii = i + o
                if ii < 0 or ii >= N:
                    continue
                jj = np.arange(N) + sgn * o
                ok = (jj >= 0) & (jj < N)
                acc[i, ok] += M[ii, jj[ok]]; cnt[i, ok] += 1
        best = np.maximum(best, acc / np.maximum(cnt, 1))
    return best


def main():
    from ..data import load_golden
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--sites", type=int, default=150); ap.add_argument("--steps", type=int, default=3000)
    args = ap.parse_args(); torch.set_num_threads(args.threads); os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0); t0 = time.time()
    syn = synthetic_pairs(args.sites, rng)
    print(f"synthetic pairs {len(syn)} (near {np.mean([i[2] for i in syn]):.2%}) {time.time() - t0:.0f}s", flush=True)
    seqs = load_golden()
    zone = {z: [s for s in seqs if s["zone"] == z] for z in ["east", "west"]}
    real = {z: real_pairs(zone[z], rng, 20000) for z in zone}
    for z in real:
        print(f"real {z} pairs {len(real[z])} (near {np.mean([i[2] for i in real[z]]):.2%})", flush=True)
    for name, items in [("syn", syn), ("syn_east", syn + real["east"] * 3), ("syn_west", syn + real["west"] * 3)]:
        print(f"training {name} on {len(items)} pairs", flush=True)
        net = train(items, args.steps, log=True)
        torch.save(net.state_dict(), f"{args.out}/{name}.pth")
    print(f"REVISIT_TRAIN_DONE {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
