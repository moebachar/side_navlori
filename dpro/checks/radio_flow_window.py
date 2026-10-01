"""Does averaging over a short window of scans before differencing reduce the noise of radio flow?"""
import os, sys, json
import numpy as np
from scipy.optimize import least_squares
os.environ.setdefault("SIDE_NAVLORI", "X:/side_navlori"); sys.path.insert(0, "X:/side_navlori")
from dpro.data import load_golden
seqs = load_golden(); L = len(seqs[0]["landmarks"])
h = lambda x, th: th[2] - 10 * th[3] * np.log10(np.hypot(x[..., 0] - th[0], x[..., 1] - th[1]) + 0.5)
G = np.vstack([s["G"] for s in seqs]); R = np.vstack([s["R"] for s in seqs]); TH = {}
for a in range(L):
    m = ~np.isnan(R[:, a])
    if m.sum() < 15: continue
    xy, y = G[m], R[m, a]; i = int(np.argmax(y)); best = None
    for dx, dy in [(0, 0), (3, 0), (-3, 0), (0, 3), (0, -3)]:
        r = least_squares(lambda t: h(xy, t) - y, [xy[i, 0] + dx, xy[i, 1] + dy, min(y[i] + 5, -1), 2.5], loss="soft_l1", f_scale=4.0, bounds=([-10, -10, -80, 1.0], [80, 40, 0, 6.0]))
        if best is None or r.cost < best.cost: best = r
    TH[a] = best.x
out = {}
for W in [1, 3]:
    for k in [2, 4, 6, 8]:
        dy, dh = [], []
        for s in seqs:
            N = len(s["t"]); Gs, Rs = s["G"], s["R"]; hw = W // 2
            for i in range(hw, N - k - hw):
                j = i + k
                for a in TH:
                    wi = Rs[i - hw:i + hw + 1, a]; wj = Rs[j - hw:j + hw + 1, a]
                    if np.isnan(wi).any() or np.isnan(wj).any(): continue
                    gi = np.array([h(Gs[q], TH[a]) for q in range(i - hw, i + hw + 1)]).mean()
                    gj = np.array([h(Gs[q], TH[a]) for q in range(j - hw, j + hw + 1)]).mean()
                    dy.append(wj.mean() - wi.mean()); dh.append(gj - gi)
        dy, dh = np.array(dy), np.array(dh)
        out[f"W{W}_k{k}"] = dict(r=float(np.corrcoef(dy, dh)[0, 1]), res=float((dy - dh).std()), n=int(len(dy)))
        print(f"window {W} scan(s), k={k}: r {out[f'W{W}_k{k}']['r']:.2f}, residual std {out[f'W{W}_k{k}']['res']:.2f} dB, pairs {len(dy)}")
json.dump(out, open(os.environ.get("RF_WINDOW_OUT", "rf_window.json"), "w"))
