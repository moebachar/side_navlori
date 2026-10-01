"""Three real-data checks behind the radio-flow proposal (golden runs, CPU).
1. Radio flow: does the RSSI change between two scans of a run follow the change predicted by the
   log-distance model (oracle AP parameters fitted with ground truth)? By baseline k = j - i.
2. Displacement from radio flow alone (best case: AP parameters known): estimate the step x_j - x_i.
3. Revisits: DROID-style edge proposal from the fingerprint distance matrix; precision vs ground truth.
Writes rf_checks.json for the proposal page.
"""
import os, sys, json
import numpy as np
from scipy.optimize import least_squares
os.environ.setdefault("SIDE_NAVLORI", "X:/side_navlori")
sys.path.insert(0, "X:/side_navlori")
from dpro.data import load_golden

OUT = os.environ.get("RF_OUT", "rf_checks.json")
seqs = load_golden()
L = len(seqs[0]["landmarks"])
EPS = 0.5
h = lambda x, th: th[2] - 10 * th[3] * np.log10(np.hypot(x[..., 0] - th[0], x[..., 1] - th[1]) + EPS)

# oracle AP parameters (fitted with ground-truth positions, all runs pooled)
G = np.vstack([s["G"] for s in seqs]); R = np.vstack([s["R"] for s in seqs])
TH = {}
for a in range(L):
    m = ~np.isnan(R[:, a])
    if m.sum() < 15: continue
    xy, y = G[m], R[m, a]; i = int(np.argmax(y)); best = None
    for dx, dy in [(0, 0), (3, 0), (-3, 0), (0, 3), (0, -3)]:
        r = least_squares(lambda t: h(xy, t) - y, [xy[i, 0] + dx, xy[i, 1] + dy, min(y[i] + 5, -1), 2.5],
                          loss="soft_l1", f_scale=4.0, bounds=([-10, -10, -80, 1.0], [80, 40, 0, 6.0]))
        if best is None or r.cost < best.cost: best = r
    TH[a] = best.x
abs_res = np.concatenate([R[~np.isnan(R[:, a]), a] - h(G[~np.isnan(R[:, a])], TH[a]) for a in TH])
print(f"oracle fit: {len(TH)} APs, absolute residual std {abs_res.std():.2f} dB")

# 1 + 2: radio flow by baseline
KS = [1, 2, 3, 4, 6, 8, 12]
flow, disp = {}, {}
rng = np.random.default_rng(0)
scatter = {}
for k in KS:
    dy, dh, dd = [], [], []
    errs, steps, zero = [], [], []
    for s in seqs:
        N = len(s["t"]); Gs, Rs = s["G"], s["R"]
        for i in range(N - k):
            j = i + k
            aps = [a for a in TH if not np.isnan(Rs[i, a]) and not np.isnan(Rs[j, a])]
            if len(aps) < 3: continue
            yi = np.array([Rs[i, a] for a in aps]); yj = np.array([Rs[j, a] for a in aps])
            hi = np.array([h(Gs[i], TH[a]) for a in aps]); hj = np.array([h(Gs[j], TH[a]) for a in aps])
            di = np.array([np.hypot(*(Gs[i] - TH[a][:2])) for a in aps]); dj = np.array([np.hypot(*(Gs[j] - TH[a][:2])) for a in aps])
            dy += list(yj - yi); dh += list(hj - hi); dd += list(dj - di)
            # displacement from radio flow, x_i known, AP parameters known (oracle best case)
            meas = yj - yi
            f = lambda d: np.array([h(Gs[i] + d, TH[a]) for a in aps]) - hi - meas
            r = least_squares(f, np.zeros(2), loss="soft_l1", f_scale=4.0)
            true = Gs[j] - Gs[i]
            errs.append(np.hypot(*(r.x - true))); steps.append(np.hypot(*true))
    dy, dh, dd = map(np.array, (dy, dh, dd))
    res = dy - dh
    flow[k] = dict(n=int(len(dy)), r=float(np.corrcoef(dy, dh)[0, 1]), res_std=float(res.std()),
                   dy_std=float(dy.std()), dh_std=float(dh.std()), med_abs_dd=float(np.median(np.abs(dd))))
    disp[k] = dict(n=len(errs), err_med=float(np.median(errs)), step_med=float(np.median(steps)),
                   frac_better_than_zero=float(np.mean(np.array(errs) < np.array(steps))))
    idx = rng.choice(len(dy), min(700, len(dy)), replace=False)
    scatter[k] = dict(dh=[round(float(v), 1) for v in dh[idx]], dy=[round(float(v), 1) for v in dy[idx]])
    print(f"k={k:2d}: pairs {len(dy):5d} | r(dy, dh) {flow[k]['r']:.2f} | residual std {res.std():.2f} dB "
          f"(dy std {dy.std():.2f}, dh std {dh.std():.2f}) | step median {disp[k]['step_med']:.2f} m -> "
          f"radio-flow estimate error {disp[k]['err_med']:.2f} m (better than 'no move' on {disp[k]['frac_better_than_zero']:.0%})")

# 3: revisit proposals (DROID add_proximity_factors, with fingerprint distance instead of flow)
def fp_dist(Rs):
    M = np.where(np.isnan(Rs), -100.0, Rs)
    return np.sqrt(((M[:, None, :] - M[None, :, :]) ** 2).mean(-1))
rev = []
run7 = None
for s in seqs:
    N = len(s["t"]); D = fp_dist(s["R"]); Gs = s["G"]
    GD = np.hypot(Gs[:, None, 0] - Gs[None, :, 0], Gs[:, None, 1] - Gs[None, :, 1])
    cand = [(D[i, j], i, j) for i in range(N) for j in range(i + 7, N)]          # not temporal neighbours
    cand.sort()
    taken, sup = [], set()
    for d, i, j in cand:
        if (i, j) in sup: continue
        taken.append((i, j, float(d), float(GD[i, j])))
        for di in range(-2, 3):
            for dj in range(-2, 3):
                sup.add((i + di, j + dj))
        if len(taken) >= max(2, N // 6): break
    for i, j, d, g in taken:
        rev.append(dict(run=s["name"], i=i, j=j, fd=round(d, 2), gd=round(g, 2)))
    if s["name"] == "golden_run_7":
        run7 = dict(G=[[round(float(v), 2) for v in p] for p in Gs], D=[[round(float(v), 1) for v in r] for r in D],
                    GD=[[round(float(v), 1) for v in r] for r in GD], edges=[dict(i=i, j=j, fd=round(d, 2), gd=round(g, 2)) for i, j, d, g in taken])
gd = np.array([r["gd"] for r in rev])
print(f"revisit proposals: {len(rev)} pairs over 12 runs | true distance < 2 m: {np.mean(gd < 2):.0%}, < 3 m: {np.mean(gd < 3):.0%}, "
      f"median {np.median(gd):.1f} m")
r7 = np.array([e["gd"] for e in run7["edges"]])
print(f"run 7: {len(r7)} proposals, < 2 m {np.mean(r7 < 2):.0%}, < 3 m {np.mean(r7 < 3):.0%}")
# for context: distance between scans that are 7+ apart in time, at random
allgd = np.concatenate([np.hypot(*(s["G"][i] - s["G"][j])) * np.ones(1) for s in seqs for i in range(len(s["t"])) for j in range(i + 7, len(s["t"]))])
print(f"random pairs 7+ scans apart: < 2 m {np.mean(allgd < 2):.0%}, < 3 m {np.mean(allgd < 3):.0%}, median {np.median(allgd):.1f} m")

json.dump(dict(abs_res_std=float(abs_res.std()), n_aps=len(TH), flow=flow, disp=disp, scatter=scatter,
               revisits=dict(n=len(rev), lt2=float(np.mean(gd < 2)), lt3=float(np.mean(gd < 3)), med=float(np.median(gd)),
                             rand_lt2=float(np.mean(allgd < 2)), rand_lt3=float(np.mean(allgd < 3)), rand_med=float(np.median(allgd))),
               run7=run7), open(OUT, "w"))
print("written", OUT)
