"""Radio-flow bundle adjustment: DROID-SLAM's dense BA with frame pairs -> scan pairs and depths -> AP parameters.

Factors (each with a weight w, in 1/dB^2):
  pair   (i, j, a): target change dy  vs  h(x_j, th_a) - h(x_i, th_a)    (radio flow; P0 cancels)
  single (j, a):    target value  y   vs  h(x_j, th_a)                    (absolute RSSI; P0 estimated)
Priors: smoothness and speed on consecutive positions (from DPRO v1), optional wheel-odometry steps,
weak priors on each AP's P0 and fade exponent n.
Gauss-Newton with the AP block (4x4 per AP) eliminated by a Schur complement, as in DPVO / DROID-SLAM.
Every operation is torch, so gradients flow from the positions back to the targets and weights (phase 3).
"""
import torch
from ..radio_ops import predict_jac
from ..ba import CholeskySolver

SIG_ACC, V0, SIG_V = 0.3, 0.2, 0.1
N0, SIG_N = 3.0, 1.0
P00, SIG_P0 = -40.0, 15.0
N_MIN, N_MAX, P0_MIN, P0_MAX = 1.5, 6.0, -90.0, -10.0


def solve(X, TH, fac, free, n, t=None, odo=None, sig_odo=None, fix_ap=False, iters=2, ep=0.1, lmbda=1e-3,
          speed_prior=True, smooth_prior=True):
    """With wheel odometry the motion priors are switched off by the caller (odometry supersedes them)."""
    for _ in range(iters):
        X, TH = _step(X, TH, fac, free, n, t, odo, sig_odo, fix_ap, ep, lmbda, speed_prior, smooth_prior)
    return X, TH


def residuals(X, TH, fac):
    """Normalised residuals sqrt(w) * (target - prediction) of the pair and single factors."""
    out = {}
    if fac.get("pi") is not None and len(fac["pi"]):
        hi = predict_jac(X[fac["pi"]], TH[fac["pa"]])[0]; hj = predict_jac(X[fac["pj"]], TH[fac["pa"]])[0]
        out["pair"] = (fac["pt"] - (hj - hi)) * fac["pw"].sqrt()
    if fac.get("sj") is not None and len(fac["sj"]):
        out["single"] = (fac["st"] - predict_jac(X[fac["sj"]], TH[fac["sa"]])[0]) * fac["sw"].sqrt()
    return out


def _step(X, TH, fac, free, n, t, odo, sig_odo, fix_ap, ep, lmbda, speed_prior, smooth_prior=True):
    dev, dt = X.device, X.dtype
    fidx = torch.where(free[:n])[0]
    nf = len(fidx)
    pos = torch.full((X.shape[0],), -1, device=dev, dtype=torch.long); pos[fidx] = torch.arange(nf, device=dev)

    # ---- stack factors: two position slots per factor (slot 1 unused for single factors) ----
    P, JX, JT, R, Wt, A = [], [], [], [], [], []
    if fac.get("pi") is not None and len(fac["pi"]):
        hi, Jxi, Jti = predict_jac(X[fac["pi"]], TH[fac["pa"]])
        hj, Jxj, Jtj = predict_jac(X[fac["pj"]], TH[fac["pa"]])
        P.append(torch.stack([fac["pi"], fac["pj"]], 1)); JX.append(torch.stack([-Jxi, Jxj], 1))
        JT.append(Jtj - Jti); R.append(fac["pt"] - (hj - hi)); Wt.append(fac["pw"]); A.append(fac["pa"])
    if fac.get("sj") is not None and len(fac["sj"]):
        h1, Jx1, Jt1 = predict_jac(X[fac["sj"]], TH[fac["sa"]])
        P.append(torch.stack([fac["sj"], fac["sj"]], 1)); JX.append(torch.stack([Jx1, torch.zeros_like(Jx1)], 1))
        JT.append(Jt1); R.append(fac["st"] - h1); Wt.append(fac["sw"]); A.append(fac["sa"])
    npair = len(fac["pi"]) if fac.get("pi") is not None else 0
    P = torch.cat(P); JX = torch.cat(JX); JT = torch.cat(JT); R = torch.cat(R); Wt = torch.cat(Wt); A = torch.cat(A)
    PP = pos[P].clone()
    PP[npair:, 1] = -1                                   # single factors have one position

    ax, ak = torch.unique(A, return_inverse=True)
    m = len(ax)
    C = torch.zeros(m, 4, 4, device=dev, dtype=dt).index_add(0, ak, Wt[:, None, None] * JT[:, :, None] * JT[:, None, :])
    u = torch.zeros(m, 4, device=dev, dtype=dt).index_add(0, ak, (Wt * R)[:, None] * JT)
    th = TH[ax]
    C = C + torch.diag_embed(torch.tensor([0.0, 0.0, 1 / SIG_P0 ** 2, 1 / SIG_N ** 2], device=dev, dtype=dt)).expand(m, 4, 4)
    u = u + torch.stack([torch.zeros_like(th[:, 0]), torch.zeros_like(th[:, 0]), (P00 - th[:, 2]) / SIG_P0 ** 2, (N0 - th[:, 3]) / SIG_N ** 2], -1)
    C = C + lmbda * torch.eye(4, device=dev, dtype=dt) + lmbda * C * torch.eye(4, device=dev, dtype=dt)

    if nf == 0:
        if fix_ap:
            return X, TH
        return X, _retr_th(TH, ax, (torch.linalg.inv(C) @ u[..., None])[..., 0])

    # ---- position blocks B (nf x nf of 2x2), coupling E (nf x m of 2x4), gradient v ----
    Bf = torch.zeros(nf * nf, 2, 2, device=dev, dtype=dt)
    Ef = torch.zeros(nf * m, 2, 4, device=dev, dtype=dt)
    v = torch.zeros(nf, 2, device=dev, dtype=dt)
    for s in range(2):
        ps = PP[:, s]; ok = ps >= 0
        Js = JX[:, s]
        v = v.index_add(0, ps[ok], ((Wt * R)[:, None] * Js)[ok])
        Ef = Ef.index_add(0, ps[ok] * m + ak[ok], (Wt[:, None, None] * Js[:, :, None] * JT[:, None, :])[ok])
        for s2 in range(2):
            p2 = PP[:, s2]; ok2 = ok & (p2 >= 0)
            Bf = Bf.index_add(0, ps[ok2] * nf + p2[ok2], (Wt[:, None, None] * Js[:, :, None] * JX[:, s2][:, None, :])[ok2])
    B = Bf.reshape(nf, nf, 2, 2).permute(0, 2, 1, 3).reshape(2 * nf, 2 * nf)
    E = Ef.reshape(nf, m, 2, 4).permute(0, 2, 1, 3).reshape(2 * nf, 4 * m)
    v = v.reshape(-1)

    # ---- motion priors (linear smoothness, nonlinear speed) and wheel odometry ----
    B, v = _priors(X, B, v, pos, fidx, n, t, nf, speed_prior, smooth_prior)
    if odo is not None:
        B, v = _odometry(X, B, v, pos, fidx, n, odo, sig_odo, nf)

    if fix_ap:
        S = B + ep * torch.eye(2 * nf, device=dev, dtype=dt) + 1e-4 * torch.diag_embed(torch.diagonal(B))
        dX = CholeskySolver.apply(S, v[:, None])[:, 0]
        return X.index_add(0, fidx, dX.reshape(nf, 2)), TH

    Q = torch.linalg.inv(C)
    Qb = torch.block_diag(*Q) if m > 1 else Q[0]
    EQ = E @ Qb
    S = B - EQ @ E.T
    y = v - EQ @ u.reshape(-1)
    S = S + ep * torch.eye(2 * nf, device=dev, dtype=dt) + 1e-4 * torch.diag_embed(torch.diagonal(S))
    dX = CholeskySolver.apply(S, y[:, None])[:, 0]
    dth = (Qb @ (u.reshape(-1) - E.T @ dX)).reshape(m, 4)
    return X.index_add(0, fidx, dX.reshape(nf, 2)), _retr_th(TH, ax, dth)


def _add_block(B, pa, pb, blk, nf):
    Bv = B.reshape(nf, 2, nf, 2).permute(0, 2, 1, 3).reshape(nf * nf, 2, 2).index_add(0, pa * nf + pb, blk)
    return Bv.reshape(nf, nf, 2, 2).permute(0, 2, 1, 3).reshape(2 * nf, 2 * nf)


def _priors(X, B, v, pos, fidx, n, t, nf, speed_prior, smooth_prior=True):
    dev, dt = X.device, X.dtype
    lo = max(int(fidx.min()) - 1, 1)
    trip = torch.arange(lo, n - 1, device=dev)
    if smooth_prior and len(trip):
        acc = (X[trip + 1] - 2 * X[trip] + X[trip - 1]) / SIG_ACC
        coef = torch.tensor([1.0, -2.0, 1.0], device=dev, dtype=dt) / SIG_ACC
        vv = v.reshape(nf, 2)
        for s, cs in zip((-1, 0, 1), coef):
            ps = pos[trip + s]; ok = ps >= 0
            vv = vv.index_add(0, ps[ok], -cs * acc[ok])
            for s2, cs2 in zip((-1, 0, 1), coef):
                p2 = pos[trip + s2]; ok2 = ok & (p2 >= 0)
                blk = (cs * cs2) * torch.eye(2, device=dev, dtype=dt).expand(int(ok2.sum()), 2, 2)
                B = _add_block(B, ps[ok2], p2[ok2], blk, nf)
        v = vv.reshape(-1)
    if speed_prior and t is not None:
        pr = torch.arange(max(int(fidx.min()) - 1, 0), n - 1, device=dev)
        if len(pr):
            dtj = (t[pr + 1] - t[pr]).clamp(min=0.5).to(dt)
            diff = X[pr + 1] - X[pr]; ln = torch.sqrt((diff ** 2).sum(-1) + 1e-6); uv = diff / ln[:, None]
            rs = (ln - V0 * dtj) / (SIG_V * dtj)
            Ju = uv / (SIG_V * dtj)[:, None]
            vv = v.reshape(nf, 2)
            for sa, ca in ((1, 1.0), (0, -1.0)):
                pa = pos[pr + sa]; oka = pa >= 0
                vv = vv.index_add(0, pa[oka], -ca * (Ju * rs[:, None])[oka])
                for sb, cb in ((1, 1.0), (0, -1.0)):
                    pb = pos[pr + sb]; ok = oka & (pb >= 0)
                    B = _add_block(B, pa[ok], pb[ok], ca * cb * Ju[ok][:, :, None] * Ju[ok][:, None, :], nf)
            v = vv.reshape(-1)
    return B, v


def _odometry(X, B, v, pos, fidx, n, odo, sig_odo, nf):
    """r = (o_{j+1} - o_j) - (x_{j+1} - x_j), per axis, sigma per step (odo, sig_odo: (N,2), (N-1,))."""
    dev, dt = X.device, X.dtype
    pr = torch.arange(max(int(fidx.min()) - 1, 0), n - 1, device=dev)
    if not len(pr):
        return B, v
    r = (odo[pr + 1] - odo[pr]) - (X[pr + 1] - X[pr])
    w = 1.0 / sig_odo[pr] ** 2
    vv = v.reshape(nf, 2)
    for sa, ca in ((1, 1.0), (0, -1.0)):
        pa = pos[pr + sa]; oka = pa >= 0
        vv = vv.index_add(0, pa[oka], (ca * w[:, None] * r)[oka])
        for sb, cb in ((1, 1.0), (0, -1.0)):
            pb = pos[pr + sb]; ok = oka & (pb >= 0)
            blk = (ca * cb) * w[ok][:, None, None] * torch.eye(2, device=dev, dtype=dt)
            B = _add_block(B, pa[ok], pb[ok], blk, nf)
    return B, vv.reshape(-1)


def _retr_th(TH, ax, dth):
    th = TH[ax] + dth
    th = torch.stack([th[:, 0], th[:, 1], th[:, 2].clamp(P0_MIN, P0_MAX), th[:, 3].clamp(N_MIN, N_MAX)], -1)
    return TH.index_copy(0, ax, th)
