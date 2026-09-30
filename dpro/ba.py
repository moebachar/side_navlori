"""Radio bundle adjustment: DPVO's ba.py with poses -> 2D scan positions and inverse depths -> AP parameters.

Minimises   sum_e w_e ((target_e - h(x_j(e), theta_a(e))) / SIG)^2            (edges: patch k of AP a -> scan j)
          + sum_j ||x_{j+1} - 2 x_j + x_{j-1}||^2 / SIG_ACC^2                   (smoothness: RSSI alone overfits)
          + weak priors on (P0, n) of every AP
with Gauss-Newton steps. As in DPVO, the structure block (here 4x4 per AP instead of 1x1 per patch) is eliminated
with a Schur complement and the reduced system over positions is solved with a Cholesky solver that returns zero
instead of crashing. Positions outside the optimisation window (and the gauge scans) stay fixed.
Everything is differentiable, so a loss on the positions trains the targets and weights that the network outputs.
"""
import torch
from .radio_ops import predict_jac

SIG = 5.0            # dB: residual unit, a weight of 1 means "trust this edge to ~5 dB"
SIG_ACC = 0.3        # m: second difference of consecutive scan positions (from the feasibility test)
N0, SIG_N = 3.0, 1.0            # path-loss exponent prior
P00, SIG_P0 = -40.0, 15.0       # reference power prior (dBm at 1 m)
N_MIN, N_MAX, P0_MIN, P0_MAX = 1.5, 6.0, -90.0, -10.0


class CholeskySolver(torch.autograd.Function):
    """DPVO's solver: don't crash training if the decomposition fails."""
    @staticmethod
    def forward(ctx, H, b):
        U, info = torch.linalg.cholesky_ex(H)
        if torch.any(info):
            ctx.failed = True
            return torch.zeros_like(b)
        xs = torch.cholesky_solve(b, U)
        ctx.save_for_backward(U, xs)
        ctx.failed = False
        return xs

    @staticmethod
    def backward(ctx, grad_x):
        if ctx.failed:
            return None, None
        U, xs = ctx.saved_tensors
        dz = torch.cholesky_solve(grad_x, U)
        dH = -torch.matmul(xs, dz.transpose(-1, -2))
        return dH, dz


def BA(X, TH, target, weight, jj, aa, free, n, ep=0.1, lmbda=1e-3, structure_only=False, iters=1, t=None):
    """One or more Gauss-Newton steps.
    X (N, 2) scan positions, TH (L, 4) AP parameters, target/weight (E,), jj (E,) scan of each edge,
    aa (E,) AP of each edge, free (N,) bool: positions the solver may move, n: number of scans in the buffer,
    t (N,) scan times in s: when given (and SPEED_PRIOR is on), adds the speed prior that fixes the scale."""
    for _ in range(iters):
        X, TH = _step(X, TH, target, weight, jj, aa, free, n, ep, lmbda, structure_only, t)
    return X, TH


SPEED_PRIOR = True
V0, SIG_V = 0.2, 0.1     # m/s: typical speed of a ground robot. RSSI alone cannot observe scale (a log-distance
                         # model absorbs any scale into P0), so without this the trajectory is free to shrink.


def _step(X, TH, target, weight, jj, aa, free, n, ep, lmbda, structure_only, t=None):
    dev, dt = X.device, X.dtype
    h, Jx, Jt = predict_jac(X[jj], TH[aa])
    r = (target - h) / SIG
    Jx, Jt = Jx / SIG, Jt / SIG
    w = weight

    # structure (AP) block: only APs that have edges
    ax, ak = torch.unique(aa, return_inverse=True)
    m = len(ax)
    C = torch.zeros(m, 4, 4, device=dev, dtype=dt).index_add(0, ak, w[:, None, None] * Jt[:, :, None] * Jt[:, None, :])
    u = torch.zeros(m, 4, device=dev, dtype=dt).index_add(0, ak, (w * r)[:, None] * Jt)
    # weak priors on P0 and n, plus Levenberg damping (DPVO adds lmbda to its depth block)
    th = TH[ax]
    C = C + torch.diag_embed(torch.tensor([0.0, 0.0, 1 / SIG_P0 ** 2, 1 / SIG_N ** 2], device=dev, dtype=dt)).expand(m, 4, 4)
    u = u + torch.stack([torch.zeros_like(th[:, 0]), torch.zeros_like(th[:, 0]),
                         (P00 - th[:, 2]) / SIG_P0 ** 2, (N0 - th[:, 3]) / SIG_N ** 2], -1)
    C = C + lmbda * torch.eye(4, device=dev, dtype=dt) + lmbda * C * torch.eye(4, device=dev, dtype=dt)
    Q = torch.linalg.inv(C)

    fidx = torch.where(free[:n])[0]
    nf = 0 if structure_only else len(fidx)
    if nf == 0:
        dth = (Q @ u[..., None])[..., 0]
        return X, _retr_th(TH, ax, dth)

    pos = torch.full((X.shape[0],), -1, device=dev, dtype=torch.long); pos[fidx] = torch.arange(nf, device=dev)
    pe = pos[jj]; v_e = pe >= 0
    # pose block B (2nf x 2nf), coupling E (2nf x 4m), gradient v
    B = torch.zeros(nf, 2, 2, device=dev, dtype=dt).index_add(0, pe[v_e], (w[:, None, None] * Jx[:, :, None] * Jx[:, None, :])[v_e])
    B = torch.block_diag(*B) if nf > 1 else B[0]
    v = torch.zeros(nf, 2, device=dev, dtype=dt).index_add(0, pe[v_e], ((w * r)[:, None] * Jx)[v_e]).reshape(-1)
    Eb = (w[:, None, None] * Jx[:, :, None] * Jt[:, None, :])[v_e]            # (e, 2, 4)
    E = torch.zeros(nf * m, 2, 4, device=dev, dtype=dt).index_add(0, pe[v_e] * m + ak[v_e], Eb)
    E = E.reshape(nf, m, 2, 4).permute(0, 2, 1, 3).reshape(2 * nf, 4 * m)

    # smoothness prior on consecutive positions (linear): r = (x_{j+1} - 2 x_j + x_{j-1}) / SIG_ACC
    lo = max(int(fidx.min()) - 1, 1)
    trip = torch.arange(lo, n - 1, device=dev)
    if len(trip):
        acc = (X[trip + 1] - 2 * X[trip] + X[trip - 1]) / SIG_ACC                    # (T, 2)
        coef = torch.tensor([1.0, -2.0, 1.0], device=dev, dtype=dt) / SIG_ACC
        Bp = torch.zeros(nf, nf, device=dev, dtype=dt); vp = torch.zeros(nf, 2, device=dev, dtype=dt)
        for s, cs in zip((-1, 0, 1), coef):
            ps = pos[trip + s]; ok = ps >= 0
            vp = vp.index_add(0, ps[ok], -cs * acc[ok])
            for s2, cs2 in zip((-1, 0, 1), coef):
                ps2 = pos[trip + s2]; ok2 = ok & (ps2 >= 0)
                Bp = Bp.index_put((ps[ok2], ps2[ok2]), cs * cs2 * torch.ones(int(ok2.sum()), device=dev, dtype=dt), accumulate=True)
        B = B + torch.kron(Bp, torch.eye(2, device=dev, dtype=dt))
        v = v + vp.reshape(-1)

    # speed prior on consecutive scans (the scale gauge): r = (||x_{j+1} - x_j|| - V0 dt) / (SIG_V dt)
    if SPEED_PRIOR and t is not None:
        pr = torch.arange(max(int(fidx.min()) - 1, 0), n - 1, device=dev)
        if len(pr):
            dtj = (t[pr + 1] - t[pr]).clamp(min=0.5).to(dt)
            diff = X[pr + 1] - X[pr]; ln = torch.sqrt((diff ** 2).sum(-1) + 1e-6); uv = diff / ln[:, None]
            rs = (ln - V0 * dtj) / (SIG_V * dtj)
            Ju = uv / (SIG_V * dtj)[:, None]                                        # d r / d x_{j+1}; -Ju for x_j
            for (sa, ca) in ((1, 1.0), (0, -1.0)):
                pa = pos[pr + sa]; oka = pa >= 0
                v = v.reshape(nf, 2).index_add(0, pa[oka], -ca * (Ju * rs[:, None])[oka]).reshape(-1)
                for (sb, cb) in ((1, 1.0), (0, -1.0)):
                    pb = pos[pr + sb]; ok = oka & (pb >= 0)
                    blk = ca * cb * Ju[ok][:, :, None] * Ju[ok][:, None, :]
                    Bv = B.reshape(nf, 2, nf, 2).permute(0, 2, 1, 3).reshape(nf * nf, 2, 2)
                    Bv = Bv.index_add(0, pa[ok] * nf + pb[ok], blk)
                    B = Bv.reshape(nf, nf, 2, 2).permute(0, 2, 1, 3).reshape(2 * nf, 2 * nf)

    # Schur complement: eliminate the AP parameters (DPVO eliminates the patch depths)
    Qb = torch.block_diag(*Q) if m > 1 else Q[0]
    EQ = E @ Qb
    S = B - EQ @ E.T
    y = v - EQ @ u.reshape(-1)
    S = S + ep * torch.eye(2 * nf, device=dev, dtype=dt) + 1e-4 * torch.diag_embed(torch.diagonal(S))   # DPVO block_solve damping
    dX = CholeskySolver.apply(S, y[:, None])[:, 0]
    dth = (Qb @ (u.reshape(-1) - E.T @ dX)).reshape(m, 4)

    X = X.index_add(0, fidx, dX.reshape(nf, 2))
    return X, _retr_th(TH, ax, dth)


def _retr_th(TH, ax, dth):
    th = TH[ax] + dth
    th = torch.stack([th[:, 0], th[:, 1], th[:, 2].clamp(P0_MIN, P0_MAX), th[:, 3].clamp(N_MIN, N_MAX)], -1)
    return TH.index_copy(0, ax, th)
