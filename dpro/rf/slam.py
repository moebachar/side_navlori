"""Radio Flow SLAM runtime (DROID-SLAM frontend + backend, for WiFi scans).

Streaming: for each scan c (its averaged readings need scan c+1, so estimates lag one scan, ~4 s):
initialise its position with the damped motion model, initialise newly heard APs, add radio-flow links to
the K previous scans (and absolute factors if enabled), and run a few solver steps on the last positions.
After the run, the backend adds revisit links and runs a global solve over all positions.

Weights: "hand" mode uses a fixed noise model (random noise of averaged readings + unshared
place-dependent error), times one multiplier per factor type chosen on synthetic sites.
"net" mode (phase 3) lets a network set targets and weights.
"""
from dataclasses import dataclass, field
import numpy as np
import torch

from ..radio_ops import init_ap, predict
from . import ba as rba
from .graph import windowed, fingerprint_distance, propose_pairs, sigma2_pair, sigma2_single


@dataclass
class RFConfig:
    K: int = 8                    # radio-flow links to the K previous scans
    W: int = 3                    # readings averaged over W scans
    use_flow: bool = True
    use_abs: bool = False
    revisits: str = "none"        # none | raw | scores
    c_flow: float = 0.1           # weight multipliers (chosen on synthetic sites)
    c_abs: float = 0.1
    c_rev: float = 0.1
    sig_n: float = 3.5            # noise model (dB): random part per reading, place-dependent part, its correlation length (m)
    sig_s: float = 5.0
    ell: float = 3.5
    step_m: float = 0.8           # typical distance between consecutive scans
    n_init: int = 8
    opt_window: int = 12
    iters_init: int = 12
    iters_scan: int = 3
    iters_final: int = 12
    fixedp: int = 3
    damping: float = 0.5
    robust_c: float = 2.0         # Cauchy scale on a revisit link's RMS normalised residual
    max_rev: int = 0              # 0 = N // 6
    odometry: bool = False
    odo_sig0: float = 0.05        # m per step
    odo_sig_rel: float = 0.05     # fraction of the step
    learned_iters: int = 12       # phase 3: learned updates after the hand-weighted global solve
    learned_stream: int = 0       # phase 3: learned updates after each new scan (0 = off)


class RadioFlowSLAM:
    def __init__(self, cfg=RFConfig(), seed=0, oracle_th=None, revisit_score=None, net=None):
        self.cfg, self.rng = cfg, np.random.default_rng(seed)
        self.oracle_th = oracle_th              # {ap: (px, py, P0, n)} -> AP parameters fixed (ceiling test)
        self.revisit_score = revisit_score      # (N, N) probability that two scans are within 3 m (phase 2)
        self.net = net

    # ---------------------------------------------------------------- factors
    def _pair(self, i, j, a, Y, C, kind="flow", d=None, score=1.0):
        c = self.cfg
        s2 = sigma2_pair(j - i, C[i, a], C[j, a], c.sig_n, c.sig_s, c.ell, c.step_m, d)
        w = (c.c_flow if kind == "flow" else c.c_rev) / s2
        self.F["pi"].append(i); self.F["pj"].append(j); self.F["pa"].append(a); self.F["pt"].append(Y[j, a] - Y[i, a]); self.F["pw"].append(w)
        self.F["pk"].append(0 if kind == "flow" else 1)
        # extra inputs for the learned update (phase 3)
        self.F["ci"].append(C[i, a]); self.F["cj"].append(C[j, a]); self.F["yi"].append(Y[i, a]); self.F["yj"].append(Y[j, a])
        self.F["sc"].append(score); self.F["fd"].append(self.FD[i, j]); self.F["ns"].append(self.NS[i, j])

    def factor_dict(self):
        """Pair factors with every input of the learned update (tensors)."""
        L = lambda k, dt=torch.float64: torch.tensor(self.F[k], dtype=dt)
        f = dict(pi=L("pi", torch.long), pj=L("pj", torch.long), pa=L("pa", torch.long), dy=L("pt"), w_hand=L("pw"),
                 ci=L("ci"), cj=L("cj"), yi=L("yi"), yj=L("yj"), kind=L("pk"), score=L("sc"), fd=L("fd"), ns=L("ns"))
        f["k"] = (f["pj"] - f["pi"]).double().clamp(max=12.0)
        ws = getattr(self, "wscale_final", None)
        if ws is not None and len(ws) == len(f["w_hand"]):     # start from phase 1's robust revisit weights
            f["w_hand"] = f["w_hand"] * ws
        return f

    def learned_refine(self, X, TH, net, iters, free, n, grad=False, f=None, state=None):
        """Iterate: network on every pair factor -> corrected targets and weights -> 2 solver steps."""
        from .net import factor_features, DIM
        f = f if f is not None else self.factor_dict()
        if len(f["pi"]) == 0:
            return X, TH, []
        state = state if state is not None else torch.zeros(len(f["pi"]), DIM)
        prev = torch.zeros(len(f["pi"]), dtype=torch.float64)
        traj = []
        with torch.set_grad_enabled(grad):
            for _ in range(iters):
                X, TH = X.detach(), TH.detach()
                feats, dh = factor_features(X, TH, f, prev)
                state, delta, z = net(state, feats, f)
                target = f["dy"] + 10.0 * delta.double()
                w = f["w_hand"] * torch.exp(z.double())
                fac = dict(pi=f["pi"], pj=f["pj"], pa=f["pa"], pt=target, pw=w)
                if len(self.F["sj"]):                      # absolute factors stay hand-weighted
                    fac.update(sj=torch.tensor(self.F["sj"]), sa=torch.tensor(self.F["sa"]),
                               st=torch.tensor(self.F["st"], dtype=torch.float64), sw=torch.tensor(self.F["sw"], dtype=torch.float64))
                X, TH = rba.solve(X, TH, fac, free, n, t=self.t, odo=self.odo, sig_odo=self.sig_odo, iters=2,
                                  speed_prior=not self.cfg.odometry, smooth_prior=not self.cfg.odometry)
                hi = predict(X[f["pi"]], TH[f["pa"]]); hj = predict(X[f["pj"]], TH[f["pa"]])
                prev = (target - (hj - hi)).detach()
                traj.append(dict(X=X, TH=TH, target=target, w=w))
        return X, TH, traj

    def _single(self, j, a, Y, C):
        c = self.cfg
        self.F["sj"].append(j); self.F["sa"].append(a); self.F["st"].append(Y[j, a]); self.F["sw"].append(c.c_abs / sigma2_single(C[j, a], c.sig_n, c.sig_s))

    def _tensors(self, wscale=None):
        T = {}
        for k in ["pi", "pj", "pa", "sj", "sa"]:
            T[k] = torch.tensor(self.F[k], dtype=torch.long)
        for k in ["pt", "pw", "st", "sw"]:
            T[k] = torch.tensor(self.F[k], dtype=torch.float64)
        if wscale is not None:
            T["pw"] = T["pw"] * wscale
        return T

    def _solve(self, X, TH, free, n, iters, wscale=None):
        if len(self.F["pi"]) + len(self.F["sj"]) == 0:
            return X, TH
        T = self._tensors(wscale)
        return rba.solve(X, TH, T, free, n, t=self.t, odo=self.odo, sig_odo=self.sig_odo,
                         fix_ap=self.oracle_th is not None, iters=iters,
                         speed_prior=not self.cfg.odometry, smooth_prior=not self.cfg.odometry)

    # ---------------------------------------------------------------- run
    @torch.no_grad()
    def run(self, seq):
        c = self.cfg
        R, G = seq["R"], seq["G"]
        N, L = R.shape
        self.t = torch.tensor(seq["t"])
        Y, C = windowed(R, c.W)
        self.Y, self.C = Y, C
        self.FD = fingerprint_distance(R)
        hY = (~np.isnan(Y)).astype(float); self.NS = hY @ hY.T
        self.F = {k: [] for k in ["pi", "pj", "pa", "pt", "pw", "pk", "sj", "sa", "st", "sw", "ci", "cj", "yi", "yj", "sc", "fd", "ns"]}
        X = torch.zeros(N, 2, dtype=torch.float64)
        TH = torch.zeros(L, 4, dtype=torch.float64); seen = np.zeros(L, bool)
        usable = np.ones(L, bool)
        if self.oracle_th is not None:
            usable[:] = False
            for a, th in self.oracle_th.items():
                TH[a] = torch.tensor(th); seen[a] = True; usable[a] = True
        self.odo = self.sig_odo = None
        if c.odometry:
            o = np.asarray(seq["odom"], float)
            self.odo = torch.tensor(o)
            steps = np.hypot(*np.diff(o, axis=0).T)
            self.sig_odo = torch.tensor(c.odo_sig0 + c.odo_sig_rel * steps)
        Xc = np.zeros((N, 2))
        for j in range(N):
            if j < c.fixedp:
                X[j] = torch.tensor(G[j])
            elif j >= 2:
                if c.odometry:
                    X[j] = X[j - 1] + (self.odo[j] - self.odo[j - 1])
                else:
                    fac = float((self.t[j] - self.t[j - 1]) / (self.t[j - 1] - self.t[j - 2]).clamp(min=1e-3))
                    X[j] = X[j - 1] + c.damping * fac * (X[j - 1] - X[j - 2])
            else:
                X[j] = X[j - 1]
            for a in np.where(~np.isnan(Y[j]) & usable & ~seen)[0]:
                hj = np.where(~np.isnan(Y[:j + 1, a]))[0]
                TH[a] = init_ap(X[hj], torch.tensor(Y[hj, a]), offset=torch.tensor(self.rng.normal(0, 2.0, 2)))
                seen[a] = True
            heard = np.where(~np.isnan(Y[j]) & usable)[0]
            if c.use_flow:
                for k in range(1, c.K + 1):
                    i = j - k
                    if i < 0:
                        break
                    for a in heard:
                        if not np.isnan(Y[i, a]):
                            self._pair(i, j, a, Y, C)
            if c.use_abs:
                for a in heard:
                    self._single(j, a, Y, C)
            n = j + 1
            free = torch.zeros(N, dtype=torch.bool)
            if n == c.n_init:
                free[c.fixedp:n] = True
                X, TH = self._solve(X, TH, free, n, c.iters_init)
            elif n > c.n_init:
                free[max(c.fixedp, n - c.opt_window):n] = True
                X, TH = self._solve(X, TH, free, n, c.iters_scan)
                if self.net is not None and c.learned_stream > 0:
                    X, TH, _ = self.learned_refine(X, TH, self.net, c.learned_stream, free, n)
            Xc[j] = X[j].numpy()
        if N < c.n_init:
            free = torch.zeros(N, dtype=torch.bool); free[c.fixedp:N] = True
            X, TH = self._solve(X, TH, free, N, c.iters_init)
            Xc[:] = X.numpy()

        # ---- backend: revisit links + global solve ----
        rev_pairs = []
        if c.revisits != "none":
            maxp = c.max_rev or max(2, N // 6)
            if c.revisits == "raw":
                rev_pairs = propose_pairs(fingerprint_distance(R), max_pairs=maxp)
            else:
                rev_pairs = propose_pairs(self.revisit_score, max_pairs=maxp, higher_is_better=True)
            start = len(self.F["pi"])
            edge_of = []
            for e, (i, jj, sc) in enumerate(rev_pairs):
                for a in np.where(~np.isnan(Y[i]) & ~np.isnan(Y[jj]) & usable & seen)[0]:
                    self._pair(i, jj, a, Y, C, kind="rev", d=1.0, score=(sc if c.revisits == "scores" else 1.0))
                    if c.revisits == "scores":
                        self.F["pw"][-1] *= sc
                    edge_of.append(e)
            edge_of = np.array(edge_of, int)
        free = torch.zeros(N, dtype=torch.bool); free[c.fixedp:N] = True
        wscale = None
        for it in range(c.iters_final):
            X, TH = self._solve(X, TH, free, N, 1, wscale)
            if rev_pairs and len(edge_of):
                r = rba.residuals(X, TH, self._tensors())["pair"][start:].numpy()
                rms = np.sqrt(np.bincount(edge_of, r ** 2, minlength=len(rev_pairs)) / np.maximum(np.bincount(edge_of, minlength=len(rev_pairs)), 1))
                s_edge = 1.0 / (1.0 + (rms / c.robust_c) ** 2)
                ws = np.ones(len(self.F["pi"])); ws[start:] = s_edge[edge_of]
                wscale = torch.tensor(ws)
        self.rev_pairs = rev_pairs
        self.rev_weight = (wscale.numpy()[start:] if (rev_pairs and wscale is not None) else None)
        self.wscale_final = wscale
        self.X_hand = X.clone()
        if self.net is not None and c.learned_iters > 0:
            X, TH, _ = self.learned_refine(X, TH, self.net, c.learned_iters, free, N)
        self.TH, self.seen = TH, seen
        return dict(X=X.detach().numpy(), Xc=Xc, TH=TH.detach().numpy(), seen=seen)
