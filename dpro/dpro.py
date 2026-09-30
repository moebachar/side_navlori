"""DPRO runtime: DPVO's dpvo.py (streaming, sliding window, inactive edges, damped motion model) for WiFi scans.

Per new scan: initialise its position with the damped linear motion model, cut M radio patches, add forward edges
(recent patches -> new scan) and backward edges (new patches -> recent scans), run the update operator + radio BA.
After N_INIT scans: 12 updates (DPVO's initialisation); then UPDATES_PER_SCAN per scan. Edges whose patch is older
than REMOVAL_WINDOW leave the network but stay in the BA with their last target and weight (DPVO's inactive
edges); positions older than OPT_WINDOW are frozen. terminate(): 12 final updates over the whole trajectory.

mode="net": targets and weights from the network.  mode="off": the same solver with the network off
(target = observed RSSI, weight 1 where heard, 0 where not), the key comparison of the brief.
mode="priors": every RSSI weight 0, so only the motion model and the priors act (what the solver does with no radio).
The first FIXEDP positions are the gauge (RSSI cannot fix where the map sits, its rotation or scale).
"""
from dataclasses import dataclass
import numpy as np
import torch

from .net import Patchifier, edge_features, fingerprint_sim, init_thetas, _edges, DIM
from .radio_ops import predict, norm_rssi
from .ba import BA


@dataclass
class Config:
    M: int = 6                   # patches per scan (DPVO: 80 per frame)
    PATCH_LIFETIME: int = 6      # scans (DPVO: 12 frames)
    OPT_WINDOW: int = 12
    REMOVAL_WINDOW: int = 20
    N_INIT: int = 8
    INIT_UPDATES: int = 12
    UPDATES_PER_SCAN: int = 2
    FINAL_UPDATES: int = 12
    MOTION_DAMPING: float = 0.5
    FIXEDP: int = 3
    STRATEGY: str = "random"     # patch selection: random (DPVO) or strongest
    OFF_WEIGHT: float = 1.0      # mode="off": constant weight of every heard reading (1 = the brief's equal weights)


class DPRO:
    def __init__(self, network=None, cfg=Config(), mode="net", seed=0):
        self.net_ = network; self.cfg = cfg; self.mode = mode
        self.rng = np.random.default_rng(seed)
        if network is not None:
            network.eval()

    @torch.no_grad()
    def run(self, seq, record=False):
        """Stream the scans of `seq` (t, R, and G for the gauge scans only) and return the final positions (N, 2).
        record=True keeps a snapshot after every scan in self.snaps: (n, positions so far, estimated AP positions)."""
        c = self.cfg
        self.snaps = []
        R = seq["R"]; N, L = R.shape
        Rt = torch.tensor(R); self.MK = (~torch.isnan(Rt)).float()
        self.RN = torch.where(self.MK > 0, norm_rssi(torch.nan_to_num(Rt, -70.0)), torch.zeros_like(Rt)).float()
        self.Robs = torch.nan_to_num(Rt, -100.0).double()
        self.FS = fingerprint_sim(Rt); self.t = torch.tensor(seq["t"]); G = torch.tensor(seq["G"])
        self.R = R
        self.X = torch.zeros(N, 2, dtype=torch.float64)
        self.TH = torch.zeros(L, 4, dtype=torch.float64); self.seen = np.zeros(L, bool)
        self.ii = torch.zeros(0, dtype=torch.long); self.aa = torch.zeros(0, dtype=torch.long)
        self.imap = torch.zeros(0, DIM)
        self.kk = torch.zeros(0, dtype=torch.long); self.jj = torch.zeros(0, dtype=torch.long)
        self.netstate = torch.zeros(0, DIM)
        self.target = torch.zeros(0, dtype=torch.float64); self.weight = torch.zeros(0, dtype=torch.float64)
        self.inac = dict(kk=[], jj=[], target=[], weight=[])
        self.n = 0; self.initialized = False

        for j in range(N):
            # position init: gauge scans from GT, else damped linear motion model (DPVO's DAMPED_LINEAR)
            if j < c.FIXEDP:
                self.X[j] = G[j]
            elif j >= 2:
                fac = float((self.t[j] - self.t[j - 1]) / (self.t[j - 1] - self.t[j - 2]).clamp(min=1e-3))
                self.X[j] = self.X[j - 1] + c.MOTION_DAMPING * fac * (self.X[j - 1] - self.X[j - 2])
            else:
                self.X[j] = self.X[j - 1]
            # radio patches of the new scan (context from the causal window)
            sel = Patchifier.select(self.MK[j].numpy() > 0, c.M, self.rng, c.STRATEGY, R[j])
            ii_new = torch.full((len(sel),), j, dtype=torch.long); aa_new = torch.tensor(sel, dtype=torch.long)
            if self.net_ is not None and len(sel):
                self.imap = torch.cat([self.imap, self.net_.patchify(self.RN, self.MK, ii_new, aa_new)])
            else:
                self.imap = torch.cat([self.imap, torch.zeros(len(sel), DIM)])
            self.ii = torch.cat([self.ii, ii_new]); self.aa = torch.cat([self.aa, aa_new])
            self.TH, self.seen = init_thetas(self.TH, self.seen, R, self.X, [j], self.rng)
            self.n = j + 1
            # forward and backward edges (DPVO's __edges_forw / __edges_back)
            k1, j1 = _edges(self.ii, max(0, j - c.PATCH_LIFETIME), j, j, j + 1, c.PATCH_LIFETIME)
            k2, j2 = _edges(self.ii, j, j + 1, max(0, j - c.PATCH_LIFETIME), j + 1, c.PATCH_LIFETIME)
            self._append(torch.cat([k1, k2]), torch.cat([j1, j2]))

            if self.n == c.N_INIT and not self.initialized:
                self.initialized = True
                for _ in range(c.INIT_UPDATES):
                    self.update()
            elif self.initialized:
                for _ in range(c.UPDATES_PER_SCAN):
                    self.update()
                self._retire_old()
            if record:
                self.snaps.append((self.n, self.X[:self.n].numpy().copy(), self.TH[torch.tensor(self.seen)][:, :2].numpy().copy()))
        if not self.initialized:
            self.initialized = True
        for _ in range(c.FINAL_UPDATES):
            self.update(global_ba=True)
        return self.X.numpy().copy()

    def _append(self, kk, jj):
        self.kk = torch.cat([self.kk, kk]); self.jj = torch.cat([self.jj, jj])
        self.netstate = torch.cat([self.netstate, torch.zeros(len(kk), DIM)])
        self.target = torch.cat([self.target, torch.zeros(len(kk), dtype=torch.float64)])
        self.weight = torch.cat([self.weight, torch.zeros(len(kk), dtype=torch.float64)])

    def _retire_old(self):
        old = self.ii[self.kk] < self.n - self.cfg.REMOVAL_WINDOW
        if old.any():
            for k, v in (("kk", self.kk), ("jj", self.jj), ("target", self.target), ("weight", self.weight)):
                self.inac[k].append(v[old])
            keep = ~old
            self.kk, self.jj = self.kk[keep], self.jj[keep]
            self.netstate, self.target, self.weight = self.netstate[keep], self.target[keep], self.weight[keep]

    def update(self, global_ba=False):
        c = self.cfg; n = self.n
        if len(self.kk) == 0:
            return
        ii, aa = self.ii[self.kk], self.aa[self.kk]
        if self.mode == "net":
            corr = edge_features(self.RN, self.MK, self.FS, self.t, self.X, self.TH, n, ii, self.jj, aa)
            self.netstate, delta, weight = self.net_.update(self.netstate, self.imap[self.kk], corr, ii, self.jj, self.kk, aa)
            self.target = predict(self.X[self.jj], self.TH[aa]) + 10.0 * delta.double()
            self.weight = weight.double()
        elif self.mode == "off":
            heard = self.MK[self.jj, aa] > 0
            self.target = torch.where(heard, self.Robs[self.jj, aa], predict(self.X[self.jj], self.TH[aa]))
            self.weight = heard.double() * c.OFF_WEIGHT
        else:                                          # "priors": no RSSI at all, only the motion priors act
            self.target = predict(self.X[self.jj], self.TH[aa])
            self.weight = torch.zeros(len(self.jj), dtype=torch.float64)
        kk, jj, tg, wt = self.kk, self.jj, self.target, self.weight
        if self.inac["kk"]:
            kk = torch.cat(self.inac["kk"] + [kk]); jj = torch.cat(self.inac["jj"] + [jj])
            tg = torch.cat(self.inac["target"] + [tg]); wt = torch.cat(self.inac["weight"] + [wt])
        t0 = c.FIXEDP if (global_ba or not self.initialized) else max(c.FIXEDP, n - c.OPT_WINDOW)
        free = torch.zeros(self.X.shape[0], dtype=torch.bool); free[t0:n] = True
        self.X, self.TH = BA(self.X, self.TH, tg, wt, jj, self.aa[kk], free, n, iters=2, t=self.t)
