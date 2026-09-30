"""DPRO network: DPVO's net.py (Update, Patchifier, CorrBlock, VONet.forward) for WiFi scans.

What maps to what (DPVO -> DPRO):
  frame (image)                  -> scan (RSSI over the APs heard, with a mask)
  patch (3x3 pixels, one frame)  -> radio patch: one AP over the last K_PATCH scans (causal, so it runs online)
  inverse depth of a patch       -> AP parameters theta_a = (px, py, P0, n), shared by all patches of that AP
  correlation volume             -> edge features: observed vs predicted RSSI window of the AP around scan j, their
                                    residual, fingerprint similarity of scans i and j, time gap, current geometry
  flow correction delta (2D)     -> RSSI correction delta (dB); target = current prediction h + delta
  confidence weight (2D)         -> confidence weight (1D)

Deviations from the brief, both to keep "nothing site-specific is learned": no AP identity embedding, and the scan
feature networks are replaced by a fixed, vocabulary-free fingerprint similarity (mean |dRSSI| over shared APs,
share of shared APs).
"""
import numpy as np
import torch
import torch.nn as nn

from .blocks import GatedResidual, SoftAgg, GradientClip, neighbors
from .radio_ops import predict, norm_rssi, init_ap
from .ba import BA

DIM = 64
K_PATCH = 5          # scans per radio patch (causal window ending at the patch's scan)
WIN = 2              # edge window: scans j-2 .. j+2 around the target scan
CORR_DIM = 5 * (2 * WIN + 1) + 6
DSCALE = 10.0        # dB per unit of the delta head


class Update(nn.Module):
    """DPVO's update operator, DIM 384 -> 64, 2D outputs -> 1D, one extra soft aggregation over edges of the same AP."""
    def __init__(self):
        super().__init__()
        self.c1 = nn.Sequential(nn.Linear(DIM, DIM), nn.ReLU(inplace=True), nn.Linear(DIM, DIM))
        self.c2 = nn.Sequential(nn.Linear(DIM, DIM), nn.ReLU(inplace=True), nn.Linear(DIM, DIM))
        self.norm = nn.LayerNorm(DIM, eps=1e-3)
        self.agg_kk = SoftAgg(DIM)          # edges of the same patch
        self.agg_aa = SoftAgg(DIM)          # edges of the same AP (the landmark shared by several patches)
        self.agg_ij = SoftAgg(DIM)          # edges between the same pair of scans
        self.gru = nn.Sequential(nn.LayerNorm(DIM, eps=1e-3), GatedResidual(DIM),
                                 nn.LayerNorm(DIM, eps=1e-3), GatedResidual(DIM))
        self.corr = nn.Sequential(nn.Linear(CORR_DIM, DIM), nn.ReLU(inplace=True), nn.Linear(DIM, DIM),
                                  nn.LayerNorm(DIM, eps=1e-3), nn.ReLU(inplace=True), nn.Linear(DIM, DIM))
        self.d = nn.Sequential(nn.ReLU(inplace=False), nn.Linear(DIM, 1), GradientClip())
        self.w = nn.Sequential(nn.ReLU(inplace=False), nn.Linear(DIM, 1), GradientClip(), nn.Sigmoid())

    def forward(self, net, inp, corr, ii, jj, kk, aa):
        net = net + inp + self.corr(corr)
        net = self.norm(net)
        ix, jx = neighbors(kk, jj)
        mask_ix = (ix >= 0).float()[:, None]; mask_jx = (jx >= 0).float()[:, None]
        net = net + self.c1(mask_ix * net[ix])
        net = net + self.c2(mask_jx * net[jx])
        net = net + self.agg_kk(net, kk)
        net = net + self.agg_aa(net, aa)
        net = net + self.agg_ij(net, ii * 12345 + jj)
        net = self.gru(net)
        return net, self.d(net)[:, 0], self.w(net)[:, 0]


class Patchifier(nn.Module):
    """Picks M heard APs per scan (random, as DPVO picks random pixels) and encodes each radio patch into a context
    feature (DPVO's inet)."""
    def __init__(self):
        super().__init__()
        self.inet = nn.Sequential(nn.Conv1d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Flatten(), nn.Linear(64 * K_PATCH, DIM))

    @staticmethod
    def select(MK_scan, M, rng, strategy="random", R_scan=None):
        heard = np.where(MK_scan)[0]
        if len(heard) <= M:
            return heard
        if strategy == "strongest":
            return heard[np.argsort(-R_scan[heard])[:M]]
        return rng.choice(heard, M, replace=False)

    def forward(self, RN, MK, ii, aa):
        """RN, MK (N, L) float tensors; ii, aa (P,) long: patch scan and AP -> context features (P, DIM)."""
        offs = torch.arange(-K_PATCH + 1, 1, device=RN.device)
        js = ii[:, None] + offs
        valid = (js >= 0).float(); js = js.clamp(min=0)
        a = aa[:, None].expand_as(js)
        x = torch.stack([RN[js, a] * valid, MK[js, a] * valid, valid], 1)      # (P, 3, K)
        return self.inet(x)


def fingerprint_sim(R):
    """(N, N, 2) vocabulary-free similarity between scans: mean |dRSSI| over shared APs (/10 dB), share of shared APs."""
    MK = ~torch.isnan(R); Rz = torch.nan_to_num(R, 0.0)
    m = MK.double()
    common = m @ m.T
    union = m.sum(1)[:, None] + m.sum(1)[None, :] - common
    mad = (m[:, None, :] * m[None, :, :] * (Rz[:, None, :] - Rz[None, :, :]).abs()).sum(-1) / common.clamp(min=1)
    mad = torch.where(common > 0, mad / 10.0, torch.full_like(mad, 3.0))
    return torch.stack([mad, common / union.clamp(min=1)], -1)


def edge_features(RN, MK, FS, t, X, TH, n, ii, jj, aa):
    """DPVO's correlation lookup, radio version. All geometry inputs are detached current estimates."""
    offs = torch.arange(-WIN, WIN + 1, device=RN.device)
    js = jj[:, None] + offs
    valid = ((js >= 0) & (js < n)).double(); js = js.clamp(0, n - 1)
    a = aa[:, None].expand_as(js)
    obs = RN[js, a].double() * valid
    msk = MK[js, a].double() * valid
    pred = norm_rssi(predict(X[js], TH[a])) * valid
    res = (obs - pred) * msk
    d = torch.sqrt(((X[jj] - TH[aa, :2]) ** 2).sum(-1) + 1e-6)
    extra = torch.stack([FS[ii, jj, 0], FS[ii, jj, 1], (t[jj] - t[ii]) / 20.0, torch.log(d + 0.5),
                         (TH[aa, 2] + 40.0) / 15.0, TH[aa, 3] - 3.0], -1)
    return torch.cat([obs, msk, pred, res, valid, extra], -1).float()


def init_thetas(TH, seen, R, X, scans, rng, offset_sig=2.0):
    """Initialise the APs heard for the first time in `scans` (DPVO's random depth init)."""
    for j in scans:
        for a in np.where(~np.isnan(R[j]))[0]:
            if seen[a]:
                continue
            hj = np.where(~np.isnan(R[:j + 1, a]))[0]
            off = torch.tensor(rng.normal(0, offset_sig, 2))
            TH[a] = init_ap(X[hj], torch.tensor(R[hj, a]), offset=off)
            seen[a] = True
    return TH, seen


class RONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.patchify = Patchifier()
        self.update = Update()
        self.DIM = DIM

    def forward(self, seg, STEPS=18, M=6, R_LIFE=6, N_INIT=8, fixedp=2, structure_only=False, rng=None):
        """DPVO's VONet.forward for training: start with N_INIT scans, add one scan per step after step 8,
        update + 2 BA steps per iteration. seg: dict with t (N,), G (N, 2), R (N, L) dBm with nan = not heard."""
        rng = rng or np.random.default_rng()
        R = seg["R"]; N, L = R.shape
        Rt = torch.tensor(R); MK = (~torch.isnan(Rt)).float(); RN = torch.where(MK > 0, norm_rssi(torch.nan_to_num(Rt, -70.0)), torch.zeros_like(Rt)).float()
        FS = fingerprint_sim(Rt); t = torch.tensor(seg["t"]); G = torch.tensor(seg["G"])

        pi, pa = [], []
        for j in range(N):
            sel = Patchifier.select(MK[j].numpy() > 0, M, rng)
            pi += [j] * len(sel); pa += list(sel)
        ii = torch.tensor(pi, dtype=torch.long); aa = torch.tensor(pa, dtype=torch.long)
        imap = self.patchify(RN, MK, ii, aa)

        X = G.clone()
        if not structure_only:
            X[fixedp:] = G[fixedp - 1]                  # DPVO starts every pose at the identity
        TH = torch.zeros(L, 4, dtype=torch.float64); seen = np.zeros(L, bool)
        n = min(N_INIT, N)
        TH, seen = init_thetas(TH, seen, R, X, range(n), rng)

        kk, jj = _edges(ii, 0, n, 0, n, R_LIFE)
        net = torch.zeros(len(kk), DIM)
        traj = []
        while len(traj) < STEPS:
            X = X.detach(); TH = TH.detach()
            if len(traj) >= 8 and n < N:
                if not structure_only:
                    X[n] = X[n - 1]
                TH, seen = init_thetas(TH, seen, R, X, [n], rng)
                k1, j1 = _edges(ii, max(0, n - R_LIFE), n, n, n + 1, R_LIFE)       # old patches -> new scan
                k2, j2 = _edges(ii, n, n + 1, max(0, n - R_LIFE), n + 1, R_LIFE)   # new patches -> recent scans
                kk = torch.cat([k1, k2, kk]); jj = torch.cat([j1, j2, jj])
                net = torch.cat([torch.zeros(len(k1) + len(k2), DIM), net])
                n += 1
            corr = edge_features(RN, MK, FS, t, X, TH, n, ii[kk], jj, aa[kk])
            net, delta, weight = self.update(net, imap[kk], corr, ii[kk], jj, kk, aa[kk])
            target = predict(X[jj], TH[aa[kk]]) + DSCALE * delta.double()
            free = torch.arange(N) >= fixedp
            X, TH = BA(X, TH, target, weight.double(), jj, aa[kk], free, n, structure_only=structure_only, iters=2, t=t)
            traj.append(dict(X=X[:n], G=G[:n], target=target, weight=weight, jj=jj, aa=aa[kk], n=n))
        return traj


def _edges(ii, p0, p1, s0, s1, R_LIFE):
    """Edges from patches whose scan is in [p0, p1) to scans in [s0, s1), within R_LIFE scans."""
    kk = torch.where((ii >= p0) & (ii < p1))[0]
    if len(kk) == 0:
        return kk, kk.clone()
    js = torch.arange(s0, s1)
    K, J = torch.meshgrid(kk, js, indexing="ij")
    K, J = K.reshape(-1), J.reshape(-1)
    ok = (ii[K] - J).abs() <= R_LIFE
    return K[ok], J[ok]
