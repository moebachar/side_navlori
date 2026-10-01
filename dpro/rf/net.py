"""Phase 3: learned update operator for Radio Flow SLAM (DROID-SLAM's update operator, per radio-flow factor).

One hidden state per factor (scan pair i-j, AP a). Each update mixes information over the factors of the same
scan pair (DROID's global pooling over an image), of the same AP (the shared structure, as DPVO's patch
aggregation) and of the same destination scan, then predicts
    a correction delta (dB) to the target change, and a log-multiplier z on the hand weight.
Unlike DROID-SLAM, the target is anchored on the measurement (target = measured change + delta) and both heads
start at zero, so an untrained network reproduces the hand-weighted phase-1 solver exactly.
"""
import numpy as np
import torch
import torch.nn as nn

from ..blocks import GatedResidual, SoftAgg, GradientClip
from ..radio_ops import predict

DIM = 64
F_IN = 17


def factor_features(X, TH, f, prev_resid):
    """Per-factor inputs. f: dict of tensors pi, pj, pa, dy (measured change), ci, cj (readings averaged),
    yi, yj (averaged values), k (scans apart), kind (0 temporal, 1 revisit), score, fd (fingerprint distance), ns (shared APs)."""
    hi = predict(X[f["pi"]], TH[f["pa"]]); hj = predict(X[f["pj"]], TH[f["pa"]])
    dh = hj - hi
    di = torch.sqrt(((X[f["pi"]] - TH[f["pa"], :2]) ** 2).sum(-1) + 1e-6)
    dj = torch.sqrt(((X[f["pj"]] - TH[f["pa"], :2]) ** 2).sum(-1) + 1e-6)
    feats = torch.stack([
        f["dy"] / 10, (f["yi"] + 70) / 15, (f["yj"] + 70) / 15, 1 / f["ci"], 1 / f["cj"],
        dh / 10, (f["dy"] - dh) / 10, prev_resid / 10, torch.log(di + 0.5), torch.log(dj + 0.5),
        (TH[f["pa"], 2] + 40) / 15, TH[f["pa"], 3] - 3, f["k"] / 8, f["kind"], f["score"], f["fd"] / 10, f["ns"] / 10], -1)
    return feats.float(), dh


class RFUpdate(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(F_IN, DIM), nn.ReLU(inplace=True), nn.Linear(DIM, DIM))
        self.norm = nn.LayerNorm(DIM, eps=1e-3)
        self.agg_edge = SoftAgg(DIM)
        self.agg_ap = SoftAgg(DIM)
        self.agg_scan = SoftAgg(DIM)
        self.gru = nn.Sequential(nn.LayerNorm(DIM, eps=1e-3), GatedResidual(DIM), nn.LayerNorm(DIM, eps=1e-3), GatedResidual(DIM))
        self.d = nn.Sequential(nn.ReLU(inplace=False), nn.Linear(DIM, 1), GradientClip())
        self.w = nn.Sequential(nn.ReLU(inplace=False), nn.Linear(DIM, 1), GradientClip())
        for head in (self.d, self.w):
            nn.init.zeros_(head[1].weight); nn.init.zeros_(head[1].bias)

    def forward(self, net, feats, f):
        net = self.norm(net + self.inp(feats))
        edge = f["pi"] * 100000 + f["pj"]
        net = net + self.agg_edge(net, edge)
        net = net + self.agg_ap(net, f["pa"])
        net = net + self.agg_scan(net, f["pj"])
        net = self.gru(net)
        return net, self.d(net)[:, 0], self.w(net)[:, 0].clamp(-5.0, 2.0)
