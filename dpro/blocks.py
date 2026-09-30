"""DPVO's blocks.py (GatedResidual, SoftAgg, GradientClip) with the torch_scatter calls replaced by plain torch,
plus DPVO's fastba.neighbors (temporal neighbours of an edge along its patch), written in torch."""
import torch
import torch.nn as nn


class GatedResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.res = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))

    def forward(self, x):
        return x + self.gate(x) * self.res(x)


def scatter_softmax(g, jx, G):
    gmax = torch.full((G, g.shape[1]), -1e9, device=g.device, dtype=g.dtype)
    gmax = gmax.scatter_reduce(0, jx[:, None].expand_as(g), g, reduce="amax", include_self=True)
    e = torch.exp(g - gmax[jx])
    s = torch.zeros(G, g.shape[1], device=g.device, dtype=g.dtype).index_add(0, jx, e)
    return e / s[jx]


class SoftAgg(nn.Module):
    """Soft aggregation over the edges that share a group id (same patch, same AP, same scan pair)."""
    def __init__(self, dim):
        super().__init__()
        self.f = nn.Linear(dim, dim)
        self.g = nn.Linear(dim, dim)
        self.h = nn.Linear(dim, dim)

    def forward(self, x, ix):
        _, jx = torch.unique(ix, return_inverse=True)
        G = int(jx.max()) + 1
        w = scatter_softmax(self.g(x), jx, G)
        y = torch.zeros(G, x.shape[1], device=x.device, dtype=x.dtype).index_add(0, jx, self.f(x) * w)
        return self.h(y)[jx]


class GradClip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_x):
        grad_x = torch.where(torch.isnan(grad_x), torch.zeros_like(grad_x), grad_x)
        return grad_x.clamp(min=-0.01, max=0.01)


class GradientClip(nn.Module):
    def forward(self, x):
        return GradClip.apply(x)


def neighbors(kk, jj):
    """For every edge, the edge of the same patch with the previous / next scan (-1 if none)."""
    order = torch.argsort(kk * 1_000_000 + jj)
    ks = kk[order]
    same = ks[1:] == ks[:-1]
    ix = torch.full_like(kk, -1); jx = torch.full_like(kk, -1)
    ix[order[1:][same]] = order[:-1][same]
    jx[order[:-1][same]] = order[1:][same]
    return ix, jx
