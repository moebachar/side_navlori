"""Synthetic WiFi sites and robot runs for DPRO pretraining (DPVO pretrains on synthetic TartanAir).

A site = a grid of corridors, wall lines that cost a few dB each, APs with random log-distance parameters and a
static shadowing field per AP. A run = a random walk along the corridors with a slow scan every ~4 s; each AP is
read at a random instant of the 3.5 s scan window (motion smear), with per-reading noise, a detection threshold and
random drop-outs. The ranges below are set so the synthetic data matches what the golden runs show
(fitted path-loss exponent ~4.5, ~6 dB model residual, ~half the APs heard per scan, ~0.7 m smear).
"""
import numpy as np


def make_site(rng):
    sp = rng.uniform(6, 12); nx = int(rng.integers(3, 7)); ny = int(rng.integers(2, 4))
    nodes = np.array([(i * sp, j * sp) for j in range(ny) for i in range(nx)], float)
    nodes += rng.normal(0, 0.6, nodes.shape)
    idx = lambda i, j: j * nx + i
    edges = [(idx(i, j), idx(i + 1, j)) for j in range(ny) for i in range(nx - 1)] + \
            [(idx(i, j), idx(i, j + 1)) for j in range(ny - 1) for i in range(nx)]
    keep = [e for e in edges if rng.random() > 0.2] or edges
    adj = {k: [] for k in range(len(nodes))}
    for a, b in keep:
        adj[a].append(b); adj[b].append(a)
    lo, hi = nodes.min(0) - 5, nodes.max(0) + 5
    walls = [np.cumsum(rng.uniform(3, 8, 40)) + lo[d] - 2 for d in range(2)]
    walls = [w[w < hi[d] + 2] for d, w in enumerate(walls)]
    L = int(rng.integers(12, 30))
    ap = rng.uniform(lo, hi, (L, 2))
    K = 16
    ell = rng.uniform(2, 5, L)
    ang = rng.uniform(0, 2 * np.pi, (L, K)); mag = rng.uniform(0.5, 1.5, (L, K)) * (2 * np.pi / ell[:, None])
    return dict(nodes=nodes, adj=adj, walls=walls, wall_db=rng.uniform(3, 8), ap=ap,
                P0=rng.uniform(-42, -28, L), n=rng.uniform(2.0, 3.2, L),
                kvec=np.stack([mag * np.cos(ang), mag * np.sin(ang)], -1), phase=rng.uniform(0, 2 * np.pi, (L, K)),
                shadow=rng.uniform(3, 7.5, L), noise=rng.uniform(2.5, 4.5), thr=rng.uniform(-88, -82), pdrop=rng.uniform(0.05, 0.25))


def field(site, P):
    """Mean RSSI (dBm) of every AP at positions P (T, 2) -> (T, L): log-distance + wall losses + static shadowing."""
    ap = site["ap"]
    d = np.sqrt(((P[:, None, :] - ap[None]) ** 2).sum(-1))
    base = site["P0"][None] - 10 * site["n"][None] * np.log10(d + 0.5)
    nw = 0
    for dim in range(2):
        w = site["walls"][dim]
        a = np.minimum(P[:, None, dim], ap[None, :, dim]); b = np.maximum(P[:, None, dim], ap[None, :, dim])
        nw = nw + ((w[None, None, :] > a[..., None]) & (w[None, None, :] < b[..., None])).sum(-1)
    arg = np.einsum("tc,lkc->tlk", P, site["kvec"]) + site["phase"][None]
    sh = site["shadow"][None] * np.sqrt(2.0 / arg.shape[-1]) * np.cos(arg).sum(-1)
    return base - site["wall_db"] * nw + sh


def make_run(site, rng, n_scans=15, dt=0.1):
    nodes, adj = site["nodes"], site["adj"]
    v = rng.uniform(0.12, 0.3); T = rng.uniform(3.8, 4.8); D = 3.5
    dur = 5.0 + n_scans * T + 5.0
    starts = [k for k in adj if adj[k]]                                   # nodes with at least one corridor
    cur = int(starts[int(rng.integers(len(starts)))]); prev = -1
    pts = [nodes[cur]] * int(5.0 / dt)                                  # 5 s still at the start
    lat = rng.uniform(-0.3, 0.3)
    while len(pts) * dt < dur:
        nb = [k for k in adj[cur] if k != prev] or adj[cur]
        nxt = nb[int(rng.integers(len(nb)))]
        a, b = nodes[cur], nodes[nxt]; seg = np.linalg.norm(b - a)
        u = (b - a) / seg; nrm = np.array([-u[1], u[0]])
        for s in np.arange(0, seg, v * dt):
            lat = np.clip(lat + rng.normal(0, 0.02), -0.4, 0.4)
            pts.append(a + u * s + nrm * lat)
        prev, cur = cur, nxt
    P = np.array(pts); tt = np.arange(len(P)) * dt
    ts = 2.0 + np.cumsum(np.r_[0, rng.uniform(T - 0.3, T + 0.3, n_scans - 1)])
    L = len(site["ap"])
    t_read = ts[:, None] + rng.uniform(0, D, (n_scans, L))                  # each AP read at its own instant
    Pr = np.stack([np.interp(t_read.ravel(), tt, P[:, 0]), np.interp(t_read.ravel(), tt, P[:, 1])], -1)
    F = field(site, Pr).reshape(n_scans, L, L)[:, np.arange(L), np.arange(L)]
    R = F + rng.normal(0, site["noise"], F.shape)
    R[(R < site["thr"]) | (rng.random(R.shape) < site["pdrop"])] = np.nan
    R = np.round(R)
    tm = ts + D / 2
    G = np.stack([np.interp(tm, tt, P[:, 0]), np.interp(tm, tt, P[:, 1])], -1)
    return dict(name="sim", t=tm - tm[0], G=G, R=R, Rtrue=field(site, G))


def sample_segment(rng, length=15):
    """One training clip from a fresh synthetic site; drops APs never heard in the clip."""
    for _ in range(20):
        site = make_site(rng)
        run = make_run(site, rng, length)
        keep = ~np.all(np.isnan(run["R"]), 0)
        if keep.sum() >= 4 and (~np.isnan(run["R"])).sum(1).min() >= 2:
            run["R"], run["Rtrue"] = run["R"][:, keep], run["Rtrue"][:, keep]
            return run
    return run
