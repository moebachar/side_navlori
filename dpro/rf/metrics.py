"""Trajectory metrics for the radio-flow experiments.

aligned : median error after the best rigid alignment of the whole run (reflection allowed), as in the literature.
online  : the causal estimates (each scan as estimated when it was processed), aligned using only the first
          30 s of the run, then no further alignment; median error over the rest of the run.
drift5  : error of the displacement over every stretch of 5 m travelled (rotation from the whole-run fit).
"""
import numpy as np
from ..evaluate import umeyama


def aligned(X, G):
    s, R, t = umeyama(X, G)
    return float(np.median(np.hypot(*(X @ R.T + t - G).T)))


def online(Xc, G, t, t_align=30.0, min_pts=4):
    idx = np.where(t <= t_align)[0]
    if len(idx) < min_pts:
        idx = np.arange(min(min_pts, len(t)))
    s, R, tr = umeyama(Xc[idx], G[idx])
    rest = np.where(t > t_align)[0]
    if len(rest) == 0:
        rest = np.arange(len(t))
    return float(np.median(np.hypot(*(Xc[rest] @ R.T + tr - G[rest]).T)))


def drift5(X, G, L=5.0):
    s, R, tr = umeyama(X, G)
    path = np.r_[0, np.cumsum(np.hypot(*np.diff(G, axis=0).T))]
    errs = []
    for i in range(len(G)):
        j = np.searchsorted(path, path[i] + L)
        if j >= len(G):
            break
        errs.append(np.hypot(*((X[j] - X[i]) @ R.T - (G[j] - G[i]))))
    return float(np.median(errs)) if errs else float("nan")


def all_metrics(X, Xc, G, t):
    return dict(aligned=aligned(X, G), online=online(Xc, G, t), drift5=drift5(X, G))
