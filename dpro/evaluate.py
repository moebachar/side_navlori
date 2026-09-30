"""Trajectory metrics: absolute trajectory error after 2D alignment (rigid, reflection allowed, optional scale)."""
import numpy as np
import torch


def umeyama(P, G, scale=False, reflect=True):
    """Best R, t (and s) mapping P onto G. Reflection allowed: RSSI cannot tell a map from its mirror image."""
    mp, mg = P.mean(0), G.mean(0)
    A, B = P - mp, G - mg
    U, S, Vt = np.linalg.svd(A.T @ B)
    R = (U @ Vt).T
    if not reflect and np.linalg.det(R) < 0:
        U[:, -1] *= -1; R = (U @ Vt).T; S = S.copy(); S[-1] *= -1
    s = S.sum() / (A ** 2).sum() if scale else 1.0
    return s, R, mg - s * mp @ R.T


def ate(P, G, scale=False, reflect=True):
    """Per-scan position error (m) after alignment."""
    s, R, t = umeyama(P, G, scale, reflect)
    return np.hypot(*(s * P @ R.T + t - G).T)


def aligned(P, G, scale=False, reflect=True):
    s, R, t = umeyama(P, G, scale, reflect)
    return s * P @ R.T + t


def kabsch_loss(P, G):
    """Training loss (torch): mean error after a rigid alignment computed on detached positions (DPVO detaches its
    Umeyama scale the same way). P, G (n, 2) float64 tensors."""
    s, R, t = umeyama(P.detach().numpy(), G.numpy())
    R = torch.tensor(R); t = torch.tensor(t)
    return torch.sqrt(((P @ R.T + t - G) ** 2).sum(-1) + 1e-9).mean()
