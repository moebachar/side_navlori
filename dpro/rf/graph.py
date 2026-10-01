"""Scan graph for radio flow: averaged readings, links between scans, revisit proposals.

DROID-SLAM builds its frame graph from neighbouring frames plus long-range pairs taken from an all-pairs
distance matrix (most similar first, near-duplicates suppressed). Here the frames are WiFi scans and the
distance is a fingerprint distance.
"""
import numpy as np


def windowed(R, W=3):
    """Centred W-scan average of each AP's RSSI. A value needs at least 2 readings in its window (1 when W = 1).
    R (N, L) dBm with nan = not heard -> Y (N, L) with nan where unavailable, C (N, L) readings averaged."""
    N, L = R.shape
    h, need = W // 2, (1 if W == 1 else 2)
    Y = np.full((N, L), np.nan); C = np.zeros((N, L), int)
    for j in range(N):
        win = R[max(0, j - h): j + h + 1]
        c = (~np.isnan(win)).sum(0)
        s = np.where(np.isnan(win), 0.0, win).sum(0)
        ok = c >= need
        Y[j, ok] = s[ok] / c[ok]; C[j] = c
    return Y, C


def fingerprint_distance(R, fill=-100.0):
    """RMS RSSI difference between every pair of scans, unheard APs at `fill` dBm (notebook section 2: r = 0.81 with distance)."""
    M = np.where(np.isnan(R), fill, R)
    return np.sqrt(((M[:, None, :] - M[None, :, :]) ** 2).mean(-1))


def propose_pairs(D, min_gap=7, max_pairs=None, nms=2, higher_is_better=False):
    """DROID-SLAM's add_proximity_factors on a scan-to-scan score matrix: pairs at least `min_gap` scans
    apart, best first, each taken pair suppressing its neighbours within `nms` scans."""
    N = D.shape[0]
    max_pairs = max_pairs if max_pairs is not None else max(2, N // 6)
    cand = [(D[i, j], i, j) for i in range(N) for j in range(i + min_gap, N)]
    cand.sort(reverse=higher_is_better)
    taken, sup = [], set()
    for d, i, j in cand:
        if (i, j) in sup:
            continue
        taken.append((i, j, float(d)))
        for di in range(-nms, nms + 1):
            for dj in range(-nms, nms + 1):
                sup.add((i + di, j + dj))
        if len(taken) >= max_pairs:
            break
    return taken


def sigma2_pair(k, ci, cj, sig_n, sig_s, ell, step_m, d=None):
    """Noise of a radio-flow factor (dB^2): random noise of both averaged ends + the part of the
    place-dependent error that the two scans do not share (exponential correlation over `ell` metres)."""
    dist = k * step_m if d is None else d
    return sig_n ** 2 * (1.0 / ci + 1.0 / cj) + 2 * sig_s ** 2 * (1 - np.exp(-dist / ell))


def sigma2_single(c, sig_n, sig_s):
    return sig_n ** 2 / c + sig_s ** 2
