"""Radio counterpart of DPVO's projective_ops: the propagation model plays the role of the camera projection.

DPVO:  patch k (pixel + inverse depth) seen from frame i, reprojected into frame j with the poses.
DPRO:  AP a (position, reference power, path-loss exponent) "reprojected" into scan j with the scan position:
       h(x_j, theta_a) = P0_a - 10 n_a log10(||x_j - p_a|| + EPS_D)
"""
import math
import torch

LOG10 = math.log(10.0)
EPS_D = 0.5          # m, keeps the log finite when the robot passes under the AP
RSSI_MU, RSSI_SIG = -70.0, 15.0


def norm_rssi(r):
    """dBm -> network units (0 where not heard is handled by the mask channel)."""
    return (r - RSSI_MU) / RSSI_SIG


def predict(x, th):
    """h(x, theta): expected RSSI (dBm) at positions x (..., 2) for AP parameters th (..., 4) = (px, py, P0, n)."""
    d = torch.sqrt(((x - th[..., :2]) ** 2).sum(-1) + 1e-6)
    return th[..., 2] - 10.0 * th[..., 3] * torch.log10(d + EPS_D)


def predict_jac(x, th):
    """h and its Jacobians w.r.t. the scan position (..., 2) and the AP parameters (..., 4)."""
    diff = x - th[..., :2]
    d = torch.sqrt((diff ** 2).sum(-1) + 1e-6)
    lg = torch.log10(d + EPS_D)
    h = th[..., 2] - 10.0 * th[..., 3] * lg
    g = -10.0 * th[..., 3] / (LOG10 * (d + EPS_D))            # dh/dd
    Jx = (g / d)[..., None] * diff
    Jth = torch.stack([-Jx[..., 0], -Jx[..., 1], torch.ones_like(h), -10.0 * lg], -1)
    return h, Jx, Jth


def init_ap(xs, rssi, n0=3.0, d0=2.5, offset=None):
    """Initial AP parameters from the scans that heard it (DPVO initialises depth at random / the median):
    position near the scan with the strongest reading, P0 so that that reading sits d0 metres away."""
    k = int(torch.argmax(rssi))
    p = xs[k] + (offset if offset is not None else 0.0)
    P0 = rssi[k] + 10.0 * n0 * math.log10(d0 + EPS_D)
    return torch.tensor([float(p[0]), float(p[1]), float(P0), n0])
