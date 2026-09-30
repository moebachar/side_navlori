#!/usr/bin/env python
# Standalone end-to-end test (runs in the notebook kernel venv). Builds protocol-A + B splits from
# run2, marshals through the CNNLoc sub-venv runner, reports position error. De-risks before wiring
# the notebook cell.
import sys, os, subprocess, tempfile, numpy as np
sys.path.append("/mnt/x/side_navlori/dataset_pipeline/export")
from load_dataset import Dataset

CNNLOC_REPO = "/root/navlori/mrepos_wifi/cnnloc"
CNNLOC_PY   = "/root/navlori/venvs_wifi/cnnloc/bin/python"
RUNNER      = os.path.join(CNNLOC_REPO, "cnnloc_runner.py")

ds = Dataset("/mnt/x/side_navlori/data/run2")
M, bssids, t_ns = ds.wifi_matrix()
gx, gy, _ = ds.gt_at(t_ns); XY = np.c_[gx, gy]; N = len(t_ns)
print("scans", N, "aps", M.shape[1])

def pad520(m):
    z = np.full((m.shape[0], 520), -100.0)   # -100 => not heard => maps to 0 in normalizeX_powed
    z[:, :m.shape[1]] = m
    return z

def cnnloc(Mtr, Ptr, Mte):
    Xtr, Xte = pad520(Mtr), pad520(Mte)
    ytr = np.c_[Ptr, np.zeros(len(Ptr)), np.zeros(len(Ptr))]   # [X, Y, floor=0, building=0]
    with tempfile.TemporaryDirectory() as td:
        ip, op = os.path.join(td, "in.npz"), os.path.join(td, "out.npz")
        np.savez(ip, Xtr=Xtr, ytr=ytr, Xte=Xte)
        r = subprocess.run([CNNLOC_PY, RUNNER, ip, op], cwd=CNNLOC_REPO,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(r.stdout.decode()[-600:])
        r.check_returncode()
        return np.load(op)["pred"]

# protocol A: interleaved 4-fold
K = 4; foldA = np.arange(N) % K
pA = np.zeros((N, 2))
for f in range(K):
    tr, te = foldA != f, foldA == f
    pA[te] = cnnloc(M[tr], XY[tr], M[te])
eA = np.hypot(pA[:,0]-XY[:,0], pA[:,1]-XY[:,1])
print(f"CNNLoc A(in-map)  med {np.median(eA):.2f} m  mean {eA.mean():.2f}  <=2m {(eA<=2).mean():.0%}")

# protocol B: time-forward 60/40
sB = int(round(0.60*N))
pB = cnnloc(M[:sB], XY[:sB], M[sB:])
eB = np.hypot(pB[:,0]-XY[sB:,0], pB[:,1]-XY[sB:,1])
print(f"CNNLoc B(fwd)     med {np.median(eB):.2f} m  mean {eB.mean():.2f}")
