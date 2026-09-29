import sys, os, subprocess, tempfile, numpy as np
sys.path.append("/content/data/scripts")
from load_dataset import Dataset
PY   = "/root/navlori/venvs_wifi/kim/bin/python"
RUN  = "/root/navlori/mrepos_wifi/kimdnn/kim_runner.py"

ds = Dataset("/content/data/run2")
M, bssids, t_ns = ds.wifi_matrix()
gx, gy, _ = ds.gt_at(t_ns); XY = np.c_[gx, gy]; N = len(t_ns)

def pad520(m):
    z = np.full((m.shape[0], 520), -100.0); z[:, :m.shape[1]] = m; return z
def kim(Mtr, Ptr, Mte):
    with tempfile.TemporaryDirectory() as td:
        ip, op = td+"/in.npz", td+"/out.npz"
        np.savez(ip, Xtr=pad520(Mtr), ytr=np.c_[Ptr, np.zeros(len(Ptr)), np.zeros(len(Ptr))], Xte=pad520(Mte))
        r = subprocess.run([PY, RUN, ip, op], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(r.stdout.decode()[-300:]); r.check_returncode()
        return np.load(op)["pred"]

K = 4; foldA = np.arange(N) % K; pA = np.zeros((N, 2))
for f in range(K):
    tr, te = foldA != f, foldA == f
    pA[te] = kim(M[tr], XY[tr], M[te])
eA = np.hypot(pA[:,0]-XY[:,0], pA[:,1]-XY[:,1])
print(f"Kim A(in-map)  med {np.median(eA):.2f} m  mean {eA.mean():.2f}  <=2m {(eA<=2).mean():.0%}")
sB = int(round(0.60*N)); pB = kim(M[:sB], XY[:sB], M[sB:])
eB = np.hypot(pB[:,0]-XY[sB:,0], pB[:,1]-XY[sB:,1])
print(f"Kim B(fwd)     med {np.median(eB):.2f} m  mean {eB.mean():.2f}")
