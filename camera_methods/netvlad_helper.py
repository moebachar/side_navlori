"""B3 NetVLAD via the official hloc toolbox: stage images, then top-1 poses.

Usage: netvlad_helper.py stage | post
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMAGES, RUNS, load_index, write_pred

root = os.path.join(RUNS, "netvlad")


def stage():
    idx = load_index()
    for sub, sel in [("db", (idx.split == "train") & idx.db_keep),
                     ("query", idx.split == "test")]:
        d = os.path.join(root, "images", sub)
        os.makedirs(d, exist_ok=True)
        for fn in idx[sel].filename:
            dst = os.path.join(d, fn)
            if not os.path.lexists(dst):
                os.symlink(os.path.join(IMAGES, fn), dst)
    print("staged image symlinks under", os.path.join(root, "images"))


def post():
    import h5py
    import numpy as np
    idx = load_index().set_index("filename")
    with h5py.File(os.path.join(root, "global-feats-netvlad.h5"), "r") as fd:
        db_names = sorted(fd["db"].keys())
        q_names = sorted(fd["query"].keys())
        Ddb = np.stack([fd[f"db/{n}"]["global_descriptor"][...]
                        for n in db_names]).astype(np.float32)
        Dq = np.stack([fd[f"query/{n}"]["global_descriptor"][...]
                       for n in q_names]).astype(np.float32)
    S = Dq @ Ddb.T
    nn = S.argmax(1)
    match = np.array(db_names)[nn]
    g = idx.loc[match]
    write_pred("netvlad", q_names, g.x.values, g.y.values, g.yaw.values,
               extra=dict(sim=S.max(1), match=match))
    print(f"netvlad: top-1 poses for {len(q_names)} queries written")


globals()[sys.argv[1]]()
