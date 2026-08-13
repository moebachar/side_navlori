"""B2 APR adapter: labels CSVs + config for MS-Transformer (official repo untouched).

Pose convention: the loader applies no transform, so we feed base-frame planar
poses directly — t=(x, y, 0), scalar-first quaternion (cos(yaw/2), 0, 0, sin(yaw/2)).
"""
import json
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RUNS, load_index

repo = sys.argv[1]
out = os.path.join(RUNS, "mst")
os.makedirs(out, exist_ok=True)
idx = load_index()


def csv(sel, name, split):
    rows = idx[sel]
    with open(os.path.join(out, name), "w") as f:
        f.write("scene,split,seq,img_path,t1,t2,t3,q1,q2,q3,q4\n")
        for r in rows.itertuples():
            f.write(f"myroom,{split},0,{r.filename},{r.x},{r.y},0.0,"
                    f"{np.cos(r.yaw / 2)},0.0,0.0,{np.sin(r.yaw / 2)}\n")
    print(name, len(rows), "rows")


csv((idx.split == "train") & idx.db_keep, "myroom_train.csv", "train")
csv(idx.split == "test", "myroom_test.csv", "test")

cfg = json.load(open(os.path.join(repo, "7Scenes_config.json")))
cfg["ems-transposenet"]["num_scenes"] = 1
cfg["general"]["n_workers"] = 2
cfg["general"]["n_freq_print"] = 50
json.dump(cfg, open(os.path.join(out, "myroom_config.json"), "w"), indent=1)
print("myroom_config.json written (num_scenes=1)")
