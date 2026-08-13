"""Dump per-frame MS-Transformer poses (their code as an unchanged library).

The official test mode logs errors but never writes poses; this mirrors its test
path exactly (batch 1, shuffle False, scene=None) and saves pred.csv.

Usage: python mst_dump.py <repo_root> <checkpoint.pth>
"""
import os
import sys
CM = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CM)
import numpy as np
import torch
import torchvision.ops
import torchvision.ops.misc

if not hasattr(torchvision.ops, "_new_empty_tensor"):
    torchvision.ops._new_empty_tensor = lambda x, shape: x.new_empty(shape)
if not hasattr(torchvision.ops.misc, "_output_size"):
    torchvision.ops.misc._output_size = (
        lambda dim, input, size=None, scale_factor=None: [])
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from common import IMAGES, RUNS, load_index, write_pred  # noqa: E402

repo, ckpt = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
sys.path.insert(0, repo)
os.chdir(repo)
import json  # noqa: E402
from util import utils  # noqa: E402
from datasets.CameraPoseDataset import CameraPoseDataset  # noqa: E402
from models.pose_regressors import get_model  # noqa: E402

cfg = json.load(open(os.path.join(RUNS, "mst", "myroom_config.json")))
config = {**cfg["ems-transposenet"], **cfg["general"]}
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = get_model("ems-transposenet", "./models/backbones/efficient-net-b0.pth",
                  config).to(dev)
model.load_state_dict(torch.load(ckpt, map_location=dev))
model.eval()

transform = utils.test_transforms.get("baseline")
ds = CameraPoseDataset(IMAGES, os.path.join(RUNS, "mst", "myroom_test.csv"), transform)
dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
P = []
with torch.no_grad():
    for mb in dl:
        for k, v in mb.items():
            mb[k] = v.to(dev)
        mb["scene"] = None
        P.append(model(mb).get("pose").cpu().numpy())
P = np.concatenate(P)
q = load_index()
q = q[q.split == "test"]
yaw = 2 * np.arctan2(P[:, 6], P[:, 3])
write_pred("mst", q.filename, P[:, 0], P[:, 1], yaw)
print("mst: poses written", P.shape)
