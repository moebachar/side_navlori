"""R2b recent retrieval: DINOv2-SALAD (CVPR 2024), official model via torch.hub.

Pinned to serizba/salad main commit; official eval transform (322x322).
Requires: pip install "pytorch-lightning>=2.1,<3". ~3-6 min on T4.
"""
import os
import sys
import time
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMAGES, RUNS, load_index, write_pred

PIN = "6aede13a3f6c25750bf7fde10209c06cb73060bb"
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.hub.load(f"serizba/salad:{PIN}", "dinov2_salad", trust_repo=True).eval().to(dev)
tf = T.Compose([T.Resize((322, 322), interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


@torch.inference_mode()
def descriptors(files):
    D = np.zeros((len(files), 8448), np.float32)
    for i in range(0, len(files), 32):
        batch = torch.stack([tf(Image.open(os.path.join(IMAGES, f)).convert("RGB"))
                             for f in files[i:i + 32]]).to(dev)
        with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
            D[i:i + len(batch)] = model(batch).float().cpu().numpy()
    return D


idx = load_index()
db = idx[(idx.split == "train") & idx.db_keep].reset_index(drop=True)
q = idx[idx.split == "test"].reset_index(drop=True)
t0 = time.time()
Ddb, Dq = descriptors(list(db.filename)), descriptors(list(q.filename))
S = Dq @ Ddb.T
nn = S.argmax(1)
os.makedirs(os.path.join(RUNS, "salad"), exist_ok=True)
np.savez_compressed(os.path.join(RUNS, "salad", "desc.npz"), Ddb=Ddb, Dq=Dq,
                    db=db.filename.values.astype(str), q=q.filename.values.astype(str))
write_pred("salad", q.filename, db.x.values[nn], db.y.values[nn], db.yaw.values[nn],
           extra=dict(sim=S.max(1), match=db.filename.values[nn]))
print(f"salad: {len(db)} db / {len(q)} queries in {time.time() - t0:.0f}s on {dev}")
