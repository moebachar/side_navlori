"""B1 retrieval floor: frozen torchvision ResNet50 (IMAGENET1K_V2) + GeM pooling.

Top-1 nearest neighbor in descriptor space; predicted pose = matched database
frame's GT. Database = train split, resting episodes deduplicated. ~2-4 min on T4.
"""
import os
import sys
import time
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms as T
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMAGES, RUNS, load_index, write_pred

dev = "cuda" if torch.cuda.is_available() else "cpu"
m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
body = torch.nn.Sequential(*list(m.children())[:-2]).eval().to(dev)
tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def gem(x, p=3.0, eps=1e-6):
    return x.clamp(min=eps).pow(p).mean((-2, -1)).pow(1.0 / p)


@torch.inference_mode()
def descriptors(files):
    D = np.zeros((len(files), 2048), np.float32)
    for i in range(0, len(files), 48):
        batch = torch.stack([tf(Image.open(os.path.join(IMAGES, f)).convert("RGB"))
                             for f in files[i:i + 48]]).to(dev)
        with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
            d = gem(body(batch)).float()
        D[i:i + len(batch)] = torch.nn.functional.normalize(d, dim=1).cpu().numpy()
    return D


idx = load_index()
db = idx[(idx.split == "train") & idx.db_keep].reset_index(drop=True)
q = idx[idx.split == "test"].reset_index(drop=True)
t0 = time.time()
Ddb, Dq = descriptors(list(db.filename)), descriptors(list(q.filename))
S = Dq @ Ddb.T
nn = S.argmax(1)
os.makedirs(os.path.join(RUNS, "gem"), exist_ok=True)
np.savez_compressed(os.path.join(RUNS, "gem", "desc.npz"), Ddb=Ddb, Dq=Dq,
                    db=db.filename.values.astype(str), q=q.filename.values.astype(str))
write_pred("gem", q.filename, db.x.values[nn], db.y.values[nn], db.yaw.values[nn],
           extra=dict(sim=S.max(1), match=db.filename.values[nn]))
print(f"gem: {len(db)} db / {len(q)} queries in {time.time() - t0:.0f}s on {dev}")
