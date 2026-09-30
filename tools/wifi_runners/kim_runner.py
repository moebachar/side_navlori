#!/usr/bin/env python
# Kim, Lee & Huang, "A scalable DNN architecture for multi-building/-floor indoor localization"
# (Big Data Analytics, 2018). Their released code does building/floor + RP classification; per the
# benchmark brief the coordinate-regression adaptation is: keep their SAE (exact architecture &
# training recipe from scalable_indoor_localization.py) and attach a 2-output (x,y) regression head
# in place of the multi-label classifier. Same Py3.7/TF1/Keras stack.
#   in:  Xtr (n,520) RSS dBm (-100=not heard), ytr (n,4)=[X,Y,floor,building], Xte (m,520)
#   out: pred (m,2) = [X, Y] in metres.  CPU-forced (tiny model; TF1.15 needs CUDA10, box has CUDA12).
#   usage: python kim_runner.py in.npz out.npz [seed]   (the optional seed only seeds numpy/TF)
import sys, os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import random
import numpy as np, warnings
warnings.filterwarnings("ignore")

inp, outp = sys.argv[1], sys.argv[2]
seed = int(sys.argv[3]) if len(sys.argv) > 3 else None
d = np.load(inp)
Xtr, ytr, Xte = d["Xtr"].astype("float64"), d["ytr"].astype("float64"), d["Xte"].astype("float64")
if seed is not None:
    import tensorflow as tf
    random.seed(seed); np.random.seed(seed); tf.set_random_seed(seed)

from keras.layers import Dense, Dropout
from keras.models import Sequential

INPUT_DIM = 520
SAE_HIDDEN = [256, 128, 64, 128, 256]     # their default --sae_hidden_layers
SAE_ACT, SAE_BIAS = "relu", False
CLF_HIDDEN = [128, 128]                    # their default --classifier_hidden_layers
CLF_ACT, CLF_BIAS, DROPOUT = "relu", False, 0.0
EPOCHS, BATCH = 20, 10                     # their defaults

# Kim standardises RSS after coding the missing AP as -110 dBm; not-heard is -100 in our matrix.
def prep(x):
    return np.where(x <= -100, -110.0, x)
Xtr, Xte = prep(Xtr), prep(Xte)
mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1.0
Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

# --- SAE, built & trained exactly as their code ---
model = Sequential()
model.add(Dense(SAE_HIDDEN[0], input_dim=INPUT_DIM, activation=SAE_ACT, use_bias=SAE_BIAS))
for u in SAE_HIDDEN[1:]:
    model.add(Dense(u, activation=SAE_ACT, use_bias=SAE_BIAS))
model.add(Dense(INPUT_DIM, activation=SAE_ACT, use_bias=SAE_BIAS))
model.compile(optimizer="adam", loss="mse")
model.fit(Xtr, Xtr, batch_size=BATCH, epochs=EPOCHS, verbose=0)
for _ in range((len(SAE_HIDDEN) + 1) // 2):   # keep encoder (through bottleneck), their exact rule
    model.pop()

# --- regression head in place of the classifier (the single-building adaptation) ---
model.add(Dropout(DROPOUT))
for u in CLF_HIDDEN:
    model.add(Dense(u, activation=CLF_ACT, use_bias=CLF_BIAS)); model.add(Dropout(DROPOUT))
model.add(Dense(2, activation="linear"))
model.compile(optimizer="adam", loss="mse")
ymu, ysd = ytr[:, :2].mean(0), ytr[:, :2].std(0); ysd[ysd == 0] = 1.0
model.fit(Xtr, (ytr[:, :2] - ymu) / ysd, batch_size=BATCH, epochs=EPOCHS, verbose=0)

pred = model.predict(Xte) * ysd + ymu
np.savez(outp, pred=pred)
print("KIM_OK", pred.shape)
