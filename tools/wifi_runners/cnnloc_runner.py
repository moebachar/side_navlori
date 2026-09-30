#!/usr/bin/env python
# CNNLoc (Song et al., IEEE Access 2019 / UTS) official EncoderDNN, used UNCHANGED as a library.
# Runs in its own Py3.7 / TF1.15 / Keras2.2.5 sub-venv. Marshals data via .npz:
#   in:  Xtr (n,520) padded RSS in dBm (-100 = not heard), ytr (n,4)=[X,Y,floor,building], Xte (m,520)
#   out: pred (m,2) = [X, Y] in metres
# Location branch only (single floor/building here); building/floor branches disabled via module globals
# without editing their source. CPU-forced: TF1.15 needs CUDA10 (box has CUDA12); models are tiny so
# results are identical and instant on CPU.
# Usage: python cnnloc_runner.py in.npz out.npz [seed], run with cwd = their repo (it saves its .h5 models
# there, so never run two CNNLoc fits in the same cwd at once). The optional seed only seeds numpy/TF.
import sys, os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, os.getcwd())                   # their encoder_model.py lives in the cwd (their repo)
import random
import numpy as np

inp, outp = sys.argv[1], sys.argv[2]
seed = int(sys.argv[3]) if len(sys.argv) > 3 else None
d = np.load(inp)
Xtr, ytr, Xte = d["Xtr"].astype("float64"), d["ytr"].astype("float64"), d["Xte"].astype("float64")

import warnings; warnings.filterwarnings("ignore")
import tensorflow as tf
if seed is not None:
    random.seed(seed); np.random.seed(seed); tf.set_random_seed(seed)
import encoder_model as em
from keras.optimizers import RMSprop
from keras.losses import MSE

# location-only; leave their code on disk untouched, just flip the module-level switches
em.Train_AE = True;        # train the SAE on THIS split (fair per-dataset pipeline)
em.Train_Location = True;  em.Run_Location = True
em.Train_Floor = False;    em.Run_Floor = False;  em.Train_New_Floor = False
em.Train_Building = False;  em.Run_Building = False

m = em.EncoderDNN()
# hyper-params exactly as their main.py (RMSprop config)
m.patience = 3; m.b = 2.8
m.epoch_AE = 40; m.epoch_position = 60; m.epoch_floor = 40; m.epoch_building = 40
m.dropout = 0.7; m.loss = MSE; m.opt = RMSprop(lr=0.001)

m.fit(Xtr, ytr, valid_x=Xtr, valid_y=ytr)     # val not used for stopping (callbacks commented out upstream)
_, _, lo, la = m.predict(Xte)
pred = np.c_[np.ravel(lo), np.ravel(la)]
np.savez(outp, pred=pred)
print("CNNLOC_OK", pred.shape)
