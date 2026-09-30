"""DPRO: Deep Patch Radio Odometry. A small re-implementation of DPVO (Teed, Lipson, Deng, NeurIPS 2023) for WiFi
RSSI scans instead of images. Files mirror DPVO's: radio_ops (projective_ops), ba, blocks, net, dpro (dpvo), train,
plus data (golden runs), sim (synthetic sites) and evaluate (ATE)."""
from .net import RONet
from .dpro import DPRO, Config
