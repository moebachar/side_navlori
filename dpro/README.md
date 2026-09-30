# dpro: Deep Patch Radio Odometry

A small re-implementation of **DPVO** (Teed, Lipson, Deng, *Deep Patch Visual Odometry*, NeurIPS 2023) that tracks a robot from **WiFi RSSI scans** instead of images, with **no radio map**. The design follows the *DPRO — Deep Patch Radio Odometry* brief. Results and figures are in `side_navlori_wifi.ipynb` §4.

The idea is DPVO's. A network proposes corrected measurements and a confidence for each one. A differentiable bundle adjustment turns them into positions. Here the access points (APs) are the landmarks: their position and path-loss parameters are estimated online, in the role of DPVO's patch depths.

## Files, next to their DPVO counterparts

| dpro | DPVO | What it does |
|---|---|---|
| `radio_ops.py` | `projective_ops.py` | Log-distance model h(x, θ) = P0 − 10 n log10(‖x − p‖ + 0.5) and its Jacobians. It plays the role of the camera projection. |
| `ba.py` | `ba.py` | Gauss-Newton over 2D scan positions and AP parameters (px, py, P0, n). APs are eliminated with a Schur complement using 4×4 blocks. Cholesky solver that returns zero on failure. Differentiable. |
| `blocks.py` | `blocks.py`, `fastba.neighbors` | GatedResidual, SoftAgg, GradientClip and temporal neighbours, in plain torch (no torch_scatter). |
| `net.py` | `net.py` | Patchifier (radio patches and context encoder), Update operator, edge features (the correlation lookup), `RONet.forward` (the training loop of `VONet.forward`). |
| `dpro.py` | `dpvo.py` | Streaming runtime: damped linear motion model, forward and backward edges, sliding optimisation window, inactive edges kept in the BA, 12 init updates and 12 final updates. |
| `train.py` | `train.py` | AdamW with OneCycle. Structure-only first steps. Per-iteration pose loss after alignment (detached), plus an RSSI loss on synthetic clips (the flow loss). |
| `sim.py` | TartanAir | Synthetic sites: corridors, walls, APs, shadowing, noise, drop-outs and scan smear, calibrated on the golden runs. |
| `data.py` | data readers | Golden runs as scan sequences. One landmark = one radio × band (virtual BSSIDs merged). |
| `evaluate.py` | `evaluate_*.py` | ATE after 2D alignment (reflection allowed, optional scale). |

## Radio counterparts of DPVO's pieces

- **Radio patch:** one AP over the last 5 scans, for up to M = 6 heard APs per scan, picked at random as DPVO picks pixels.
  - The window is causal, so the system runs online.
  - All patches of an AP share its parameters θ.
- **Edge (patch k → scan j):** the network sees:
  - the AP's observed and predicted RSSI in scans j−2…j+2, with the mask and the residual;
  - a fingerprint similarity between the patch's scan and scan j;
  - the time gap;
  - the current distance to the AP and θ.

  It outputs δ (dB) and a weight w. The target is h(x_j, θ_a) + δ, as DPVO's target is the reprojection + flow.
- **"Not heard" is information.** Edges exist where the AP was not heard. The network decides what target and weight they get.

## Deviations from DPVO and the brief, and why

1. **Smoothness prior** on consecutive positions (σ = 0.3 m on the second difference). Without it, the RSSI-only solver bends the path to fit noise; the feasibility test showed 1.1 → 2.1 m error from a near-true start.
2. **Speed prior** of 0.2 ± 0.1 m/s. RSSI cannot observe scale, because a log-distance model absorbs any scale into P0, so without it trajectories shrink (to 0.56 of their length on the golden runs). DPVO has the same problem with a single camera and reports scale-aligned errors.
3. **Gauge:** the first 3 scan positions are given. The robot stands still at the start, so this fixes only the starting point. Evaluation aligns rotation and reflection.
4. **No AP identity embedding and no learned per-scan encoders.** Both would tie the network to one site's AP list. They are replaced by a fixed fingerprint similarity, so nothing site-specific is learned.
5. **Sizes:** DIM 64 (DPVO 384), M = 6 patches per scan (80), patch lifetime 6 scans (12 frames), about 5k training clips (240k).
6. **No keyframing and no loop closure.** Runs are 11 to 90 scans.

## Use

```python
import sys; sys.path.insert(0, "/mnt/x/side_navlori")
import torch
from dpro import RONet, DPRO, Config
from dpro.data import load_golden
net = RONet(); net.load_state_dict(torch.load("/root/navlori/runs/dpro/east/final.pth", weights_only=True))
seq = load_golden(["golden_run_5"])[0]
X = DPRO(net, Config(), mode="net").run(seq)      # (N, 2); mode="off" / "priors" for the non-learned solvers
```

Training runs on the CPU in about 45 min, with no GPU lock needed: `tools/notebook_runs/train_dpro.sh`. It trains on synthetic data first, then fine-tunes once on the east runs and once on the west runs. Checkpoints go to `/root/navlori/runs/dpro/{sim,east,west}/`.
