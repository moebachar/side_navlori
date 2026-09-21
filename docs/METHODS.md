# Methods — evaluation & how to re-run

The evaluation philosophy (see AGENT_HANDOFF §2): **classic baselines + recent
SOTA competitors, competitors run from their official code UNCHANGED in isolated
sub-venvs**, scored under fixed protocols, reporting **median + mean localization
error in metres** (and room-level accuracy). The method shortlist came from a
deep-research pass saved at `compass_artifact_wf-*.md` (repo root).

## WiFi (`side_navlori_wifi.ipynb`, §3)

### Protocols (mirror the camera notebook)
- **A — in-map:** interleaved **4-fold**; every scan predicted by a model that
  never saw it (interpolation inside the surveyed route).
- **B — forward:** time-forward **60/40** (train first 60 % of the route, test
  last 40 %) — extrapolation to unseen space.

### Harness (in the notebook)
`M, bssids, t_ns = ds.wifi_matrix()` → GT `XY = gt_at(t_ns)`; then `report(name,
predA, predB)` fills `RESULTS[name]` (medians/means/≤2 m) and `PRED[name]`
(protocol-A predictions, used by the §3d map and §3e GIF); `by_protocol(fn)`
runs any `predict(Mtr, Ptr, Mte)` under both protocols.

### The five methods
- **Baselines (in-notebook, RSSI-only, memory-based):** RADAR (kNN, k=1),
  Weighted-kNN (k=5, "powed" RSS representation), Horus (per-cell Gaussian
  likelihood).
- **CNNLoc** (Song et al., IEEE Access 2019 — SAE + 1D-CNN) and **Kim** (Kim et
  al., Big Data Analytics 2018 — SAE + regression head): official/faithful code,
  each in its **own Py3.7/TF1.15 sub-venv**, driven from the notebook by
  subprocess with `.npz` marshalling. Runners:
  `/root/navlori/mrepos_wifi/cnnloc/cnnloc_runner.py`,
  `/root/navlori/mrepos_wifi/kimdnn/kim_runner.py`. Inputs are padded to the
  **520-column UJIIndoorLoc layout** with `y = [X, Y, 0, 0]`; building/floor
  heads disabled (single floor); CPU-forced. The §3b cell defines a `SOTA` dict
  + `sota_predict(name)` wrapper — add a new competitor by adding a runner + an
  entry there.

### Results so far (run2, 52 scans) — the expected story
In-map, the memory baselines **beat** the deep nets (WkNN best, ~1.72 m median);
the deep nets **overfit** on 52 scans (CNNLoc ~3.2 m, Kim ~2.2 m). Protocol B:
**all** methods ~8–10 m → WiFi maps what it has seen, it does not extrapolate.
This is why the **bigger dataset matters** — it's the real test of the deep
methods.

### Re-running on the new dataset
1. Point the setup cell `DATA` at the new run (e.g. `/content/data/run3`), making
   sure that folder exists under `/root/navlori/data/` (ENVIRONMENT.md).
2. Execute headless (ENVIRONMENT.md nbconvert command). §3 baselines + both
   sub-venv deep methods re-run automatically (a few minutes).
3. There is **no fixed random seed** → ~0.5 m run-to-run wobble on small N (pure
   small-sample noise). For final paper numbers, **seed and/or average** several
   runs of the deep methods.
- Dropped for now: a GNN competitor (**IndoorGNN**) — its GitLab clone failed and
  the user chose to skip it. It's the natural next competitor if wanted
  (adapt its classification head to (x,y) regression, add a runner + SOTA entry).

## Camera (`side_navlori_camera.ipynb`)

Evaluated and committed (commit `621a027`): **6 methods + DPVO** (deep patch
visual odometry), each run on the GPU; the exact list + cells are in the
notebook. **DPVO** runs in its **own legacy sub-venv** (built with `uv venv
--python 3.x`, the pattern predating the micromamba WiFi venvs) with
`TORCH_CUDA_ARCH_LIST="6.1;7.5"` for the sm_61 GPU. Setup cell uses
`DATA = "/content/data/run1"`; re-run on a new run by repointing `DATA` and
re-executing headless (inline backend so the figures embed).

## Reproducibility notes
- Notebooks are committed **with outputs + figures embedded** — always re-execute
  headless with the inline backend after changing anything, so the committed
  artifact matches the code.
- The heavy demo videos / slide assets from the presentation side-quest live in
  `figures/` (gitignored): `data_collection.mp4` (3-panel demo),
  `demo_cockpit_big.mp4`, `demo_slam_timelapse_big.mp4`, `slide_map_big.png`,
  `slide_robot.png`. Not part of the pipeline — reference only.
