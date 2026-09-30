# Brief: side_navlori agent (per-sensor notebooks)

Read this first. `AGENT_HANDOFF.md`, `ENVIRONMENT.md`, `DATA.md` and `METHODS.md` give the background; they predate the 2026-09-29 cleanup, so where they disagree with this file, this file wins.

## Mission

The original goal of side_navlori: for each sensor of the real TurtleBot3, one clean, pitchable notebook (`side_navlori_<sensor>.ipynb`) that:
1. shows the sensor;
2. cleans and transforms it;
3. evaluates 2–3 classic baselines plus 1–2 recent state-of-the-art methods, using their **official code unchanged**, on the local GPU;
4. later, brainstrom innovate and adds a personal contribution.
5. make Mohamed learn and have insights along the path by not scilently passing things, but by including him in decision and inlightning him.

The numbers are for a journal paper: fair, reproducible, honest (median / mean error in metres, clear protocols).

## Where things stand

- **`side_navlori_camera.ipynb`: done on the old `run2`** (commit 621a027, 2026-09-17).
  - GeM, NetVLAD, SALAD, MST, ACE, Reloc3r and DPVO, under two protocols.
  - Protocol A (in-map) medians: ACE 3.9 cm, MST 10.4, NetVLAD 19.5, SALAD 20.1, GeM 21.2, Reloc3r 28.2 cm.
  - Protocol B (unseen space): all methods collapse to 2.4–6.3 m.
  - DPVO visual odometry: ATE 11.4 cm.
- **`side_navlori_wifi.ipynb`: re-run on the 12 golden runs (2026-09-30)**, 370 scans, 120 CESI BSSIDs (24 physical radios). Protocols P0 pooled random (leaky reference), P1 leave-one-run-out, P2 unseen 5 m blocks; 5 seeds. Median error P0 / P1 / P2: WkNN 1.49 / 2.28 / 4.39 m, Kim 2.20 / 3.13 / 3.99, RADAR 2.02 / 3.19 / 4.77, Horus 1.93 / 3.27 / 4.63, CNNLoc 5.50 / 7.40 / 7.80. Runners: `tools/wifi_runners/`. Earlier pilot on `run2` (below):
  - Baselines RADAR, WkNN, Horus; CNNLoc and Kim run unchanged in their own Python 3.7 / TF1 sub-venvs.
  - WkNN is best in-map (~1.7 m median). Protocol B gives ~8–10 m for all methods.
  - The deep models overfit on 52 scans; "the big dataset is their real test".
- **DPRO (Deep Patch Radio Odometry), the first personal contribution (2026-09-30):**
  - Package `dpro/`: a re-implementation of DPVO for WiFi, tracking with no radio map (see `dpro/README.md`). Results in the WiFi notebook §4.
  - Trained on synthetic sites calibrated to the golden runs; CPU training ~45 min (`tools/notebook_runs/train_dpro.sh`).
  - Mean per-run median ATE on the 12 golden runs, zero-shot: **1.51 m**. Same solver with the network off: 2.70 m; priors only: 3.57 m. DPRO is better on 9/12 runs.
  - Fine-tuning on one zone's real clips did not help (1.74 m).
- **IMU and wheel-odometry notebooks: not started.**
- **The big dataset now exists:** 12 golden runs (25 min, 262 m, ~370 WiFi scans, 45k camera frames), with ground truth from lidar SLAM + gyroscope placed on 26 surveyed AprilTags (held-out accuracy ~15 cm median). They are in `data\golden_run_1..12` (a link to the vault). `data\run1..run8` are the older recordings, `run2` included.

## Backlog, in priority order

1. ~~**Update the docs and runners to the new layout.**~~ Done 2026-09-30: notebook setup cells read `/mnt/x/side_navlori/data/<run>` and the loader from `dataset_pipeline/export`; runners point at `tools/`; camera runners take the GPU lock; `ENVIRONMENT.md` / `DATA.md` rewritten.
2. **Re-run WiFi and camera on the golden runs.** WiFi done (P0/P1/P2 above); camera next, same protocols, under the GPU lock.
   - Define protocols across runs (e.g. train on some runs, test on others: in-map and unseen-area), with seeds and several folds.
   - Keep the methods and their official code unchanged.
3. **IMU notebook:** classic PDR / strapdown integration and learned inertial odometry (e.g. RoNIN, TLIO-style), evaluated on the golden runs.
4. **Wheel-odometry notebook:** differential-drive dead reckoning, with and without gyroscope heading, plus a learned correction baseline.
   - The wheel velocity columns named `*_vel_radps` actually hold **m/s**.
   - The ground truth uses the gyroscope for heading, so odometry drift below ~15 cm is inside the ground-truth accuracy. Say so in the notebook.

## Rules

- **You work in `X:\side_navlori` (its own git repo, branch `main`).** Commit small, push when a notebook section works. Only the notebooks, `tools/`, `docs/` and `slides_progress/PROMPTS.md` are versioned; data is not.
- **The robot data is read-only for you.** It belongs to the dataset agent, who works in navlori-fusion (`docs/agents/dataset.md` there). If you find a data problem, tell the user rather than editing `data\` or `dataset_pipeline\`.
- **One GPU for four agents:** take the lock before any GPU job. From WSL:
  ```bash
  /root/navlori/venv/bin/python /mnt/x/navlori-fusion/scripts/gpu_lock.py run --wait --who side_navlori --what "camera notebook" -- <command>
  ```
  For detached papermill runs, `acquire` first and make `release --who side_navlori` the runner's last step. GPU memory: the ACE and DPVO lessons in `METHODS.md` still apply (fresh process per fold, `TORCH_CUDA_ARCH_LIST=6.1`).
- **No new top-level folders on X:.** Scratch goes in the WSL home or the session scratchpad.
