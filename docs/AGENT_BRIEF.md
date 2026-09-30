# Brief: side_navlori agent (per-sensor notebooks)

Read this first. `AGENT_HANDOFF.md`, `ENVIRONMENT.md`, `DATA.md` and `METHODS.md` give the background; they predate the 2026-09-29 cleanup, so where they disagree with this file, this file wins.

## Mission

The original goal of side_navlori: for each sensor of the real TurtleBot3, one clean, pitchable notebook (`side_navlori_<sensor>.ipynb`) that:
1. shows the sensor;
2. cleans and transforms it;
3. evaluates 2–3 classic baselines plus 1–2 recent state-of-the-art methods, using their **official code unchanged**, on the local GPU;
4. later, adds a personal contribution.

The numbers are for a journal paper: fair, reproducible, honest (median / mean error in metres, clear protocols).

## Where things stand

- **`side_navlori_camera.ipynb`: done on the old `run2`** (commit 621a027, 2026-09-17).
  - GeM, NetVLAD, SALAD, MST, ACE, Reloc3r and DPVO, under two protocols.
  - Protocol A (in-map) medians: ACE 3.9 cm, MST 10.4, NetVLAD 19.5, SALAD 20.1, GeM 21.2, Reloc3r 28.2 cm.
  - Protocol B (unseen space): all methods collapse to 2.4–6.3 m.
  - DPVO visual odometry: ATE 11.4 cm.
- **`side_navlori_wifi.ipynb`: done on `run2`** (52 scans, 59 APs).
  - Baselines RADAR, WkNN, Horus; CNNLoc and Kim run unchanged in their own Python 3.7 / TF1 sub-venvs.
  - WkNN is best in-map (~1.7 m median). Protocol B gives ~8–10 m for all methods.
  - The deep models overfit on 52 scans; "the big dataset is their real test".
- **IMU and wheel-odometry notebooks: not started.**
- **The big dataset now exists:** 12 golden runs (25 min, 262 m, ~370 WiFi scans, 45k camera frames), with ground truth from lidar SLAM + gyroscope placed on 26 surveyed AprilTags (held-out accuracy ~15 cm median). They are in `data\golden_run_1..12` (a link to the vault). `data\run1..run8` are the older recordings, `run2` included.

## Backlog, in priority order

1. **Update the docs and runners to the new layout** (short, do it first).
   - `data\` is now a link to `X:\navlori-data\robot`.
   - The exporters and loader moved to `dataset_pipeline\export\` (`load_dataset.py`).
   - The WSL runner scripts moved to `tools\notebook_runs\`. They still point at `/root/navlori/scripts_local` and `/root/navlori/data`, which no longer exist.
   - In WSL, read data from `/mnt/x/side_navlori/data/...`. If I/O is too slow, copy the runs you need to a native WSL folder, and delete the copy when done.
   - Rewrite `ENVIRONMENT.md` / `DATA.md` accordingly.
2. **Re-run WiFi and camera on the golden runs.**
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
