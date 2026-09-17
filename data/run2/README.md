# run2 — TurtleBot3 recording 2026-09-16 (CESI)

Second recording, **same format as `../run1/`** (load with the shared
`../scripts/load_dataset.py`: `Dataset("…/data/run2")`). A ~220 s drive through
the CESI building — start room → corridor → far room (a real ~19 m multi-room
route, not a single-room loop). Recorded on the TurtleBot3 Waffle Pi with the
sensor pipeline reworked for higher rates.

## What's different from run1
| | run1 (2026-07-24) | run2 (2026-09-16) |
|---|---|---|
| IMU / odom | ~20 Hz | **~127 Hz** (turtlebot3_node publish timer 50→5 ms) |
| camera | 820×616 | **1280×720** RGB JPEG, ~30 Hz (6252 frames) |
| wifi | 28 scans / 56 APs | **52 scans / 78 APs** (~2900 rows) |
| route | one small closed loop | ~19 m, multi-room (corridor) |
| duration | 116.8 s | 220.7 s |

## Layout (per-modality, raw asynchronous stamps, epoch ns)
- `camera/camera.csv` + `camera/images/<t_ns>.jpg` — **images are gitignored** (in the bag only)
- `imu/imu.csv`, `mag/mag.csv` (dead sensor, zeros), `wheel_odom/{odom,joint_states}.csv`
- `wifi/wifi.csv` (+ `wifi_raw.jsonl`) — key on `bssid`; read with `keep_default_na=False`
- `lidar/scans.npz` — ragged 236–258 beams/scan, ~9 Hz
- `ground_truth/gt_pose.csv` — lidar-SLAM poses at scan stamps (+ `map.png`, `map_points.npz`, `gt_overview.png`)
- `calib/` — copied from run1 (⚠ `camera_intrinsics_nominal.yaml` is the 820×616 prior; **run2 camera is 1280×720** — recalibrate before metric camera work; the true `camera_info` is in the bag)

## Ground truth
Built with `../scripts/build_ground_truth.py` (PLICP scan-to-map + refinement):
**2008/2008 scans matched, 0 fallbacks, median residual 7.8 mm.** Wheel odometry
drifts ~1.8 m over the 19 m route (heavy rotation → slip) — use the lidar GT, not
odom. See `ground_truth/gt_overview.png`.

## Caveats
Captured on a **low battery (~29 %)** — the motors browned out intermittently
(brief `cmd_vel` dropouts) and the run ends where the pack sagged. All sensor
streams are intact and the SLAM is clean, but it's a rough take, not a pristine
protocol run. Source bag: `navlori_20260916_091441` (rosbag2, sqlite3; gitignored).
