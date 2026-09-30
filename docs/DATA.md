# Data: layout, loader, pipeline

The robot data lives in the **vault `X:\navlori-data`** (git + DVC). `X:\side_navlori\data`
is a junction to its `robot/` folder (`/mnt/x/side_navlori/data` from WSL).
**Read-only for this agent:** the dataset agent owns the data and the pipeline
(`X:\navlori-fusion\docs\agents\dataset.md`). Report data problems to the user.

## Robot and sensors

**TurtleBot3 Waffle Pi**, ROS 2 Humble, recorded as rosbag2. Topics:
`/camera/image_raw/compressed` (imx219), `/imu` `/odom` `/joint_states` `/magnetic_field`
(OpenCR, one shared timer), `/scan` (LDS-02 lidar, ~9.5 Hz), `/wifi/rssi` (custom
`wifi_scanner`, ~1 scan / 4 s), `/tf` `/tf_static` `/battery_state`. `/cmd_vel` was never
recorded. The magnetometer is dead (all zeros).

## What is in `data\`

| folder | what |
|---|---|
| `golden_run_1..12` | **the main dataset** (2026-09-24): 12 runs, 25 min, 262 m, ~370 WiFi scans, 45k frames. Ground truth in the **building frame**. |
| `run1`, `run2` | older laps (2026-07-24, 2026-09-16), used by the current notebooks. GT in each run's own **SLAM start frame**: not comparable with the golden runs. |
| `run3..run8` | 2026-09-17 recordings, SLAM-frame GT (superseded by the golden runs). |
| `golden_floorplan/` | floor plan, `plan_transform.json`, all paths on the plan, **`golden_runs_summary.csv`** (per-run duration, length, tags, GT accuracy, scan counts). |
| `tags_ground_truth.json` | surveyed positions of the 26 AprilTags (building frame). |
| `calib/` | Kalibr camera calibration at 640×480 (`kalibr_640x480/`), calibration targets. |

Each `*.dvc` file next to a folder is its DVC pointer. Don't touch them.

## Run folder layout (all runs share it)

```
<run>/
  camera/       images/<t_ns>.jpg, camera.csv (t_ns, t_bag_ns, filename)          ~29 Hz, 640x480 (golden)
  imu/          imu.csv (t_ns, t_bag_ns, wx..wz, ax..az, qx..qw)                   ~145 Hz (golden)
  wheel_odom/   odom.csv (x, y, yaw, v_lin, w_ang), joint_states.csv              ~145 Hz (golden)
  mag/          mag.csv                                                            dead
  wifi/         wifi.csv (long: t_start_ns, t_end_ns, scan_idx, bssid, ssid, rssi_dbm, freq_mhz, last_seen_ms), wifi_raw.jsonl
  lidar/        scans.npz (ragged CSR: offsets/ranges/angles/t_ns), scans_meta.csv
  ground_truth/ gt_pose.csv (t_ns, x, y, yaw, quality, ...), gt_info.json, method.md, maps and plots
  calib/        camera_intrinsics.yaml (Kalibr, golden), extrinsics.yaml, robot.yaml, apriltags.yaml
  README.md     dataset card
```
Timestamps everywhere: raw async int64 nanoseconds, no resampling.

**Known quirk:** in `joint_states.csv` the columns `left_vel_radps` / `right_vel_radps`
hold **m/s** (wheel surface speed), not rad/s. A rename is pending the user's approval.

## Ground truth of the golden runs

Gyro heading + lidar-SLAM distances (PLICP), placed **rigidly per run** in the building
frame by a joint least-squares adjustment over all AprilTag sightings of all runs.
Not drift-corrected by the tags. Accuracy per run is in `golden_runs_summary.csv`:
held-out tag error ~2–24 cm median (run 9: 43 cm); runs 2 and 8 are map-matched to the
floor plan instead; run 1 has no held-out check. Details: `<run>/ground_truth/method.md`.
**Consequence:** errors below ~15 cm are inside the GT accuracy; say so when reporting.

## Loading a run

```python
import sys
sys.path.append("/mnt/x/side_navlori/dataset_pipeline/export")
from load_dataset import Dataset
ds = Dataset("/mnt/x/side_navlori/data/golden_run_7")
ds.camera / ds.imu / ds.odom / ds.joints / ds.wifi / ds.gt / ds.lidar   # DataFrames / arrays
M, bssids, t_ns = ds.wifi_matrix(max_age_ms=4000, fill_dbm=-100)       # (n_scan, n_ap) RSSI matrix
gx, gy, yaw = ds.gt_at(t_ns)                                            # GT pose at any stamps
```
`wifi_matrix()` is per run: its AP columns differ between runs. For cross-run work, build
one shared BSSID vocabulary and align every run's matrix to it.

## The pipeline (read-only reference)

`dataset_pipeline/` = `X:\navlori-fusion\scripts\dataset`:
- `export/`: bag → run folder (`export_{camera,telemetry,lidar,wifi,calib}.py`,
  `build_ground_truth.py` = PLICP SLAM) and `load_dataset.py`.
- `golden/`: the golden-run ground truth (tag detection `tags_detect.py`, tag network
  `tag_ba2.py`, packaging `golden_package.py`, floor-plan registration), plus
  `alternatives/` and `diagnostics/`.

Raw rosbags are in the vault's `raw/` folder.
