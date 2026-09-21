# Data — layout, pipeline, and the new recordings

## Robot & modalities

**TurtleBot3 Waffle Pi**, ROS2 Humble, recorded as rosbag2 (sqlite3 `.db3`).
Topics: `/camera/image_raw/compressed` (imx219), `/imu` `/odom` `/joint_states`
`/magnetic_field` (OpenCR, one shared timer), `/scan` (LDS-02 lidar, ~9.6 Hz),
`/wifi/rssi` (custom `wifi_scanner`, JSON, ~1 scan / 4 s), `/tf` `/tf_static`
`/battery_state` `/cmd_vel`.

## Run folder layout (run1, run2 — the target shape)

```
data/runN/
  camera/    images/<t_ns>.jpg , camera.csv (t_ns,t_bag_ns,filename)
  imu/       imu.csv
  wheel_odom/ odom.csv , joint_states.csv
  mag/       mag.csv           (magnetometer is DEAD/all-zero, kept for completeness)
  wifi/      wifi.csv (long: t_start_ns,t_end_ns,scan_idx,bssid,ssid,rssi_dbm,...) , wifi_raw.jsonl
  lidar/     scans.npz (ragged CSR: offsets/ranges/angles/t_ns/...) , scans_meta.csv
  ground_truth/ gt_pose.csv (t_ns,x,y,yaw,n_matched,inlier_frac,rmse_m,quality) ,
                map_points.npz (xy,normals) , gt_overview.png , map.png/.yaml , method.md
  calib/     camera_intrinsics_nominal.yaml , extrinsics.yaml , robot.yaml
  README.md  (dataset card)
```

Timestamp policy everywhere: **raw async int64 nanoseconds**, no resampling.

## Loading the data (`data/scripts/load_dataset.py`)

```python
from load_dataset import Dataset
ds = Dataset("/content/data/run2")
ds.camera / ds.imu / ds.odom / ds.joints / ds.wifi / ds.gt / ds.lidar   # DataFrames / arrays
M, bssids, t_ns = ds.wifi_matrix(max_age_ms=4000, fill_dbm=-100)  # (n_scan, n_ap) RSSI matrix
gx, gy, yaw   = ds.gt_at(t_ns)                                    # GT pose interpolated at any stamps
```

## The pipeline (`data/scripts/`)

| script | reads | writes | path constants (hardcoded at top) |
|---|---|---|---|
| `export_camera.py` | bag | `<run>/camera/` | `BAGDIR`, `OUTROOT` (=`<run>`) |
| `export_telemetry.py` | bag | `<run>/imu,mag,wheel_odom/` | `BAGDIR`, `OUTROOT` (=`<run>`) |
| `export_lidar.py` | bag | `<run>/lidar/` | `BAGDIR`, `OUTDIR` (=`<run>/lidar`) |
| `export_wifi.py` | bag | `<run>/wifi/` | `BAGDIR`, `OUTDIR` (=`<run>/wifi`) |
| `export_calib.py` | bag | `<run>/calib/` | `BAG`, `OUT` (extrinsics from `/tf_static` + nominal intrinsics) |
| `build_ground_truth.py` | `<run>/lidar/scans.npz` + `<run>/wheel_odom/odom.csv` | `<run>/ground_truth/` | `ROOT` (=`<run>`) |

`build_ground_truth.py` is **PLICP scan-to-map SLAM** (deskew → point-to-line
ICP → voxel map + refinement); on run2 it hit ~7.8 mm median. It uses
`matplotlib.use("Agg")` internally.

### Driving the exporters per bag

They hardcode their input/output paths, so per bag you either edit the two
constant lines, or **patch them in memory and exec** (what the previous agent
did — lets you batch without touching the files):

```python
import re
def run_script(path, repls):          # repls: list of (regex, replacement) on the constant lines
    src = open(path).read()
    for pat, rep in repls:
        src, n = re.subn(pat, rep, src, count=1); assert n == 1, pat
    g = {"__name__": "__main__", "__file__": path}
    exec(compile(src, path, "exec"), g)

BAG = "/mnt/x/side_navlori/data/navlori_big"; RUN = "/mnt/x/side_navlori/data/run3"
run_script(".../export_camera.py",
           [(r'BAGDIR = Path\(r".*?"\)', f'BAGDIR = Path(r"{BAG}")'),
            (r'OUTROOT = Path\(r".*?"\)', f'OUTROOT = Path(r"{RUN}")')])
# lidar/wifi use OUTDIR = f"{RUN}/lidar" | f"{RUN}/wifi"; GT uses ROOT = r"{RUN}" (no Path()).
```
Run camera export to a **native path** (e.g. `/root/navlori/_tmp/...`) when
speed matters — writing thousands of JPEGs onto the `/mnt/x` mount is slow.

## The NEW recordings (2026-09-17) — status & the finalize task

Six runs recorded on the physical robot at **camera 640×480 @ 30 fps RGB888**,
IMU/odom ~144 Hz, scan ~9.6 Hz, wifi ~1/4 s. `/cmd_vel` is **absent** in all
bags (the controller published on another topic — harmless; motion is fully in
`/odom`+`/imu`+`/scan`).

| staging name | raw bag | duration | camera | wifi scans | SLAM med / p95 |
|---|---|---|---|---|---|
| `big`        | `data/navlori_big`        | 4.0 min | 7254 | 58 | 7.7 / 14.9 mm |
| `0917_152255`| `data/navlori_0917_152255`| 1.8 min | 3118 | 25 | 8.1 / 12.0 mm |
| `0917_152704`| `data/navlori_0917_152704`| 1.9 min | 3478 | 28 | 8.9 / 12.9 mm |
| `0917_151700`| `data/navlori_0917_151700`| 4.3 min | 7705 | 62 | 10.8 / 19.4 mm |
| `0917_151405`| `data/navlori_0917_151405`| 2.2 min | 3878 | 32 | 11.3 / 17.8 mm |
| `0917_150450`| `data/navlori_0917_150450`| 6.5 min | 11620 | 94 | 12.3 / 38.9 mm |

**`data/staging/<name>/`** already has **lidar + telemetry (imu/mag/wheel_odom)
+ ground_truth** built. It does **NOT** yet have **camera/** or **wifi/**.
Compare all six trajectories in **`data/staging/_compare_gt.png`**.

### Finalize procedure

1. **User picks keepers** (all SLAM'd fine; `0917_150450` has the most SLAM
   stress, `0917_152255` the least coverage — but it's a coverage/route call).
   Don't delete a run without the user.
2. For each keeper, **add the two missing modalities** into its staging folder:
   `export_camera.py` and `export_wifi.py` with `BAGDIR=data/navlori_<name>`,
   `OUTROOT`/`OUTDIR` pointing at `data/staging/<name>`. Also run
   `export_calib.py` for `extrinsics.yaml`.
3. **Promote:** rename `data/staging/<name>` → `data/run3` (then run4, …), write
   `data/runN/README.md` (dataset card: date, route, sensor rates, differences
   vs run1/run2 — note **640×480** camera and absent `/cmd_vel`), and **copy the
   folder into `/root/navlori/data/`** so the notebooks can load it (see
   ENVIRONMENT.md — two data copies).
4. **Delete the raw `data/navlori_*` bags** once export is verified (large,
   gitignored).

## Camera calibration (pending)

The new runs use camera **640×480** — but no on-robot calibration exists; runs
so far ship **nominal** intrinsics only (`camera_intrinsics_nominal.yaml`,
provenance = Pi-Cam-v2 reference, ~1–2 % focal accuracy — and the existing one
is for the **820×616** mode, not 640×480). To upgrade:

- A print-ready checkerboard is at **`calib/checkerboard_9x6_25mm_A4.pdf`**
  (9×6 inner corners, 25 mm squares — measure after printing at 100 %).
- Calibration is a **short dedicated session on the robot** (film the board with
  the camera at 640×480), then `cv2.calibrateCamera` → write a ROS camera_info
  YAML into each run's `calib/`. Intrinsics are fixed per camera+resolution, so
  it can be done any time as long as the camera stays 640×480. Until then, use a
  640×480-scaled nominal YAML.
- Extrinsics (sensor↔base TF) and the robot card come from `export_calib.py`
  (reads the bag's `/tf_static`), independent of intrinsics.
