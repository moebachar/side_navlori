# side_navlori TurtleBot dataset (`/data`)

One continuous indoor recording from a **TurtleBot3 Waffle Pi** driving a
single closed rectangular loop (~17.8 m GT path length, 116.8 s, one
clockwise lap, robot stationary for the first ~4 s and last ~9 s) in a
lab/office room (~7 × 4.5 m incl. alcoves) at CESI, France, on
**2026-07-24**.

Built directly from the raw ROS 2 bag `../dataset_full_01/` by the scripts in
`scripts/` — every byte here is reproducible from the bag. This directory
supersedes `../dataset_turtlebot/` (whose `ground_truth.csv` was actually a
copy of wheel odometry — see `ground_truth/method.md`).

**Timestamp policy: every modality keeps its own raw asynchronous
timestamps.** Nothing is resampled onto a common clock. All stamps are
**epoch nanoseconds (int64)**: `t_ns` = the sensor header stamp (use this),
`t_bag_ns` = bag receive time (transport latency analysis only; camera
arrives ~58 ms after capture, telemetry ~4-7 ms, lidar ~0.4 ms).

## Layout

```
data/
├── README.md                     <- this file
├── camera/
│   ├── camera.csv                3465 rows: t_ns, t_bag_ns, filename
│   └── images/<t_ns>.jpg         3465 JPEGs, 820x616 RGB, ~30 Hz (byte-exact from the bag)
├── imu/imu.csv                   2335 rows @ ~20 Hz: t_ns, t_bag_ns, wx wy wz, ax ay az, qx qy qz qw
├── mag/mag.csv                   2335 rows: DEAD SENSOR, all zeros (kept for completeness)
├── wheel_odom/
│   ├── odom.csv                  2335 rows @ ~20 Hz: t_ns, t_bag_ns, x y yaw qz qw, v_lin w_ang
│   └── joint_states.csv          2335 rows: left/right wheel angle [rad] + velocity [rad/s]
├── wifi/
│   ├── wifi.csv                  1429 rows (28 scans x ~51 APs): t_start_ns, t_end_ns, t_bag_ns,
│   │                             scan_idx, bssid, ssid, rssi_dbm, freq_mhz, last_seen_ms
│   └── wifi_raw.jsonl            the 28 raw scan JSONs, verbatim + t_bag_ns
├── lidar/
│   ├── scans.npz                 1004 ragged scans @ ~8.6 Hz mean (see below; authoritative over the csv)
│   └── scans_meta.csv            per-scan metadata (float cols rounded at the last ULP; npz is exact)
├── ground_truth/
│   ├── gt_pose.csv               1004 poses AT RAW LIDAR STAMPS: t_ns, x, y, yaw + quality cols
│   ├── method.md                 how GT was built + validation (READ THIS before evaluating)
│   ├── map.png + map.yaml        RAW georeferenced 1 cm occupancy grid (pixel<->world via map.yaml)
│   ├── map_preview.png           decorated view of the same grid (axes/trajectory; not georeferenced)
│   ├── map_points.npz, gt_overview.png
├── calib/
│   ├── extrinsics.yaml           static TF chain + all sensor poses in base_footprint
│   ├── camera_intrinsics_nominal.yaml   NOMINAL (uncalibrated!) Pi Cam v2 intrinsics
│   └── robot.yaml                platform description
└── scripts/                      the exporters + GT builder (provenance; rerunnable)
```

## Modalities

| Modality | Rate | Count | Stamp span (rel. s) | Notes |
|---|---|---|---|---|
| camera | ~30 Hz | 3465 | 0.000 – 116.847 | forward-facing, ~10 cm above floor; motion blur during turns (max 0.61 rad/s) |
| imu | ~20 Hz | 2335 | 0.098 – 116.866 | gyro + accel + orientation quaternion (OpenCR filter); covariances were all zeros → omitted |
| wheel_odom | ~20 Hz | 2335 | 0.098 – 116.866 | pose = RAW dead reckoning (drifts 25 cm / 3.6° over the run); `v_lin`,`w_ang` = commanded-frame twist; `qz,qw` give lossless yaw |
| joint_states | ~20 Hz | 2335 | 0.098 – 116.866 | raw wheel angles — rawest odometry source |
| mag | ~20 Hz | 2335 | 0.098 – 116.866 | all zeros (sensor never published data) — do not use |
| wifi | ~0.24 Hz | 28 scans / 1429 rows | −2.03 – 109.34 (t_start) | 56 unique BSSIDs, ~51 APs/scan, 2.4+5 GHz |
| lidar | ~8.6 Hz mean (median interval 120 ms) | 1004 | 0.186 – 116.816 | 360°, ragged 247–287 beams/rev, ranges+intensities |
| ground truth | at lidar stamps | 1004 | 0.186 – 116.816 | **σ ≈ 2–3 mm / 0.2°** (see method.md) |

(rel. s = seconds after the first camera stamp `1784885672193942473`.)

### Quirks you must know

1. **imu / mag / odom / joint_states share IDENTICAL `t_ns` values row-for-row.**
   The TB3 driver stamps all four from one ~20 Hz control tick. This is the
   raw truth from the bag, not an export artifact.
2. **Ground truth exists only at lidar stamps (~8.6 Hz mean).** Interpolate to
   camera/imu/wifi stamps yourself (snippet below). The dataset deliberately
   ships no resampled products.
3. **`wheel_odom/odom.csv` pose columns are dead reckoning** in the `odom`
   frame — after the gauge alignment (see method.md) they start aligned with
   GT and drift away. Using them as a model *input* is legitimate;
   evaluating against them as if they were truth is not.
4. **WiFi `last_seen_ms`** = age of the AP observation at scan read-out
   (iw-style). 19% of rows are cached entries older than the ~3.6 s scan
   window (up to 29 s → the robot may have moved up to ~5 m since that RSSI
   was actually measured). Filter `last_seen_ms <= 4000` for clean
   fingerprints. Also: one scan window spans ~3.6 s of robot motion (~0.7 m
   of path at cruise); `t_start_ns`/`t_end_ns` bound it — a reasonable
   single stamp is `t_end_ns - last_seen_ms*1e6` per row (beware: for stale
   cached rows this can precede the recording start by up to ~23 s).
5. **Hidden SSIDs**: 79 wifi rows have empty-string SSID. Read with
   `pd.read_csv(..., keep_default_na=False)` or they become NaN. Key on
   `bssid`, never on `ssid`.
6. **Lidar is ragged**: beam count varies 247–287 per revolution, and the
   per-beam angles must come from the stored `angles` array (convention
   `theta_i = angle_min + (i*(angle_max-angle_min))/(n-1)`, evaluated in
   exactly that order if you want bitwise reproduction; validated
   empirically — the driver's `angle_increment` is self-inconsistent by up
   to one beam width and is stored only as `angle_increment_raw`). ~15% of
   ranges are NaN (invalid returns), and **intensities are NaN at the same
   slots** — use nan-aware ops. A few spurious returns reach 65 m — gate to
   `0.05 < r < 12` for this room. `scans.npz` is authoritative;
   `scans_meta.csv` float columns round at the last ULP.
7. **Camera intrinsics are NOMINAL** (`calib/camera_intrinsics_nominal.yaml`):
   the camera was never calibrated; values are the standard raspicam v2
   820×616 reference calibration (good prior, ~1–2% focal accuracy).
8. Camera runs at a true ~29.7 fps mean (median dt 33.35 ms, occasional
   ~44 ms gaps; no dropped frames).
9. First wifi scan window starts 2.03 s BEFORE the first camera frame (the
   scanner was already running); the first ~0.19 s of camera frames precede
   the first lidar/GT stamp, and the last camera frames extend ~0.03 s past
   it. Handle edges when joining (the robot is stationary at both ends, so
   clamped GT interpolation is exact there).
10. **IMU orientation quaternion has an ARBITRARY heading datum** (filter
   initialization; a constant ~−51° offset from the map/odom frame in this
   recording). Use it for roll/pitch/relative yaw only — never as absolute
   heading (remember the magnetometer is dead).

## Frames & conventions (REP-103)

- 2D planar poses: x forward, y left, z up; yaw CCW about +z, radians,
  in (−π, π]; quaternions are [x, y, z, w].
- `map` frame (ground truth) ≡ odometry frame during the initial stationary
  segment — GT and odometry are directly comparable without alignment.
- Sensor extrinsics (from `/tf_static`, see `calib/extrinsics.yaml`):
  camera optical center at (+0.076, 0.000, 0.103) m in base_footprint,
  optical axis = robot x; lidar `base_scan` at (−0.064, 0.000, 0.132) m,
  yaw 0; imu at (0, 0, 0.078) m.
- Lidar beam angles are CCW in `base_scan` (verified against odometry
  rotation direction).

## Loading

```python
import numpy as np, pandas as pd

# float_precision='round_trip' preserves the shipped float64s exactly
# (the pandas default parser is ~1 ULP lossy)
D = r"C:\Users\mbachar\side_navlori\data"
rc = dict(float_precision="round_trip")
cam  = pd.read_csv(f"{D}/camera/camera.csv", **rc)
odom = pd.read_csv(f"{D}/wheel_odom/odom.csv", **rc)
imu  = pd.read_csv(f"{D}/imu/imu.csv", **rc)
wifi = pd.read_csv(f"{D}/wifi/wifi.csv", keep_default_na=False, **rc)
gt   = pd.read_csv(f"{D}/ground_truth/gt_pose.csv", **rc)

# ragged lidar
z = np.load(f"{D}/lidar/scans.npz")
k = 100
sl = slice(z["offsets"][k], z["offsets"][k + 1])
r, th = z["ranges"][sl], z["angles"][sl]          # NaN = invalid return

# interpolate GT to any stamps (e.g. camera) — position linear, yaw unwrapped
def gt_at(t_ns):
    t  = (np.asarray(t_ns, dtype=np.int64) - gt.t_ns[0]) / 1e9
    tg = (gt.t_ns.values - gt.t_ns[0]) / 1e9
    x  = np.interp(t, tg, gt.x)
    y  = np.interp(t, tg, gt.y)
    yaw = np.interp(t, tg, np.unwrap(gt.yaw))
    return x, y, np.arctan2(np.sin(yaw), np.cos(yaw))   # clamps at ends (robot stationary there)

xc, yc, yawc = gt_at(cam.t_ns)
```

See `scripts/load_dataset.py` for a fuller helper (per-image labels, wifi
fingerprint matrix, lidar point clouds).

## Provenance

- Source: `../dataset_full_01/` rosbag2 (sqlite3), 21,304 messages, read with
  the pure-Python `rosbags` package (typestore ROS2_HUMBLE) — no ROS install.
- Exporters: `scripts/export_camera.py`, `export_telemetry.py`,
  `export_wifi.py`, `export_lidar.py`, `export_calib.py`.
- Ground truth: `scripts/build_ground_truth.py` (method + validation in
  `ground_truth/method.md`).
- Environment: Python 3.13, numpy 2.5, pandas 3.0, scipy 1.18.
- Verified 2026-08-02: image bytes identical to bag payloads; stamps
  monotonic; counts match bag metadata exactly.
