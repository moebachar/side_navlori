# Ground-truth trajectory: method & validation

`gt_pose.csv` was **built offline from the lidar scans** in this dataset
(`data/lidar/scans.npz`). It is NOT the "ground truth" of the older
`dataset_turtlebot/` export — that file was a time-cropped copy of raw wheel
odometry (verified byte-identical to `/odom`; no SLAM ran during recording).
This one is an independent, drift-free reference.

Producer script: `data/scripts/build_ground_truth.py` (pure Python:
numpy/scipy/pandas; no ROS required). Rerunning it reproduces all files in
this folder deterministically.

## Method

1. **Deskew.** Each 360° revolution takes ~0.118 s while the robot moves
   (≤0.26 m/s, ≤0.61 rad/s). Every beam is timestamped
   (`t_ns + beam_time_offset_ns`) and re-projected into the scan-stamp frame
   using the wheel-odometry motion across the sweep.
2. **Beam angles** use the empirically validated convention
   `theta_i = angle_min + i*(angle_max-angle_min)/(n-1)` (see
   `data/lidar/` notes; the driver's `angle_increment` field is
   self-inconsistent by up to one beam width and is stored only as
   provenance).
3. **Pass 1 — incremental scan-to-map matching.** Odometry-seeded
   point-to-line ICP (PLICP) of each scan against a growing map
   (1 cm voxel-mean point cloud with scan-derived normals; Huber-robust
   Gauss-Newton on (x, y, yaw)).
4. **Refinement rounds (6 + 4 polish).** Freeze the map built from current
   poses, re-match every scan with a weak odometry motion-model prior
   (constrains only directions the scan geometry leaves free, e.g. the
   along-wall direction next to long parallel walls), rebuild, repeat.
   Converged to max pose change 6.3 mm (p50 0.95 mm) in the final round;
   polish rounds use a tighter 0.15 m gate / 0.03 m Huber scale.
5. **Gauge.** The map frame is defined as the odometry frame during the
   initial stationary segment: one final rigid transform aligns the mean GT
   pose of scans 0–24 (robot immobile) to the mean odometry pose at the same
   stamps. GT and wheel odometry are therefore directly comparable with no
   further alignment.

## Validation (from the final build)

| Check | Result |
|---|---|
| Scans matched | 1004 / 1004 (`quality == "good"` for all; no fallbacks) |
| Point-to-line RMSE (per scan) | median 9.1 mm, p95 13.2 mm, max 16.7 mm |
| Inlier fraction (<10 cm) | median 1.000, min 0.985 |
| Loop closure (last 40 scans matched to a map built ONLY from the first 25) | median 5.1 mm / 0.097°, max 12.7 mm / 0.485° (point-to-line residuals; an independent point-to-point re-check lands ~10% higher) |
| Stationary pose scatter, start (25 scans) | std 2.2 / 1.9 mm, yaw 0.195° |
| Stationary pose scatter, end (40 scans) | std 1.3 / 1.4 mm, yaw 0.225° |
| GT-vs-odom twist consistency (raw finite diff at scan stamps) | v corr 0.954 (RMSE 0.026 m/s), w corr 0.922 (RMSE 0.047 rad/s) |

**Estimated accuracy: ~2–3 mm (1σ) per pose in position, ~0.2° in yaw,
with global consistency across the loop of ~5 mm.** The stationary scatter is
a direct empirical measurement of per-pose noise; the loop-closure check
bounds accumulated drift over the full ~17.8 m lap. Note the trajectory only
revisits its start region (single loop, no mid-run overlaps within 0.4 m), so
revisit-based validation covers start-vs-end consistency; mid-run quality is
supported by the per-scan residuals, the odometry cross-check (no steps
> 8.3 mm in the GT-odom difference signal), and physical-plausibility bounds.

## What this exposes about wheel odometry

Raw wheel odometry (`data/wheel_odom/odom.csv` pose columns) drifts against
this reference by **24.8 cm / 3.6° at the end of the 117 s run** (max 25.9 cm
/ 4.3°), growing roughly linearly with distance travelled — see
`gt_overview.png`. Any model consuming odometry as an input and evaluating
against this GT now faces an honest problem.

## Files

- `gt_pose.csv` — `t_ns` (raw lidar header stamps, ~8.6 Hz mean, epoch ns),
  `x, y, yaw` (base_footprint in the map frame, REP-103, yaw in (-pi, pi]),
  per-scan quality columns (`n_matched`, `inlier_frac`, `rmse_m`, `quality`).
  GT exists ONLY at scan stamps — interpolate to other sensors' stamps
  yourself (see README loading snippet); nothing in this dataset is resampled.
- `map_points.npz` — `xy` (N×2 float64 map points), `normals` (N×2 float32).
  Useful for map-based localization baselines or visualization.
- `map.png` / `map.yaml` — RAW georeferenced 1 cm hit-count occupancy grid
  (grayscale, exactly `width_px × height_px`; pixel↔world mapping in
  `map.yaml`).
- `map_preview.png` — decorated view of the grid with axes + GT trajectory
  (human viewing only, not georeferenced).
- `gt_overview.png` — map + GT vs odometry trajectories, odometry drift curve.

## Caveats

- Camera stamps start ~0.19 s before the first scan stamp and extend ~0.03 s
  after the last; the first wifi window opens ~2.2 s before the first scan.
  Clamp or drop when interpolating GT there (the robot is stationary at both
  ends, so clamping is exact in practice).
- GT yaw is the base_footprint heading; the camera optical axis is aligned
  with it (see `calib/extrinsics.yaml`).
- z, roll, pitch are not observed (planar 2D GT).
