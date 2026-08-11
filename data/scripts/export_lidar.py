"""Export /scan (sensor_msgs/LaserScan) from a rosbag2 to a ragged/CSR npz + per-scan meta CSV.

Dataset: TurtleBot3 Waffle Pi, 360-degree LDS-02 lidar.
Bag:     C:/Users/mbachar/side_navlori/dataset_full_01 (rosbag2, sqlite3)
Output:  C:/Users/mbachar/side_navlori/data/lidar/scans.npz
         C:/Users/mbachar/side_navlori/data/lidar/scans_meta.csv

Notes / policy:
- Raw async timestamps. No resampling, no interpolation, no rounding of stamps.
  All stamps handled as int64 nanoseconds (never through float64).
- Beam count VARIES per scan (~247..287): ragged data stored flat with CSR
  offsets; scan k owns flat slice offsets[k]:offsets[k+1].
- ANGLE CONVENTION (empirically validated against seam continuity and
  stationary-scan consistency):
      theta_i = angle_min + i * (angle_max - angle_min) / (n - 1),  i = 0..n-1
  The published angle_increment field is inconsistent with the published
  endpoints (off by up to one beam width); it is stored as
  angle_increment_raw for provenance but NEVER used to build angles.
- ranges: NaN kept as NaN (invalid return).
- beam_time_offset_ns[i] = i * int(round(time_increment * 1e9)); beam i
  timestamp = t_ns + beam_time_offset_ns[i].
"""

from pathlib import Path

import numpy as np
import pandas as pd
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

BAGDIR = Path(r"C:\Users\mbachar\side_navlori\dataset_full_01")
OUTDIR = Path(r"C:\Users\mbachar\side_navlori\data\lidar")
TOPIC = "/scan"


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ts = get_typestore(Stores.ROS2_HUMBLE)

    scans = []
    with Reader(BAGDIR) as reader:
        for conn, t_bag_ns, raw in reader.messages():
            if conn.topic != TOPIC:
                continue
            msg = ts.deserialize_cdr(raw, conn.msgtype)
            h = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec  # int ns
            scans.append((h, int(t_bag_ns), msg))

    # Order by header stamp (stable); stamps themselves are untouched.
    scans.sort(key=lambda s: s[0])
    n_scans = len(scans)

    t_ns = np.array([s[0] for s in scans], dtype=np.int64)
    t_bag = np.array([s[1] for s in scans], dtype=np.int64)
    n_beams = np.array([len(s[2].ranges) for s in scans], dtype=np.int64)

    offsets = np.zeros(n_scans + 1, dtype=np.int64)
    np.cumsum(n_beams, out=offsets[1:])
    total = int(offsets[-1])

    ranges = np.empty(total, dtype=np.float32)
    intensities = np.empty(total, dtype=np.float32)
    angles = np.empty(total, dtype=np.float64)
    beam_time_offset_ns = np.empty(total, dtype=np.int64)

    angle_min = np.empty(n_scans, dtype=np.float64)
    angle_max = np.empty(n_scans, dtype=np.float64)
    angle_increment_raw = np.empty(n_scans, dtype=np.float64)
    time_increment_s = np.empty(n_scans, dtype=np.float64)
    scan_time_s = np.empty(n_scans, dtype=np.float64)
    range_min = np.empty(n_scans, dtype=np.float64)
    range_max = np.empty(n_scans, dtype=np.float64)
    n_finite = np.empty(n_scans, dtype=np.int64)

    for k, (_, _, msg) in enumerate(scans):
        a, b = int(offsets[k]), int(offsets[k + 1])
        n = b - a

        r = np.asarray(msg.ranges, dtype=np.float32)  # NaN kept as NaN
        ranges[a:b] = r
        intensities[a:b] = np.asarray(msg.intensities, dtype=np.float32)

        # Angle convention: endpoints-based, NOT angle_increment.
        amin = float(msg.angle_min)
        amax = float(msg.angle_max)
        i = np.arange(n, dtype=np.float64)
        angles[a:b] = amin + i * (amax - amin) / (n - 1)

        dt_ns = int(round(float(msg.time_increment) * 1e9))
        beam_time_offset_ns[a:b] = np.arange(n, dtype=np.int64) * dt_ns

        angle_min[k] = amin
        angle_max[k] = amax
        angle_increment_raw[k] = float(msg.angle_increment)
        time_increment_s[k] = float(msg.time_increment)
        scan_time_s[k] = float(msg.scan_time)
        range_min[k] = float(msg.range_min)
        range_max[k] = float(msg.range_max)
        n_finite[k] = int(np.sum(np.isfinite(r) & (r > 0.01)))

    np.savez_compressed(
        OUTDIR / "scans.npz",
        t_ns=t_ns,
        t_bag_ns=t_bag,
        offsets=offsets,
        ranges=ranges,
        intensities=intensities,
        angles=angles,
        beam_time_offset_ns=beam_time_offset_ns,
        angle_min=angle_min,
        angle_max=angle_max,
        angle_increment_raw=angle_increment_raw,
        time_increment_s=time_increment_s,
        scan_time_s=scan_time_s,
        range_min=range_min,
        range_max=range_max,
        n_beams=n_beams,
    )

    meta = pd.DataFrame(
        {
            "t_ns": t_ns,
            "t_bag_ns": t_bag,
            "n_beams": n_beams,
            "n_finite": n_finite,
            "angle_min": angle_min,
            "angle_max": angle_max,
            "angle_increment_raw": angle_increment_raw,
            "time_increment_s": time_increment_s,
            "scan_time_s": scan_time_s,
        }
    )
    meta.to_csv(OUTDIR / "scans_meta.csv", index=False)

    print(f"scans: {n_scans}")
    print(f"total beams: {total}")
    print(f"n_beams range: [{n_beams.min()}, {n_beams.max()}]")
    print(f"wrote {OUTDIR / 'scans.npz'}")
    print(f"wrote {OUTDIR / 'scans_meta.csv'}")


if __name__ == "__main__":
    main()
