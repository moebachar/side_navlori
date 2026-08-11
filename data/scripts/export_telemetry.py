"""Export telemetry modality from a TurtleBot3 Waffle Pi rosbag2 recording.

Topics (all 20 Hz, driver stamps all four from the same control tick):
  /imu            sensor_msgs/Imu          -> data/imu/imu.csv
  /magnetic_field sensor_msgs/MagneticField-> data/mag/mag.csv
  /odom           nav_msgs/Odometry        -> data/wheel_odom/odom.csv
  /joint_states   sensor_msgs/JointState   -> data/wheel_odom/joint_states.csv

Policy: raw async timestamps preserved exactly as published (integer ns,
no resampling, no interpolation, no rounding). t_ns = header stamp,
t_bag_ns = bag receive time, both epoch nanoseconds as int64.

/odom is RAW WHEEL ODOMETRY (dead reckoning, frame_id='odom',
child_frame_id='base_footprint'); pose is kept as-is.
/magnetic_field sensor is dead (all zeros) but exported for completeness.

Rerunnable: python export_telemetry.py
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

BAGDIR = Path(r"C:\Users\mbachar\side_navlori\dataset_full_01")
OUTROOT = Path(r"C:\Users\mbachar\side_navlori\data")

TOPICS = {"/imu", "/magnetic_field", "/odom", "/joint_states"}


def stamp_ns(msg):
    """Header stamp in epoch ns, pure integer arithmetic (never float)."""
    return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec


def main():
    imu_rows = []
    mag_rows = []
    odom_rows = []
    js_rows = []

    ts = get_typestore(Stores.ROS2_HUMBLE)
    with Reader(BAGDIR) as reader:
        for conn, t_bag_ns, raw in reader.messages():
            if conn.topic not in TOPICS:
                continue
            msg = ts.deserialize_cdr(raw, conn.msgtype)
            t_ns = stamp_ns(msg)

            if conn.topic == "/imu":
                av = msg.angular_velocity
                la = msg.linear_acceleration
                q = msg.orientation
                imu_rows.append(
                    (t_ns, t_bag_ns, av.x, av.y, av.z, la.x, la.y, la.z,
                     q.x, q.y, q.z, q.w)
                )
            elif conn.topic == "/magnetic_field":
                m = msg.magnetic_field
                mag_rows.append((t_ns, t_bag_ns, m.x, m.y, m.z))
            elif conn.topic == "/odom":
                p = msg.pose.pose.position
                q = msg.pose.pose.orientation
                yaw = math.atan2(
                    2.0 * (q.w * q.z + q.x * q.y),
                    1.0 - 2.0 * (q.y * q.y + q.z * q.z),
                )
                odom_rows.append(
                    (t_ns, t_bag_ns, p.x, p.y, yaw, q.z, q.w,
                     msg.twist.twist.linear.x, msg.twist.twist.angular.z)
                )
            elif conn.topic == "/joint_states":
                # Map wheel columns BY NAME per message, never by index.
                idx = {name: i for i, name in enumerate(msg.name)}
                li = idx["wheel_left_joint"]
                ri = idx["wheel_right_joint"]
                js_rows.append(
                    (t_ns, t_bag_ns,
                     msg.position[li], msg.position[ri],
                     msg.velocity[li], msg.velocity[ri])
                )

    def build(rows, float_cols):
        cols = ["t_ns", "t_bag_ns"] + float_cols
        df = pd.DataFrame(rows, columns=cols)
        df["t_ns"] = np.array([r[0] for r in rows], dtype=np.int64)
        df["t_bag_ns"] = np.array([r[1] for r in rows], dtype=np.int64)
        return df

    outputs = [
        (OUTROOT / "imu" / "imu.csv",
         build(imu_rows, ["wx", "wy", "wz", "ax", "ay", "az",
                          "qx", "qy", "qz", "qw"])),
        (OUTROOT / "mag" / "mag.csv",
         build(mag_rows, ["mx", "my", "mz"])),
        (OUTROOT / "wheel_odom" / "odom.csv",
         build(odom_rows, ["x", "y", "yaw", "qz", "qw", "v_lin", "w_ang"])),
        (OUTROOT / "wheel_odom" / "joint_states.csv",
         build(js_rows, ["left_pos_rad", "right_pos_rad",
                         "left_vel_radps", "right_vel_radps"])),
    ]

    for path, df in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"wrote {path}  rows={len(df)}")


if __name__ == "__main__":
    main()
