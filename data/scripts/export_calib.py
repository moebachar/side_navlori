"""Export calibration files for the side_navlori TurtleBot dataset.

- extrinsics.yaml: static TF chain from the bag (full precision) + every sensor
  frame flattened to base_footprint via proper quaternion composition.
- camera_intrinsics_nominal.yaml: NOMINAL Pi Camera v2 intrinsics for the
  820x616 mode. The bag's /camera/camera_info is empty (all zeros, never
  calibrated). 820x616 is the full-FoV 1/4-scale mode of the IMX219
  (3280x2464); the UbiquityRobotics raspicam_node reference calibration
  camerav2_410x308.yaml is the same optical configuration at 1/8 scale, so
  fx, fy, cx, cy are scaled by exactly 2 and distortion is unchanged.
- robot.yaml: platform description with nominal ROBOTIS TurtleBot3 specs.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

BAG = r"C:\Users\mbachar\side_navlori\dataset_full_01"
OUT = r"C:\Users\mbachar\side_navlori\data\calib"
import os
os.makedirs(OUT, exist_ok=True)

ts = get_typestore(Stores.ROS2_HUMBLE)
static = {}   # (parent, child) -> (xyz, quat xyzw)
wheels = {}
with Reader(BAG) as reader:
    for conn, t, raw in reader.messages():
        if conn.topic == "/tf_static":
            msg = ts.deserialize_cdr(raw, conn.msgtype)
            for tr in msg.transforms:
                tl, q = tr.transform.translation, tr.transform.rotation
                static[(tr.header.frame_id, tr.child_frame_id)] = (
                    np.array([tl.x, tl.y, tl.z]), np.array([q.x, q.y, q.z, q.w]))
        elif conn.topic == "/tf" and len(wheels) < 2:
            msg = ts.deserialize_cdr(raw, conn.msgtype)
            for tr in msg.transforms:
                if "wheel" in tr.child_frame_id and tr.child_frame_id not in wheels:
                    tl = tr.transform.translation
                    wheels[tr.child_frame_id] = np.array([tl.x, tl.y, tl.z])
        if len(static) >= 8 and len(wheels) >= 2:
            break

def compose(chain):
    """Compose transforms along a parent->child chain; returns (xyz, quat_xyzw)."""
    p = np.zeros(3)
    r = Rotation.identity()
    for key in chain:
        xyz, q = static[key]
        p = p + r.apply(xyz)
        r = r * Rotation.from_quat(q)
    return p, r.as_quat()

chains = {
    "camera_rgb_optical_frame": [("base_footprint", "base_link"), ("base_link", "camera_link"),
                                 ("camera_link", "camera_rgb_frame"),
                                 ("camera_rgb_frame", "camera_rgb_optical_frame")],
    "camera_rgb_frame": [("base_footprint", "base_link"), ("base_link", "camera_link"),
                         ("camera_link", "camera_rgb_frame")],
    "base_scan": [("base_footprint", "base_link"), ("base_link", "base_scan")],
    "imu_link": [("base_footprint", "base_link"), ("base_link", "imu_link")],
    "base_link": [("base_footprint", "base_link")],
}

def fmt_list(a):
    return "[" + ", ".join(f"{v:.10g}" for v in a) + "]"

lines = []
lines.append("# Sensor extrinsics for the side_navlori TurtleBot3 Waffle Pi dataset.")
lines.append("# Source: /tf_static of rosbag dataset_full_01 (values verbatim, full precision).")
lines.append("# Conventions: REP-103. base_footprint sits on the ground plane under the robot")
lines.append("# center; x forward, y left, z up; quaternions are [x, y, z, w].")
lines.append("# 'flattened' entries are the composed transform base_footprint -> frame.")
lines.append("")
lines.append("raw_static_tf:")
for (parent, child), (xyz, q) in sorted(static.items()):
    lines.append(f"  - parent: {parent}")
    lines.append(f"    child: {child}")
    lines.append(f"    translation_m: {fmt_list(xyz)}")
    lines.append(f"    quaternion_xyzw: {fmt_list(q)}")
lines.append("")
lines.append("flattened_to_base_footprint:")
for name, chain in chains.items():
    p, q = compose(chain)
    lines.append(f"  {name}:")
    lines.append(f"    translation_m: {fmt_list(p)}")
    lines.append(f"    quaternion_xyzw: {fmt_list(q)}")
    if name == "camera_rgb_optical_frame":
        lines.append("    # optical convention: z forward (viewing axis), x right, y down.")
        lines.append("    # NOTE: the URDF quaternion is a slightly rounded version of the exact")
        lines.append("    # optical rotation (-0.5, 0.5, -0.5, 0.5); the difference is ~0.065 deg.")
    if name == "base_scan":
        lines.append("    # lidar frame: x forward, angles CCW about +z (verified empirically).")
lines.append("")
lines.append("wheels:")
lines.append(f"  # from first /tf wheel transforms (URDF joint origins)")
for name, xyz in sorted(wheels.items()):
    lines.append(f"  {name}:")
    lines.append(f"    translation_in_base_link_m: {fmt_list(xyz)}")
lines.append("  nominal_wheel_radius_m: 0.033      # ROBOTIS TB3 spec (not calibrated)")
sep = abs(wheels.get("wheel_left_link", [0, 0.1435, 0])[1] - wheels.get("wheel_right_link", [0, -0.1435, 0])[1])
lines.append(f"  wheel_separation_m: {sep:.10g}       # |y_left - y_right| from TF")

with open(os.path.join(OUT, "extrinsics.yaml"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

# --- nominal intrinsics, scaled x2 from camerav2_410x308.yaml ---
ref = dict(
    fx=322.0704122808738, fy=320.8673986158544, cx=199.2680620421962, cy=155.2533082600705,
    D=[0.1639958233797625, -0.271840030972792, 0.001055841660100477, -0.00166555973740089, 0.0],
)
s = 2.0
intr = f"""# NOMINAL camera intrinsics for /camera/image_raw (820x616) — NOT calibrated
# on this robot. The bag's /camera/camera_info is empty (all zeros): the camera
# was never calibrated during collection.
#
# Provenance: UbiquityRobotics/raspicam_node camera_info/camerav2_410x308.yaml
# (Raspberry Pi Camera v2, IMX219, full-FoV binned mode). 820x616 is the same
# optical mode at exactly 2x resolution, so fx, fy, cx, cy scale by 2 and
# plumb_bob distortion coefficients are unchanged.
# Treat as a good prior (~1-2% focal accuracy across units), not a calibration.
image_width: 820
image_height: 616
camera_name: raspicam_v2_820x616_nominal
camera_matrix:
  rows: 3
  cols: 3
  data: [{ref['fx']*s:.10g}, 0, {ref['cx']*s:.10g}, 0, {ref['fy']*s:.10g}, {ref['cy']*s:.10g}, 0, 0, 1]
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: 5
  data: [{', '.join(f"{d:.10g}" for d in ref['D'])}]
rectification_matrix:
  rows: 3
  cols: 3
  data: [1, 0, 0, 0, 1, 0, 0, 0, 1]
fov_deg_approx: {{horizontal: {2*np.degrees(np.arctan(410/(ref['fx']*s))):.1f}, vertical: {2*np.degrees(np.arctan(308/(ref['fy']*s))):.1f}}}
"""
with open(os.path.join(OUT, "camera_intrinsics_nominal.yaml"), "w", encoding="utf-8") as f:
    f.write(intr)

robot = """# Platform description — side_navlori dataset
platform: TurtleBot3 Waffle Pi
manufacturer: ROBOTIS
onboard_computer: Raspberry Pi (ROS 2, rosbag2 recording)
mcu: OpenCR 1.0 (publishes /imu, /odom, /joint_states, /magnetic_field at a
  common ~20 Hz tick with identical header stamps)
sensors:
  camera:
    model: Raspberry Pi Camera v2 (Sony IMX219)
    mode: 820x616 @ ~30 Hz, JPEG compressed
    mount: front of robot, ~0.103 m above ground, forward-facing (see calib/extrinsics.yaml)
  lidar:
    model: LDS-02 (360 deg, ~8.6 Hz mean / median interval 120 ms,
      variable 247-287 beams/rev)
    note: recorded for ground-truth generation; angles CCW, frame base_scan
  imu:
    model: OpenCR onboard IMU (gyro + accel + orientation filter)
    rate: ~20 Hz as recorded
    note: the orientation quaternion has an ARBITRARY heading datum (filter
      initialization; ~-51 deg from the map/odom frame in this recording,
      constant throughout). Use it for roll/pitch/relative yaw only.
  magnetometer:
    status: DEAD - all samples exactly zero in this recording; do not use
  wheel_encoders:
    topics: [/odom, /joint_states]
    note: /odom yaw incorporates the OpenCR gyro (TB3 firmware fuses it);
      /odom pose is dead reckoning in the 'odom' frame
  wifi:
    tool: custom scanner publishing /wifi/rssi JSON (iw-style scan dumps)
    rate: one scan per ~4.1 s; scan window ~3.6 s while robot moves
nominal_specs:
  wheel_radius_m: 0.033
  wheel_separation_m: 0.287   # ROBOTIS spec; the TF-derived value in
                              # extrinsics.yaml is 0.288 (|y_left - y_right|)
  max_lin_vel_mps: 0.26
  max_ang_vel_radps: 1.82
collection:
  site: indoor lab/office room (~7 x 4.5 m incl. alcoves), CESI campus, France
  date_utc: 2026-07-24 (epoch-ns timestamps in all files)
  trajectory: single closed rectangular loop (GT path length ~17.8 m,
    trajectory bounding box 5.9 x 3.4 m), ~117 s, one lap CW,
    stationary for the first ~4 s and last ~9 s
"""
with open(os.path.join(OUT, "robot.yaml"), "w", encoding="utf-8") as f:
    f.write(robot)

print("wrote", os.listdir(OUT))
print("\n--- extrinsics.yaml flattened section preview ---")
print("\n".join(lines[lines.index("flattened_to_base_footprint:"):]))
