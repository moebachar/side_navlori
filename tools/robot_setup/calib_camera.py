#!/usr/bin/env python3
import argparse, os, sys, numpy as np, cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--cols", type=int, required=True, help="inner corners per row")
    p.add_argument("--rows", type=int, required=True, help="inner corners per column")
    p.add_argument("--square", type=float, required=True, help="square edge in mm")
    p.add_argument("--n", type=int, default=40, help="target good views")
    p.add_argument("--topic", default="/camera/image_raw")
    p.add_argument("--move", type=float, default=30.0, help="min px move between kept views")
    return p.parse_args()

class Calib(Node):
    def __init__(self, a):
        super().__init__("headless_calib"); self.a = a
        self.pattern = (a.cols, a.rows)
        self.objp = np.zeros((a.cols * a.rows, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:a.cols, 0:a.rows].T.reshape(-1, 2)
        self.objp *= (a.square / 1000.0)
        self.objpoints = []; self.imgpoints = []; self.last = None; self.size = None; self.cov = set()
        self.flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
        self.crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.create_subscription(Image, a.topic, self.cb, 10)
        print(f"[calib] {a.cols}x{a.rows} board, {a.square} mm squares. Move it across the WHOLE frame + tilts. Ctrl-C to finish.")

    def cb(self, msg):
        if len(self.objpoints) >= self.a.n: return
        try:
            arr = np.frombuffer(bytes(msg.data), np.uint8)
            img = arr.reshape(msg.height, msg.step)[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
        except Exception as e:
            print("  frame decode err:", e); return
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY); self.size = gray.shape[::-1]
        ok, corners = cv2.findChessboardCorners(gray, self.pattern, self.flags)
        if not ok: return
        c = corners.reshape(-1, 2).mean(0)
        if self.last is not None and np.hypot(*(c - self.last)) < self.a.move: return
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.crit)
        self.objpoints.append(self.objp.copy()); self.imgpoints.append(corners); self.last = c
        self.cov.add((int(c[0] // (msg.width / 3)), int(c[1] // (msg.height / 3))))
        print(f"  kept {len(self.objpoints)}/{self.a.n}   frame-coverage {len(self.cov)}/9   board@({c[0]:.0f},{c[1]:.0f})")

def write_yaml(path, w, h, K, D):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    d = D.reshape(-1).tolist()
    while len(d) < 5: d.append(0.0)
    txt = f"""# Camera intrinsics for /camera/image_raw ({w}x{h}) - CALIBRATED on-robot (checkerboard, cv2).
image_width: {w}
image_height: {h}
camera_name: imx219_{w}x{h}
camera_matrix:
  rows: 3
  cols: 3
  data: [{fx:.8g}, 0, {cx:.8g}, 0, {fy:.8g}, {cy:.8g}, 0, 0, 1]
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: 5
  data: [{', '.join(f'{v:.8g}' for v in d[:5])}]
rectification_matrix:
  rows: 3
  cols: 3
  data: [1, 0, 0, 0, 1, 0, 0, 0, 1]
projection_matrix:
  rows: 3
  cols: 4
  data: [{fx:.8g}, 0, {cx:.8g}, 0, 0, {fy:.8g}, {cy:.8g}, 0, 0, 0, 1, 0]
"""
    os.makedirs(os.path.dirname(path), exist_ok=True); open(path, "w").write(txt)

def main():
    a = parse(); rclpy.init(); node = Calib(a)
    try:
        while rclpy.ok() and len(node.objpoints) < a.n:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    n = len(node.objpoints); print(f"\n[calib] collected {n} views; solving...")
    if n < 8:
        print("NOT ENOUGH (need >=8, want 20-40). Re-run and cover more of the frame."); rclpy.shutdown(); sys.exit(1)
    rms, K, D, _, _ = cv2.calibrateCamera(node.objpoints, node.imgpoints, node.size, None, None)
    w, h = node.size
    print(f"\nRMS reprojection error: {rms:.3f} px   (great <0.3, good <0.6, redo if >1.0)")
    print("K =\n", K, "\nD =", D.reshape(-1))
    url = os.path.expanduser(f"~/.ros/camera_info/imx219__base_soc_i2c0mux_i2c_1_imx219_10_{w}x{h}.yaml")
    ds  = os.path.expanduser(f"~/camera_intrinsics_{w}x{h}.yaml")
    write_yaml(url, w, h, K, D); write_yaml(ds, w, h, K, D)
    print(f"\nWROTE:\n  {url}\n  {ds}")
    rclpy.shutdown()

if __name__ == "__main__":
    main()
