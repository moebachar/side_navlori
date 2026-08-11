"""Convenience loaders for the side_navlori TurtleBot dataset.

Usage:
    from load_dataset import Dataset
    ds = Dataset(r"C:\\Users\\mbachar\\side_navlori\\data")
    ds.camera            # DataFrame: t_ns, t_bag_ns, filename, x, y, yaw (GT-interpolated)
    ds.gt                # DataFrame: raw GT at lidar stamps
    ds.lidar_points(k)   # (n,2) points of scan k in base_scan frame (NaN-filtered)
    ds.wifi_matrix()     # (28, n_ap) RSSI fingerprint matrix + stamps + bssid list

All timestamps stay raw (epoch ns, int64). GT interpolation is the ONLY
derived quantity, computed on the fly and clearly separated from raw columns.
"""
import os
import numpy as np
import pandas as pd


class Dataset:
    def __init__(self, root):
        self.root = root
        # round_trip: the default pandas float parser is ~1 ULP lossy;
        # this preserves the shipped float64 values exactly.
        rc = dict(float_precision="round_trip")
        self.camera = pd.read_csv(os.path.join(root, "camera", "camera.csv"), **rc)
        self.imu = pd.read_csv(os.path.join(root, "imu", "imu.csv"), **rc)
        self.odom = pd.read_csv(os.path.join(root, "wheel_odom", "odom.csv"), **rc)
        self.joints = pd.read_csv(os.path.join(root, "wheel_odom", "joint_states.csv"), **rc)
        self.wifi = pd.read_csv(os.path.join(root, "wifi", "wifi.csv"),
                                keep_default_na=False, **rc)
        self.gt = pd.read_csv(os.path.join(root, "ground_truth", "gt_pose.csv"), **rc)
        self._z = np.load(os.path.join(root, "lidar", "scans.npz"))
        # convenience: GT pose at camera stamps (clamped at ends; robot stationary there)
        x, y, yaw = self.gt_at(self.camera.t_ns.values)
        self.camera = self.camera.assign(x=x, y=y, yaw=yaw)

    def gt_at(self, t_ns):
        """Interpolate GT (x, y, yaw) at arbitrary epoch-ns stamps."""
        t0 = int(self.gt.t_ns.iloc[0])
        t = (np.asarray(t_ns, dtype=np.int64) - t0) / 1e9
        tg = (self.gt.t_ns.values.astype(np.int64) - t0) / 1e9
        x = np.interp(t, tg, self.gt.x.values)
        y = np.interp(t, tg, self.gt.y.values)
        yaw = np.interp(t, tg, np.unwrap(self.gt.yaw.values))
        return x, y, np.arctan2(np.sin(yaw), np.cos(yaw))

    def image_path(self, i):
        return os.path.join(self.root, "camera", "images", self.camera.filename.iloc[i])

    def lidar_scan(self, k):
        """Raw arrays of scan k: dict(t_ns, ranges, angles, intensities, beam_time_offset_ns)."""
        z = self._z
        sl = slice(int(z["offsets"][k]), int(z["offsets"][k + 1]))
        return dict(t_ns=int(z["t_ns"][k]), ranges=z["ranges"][sl], angles=z["angles"][sl],
                    intensities=z["intensities"][sl],
                    beam_time_offset_ns=z["beam_time_offset_ns"][sl])

    def lidar_points(self, k, r_min=0.05, r_max=12.0):
        """(n,2) valid points of scan k in the base_scan frame."""
        s = self.lidar_scan(k)
        r, th = s["ranges"].astype(np.float64), s["angles"]
        ok = np.isfinite(r) & (r > r_min) & (r < r_max)
        return np.c_[r[ok] * np.cos(th[ok]), r[ok] * np.sin(th[ok])]

    def wifi_matrix(self, max_age_ms=4000.0, fill_dbm=-100.0):
        """RSSI fingerprint matrix.

        Returns (M, bssids, t_ns): M is (n_scans, n_bssids) float with
        fill_dbm where an AP was absent (or its cached entry older than
        max_age_ms); t_ns is the scan-end stamp per row.
        """
        w = self.wifi[self.wifi.last_seen_ms <= max_age_ms]
        bssids = sorted(w.bssid.unique())
        col = {b: i for i, b in enumerate(bssids)}
        scans = sorted(w.scan_idx.unique())
        row = {s: i for i, s in enumerate(scans)}
        M = np.full((len(scans), len(bssids)), fill_dbm)
        for r in w.itertuples():
            M[row[r.scan_idx], col[r.bssid]] = r.rssi_dbm
        t_end = w.groupby("scan_idx").t_end_ns.first().loc[scans].values
        return M, bssids, t_end


if __name__ == "__main__":
    ds = Dataset(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print("camera:", len(ds.camera), "rows; GT-labelled columns:", list(ds.camera.columns))
    print("gt span:", (ds.gt.t_ns.iloc[-1] - ds.gt.t_ns.iloc[0]) / 1e9, "s")
    M, bssids, t = ds.wifi_matrix()
    print("wifi matrix:", M.shape, "APs:", len(bssids))
    print("scan 100 points:", ds.lidar_points(100).shape)
