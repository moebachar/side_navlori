# side_navlori

Experimental sandbox for the **navlori** robot-localization project — a free
space to test new ideas on a real TurtleBot3 Waffle Pi recording, away from the
main project's assumptions.

- `side_navlori.ipynb` — the working notebook (developed here, run on Colab).
- `data/` — the canonical dataset (camera / imu / wheel odometry / wifi / lidar
  / lidar-built ground truth). See `data/README.md` for the full dataset card.
  The heavy payloads (`data/camera/images/`, raw rosbag) are excluded from git —
  the complete dataset travels to Colab as `data.zip` on Google Drive.
- `data/scripts/` — exporters + ground-truth builder (everything is
  reproducible from the raw bag `dataset_full_01/`, not tracked here).
