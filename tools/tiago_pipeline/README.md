# TIAGo++ Strasbourg pipeline (2026-11 campaign)

ROS 1 bags of the PAL TIAGo++ (`tiago-201c`) → packaged runs in the vault (`navlori-data/robot/tiago_*`), same layout as
the TurtleBot golden runs. Runs in WSL with `/root/navlori/venv` (rosbags, OpenCV, scipy).

| step | script | output |
|---|---|---|
| 1 | `process_run.py <bag>` (the on-robot script, `raw/tiago_bags/scripts/`) | per-run lidar SLAM, CSVs, images (`staging_tiago/processed_onrobot/`) |
| 2 | `tags_detect.py` | AprilTag detections (tag36h11, 15 cm), tag centres in each run's map (`staging_tiago/tags/`) |
| 3 | `tag_network.py --ref run01 --exclude run_final,...` | one rigid transform per run + tag positions, held-out errors (`network.json`) |
| 4 | `merge_map.py` | merged map, wall agreement, occupancy image for the floor plan (`tiago_common/`) |
| 5 | `package.py` | `tiago_run_01..12`, `tiago_demo_1/2` in the golden-run layout |
| 6 | `make_videos.py` | `ground_truth/video_tags_slam.mp4` and `gt_on_map.png` per run |

Results of the first pass (2026-11-16): all 12 runs + 2 demo runs placed in one frame; held-out tag error 6-25 cm
median per run (13 cm over runs); walls of different runs agree to ~1 cm (runs 02/03 ~5 cm).
Next: register the merged map on the floor plan (LibreCAD) and calibrate the head camera.
