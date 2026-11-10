# Rebuilding the TurtleBot3 SD card

Everything custom we put on the robot in September 2026, recovered from the work log after the SD card was
physically broken (2026-10). The datasets were not on the card; they are safe in the vault.

Robot: TurtleBot3 **Waffle Pi**, Raspberry Pi 4, OpenCR, lidar **LDS-02**, camera **imx219** (Pi camera v2).

## 1. Base system (~2 h)
1. Flash **Ubuntu Server 22.04 (64-bit) for Raspberry Pi** on a 32-64 GB A1/A2 microSD (Raspberry Pi Imager).
2. Follow the ROBOTIS TurtleBot3 e-Manual, **Humble** tab, "SBC setup" and "OpenCR setup":
   ROS 2 Humble, `~/turtlebot3_ws` built from source, `LDS_MODEL=LDS-02`, `TURTLEBOT3_MODEL=waffle_pi`.
3. In `~/.bashrc`: `export ROS_DOMAIN_ID=30`, `export TURTLEBOT3_MODEL=waffle_pi`, `export LDS_MODEL=LDS-02`.
4. Login user `ubuntu`, password `cesi` (as before).

## 2. WiFi: join the laptop hotspot `poppy_net`
```
sudo apt install wpasupplicant
sudo cp 60-wifi.yaml /etc/netplan/60-wifi.yaml      # or the one-line printf below
sudo chmod 600 /etc/netplan/60-wifi.yaml && sudo netplan apply
ip -4 addr show wlan0
```
One-line version: `printf 'network: {version: 2, wifis: {wlan0: {dhcp4: true, access-points: {poppy_net: {password: cesicesi}}}}}\n' | sudo tee /etc/netplan/60-wifi.yaml`

## 3. IMU / odometry at ~175 Hz (no firmware flash)
`~/turtlebot3_ws/src/turtlebot3/turtlebot3_node/src/turtlebot3.cpp`, around line 225:
`publish_timer(std::chrono::milliseconds(50))` -> **`milliseconds(5)`**, then
`cd ~/turtlebot3_ws && colcon build --packages-select turtlebot3_node`. Check: `ros2 topic hz /imu` (~175 Hz).

## 4. WiFi scanner package
```
cp -r wifi_scanner ~/turtlebot3_ws/src/
cd ~/turtlebot3_ws && colcon build --packages-select wifi_scanner && source install/setup.bash
echo "ubuntu ALL=(ALL) NOPASSWD: /usr/sbin/iw" | sudo tee /etc/sudoers.d/iw     # the node calls `sudo -n iw`
ros2 run wifi_scanner scanner          # publishes JSON on /wifi/rssi, ~0.25 Hz
```
The package files (`setup.py`, `package.xml`) were rewritten; `scanner.py` is the original code.
Scanning freezes SSH over wlan0 for ~3-4 s per scan: expected.

## 5. Camera
`sudo apt install ros-humble-camera-ros` (or build camera_ros). Default format NV21 breaks JPEG; always run:
`ros2 run camera_ros camera_node --ros-args -p format:=RGB888 -p width:=1280 -p height:=720`
Intrinsics: re-run `python3 calib_camera.py --cols 9 --rows 6 --square 24` (adapt to the board), or copy the
calibration from the vault (`navlori-data`, calib folder).

## 6. Teleop and recording
- `python3 nav_web_teleop.py`, then open `http://<pi-ip>:8080` on the laptop: hold arrows (combos = arcs), a/z speed.
- Record (5 SSH terminals): camera (step 5), `ros2 launch turtlebot3_bringup robot.launch.py` (wait for `Run!`),
  `ros2 run wifi_scanner scanner`, teleop, then
  `ros2 bag record -o ~/bags/navlori_$(date +%Y%m%d_%H%M%S) /imu /odom /joint_states /scan /camera/image_raw/compressed /camera/camera_info /wifi/rssi /tf /tf_static /battery_state /cmd_vel /magnetic_field`
- Lessons: charge the LiPo (at 11 V the motors drop out); restart bringup before every run and don't touch the
  robot for 5 s after `Run!` (gyro calibrates at start; runs 4 and 9 have a gyro bias).

## 7. Back up the finished card
On a Linux box: `sudo dd if=/dev/sdX of=turtlebot_sd.img bs=4M status=progress`, then compress and store it.
