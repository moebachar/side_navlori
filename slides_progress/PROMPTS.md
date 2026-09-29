# Prompts for the slide agent

Paste the **deck brief** once, then one slide prompt at a time together with that slide's folder.

---

## Deck brief (paste first)

You are designing a 7-slide progress update (16:9) for research supervisors. Order: Slide 0 (where we are) → Slides 1–5 → Slide 6 (what this dataset adds). They already know the project (indoor robot localization from WiFi, IMU, wheel odometry and camera) and the robot. The slides present **finished results**: what now exists and how good it is. Do not add sections about difficulties or to-dos.

Style rules for the whole deck:
- Clean, light background, lots of white space, one strong visual per slide. At most ~30 words of running text per slide; numbers, labels, tables and references don't count.
- Academic citations: numbered in square brackets in the slide body, e.g. [1], with the full references in a small grey footer on the same slide. Keep the numbering exactly as given in each slide prompt.
- Use my uploaded images and videos exactly as they are. Never redraw or re-plot my data images. **You** design everything else: diagrams, icons, arrows, callouts, number tiles and charts.
- One sensor colour code across all slides: WiFi = purple `#7b2cbf`, IMU = orange `#f77f00`, wheel odometry = green `#2a9d3f`, camera = blue `#1d4ed8`, lidar = grey `#6b7280`, ground truth = black, markers = gold `#f2b705`.
- Equations are typeset properly (LaTeX-style), small, and always sit next to the thing they describe.
- Every slide has a short statement-style title (a claim, not a topic word).

---

## Slide 0 — Where we are (opener)  ·  no files, you draw everything

**Title:** "Where we are: a complete, validated data-acquisition protocol"

Draw a horizontal 5-stage protocol, like the "Methods" figure of a dataset paper: numbered stages connected by arrows, each with an icon, the method, its reference and its measured output. All 5 stages carry a green ✓ (done). Add one faded 6th box at the far right, *"Benchmarking"*, without a tick.

| # | Stage | Method [ref] | Measured output |
|---|---|---|---|
| 1 | **Sensor calibration** | Kalibr camera calibration [1], following the OpenVINS procedure [2]; AprilGrid target | intrinsics K, reprojection error 0.4 px |
| 2 | **Marker survey** | 26 AprilTag markers (tag36h11, 15 cm) [3, 4] fixed at known floor-plan coordinates | building frame in metres |
| 3 | **Recording** | TurtleBot3 on ROS 2 [5]; 12 runs, 5 s standing still at start and end, markers in view along the path | camera 30 Hz · IMU 145 Hz · wheel odometry 145 Hz · 2-D lidar 10 Hz · WiFi 0.25 Hz |
| 4 | **Ground truth** | lidar scan-to-map point-to-line ICP [6] for distance, gyroscope for heading, bias from a Theil–Sen fit [7] or from standstill; markers measured by planar PnP (IPPE) [8] and adjusted jointly with a robust (Cauchy) cost [9] | pose at 10 Hz in building coordinates |
| 5 | **Validation** | leave-one-marker-out: hide one marker sighting, re-solve, measure the error there | **15 cm** median · walls of different runs agree within 6–7 cm |

- Under the stages, one line: *"Every stage ends with a measured quality number, not an assumption."*
- **Footer references (small, grey):**
  [1] P. Furgale, J. Rehder, R. Siegwart, "Unified temporal and spatial calibration for multi-sensor systems," IROS 2013, pp. 1280–1286.
  [2] P. Geneva, K. Eckenhoff, W. Lee, Y. Yang, G. Huang, "OpenVINS: A research platform for visual-inertial estimation," ICRA 2020, pp. 4666–4672.
  [3] E. Olson, "AprilTag: A robust and flexible visual fiducial system," ICRA 2011, pp. 3400–3407.
  [4] J. Wang, E. Olson, "AprilTag 2: Efficient and robust fiducial detection," IROS 2016, pp. 4193–4198.
  [5] S. Macenski, T. Foote, B. Gerkey, C. Lalancette, W. Woodall, "Robot Operating System 2: Design, architecture, and uses in the wild," Science Robotics 7(66), 2022.
  [6] A. Censi, "An ICP variant using a point-to-line metric," ICRA 2008, pp. 19–25.
  [7] P. K. Sen, "Estimates of the regression coefficient based on Kendall's tau," J. Amer. Statist. Assoc. 63(324), 1968, pp. 1379–1389.
  [8] T. Collins, A. Bartoli, "Infinitesimal plane-based pose estimation," IJCV 109(3), 2014, pp. 252–286.
  [9] B. Triggs, P. McLauchlan, R. Hartley, A. Fitzgibbon, "Bundle adjustment — a modern synthesis," Vision Algorithms: Theory and Practice, LNCS 1883, 2000, pp. 298–372.

---

## Slide 1 — Calibration  ·  folder `slide1_calibration`

**Title:** "Our camera, measured to the pixel with Kalibr"
**Subtitle:** *Kalibr: the reference calibration toolbox from ETH Zürich's Autonomous Systems Lab.*

Tell it as a left-to-right pipeline with **Kalibr as the engine in the middle**: *what Kalibr sees → what Kalibr solves → what we get*.
- **Left, "Kalibr sees":** `kalibr_detections_montage.jpg`, 6 of the calibration views with Kalibr's own detections drawn on: green boxes = AprilGrid tags, red dots = the corners it measured (165–190 of 192 per view). Caption: *AprilGrid target (8 × 6 tags, 3 cm), filmed by the robot's camera*.
- **Centre, "Kalibr solves":** draw a minimal pinhole-camera sketch yourself: a 3-D point, a ray through the optical centre, and the image plane, with `f` and the principal point `(cx, cy)` labelled. Under it, only the pinhole model:
  $$u = f_x \frac{X}{Z} + c_x \qquad v = f_y \frac{Y}{Z} + c_y$$
  Next to it, show the result as the camera matrix with the real numbers:
  $$K = \begin{pmatrix} 1275.7 & 0 & 338.7 \\ 0 & 1274.0 & 291.8 \\ 0 & 0 & 1 \end{pmatrix}\ \text{px}$$
- **Right, "we get":** `lens_distortion_map.png`, a real camera frame with arrows showing how far the lens displaces each pixel (arrows ×10). Caption: *lens distortion measured and removed: 0 px at the centre, ≤ 6.5 px at the edges*.
- **Bottom strip, 3 datasheet tiles:**
  - focal length **1275.7 px** (± 0.8 %)
  - reprojection error **0.4 px**
  - lens distortion **≤ 6.5 px**, corrected
- Footnote: *Kalibr · pinhole + radial-tangential model · 640 × 480*.

---

## Slide 2 — Floor plan & markers  ·  folder `slide2_floorplan_markers`

**Title:** "A floor plan the robot can read"

Make the building the hero.
- `floorplan_26_markers.png` fills about 70 % of the slide, full width. The gold squares are the markers.
- Draw a **magnifier bubble**: a circle around one gold square with a zoom cone opening onto `markers_seen_by_robot.jpg`, which shows 4 real camera frames with markers detected (red outlines).
- Three floating badges near the plan, each with a small icon you design:
  - **26 markers**
  - **15 cm AprilTags** (tag36h11)
  - **positions in plan coordinates (m)**
- One-line takeaway at the bottom: *"Each marker is a known point on the plan: when the camera sees one, the robot's position is pinned to the building."*

---

## Slide 3 — The dataset  ·  folder `slide3_dataset`

**Title:** "12 runs, 262 metres, one format"

Infographic layout, like a magazine data page.
- **Top half:** `all_12_paths_on_floorplan.png` (every colour is one run).
- **Bottom half:** a row of big-number tiles, each with a sensor icon in its sensor colour:
  - **25 min** of driving (runs from 5 m to 71 m)
  - **45,748** camera frames (640 × 480, 30 Hz)
  - **219,732** IMU samples (145 Hz)
  - **219,697** wheel-odometry samples (145 Hz)
  - **14,704** lidar scans (10 Hz)
  - **370** WiFi scans (~100 access points per run)
- **Right margin:** draw a small folder tree: `golden_run_N/` → `camera · imu · wheel_odom · lidar · wifi · ground_truth · calib`, captioned *same layout for every run, one loader*.
- Thin ribbon under the title: *Each run: 5 s still at start and end · slow drive · markers in view along the way.*

---

## Slide 4 — Ground truth  ·  folder `slide4_ground_truth`

**Title:** "Ground truth: where the robot really was"

Top third is a process diagram; bottom two-thirds are the proof.
- **Draw a flow diagram yourself** (icons + arrows):
  `Lidar SLAM` (how far it moved) + `Gyroscope` (which way it faced) → `Trajectory` → `+ Markers seen by the camera + marker map` → `Placed on the building plan` → **`Ground truth, 10 Hz`**.
- Put one equation under each stage:
  - heading: $\theta(t) = \int_0^t (\omega_z - b)\,d\tau$
  - position: $p_{k+1} = p_k + R(\theta_k)\,\Delta p_k^{\text{lidar}}$
  - placement: $\min_{R,\,t}\ \sum_j \lVert R\,p_j + t - T_j \rVert^2$, where $T_j$ = marker positions
- **Bottom left:** `example_run7_on_plan.png`, the longest run (71 m); colour = time.
- **Bottom right:** `gt_accuracy_blind_marker_test.png`. Above it, one big headline number: **15 cm**, with the caption *median error at a marker the solver never saw*.
- Small footnote: *walls mapped in different runs coincide within 6–7 cm*.

---

## Slide 5 — Visualisation  ·  folder `slide5_visualization`

**Title:** "One run, every sensor, one map"

Cinema layout: the video *is* the slide.
- `golden_run_10_replay.mp4` fills the slide under the title, edge to edge (16:9, autoplay, loop, muted). It is 50 s long: a 2.5-min run played at 3× speed.
  - **Left half:** the 4 sensor streams exactly as recorded: camera (markers boxed in gold when detected), IMU, wheel odometry, WiFi.
  - **Right half:** the lidar map drawing itself on the building floor plan, the robot's path, and each marker turning gold with a sight line the moment the camera sees it.
- Use `golden_run_10_replay_still.jpg` as the poster frame (shown before the video plays and in the PDF export).
- Just one caption line under the video: *"Left: the four sensors as recorded. Right: the map drawn live on the floor plan; a marker turns gold when the camera sees it."*
- No bullet points, no extra labels on top of the video.

---

## Slide 6 — What this dataset adds (closer)  ·  no files, you build the table

**Title:** "What this dataset adds to the public ones"

A clean comparison table in the style of a dataset paper: one row per dataset, ✓ / – cells, our row last and highlighted (light gold background, bold). Colour the sensor column headers with the deck's sensor colours.

| Dataset | Platform | WiFi | IMU | Wheel odom. | Camera | Lidar | Ground truth | Scale |
|---|---|---|---|---|---|---|---|---|
| UJIIndoorLoc [1] | smartphones (25 devices) | ✓ | – | – | – | – | predefined reference points | 21k fingerprints, 3 buildings |
| Indoor Location Competition 2.0 [2] | hand-held smartphone | ✓ | ✓ | – | – | – | waypoints labelled on the floor plan by the surveyor | hundreds of buildings |
| IMUWiFine [3] | smartphone | ✓ | ✓ | – | – | – | fine-grained reference points | 120 trajectories, 14.2 km, 10.4 h |
| RoNIN [4] | smartphone | – | ✓ | – | – | – | visual-inertial tracking phone on a harness | 42.7 h, 100 people |
| OpenLORIS-Scene [5] | wheeled robot | – | ✓ | ✓ | ✓ | ✓ | motion capture (office) / offline lidar SLAM | 5 scenes |
| LuViRA [6] | service robot | 5G radio, not WiFi | ✓ | – | ✓ | – | motion capture (0.5 mm), one lab room | 89 trajectories of 20–50 s |
| **Ours** | **TurtleBot3 robot** | **✓** | **✓** | **✓** | **✓** | **✓** | **lidar SLAM + gyroscope, placed on 26 surveyed markers; accuracy measured on held-out markers (15 cm median)** | **12 runs, 25 min, 262 m, one building floor** |

- Takeaway line under the table: *"Among the datasets compared here, the only one pairing WiFi fingerprints with a robot's IMU, wheel odometry, camera and lidar across a whole building floor, with ground-truth accuracy that is measured, not assumed."*
- **Footer references (small, grey):**
  [1] J. Torres-Sospedra et al., "UJIIndoorLoc: A new multi-building and multi-floor database for WLAN fingerprint-based indoor localization problems," IPIN 2014, pp. 261–270.
  [2] Y. Shu, Q. Xu, J. Liu, R. Roy Choudhury, N. Trigoni, V. Bahl, "Indoor Location Competition 2.0 Dataset," Microsoft Research, 2021.
  [3] M. Nurpeiissov, A. Kuzdeuov, A. Assylkhanov, Y. Khassanov, H. A. Varol, "End-to-end sequential indoor localization using smartphone inertial sensors and WiFi," IEEE/SICE SII 2022, pp. 566–571.
  [4] S. Herath, H. Yan, Y. Furukawa, "RoNIN: Robust neural inertial navigation in the wild: Benchmark, evaluations, & new methods," ICRA 2020.
  [5] X. Shi et al., "Are we ready for service robots? The OpenLORIS-Scene datasets for lifelong SLAM," ICRA 2020.
  [6] I. Yaman et al., "The LuViRA dataset: Synchronized vision, radio, and audio sensors for indoor localization," ICRA 2024.
