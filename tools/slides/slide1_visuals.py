#!/usr/bin/env python
# Slide 1 visuals: (a) the 6 target views as Kalibr's AprilGrid detector sees them, (b) the lens distortion Kalibr corrects.
import numpy as np, cv2, yaml
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SN = "/mnt/x/side_navlori"; S1 = SN + "/slides_progress/slide1_calibration"; X1 = SN + "/slides_progress/_extra/slide1_calibration"
d = np.load(f"{X1}/kalibr_detections.npz")
R, C, TR, TC = int(d["grid_rows"]), int(d["grid_cols"]), int(d["tag_rows"]), int(d["tag_cols"])
UP = 2
tiles = []
for k in range(1, 7):
    img = cv2.imread(f"{X1}/target_view_{k}.jpg"); img = cv2.resize(img, None, fx=UP, fy=UP, interpolation=cv2.INTER_CUBIC)
    c, idx = d[f"corners_{k}"], d[f"idx_{k}"].astype(int); P = {int(i): p for i, p in zip(idx, c)}
    ov = img.copy(); ntag = 0
    for tr in range(TR):
        for tc in range(TC):
            q = [(2 * tr) * C + 2 * tc, (2 * tr) * C + 2 * tc + 1, (2 * tr + 1) * C + 2 * tc + 1, (2 * tr + 1) * C + 2 * tc]
            to16 = lambda j: tuple(np.round((P[j] + 0.5) * UP * 16).astype(int))
            if all(j in P for j in q):          # tag with all 4 corners kept: filled box
                ntag += 1
                pts = np.array([(P[j] + 0.5) * UP for j in q], np.float32)
                cv2.fillPoly(ov, [np.round(pts * 16).astype(np.int32)], (80, 200, 60), cv2.LINE_AA, shift=4)
            for a, b in zip(q, q[1:] + q[:1]):   # outline every edge whose two corners were kept
                if a in P and b in P:
                    cv2.line(img, to16(a), to16(b), (40, 170, 20), 2, cv2.LINE_AA, shift=4)
    img = cv2.addWeighted(ov, 0.28, img, 0.72, 0)
    for p in c:
        cv2.circle(img, tuple(np.round((p + 0.5) * UP * 16).astype(int)), 4 * 16, (30, 30, 230), -1, cv2.LINE_AA, shift=4)
    cv2.rectangle(img, (0, 0), (img.shape[1], 46), (255, 255, 255), -1)
    cv2.putText(img, f"Kalibr: {len(c)}/{R*C} corners found", (14, 33), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
    tiles.append(img)
    print(f"view {k}: {ntag} tags, {len(c)} corners", flush=True)
pad = lambda t: cv2.copyMakeBorder(t, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=(255, 255, 255))
grid = np.vstack([np.hstack([pad(t) for t in tiles[:3]]), np.hstack([pad(t) for t in tiles[3:]])])
cv2.imwrite(f"{S1}/kalibr_detections_montage.jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
cv2.imwrite(f"{X1}/kalibr_detection_single_view.jpg", tiles[0], [cv2.IMWRITE_JPEG_QUALITY, 92])

# (b) distortion field: for every corrected pixel, where it sits in the raw image
cc = yaml.safe_load(open(f"{SN}/calib/kalibr_640x480/calib_cam-camchain.yaml"))["cam0"]
fx, fy, cx, cy = cc["intrinsics"]; K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]); D = np.array(cc["distortion_coeffs"])
W, H = cc["resolution"]
mx, my = cv2.initUndistortRectifyMap(K, D, None, K, (W, H), cv2.CV_32FC1)
u, v = np.meshgrid(np.arange(W), np.arange(H)); dx, dy = mx - u, my - v; mag = np.hypot(dx, dy)
print(f"distortion shift: max {mag.max():.1f} px, corners {mag[[0,0,-1,-1],[0,-1,0,-1]].round(1)}, centre {mag[H//2, W//2]:.2f} px, "
      f"median {np.median(mag):.1f} px", flush=True)
import pandas as pd
cam = pd.read_csv(f"{SN}/data/golden_run_11/camera/camera.csv")
frame = cv2.cvtColor(cv2.imread(f"{SN}/data/golden_run_11/camera/images/{cam.filename.iloc[len(cam)//2]}"), cv2.COLOR_BGR2RGB)
EX = 10
fig, ax = plt.subplots(figsize=(9, 6.9))
ax.imshow(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), cmap="gray", vmin=-120, vmax=330)
cs = ax.contour(mag, levels=[1, 2, 4, 6], colors="#6b7280", linewidths=1.0); ax.clabel(cs, fmt="%d px", fontsize=10)
s_ = 40; yy, xx = np.mgrid[s_ // 2:H:s_, s_ // 2:W:s_]
q = ax.quiver(xx, yy, dx[yy, xx] * EX, dy[yy, xx] * EX, mag[yy, xx], angles="xy", scale_units="xy", scale=1,
              cmap="plasma", clim=(0, np.ceil(mag.max())), width=0.0045)
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
cb = fig.colorbar(q, ax=ax, fraction=0.035, pad=0.02); cb.set_label("pixel shift (px)")
ax.text(8, H - 10, f"arrows x{EX} for visibility", fontsize=10, color="k", bbox=dict(fc="white", ec="none", alpha=0.8))
fig.savefig(f"{S1}/lens_distortion_map.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("SLIDE1_VISUALS_DONE", flush=True)
