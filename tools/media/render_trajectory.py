import sys
import numpy as np
import cv2
import pandas as pd

def main():
    scale = float(sys.argv[1])
    tx = float(sys.argv[2])
    ty = float(sys.argv[3])
    theta = float(sys.argv[4])
    run_name = sys.argv[5]
    mirror_x = sys.argv[6].lower() == 'true' if len(sys.argv) > 6 else False
    mirror_y = sys.argv[7].lower() == 'true' if len(sys.argv) > 7 else False
    
    floorplan = cv2.imread("/mnt/x/side_navlori/floorplan.jpg")
    poses_df = pd.read_csv(f"/mnt/x/side_navlori/data/{run_name}/ground_truth/gt_pose.csv")
    pts = poses_df[['x', 'y']].values
    
    if mirror_x: pts[:, 0] = -pts[:, 0]
    if mirror_y: pts[:, 1] = -pts[:, 1]
    
    th = np.radians(theta)
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])
    pts_tf = (pts @ R.T) * scale + np.array([tx, ty])
    
    overlay = floorplan.copy()
    h, w = overlay.shape[:2]
    
    for x, y in pts_tf:
        ix, iy = int(x), int(y)
        if 0 <= ix < w and 0 <= iy < h:
            cv2.circle(overlay, (ix, iy), 6, (255, 0, 0), -1)
            
    cv2.imwrite(f"/mnt/x/side_navlori/{run_name}_trajectory_overlay.png", overlay)
    print(f"Saved {run_name}_trajectory_overlay.png")

if __name__ == "__main__":
    main()
