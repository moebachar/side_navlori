import sys
import numpy as np
import cv2

def main():
    scale = float(sys.argv[1])
    tx = float(sys.argv[2])
    ty = float(sys.argv[3])
    theta = float(sys.argv[4])
    
    floorplan = cv2.imread("/mnt/x/side_navlori/floorplan.jpg")
    map_data = np.load("/mnt/x/side_navlori/data/run3/ground_truth/map_points.npz")
    pts = map_data["xy"]
    
    if len(pts) > 5000:
        pts = pts[::len(pts)//5000]
        
    th = np.radians(theta)
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])
    pts_tf = (pts @ R.T) * scale + np.array([tx, ty])
    
    overlay = floorplan.copy()
    h, w = overlay.shape[:2]
    
    for x, y in pts_tf:
        ix, iy = int(x), int(y)
        if 0 <= ix < w and 0 <= iy < h:
            cv2.circle(overlay, (ix, iy), 3, (0, 0, 255), -1)
            
    # Crop around the points
    min_x = max(0, int(np.min(pts_tf[:, 0])) - 200)
    max_x = min(w, int(np.max(pts_tf[:, 0])) + 200)
    min_y = max(0, int(np.min(pts_tf[:, 1])) - 200)
    max_y = min(h, int(np.max(pts_tf[:, 1])) + 200)
    
    crop = overlay[min_y:max_y, min_x:max_x]
    
    cv2.imwrite("/mnt/x/side_navlori/overlay_crop.png", crop)
    print(f"Saved overlay_crop.png with shape {crop.shape[1]}x{crop.shape[0]}")

if __name__ == "__main__":
    main()
