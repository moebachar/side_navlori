import sys
import numpy as np
import cv2

def main():
    scale = float(sys.argv[1])
    tx = float(sys.argv[2])
    ty = float(sys.argv[3])
    theta = float(sys.argv[4])
    
    floorplan = cv2.imread("/mnt/x/side_navlori/floorplan.jpg")
    if floorplan is None:
        print("Could not load floorplan.")
        sys.exit(1)
        
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
            cv2.circle(overlay, (ix, iy), 2, (0, 0, 255), -1)
            
    cv2.imwrite("/mnt/x/side_navlori/overlay.png", overlay)
    print(f"Saved overlay.png with shape {w}x{h}")

if __name__ == "__main__":
    main()
