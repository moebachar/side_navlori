import cv2
import numpy as np
import pandas as pd
import os

runs = {
    'run3': {'scale': 244.0, 'tx': 12325.0, 'ty': 3865.0, 'theta': -66.8, 'flip_x': True, 'flip_y': False, 'color': (255, 0, 0)},     # Blue
    'run4': {'scale': 232.9, 'tx': 3549.0, 'ty': 7672.0, 'theta': -32.5, 'flip_x': True, 'flip_y': False, 'color': (0, 200, 0)},       # Green
    'run5': {'scale': 236.7, 'tx': 2967.0, 'ty': -1822.0, 'theta': -161.6, 'flip_x': True, 'flip_y': False, 'color': (0, 165, 255)},    # Orange
    'run7': {'scale': 239.5, 'tx': 11638.0, 'ty': 4957.0, 'theta': 58.7, 'flip_x': True, 'flip_y': False, 'color': (255, 0, 255)}      # Purple
}

def draw_point_with_text(img, point, text, color, is_start):
    # point is (x, y)
    ix, iy = int(point[0]), int(point[1])
    
    marker_color = (0, 255, 0) if is_start else (0, 0, 255) # Green for start, Red for end
    radius = 15
    thickness = -1 if is_start else 4
    
    if is_start:
        cv2.circle(img, (ix, iy), radius, marker_color, thickness)
    else:
        # draw a square for end
        cv2.rectangle(img, (ix - radius, iy - radius), (ix + radius, iy + radius), marker_color, thickness)
        
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_thickness = 4
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    
    text_x = ix + 20
    text_y = iy + 10
    
    # White background for text for readability
    cv2.rectangle(img, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), (255, 255, 255), -1)
    cv2.putText(img, text, (text_x, text_y), font, font_scale, color, font_thickness)
    

def main():
    print("Loading floorplan...")
    floorplan = cv2.imread("/mnt/x/side_navlori/floorplan.jpg")
    
    for run_name, params in runs.items():
        print(f"Processing {run_name}...")
        csv_path = f"/mnt/x/side_navlori/data/{run_name}/ground_truth/gt_pose.csv"
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found.")
            continue
            
        poses_df = pd.read_csv(csv_path)
        pts = poses_df[['x', 'y']].values
        
        if params['flip_x']: pts[:, 0] = -pts[:, 0]
        if params['flip_y']: pts[:, 1] = -pts[:, 1]
        
        th = np.radians(params['theta'])
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s], [s, c]])
        
        pts_tf = (pts @ R.T) * params['scale'] + np.array([params['tx'], params['ty']])
        pts_tf = np.int32(pts_tf)
        
        # Draw path using polylines
        cv2.polylines(floorplan, [pts_tf], isClosed=False, color=params['color'], thickness=6)
        
        # Draw Start and End
        if len(pts_tf) > 0:
            start_pt = pts_tf[0]
            end_pt = pts_tf[-1]
            draw_point_with_text(floorplan, start_pt, f"{run_name.upper()} Start", params['color'], is_start=True)
            draw_point_with_text(floorplan, end_pt, f"{run_name.upper()} End", params['color'], is_start=False)

    out_path = "/mnt/x/side_navlori/all_runs_overlay.png"
    cv2.imwrite(out_path, floorplan)
    print(f"Saved combined overlay to {out_path}")

if __name__ == "__main__":
    main()
