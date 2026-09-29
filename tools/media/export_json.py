import numpy as np
import json
import os

def main():
    out = {}
    runs = ['run3', 'run4', 'run5', 'run6', 'run7', 'run8']
    
    for run in runs:
        path = f"/mnt/x/side_navlori/data/{run}/ground_truth/map_points.npz"
        if os.path.exists(path):
            pts = np.load(path)["xy"]
            if len(pts) > 5000:
                pts = pts[::len(pts)//5000]
            out[run] = pts.tolist()
            
    with open("/mnt/x/side_navlori/points.js", "w") as f:
        f.write("const slam_runs = " + json.dumps(out) + ";\n")
        f.write("let slam_points = slam_runs['run3'];\n")
        
    print("Exported all runs to points.js")

if __name__ == "__main__":
    main()
