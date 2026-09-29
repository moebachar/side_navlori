import cv2
import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
import sys

def main():
    bag_path = "/mnt/x/side_navlori/kalibr_test_5"
    topic = "/camera/image_raw"
    out_video = "/root/.gemini/antigravity/brain/852ac0c4-f018-4e40-bee5-f03622572d8f/artifacts/bag_video.mp4"
    
    ts = get_typestore(Stores.ROS2_HUMBLE)
    
    writer = None
    count = 0
    print("Reading bag and writing video...")
    with Reader(bag_path) as reader:
        for conn, timestamp, rawdata in reader.messages():
            if conn.topic == topic:
                count += 1
                if count % 2 != 0: continue # Process every 2nd frame (reduce size/time)
                
                msg = ts.deserialize_cdr(rawdata, conn.msgtype)
                
                if msg.encoding in ["mono8", "8UC1"]:
                    img = np.array(msg.data, dtype=np.uint8).reshape((msg.height, msg.width))
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif msg.encoding in ["rgb8", "bgr8", "8UC3"]:
                    img_bgr = np.array(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
                    if msg.encoding == "rgb8":
                        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
                else:
                    continue
                    
                if writer is None:
                    # 15 fps since we skip every 2nd frame (assuming ~30fps bag)
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(out_video, fourcc, 15.0, (msg.width, msg.height))
                
                writer.write(img_bgr)

    if writer is not None:
        writer.release()
        print(f"Video written to {out_video} with {count//2} frames.")
    else:
        print("Failed to write video.")

if __name__ == "__main__":
    main()
