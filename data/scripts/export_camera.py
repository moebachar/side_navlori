"""Export /camera/image_raw/compressed from the rosbag2 to data/camera/.

- Writes each CompressedImage's JPEG bytes verbatim to data/camera/images/<t_ns>.jpg
  (t_ns = header stamp in epoch nanoseconds, integer arithmetic only).
- Writes data/camera/camera.csv with columns: t_ns, t_bag_ns, filename.
- Policy: raw async timestamps. No resampling, no interpolation, no rounding.

Rerunnable: overwrites camera.csv and image files deterministically.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

BAGDIR = Path(r"C:\Users\mbachar\side_navlori\dataset_full_01")
OUTROOT = Path(r"C:\Users\mbachar\side_navlori\data")
TOPIC = "/camera/image_raw/compressed"

IMGDIR = OUTROOT / "camera" / "images"
CSVPATH = OUTROOT / "camera" / "camera.csv"


def main() -> None:
    IMGDIR.mkdir(parents=True, exist_ok=True)

    ts = get_typestore(Stores.ROS2_HUMBLE)

    t_ns_list: list[int] = []
    t_bag_ns_list: list[int] = []
    filenames: list[str] = []

    with Reader(BAGDIR) as reader:
        for conn, t_bag_ns, raw in reader.messages():
            if conn.topic != TOPIC:
                continue
            msg = ts.deserialize_cdr(raw, conn.msgtype)
            # Integer nanosecond arithmetic only -- never float.
            h = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            fname = f"{h}.jpg"
            (IMGDIR / fname).write_bytes(bytes(msg.data))
            t_ns_list.append(h)
            t_bag_ns_list.append(int(t_bag_ns))
            filenames.append(fname)

    df = pd.DataFrame(
        {
            "t_ns": np.array(t_ns_list, dtype=np.int64),
            "t_bag_ns": np.array(t_bag_ns_list, dtype=np.int64),
            "filename": filenames,
        }
    )
    df.to_csv(CSVPATH, index=False)
    print(f"Wrote {len(df)} rows to {CSVPATH}")
    print(f"Wrote {len(filenames)} images to {IMGDIR}")


if __name__ == "__main__":
    main()
