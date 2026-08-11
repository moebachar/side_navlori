"""Export /wifi/rssi (std_msgs/String, JSON payload) from the rosbag2 bag to:

  1. data/wifi/wifi_raw.jsonl  - one line per message: the parsed JSON object
     with every original field preserved verbatim, plus an added key
     t_bag_ns (int, bag receive time in epoch nanoseconds).
  2. data/wifi/wifi.csv        - long format, one row per (scan, AP):
     t_start_ns, t_end_ns, t_bag_ns, scan_idx, bssid, ssid, rssi_dbm,
     freq_mhz, last_seen_ms.

Timestamp policy: raw async timestamps, no resampling / interpolation /
rounding of stamps. t_bag_ns stays integer end-to-end. t_start / t_end
originate as float unix seconds inside the JSON payload (unavoidable);
they are converted to int ns via int(round(x * 1e9)).

Note on semantics: last_seen_ms is the age of the AP observation at scan
read-out (like 'iw' output); values can exceed the ~3.6 s scan window
(cached entries up to ~29 s stale).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

BAGDIR = Path(r"C:\Users\mbachar\side_navlori\dataset_full_01")
OUTDIR = Path(r"C:\Users\mbachar\side_navlori\data\wifi")
TOPIC = "/wifi/rssi"

CSV_COLUMNS = [
    "t_start_ns", "t_end_ns", "t_bag_ns", "scan_idx",
    "bssid", "ssid", "rssi_dbm", "freq_mhz", "last_seen_ms",
]


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    ts = get_typestore(Stores.ROS2_HUMBLE)

    scans = []  # (t_bag_ns:int, obj:dict) in bag order
    with Reader(BAGDIR) as reader:
        for conn, t_bag_ns, raw in reader.messages():
            if conn.topic != TOPIC:
                continue
            msg = ts.deserialize_cdr(raw, conn.msgtype)
            obj = json.loads(msg.data)
            scans.append((int(t_bag_ns), obj))

    # 1) JSONL: parsed object + t_bag_ns, original fields preserved verbatim.
    jsonl_path = OUTDIR / "wifi_raw.jsonl"
    with open(jsonl_path, "w", encoding="utf-8", newline="\n") as f:
        for t_bag_ns, obj in scans:
            out = dict(obj)  # keep original keys and their order
            out["t_bag_ns"] = t_bag_ns
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    # 2) Long-format CSV: one row per (scan, AP).
    rows = []
    for scan_idx, (t_bag_ns, obj) in enumerate(scans):
        t_start_ns = int(round(obj["t_start"] * 1e9))
        t_end_ns = int(round(obj["t_end"] * 1e9))
        for ap in obj["aps"]:
            rows.append({
                "t_start_ns": t_start_ns,
                "t_end_ns": t_end_ns,
                "t_bag_ns": t_bag_ns,
                "scan_idx": scan_idx,
                "bssid": ap["bssid"],
                "ssid": ap["ssid"],
                "rssi_dbm": ap["rssi_dbm"],
                "freq_mhz": ap["freq_mhz"],
                "last_seen_ms": ap["last_seen_ms"],
            })

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    for col in ("t_start_ns", "t_end_ns", "t_bag_ns"):
        df[col] = df[col].astype(np.int64)
    df["scan_idx"] = df["scan_idx"].astype(np.int64)

    csv_path = OUTDIR / "wifi.csv"
    df.to_csv(csv_path, index=False)

    n_scans = len(scans)
    n_rows = len(df)
    sum_len_aps = sum(len(obj["aps"]) for _, obj in scans)
    sum_n_field = sum(obj["n"] for _, obj in scans)
    print(f"messages (scans): {n_scans}")
    print(f"csv rows:         {n_rows}")
    print(f"sum(len(aps)):    {sum_len_aps}")
    print(f"sum(n fields):    {sum_n_field}")
    print(f"wrote {jsonl_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
