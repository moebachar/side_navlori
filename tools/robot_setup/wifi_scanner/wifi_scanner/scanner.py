import json
import re
import subprocess
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class WifiScanner(Node):
    def __init__(self):
        super().__init__('wifi_scanner')
        self.declare_parameter('interface', 'wlan0')
        self.declare_parameter('period_sec', 0.5)
        self.declare_parameter('include_ssid_substring', '')
        self.interface = self.get_parameter('interface').value
        self.period = float(self.get_parameter('period_sec').value)
        self.ssid_filter = self.get_parameter('include_ssid_substring').value
        self.pub = self.create_publisher(String, '/wifi/rssi', 10)
        self.get_logger().info(
            f'WiFi scanner on {self.interface}, '
            f'ssid filter: {self.ssid_filter or "OFF (recording raw)"}')

    def scan_once(self):
        t_start = self.get_clock().now().nanoseconds / 1e9
        try:
            out = subprocess.run(
                ['sudo', '-n', 'iw', 'dev', self.interface, 'scan'],
                capture_output=True, text=True, timeout=15.0)
        except subprocess.TimeoutExpired:
            self.get_logger().warning('scan timeout')
            return None
        t_end = self.get_clock().now().nanoseconds / 1e9
        if out.returncode != 0:
            self.get_logger().warning(
                f'scan failed: {out.stderr.strip()[:120]}')
            return None
        aps = []
        ap = None
        for line in out.stdout.splitlines():
            m = re.match(r'^BSS ([0-9a-f:]{17})', line)
            if m:
                if ap:
                    aps.append(ap)
                ap = {'bssid': m.group(1), 'ssid': '', 'rssi_dbm': None,
                      'freq_mhz': None, 'last_seen_ms': None}
                continue
            if ap is None:
                continue
            s = line.strip()
            if s.startswith('signal:'):
                try:
                    ap['rssi_dbm'] = float(s.split()[1])
                except (IndexError, ValueError):
                    pass
            elif s.startswith('SSID:'):
                ap['ssid'] = s[5:].strip()
            elif s.startswith('freq:'):
                try:
                    ap['freq_mhz'] = float(s.split()[1])
                except (IndexError, ValueError):
                    pass
            elif s.startswith('last seen:'):
                mm = re.search(r'(\d+)\s*ms', s)
                if mm:
                    ap['last_seen_ms'] = int(mm.group(1))
        if ap:
            aps.append(ap)
        if self.ssid_filter:
            aps = [a for a in aps if self.ssid_filter in a['ssid']]
        return {'t_start': round(t_start, 6), 't_end': round(t_end, 6),
                'n': len(aps), 'aps': aps}

    def run(self):
        while rclpy.ok():
            result = self.scan_once()
            if result is not None:
                msg = String()
                msg.data = json.dumps(result)
                self.pub.publish(msg)
                self.get_logger().info(
                    f"scan: {result['n']} APs in "
                    f"{result['t_end'] - result['t_start']:.2f}s")
            time.sleep(self.period)


def main():
    rclpy.init()
    node = WifiScanner()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
