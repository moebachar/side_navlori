#!/usr/bin/env python3
"""navlori web teleop — drive from the laptop browser, real game controls."""
import threading, time, subprocess, signal, os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

PORT, DEADMAN = 8080, 0.3
BAGDIR = os.path.expanduser("~/bags")
TOPICS = ("/imu /odom /joint_states /scan /camera/image_raw/compressed "
          "/camera/camera_info /wifi/rssi /tf /tf_static /battery_state "
          "/cmd_vel /magnetic_field")
state = {"vx": 0.0, "wz": 0.0, "t": 0.0}
lock = threading.Lock()
rec = {"proc": None}

HTML = """<!doctype html><html><head><meta charset=utf-8><title>navlori teleop</title>
<style>body{font-family:system-ui;background:#111;color:#eee;text-align:center;padding:2em}
#s{font-size:1.4em;margin:1em;font-variant-numeric:tabular-nums}
button{font-size:1.1em;padding:.6em 1.2em;margin:.3em;border-radius:8px;border:0;cursor:pointer}
.k{display:inline-block;width:3em;height:3em;line-height:3em;margin:.15em;background:#333;border-radius:8px}
.on{background:#2a7}</style></head><body>
<h2>navlori teleop</h2><div>click here once, then use the arrow keys</div>
<div style="margin:1em"><div><span id=ku class=k>&uarr;</span></div>
<div><span id=kl class=k>&larr;</span><span id=kd class=k>&darr;</span><span id=kr class=k>&rarr;</span></div></div>
<div id=s>vx 0.00  wz 0.00  spd 0.12</div>
<button onclick="rec(1)">&#9679; Record</button><button onclick="rec(0)">&#9632; Stop rec</button>
<button onclick="held={};upd()">Stop</button>
<script>
let held={},lin=0.12,ang=0.9,LMAX=0.22,AMAX=1.5;
const ids={ArrowUp:'ku',ArrowDown:'kd',ArrowLeft:'kl',ArrowRight:'kr'};
function upd(){let vx=0,wz=0;
 if(held.ArrowUp)vx+=lin; if(held.ArrowDown)vx-=lin;
 if(held.ArrowLeft)wz+=ang; if(held.ArrowRight)wz-=ang;
 fetch(`/drive?vx=${vx}&wz=${wz}`);
 document.getElementById('s').textContent=`vx ${vx.toFixed(2)}  wz ${wz.toFixed(2)}  spd ${lin.toFixed(2)}`;
 for(const k in ids)document.getElementById(ids[k]).classList.toggle('on',!!held[k]);}
function rec(o){fetch(`/rec?on=${o}`);}
onkeydown=e=>{if(e.key in ids){held[e.key]=1;e.preventDefault();}
 else if(e.key=='a'){lin=Math.min(lin+0.02,LMAX);ang=Math.min(ang+0.1,AMAX);}
 else if(e.key=='z'){lin=Math.max(lin-0.02,0.02);ang=Math.max(ang-0.1,0.1);}
 else if(e.key==' '){held={};}};
onkeyup=e=>{if(e.key in ids){delete held[e.key];e.preventDefault();}};
onblur=()=>{held={};};
setInterval(upd,50);
</script></body></html>"""

class H(BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def do_GET(self):
        u=urlparse(self.path); q=parse_qs(u.query)
        if u.path=='/':
            self.send_response(200); self.send_header('Content-Type','text/html'); self.end_headers()
            self.wfile.write(HTML.encode())
        elif u.path=='/drive':
            with lock:
                state["vx"]=float(q.get("vx",["0"])[0]); state["wz"]=float(q.get("wz",["0"])[0]); state["t"]=time.time()
            self.send_response(204); self.end_headers()
        elif u.path=='/rec':
            on=q.get("on",["0"])[0]=="1"
            if on and not rec["proc"]:
                os.makedirs(BAGDIR,exist_ok=True); n=time.strftime("navlori_%Y%m%d_%H%M%S")
                rec["proc"]=subprocess.Popen(["bash","-lc",f"ros2 bag record -o {BAGDIR}/{n} {TOPICS}"],
                    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,preexec_fn=os.setsid)
            elif not on and rec["proc"]:
                os.killpg(os.getpgid(rec["proc"].pid),signal.SIGINT)
                try: rec["proc"].wait(timeout=10)
                except subprocess.TimeoutExpired: os.killpg(os.getpgid(rec["proc"].pid),signal.SIGKILL)
                rec["proc"]=None
            self.send_response(204); self.end_headers()
        else:
            self.send_response(404); self.end_headers()

class Drive(Node):
    def __init__(self):
        super().__init__('nav_web_teleop')
        self.pub=self.create_publisher(Twist,'cmd_vel',10)
        self.create_timer(0.05,self.tick)
    def tick(self):
        with lock:
            fresh=time.time()-state["t"]<DEADMAN
            vx=state["vx"] if fresh else 0.0; wz=state["wz"] if fresh else 0.0
        t=Twist(); t.linear.x=vx; t.angular.z=wz; self.pub.publish(t)

def main():
    rclpy.init(); node=Drive()
    ThreadingHTTPServer.daemon_threads=True
    srv=ThreadingHTTPServer(("0.0.0.0",PORT),H)
    threading.Thread(target=srv.serve_forever,daemon=True).start()
    ip=subprocess.run(["bash","-lc","hostname -I | awk '{print $1}'"],capture_output=True,text=True).stdout.strip()
    print(f"\n>>> open  http://{ip}:{PORT}  in your laptop browser\n")
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.pub.publish(Twist()); node.destroy_node(); rclpy.shutdown()

if __name__=='__main__':
    main()
