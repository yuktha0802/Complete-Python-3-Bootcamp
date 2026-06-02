#!/usr/bin/env python3
"""
x5_swarm_hybrid.py — One-time intercept mission, dual-panel display.

DISPLAY LAYOUT (1280 × 480 window):
  ┌────────────────────────┬────────────────────────┐
  │  INTERCEPTOR CAM (FPV) │  THIRD-PERSON CHASE    │
  │  ROS feed → mss → synth│  synthetic chase cam   │
  │  drone icons overlaid  │  both drones visible   │
  └────────────────────────┴────────────────────────┘
       640 × 480                  640 × 480

MISSION: one-time auto-lock on nearest target drone → NAVIGATE → PURSUE
→ SHADOW for 60 s → DONE.  No re-targeting.

═══════════════════════════════════════════════════════════════
BRIDGE COMMAND (run before this script):
  source /opt/ros/jazzy/setup.bash && ros2 run ros_gz_bridge parameter_bridge \
    /model/target_drone/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist \
    /model/target_drone_3/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist \
    /model/target_drone_4/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist \
    /model/target_drone_5/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist \
    /model/target_drone_6/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist \
    /model/target_drone_7/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist \
    /model/x5_uav/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist \
    /world/defense_zone/model/x5_uav/link/camera_link/sensor/forward_camera/image@sensor_msgs/msg/Image[gz.msgs.Image \
    /world/defense_zone/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V

YAW NOTE:
  angular.z has NO effect on X5 (MulticopterVelocityControl, Gazebo Harmonic).
  Yaw is permanently fixed at 0° → body frame ≡ world frame forever.
  CAM_HDG = 0.0 is used for FPV render/projection.
  self.x5_hdg is updated from velocity for HUD display only.
═══════════════════════════════════════════════════════════════
"""

import math
import subprocess
import threading
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from tf2_msgs.msg import TFMessage

# Optional live camera capture
try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False

# Optional ROS camera bridge
try:
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image
    CVBRIDGE_AVAILABLE = True
except ImportError:
    CVBRIDGE_AVAILABLE = False

# ── Display constants ───────────────────────────────────────────────────────
PANEL_W = 640
PANEL_H = 480
TOTAL_W = 1280
TOTAL_H = 480

# ── 3-D camera constants ────────────────────────────────────────────────────
CAM_PITCH = 0.30          # FPV: slight downward tilt
CAM_HDG   = 0.0           # FIXED — physical yaw immutable; camera always faces +X
FOV_H     = math.pi       # 180° horizontal FOV
FOCAL     = PANEL_W / (2 * math.tan(FOV_H / 2))
NEAR      = 0.30          # near-plane clip distance

CHASE_BACK   = 25.0       # chase cam: metres behind X5
CHASE_UP     = 12.0       # chase cam: metres above X5

# ── Fog ────────────────────────────────────────────────────────────────────
FOG_START = 40.0
FOG_END   = 200.0
FOG_BGR   = np.array([200, 205, 185], dtype=np.float32)

# ── Toolbar crop for mss window capture ────────────────────────────────────
TOOLBAR_H = 52

# ── Defense world geometry ─────────────────────────────────────────────────
BUILDINGS = [
    ( 6,  6, 7.0, 5.0,  9.0, (130,140,150), (100,110,120)),
    ( 6,  6, 4.0, 3.0,  5.0, (115,125,135), ( 88, 98,108)),
    (-18,  9,11.0, 3.0,  4.0, (128,138,148), ( 95,105,115)),
    (-18,  0,11.0, 3.0,  4.0, (128,138,148), ( 95,105,115)),
    (-18, -9,11.0, 3.0,  4.0, (128,138,148), ( 95,105,115)),
    ( 22,-14, 4.5, 4.5,  5.0, (105,112,120), ( 78, 85, 92)),
    ( 33,-14, 4.5, 4.5,  5.0, (105,112,120), ( 78, 85, 92)),
    (  8,-21, 3.5, 3.0,  3.0, (110,118,125), ( 82, 90, 97)),
    ( 18,-21, 3.5, 3.0,  3.0, (110,118,125), ( 82, 90, 97)),
    ( -5,-21, 3.5, 2.5,  3.0, (110,118,125), ( 82, 90, 97)),
    ( 30,  5, 2.5, 2.5,  4.0, (108,116,124), ( 80, 88, 96)),
    ( 46, 46, 1.5, 1.5, 15.0, (130,140,150), ( 98,108,118)),
    (-46, 46, 1.5, 1.5, 15.0, (130,140,150), ( 98,108,118)),
    ( 46,-46, 1.5, 1.5, 15.0, (130,140,150), ( 98,108,118)),
    (-46,-46, 1.5, 1.5, 15.0, (130,140,150), ( 98,108,118)),
    (  0, 51,51.0, 1.0,  4.0, (108,112,116), ( 82, 86, 90)),
    (  0,-51,51.0, 1.0,  4.0, (108,112,116), ( 82, 86, 90)),
    ( 51,  0, 1.0,51.0,  4.0, (108,112,116), ( 82, 86, 90)),
    (-51,  0, 1.0,51.0,  4.0, (108,112,116), ( 82, 86, 90)),
]

ROADS = [
    ((-60, 0),(60, 0), 8), ((0,-60),(0, 60), 8),
    ((-60,25),(60,25), 5), ((-60,-25),(60,-25), 5),
    ((25,-60),(25,60), 5), ((-25,-60),(-25,60), 5),
]

# ── Nav constants ───────────────────────────────────────────────────────────
NAV_MAX   = 25.0;  NAV_SLOW  = 15.0
PUR_MAX   = 25.0;  PUR_MIN   =  3.0
MAX_VRT   = 12.0
INTC_DIST =  3.0;  SHAD_DIST = 18.0
SHAD_DUR  = 60.0;  LOST_T    =  2.0
SRCH_SPD  = 18.0
SEARCH_ALT     = 20.0
SEARCH_TIMEOUT = 20.0
SEARCH_ZONES   = [
    (  0,  0), ( 30, 30), (-30, 30),
    ( 30,-30), (-30,-30), (  0, 45),
    (  0,-45), ( 45,  0), (-45,  0),
]

class State:
    SEARCH   = "SEARCH"
    NAVIGATE = "NAVIGATE"
    PURSUE   = "PURSUE"
    SHADOW   = "SHADOW"
    DONE     = "DONE"


# ══════════════════════════════════════════════════════════════════════════════
#  3-D rendering helpers
# ══════════════════════════════════════════════════════════════════════════════

def w2c(pt, pos, heading, pitch):
    """World point → camera space (xc=depth, yc=right→left, zc=up)."""
    dx, dy, dz = pt[0]-pos[0], pt[1]-pos[1], pt[2]-pos[2]
    ch, sh = math.cos(heading), math.sin(heading)
    xh, yh = dx*ch+dy*sh, -dx*sh+dy*ch
    cp, sp = math.cos(pitch), math.sin(pitch)
    return xh*cp - dz*sp, yh, xh*sp + dz*cp

def proj(xc, yc, zc, w=PANEL_W, h=PANEL_H, focal=FOCAL):
    if xc <= 0:
        return None
    foc = w / (2 * math.tan(FOV_H / 2))
    return int(w/2 - foc*yc/xc), int(h/2 - foc*zc/xc)

def clip_poly(cam_pts, near=NEAR):
    output = []
    n = len(cam_pts)
    for i in range(n):
        cur  = cam_pts[i]
        prev = cam_pts[i-1]
        ci = cur[0]  >= near
        pi = prev[0] >= near
        if ci:
            if not pi:
                t = (near-prev[0])/(cur[0]-prev[0])
                output.append((near,
                               prev[1]+t*(cur[1]-prev[1]),
                               prev[2]+t*(cur[2]-prev[2])))
            output.append(cur)
        elif pi:
            t = (near-prev[0])/(cur[0]-prev[0])
            output.append((near,
                          prev[1]+t*(cur[1]-prev[1]),
                          prev[2]+t*(cur[2]-prev[2])))
    return output

def fog(col_bgr, dist):
    t = min(1.0, max(0.0, (dist-FOG_START)/(FOG_END-FOG_START)))
    c = np.array(col_bgr, dtype=np.float32)
    return tuple(int(v) for v in c*(1-t)+FOG_BGR*t)


def render_scene(pos, heading, pitch, w=PANEL_W, h=PANEL_H):
    """Render the defense world from an arbitrary camera pose."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    foc   = w / (2 * math.tan(FOV_H / 2))

    def _proj(xc, yc, zc):
        if xc <= 0:
            return None
        return int(w/2 - foc*yc/xc), int(h/2 - foc*zc/xc)

    # Sky
    sky_h = int(h * 0.40)
    ys = np.arange(sky_h, dtype=np.float32) / max(sky_h, 1)
    frame[:sky_h,:,0] = np.clip(195+ys*10, 0, 255)[:,None]
    frame[:sky_h,:,1] = np.clip(160+ys*30, 0, 255)[:,None]
    frame[:sky_h,:,2] = np.clip( 95+ys*60, 0, 255)[:,None]
    hb = min(sky_h+20, h)
    if hb > sky_h:
        yh = np.arange(sky_h, hb, dtype=np.float32)
        t2 = (yh - sky_h) / max(hb-sky_h, 1)
        frame[sky_h:hb,:,0] = np.clip(205-t2*50, 0, 255)[:,None]
        frame[sky_h:hb,:,1] = np.clip(185-t2*35, 0, 255)[:,None]
        frame[sky_h:hb,:,2] = np.clip(165-t2*30, 0, 255)[:,None]

    # Ground
    gs = np.arange(hb, h, dtype=np.float32)
    t3 = (gs - hb) / max(h-hb, 1)
    frame[hb:,:,0] = np.clip( 90-t3*15, 0, 255)[:,None]
    frame[hb:,:,1] = np.clip(145-t3*20, 0, 255)[:,None]
    frame[hb:,:,2] = np.clip(165-t3*20, 0, 255)[:,None]

    def gpt(wx, wy):
        xc,yc,zc = w2c((wx,wy,0.05), pos, heading, pitch)
        if xc <= NEAR:
            return None
        sx,sy = _proj(xc, yc, zc)
        return (sx,sy) if sx and 0<=sx<w and 0<=sy<h else None

    # Ground grid
    for v in range(-80, 81, 10):
        p1=gpt(v,-80); p2=gpt(v, 80)
        if p1 and p2:
            d=abs(v-pos[0]); a=max(0,int(50*(1-d/90)))
            cv2.line(frame,p1,p2,(int(a*0.8),int(a*1.0),int(a*1.1)),1)
        p3=gpt(-80,v); p4=gpt(80,v)
        if p3 and p4:
            d=abs(v-pos[1]); a=max(0,int(50*(1-d/90)))
            cv2.line(frame,p3,p4,(int(a*0.8),int(a*1.0),int(a*1.1)),1)

    # Roads
    for (x1,y1),(x2,y2),rw in ROADS:
        pts=[]; steps=8
        for s in range(steps+1):
            t=s/steps
            p=gpt(x1+t*(x2-x1), y1+t*(y2-y1))
            if p: pts.append(p)
        if len(pts) > 1:
            dist = math.sqrt(((x1+x2)/2-pos[0])**2+((y1+y2)/2-pos[1])**2)
            ft   = min(1.0,max(0.0,(dist-FOG_START)/(FOG_END-FOG_START)))
            gray = int(62*(1-ft)+185*ft)
            thick= max(1,int((rw/2)*(1-dist/150)))
            for i in range(len(pts)-1):
                cv2.line(frame,pts[i],pts[i+1],(gray,gray,gray),thick)

    # Buildings (painter's algorithm)
    polys = []
    for cx_,cy_,hw_,hd_,ht_,top_col,side_col in BUILDINGS:
        dist = math.sqrt((cx_-pos[0])**2+(cy_-pos[1])**2)
        if dist > FOG_END+20: continue
        cw_=[
            (cx_-hw_,cy_-hd_,0),(cx_+hw_,cy_-hd_,0),(cx_+hw_,cy_+hd_,0),(cx_-hw_,cy_+hd_,0),
            (cx_-hw_,cy_-hd_,ht_),(cx_+hw_,cy_-hd_,ht_),(cx_+hw_,cy_+hd_,ht_),(cx_-hw_,cy_+hd_,ht_),
        ]
        cc_=[w2c(p,pos,heading,pitch) for p in cw_]
        faces=[
            ([4,5,6,7],(0,0,1), top_col),
            ([0,1,5,4],(0,-1,0),side_col),
            ([1,2,6,5],(1,0,0), tuple(int(c*0.85) for c in side_col)),
            ([2,3,7,6],(0,1,0), tuple(int(c*0.70) for c in side_col)),
            ([3,0,4,7],(-1,0,0),tuple(int(c*0.78) for c in side_col)),
        ]
        for idxs,normal,col in faces:
            fc=np.mean([cw_[i] for i in idxs],axis=0)
            tc=np.array([pos[0]-fc[0],pos[1]-fc[1],pos[2]-fc[2]])
            if np.dot(np.array(normal),tc) <= 0: continue
            clipped=clip_poly([cc_[i] for i in idxs])
            if len(clipped) < 3: continue
            pts=[_proj(*p) for p in clipped]
            pts=[p for p in pts if p is not None]
            if len(pts) < 3: continue
            depth=float(np.mean([p[0] for p in clipped]))
            polys.append((depth, pts, fog(col,dist)))

    polys.sort(key=lambda x: -x[0])
    for _,pts,col in polys:
        try:
            arr=np.array(pts,dtype=np.int32)
            cv2.fillPoly(frame,[arr],col)
            cv2.polylines(frame,[arr],True,tuple(max(0,c-20) for c in col),1)
        except:
            pass

    return frame


def chase_cam_params(x5_pos, x3_pos):
    """Compute chase camera position/heading/pitch behind X5 looking toward X3."""
    if x5_pos is None:
        return (0, 0, 30), 0.0, 0.35
    if x3_pos is None:
        # Look in X5's forward direction (world +X)
        return (x5_pos[0]-CHASE_BACK, x5_pos[1], x5_pos[2]+CHASE_UP), 0.0, 0.25

    # Direction from X5 toward X3
    dx = x3_pos[0] - x5_pos[0]
    dy = x3_pos[1] - x5_pos[1]
    dz = x3_pos[2] - x5_pos[2]
    horiz = math.sqrt(dx**2 + dy**2)
    if horiz < 0.1:
        return (x5_pos[0], x5_pos[1], x5_pos[2]+CHASE_UP+5), 0.0, 0.5

    nx, ny = dx/horiz, dy/horiz
    cam = (
        x5_pos[0] - nx * CHASE_BACK,
        x5_pos[1] - ny * CHASE_BACK,
        x5_pos[2] + CHASE_UP,
    )
    # Camera looks from its position toward X3
    look_dx = x3_pos[0] - cam[0]
    look_dy = x3_pos[1] - cam[1]
    look_dz = x3_pos[2] - cam[2]
    look_h  = math.sqrt(look_dx**2 + look_dy**2)
    hdg     = math.atan2(look_dy, look_dx)
    pitch   = math.atan2(-look_dz, max(look_h, 0.1))
    return cam, hdg, pitch


# ── Drone icon rendering ────────────────────────────────────────────────────

def draw_drone_icon(img, cx, cy, arm, color, label="", is_target=False, dist=None):
    """
    Draw a proper top-down drone icon (cross with rotors).
    arm: arm length in pixels
    """
    if not (0 <= cx < img.shape[1] and 0 <= cy < img.shape[0]):
        return

    body_r = max(3, arm // 4)
    rotor_r = max(2, arm // 3)

    # Body
    cv2.circle(img, (cx, cy), body_r, color, -1)

    # 4 arms at 45°, 135°, 225°, 315°
    for deg in [45, 135, 225, 315]:
        rad = math.radians(deg)
        ax = int(cx + arm * math.cos(rad))
        ay = int(cy + arm * math.sin(rad))
        cv2.line(img, (cx, cy), (ax, ay), color, 2)
        # Rotor disc (filled for target, outline for X5)
        if is_target:
            cv2.circle(img, (ax, ay), rotor_r, color, -1)
            # Spinning cross inside rotor
            cv2.line(img, (ax-rotor_r, ay), (ax+rotor_r, ay), (0,0,0), 1)
            cv2.line(img, (ax, ay-rotor_r), (ax, ay+rotor_r), (0,0,0), 1)
        else:
            cv2.circle(img, (ax, ay), rotor_r, color, 1)

    # Target lock box
    if is_target:
        pad = arm + 6
        cv2.rectangle(img, (cx-pad, cy-pad), (cx+pad, cy+pad), color, 1)
        # Corner ticks
        tick = 8
        for sx_, sy_ in [(-1,-1),(1,-1),(-1,1),(1,1)]:
            ox, oy = cx+sx_*pad, cy+sy_*pad
            cv2.line(img, (ox,oy), (ox+sx_*tick,oy),       color, 2)
            cv2.line(img, (ox,oy), (ox,         oy+sy_*tick), color, 2)

    # Label + distance
    dist_txt = f" {dist:.0f}m" if dist is not None else ""
    full_label = f"{'★' if is_target else ''}{label}{dist_txt}"
    # Background pill
    (tw, th_), bl = cv2.getTextSize(full_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    tx = cx + arm + 5
    ty = cy - 4
    cv2.rectangle(img, (tx-2, ty-th_-2), (tx+tw+2, ty+bl), (0,0,0), -1)
    cv2.putText(img, full_label, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def project_onto_panel(world_pos, cam_pos, heading, pitch, w=PANEL_W, h=PANEL_H):
    """Project world_pos onto a panel rendered from cam_pos/heading/pitch."""
    if cam_pos is None:
        return None
    xc, yc, zc = w2c(world_pos, cam_pos, heading, pitch)
    if xc <= NEAR:
        return None
    fov_v = FOV_H * h / w
    ah = math.atan2(yc, xc)
    av = math.atan2(zc, xc)
    if abs(ah) > FOV_H/2*0.97 or abs(av) > fov_v/2*0.97:
        return None
    px = int(w/2 - (ah/(FOV_H/2))*(w/2))
    py = int(h/2 - (av/(fov_v/2))*(h/2))
    dist = math.sqrt(sum((world_pos[i]-cam_pos[i])**2 for i in range(3)))
    return px, py, dist


# ── Text with background ────────────────────────────────────────────────────

def tb(img, text, x, y, sc, col, th=1):
    (tw, th_), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, sc, th)
    cv2.rectangle(img, (x-2,y-th_-2), (x+tw+2,y+bl), (0,0,0), -1)
    cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, th)


# ── Live Gazebo window capture ──────────────────────────────────────────────

class GazeboCamCapture:
    def __init__(self):
        self.sct     = mss.mss() if MSS_AVAILABLE else None
        self.monitor = None
        self._find()

    def _find(self):
        if not MSS_AVAILABLE:
            return
        keywords = ['camera','x5','interceptor','image','forward']
        try:
            result = subprocess.run(['xdotool','search','--name',''],
                                    capture_output=True, text=True, timeout=5)
            for wid in result.stdout.strip().split('\n'):
                if not wid:
                    continue
                try:
                    nr = subprocess.run(['xdotool','getwindowname',wid],
                                        capture_output=True, text=True, timeout=2)
                    name = nr.stdout.strip()
                    if not any(k.lower() in name.lower() for k in keywords):
                        continue
                    gr = subprocess.run(['xdotool','getwindowgeometry','--shell',wid],
                                        capture_output=True, text=True, timeout=2)
                    info = dict(l.split('=',1) for l in gr.stdout.strip().split('\n') if '=' in l)
                    W = int(info.get('WIDTH',0))
                    H = int(info.get('HEIGHT',0))
                    if W > 200 and H > 150:
                        self.monitor = {
                            'top':    int(info['Y']) + TOOLBAR_H,
                            'left':   int(info['X']),
                            'width':  W,
                            'height': max(1, H - TOOLBAR_H),
                        }
                        print(f"[cam] {name}  {W}x{H}")
                        return
                except Exception:
                    continue
        except Exception as e:
            print(f"[cam] {e}")

    def capture(self):
        if not self.sct:
            return None
        if not self.monitor:
            self._find()
            return None
        try:
            shot  = self.sct.grab(self.monitor)
            frame = np.array(shot)
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        except Exception:
            self.monitor = None
            return None


# ══════════════════════════════════════════════════════════════════════════════
#  ROS Node
# ══════════════════════════════════════════════════════════════════════════════

class X5SwarmHybrid(Node):

    def __init__(self):
        super().__init__("x5_interceptor")

        # Command publisher
        self.cmd_pub = self.create_publisher(Twist, "/model/x5_uav/cmd_vel", 10)

        # Pose subscriber (TFMessage from dynamic_pose bridge)
        self.create_subscription(TFMessage,
            "/world/defense_zone/dynamic_pose/info", self.pose_cb, 10)

        # ROS camera feed (optional)
        self._ros_frame      = None
        self._ros_frame_lock = threading.Lock()
        if CVBRIDGE_AVAILABLE:
            self._bridge = CvBridge()
            cam_qos = QoSProfile(depth=1)
            cam_qos.reliability = ReliabilityPolicy.BEST_EFFORT
            self.create_subscription(
                Image,
                "/world/defense_zone/model/x5_uav/link/"
                "camera_link/sensor/forward_camera/image",
                self._ros_cam_cb,
                cam_qos,
            )

        # Pose state
        self.x5_pos  = None
        self.x5_hdg  = 0.0       # movement heading — HUD display only
        self.x5_vel  = (0.0, 0.0, 0.0)
        self.x5_vt   = time.time()

        # Target drones
        self.targets   = {}       # name → (x,y,z)
        self.tgt_vel   = {}       # name → (vx,vy,vz)
        self.tgt_prev  = {}       # name → ((x,y,z), timestamp)

        # Mission state (one-time, no re-targeting)
        self.selected   = None
        self.last_view  = time.time()
        self.state      = State.SEARCH
        self.shad_t     = 0.0
        self.sang       = 0.0
        self.srad       = 3.0
        self.st         = time.time()
        self.search_phase   = "CLIMB"
        self.search_zone_idx = 0
        self.search_phase_t  = time.time()

        # Live capture
        self.cam_capture = GazeboCamCapture()

        # OpenCV window
        cv2.namedWindow("X5 INTERCEPTOR", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("X5 INTERCEPTOR", TOTAL_W, TOTAL_H)

        self.create_timer(1/30, self.loop)

        self.get_logger().info(
            "X5 ready  |  one-time mission  |  "
            "YAW FIXED (CAM_HDG=0.0)  |  dual-panel display"
        )

    # ── ROS camera callback ────────────────────────────────────────────────
    def _ros_cam_cb(self, msg):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, "bgr8")
            with self._ros_frame_lock:
                self._ros_frame = frame
        except Exception:
            pass

    # ── Pose callback ──────────────────────────────────────────────────────
    def pose_cb(self, msg):
        """
        Parse TFMessage from dynamic_pose bridge.
        Uses child_frame_id for reliable drone identification.
        Handles scoped names like 'defense_zone::x5_uav'.
        """
        now = time.time()
        for tf in msg.transforms:
            raw   = tf.child_frame_id
            parts = raw.split("::")
            if len(parts) > 2:
                continue          # sub-link — skip
            name = parts[-1]

            t   = tf.transform.translation
            pos = (t.x, t.y, t.z)

            if name == "x5_uav":
                if self.x5_pos is not None:
                    dt = now - self.x5_vt
                    if dt > 0.05:
                        dx = pos[0] - self.x5_pos[0]
                        dy = pos[1] - self.x5_pos[1]
                        dz = pos[2] - self.x5_pos[2]
                        if math.sqrt(dx**2+dy**2)/dt > 0.3:
                            self.x5_hdg = math.atan2(dy, dx)  # movement hdg, HUD only
                        self.x5_vel = (dx/dt, dy/dt, dz/dt)
                        self.x5_vt  = now
                self.x5_pos = pos

            elif "target_drone" in name:
                if name in self.tgt_prev:
                    pp, pt = self.tgt_prev[name]
                    tdt = now - pt
                    if tdt > 0.05:
                        self.tgt_vel[name] = (
                            (pos[0]-pp[0])/tdt,
                            (pos[1]-pp[1])/tdt,
                            (pos[2]-pp[2])/tdt,
                        )
                self.tgt_prev[name] = (pos, now)
                self.targets[name]  = pos

    # ── FPV camera source ──────────────────────────────────────────────────
    def _get_fpv_frame(self):
        """Returns 640×480 FPV frame from: mss > ROS topic > synthetic."""
        # 1. mss live capture
        live = self.cam_capture.capture()
        if live is not None and live.shape[0] > 10 and np.mean(live) > 20:
            return cv2.resize(live, (PANEL_W, PANEL_H))
        # 2. ROS camera topic
        with self._ros_frame_lock:
            ros = self._ros_frame
        if ros is not None:
            return cv2.resize(ros, (PANEL_W, PANEL_H))
        # 3. Synthetic first-person (CAM_HDG=0.0 — yaw is immutable)
        pos = self.x5_pos if self.x5_pos else (0, 0, 10)
        return render_scene(pos, CAM_HDG, CAM_PITCH)

    # ── Chase camera frame ─────────────────────────────────────────────────
    def _get_chase_frame(self):
        """Returns 640×480 third-person chase-cam frame (always synthetic)."""
        sel_pos = self.targets.get(self.selected) if self.selected else None
        cam_pos, hdg, pitch = chase_cam_params(self.x5_pos, sel_pos)
        return render_scene(cam_pos, hdg, pitch)

    # ── Main loop ──────────────────────────────────────────────────────────
    def loop(self):
        # ── Build frames ───────────────────────────────────────────────────
        fpv_frame   = self._get_fpv_frame()
        chase_frame = self._get_chase_frame()

        # ── Auto-lock (one-time, never changes after selection) ────────────
        # Projects using FPV camera (heading=0, pos=x5_pos)
        fpv_icons = {}
        for lbl, p in self.targets.items():
            res = project_onto_panel(p, self.x5_pos, CAM_HDG, CAM_PITCH)
            if res:
                fpv_icons[lbl] = res

        if self.state == State.SEARCH and self.selected is None and fpv_icons:
            nearest = min(fpv_icons.items(), key=lambda x: x[1][2])
            self.selected = nearest[0]
            self.state    = State.NAVIGATE
            self.get_logger().info(
                f"Mission locked: {self.selected}  dist={nearest[1][2]:.1f}m"
            )

        # ── Draw drone icons on FPV panel ──────────────────────────────────
        for lbl, (px, py, dist) in fpv_icons.items():
            is_sel = (lbl == self.selected)
            color  = (0, 60, 255) if is_sel else (60, 220, 60)
            arm    = max(6, int(50 / max(dist, 1)))
            draw_drone_icon(fpv_frame, px, py, arm, color, lbl, is_sel, dist)

        # ── Draw drone icons on chase panel ───────────────────────────────
        sel_pos = self.targets.get(self.selected) if self.selected else None
        cam_pos, c_hdg, c_pitch = chase_cam_params(self.x5_pos, sel_pos)

        # Target drones on chase view
        for lbl, p in self.targets.items():
            res = project_onto_panel(p, cam_pos, c_hdg, c_pitch)
            if res:
                px, py, dist = res
                is_sel = (lbl == self.selected)
                color  = (0, 60, 255) if is_sel else (60, 220, 60)
                arm    = max(6, int(50 / max(dist, 1)))
                draw_drone_icon(chase_frame, px, py, arm, color, lbl, is_sel, dist)

        # X5 on chase view (blue)
        if self.x5_pos:
            res = project_onto_panel(self.x5_pos, cam_pos, c_hdg, c_pitch)
            if res:
                px, py, dist = res
                arm = max(8, int(60 / max(dist, 1)))
                draw_drone_icon(chase_frame, px, py, arm, (200, 140, 0), "X5", False, dist)

        # ── Crosshair on FPV ───────────────────────────────────────────────
        cx_, cy_ = PANEL_W//2, PANEL_H//2
        cv2.line(fpv_frame, (cx_-18, cy_), (cx_+18, cy_), (0,220,0), 1)
        cv2.line(fpv_frame, (cx_, cy_-18), (cx_, cy_+18), (0,220,0), 1)
        cv2.circle(fpv_frame, (cx_, cy_), 28, (0,180,0), 1)

        # ── State machine ──────────────────────────────────────────────────
        self._sm(fpv_icons)

        # ── HUD overlays ───────────────────────────────────────────────────
        self._hud_fpv(fpv_frame, fpv_icons)
        self._hud_chase(chase_frame)

        # ── Combine panels side by side ────────────────────────────────────
        # Divider line
        combined = np.hstack([fpv_frame, chase_frame])
        cv2.line(combined, (PANEL_W, 0), (PANEL_W, TOTAL_H), (0, 200, 200), 2)

        # Panel labels
        cv2.putText(combined, "FPV - INTERCEPTOR CAM",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,200), 1)
        cv2.putText(combined, "THIRD PERSON - CHASE VIEW",
                    (PANEL_W+8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,200), 1)

        cv2.imshow("X5 INTERCEPTOR", combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()

    # ── State machine ──────────────────────────────────────────────────────
    def _sm(self, fpv_icons):
        if self.state == State.SEARCH:
            self._search()
            return
        if not self.selected or self.state == State.DONE:
            return
        sp = self.targets.get(self.selected)
        if sp is None:
            return
        if self.state == State.NAVIGATE:
            self._nav(sp)
            if self.selected in fpv_icons:
                self.state     = State.PURSUE
                self.last_view = time.time()
        elif self.state == State.PURSUE:
            if self.selected in fpv_icons:
                self.last_view = time.time()
                _, _, dist = fpv_icons[self.selected]
                if dist < INTC_DIST:
                    self.state  = State.SHADOW
                    self.shad_t = time.time()
                    return
                self._pursue(sp)
            elif time.time() - self.last_view > LOST_T:
                self.state = State.NAVIGATE
        elif self.state == State.SHADOW:
            if time.time() - self.shad_t >= SHAD_DUR:
                self.get_logger().info("Mission complete — DONE")
                self.state = State.DONE
                self.stop()
                return
            self._shadow(sp)

    # ── Search ─────────────────────────────────────────────────────────────
    def _search(self):
        now = time.time()
        tw  = Twist()

        if self.search_phase == "CLIMB":
            if self.x5_pos:
                alt_err = SEARCH_ALT - self.x5_pos[2]
                tw.linear.z = float(np.clip(alt_err*0.9, -2.0, 8.0))
                if abs(alt_err) < 2.0:
                    self.search_phase   = "SPIRAL"
                    self.search_phase_t = now
                    self.sang = 0.0; self.srad = 3.0
                    self.get_logger().info("Search altitude — spiraling")
            else:
                tw.linear.z = 3.0
            self.cmd_pub.publish(tw)

        elif self.search_phase == "SPIRAL":
            elapsed = now - self.search_phase_t
            if elapsed > SEARCH_TIMEOUT:
                self.search_zone_idx = (self.search_zone_idx+1) % len(SEARCH_ZONES)
                self.search_phase    = "TRANSIT"
                self.search_phase_t  = now
                zx,zy = SEARCH_ZONES[self.search_zone_idx]
                self.get_logger().info(f"Transit to zone {self.search_zone_idx} ({zx},{zy})")
            else:
                self.srad  = 3.0 + 0.4*elapsed
                self.sang += SRCH_SPD/self.srad*(1/30)
                tw.linear.x = SRCH_SPD*math.cos(self.sang)
                tw.linear.y = SRCH_SPD*math.sin(self.sang)
                if self.x5_pos:
                    tw.linear.z = float(np.clip((SEARCH_ALT-self.x5_pos[2])*0.5,-2.0,3.0))
                self.cmd_pub.publish(tw)

        elif self.search_phase == "TRANSIT":
            zx,zy = SEARCH_ZONES[self.search_zone_idx]
            if self.x5_pos:
                dx,dy = zx-self.x5_pos[0], zy-self.x5_pos[1]
                dz    = SEARCH_ALT-self.x5_pos[2]
                dist  = math.sqrt(dx**2+dy**2)
                if dist < 5.0:
                    self.search_phase   = "SPIRAL"
                    self.search_phase_t = now
                    self.sang=0.0; self.srad=3.0
                else:
                    spd = min(NAV_MAX, 18.0)
                    tw.linear.x = spd*dx/dist
                    tw.linear.y = spd*dy/dist
                    tw.linear.z = float(np.clip(dz*0.5,-2.0,4.0))
                    self.cmd_pub.publish(tw)

    # ── Navigation ─────────────────────────────────────────────────────────
    def _nav(self, tp):
        if not self.x5_pos: return
        dx,dy,dz = tp[0]-self.x5_pos[0], tp[1]-self.x5_pos[1], tp[2]-self.x5_pos[2]
        dist = math.sqrt(dx**2+dy**2+dz**2)
        if dist < 1.0: self.stop(); return
        spd = min(NAV_MAX, max(2.0, NAV_MAX*dist/NAV_SLOW))
        tw  = Twist()
        tw.linear.x,tw.linear.y,tw.linear.z = spd*dx/dist, spd*dy/dist, spd*dz/dist
        self.cmd_pub.publish(tw)

    # ── Pursuit ────────────────────────────────────────────────────────────
    def _pursue(self, sp):
        if not self.x5_pos: return
        dx,dy,dz = sp[0]-self.x5_pos[0], sp[1]-self.x5_pos[1], sp[2]-self.x5_pos[2]
        d3 = math.sqrt(dx**2+dy**2+dz**2)
        if d3 < 1.0: self.stop(); return
        speed = float(np.clip(PUR_MAX*min(1.0,d3/15.0), PUR_MIN, PUR_MAX))
        tw = Twist()
        tw.linear.x = float(np.clip(speed*dx/d3, -PUR_MAX, PUR_MAX))
        tw.linear.y = float(np.clip(speed*dy/d3, -PUR_MAX, PUR_MAX))
        tw.linear.z = float(np.clip(speed*dz/d3, -MAX_VRT, MAX_VRT))
        self.cmd_pub.publish(tw)

    # ── Shadow ─────────────────────────────────────────────────────────────
    def _shadow(self, tp):
        if not self.x5_pos: return
        tvx,tvy,tvz = self.tgt_vel.get(self.selected,(0,0,0))
        tspd = math.sqrt(tvx**2+tvy**2)
        if tspd > 0.5:
            nx,ny = tvx/tspd, tvy/tspd
            sx,sy,sz = tp[0]-nx*SHAD_DIST, tp[1]-ny*SHAD_DIST, tp[2]
        else:
            te = time.time()-self.shad_t
            sx = tp[0]+SHAD_DIST*math.cos(te*0.5)
            sy = tp[1]+SHAD_DIST*math.sin(te*0.5)
            sz = tp[2]
        dx,dy,dz = sx-self.x5_pos[0], sy-self.x5_pos[1], sz-self.x5_pos[2]
        ds   = math.sqrt(dx**2+dy**2+dz**2)
        corr = min(5.0, ds*1.5)
        if ds > 0.3:
            vx = corr*dx/ds+tvx*0.9; vy=corr*dy/ds+tvy*0.9; vz=corr*dz/ds+tvz*0.5
        else:
            vx,vy,vz = tvx*0.9, tvy*0.9, tvz*0.5
        tw = Twist()
        tw.linear.x = float(np.clip(vx,-NAV_MAX,NAV_MAX))
        tw.linear.y = float(np.clip(vy,-NAV_MAX,NAV_MAX))
        tw.linear.z = float(np.clip(vz,-4,4))
        self.cmd_pub.publish(tw)

    def stop(self): self.cmd_pub.publish(Twist())

    # ── FPV HUD ────────────────────────────────────────────────────────────
    def _hud_fpv(self, panel, icons):
        SC = {State.SEARCH:(0,200,255), State.NAVIGATE:(0,220,255),
              State.PURSUE:(50,255,50), State.SHADOW:(0,140,255),
              State.DONE:(100,255,100)}
        sc = SC.get(self.state,(255,255,255))
        y  = 30
        tb(panel, f"STATE: {self.state}", 10, y, 0.7, sc, 2); y += 30

        if self.state == State.SEARCH:
            zx,zy = SEARCH_ZONES[self.search_zone_idx]
            tb(panel,f"PHASE: {self.search_phase}  Z{self.search_zone_idx}({zx},{zy})",10,y,0.45,(255,200,0)); y+=22
            if self.search_phase == "SPIRAL":
                rem = max(0, SEARCH_TIMEOUT-(time.time()-self.search_phase_t))
                tb(panel,f"Next zone in {rem:.0f}s",10,y,0.42,(200,170,0)); y+=20

        if self.x5_pos:
            spd = math.sqrt(sum(v**2 for v in self.x5_vel))
            tb(panel,f"ALT  {self.x5_pos[2]:.1f}m",  10,y,0.55,(210,210,210)); y+=22
            tb(panel,f"SPD  {spd:.1f}m/s",            10,y,0.55,(210,210,210)); y+=22
            tb(panel,f"HDG  {math.degrees(self.x5_hdg)%360:.0f}° (move)",10,y,0.45,(180,140,0)); y+=20

        if self.state == State.DONE:
            cx_,cy_ = PANEL_W//2, PANEL_H//2
            cv2.putText(panel,"MISSION COMPLETE",(cx_-140,cy_),
                        cv2.FONT_HERSHEY_SIMPLEX,1.0,(60,255,60),2)

        if self.selected:
            sp_ = self.targets.get(self.selected)
            if sp_ and self.x5_pos:
                dist = math.sqrt(sum((self.x5_pos[i]-sp_[i])**2 for i in range(3)))
                col  = (0,60,255)
                rx   = PANEL_W-210
                tb(panel,f"TGT {self.selected}",rx,28,0.6,col,2)
                tb(panel,f"DIST {dist:.1f}m",    rx,50,0.5,(210,210,210))
                tb(panel,f"ALT  {sp_[2]:.1f}m",  rx,68,0.5,(210,210,210))
                if self.state == State.SHADOW:
                    rem = max(0,SHAD_DUR-(time.time()-self.shad_t))
                    tb(panel,f"SHADOW {rem:.0f}s",rx,88,0.5,(0,140,255))

        # Bottom bar
        yb = PANEL_H-12
        tb(panel,f"DRONES:{len(self.targets)}  IN VIEW:{len(icons)}",
           8,yb,0.45,(170,170,170))

    # ── Chase HUD ──────────────────────────────────────────────────────────
    def _hud_chase(self, panel):
        if not self.x5_pos:
            return
        sel_pos = self.targets.get(self.selected) if self.selected else None

        # X5 info
        y  = 30
        spd = math.sqrt(sum(v**2 for v in self.x5_vel))
        tb(panel,f"X5  ALT {self.x5_pos[2]:.1f}m  SPD {spd:.1f}m/s",
           8,y,0.48,(200,160,0)); y+=22

        if sel_pos:
            dist = math.sqrt(sum((self.x5_pos[i]-sel_pos[i])**2 for i in range(3)))
            bearing = math.degrees(math.atan2(
                sel_pos[1]-self.x5_pos[1], sel_pos[0]-self.x5_pos[0]))
            tb(panel,f"TGT  DIST {dist:.1f}m  BRG {bearing:.0f}°",
               8,y,0.48,(0,60,255)); y+=22
            tvx,tvy,_ = self.tgt_vel.get(self.selected,(0,0,0))
            tspd = math.sqrt(tvx**2+tvy**2)
            tb(panel,f"TGT SPD {tspd:.1f}m/s  ALT {sel_pos[2]:.1f}m",
               8,y,0.48,(0,60,255))

        # Border
        cv2.rectangle(panel,(0,0),(PANEL_W-1,PANEL_H-1),(0,160,200),2)


def main():
    rclpy.init()
    node = X5SwarmHybrid()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
