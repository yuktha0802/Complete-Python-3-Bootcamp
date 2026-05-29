#!/usr/bin/env python3
"""
X5 Interceptor Drone
====================
YAW-ROTATION FIX
────────────────
angular.z has NO effect on X5 under either VelocityControl or
MulticopterVelocityControl in this Gazebo Harmonic setup.
Consequence: X5's yaw is **permanently fixed** at its initial orientation
(yaw = 0, i.e. body +X == world +X).

Fix: since body frame never rotates, body frame ≡ world frame forever.
We compute the 3-D pursuit vector in world frame and send it directly as
linear velocity — no rotation required.

    cmd.linear.x  =  world +X velocity component
    cmd.linear.y  =  world +Y velocity component
    cmd.linear.z  =  world +Z velocity component

Navigation uses dead-reckoning (velocity integration) to track both drones.
Camera is displayed as an Aircraft-Control HUD; detection refines pursuit.

Topics required (already in your bridge command):
  /target_drone/cmd_vel         geometry_msgs/Twist  (X3 velocity — subscribed)
  /model/x5_uav/cmd_vel         geometry_msgs/Twist  (X5 command — published)
  /world/defense_zone/model/x5_uav/link/camera_link/sensor/forward_camera/image
                                sensor_msgs/Image

Spawn positions (must match Gazebo world):
  X3 (target)     : (20, 20, 15)
  X5 (interceptor): ( 0,  0, 10)
"""

import math
import threading
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

# ── optional YOLO ─────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

# ── constants ──────────────────────────────────────────────────────────────────
X3_SPAWN = np.array([20.0, 20.0, 15.0])
X5_SPAWN = np.array([ 0.0,  0.0, 10.0])

CONTROL_HZ      = 20        # Hz
DISPLAY_HZ      = 30        # Hz

MAX_SPEED       = 8.0       # m/s  pursuit speed cap
KP_PURSUIT      = 1.2       # proportional gain
LEAD_FACTOR     = 1.0       # seconds to lead X3 by
INTERCEPT_DIST  = 3.0       # m   — declare intercept inside this radius
MIN_ALT         = 3.0       # m   — floor
HOVER_DURATION  = 2.0       # s   — stabilisation at spawn

CAM_TOPIC = (
    "/world/defense_zone/model/x5_uav/link/"
    "camera_link/sensor/forward_camera/image"
)
CAM_FOV_DEG = 180.0         # horizontal FOV of X5 camera


# ══════════════════════════════════════════════════════════════════════════════
class X5Interceptor(Node):

    # ── init ──────────────────────────────────────────────────────────────────
    def __init__(self):
        super().__init__("x5_interceptor")

        # publishers / subscribers
        self.cmd_pub = self.create_publisher(Twist, "/model/x5_uav/cmd_vel", 10)

        self.create_subscription(
            Twist, "/target_drone/cmd_vel", self._x3_vel_cb, 10
        )

        cam_qos = QoSProfile(depth=1)
        cam_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.create_subscription(Image, CAM_TOPIC, self._camera_cb, cam_qos)

        self.bridge = CvBridge()

        # ── dead-reckoning position state ────────────────────────────────────
        self.x5_pos  = X5_SPAWN.copy()   # integrated from own commands
        self.x3_pos  = X3_SPAWN.copy()   # integrated from X3 vel subscription
        self.x5_vel  = np.zeros(3)       # last sent command
        self.x3_vel  = np.zeros(3)       # latest received X3 velocity
        self._last_t = time.time()

        # ── camera state ─────────────────────────────────────────────────────
        self._frame      = None
        self._frame_lock = threading.Lock()
        self._det_box    = None          # (x1,y1,x2,y2) or None

        # ── optional YOLO ────────────────────────────────────────────────────
        self.yolo = None
        if _YOLO_AVAILABLE:
            try:
                self.yolo = YOLO("yolov8n.pt")
                self.get_logger().info("YOLO loaded (yolov8n)")
            except Exception as e:
                self.get_logger().warn(f"YOLO load failed: {e}")

        # ── state machine ────────────────────────────────────────────────────
        self.state       = "HOVER"
        self._state_t    = time.time()

        # ── timers ───────────────────────────────────────────────────────────
        self.create_timer(1.0 / CONTROL_HZ, self._control_loop)
        self.create_timer(1.0 / DISPLAY_HZ,  self._display_loop)

        self.get_logger().info(
            "X5 Interceptor ready\n"
            "  YAW FIX: angular.z ignored — body frame == world frame (yaw=0)\n"
            "  Navigation: dead-reckoning via velocity integration\n"
            f"  X5 spawn={X5_SPAWN.tolist()}  X3 spawn={X3_SPAWN.tolist()}"
        )

    # ── callbacks ─────────────────────────────────────────────────────────────
    def _x3_vel_cb(self, msg: Twist):
        self.x3_vel = np.array([msg.linear.x, msg.linear.y, msg.linear.z])

    def _camera_cb(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self._frame_lock:
                self._frame = frame
        except Exception as e:
            self.get_logger().warn(f"Camera cb error: {e}", throttle_duration_sec=5)

    # ── position integration ──────────────────────────────────────────────────
    def _integrate(self):
        now = time.time()
        dt  = min(now - self._last_t, 0.1)   # cap at 100 ms
        self._last_t = now

        self.x3_pos += self.x3_vel  * dt
        self.x5_pos += self.x5_vel  * dt

        # altitude floor
        self.x3_pos[2] = max(self.x3_pos[2], MIN_ALT)
        self.x5_pos[2] = max(self.x5_pos[2], MIN_ALT)

    # ── pursuit velocity (world frame) ────────────────────────────────────────
    def _compute_pursuit_vel(self) -> np.ndarray:
        """
        Lead pursuit:  aim at where X3 will be in LEAD_FACTOR seconds.
        Returns world-frame velocity vector (clamped to MAX_SPEED).

        KEY FIX: Because yaw is fixed (angular.z non-functional), the
        returned vector is sent directly as cmd.linear.[x,y,z] — no
        coordinate-frame rotation needed.
        """
        dist = float(np.linalg.norm(self.x3_pos - self.x5_pos))
        lead = LEAD_FACTOR * (dist / max(MAX_SPEED, 1.0))
        aim  = self.x3_pos + self.x3_vel * lead

        delta = aim - self.x5_pos
        d     = float(np.linalg.norm(delta))
        if d < 0.01:
            return np.zeros(3)

        speed = min(KP_PURSUIT * d, MAX_SPEED)
        return (delta / d) * speed

    # ── detection ─────────────────────────────────────────────────────────────
    def _detect_x3(self, frame) -> tuple | None:
        """Return (x1,y1,x2,y2) bounding box of X3 or None."""
        if frame is None:
            return None

        # ── YOLO attempt (airplane / bird / kite classes) ─────────────────
        if self.yolo is not None:
            try:
                res = self.yolo(frame, verbose=False, conf=0.3, classes=[4, 14, 33])
                if res and len(res[0].boxes):
                    b = res[0].boxes[0]
                    return tuple(map(int, b.xyxy[0]))
            except Exception:
                pass

        # ── colour / contour fallback ─────────────────────────────────────
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # orange UAV body
        mask = cv2.inRange(hsv, np.array([5, 80, 80]), np.array([30, 255, 255]))
        # also catch grey/dark against bright sky
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sky  = cv2.inRange(gray, 180, 255)
        dark = cv2.bitwise_and(cv2.bitwise_not(sky), cv2.bitwise_not(sky))
        combined = cv2.bitwise_or(mask, dark)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,
                                    np.ones((3, 3), np.uint8))

        cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c    = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if 40 < area < 8000:
                x, y, w, h = cv2.boundingRect(c)
                return (x, y, x + w, y + h)
        return None

    # ── control loop ──────────────────────────────────────────────────────────
    def _control_loop(self):
        self._integrate()

        dist    = float(np.linalg.norm(self.x3_pos - self.x5_pos))
        elapsed = time.time() - self._state_t
        cmd     = Twist()

        # ── HOVER: stabilise at spawn ──────────────────────────────────────
        if self.state == "HOVER":
            if elapsed >= HOVER_DURATION:
                self.state    = "PURSUE"
                self._state_t = time.time()
                self.get_logger().info(
                    f"→ PURSUE  X3 est. pos={self.x3_pos.round(1)}  "
                    f"dist={dist:.1f} m"
                )
            # hold position (zero cmd)

        # ── PURSUE: lead-pursuit toward X3 ────────────────────────────────
        elif self.state == "PURSUE":
            if dist <= INTERCEPT_DIST:
                self.state    = "INTERCEPT"
                self._state_t = time.time()
                self.get_logger().info("★ INTERCEPT achieved!")
            else:
                vel = self._compute_pursuit_vel()
                # ── THE YAW FIX ────────────────────────────────────────────
                # angular.z does nothing; yaw stays at 0.
                # Body frame == world frame, so set linear directly:
                cmd.linear.x = float(vel[0])
                cmd.linear.y = float(vel[1])
                cmd.linear.z = float(vel[2])
                cmd.angular.z = 0.0      # explicit no-op
                self.x5_vel   = vel.copy()

        # ── INTERCEPT: shadow target ───────────────────────────────────────
        elif self.state == "INTERCEPT":
            # mirror X3 velocity to stay close
            cmd.linear.x = float(self.x3_vel[0])
            cmd.linear.y = float(self.x3_vel[1])
            cmd.linear.z = float(self.x3_vel[2])
            self.x5_vel  = self.x3_vel.copy()

            if dist > INTERCEPT_DIST * 2.5:
                self.state    = "PURSUE"
                self._state_t = time.time()
                self.get_logger().info(f"Re-engaging pursuit  dist={dist:.1f} m")

        self.cmd_pub.publish(cmd)

    # ── HUD drawing ───────────────────────────────────────────────────────────
    def _draw_hud(self, canvas: np.ndarray) -> np.ndarray:
        H, W = canvas.shape[:2]
        PANEL = 230             # right-panel width (px)
        cam_w = W - PANEL

        # ── right panel background ────────────────────────────────────────
        overlay = canvas.copy()
        cv2.rectangle(overlay, (cam_w, 0), (W, H), (15, 15, 25), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
        cv2.line(canvas, (cam_w, 0), (cam_w, H), (0, 180, 255), 1)

        # ── helpers ───────────────────────────────────────────────────────
        def txt(text, y, color=(210, 210, 210), scale=0.4, bold=1):
            cv2.putText(canvas, text, (cam_w + 6, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, bold)

        def sep(y):
            cv2.line(canvas, (cam_w + 4, y), (W - 4, y), (60, 60, 80), 1)

        # ── title ─────────────────────────────────────────────────────────
        txt("AIRCRAFT CTRL", 22, (0, 220, 255), 0.5, 2)
        sep(28)

        # ── X5 block ──────────────────────────────────────────────────────
        spd5 = float(np.linalg.norm(self.x5_vel))
        txt("[ X5  INTERCEPTOR ]", 46, (80, 255, 80))
        txt(f"POS  {self.x5_pos[0]:+6.1f} {self.x5_pos[1]:+6.1f}", 62)
        txt(f"ALT  {self.x5_pos[2]:5.1f} m",  78)
        txt(f"SPD  {spd5:5.2f} m/s",           94)
        txt("YAW  FIXED 0° (angular.z n/a)", 110, (255, 165, 0))
        sep(116)

        # ── X3 block ──────────────────────────────────────────────────────
        spd3 = float(np.linalg.norm(self.x3_vel))
        txt("[ X3  TARGET      ]", 132, (80, 120, 255))
        txt(f"EST  {self.x3_pos[0]:+6.1f} {self.x3_pos[1]:+6.1f}", 148)
        txt(f"ALT  {self.x3_pos[2]:5.1f} m",  164)
        txt(f"SPD  {spd3:5.2f} m/s",           180)
        sep(186)

        # ── engagement ────────────────────────────────────────────────────
        delta      = self.x3_pos - self.x5_pos
        dist       = float(np.linalg.norm(delta))
        bearing    = math.degrees(math.atan2(delta[1], delta[0]))
        closure    = -float(np.dot(delta / max(dist, 0.01), self.x5_vel))

        txt("ENGAGEMENT", 202, (0, 220, 255))
        txt(f"DIST     {dist:6.1f} m",   218)
        txt(f"BEARING  {bearing:+6.0f}°", 234)
        txt(f"CLOSURE  {closure:+5.1f} m/s", 250)
        sep(256)

        # ── state badge ───────────────────────────────────────────────────
        state_col = {
            "HOVER":     (0, 230, 230),
            "PURSUE":    (0, 165, 255),
            "INTERCEPT": (0, 255, 60),
        }.get(self.state, (200, 200, 200))
        cv2.rectangle(canvas, (cam_w + 4, H - 36), (W - 4, H - 4),
                      (30, 30, 40), -1)
        cv2.putText(canvas, f" {self.state}", (cam_w + 8, H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_col, 2)

        # ── mini-map (bottom-left of camera area) ─────────────────────────
        MM  = 120          # mini-map size px
        PAD = 8
        mx0 = PAD
        my0 = H - MM - PAD
        cv2.rectangle(canvas, (mx0, my0), (mx0 + MM, my0 + MM),
                      (20, 20, 30), -1)
        cv2.rectangle(canvas, (mx0, my0), (mx0 + MM, my0 + MM),
                      (80, 80, 100), 1)
        cv2.putText(canvas, "MAP", (mx0 + 4, my0 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 140), 1)

        # scale: world ±40 m around midpoint → MM pixels
        mid   = (self.x5_pos[:2] + self.x3_pos[:2]) / 2.0
        scale = MM / 80.0

        def world_to_mm(wx, wy):
            px = int(mx0 + MM // 2 + (wx - mid[0]) * scale)
            py = int(my0 + MM // 2 - (wy - mid[1]) * scale)
            return (np.clip(px, mx0 + 2, mx0 + MM - 2),
                    np.clip(py, my0 + 2, my0 + MM - 2))

        # X3 (red)
        p3 = world_to_mm(self.x3_pos[0], self.x3_pos[1])
        cv2.circle(canvas, p3, 5, (50, 80, 255), -1)
        cv2.putText(canvas, "X3", (p3[0] + 6, p3[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (50, 80, 255), 1)

        # X5 (green)
        p5 = world_to_mm(self.x5_pos[0], self.x5_pos[1])
        cv2.circle(canvas, p5, 5, (50, 220, 50), -1)
        cv2.putText(canvas, "X5", (p5[0] + 6, p5[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (50, 220, 50), 1)

        # pursuit vector line
        cv2.arrowedLine(canvas, p5, p3, (0, 165, 255), 1, tipLength=0.3)

        # ── camera crosshair + detection box ──────────────────────────────
        cx_c = cam_w // 2
        cy_c = H // 2
        cv2.line(canvas, (cx_c - 25, cy_c), (cx_c + 25, cy_c), (0, 255, 0), 1)
        cv2.line(canvas, (cx_c, cy_c - 25), (cx_c, cy_c + 25), (0, 255, 0), 1)
        cv2.circle(canvas, (cx_c, cy_c), 45, (0, 255, 0), 1)

        if self._det_box:
            x1, y1, x2, y2 = self._det_box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 60, 255), 2)
            cv2.putText(canvas, "X3 LOCKED", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 60, 255), 1)
            cx_d = (x1 + x2) // 2
            cy_d = (y1 + y2) // 2
            cv2.line(canvas, (cx_c, cy_c), (cx_d, cy_d), (0, 60, 255), 1)

            # bearing from pixel offset (180° FOV / cam width)
            # pixel offset → angle offset relative to forward direction
            px_offset  = cx_d - (cam_w // 2)
            cam_bearing = (px_offset / (cam_w / 2)) * (CAM_FOV_DEG / 2)
            cv2.putText(canvas, f"cam {cam_bearing:+.0f}°",
                        (x1, y2 + 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (0, 180, 255), 1)

        # ── altitude bar (left edge) ───────────────────────────────────────
        bar_h = int(np.clip(self.x5_pos[2] / 50.0, 0, 1) * (H - 60))
        cv2.rectangle(canvas, (4, H - 50 - bar_h), (18, H - 50),
                      (0, 200, 50), -1)
        cv2.rectangle(canvas, (4, 30), (18, H - 50), (70, 70, 90), 1)
        cv2.putText(canvas, "ALT", (1, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (180, 180, 200), 1)

        return canvas

    # ── display loop ──────────────────────────────────────────────────────────
    def _display_loop(self):
        with self._frame_lock:
            raw = self._frame.copy() if self._frame is not None else None

        if raw is None:
            canvas = np.zeros((480, 1280, 3), dtype=np.uint8)
            cv2.putText(canvas, "Waiting for camera...", (480, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 120, 255), 2)
        else:
            canvas = raw.copy()

        # run detection on camera portion only
        H, W = canvas.shape[:2]
        cam_view = canvas[:, :W - 230]
        self._det_box = self._detect_x3(cam_view)

        canvas = self._draw_hud(canvas)
        cv2.imshow("X5 Interceptor — Aircraft Control Panel", canvas)
        cv2.waitKey(1)


# ══════════════════════════════════════════════════════════════════════════════
def main():
    rclpy.init()
    node = X5Interceptor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())    # stop motors
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
