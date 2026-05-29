#!/usr/bin/env python3
"""
X3 Target Drone — random waypoint flight.
Spawns airborne at (20, 20, 15).  Changes direction every 3 s.
Topic: /target_drone/cmd_vel  (bridged → /model/target_drone/cmd_vel)
"""

import math
import random
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

SPAWN_POS   = (20.0, 20.0, 15.0)
TARGET_ALT  = 15.0          # m  — altitude to hold
MIN_ALT     = 8.0
MAX_ALT     = 25.0
SPEED_MIN   = 3.0           # m/s
SPEED_MAX   = 6.0
DIR_CHANGE  = 3.0           # seconds between direction changes
ALT_KP      = 0.5           # altitude correction gain


class X3TargetDrone(Node):
    def __init__(self):
        super().__init__("x3_target_drone")
        self.pub = self.create_publisher(Twist, "/target_drone/cmd_vel", 10)

        self.vx = 0.0
        self.vy = 0.0
        self.alt = SPAWN_POS[2]         # estimated altitude (integrated)
        self.dir_timer = time.time()
        self.phase = "hover"
        self.phase_start = time.time()

        self.timer = self.create_timer(0.05, self.control_loop)   # 20 Hz
        self.get_logger().info(
            f"X3 Target Drone ready  |  spawn={SPAWN_POS}  |  "
            f"topic=/target_drone/cmd_vel"
        )

    # ------------------------------------------------------------------ #
    def _new_direction(self):
        speed = random.uniform(SPEED_MIN, SPEED_MAX)
        angle = random.uniform(0.0, 2.0 * math.pi)
        self.vx = speed * math.cos(angle)
        self.vy = speed * math.sin(angle)
        self.get_logger().info(
            f"X3 new heading: vx={self.vx:+.1f}  vy={self.vy:+.1f}  "
            f"speed={speed:.1f} m/s"
        )

    # ------------------------------------------------------------------ #
    def control_loop(self):
        now   = time.time()
        elapsed = now - self.phase_start
        msg   = Twist()

        # --- Phase: initial hover to stabilise -------------------------
        if self.phase == "hover":
            if elapsed >= 2.0:
                self._new_direction()
                self.phase = "fly"
                self.phase_start = now
            # stay put
            self.pub.publish(msg)
            return

        # --- Phase: random flight --------------------------------------
        if now - self.dir_timer >= DIR_CHANGE:
            self._new_direction()
            self.dir_timer = now

        # Altitude soft-hold: nudge z if drifting
        alt_err = TARGET_ALT - self.alt
        vz = ALT_KP * alt_err
        vz = max(-2.0, min(2.0, vz))

        msg.linear.x = self.vx
        msg.linear.y = self.vy
        msg.linear.z = vz
        self.pub.publish(msg)

        # Integrate estimated altitude (rough dead-reckoning)
        self.alt += vz * 0.05
        self.alt = max(MIN_ALT, min(MAX_ALT, self.alt))


# ------------------------------------------------------------------ #
def main():
    rclpy.init()
    node = X3TargetDrone()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub.publish(Twist())        # stop motors
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
