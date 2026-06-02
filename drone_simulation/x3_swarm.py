#!/usr/bin/env python3
"""
x3_swarm.py — Target drones with wide random flight paths.
High speed + long direction intervals = drones travel far across the map.

═══════════════════════════════════════════════════════════════
BRIDGE COMMAND (add these to your existing bridge call):
  /model/target_drone/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist
  /model/target_drone_3/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist
  /model/target_drone_4/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist
  /model/target_drone_5/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist
  /model/target_drone_6/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist
  /model/target_drone_7/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist

Set NUM_DRONES below to match how many target drones are spawned
in your Gazebo world (default 6). Unused topics are silently ignored
by Gazebo but waste bridge resources if bridged unnecessarily.
═══════════════════════════════════════════════════════════════
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import random
import math
import time

# ── Configuration ──────────────────────────────────────────────────────────
NUM_DRONES  = 6          # must match number of target drones spawned in Gazebo
SPEED       = 8.0        # m/s
ALT_MIN     = 8.0        # m  — floor (prevent ground collision)
ALT_MAX     = 30.0       # m  — ceiling
ALT_HOLD    = 15.0       # m  — spawn altitude (from world file)

# Each drone gets its own direction-change interval for varied behaviour
CHANGE_INTERVALS = [8.0, 9.0, 10.0, 11.0, 8.5, 9.5]

# Topic for each drone — must match Gazebo model names
def _topic(i):
    if i == 0:
        return "/model/target_drone/cmd_vel"
    return f"/model/target_drone_{i+2}/cmd_vel"   # 0→drone, 1→drone_3, 2→drone_4 …

DRONE_TOPICS = [_topic(i) for i in range(NUM_DRONES)]


class X3Swarm(Node):
    def __init__(self):
        super().__init__("x3_swarm")
        self.drone_pubs = [
            self.create_publisher(Twist, t, 10) for t in DRONE_TOPICS
        ]
        self.last_change  = [0.0]  * NUM_DRONES
        self.curr_vz      = [0.0]  * NUM_DRONES   # track z vel for clamping
        self.est_alt      = [ALT_HOLD] * NUM_DRONES  # dead-reckoned altitude
        self.started      = False
        self.start_time   = time.time()
        self.create_timer(0.05, self.update)   # 20 Hz
        self.get_logger().info(
            f"X3 Swarm: {NUM_DRONES} drones on topics {DRONE_TOPICS[0]} … "
            f"{DRONE_TOPICS[-1]}  — launching in 2 s"
        )

    # ── helpers ────────────────────────────────────────────────────────────
    def new_direction(self, idx):
        angle = random.uniform(0, 2 * math.pi)
        t     = Twist()
        t.linear.x = SPEED * math.cos(angle)
        t.linear.y = SPEED * math.sin(angle)

        # Altitude correction: nudge toward ALT_HOLD if out of bounds
        alt = self.est_alt[idx]
        if alt < ALT_MIN + 2.0:
            t.linear.z = random.uniform(1.0, 2.5)   # climb
        elif alt > ALT_MAX - 2.0:
            t.linear.z = random.uniform(-2.5, -1.0) # descend
        else:
            t.linear.z = random.uniform(-1.0, 1.0)  # slight variation

        self.curr_vz[idx] = t.linear.z
        self.drone_pubs[idx].publish(t)
        self.get_logger().info(
            f"Drone {idx+1}: vx={t.linear.x:.1f}  vy={t.linear.y:.1f}  "
            f"vz={t.linear.z:.1f}  est_alt={alt:.1f}m"
        )

    # ── main update ────────────────────────────────────────────────────────
    def update(self):
        now = time.time()

        # Integrate estimated altitude at 20 Hz (dt=0.05)
        for idx in range(NUM_DRONES):
            self.est_alt[idx] += self.curr_vz[idx] * 0.05
            self.est_alt[idx]  = max(ALT_MIN, min(ALT_MAX, self.est_alt[idx]))

        if not self.started and now - self.start_time >= 2.0:
            self.started = True
            for i in range(NUM_DRONES):
                self.new_direction(i)
                self.last_change[i] = now
            self.get_logger().info("All target drones moving!")
            return

        if not self.started:
            return

        for i in range(NUM_DRONES):
            if now - self.last_change[i] >= CHANGE_INTERVALS[i]:
                self.new_direction(i)
                self.last_change[i] = now


def main():
    rclpy.init()
    node = X3Swarm()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        for p in node.drone_pubs:
            p.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
