#!/usr/bin/env python3
"""Candidate Track 1 DeepSpeed driver: pure pursuit with preview braking + LiDAR guard.

Copy BOTH driver.py and hybrid_core.py to race_ws/src/team_driver/team_driver/
Back up your original driver.py first. The official centreline stays untouched.

Original implementation; standard algorithm inspirations:
NTU DeepSpeed hackathon documentation, and open F1TENTH gap-follow examples.
"""
import math
import hashlib
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

from team_driver.hybrid_core import HybridController


class Driver(Node):
    def __init__(self):
        super().__init__('driver')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('raceline_csv', str(Path(get_package_share_directory('team_driver')) / 'config' / 'icra26_clearance.csv'))
        self.declare_parameter('max_speed', 6.2)          # m/s: change one parameter at a time
        self.declare_parameter('lateral_accel', 10.0)      # m/s^2
        self.declare_parameter('brake_accel', 3.5)        # m/s^2
        self.declare_parameter('max_range', 8.0)         # metres: LiDAR clipping distance

        self.declare_parameter('accel_limit', 4.5)
        self.declare_parameter('lookahead_scale', 0.90)
        self.declare_parameter('project_progress', False)
        self.declare_parameter('use_speed_profile', False)
        self.declare_parameter('response_delay', 0.0)
        self.declare_parameter('prediction_horizon', 0.0)

        path = Path(str(self.get_parameter('raceline_csv').value))
        if not path.is_file():
            raise FileNotFoundError(f"Centreline missing: {path}. Check repository mount.")
        waypoints = np.loadtxt(str(path), delimiter=',', comments='#')
        self.controller = HybridController(
            waypoints,
            max_speed=float(self.get_parameter('max_speed').value),
            lateral_accel=float(self.get_parameter('lateral_accel').value),
            brake_accel=float(self.get_parameter('brake_accel').value),
            accel_limit=float(self.get_parameter('accel_limit').value),
            lookahead_scale=float(self.get_parameter('lookahead_scale').value),
            project_progress=bool(self.get_parameter('project_progress').value),
            use_speed_profile=bool(self.get_parameter('use_speed_profile').value),
            response_delay=float(self.get_parameter('response_delay').value),
            prediction_horizon=float(self.get_parameter('prediction_horizon').value),
        )
        self.max_range = float(self.get_parameter('max_range').value)
        self.position = None
        self.yaw = 0.0
        self.speed = 0.0
        self.yaw_rate = 0.0
        self.beta = 0.0
        self.odom_stamp = None
        self.last_stamp = None
        self.scan_count = 0
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, str(self.get_parameter('drive_topic').value), 10)
        self.create_subscription(
            LaserScan, str(self.get_parameter('scan_topic').value), self.scan_callback, 10)
        self.create_subscription(
            Odometry, str(self.get_parameter('odom_topic').value), self.odom_callback, 10)
        self.get_logger().info(f"Hybrid loaded {len(waypoints)} points from {path}")
        settings = {name: self.get_parameter(name).value for name in
                    ('max_speed', 'lateral_accel', 'brake_accel', 'accel_limit',
                     'lookahead_scale', 'project_progress', 'max_range',
                     'use_speed_profile', 'response_delay', 'prediction_horizon')}
        self.get_logger().info(f"CONTROLLER_CONFIG {settings} "
                               f"line_sha256={hashlib.sha256(path.read_bytes()).hexdigest()}")

    def odom_callback(self, msg):
        self.odom_stamp = float(msg.header.stamp.sec) + 1e-9 * float(msg.header.stamp.nanosec)
        self.position = (float(msg.pose.pose.position.x), float(msg.pose.pose.position.y))
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y),
                              1.0 - 2.0*(q.y*q.y + q.z*q.z))
        self.speed = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)
        self.yaw_rate = float(msg.twist.twist.angular.z)
        self.beta = math.atan2(float(msg.twist.twist.linear.y), float(msg.twist.twist.linear.x))

    def scan_callback(self, msg):
        if self.position is None:
            return
        ranges = np.asarray(msg.ranges, dtype=np.float64)
        ranges = np.nan_to_num(ranges, nan=0.0, posinf=self.max_range, neginf=0.0)
        ranges = np.clip(ranges, 0.0, self.max_range)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        stamp = float(msg.header.stamp.sec) + 1e-9 * float(msg.header.stamp.nanosec)
        dt = 0.025 if self.last_stamp is None else stamp - self.last_stamp
        self.last_stamp = stamp
        age = stamp - self.odom_stamp
        self.controller.motion_state = ((self.yaw_rate, self.beta)
            if 0 <= age <= .03 else None)
        cmd = self.controller.step(
            self.position[0], self.position[1], self.yaw, self.speed,
            ranges, angles, dt=dt)
        out = AckermannDriveStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.drive.steering_angle = float(cmd.steering)
        out.drive.speed = float(cmd.speed)
        self.drive_pub.publish(out)
        self.scan_count += 1
        if self.scan_count % 8 == 0:
            self.get_logger().info(
                f"{cmd.mode} v_cmd={cmd.speed:.2f} steering={cmd.steering:+.2f} "
                f"track_error={cmd.path_error:.2f} front={cmd.front_range:.2f} "
                f"sim_t={stamp:.3f} x={self.position[0]:.3f} "
                f"y={self.position[1]:.3f} yaw={self.yaw:.3f} "
                f"v_actual={self.speed:.3f} "
                f"v_plan={cmd.planned_speed:.3f} v_guard={cmd.guarded_speed:.3f} "
                f"odom_age={stamp - self.odom_stamp:.4f} dt={dt:.4f} "
                f"idx={cmd.nearest_index} yaw_rate={self.yaw_rate:.4f} "
                f"beta={self.beta:.4f} pred_delta={self.controller.prediction_delta:.4f}")


def main(args=None):
    rclpy.init(args=args)
    node = Driver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
