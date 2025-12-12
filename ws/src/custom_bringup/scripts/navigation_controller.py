#!/usr/bin/env python3
import rospy
import numpy as np
import math
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

import local_occupancy_movement as lom


def angle_normalize(a):
    """Normalize angle to [-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


class NavigationController:
    def __init__(self):
        rospy.init_node("navigation_controller")

        # subscribers
        rospy.Subscriber("/local_costmap", OccupancyGrid, self.local_costmap_cb)
        rospy.Subscriber("/odom", Odometry, self.odom_cb)

        # publishers
        self.debug_pub = rospy.Publisher("/debug_map", OccupancyGrid, queue_size=1)
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        # occupancy movement module (your object)
        self.local_occupancy_movement = lom.LocalOccupancyNavigator()

        # storage
        self.local_map_msg = None         # raw OccupancyGrid msg used by trigger()
        self.have_map = False

        self.odom = None                  # latest Odometry msg
        self.have_odom = False

        # control parameters (tweak to taste)
        self.rot_k = 1.2          # angular P gain
        self.rot_max = 0.8        # max angular speed (rad/s)
        self.lin_k = 0.8          # linear P gain (used to scale speed by distance)
        self.lin_max = 0.35       # max linear speed (m/s)

        # thresholds
        self.angle_tol = 0.04     # rad ~ 2.3 deg
        self.dist_tol = 0.03      # meters

    # -------------------------
    # ROS callbacks
    # -------------------------
    def local_costmap_cb(self, msg: OccupancyGrid):
        self.local_map_msg = msg
        self.have_map = True

    def odom_cb(self, msg: Odometry):
        self.odom = msg
        self.have_odom = True

    # -------------------------
    # helpers: odom pose
    # -------------------------
    def get_odom_pose(self):
        """Return (x, y, yaw) from latest odom message."""
        if not self.have_odom:
            return None
        px = self.odom.pose.pose.position.x
        py = self.odom.pose.pose.position.y
        q = self.odom.pose.pose.orientation
        # convert quaternion to yaw
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny, cosy)
        return px, py, yaw

    # -------------------------
    # coordinate conversion
    # -------------------------
    def robot_relative_to_world(self, rel_x, rel_y, odom_yaw, odom_x, odom_y):
        """
        Convert your robot-relative coordinates (rel_x, rel_y) into world coords.
        User's convention: -Y is forward, +X is right.
        We map that to the standard robot frame (x_forward, y_left):
            x_forward = -rel_y
            y_left    = -rel_x
        Then rotate by odom_yaw (standard) and translate by odom position.
        """
        x_fwd = -rel_y
        y_left = -rel_x

        wx = x_fwd * math.cos(odom_yaw) - y_left * math.sin(odom_yaw)
        wy = x_fwd * math.sin(odom_yaw) + y_left * math.cos(odom_yaw)

        return odom_x + wx, odom_y + wy

    def robot_vector_to_world_vector(self, fx, fy, odom_yaw):
        """Convert a robot-relative direction vector (fx, fy) to world direction vector."""
        # map to standard robot-frame (x forward, y left)
        x_fwd = -fy
        y_left = -fx

        wx = x_fwd * math.cos(odom_yaw) - y_left * math.sin(odom_yaw)
        wy = x_fwd * math.sin(odom_yaw) + y_left * math.cos(odom_yaw)

        return wx, wy

    # -------------------------
    # low-level motion primitives (blocking)
    # -------------------------
    def rotate_to_angle(self, desired_yaw):
        """Rotate to desired yaw (world frame) using odom yaw feedback."""
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            pose = self.get_odom_pose()
            if pose is None:
                rospy.sleep(0.02)
                continue
            _, _, yaw = pose
            err = angle_normalize(desired_yaw - yaw)
            if abs(err) < self.angle_tol:
                break
            cmd = Twist()
            cmd.angular.z = max(-self.rot_max, min(self.rot_max, self.rot_k * err))
            # small linear damp to avoid drift (optional)
            cmd.linear.x = 0.0
            self.cmd_pub.publish(cmd)
            rate.sleep()

        # stop
        self.cmd_pub.publish(Twist())
        rospy.sleep(0.08)

    def drive_to_point(self, target_x, target_y):
        """Drive to (target_x, target_y) in world frame using odom feedback.
           Keeps heading toward the point by commanding angular vel to reduce heading error.
        """
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            pose = self.get_odom_pose()
            if pose is None:
                rospy.sleep(0.02)
                continue
            ox, oy, yaw = pose
            dx = target_x - ox
            dy = target_y - oy
            dist = math.hypot(dx, dy)
            if dist < self.dist_tol:
                break

            # desired angle in world to face the target
            target_angle = math.atan2(dy, dx)
            ang_err = angle_normalize(target_angle - yaw)

            # proportional controllers
            lin = min(self.lin_max, self.lin_k * dist)
            ang = max(-self.rot_max, min(self.rot_max, self.rot_k * ang_err))

            cmd = Twist()
            cmd.linear.x = lin
            cmd.angular.z = ang
            self.cmd_pub.publish(cmd)
            rate.sleep()

        # stop
        self.cmd_pub.publish(Twist())
        rospy.sleep(0.08)

    # -------------------------
    # main one-shot routine
    # -------------------------
    def run_once(self, timeout=10.0):
        """Wait for a map + odom, run trigger(), move and rotate, then exit."""
        start_t = rospy.Time.now()
        rate = rospy.Rate(10)

        rospy.loginfo("Waiting for /local_costmap and /odom...")
        while not rospy.is_shutdown():
            if self.have_map and self.have_odom:
                break
            if (rospy.Time.now() - start_t).to_sec() > timeout:
                rospy.logerr("Timed out waiting for map/odom")
                return
            rate.sleep()

        rospy.loginfo("Got map & odom — running trigger()")
        # call your trigger with raw OccupancyGrid message (your trigger expects msg)
        msg, origin, end_position, goal_forward_vector = self.local_occupancy_movement.trigger(self.local_map_msg)

        # publish debug map if returned
        if msg is not None:
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "map"
            self.debug_pub.publish(msg)

        # get current odom pose
        od = self.get_odom_pose()
        if od is None:
            rospy.logerr("No odom pose available")
            return
        od_x, od_y, od_yaw = od

        # convert end_position (robot-relative) -> world
        ex_r, ey_r = end_position      # user coords: +X right, -Y forward
        target_world_x, target_world_y = self.robot_relative_to_world(ex_r, ey_r, od_yaw, od_x, od_y)

        rospy.loginfo(f"Target world: ({target_world_x:.3f}, {target_world_y:.3f})")

        # 1) rotate to face target
        target_angle = math.atan2(target_world_y - od_y, target_world_x - od_x)
        self.rotate_to_angle(target_angle)

        # 2) drive to target (closed-loop using odom)
        self.drive_to_point(target_world_x, target_world_y)

        # 3) compute goal facing angle (from goal_forward_vector) -> rotate to it
        gf_x, gf_y = goal_forward_vector
        gwx, gwy = self.robot_vector_to_world_vector(gf_x, gf_y, od_yaw)
        desired_yaw = math.atan2(gwy, gwx)
        self.rotate_to_angle(desired_yaw)

        rospy.loginfo("Move + final rotation complete.")

        # stop and exit
        self.cmd_pub.publish(Twist())
        rospy.sleep(0.2)
        rospy.signal_shutdown("One-shot navigation complete.")


if __name__ == "__main__":
    nav = NavigationController()
    nav.run_once()
