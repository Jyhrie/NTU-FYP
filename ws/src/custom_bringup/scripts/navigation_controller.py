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

    def get_yaw_from_odom(self, odom):
        q = odom.pose.pose.orientation
        yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        return yaw

    def run_once(self):
        rospy.loginfo("Waiting for /local_costmap and /odom...")
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            if self.have_map and self.have_odom:
                break
            rate.sleep()

        msg, origin, end_position, goal_forward_vector = self.local_occupancy_movement.trigger(self.local_map_msg)
        if msg is not None:
            self.debug_pub.publish(msg)

        print(origin, end_position, goal_forward_vector)

         # 1. Compute angle to goal

        res = self.local_occupancy_movement.resolution
        dx = (end_position.x - origin.x) * res
        dy = (end_position.y - origin.y) * res

        target_angle = math.atan2(dy, dx)

        current_yaw = self.get_yaw_from_odom(self.odom)

        angle_error = angle_normalize(target_angle - current_yaw)

        
        twist = Twist()
        while abs(angle_error) > self.angle_tol and not rospy.is_shutdown():
            twist.angular.z = max(-self.rot_max, min(self.rot_max, self.rot_k * angle_error))
            twist.linear.x = 0.0  # rotate in place
            self.cmd_pub.publish(twist)
            rospy.sleep(0.05)
            current_yaw = self.get_yaw_from_odom(self.odom)
            angle_error = angle_normalize(target_angle - current_yaw)

        distance = math.hypot(dx, dy)
        while distance > self.dist_tol and not rospy.is_shutdown():
            twist.linear.x = max(-self.lin_max, min(self.lin_max, self.lin_k * distance))
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)
            rospy.sleep(0.05)
            
            # recompute position relative to robot
            dx = (end_position.x - origin.x) * res
            dy = (end_position.y - origin.y) * res
            distance = math.hypot(dx, dy)

        goal_angle = math.atan2(goal_forward_vector.y, goal_forward_vector.x)
        current_yaw = self.get_yaw_from_odom(self.odom)
        angle_error = angle_normalize(goal_angle - current_yaw)

        while abs(angle_error) > self.angle_tol and not rospy.is_shutdown():
            twist.angular.z = max(-self.rot_max, min(self.rot_max, self.rot_k * angle_error))
            twist.linear.x = 0.0
            self.cmd_pub.publish(twist)
            rospy.sleep(0.05)
            current_yaw = self.get_yaw_from_odom(self.odom)
            angle_error = angle_normalize(goal_angle - current_yaw)



if __name__ == "__main__":
    nav = NavigationController()
    nav.run_once()
