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
    
    def numpy_to_transform(self, grid, x, y):
        size_x = grid.shape[1]
        size_y = grid.shape[0]

        return x, size_y - y - 1
    
    def normalize_angle(self, a):
        return math.atan2(math.sin(a), math.cos(a))


    def run_once(self):
        rospy.loginfo("Waiting for /local_costmap and /odom...")
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            if self.have_map and self.have_odom:
                break
            rate.sleep()

        # get local map info
        msg, origin, end_position, goal_forward_vector = self.local_occupancy_movement.trigger(self.local_map_msg)
        if msg is not None:
            self.debug_pub.publish(msg)

        print("Origin:", origin)
        print("End Position:", end_position)
        print("Goal Forward Vector:", goal_forward_vector)

        # data = np.array(msg.data, dtype=np.int8)
        # map = data.reshape((self.map_height, self.map_width))

        # self.numpy_to_transform(map, origin.x, origin.y)
        res = self.local_occupancy_movement.resolution

        # Compute relative position in meters
        dx_rel = end_position.x - origin.x
        dy_rel = origin.y - end_position.y  # NEGATE dy because -Y is forward
        dx = dx_rel * res
        dy = dy_rel * res  # NEGATE dy because -Y is forward

        print(dx_rel, dy_rel)

        twist = Twist()

        # ----------------------
        # Rotate to face the goal
        # ----------------------
        # target_angle = math.atan2(dy, dx)
        # print(target_angle)
        # current_yaw = self.get_yaw_from_odom(self.odom)
        # angle_error = angle_normalize(target_angle - current_yaw)
        # angle_error = (target_angle - current_yaw + math.pi) % (2*math.pi) - math.pi

        
        # while abs(angle_error) > self.angle_tol and not rospy.is_shutdown():
        #     twist.angular.z = max(-self.rot_max, min(self.rot_max, self.rot_k * angle_error))
        #     twist.linear.x = 0.0
        #     self.cmd_pub.publish(twist)
        #     rospy.sleep(0.05)
        #     current_yaw = self.get_yaw_from_odom(self.odom)
        #     angle_error = angle_normalize(target_angle - current_yaw)

        # # ----------------------
        # # Move straight to goal
        # # ----------------------
        distance = math.hypot(dx, dy)
        print(distance)
        # while distance > self.dist_tol and not rospy.is_shutdown():
        #     twist.linear.x = max(-self.lin_max, min(self.lin_max, self.lin_k * distance))
        #     twist.angular.z = 0.0
        #     self.cmd_pub.publish(twist)
        #     rospy.sleep(0.05)

        #     dx = (end_position.x - origin.x) * res
        #     dy = (origin.y - end_position.y) * res  # still negate dy
        #     distance = math.hypot(dx, dy)

        # # ----------------------
        # # Rotate to match goal orientation
        # # ----------------------


        goal_angle = math.atan2(goal_forward_vector.x, goal_forward_vector.y)

        forward = (0, -1)
        goal = (goal_forward_vector.x, goal_forward_vector.y)

        dot = forward[0]*goal[0] + forward[1]*goal[1]
        cross = forward[0]*goal[1] - forward[1]*goal[0]

        current_yaw = self.get_yaw_from_odom(self.odom)

        rotation_angle = math.atan2(cross, dot)

        target_yaw = current_yaw - rotation_angle
        target_yaw = self.normalize_angle(target_yaw)

        # --- control params ---
        Kp = 1.5
        max_ang = 1.0
        tol = math.radians(2)

        cmd = Twist()
        rate = rospy.Rate(30)

        # --- rotate in place ---
        while not rospy.is_shutdown():
            current_yaw = self.get_yaw_from_odom(self.odom)

            error = target_yaw - current_yaw
            error = math.atan2(math.sin(error), math.cos(error))  # normalize

            if abs(error) < tol:
                break

            ang = Kp * error
            ang = max(-max_ang, min(max_ang, ang))

            cmd.linear.x = 0.0
            cmd.angular.z = ang
            self.cmd_pub.publish(cmd)

            rate.sleep()

        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)




if __name__ == "__main__":
    nav = NavigationController()
    nav.run_once()
