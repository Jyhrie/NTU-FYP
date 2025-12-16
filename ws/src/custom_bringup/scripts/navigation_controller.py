#!/usr/bin/env python
import rospy
import numpy as np
import math
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

import tf.transformations
from vectors import Vector2
import utils

import local_occupancy_movement as lom

MAX_MOVEMENT_SPEED = 0.25
MAX_ANGULAR_SPEED = 0.35

HUG_DISTANCE = 0.2  # meters

class NavigationController:
    def __init__(self):
        rospy.init_node("navigation_controller")

        # subscribers
        rospy.Subscriber("/local_costmap", OccupancyGrid, self.local_costmap_cb)
        rospy.Subscriber("/odom", Odometry, self.odom_cb)

        # publishers
        self.debug_pub = rospy.Publisher("/debug_map", OccupancyGrid, queue_size=1)
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        self.local_occupancy_movement = lom.LocalOccupancyNavigator()

        self.have_map = False
        self.have_odom = False

        self.row, self.pitch, self.yaw = 0,0,0
        pass

    def local_costmap_cb(self, msg):
        self.local_map_msg = msg
        self.have_map = True

    def odom_cb(self, msg):
        self.odom = msg

        if msg is not None:
            self.have_odom = True
            q = msg.pose.pose.orientation
            quaternion = [q.x, q.y, q.z, q.w]
            self.roll, self.pitch, self.yaw = tf.transformations.euler_from_quaternion(quaternion)


    def display_debug_map(self, msg):
        #msg, normal_vec, inlier = self.local_occupancy_movement.trigger(self.local_map_msg)
        if msg is not None:
            self.debug_pub.publish(msg)
        return
    
    def get_local_route(self, samples=5):
        """
        gets the local route from the local occupancy movement module
        """

        rate = rospy.Rate(3)  # 5 Hz

        average_inlier_vec = []
        inlier_list = []

        for i in range(0,samples):
            msg, avg_inlier, inlier = self.local_occupancy_movement.trigger(self.local_map_msg)
            average_inlier_vec.append(avg_inlier)
            inlier_list.append(inlier)
            rate.sleep()

        # Take up to 5 samples (or fewer if not enough)

        inlier_point_list_x = []
        inlier_point_list_y = []

        print(inlier_list)
        for sample in inlier_list:
            #get first n
            inlier_len = len(sample)
            get_count = min(3, inlier_len)
            for i in range(0,get_count):
                inlier_point_list_x.append(sample[i].x)
                inlier_point_list_y.append(sample[i].y)
                
        inlier_point_list_x.sort()        
        inlier_point_list_y.sort()

        if inlier_point_list_x is not None:
            median_inlier = Vector2(
                inlier_point_list_x[len(inlier_point_list_x)//2],
                inlier_point_list_y[len(inlier_point_list_y)//2]
            )

        #compute average normal vector
        normal_vec_sum = Vector2(0,0)
        for vec in average_inlier_vec:
            normal_vec_sum.add(vec)
            average_normal_vec_median = normal_vec_sum.normalize()

        normal_vec_median = average_normal_vec_median

        res = self.local_map_msg.info.resolution

        #compute target point to hug wall
        target_point = Vector2(median_inlier.x + (normal_vec_median.x * (HUG_DISTANCE) / res), #negative as we want the < direction
                                median_inlier.y + (normal_vec_median.y * HUG_DISTANCE / res)) #postitive as we want ^ direction
        
        #move to target point
        cx = self.local_occupancy_movement.map_width // 2
        cy = self.local_occupancy_movement.map_height // 2

        dx = target_point.x - cx
        dy = target_point.y - cy

        target_angle = math.atan2(dy, dx)

        print("Robot Position (grid coords):", cx, cy)
        print("Median Inlier: ", median_inlier)
        print("Target Point:", target_point)
        print("Target Angle (rad):", target_angle)
        print("Current Yaw (rad):", self.yaw)
        print("Normal Vector:", normal_vec_median)
        angle_diff = utils.normalize_angle(target_angle - self.yaw)
        print("Angle Diff (rad):", angle_diff)

        mx = int(round(target_point.x))
        my = int(round(target_point.y))

        width = msg.info.width
        height = msg.info.height

        grid = np.array(msg.data, dtype=np.int8).reshape((height, width))
        grid[my, mx] = 2  # set cell

        # flatten back to msg.data
        msg.data = grid.flatten().tolist()

        
        self.display_debug_map(msg)

        pass


    def update_global_costmap(self):
        """
        updates the global costmap from the /map topic
        """
        pass

    def check_against_global_map(self):
        """
        checks against the global costmap to see if robot is turning into a spot where it has been before,
        """
        pass

    def run(self):
        rate = rospy.Rate(30)  # 5 Hz
        while not rospy.is_shutdown():
            if self.have_map and self.have_odom:
                try:
                    user_input = raw_input("Press A to run local route: ").strip().lower()
                    if user_input == 'a':
                        rospy.loginfo("Running local route")
                        self.get_local_route(samples=5)
                except KeyboardInterrupt:
                    rospy.loginfo("Exiting...")
                    break
            rate.sleep()


if __name__ == "__main__":
    nav = NavigationController()
    nav.run()
