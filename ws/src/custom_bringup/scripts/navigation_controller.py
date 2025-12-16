#!/usr/bin/env python3
import rospy
import numpy as np
import math
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from vectors import Vector2

import local_occupancy_movement as lom

import sys, select, termios, tty

MAX_MOVEMENT_SPEED = 0.25
MAX_ANGULAR_SPEED = 0.35

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
        pass

    def local_costmap_cb(self, msg: OccupancyGrid):
        self.local_map_msg = msg
        self.have_map = True

    def odom_cb(self, msg: Odometry):
        self.odom = msg
        self.have_odom = True

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

        average_normal_vec = Vector2(0,0)
        for i in range(0,samples):
            msg, normal_vec, inlier = self.local_occupancy_movement.trigger(self.local_map_msg)
            self.display_debug_map(msg)
            rate.sleep()

        #if turned hug dist is > thresh, get to hug dist first.
        #to get to hug dist, get first point of detected spot, and move to projected distance perp to wall
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
                    user_input = input("Press A to run local route: ").strip().lower()
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
