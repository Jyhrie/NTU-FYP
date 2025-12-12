#!/usr/bin/env python3
import rospy
import numpy as np
import math
from nav_msgs.msg import OccupancyGrid
import local_occupancy_movement as lom

class NavigationController:
    def __init__(self):

        rospy.init_node("navigation_controller")
        rospy.Subscriber("/local_costmap", OccupancyGrid, self.local_occupancy_callback)
        self.debug_pub = rospy.Publisher("/debug_map", OccupancyGrid, queue_size=10)

        self.local_occupancy_movement = lom.LocalOccupancyNavigator()

        local_map_data = None
        local_map_width = None
        local_map_height = None
        local_resolution = None
        local_map_origin = None

        self.local_map = None
        self.map_data = None

    def run(self):
        rate = rospy.Rate(5)  # 5 Hz
        while not rospy.is_shutdown():
            if self.map_data is not None:
                self.map_data = self.local_occupancy_movement.trigger(self.map_data)
                self.publish_debug_map()
            rate.sleep()

    def publish_debug_map(self):
        print("publishing debug map")
        if self.local_map is None or self.local_map_origin is None:
            return

        # Use local map instead of self.map
        self.grid = self.local_map.copy()

        # Create OccupancyGrid message
        msg = OccupancyGrid()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"

        msg.info.resolution = self.local_resolution
        msg.info.width = self.local_map_width
        msg.info.height = self.local_map_height
        msg.info.origin = self.local_map_origin

        msg.data = self.grid.astype(np.int8).ravel()

        self.debug_pub.publish(msg)


    def local_occupancy_callback(self, msg):
        self.local_map_data = msg
        self.local_map_width  = msg.info.width
        self.local_map_height = msg.info.height
        self.local_resolution = msg.info.resolution
        self.local_map_origin = msg.info.origin

        data = np.array(msg.data, dtype=np.int8)
        self.local_map = data.reshape((self.local_map_height, self.local_map_width))


if __name__ == "__main__":
    nav = NavigationController()
    nav.run()