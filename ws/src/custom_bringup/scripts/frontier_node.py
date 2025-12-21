#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PointStamped
# Assuming your FrontierDetector is in frontier_finder.py
from frontier_finder import FrontierDetector 

class FrontierNode:
    def __init__(self):
        rospy.init_node('frontier_explorer_node')

        # 1. State Variables
        self.latest_map = None
        self.latest_costmap = None
        self.detector = None

        # 2. Subscribers
        # We listen to gmapping for the frontier search
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        # We listen to global_costmap for the safety/inflation check
        self.costmap_sub = rospy.Subscriber("/move_base/global_costmap/costmap", OccupancyGrid, self.costmap_callback)

        # 3. Publishers
        self.frontier_map_pub = rospy.Publisher('/detected_frontiers', OccupancyGrid, queue_size=1)

        # 4. Timer - Run detection at a controlled rate (e.g., every 2 seconds)
        # Doing this in a timer prevents the callback from hanging if gmapping is fast
        self.timer = rospy.Timer(rospy.Duration(2.0), self.process_frontiers)

    def map_callback(self, msg):
        self.latest_map = msg
        # Initialize detector once we have map metadata
        if self.detector is None:
            self.detector = FrontierDetector(
                map_width=msg.info.width,
                map_height=msg.info.height,
                resolution=msg.info.resolution,
                origin_x=msg.info.origin.position.x,
                origin_y=msg.info.origin.position.y
            )

    def costmap_callback(self, msg):
        self.latest_costmap = msg

    def process_frontiers(self, event):
        # Only run if we have both datasets and detector is ready
        if self.latest_map is None or self.latest_costmap is None or self.detector is None:
            rospy.logwarn_throttle(5, "Waiting for Map and Costmap to be available...")
            return

        # Ensure maps are the same size before processing
        if len(self.latest_map.data) != len(self.latest_costmap.data):
            rospy.logerr_throttle(5, "Map and Costmap dimensions do not match! Check your ROS config.")
            return

        rospy.loginfo("Detecting frontiers...")
        
        # 5. Use the detector with both map_data and costmap_data
        # Update your FrontierDetector.detect_frontiers to accept two arguments
        centroids, frontier_map_data = self.detector.detect_frontiers(
            self.latest_map.data, 
            self.latest_costmap.data
        )

        # 6. Publish the debug map (visualization)
        self.publish_frontier_map(self.latest_map, frontier_map_data)

        # 7. Log results
        rospy.loginfo(f"Found {len(centroids)} safe, valid frontier clusters.")
        for i, (wx, wy) in enumerate(centroids):
            rospy.logdebug(f"Frontier {i}: x={wx:.2f}, y={wy:.2f}")

    def publish_frontier_map(self, original_msg, debug_data):
        """Helper to package the 1D list back into a ROS OccupancyGrid"""
        f_map = OccupancyGrid()
        f_map.header.stamp = rospy.Time.now()
        f_map.header.frame_id = original_msg.header.frame_id
        f_map.info = original_msg.info 
        f_map.data = debug_data
        self.frontier_map_pub.publish(f_map)

if __name__ == '__main__':
    try:
        node = FrontierNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass