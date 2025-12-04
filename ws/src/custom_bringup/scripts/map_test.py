#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
import math

# --- Configuration ---
LINEAR_SPEED = 0.15      # m/s
ANGULAR_SPEED = 0.15      # rad/s
DESIRED_DISTANCE = 0.3   # meters from wall
RATE_HZ = 10

NODE_NAME = 'hug_wall_map_node'
CMD_TOPIC = '/cmd_vel'
SCAN_TOPIC = '/scan'
MAP_TOPIC = '/map'

# Threshold to detect unknown space in map (-1 = unknown)
UNKNOWN_THRESHOLD = -1
MAP_EDGE_MARGIN = 0.2  # meters

class HugWallMap:
    def __init__(self):
        rospy.init_node(NODE_NAME, anonymous=False)
        rospy.loginfo("--- Wall Hug with Map Awareness ---")
        
        self.cmd_pub = rospy.Publisher(CMD_TOPIC, Twist, queue_size=1)
        rospy.Subscriber(SCAN_TOPIC, LaserScan, self.scan_callback)
        rospy.Subscriber(MAP_TOPIC, OccupancyGrid, self.map_callback)
        
        self.twist = Twist()
        self.scan = None
        self.map_data = None
        self.rate = rospy.Rate(RATE_HZ)

    def scan_callback(self, msg):
        self.scan = msg

    def map_callback(self, msg):
        self.map_data = msg

    def check_map_ahead(self):
        """
        Check if there is unknown or obstacle space ahead in the map
        within a small margin in front of the robot.
        """
        if self.map_data is None:
            return False  # treat as safe if map not ready

        resolution = self.map_data.info.resolution
        width = self.map_data.info.width
        height = self.map_data.info.height

        # Assume robot is at center of map grid
        cx = int(width / 2)
        cy = int(height / 2)

        margin_cells = int(MAP_EDGE_MARGIN / resolution)
        
        # Check a small rectangle of cells ahead
        for dx in range(0, margin_cells):
            cell_index = (cy * width) + (cx + dx)
            if 0 <= cell_index < len(self.map_data.data):
                if self.map_data.data[cell_index] == UNKNOWN_THRESHOLD:
                    return True
        return False

    def run(self):
        while not rospy.is_shutdown():
            if self.scan is None:
                self.rate.sleep()
                continue

            # --- Filter LIDAR ranges ---
            valid_ranges = [r for r in self.scan.ranges if not (math.isinf(r) or math.isnan(r))]
            scan_len = len(valid_ranges)
            if scan_len == 0:
                rospy.logwarn("No valid LIDAR data")
                self.rate.sleep()
                continue

            # --- Indices for front and right ---
            right_index = int(scan_len * 0.25)       # ~90 deg right
            front_index = int(scan_len * 0.0)        # 0 deg front

            # Safe slicing to avoid index errors
            right_slice = valid_ranges[max(0,right_index-2):min(scan_len,right_index+3)]
            front_slice = valid_ranges[max(0,front_index-2):min(scan_len,front_index+3)]

            right_dist = min(right_slice) if right_slice else float('inf')
            front_dist = min(front_slice) if front_slice else float('inf')

            # --- Wall-following logic ---
            #map_edge_ahead = self.check_map_ahead()
            map_edge_ahead = False

            if front_dist < DESIRED_DISTANCE or map_edge_ahead:
                # Obstacle ahead OR approaching unknown map edge
                self.twist.linear.x = 0.0
                self.twist.angular.z = ANGULAR_SPEED  # turn left
            else:
                # Move forward
                self.twist.linear.x = LINEAR_SPEED
                error = DESIRED_DISTANCE - right_dist
                self.twist.angular.z = error * 1.5  # P-controller

            self.cmd_pub.publish(self.twist)
            self.rate.sleep()


if __name__ == '__main__':
    try:
        hug_wall_map = HugWallMap()
        hug_wall_map.run()
    except rospy.ROSInterruptException:
        pass
