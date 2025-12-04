#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
import math

# --- Configuration ---
LINEAR_SPEED = 0.15      # m/s
ANGULAR_SPEED = 0.3      # rad/s
DESIRED_DISTANCE = 0.3   # meters from wall
RATE_HZ = 10

NODE_NAME = 'hug_wall_map_node'
CMD_TOPIC = '/cmd_vel'
SCAN_TOPIC = '/scan'
MAP_TOPIC = '/map'

# threshold to detect unexplored/unknown space in map (-1 = unknown)
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

        # Convert margin in meters to map cells
        resolution = self.map_data.info.resolution
        width = self.map_data.info.width
        height = self.map_data.info.height

        # Robot is assumed at the center of the map grid for simplicity
        cx = int(width / 2)
        cy = int(height / 2)

        margin_cells = int(MAP_EDGE_MARGIN / resolution)
        
        # Check a column of cells ahead
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

            # --- LIDAR distances ---
            scan_len = len(self.scan.ranges)
            right_index = int(scan_len * 0.25)       # 90 deg to right
            front_index = int(scan_len * 0.0)        # 0 deg = front

            right_dist = min(self.scan.ranges[right_index-2:right_index+2])
            front_dist = min(self.scan.ranges[front_index-2:front_index+2])

            # --- Wall-following logic ---
            map_edge_ahead = self.check_map_ahead()

            if front_dist < DESIRED_DISTANCE or map_edge_ahead:  
                # Obstacle ahead OR approaching unknown map edge
                self.twist.linear.x = 0.0
                self.twist.angular.z = ANGULAR_SPEED  # turn left
            else:
                # Move forward
                self.twist.linear.x = LINEAR_SPEED
                error = DESIRED_DISTANCE - right_dist
                self.twist.angular.z = error * 1.5

            self.cmd_pub.publish(self.twist)
            self.rate.sleep()


if __name__ == '__main__':
    try:
        hug_wall_map = HugWallMap()
        hug_wall_map.run()
    except rospy.ROSInterruptException:
        pass
