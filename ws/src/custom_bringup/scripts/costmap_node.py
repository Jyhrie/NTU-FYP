#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
import json
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist

class CostmapNode:
    def __init__(self):
        rospy.init_node("costmap_node")

        # Configuration: lower value = stricter hallway centering
        self.hallway_threshold_div = 4  

        # Publisher for the processed hallway costmap
        self.costmap_pub = rospy.Publisher("/map/costmap_global", OccupancyGrid, queue_size=1, latch=True)

        rospy.loginfo("Waiting for the very first map message...")

        # Subscriber to the raw SLAM map
        rospy.Subscriber("/map", OccupancyGrid, self.map_callback)

        rospy.loginfo("Python 3 Costmap Node Ready. Processing /map -> /map/costmap_global")

    def map_callback(self, msg):
        # 1. Process the map using your iterative dilation logic
        cost_array = self.calc_cost_map(msg)

        # 2. Prepare the OccupancyGrid message
        out_msg = OccupancyGrid()
        out_msg.header = msg.header
        out_msg.header.stamp = rospy.Time.now()
        out_msg.info = msg.info

        # 3. Normalize to 0-100 range for Rviz visualization
        max_val = np.max(cost_array)
        if max_val > 0:
            # High cost (walls) = 100, Low cost (centers) = 0
            normalized = (cost_array.astype(float) / max_val * 100)
            # OccupancyGrid data must be a list of int8
            out_msg.data = normalized.flatten().astype(np.int8).tolist()
        else:
            out_msg.data = cost_array.flatten().astype(np.int8).tolist()

        self.costmap_pub.publish(out_msg)

    def is_hallway_cell(self, cost_map, p, threshold, width, height):
        """Checks if a cell is a local maximum (center of the path)"""
        val = cost_map[p[1], p[0]]
        # 8-neighbor check
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0: continue
                nx, ny = p[0] + dx, p[1] + dy
                if 0 <= nx < width and 0 <= ny < height:
                    n_val = cost_map[ny, nx]
                    if n_val < threshold or n_val > val:
                        return False
        return True

    def create_hallway_mask(self, cost_map, threshold, width, height):
        mask = np.zeros_like(cost_map, dtype=np.uint8)
        non_zero = np.transpose(np.nonzero(cost_map))
        for y, x in non_zero:
            if self.is_hallway_cell(cost_map, (x, y), threshold, width, height):
                mask[y, x] = 1
        return mask

    def calc_cost_map(self, mapdata):
        width = mapdata.info.width
        height = mapdata.info.height
        
        # Convert OccupancyGrid to numpy array
        # Note: height/width order is crucial for reshape
        map_arr = np.array(mapdata.data).reshape(height, width).astype(np.uint8)
        
        # Treat unknown (-1 or 255) as obstacles (100)
        map_arr[map_arr == 255] = 100 

        cost_map = np.zeros_like(map_arr, dtype=np.uint32)
        dilated_map = map_arr.copy()
        iterations = 0
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)

        # Pass 1: Distance from walls
        while np.any(dilated_map == 0):
            iterations += 1
            next_dilated_map = cv2.dilate(dilated_map, kernel, iterations=1)
            difference = next_dilated_map - dilated_map
            cost_map[difference > 0] = iterations
            dilated_map = next_dilated_map

        # Pass 2: Identify the "skeleton" (centerlines)
        hallway_mask = self.create_hallway_mask(cost_map, iterations // self.hallway_threshold_div, width, height)

        # Pass 3: Propagate cost outward from those centerlines
        dilated_map = hallway_mask.copy()
        final_cost_map = np.zeros_like(map_arr, dtype=np.uint32)
        cost = 1
        for _ in range(iterations):
            cost += 1
            next_dilated_map = cv2.dilate(dilated_map, kernel, iterations=1)
            difference = next_dilated_map - dilated_map
            final_cost_map[difference > 0] = cost
            dilated_map = next_dilated_map

        final_cost_map[final_cost_map > 0] -= 1
        return final_cost_map

    def nudge_robot(self):
        rospy.loginfo("Nudging robot to wake up SLAM...")
        pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        rospy.sleep(1) # Wait for publisher to connect
        
        move_msg = Twist()
        move_msg.linear.x = 0.1  # Move at 0.1 m/s
        
        # Publish for 0.5 seconds to move ~5cm
        start_time = rospy.Time.now()
        while (rospy.Time.now() - start_time) < rospy.Duration(0.5):
            pub.publish(move_msg)
            rospy.sleep(0.1)
            
        # Stop
        pub.publish(Twist())

if __name__ == "__main__":
    try:
        node = CostmapNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass