#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
import json
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist

hallway_threshold_div = 4  

def is_hallway_cell(cost_map, p, threshold, width, height):
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

def create_hallway_mask(cost_map, threshold, width, height):
    mask = np.zeros_like(cost_map, dtype=np.uint8)
    non_zero = np.transpose(np.nonzero(cost_map))
    for y, x in non_zero:
        if is_hallway_cell(cost_map, (x, y), threshold, width, height):
            mask[y, x] = 1
    return mask

def calc_cost_map(mapdata):
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
    hallway_mask = create_hallway_mask(cost_map, iterations // hallway_threshold_div, width, height)

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