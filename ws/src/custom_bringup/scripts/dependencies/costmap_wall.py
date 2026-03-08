import numpy as np
import cv2

def calc_cost_map(mapdata, obj_coords=None):
    width = mapdata.info.width
    height = mapdata.info.height
    
    # 1. Convert OccupancyGrid to numpy array (Height, Width)
    map_arr = np.array(mapdata.data).reshape(height, width).astype(np.uint8)
    
    # 2. Define Obstacles: Treat unknown (255) and occupied (100) as walls
    # We create a binary mask where 1 is an obstacle and 0 is free space
    is_obstacle = (map_arr == 100) | (map_arr == 255)
    
    # 3. Inject Object Obstacle (3x3 block)
    if obj_coords is not None:
        ocx, ocy = obj_coords
        r = 1 
        y_start, y_end = max(0, ocy-r), min(height, ocy+r+1)
        x_start, x_end = max(0, ocx-r), min(width, ocx+r+1)
        is_obstacle[y_start:y_end, x_start:x_end] = True

    # 4. Distance Transform: Calculate distance from every free cell to the nearest obstacle
    # We use CV2's built-in distanceTransform for extreme speed on the Jetson
    # DIST_L2 = Euclidean distance; Mask size 5 is high precision
    dist_to_wall = cv2.distanceTransform((~is_obstacle).astype(np.uint8), cv2.DIST_L2, 5)

    # 5. Convert Distance to Cost (0-100)
    # Walls should be 100, Centers should be 0.
    # We define a 'max_influence' (e.g., 20 cells) beyond which the cost stays 0
    max_influence = 20.0 
    
    # Formula: cost = max(0, 100 * (1 - (dist / max_influence)))
    cost_map = 100.0 * (1.0 - (dist_to_wall / max_influence))
    cost_map = np.clip(cost_map, 0, 100).astype(np.uint32)

    # Ensure actual obstacles remain at max cost
    cost_map[is_obstacle] = 100

    return cost_map