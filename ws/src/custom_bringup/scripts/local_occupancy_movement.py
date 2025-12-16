#!/usr/bin/env python3
import rospy
import numpy as np
import math
from nav_msgs.msg import OccupancyGrid
from vectors import Vector2

    
class Quaternion:
    def __init__(self, x, y, z, w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

class Transform: 
    def __init__(self, pos = Vector2(0,0), rot = Quaternion(0,0,0,1)):
        self.pos = pos
        self.rot = rot

class LocalOccupancyNavigator:
    def __init__(self):
        # rospy.init_node("local_occupancy_debug")
        # rospy.Subscriber("/local_costmap", OccupancyGrid, self.map_callback)
        # self.debug_pub = rospy.Publisher("/debug_map", OccupancyGrid, queue_size=10)

        self.map = None
        self.map_width = 0
        self.map_height = 0
        self.resolution = 0.0
        self.map_origin = None

        self.grid = None

        self.sensor_offset = Vector2(0,-7)
        self.rate = rospy.Rate(2)

    # ------------------------------------------------------------
    # SAVE incoming costmap
    # ------------------------------------------------------------
    # def map_callback(self, msg: OccupancyGrid):
    #     self.map_width  = msg.info.width
    #     self.map_height = msg.info.height
    #     self.resolution = msg.info.resolution
    #     self.map_origin = msg.info.origin

    #     data = np.array(msg.data, dtype=np.int8)
    #     self.map = data.reshape((self.map_height, self.map_width))

    # ------------------------------------------------------------
    # DRAW a vertical line (modify the grid array directly)
    # ------------------------------------------------------------
    def draw_vertical_line(self, grid):
        robot_origin = Vector2(self.map_width // 2, self.map_height // 2)
        grid[:, robot_origin.x] = 3   # cost value 1

    def boxcast_area(self, root, pos_halfwidth, pos_halfheight, root_offset, grid):  
            pos = root.copy()
            pos.add(root_offset)

            pt1 = pos.copy().add(Vector2(pos_halfwidth, pos_halfheight))
            pt2 = pos.copy().subtract(Vector2(pos_halfwidth, pos_halfheight))

            map_h, map_w = grid.shape

            # 1. Sort coordinates (Raw, unclamped)
            start_x = int(min(pt1.x, pt2.x))
            end_x   = int(max(pt1.x, pt2.x))
            start_y = int(min(pt1.y, pt2.y))
            end_y   = int(max(pt1.y, pt2.y))

            # 2. NEW: Check Bounds - Treat Map Edges as Walls
            # If any part of the box sticks out of the map, it's a hit.
            if start_x < 0 or end_x >= map_w:
                #print("Boxcast hit map edge on X axis")
                return True
            if start_y < 0 or end_y >= map_h:
                #print("Boxcast hit map edge on Y axis")
                return True 

            # 3. Clamp vals for NumPy slicing
            # (Technically redundant now if we return True above, but good for safety)
            x0 = max(0, start_x)
            x1 = min(map_w, end_x + 1)
            y0 = max(0, start_y)
            y1 = min(map_h, end_y + 1)

            # 4. Safety Check: If slice is invalid (shouldn't happen due to step 2, but keep it)
            if x1 <= x0 or y1 <= y0:
                return False

            # 5. The actual obstacle check
            return np.max(grid[y0:y1, x0:x1]) >= 100

    def raycast(self, grid):
        if grid is None:
            return None, None, None, None
        hitpoints, vert_endpoint = self.vert_boxcasts(grid)

        inliers, outliers, average_vector = self.extract_outliers(hitpoints)

        hitpoints, vert_endpoint = self.vert_boxcasts(grid)

        # for hitpoint in hitpoints:
        #     if hitpoint.x != -1 and hitpoint.y != -1:
        #         self.grid[hitpoint.y, hitpoint.x] = 2

        for outlier in outliers:
            self.grid[outlier.y, outlier.x] = 5

        
        for inlier in inliers:
            self.grid[inlier.y, inlier.x] = 6

        if not inliers:
            avg_inlier = Vector2(0, 0)

        else:
            # Compute sum of coordinates
            sum_x = sum(p.x for p in inliers)
            sum_y = sum(p.y for p in inliers)
            count = len(inliers)

            avg_inlier = Vector2(sum_x / count, sum_y / count)

        normal_vec = average_vector.normal()

        # for i in range (0,5):
        #     self.grid[int(avg_inlier.y + (normal_vec.y * i)), int(avg_inlier.x - (normal_vec.x * i))] = 2

        # self.grid[int(avg_inlier.y), int(avg_inlier.x)] = 1
        # origin = Vector2(self.map_width // 2, self.map_height // 2)
        # end_position =  avg_inlier # + Vector2(normal_vec.x * 3, normal_vec.y *3)
        # goal_forward_vector = average_vector
        # return origin, end_position, goal_forward_vector

        return normal_vec, inlier

        
    def extract_outliers(self, hitpoints, span=2):
        """
        Splits hitpoints into inliers and outliers using a spanning window method.
        Computes the average wall vector from valid points.
        """
        if not hitpoints or len(hitpoints) <= span:
            return [], hitpoints, Vector2(0, 0)

        average_vector = Vector2(0, 0)
        stop_point = len(hitpoints)
        num_vectors = 0

        for i in range(len(hitpoints) - span):
            p1 = hitpoints[i]
            p2 = hitpoints[i + span]

            dx = p2.x - p1.x
            dy = p2.y - p1.y
            length = math.hypot(dx, dy)

            if length == 0:
                continue  # identical points

            norm_x = dx / length
            norm_y = dy / length

            # Compute dot with previous average to detect sharp change
            if i > 0:
                dot_val = norm_x * average_vector.x + norm_y * average_vector.y
                if dot_val < 0.906:  # cos ~25 degrees
                    stop_point = i
                    break

            average_vector.add(Vector2(norm_x, norm_y))
            num_vectors += 1

        # Finalize average vector
        avg_vec = average_vector.normalize() if num_vectors > 0 else Vector2(0, 0)

        outliers = hitpoints[:stop_point]
        inliers = hitpoints[stop_point:]

        return inliers, outliers, avg_vec
    

    def vert_boxcasts(self, grid, scan_dist = 50):
        robot_origin = Vector2(self.map_width // 2, self.map_height // 2)
        hitpoints = []
        
        step_offset = Vector2(0,-1)
        for i in range(scan_dist):
            robot_origin.add(step_offset)
            hit = self.boxcast_area(robot_origin, 5, 7, self.sensor_offset, grid)

            hit_horizontal = self.horizontal_boxcast(robot_origin.copy(), grid, scan_dist)

            hitpoints.append(Vector2(hit_horizontal + robot_origin.x, robot_origin.y))
            if hit:
                self.draw_boxcast_hit(robot_origin, 5, 7, self.sensor_offset, grid, 3)
                return hitpoints, i
        self.draw_boxcast_hit(robot_origin, 5, 7, self.sensor_offset, grid, 3)
        return hitpoints, -1
        
            
    def horizontal_boxcast(self, root, grid, scan_dist = 50):
        step_offset = Vector2(1, 0)
        for i in range(scan_dist):
            root.add(step_offset)
            hit = self.boxcast_area(root, 7, 5, Vector2(-self.sensor_offset.y, self.sensor_offset.x), grid)
            #self.draw_boxcast_hit(root.add(Vector2(i,0)), 7, 5, Vector2(self.sensor_offset.y, self.sensor_offset.x), grid, 2)
            if hit:
                self.draw_boxcast_hit(root, 7, 5, Vector2(-self.sensor_offset.y, self.sensor_offset.x), grid, 99)
                return i
        self.draw_boxcast_hit(root, 7, 5, Vector2(-self.sensor_offset.y, self.sensor_offset.x), grid, 3)
        return -1
        
    def draw_boxcast_hit(self, center_pos, half_w, half_h, offset, grid, print_number):
            map_h, map_w = grid.shape

            # 1. Calculate Target Position
            target_pos = center_pos.copy().add(offset)
            
            # --- NEW: Clamp the Center to be visible ---
            # We ensure the center is at least 'half_w' away from edges.
            # This guarantees the full box is drawn at the very edge of the map
            # if the actual hit was further out.
            draw_x = max(half_w, min(map_w - 1 - half_w, int(target_pos.x)))
            draw_y = max(half_h, min(map_h - 1 - half_h, int(target_pos.y)))
            
            # 2. Define corners using the CLAMPED center
            # We use standard integers here since we clamped them manually above
            pt1_x = draw_x + half_w
            pt1_y = draw_y + half_h
            pt2_x = draw_x - half_w
            pt2_y = draw_y - half_h

            # 3. Sort & Clamp Indices (Standard boilerplate)
            # We still need this in case 'half_w' is larger than the map itself
            start_x = max(0, min(pt1_x, pt2_x))
            end_x   = min(map_w, max(pt1_x, pt2_x) + 1)
            
            start_y = max(0, min(pt1_y, pt2_y))
            end_y   = min(map_h, max(pt1_y, pt2_y) + 1)

            # 4. Draw
            if start_x < end_x and start_y < end_y:
                # Note: Changed 'self.grid' to 'grid' to avoid the NoneType error
                roi = grid[start_y:end_y, start_x:end_x]
                
                # Update only cells that are NOT 100
                roi[roi != 100] = print_number


    def draw_robot_footprint(self, grid):
        cx = self.map_width // 2
        cy = self.map_height // 2

        # Define footprint bounds
        x_start = max(cx - 7, 0)
        x_end   = min(cx + 7 + 1, self.map_width)   # +1 because slicing is exclusive
        y_start = max(cy - 12, 0)
        y_end   = min(cy + 5 + 1, self.map_height)

        # Set the rectangle area to occupancy value 1
        grid[y_start:y_end, x_start:x_end] = 1

    # ------------------------------------------------------------
    # CREATE debug map and publish it
    # ------------------------------------------------------------
    def trigger(self, msg):
        self.map_width  = msg.info.width
        self.map_height = msg.info.height
        self.resolution = msg.info.resolution
        self.map_origin = msg.info.origin

        data = np.array(msg.data, dtype=np.int8)
        self.map = data.reshape((self.map_height, self.map_width))

        self.grid = self.map

        self.draw_robot_footprint(self.grid)
        normal_vec, inlier =  self.raycast(self.grid)

        if self.grid is not None:

            msg = OccupancyGrid()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "map"

            msg.info.resolution = self.resolution
            msg.info.width = self.map_width
            msg.info.height = self.map_height
            msg.info.origin = self.map_origin

            msg.data = msg.data = self.grid.astype(np.int8).ravel()

        return msg, normal_vec, inlier

