#!/usr/bin/env python3
import rospy
import numpy as np
import math
from nav_msgs.msg import OccupancyGrid


class Vector2:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y

    def add(self, other):
        return Vector2(self.x + other.x, self.y + other.y)
    
    def subtract(self, other):
        return Vector2(self.x - other.x, self.y - other.y)

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
        rospy.init_node("local_occupancy_debug")

        rospy.Subscriber("/local_costmap", OccupancyGrid, self.map_callback)
        self.debug_pub = rospy.Publisher("/debug_map", OccupancyGrid, queue_size=10)

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
    def map_callback(self, msg: OccupancyGrid):
        self.map_width  = msg.info.width
        self.map_height = msg.info.height
        self.resolution = msg.info.resolution
        self.map_origin = msg.info.origin

        data = np.array(msg.data, dtype=np.int8)
        self.map = data.reshape((self.map_height, self.map_width))

    # ------------------------------------------------------------
    # DRAW a vertical line (modify the grid array directly)
    # ------------------------------------------------------------
    def draw_vertical_line(self, grid):
        robot_origin = Vector2(self.map_width // 2, self.map_height // 2)
        grid[:, robot_origin.x] = 3   # cost value 1

    def boxcast_area(self, root, pos_halfwidth, pos_halfheight, root_offset, grid):
            pos = root.add(root_offset)

            pt1 = pos.add(Vector2(pos_halfwidth, pos_halfheight))
            pt2 = pos.subtract(Vector2(pos_halfwidth, pos_halfheight))

            map_h, map_w = grid.shape

            # 1. Sort coordinates (Raw, unclamped)
            start_x = int(min(pt1.x, pt2.x))
            end_x   = int(max(pt1.x, pt2.x))
            start_y = int(min(pt1.y, pt2.y))
            end_y   = int(max(pt1.y, pt2.y))

            # 2. NEW: Check Bounds - Treat Map Edges as Walls
            # If any part of the box sticks out of the map, it's a hit.
            if start_x < 0 or end_x >= map_w:
                print("Boxcast hit map edge on X axis")
                return True
            if start_y < 0 or end_y >= map_h:
                print("Boxcast hit map edge on Y axis")
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
            return
        hitpoints, vert_endpoint = self.vert_boxcasts(grid)

        outliers, inliers = self.extract_outliers(hitpoints)

        hitpoints, vert_endpoint = self.vert_boxcasts(grid)

        for hitpoint in hitpoints:
            if hitpoint.x != -1 and hitpoint.y != -1:
                self.grid[hitpoint.y, hitpoint.x] = 2

        for outlier in outliers:
            self.grid[outlier.y, outlier.x] = 5

        
        for inlier in inliers:
            self.grid[inlier.y, inlier.x] = 6

        wall_vector = self.get_vector_from_points(inliers)
        for i in range(0,50):
            step = Vector2(wall_vector.x * i, wall_vector.y * i)
            self.grid[int(hitpoints[0].y + step.y), int(hitpoints[0].x + step.x)] = 2




        #target location will be the midpoint of the reference wall, multiplied by the normal vector, that the robot is facing
        #target_location = ""
        #wall is at vert_endpoint.

    def get_vector_from_points(self, points):
        if len(points) < 2:
            return Vector2(0,0)

        sx = 0
        sy = 0

        count = 0
        for i in range(len(points) - 1):
            dx = points[i+1].x - points[i].x
            dy = points[i+1].y - points[i].y
            sx += dx
            sy += dy
            count += 1

        return Vector2(sx / count, sy / count)

    def normal_vector(self, vector):
        return


        
    def extract_outliers(self, hitpoints):
            if not hitpoints: return [], []
            prev_x, prev_y = None, None
            prev_vec_x, prev_vec_y = None, None
            stop_point = len(hitpoints)
            for i in range(0, len(hitpoints)):
                hp = hitpoints[i]
                x, y = hp.x, hp.y
                if prev_x is None and prev_y is None:
                    prev_x, prev_y = x, y
                    continue
        
                vec_x = x - prev_x
                vec_y = y - prev_y

                length = math.sqrt(vec_x**2 + vec_y**2)

                if length > 0:
                    norm_x = vec_x / length
                    norm_y = vec_y / length
                else:
                    # Points are identical; direction is undefined (zero vector)
                    norm_x, norm_y = 0.0, 0.0

                if prev_vec_x is not None and prev_vec_y is not None:
                    print(prev_x, prev_y, "->", x, y)
                    #dot product to get angle difference
                    dot_val = (norm_x * prev_vec_x) + (norm_y * prev_vec_y)
                    print(dot_val)

                    if abs(dot_val) <= 0.706: #cos 45 degrees + leeway
                        stop_point = i
                        break

                prev_x, prev_y = x, y
                prev_vec_x, prev_vec_y = norm_x, norm_y


            outliers = hitpoints[stop_point:]
            inliers = hitpoints[:stop_point]

            return inliers, outliers
    

    def vert_boxcasts(self, grid, scan_dist = 50):
        robot_origin = Vector2(self.map_width // 2, self.map_height // 2)
        hitpoints = []
        for i in range(scan_dist):
            step_offset = Vector2(0,-i)
            hit = self.boxcast_area(robot_origin.add(step_offset), 5, 7, self.sensor_offset, grid)
            hit_horizontal = self.horizontal_boxcast(robot_origin.add(step_offset), grid, scan_dist)
            hitpoints.append(Vector2(hit_horizontal + robot_origin.x, robot_origin.y - i))
            if hit:
                self.draw_boxcast_hit(robot_origin.add(step_offset), 5, 7, self.sensor_offset, grid, 3)
                return hitpoints, i
        self.draw_boxcast_hit(robot_origin.add(step_offset), 5, 7, self.sensor_offset, grid, 3)
        return hitpoints, -1
        
            
    def horizontal_boxcast(self, root, grid, scan_dist = 50):
        for i in range(scan_dist):
            step_offset = Vector2(i, 0)
            hit = self.boxcast_area(root.add(step_offset), 7, 5, Vector2(-self.sensor_offset.y//2, self.sensor_offset.x), grid)
            #self.draw_boxcast_hit(root.add(Vector2(i,0)), 7, 5, Vector2(self.sensor_offset.y, self.sensor_offset.x), grid, 2)
            if hit:
                self.draw_boxcast_hit(root.add(step_offset), 7, 5, Vector2(-self.sensor_offset.y//2, self.sensor_offset.x), grid, 99)
                return i
        self.draw_boxcast_hit(root.add(step_offset), 7, 5, Vector2(-self.sensor_offset.y//2, self.sensor_offset.x), grid, 3)
        return -1
        
    def draw_boxcast_hit(self, center_pos, half_w, half_h, offset, grid, print_number):
            map_h, map_w = grid.shape

            # 1. Calculate Target Position
            target_pos = center_pos.add(offset)
            
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
    def publish_debug_map(self):
        if self.map is None or self.map_origin is None:
            return
        
        self.grid = self.map

        self.draw_robot_footprint(self.grid)
        self.raycast(self.grid)


        if self.grid is not None:

            msg = OccupancyGrid()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "map"

            msg.info.resolution = self.resolution
            msg.info.width = self.map_width
            msg.info.height = self.map_height
            msg.info.origin = self.map_origin

            msg.data = msg.data = self.grid.astype(np.int8).ravel()

            self.debug_pub.publish(msg)

    def run(self):
        while not rospy.is_shutdown():
            self.publish_debug_map()
            self.rate.sleep()


if __name__ == "__main__":
    nav = LocalOccupancyNavigator()
    nav.run()
