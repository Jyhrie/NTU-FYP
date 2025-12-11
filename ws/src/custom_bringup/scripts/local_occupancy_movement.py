#!/usr/bin/env python3
import rospy
import numpy as np
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

        self.rate = rospy.Rate(10)

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

    def raycast(self):
        self.grid = self.map
        if self.grid is None:
            return
        self.vert_boxcasts(self.grid)

    def vert_boxcasts(self, grid, scan_dist = 50):
        robot_origin = Vector2(self.map_width // 2, self.map_height // 2)

        for i in range(scan_dist):
            step_offset = Vector2(0,-i)
            hit = self.boxcast_area(robot_origin.add(step_offset), 5, 7, self.sensor_offset, grid)
            self.horizontal_boxcast(robot_origin.add(step_offset), grid, scan_dist)
            if hit:
                self.draw_boxcast_hit(robot_origin.add(step_offset), 5, 7, self.sensor_offset, grid, 3)
                return i
        self.draw_boxcast_hit(robot_origin.add(step_offset), 5, 7, self.sensor_offset, grid, 3)
        
            
    def horizontal_boxcast(self, root, grid, scan_dist = 50):
        for i in range(scan_dist):
            step_offset = Vector2(i, 0)
            hit = self.boxcast_area(root.add(step_offset), 7, 5, Vector2(-self.sensor_offset.y, self.sensor_offset.x), grid)
            #self.draw_boxcast_hit(root.add(Vector2(i,0)), 7, 5, Vector2(self.sensor_offset.y, self.sensor_offset.x), grid, 2)
            if hit:
                self.draw_boxcast_hit(root.add(step_offset), 7, 5, Vector2(-self.sensor_offset.y, self.sensor_offset.x), grid, 99)
                return i
        self.draw_boxcast_hit(root.add(step_offset), 7, 5, Vector2(self.sensor_offset.y, self.sensor_offset.x), grid, 3)
        
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


    # def draw_horizontal_boxcasts(self, start_x, start_y, grid):
    #     # horizontal rectangle bounds (same size as vertical BoxCast, but along x)
    #     x_start = max(start_x - 7, 0)
    #     x_end   = min(start_x + 7 + 1, self.map_width)
    #     y_start = max(start_y - 12, 0)
    #     y_end   = min(start_y + 5 + 1, self.map_height)

    #     grid[y_start:y_end, x_start:x_end] = 99


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

        self.raycast()
        self.draw_robot_footprint(self.grid)

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
