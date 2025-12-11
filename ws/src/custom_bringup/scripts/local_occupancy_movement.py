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

        self.sensor_offset = Vector2(0,7)

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

        self.grid == None

        #sort coordinates
        start_x = int(min(pt1.x, pt2.x))
        end_x   = int(max(pt1.x, pt2.x))
        start_y = int(min(pt1.y, pt2.y))
        end_y   = int(max(pt1.y, pt2.y))

        #clamp vals
        x0 = max(0, start_x)
        x1 = min(map_w, end_x + 1)
        
        y0 = max(0, start_y)
        y1 = min(map_h, end_y + 1)

        # 5. Safety Check: If the box is off-screen, x1 might be <= x0
        if x1 <= x0 or y1 <= y0:
            return False

        # 6. The actual check
        # Returns True if ANY pixel in this box is >= 100
        return np.max(grid[y0:y1, x0:x1]) >= 100

    def raycast(self):
        grid = self.map
        self.vert_boxcasts(grid)

    def vert_boxcasts(self, grid, scan_dist = 20):
        robot_origin = Vector2(self.map_width // 2, self.map_height // 2)

        for i in range(scan_dist):
            step_offset = Vector2(0,-i)
            hit = self.boxcast_area(robot_origin.add(step_offset), 7, 5, self.sensor_offset, grid)
            self.horizontal_boxcast(robot_origin.add(step_offset), grid, scan_dist)
            if hit:
                self.draw_boxcast_hit(robot_origin.add(step_offset), 7, 5, self.sensor_offset, grid)
                self.horizontal_boxcast(robot_origin.add(step_offset), grid, scan_dist)
                return i
            
    def horizontal_boxcast(self, root, grid, scan_dist = 20):
        for i in range(scan_dist):
            hit = self.boxcast_area(root.add(Vector2(i,0)), 5, 7, Vector2(self.sensor_offset.y, self.sensor_offset.x), grid)
            if hit:
                self.draw_boxcast_hit(root.add(Vector2(i,0)), 5, 7, Vector2(self.sensor_offset.y, self.sensor_offset.x), grid)
                return i
        pass
        
    def draw_boxcast_hit(self, center_pos, half_w, half_h, offset, grid):
            # 1. Define corners
            print("drawing boxcast hit at:", center_pos.x, center_pos.y)
            center_pos = center_pos.add(offset)
            pt1 = center_pos.add(Vector2(half_w, half_h))
            pt2 = center_pos.subtract(Vector2(half_w, half_h))

            map_h, map_w = grid.shape

            # 2. Sort & Clamp (Standard boilerplate)
            start_x = max(0, int(min(pt1.x, pt2.x)))
            end_x   = min(map_w, int(max(pt1.x, pt2.x)) + 1)
            
            start_y = max(0, int(min(pt1.y, pt2.y)))
            end_y   = min(map_h, int(max(pt1.y, pt2.y)) + 1)

            # 3. Draw
            if start_x < end_x and start_y < end_y:
                self.grid[start_y:end_y, start_x:end_x] = 3


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

        grid = self.map.copy()


        self.raycast()

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
