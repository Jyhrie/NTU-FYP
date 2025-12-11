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
        # Adjust your sensor offset here
        self.sensor_offset = Vector2(0, -7)

        self.rate = rospy.Rate(10)

    # ------------------------------------------------------------
    # CALLBACK: Save Incoming Map
    # ------------------------------------------------------------
    def map_callback(self, msg: OccupancyGrid):
        self.map_width  = msg.info.width
        self.map_height = msg.info.height
        self.resolution = msg.info.resolution
        self.map_origin = msg.info.origin

        data = np.array(msg.data, dtype=np.int8)
        self.map = data.reshape((self.map_height, self.map_width))

    # ------------------------------------------------------------
    # CORE: Vectorized Vertical Scan ("God Mode")
    # ------------------------------------------------------------
    def vert_boxcasts(self, grid, scan_dist=50):
        cx, cy = self.map_width // 2, self.map_height // 2
        
        # 1. Define the Scan Area (Robot to Max Distance)
        # UP is negative Y. 
        y_start = cy - scan_dist
        y_end   = cy 
        
        # Center the scan on the robot (offset by sensor.x if needed)
        # We add sensor_offset.x in case the sensor is mounted left/right
        start_x_center = cx + int(self.sensor_offset.x)
        half_w = 7
        
        x_start = start_x_center - half_w
        x_end   = start_x_center + half_w + 1

        # 2. Safety Bounds (Treat edges as walls later, but clamp for slicing)
        # Note: If y_start < 0, the robot is close to the top edge.
        # We record the offset to correct the index math later.
        y_start_clamped = max(0, y_start)
        y_end_clamped   = min(self.map_height, y_end)
        x_start_clamped = max(0, x_start)
        x_end_clamped   = min(self.map_width, x_end)

        # 3. Extract the Strip
        # This slice represents the entire path in front of the robot
        strip = grid[y_start_clamped:y_end_clamped, x_start_clamped:x_end_clamped]

        # 4. Vectorized Check
        # "Is there ANY obstacle in this row?" -> Returns 1D array of Booleans
        if strip.size == 0: return -1
        row_has_obstacle = np.any(strip >= 100, axis=1)

        # 5. Find First Hit (Bottom-Up)
        # The strip is ordered Top->Bottom. We want the hit closest to the robot.
        rows_reversed = row_has_obstacle[::-1]
        
        hit_dist = -1
        
        if np.any(rows_reversed):
            # argmax finds the index of the FIRST True value
            hit_index_local = np.argmax(rows_reversed)
            
            # Distance from the 'y_end' (Robot)
            hit_dist = hit_index_local
            
            # Calculate exact hit position for drawing
            # Position = RobotOrigin + SensorOffset + ForwardStep
            # ForwardStep is (0, -hit_dist)
            hit_pos = Vector2(cx, cy).add(self.sensor_offset).add(Vector2(0, -hit_dist))
            
            # Draw the Vertical Hit (Red = 3)
            self.draw_boxcast_hit(hit_pos, 5, 7, Vector2(0,0), grid, 3)

            # --- TRIGGER HORIZONTAL SCAN ---
            # We only run this ONCE, right where the wall is.
            self.horizontal_boxcast(hit_pos, grid, scan_dist)
            
            return hit_dist
        else:
            # No Hit: Draw the "Clear" marker at the max distance
            end_pos = Vector2(cx, cy).add(self.sensor_offset).add(Vector2(0, -scan_dist))
            self.draw_boxcast_hit(end_pos, 5, 7, Vector2(0,0), grid, 3)
            return -1

    # ------------------------------------------------------------
    # HORIZONTAL SCAN (Runs once per frame on impact)
    # ------------------------------------------------------------
    def horizontal_boxcast(self, start_pos, grid, scan_dist=50):
        # Scan Right (+X)
        for i in range(scan_dist):
            # Rotated offset (from your original logic)
            rot_offset = Vector2(-self.sensor_offset.y, self.sensor_offset.x)
            
            step_offset = Vector2(i, 0)
            check_pos = start_pos.add(step_offset)

            hit = self.boxcast_area(check_pos, 7, 5, rot_offset, grid)
            
            if hit:
                # Draw Hit (Value 99)
                self.draw_boxcast_hit(check_pos, 7, 5, rot_offset, grid, 99)
                return i
        
        # If no horizontal hit, draw end of scan
        rot_offset = Vector2(-self.sensor_offset.y, self.sensor_offset.x)
        self.draw_boxcast_hit(start_pos.add(Vector2(scan_dist, 0)), 7, 5, rot_offset, grid, 3)
        return -1

    # ------------------------------------------------------------
    # HELPER: Area Check (Used by Horizontal Scan)
    # ------------------------------------------------------------
    def boxcast_area(self, center_pos, half_w, half_h, offset, grid):
        target = center_pos.add(offset)
        tx, ty = int(target.x), int(target.y)
        
        map_h, map_w = grid.shape
        
        # 1. Bounds Check (Touching Map Edge = Hit)
        if tx - half_w < 0 or tx + half_w >= map_w: return True
        if ty - half_h < 0 or ty + half_h >= map_h: return True

        # 2. Clamp for Slicing
        x0 = max(0, tx - half_w)
        x1 = min(map_w, tx + half_w + 1)
        y0 = max(0, ty - half_h)
        y1 = min(map_h, ty + half_h + 1)

        if x1 <= x0 or y1 <= y0: return False
        
        # 3. Check for 100
        return np.max(grid[y0:y1, x0:x1]) >= 100

    # ------------------------------------------------------------
    # HELPER: Draw Hit (Clamped for Visibility)
    # ------------------------------------------------------------
    def draw_boxcast_hit(self, center_pos, half_w, half_h, offset, grid, val):
        map_h, map_w = grid.shape
        target = center_pos.add(offset)

        # CLAMP CENTER to be visible inside map
        draw_x = max(half_w, min(map_w - 1 - half_w, int(target.x)))
        draw_y = max(half_h, min(map_h - 1 - half_h, int(target.y)))

        x0 = max(0, draw_x - half_w)
        x1 = min(map_w, draw_x + half_w + 1)
        y0 = max(0, draw_y - half_h)
        y1 = min(map_h, draw_y + half_h + 1)

        if x1 > x0 and y1 > y0:
            roi = grid[y0:y1, x0:x1]
            roi[roi != 100] = val

    # ------------------------------------------------------------
    # EXECUTION
    # ------------------------------------------------------------
    def draw_robot_footprint(self, grid):
        cx, cy = self.map_width // 2, self.map_height // 2
        # Simple footprint
        x0, x1 = max(0, cx-7), min(self.map_width, cx+8)
        y0, y1 = max(0, cy-12), min(self.map_height, cy+6)
        grid[y0:y1, x0:x1] = 1

    def raycast(self):
        if self.map is None: return
        # Copy to avoid corrupting the master map buffer
        self.grid = self.map.copy()
        self.vert_boxcasts(self.grid)

    def publish_debug_map(self):
        if self.map is None: return
        self.raycast()
        self.draw_robot_footprint(self.grid)

        msg = OccupancyGrid()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.info.resolution = self.resolution
        msg.info.width = self.map_width
        msg.info.height = self.map_height
        msg.info.origin = self.map_origin
        msg.data = self.grid.ravel().tolist()
        self.debug_pub.publish(msg)

    def run(self):
        while not rospy.is_shutdown():
            self.publish_debug_map()
            self.rate.sleep()

if __name__ == "__main__":
    LocalOccupancyNavigator().run()