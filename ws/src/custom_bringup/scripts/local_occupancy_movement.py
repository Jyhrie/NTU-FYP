#!/usr/bin/env python3
import rospy
import numpy as np

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist


class LocalOccupancyNavigator:
    def __init__(self):
        rospy.init_node("local_occupancy_navigator")

        # --- Subscribers ---
        rospy.Subscriber("/localoccupancy", OccupancyGrid, self.map_callback)

        # --- Publisher ---
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        # --- State ---
        self.map = None
        self.map_width = 0
        self.map_height = 0
        self.resolution = 0.0

        # robot is ALWAYS center of map
        self.cx = 0
        self.cy = 0

        self.footprint_width = 0.4
        self.footprint_length = 0.4

        cell_width  = int(self.footprint_width / self.resolution)
        cell_length = int(self.footprint_length / self.resolution)

        self.rate = rospy.Rate(10)  # 10 Hz tick

    # ------------------------------------------------------------
    # MAP CALLBACK
    # ------------------------------------------------------------
    def map_callback(self, msg: OccupancyGrid):
        """Convert OccupancyGrid into numpy and store internally."""
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.resolution = msg.info.resolution

        # Convert to numpy grid, reshape
        data = np.array(msg.data, dtype=np.int8)
        grid = data.reshape((self.map_height, self.map_width))

        # -1 unknown → treat separately later
        self.map = grid

        # center of map = robot position
        self.cx = self.map_width // 2
        self.cy = self.map_height // 2

    def pick_navigation_to_position(self):
        """
        Pick a valid target position for the robot to navigate to.
        Returns (ix, iy) grid coordinates in self.grid.
        """
        # Scan from top-right to bottom-left
        for iy in range(self.n-1, -1, -1):       # top → bottom
            for ix in range(self.n-1, -1, -1):   # right → left
                if self.grid[iy, ix] == 0:       # free cell
                    return (ix, iy)
        
        # fallback if no free cell found
        return (self.c, self.c)  # stay in place



    # ------------------------------------------------------------
    # MAIN LOGIC TICK
    # ------------------------------------------------------------
    def tick(self):
        pass

    # def get_right_wall(self, max_scan_distance=40, threshold=50):
    #     strip_width = int(self.robot_width / self.resolution / 2)
    #     max_cells = int(max_scan_distance / self.resolution)

    #     # Slice map
    #     right_strip = self.map[self.cy - strip_width:self.cy + strip_width + 1,
    #                         self.cx + 1:self.cx + max_cells + 1]
        
    #     occupied_mask = right_strip >= threshold
    #     ys, xs = np.where(occupied_mask)

    #     xs_map = xs + self.cx + 1
    #     ys_map = ys + self.cy - strip_width
    #     points = list(zip(xs_map, ys_map))

    #     if not points:
    #         distance = None  # no wall detected

    #     else:
    #         dx_min = min([x - self.cx for x, y in points])
    #         distance = dx_min * self.resolution  # convert cells to meters


    #     return distance

    def wallhug(self):
        pass

    def run(self):
        while not rospy.is_shutdown():
            self.tick()
            self.rate.sleep()


if __name__ == "__main__":
    nav = LocalOccupancyNavigator()
    nav.run()
