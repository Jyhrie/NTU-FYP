#!/usr/bin/env python3
import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid


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
        cx = self.map_width // 2
        cy = self.map_height // 2

        grid[:, cx] = 3   # cost value 1

    def draw_vert_boxcasts(self, grid):
        cx = self.map_width // 2
        cy = self.map_height // 2

        # initial rectangle in front of robot
        x_start = max(cx - 7, 0)
        x_end   = min(cx + 7 + 1, self.map_width)
        y_start = max(cy - 12, 0)
        y_end   = min(cy + 5 + 1, self.map_height)

        # Step forward in y (decreasing y = forward)
        for dy in range(y_end, -1, -1):
            self.draw_horizontal_boxcasts(cx, dy, grid)
            # extract the current “slice” rectangle at this y position
            rect = grid[dy:y_end, x_start:x_end]

            # check if any obstacle is present (value >= 100)
            if np.any(rect >= 100):
                # first obstacle detected, mark this row with cost 2
                grid[dy:y_end, x_start:x_end] = 2
                break  # stop stepping forward

    def draw_horizontal_boxcasts(self, start_x, start_y, grid):
        # horizontal rectangle bounds (same size as vertical BoxCast, but along x)
        x_start = max(start_x - 7, 0)
        x_end   = min(start_x + 7 + 1, self.map_width)
        y_start = max(start_y - 12, 0)
        y_end   = min(start_y + 5 + 1, self.map_height)

        grid[y_start:y_end, x_start:x_end] = 99


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

        self.draw_vertical_line(grid)
        self.draw_robot_footprint(grid)
        self.draw_vert_boxcasts(grid)

        msg = OccupancyGrid()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"

        msg.info.resolution = self.resolution
        msg.info.width = self.map_width
        msg.info.height = self.map_height
        msg.info.origin = self.map_origin

        msg.data = msg.data = grid.astype(np.int8).ravel()

        self.debug_pub.publish(msg)

    def run(self):
        while not rospy.is_shutdown():
            self.publish_debug_map()
            self.rate.sleep()


if __name__ == "__main__":
    nav = LocalOccupancyNavigator()
    nav.run()
