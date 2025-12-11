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

        msg = OccupancyGrid()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"

        msg.info.resolution = self.resolution
        msg.info.width = self.map_width
        msg.info.height = self.map_height
        msg.info.origin = self.map_origin

        msg.data = grid.flatten().tolist()

        self.debug_pub.publish(msg)

    def run(self):
        while not rospy.is_shutdown():
            self.publish_debug_map()
            self.rate.sleep()


if __name__ == "__main__":
    nav = LocalOccupancyNavigator()
    nav.run()
