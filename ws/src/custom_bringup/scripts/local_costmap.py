#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
import math


class LocalCostmap:
    def __init__(self,
                 size_m=3.0,          # size of local map in meters
                 resolution=0.02):    # 2cm per cell (adjust as needed)

        self.size_m = size_m
        self.resolution = resolution

        # grid size in cells
        self.n = int(self.size_m / self.resolution)

        # center index (robot is at center of grid)
        self.c = self.n // 2    

        # allocate empty occupancy grid
        self.grid = np.zeros((self.n, self.n), dtype=np.int8)

        self.scan = None
        rospy.Subscriber("/scan", LaserScan, self.scan_cb)


    def scan_cb(self, msg: LaserScan):
        self.scan = msg


    def build(self):
        """
        Convert the latest lidar scan into a robot-centered occupancy grid.
        """

        # Clear previous grid
        self.grid.fill(0)

        if self.scan is None:
            return self.grid

        msg = self.scan
        angle = msg.angle_min

        for d in msg.ranges:
            # ignore invalid values
            if math.isinf(d) or math.isnan(d):
                angle += msg.angle_increment
                continue

            # ignore points outside local map window
            if d > self.size_m:
                angle += msg.angle_increment
                continue

            # convert polar → robot-centric (meters)
            x = d * math.cos(angle)
            y = d * math.sin(angle)

            # convert meters → grid coordinates
            ix = int(self.c + (x / self.resolution))
            iy = int(self.c + (y / self.resolution))

            # skip if out of bounds
            if 0 <= ix < self.n and 0 <= iy < self.n:
                self.grid[ix, iy] = 100  # occupied

            angle += msg.angle_increment

        return self.grid
    
if __name__ == "__main__":
    rospy.init_node("local_costmap_test")

    costmap = LocalCostmap(size_m=3.0, resolution=0.03)

    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        grid = costmap.build()

        # Example: check front occupancy
        front_slice = grid[costmap.c-10:costmap.c+10, costmap.c+40:costmap.c+70]
        front_blocked = np.any(front_slice == 100)

        print("Front blocked:", front_blocked)

        rate.sleep()