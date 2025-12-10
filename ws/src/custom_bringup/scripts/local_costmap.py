#!/usr/bin/env python3
import rospy
import numpy as np
import math

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose


class LocalCostmapPublisher:
    def __init__(self,
                 size_m=3.0,       # total local window size (3m x 3m)
                 resolution=0.03): # 3cm resolution
                 
        self.size_m = size_m
        self.resolution = resolution

        self.n = int(self.size_m / self.resolution)
        self.c = self.n // 2  # robot center index

        self.grid = np.zeros((self.n, self.n), dtype=np.int8)
        self.scan = None

        # ROS
        rospy.Subscriber("/scan", LaserScan, self.scan_cb)
        self.pub = rospy.Publisher("/local_costmap", OccupancyGrid, queue_size=1)

    def scan_cb(self, msg):
        self.scan = msg

    def build_costmap(self):
        """
        Convert LaserScan → local occupancy grid (robot-centered)
        """
        if self.scan is None:
            self.grid.fill(0)
            return

        self.grid.fill(0)
        angle = self.scan.angle_min

        for d in self.scan.ranges:
            if math.isinf(d) or math.isnan(d):
                angle += self.scan.angle_increment
                continue

            if d > self.size_m:
                angle += self.scan.angle_increment
                continue

            # polar → Cartesian
            x = d * math.cos(angle)
            y = d * math.sin(angle)

            # Cartesian → grid index
            ix = int(self.c + x / self.resolution)
            iy = int(self.c + y / self.resolution)

            if 0 <= ix < self.n and 0 <= iy < self.n:
                self.grid[ix, iy] = 100  # OCCUPIED

            angle += self.scan.angle_increment

    def publish_costmap(self):
        """
        Publish occupancy map message
        """
        msg = OccupancyGrid()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link"  # robot-centered map

        msg.info.resolution = self.resolution
        msg.info.width = self.n
        msg.info.height = self.n

        # map origin so robot is at center
        msg.info.origin = Pose()
        msg.info.origin.position.x = -self.size_m / 2.0
        msg.info.origin.position.y = -self.size_m / 2.0
        msg.info.origin.position.z = 0.0

        # flatten grid row-major
        msg.data = self.grid.flatten().tolist()

        self.pub.publish(msg)

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.build_costmap()
            self.publish_costmap()
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("local_costmap_publisher")
    node = LocalCostmapPublisher(size_m=3.0, resolution=0.03)
    node.run()