#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
import math

LINEAR_SPEED = 0.15      # m/s
ANGULAR_SPEED = 0.05      # rad/s
DESIRED_DISTANCE = 0.5   # meters from wall
RATE_HZ = 10

NODE_NAME = 'mapper_node'
CMD_TOPIC = '/cmd_vel'
SCAN_TOPIC = '/scan'
MAP_TOPIC = '/map'

MIN_FORWARD_WALL_DISTANCE = 0.4  # meters

class Transform:
    def __init__(self, translation, rotation):
        self.translation = translation
        self.rotation = rotation
        self.sample_points = 0.7

class Vector3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    
class Quaternion:
    def __init__(self, x, y, z, w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

class Mapper:

    def __init__(self):
        rospy.init_node(NODE_NAME, anonymous=False)
        rospy.loginfo("--- Custom Mapping Algoritm ---")

        self.scan = None
        self.start_transform = None
        self.map_data = None
        self.forward_wall = None

        self.cmd_pub = rospy.Publisher(CMD_TOPIC, Twist, queue_size=1)
        rospy.Subscriber(SCAN_TOPIC, LaserScan, self.scan_callback)
        rospy.Subscriber(MAP_TOPIC, OccupancyGrid, self.map_callback)

    def set_start_transform(self):
        self.start_transform = Transform((0,0,0), Quaternion(0,0,0,1))
        pass

    def get_right_wall(self, spread_deg=90):
        if self.scan is None:
            return None

        scan = self.scan

        # Convert degrees to radians
        spread_rad = math.radians(spread_deg)

        # EXACT EAST = -90 degrees
        east_angle = -math.pi / 2

        # Compute start/end angles
        start_angle = east_angle - spread_rad
        end_angle   = east_angle + spread_rad

        # Helper to convert angle -> index
        def angle_to_index(angle):
            return int((angle - scan.angle_min) / scan.angle_increment)

        i_start = angle_to_index(start_angle)
        i_end   = angle_to_index(end_angle)

        # Clamp to valid array range
        i_start = max(0, i_start)
        i_end   = min(len(scan.ranges), i_end)

        # Extract the slice
        wall_ranges = scan.ranges[i_start:i_end]

        # Filter out inf and out-of-range
        wall_ranges = [r for r in wall_ranges if scan.range_min < r < scan.range_max]

        if not wall_ranges:
            return None  # no wall detected

        # Return median distance to smooth out noise
        wall_ranges_sorted = sorted(wall_ranges)
        median_idx = len(wall_ranges_sorted) // 2
        return wall_ranges_sorted[median_idx]

    def publish_move_command(self, linear, angular):
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        self.cmd_pub.publish(twist)

    def get_correction_distance(self):
        #get 90deg distance away from wall (this acts as the hug distance) if value is inf, make robot turn right until wall is detected, peform smooth turn.
        #take n points, close to 90deg right of robot, find the median distance away from the wall.
        pass

    def get_correction_angle(self):
        #get gradient of wall from x points (this acts as the correction angle)
        pass

    def check_forward(self):
        #get forward
        pass

    def check_left(self):
        #get left
        pass

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            # if self.scan is None or self.map_data is None:
            #     rospy.loginfo("Waiting for SCAN and MAP data...")
            #     continue
            # pass

            print(self.get_right_wall(spread_deg=90))
            rate.sleep()

        
        # correction_distance = self.get_correction_distance()
        # correction_angle = self.get_correction_angle()

        # self.forward_wall = self.check_forward()
        # if self.forward_wall < MIN_FORWARD_WALL_DISTANCE:
        #     #perform left wall check
        #     pass

        # if correction_distance == math.inf:
        #     pass
        # if correction_angle == math.inf:
        #     pass

        #get 90deg distance away from wall (this acts as the hug distance) if value is inf, make robot turn right until wall is detected, peform smooth turn.
        #take n points, close to 90deg right of robot, find the median distance away from the wall.

        #get gradient of wall from x points (this acts as the correction angle)

        #get forward distance to wall (this acts as the stop distance)
        #if forward has wall, right has wall, left has no wall, perform a left turn (corner detected)
        #if forward, right, and left has wall, perfrom a reverse (dead end detected)

        #if area is fully detected, (need to find metric for this, probably robot is in a dead end that it already has been at/near)
            #let ASTAR algorithm take over to find a random spot near unexplored area (flood fill area, find spot nearest to robot and move there, then A-star there, then resume wall hug)
            #if flood fill literally FILLS the entire map, then mapping is complete, stop robot.
            #if robot is back at the start location, stop robot.






    def scan_callback(self, msg):
        self.scan = msg

    def map_callback(self, msg):
        self.map_data = msg


def start_mapping():
    mapper = Mapper()
    mapper.set_start_transform()
    mapper.run()
    pass

def start_coroutine():
    pass


if __name__ == '__main__':
    try:
        start_mapping()
    except rospy.ROSInterruptException:
        pass
