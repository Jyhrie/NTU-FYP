#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
import numpy as np
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

    def get_right_wall(self, spread_samples=10):
        if self.scan is None:
            return None

        scan = self.scan 
        
        #this is left
        right_angle = math.pi / 2  # -90° in radians
        index = int(round((right_angle - scan.angle_min) / scan.angle_increment))

        i_start = index - spread_samples
        i_end   = index + spread_samples

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

    def parallel_alignment(self, dir=1, max_bidirectional_samples=50, max_falloff=0.2):
        if self.scan is None:
            return None

        scan = self.scan 
        right_angle = math.pi / 2  # -90° in radians
        index = int(round((right_angle - scan.angle_min) / scan.angle_increment))

        i_start = index - max_bidirectional_samples
        i_end   = index + max_bidirectional_samples

        i_start = max(0, i_start)
        i_end   = min(len(scan.ranges), i_end)

        wall_ranges = scan.ranges[i_start:i_end]
        num_beams = len(wall_ranges)
        half = num_beams // 2
        front_ranges = wall_ranges[half:]  # first half → front side of wall
        back_ranges  = wall_ranges[:half][::-1]  # second half → back side of wall

        prev_front_dist = front_ranges[0]
        prev_back_dist = back_ranges[0]
        falloff_count_front = 0
        falloff_count_back = 0
        front_edge_index = len(front_ranges)
        back_edge_index = len(back_ranges)
        patience = 5
        for i in range(0,max_bidirectional_samples):
            front_dist = front_ranges[i]  # LiDAR beam at front-side
            back_dist  = back_ranges[i]   # LiDAR beam at back-side

            if prev_front_dist is not None and falloff_count_front < patience:
                if abs(front_dist - prev_front_dist) > max_falloff:
                    falloff_count_front += 1
                else:
                    prev_front_dist = front_dist
                    falloff_count_front = 0  # reset if distance normalizes
                if falloff_count_front >= patience:
                    front_edge_index = i
                    print("Front edge detected at index", i)
                    print("Front Edges: ", front_ranges[i-5], front_ranges[i-4],front_ranges[i-3], front_ranges[i-2], front_ranges[i-1], front_ranges[i])


            if prev_back_dist is not None  and falloff_count_back < patience:
                if abs(back_dist - prev_back_dist) > max_falloff:
                    falloff_count_back += 1
                else:
                    prev_back_dist = back_dist
                    falloff_count_back = 0
                if falloff_count_back >= patience:
                    back_edge_index = i
                    print("Back edge detected at index", i)


        normalized_distance_index = front_edge_index if front_edge_index < back_edge_index else back_edge_index
        sample_index = max(0, normalized_distance_index - 5)
        print("sampling_index", sample_index)
        print("Front Distance at sample index:", front_ranges[sample_index])
        print("Back Distance at sample index:", back_ranges[sample_index])

        d_front = front_ranges[sample_index]  # distance to wall at front
        d_back  = back_ranges[sample_index]   # distance to wall at back

        angle_front = scan.angle_min + (i_start + half + sample_index) * scan.angle_increment
        angle_back  = scan.angle_min + (i_start + sample_index) * scan.angle_increment

        x_front = d_front * math.cos(angle_front)
        y_front = d_front * math.sin(angle_front)

        x_back  = d_back  * math.cos(angle_back)
        y_back  = d_back  * math.sin(angle_back)

        dx = x_front - x_back
        dy = y_front - y_back
        wall_angle = math.atan2(dy, dx)  # radians

        wall_angle_deg = math.degrees(wall_angle)
        print(f"Wall angle: {wall_angle_deg:.2f}°")


        # front_valid = [r for r in front_ranges if r is not None and np.isfinite(r)]
        # back_valid  = [r for r in back_ranges  if r is not None and np.isfinite(r)]

        

        # median_front = np.median(front_valid) if front_valid else float('inf')
        # median_back  = np.median(back_valid)  if back_valid else float('inf')

        # print("Median Front Distance:", median_front)
        # print("Median Back Distance:", median_back)

        

        #wall_ranges = [r for r in wall_ranges if scan.range_min < r < scan.range_max]

        pass

    def run(self):
        rate = rospy.Rate(1)  # 10 Hz
        while not rospy.is_shutdown():
            # if self.scan is None or self.map_data is None:
            #     rospy.loginfo("Waiting for SCAN and MAP data...")
            #     continue
            # pass
            right_wall_dist = self.get_right_wall()
            self.parallel_alignment()
            # print(f"Right wall distance: {right_wall_dist}")
            # if right_wall_dist is None or right_wall_dist > 1:
            #     self.publish_move_command(0,0)  # turn right
            # else:
            #     self.publish_move_command(LINEAR_SPEED, 0)  # move forward
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
