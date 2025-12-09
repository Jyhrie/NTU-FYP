#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
import numpy as np
import math
from enum import Enum, auto

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

LINEAR_SPEED = 0.08     # m/s
ANGULAR_SPEED = 0.05      # rad/s
DESIRED_DISTANCE = 0.5   # meters from wall
RATE_HZ = 10

NODE_NAME = 'mapper_node'
CMD_TOPIC = '/cmd_vel'
SCAN_TOPIC = '/scan'
MAP_TOPIC = '/map'

INIT_HUG_DIST = 0.8
HUG_THRESH = 0.2

MAX_WALL_ANGLE = 0.2  # radians (~11°), tune to your robot

KP = 0.05

MIN_FORWARD_WALL_DISTANCE = 0.4  # meters

ROBOT_FOOTPRINT = Vector3(0.4, 0.5, 0.3)

class State(Enum):
    INIT = auto()
    START = auto()
    WALL_HUG = auto()
    REFLEX_CORNER = auto()
    ACTUAL_CORNER = auto()
    DEAD_END = auto()             # WIP
    EXPLORED_AREA = auto()
    FLOOD_FILL = auto()           # WIP
    ASTAR = auto()  # WIP
    COMPLETE = auto()             # WIP

class Mapper:
    """
        robot will always try to align with right wall
        at distance scan_max-leeway
        if left wall is ALSO detected, it will try to center itself between the 2 walls

    """

    def __init__(self):
        rospy.init_node(NODE_NAME, anonymous=False)
        rospy.loginfo("--- Custom Mapping Algoritm ---")

        self.state = State.INIT
        self.mask = None
        self.scan = None
        self.start_transform = None
        self.map_mask = None
        self.forward_wall = None
        self.target_wall_distance = INIT_HUG_DIST

        self.cmd_pub = rospy.Publisher(CMD_TOPIC, Twist, queue_size=1)
        rospy.Subscriber(SCAN_TOPIC, LaserScan, self.scan_callback)
        rospy.Subscriber(MAP_TOPIC, OccupancyGrid, self.map_callback)

    #init
    
    def set_start_transform(self):
        self.start_transform = Transform((0,0,0), Quaternion(0,0,0,1))
        pass

    #runtime

    def tick(self):
        print("\033[2J\033[H", end="")
        if self.state == State.INIT:
            self.tick_init()
            pass
        if self.state == State.START:
            pass
        if self.state == State.WALL_HUG:
            self.tick_wallhug()
            pass
        pass

    def tick_init(self):
        self.state = State.WALL_HUG

    def tick_wallhug(self):
        cmd_ang, cmd_lin = 0, 0

        dist_right, ang_right = self.get_horizontal_wall(dir=0) #get left wall
        dist_left, ang_left = self.get_horizontal_wall(dir=1)
        dist_front = self.get_front_wall(max_bidirectional_samples=20)

        print("front dist: ", dist_front)
        print("dist_left:", dist_left, " | ang_left", ang_left)
        print("dist_right:", dist_right, " | ang_right", ang_right)


        #this is to get the robot position PHYSICALLY, relative to the robot's posn

        #this is top prio after state change
        if dist_right is None or ang_right is None \
            or dist_left is None or ang_left is None \
            or dist_front is None:
            return
            
        delta_x = 0
        if dist_left < INIT_HUG_DIST:
            #get middle point, then try to nav to middle point then stay mid     
            midpoint_distance = (dist_right + dist_left)/2  
            delta_x = (dist_right - dist_left)/2

            if delta_x < 0.1: #if less than 0.2, let wall hug algo take over instead
                angular_correction = KP * ang_right  # negative to reduce error

                max_angular_speed = 0.5  # rad/s
                angular_correction = max(-max_angular_speed, min(max_angular_speed, angular_correction))
                cmd_ang = angular_correction
                print("Hugging Via Midpoint")
            else:
                if dist_right - midpoint_distance > 0:
                    cmd_ang = -0.15
                    print("Rotating In")
                else:
                    cmd_ang = 0.15
                    print("Rotating Out")
                print("Midpoint Correction")

        else:
            if abs(dist_right - INIT_HUG_DIST) < 0.2:
                angular_correction = KP * ang_right 
                max_angular_speed = 0.5  # rad/s
                angular_correction = max(-max_angular_speed, min(max_angular_speed, angular_correction))
                cmd_ang = angular_correction
                print("trying to hug")
            else:
                if abs(ang_right) > MAX_WALL_ANGLE:
                    angular_correction = KP * ang_right 
                    max_angular_speed = 0.5  # rad/s
                    angular_correction = max(-max_angular_speed, min(max_angular_speed, angular_correction))
                    cmd_ang = angular_correction
                    print("snapBack")
                else:
                    if dist_right - INIT_HUG_DIST > 0:
                        cmd_ang = -0.15
                        print("Rotating In")
                    else:
                        cmd_ang = 0.15
                        print("Rotating Out")

            

        if dist_front < 0.4:
            self.publish_move_command(0,0)
        else:
            self.publish_move_command(LINEAR_SPEED, cmd_ang)
            #stop

        #first check for state transition
        # if right wall doesnt exist
        #     if front wall doesnt exist
        #         turn right with with turning radius based on hug dist up to 270 deg until right wall is found
        #     if front wall does exist
        #         if right wall doesnt exist
        #             do on the spot right turn 90 deg and move forward till a either front or right wall is found.
        # if right wall does exist
        #     if front wall does exist
        #         if left wall doesnt exist
        #              mark as acute corner, then turn left, front wall is now right wall
        #         if left wall does exist
        #             check the map, and fill dead end with "dead end" cells
        #             reverse, turn 90deg left and go forward until dead end is passed.

            
        # dist, ang = self.parallel_alignment()

        # #this takes precedence
        # if dist < INIT_HUG_DIST + HUG_THRESH:
        #     pass
        # if dist > INIT_HUG_DIST - HUG_THRESH:
        #     pass


        # Kp = 0.05  # proportional gain (tune as needed)
        # angular_correction = -Kp * wall_angle_deg  # negative to reduce error

        # # Clamp max rotation speed
        # max_angular_speed = 0.5  # rad/s
        # angular_correction = max(-max_angular_speed, min(max_angular_speed, angular_correction))

   
    def get_front_wall(self, max_bidirectional_samples=50):
        #currently its using lidar, but swap this to use camera after

        #IN FRONT got this stupid robotic arm BLOCKING.

        if self.scan is None:
            print("Scan is None")
            return None, None

        scan = self.scan 
        base_angle = math.pi

        index = int(round((base_angle - scan.angle_min) / scan.angle_increment))

        i_start = index - max_bidirectional_samples
        i_end   = index + max_bidirectional_samples

        i_start = max(0, i_start)
        i_end   = min(len(scan.ranges), i_end)

        wall_ranges = scan.ranges[i_start:i_end]

        # Filter out any inf or nan values
        valid = [d for d in wall_ranges if math.isfinite(d)]

        if valid:
            min_dist = min(valid)
        else:
            min_dist = None   # or some default value

        return min_dist

    def get_horizontal_wall(self, dir=0, max_bidirectional_samples=50, max_falloff=0.2):
        """
        get_horizontal_wall
        
        :param dir: 0 for left, 1 for right
        """
        if self.scan is None:
            print("Scan is None")
            return None, None

        scan = self.scan 
        if dir == 0:
            base_angle = math.pi / 2
        elif dir == 1:
            base_angle = -math.pi / 2

        index = int(round((base_angle - scan.angle_min) / scan.angle_increment))

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

        d_front = front_ranges[sample_index]  # distance to wall at front
        d_back  = back_ranges[sample_index]   # distance to wall at back

        theta_front = sample_index * scan.angle_increment  # radians
        theta_back  = sample_index * -scan.angle_increment

        x_front = d_front * math.cos(theta_front)
        y_front = d_front * math.sin(theta_front)

        x_back  = d_back  * math.cos(theta_back)
        y_back  = d_back  * math.sin(theta_back)

        dx = x_front - x_back
        dy = y_front - y_back

        wall_angle = math.atan2(dy, dx)  # radians
        wall_angle_deg = math.degrees(wall_angle) - 90

        numerator   = abs(x_back * y_front - y_back * x_front)
        denominator = math.sqrt((y_back - y_front)**2 + (x_back - x_front)**2)

        if denominator == 0:
            return None, None
        wall_distance = numerator / denominator

        return wall_distance, wall_angle_deg


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
    
    #utils

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

        d_front = front_ranges[sample_index]  # distance to wall at front
        d_back  = back_ranges[sample_index]   # distance to wall at back

        theta_front = sample_index * scan.angle_increment  # radians
        theta_back  = sample_index * -scan.angle_increment

        x_front = d_front * math.cos(theta_front)
        y_front = d_front * math.sin(theta_front)

        x_back  = d_back  * math.cos(theta_back)
        y_back  = d_back  * math.sin(theta_back)

        dx = x_front - x_back
        dy = y_front - y_back

        wall_angle = math.atan2(dy, dx)  # radians
        wall_angle_deg = math.degrees(wall_angle) - 90


        numerator   = abs(x_back * y_front - y_back * x_front)
        denominator = math.sqrt((y_back - y_front)**2 + (x_back - x_front)**2)
        wall_distance = numerator / denominator

        return wall_distance, wall_angle_deg

    def run(self):
        rate = rospy.Rate(20)  # 10 Hz
        while not rospy.is_shutdown():
            # if self.scan is None or self.map_data is None:
            #     rospy.loginfo("Waiting for SCAN and MAP data...")
            #     continue
            # pass

            self.tick()
            # right_wall_dist = self.get_right_wall()
            #self.parallel_alignment()
            # # print(f"Right wall distance: {right_wall_dist}")
            # # if right_wall_dist is None or right_wall_dist > 1:
            # #     self.publish_move_command(0,0)  # turn right
            # # else:
            # #     self.publish_move_command(LINEAR_SPEED, 0)  # move forward
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
