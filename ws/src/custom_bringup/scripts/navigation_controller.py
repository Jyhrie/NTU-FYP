#!/usr/bin/env python
import rospy
import numpy as np
import math
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

import tf.transformations
from vectors import Vector2
import utils

from collections import deque


import local_occupancy_movement as lom

MAX_MOVEMENT_SPEED = 0.25
MAX_ANGULAR_SPEED = 0.15
ROBOT_SAFE_SQUARE_FOOTPRINT = 0.4

HUG_DISTANCE = 0.2  # meters
TURN_SAFE_DISTANCE = 0.2
TURN_THRESH_STEPS = 7

class CommandType:
    TURN = 0
    MOVE = 1
    MOVE_BY_VECTOR = 2
    SCAN = 3
    LOCAL_SCAN = 4
    UPDATE_MAP = 5

class Command(object):
    def __init__(self, cmd_type, target_vec=None, target_yaw=None, magnitude=None, res=None):
        self.cmd_type = cmd_type
        self.target_vec = target_vec
        self.target_yaw = target_yaw
        self.magnitude = magnitude
        self.res = res

    def __repr__(self):
        # Useful for debugging the queue
        return "Command(Type=%s, Target=%s)" % (self.cmd_type, self.target_vec)
    
class NavigationController:
    def __init__(self):
        rospy.init_node("navigation_controller")

        # subscribers
        rospy.Subscriber("/local_costmap", OccupancyGrid, self.local_costmap_cb)
        rospy.Subscriber("/odom", Odometry, self.odom_cb)

        # publishers
        self.debug_pub = rospy.Publisher("/debug_map", OccupancyGrid, queue_size=1)
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        self.local_occupancy_movement = lom.LocalOccupancyNavigator()

        self.have_map = False
        self.have_odom = False

        self.x, self.y = 0, 0
        self.roll, self.pitch, self.yaw = 0,0,0

        self._queue = deque()

        pass

    def enqueue(self, command):
        """Add a command to the end of the line."""
        self._queue.append(command)

    def dequeue(self):
        """Remove and return the next command to execute."""
        if len(self._queue) == 0:
            return None
        return self._queue.popleft()

    def cutqueue(self, command):
        """Priority: Add to the very front (next to be executed)."""
        self._queue.appendleft(command)

    def local_costmap_cb(self, msg):
        self.local_map_msg = msg
        self.have_map = True

    def odom_cb(self, msg):
        self.odom = msg

        if msg is not None:
            self.have_odom = True

            self.x = msg.pose.pose.position.x
            self.y = msg.pose.pose.position.y

            q = msg.pose.pose.orientation
            quaternion = [q.x, q.y, q.z, q.w]
            self.roll, self.pitch, self.yaw = tf.transformations.euler_from_quaternion(quaternion) #in rads.

    def display_debug_map(self, msg):
        #msg, normal_vec, inlier = self.local_occupancy_movement.trigger(self.local_map_msg)
        if msg is not None:
            self.debug_pub.publish(msg)
        return
        
    def local_mapping_decision_maker(self, samples=5):
        #init
        rate_local_poll = rospy.Rate(3)  # 3 Hz
        res = self.local_map_msg.info.resolution

        #declare initial list for polling
        average_inlier_vec = []
        inlier_list = []
        outlier_list = []

        median_inlier = None
        median_outlier = None
        last_inlier = None

        #get 5 samples of the walls at time steps of 1/rate
        for i in range(0,samples): #get samples
            msg, avg_inlier, inlier, outlier, vert_hit_distance = self.local_occupancy_movement.trigger(self.local_map_msg)
            average_inlier_vec.append(avg_inlier)
            inlier_list.append(inlier)
            outlier_list.append(outlier)
            rate_local_poll.sleep()

        #process inliers
        inlier_point_list_x = []
        inlier_point_list_y = []

        for sample in inlier_list:
            inlier_len = len(sample)
            count_inlier = min(3, inlier_len) #get minimum samples of inliers.
            for i in range(0,count_inlier):
                inlier_point_list_x.append(sample[i].x)
                inlier_point_list_y.append(sample[i].y)

        inlier_point_list_x.sort()   
        inlier_point_list_y.sort()
        
        if inlier_point_list_x:
            median_inlier = Vector2(
                inlier_point_list_x[len(inlier_point_list_x)//2],
                inlier_point_list_y[len(inlier_point_list_y)//2]
            )

        outlier_point_list_x = []
        outlier_point_list_y = []

        for sample in outlier_list:
            outlier_len = len(sample)
            count_outlier = min(1, outlier_len) #get 1 samples of inliers.
            for i in range(0,count_outlier):
                outlier_point_list_x.append(sample[i].x)
                outlier_point_list_y.append(sample[i].y)

        outlier_point_list_x.sort()   
        outlier_point_list_y.sort()
        
        if outlier_point_list_x:
            median_outlier = Vector2(
                outlier_point_list_x[len(outlier_point_list_x)//2], #perfect world, this will all return same value
                outlier_point_list_y[len(outlier_point_list_y)//2]
            )

        if median_outlier is None: #there is no outliers, wall is perfect
            inlier_point_list_x = []
            inlier_point_list_y = []

            for sample in inlier_list:
                inlier_len = len(sample)
                count_inlier = min(3, inlier_len) #get minimum samples of inliers.
                for i in range(0,count_inlier):
                    inlier_point_list_x.append(sample[inlier_len-i-1].x)
                    inlier_point_list_y.append(sample[inlier_len-i-1].y)

            inlier_point_list_x.sort()   
            inlier_point_list_y.sort()
            
            if inlier_point_list_x is not None:
                last_inlier = Vector2(
                    inlier_point_list_x[len(inlier_point_list_x)//2],
                    inlier_point_list_y[len(inlier_point_list_y)//2]
                )

        #compute average normal vector
        wall_vec_sum = Vector2(0,0)
        for vec in average_inlier_vec:
            wall_vec_sum.add(vec)
            average_wall_vec_median = wall_vec_sum.normalize()

        normal_vec_median = average_wall_vec_median.normal()
        
        #project normal vector onto 1st point of detected wall to obtain hug point
        target_point = Vector2(median_inlier.x + (normal_vec_median.x * (HUG_DISTANCE) / res), #negative as we want the < direction
                                median_inlier.y + (normal_vec_median.y * (HUG_DISTANCE) / res)) #postitive as we want ^ direction
                                
        #move to target point
        cx = self.local_occupancy_movement.map_width // 2
        cy = self.local_occupancy_movement.map_height // 2

        dx = target_point.x - cx
        dy = target_point.y - cy

        govec = Vector2(dx, dy)

        print("Average Wall Median: ", average_wall_vec_median)

        if math.hypot(dy, dx) < (ROBOT_SAFE_SQUARE_FOOTPRINT / res) and vert_hit_distance < TURN_THRESH_STEPS: #robot has no space to move forward anymore, cant turn right
            print("Vert Hit Distance: ", vert_hit_distance)
            left_vec = Vector2(-1,0)
            relative_angle = utils.angle_between(Vector2(0,-1), left_vec) 
            target_yaw = utils.normalize_angle(self.yaw - relative_angle)
            self.enqueue(Command(CommandType.TURN, target_yaw=target_yaw))

        elif math.hypot(dy, dx) < (ROBOT_SAFE_SQUARE_FOOTPRINT / res): #robot doesnt need to mvoe closer to the wall
            print(dy, dx, ROBOT_SAFE_SQUARE_FOOTPRINT / res)
            print("Point Within Region")
            relative_angle = utils.angle_between(Vector2(0, -1), average_wall_vec_median)
            target_yaw = utils.normalize_angle(self.yaw - relative_angle)

            #enqueue turn to wall tangent
            self.enqueue(Command(CommandType.TURN, target_yaw=target_yaw))
            if median_outlier is not None: #has detected an outlier
                print("Median Outlier: ", median_outlier)
                print("average_wall_vec_median: ", average_wall_vec_median)

                #project normal from first point of outlier
                projected_median_outlier_x = median_outlier.x + ((normal_vec_median.x * HUG_DISTANCE) / res)
                projected_median_outlier_y = median_outlier.y + ((normal_vec_median.y * HUG_DISTANCE) / res)

                print("Projected Normal: ", projected_median_outlier_x, projected_median_outlier_y)

                #project tangent from previous point back towards robot
                projected_median_outlier_x -= ((average_wall_vec_median.x * (TURN_SAFE_DISTANCE)) / res)
                projected_median_outlier_y -= ((average_wall_vec_median.y * (TURN_SAFE_DISTANCE)) / res)

                print("Projected Tangent: ", projected_median_outlier_x, projected_median_outlier_y)

                #delta vector between robot origin and double projected point
                stop_vec_x = cx - projected_median_outlier_x
                stop_vec_y = -(cy - projected_median_outlier_y)

                stop_vec = Vector2(stop_vec_x, stop_vec_y)

                print("Stop Vec: ", stop_vec)
                if(stop_vec.y < 0): #due to safety, robot's expected position is BEHIND its current position
                    mag_dist_to_stop_point = stop_vec.mag() * res
                    self.enqueue(Command(CommandType.MOVE, magnitude=mag_dist_to_stop_point))

                else: #scoot close to wall such that next wall tangent is detected.
                    self.enqueue(Command(CommandType.MOVE, magnitude=(0.3)*res)) #should be move forward by 1 robot's distance or however much the tolerance is on the forward projection



            elif last_inlier is not None:
                print("Last Inlier: ", last_inlier)
                print("average_wall_vec_median: ", average_wall_vec_median)

                                #project normal from first point of outlier
                projected_median_outlier_x = last_inlier.x + ((normal_vec_median.x * HUG_DISTANCE) / res)
                projected_median_outlier_y = last_inlier.y + ((normal_vec_median.y * HUG_DISTANCE) / res)

                print("Projected Normal: ", projected_median_outlier_x, projected_median_outlier_y)

                #project tangent from previous point back towards robot
                projected_median_outlier_x -= ((average_wall_vec_median.x * (TURN_SAFE_DISTANCE)) / res)
                projected_median_outlier_y -= ((average_wall_vec_median.y * (TURN_SAFE_DISTANCE)) / res)

                print("Projected Tangent: ", projected_median_outlier_x, projected_median_outlier_y)

                #delta vector between robot origin and double projected point
                stop_vec_x = cx - projected_median_outlier_x
                stop_vec_y = -(cy - projected_median_outlier_y)

                # stop_vec = Vector2(cx - median_outlier.x + ((normal_vec_median.x * HUG_DISTANCE) / res) + ((average_wall_vec_median.x * (TURN_SAFE_DISTANCE)) / res),
                #                     -(cy - median_outlier.y + ((normal_vec_median.y * HUG_DISTANCE) / res) + ((average_wall_vec_median.y * (TURN_SAFE_DISTANCE)) / res))) #need to flip the sign of average_wall_vec median for it to not overshoot since -Y is forward
                
                stop_vec = Vector2(stop_vec_x, stop_vec_y)

                # stop_vec = Vector2(cx - last_inlier.x + ((normal_vec_median.x * HUG_DISTANCE) / res) - ((average_wall_vec_median.x * (TURN_SAFE_DISTANCE)) / res),
                #                     -(cy - last_inlier.y + ((normal_vec_median.y * HUG_DISTANCE) / res) - ((average_wall_vec_median.y * (TURN_SAFE_DISTANCE)) / res)))
                mag_dist_to_stop_point = stop_vec.mag() * res

                print("Stop Vec: ", stop_vec)
                if(stop_vec.y < 0):
                    mag_dist_to_stop_point = stop_vec.mag() * res
                    self.enqueue(Command(CommandType.MOVE, magnitude=mag_dist_to_stop_point))

            self.enqueue(Command(CommandType.SCAN))
                
                
            #enqueue update local map

            
        
        else: #robot robot is to get to hug distance from the wall
            #robot needs to move out/in
            #enqueue turn to projected wall normal
            self.enqueue(Command(CommandType.MOVE_BY_VECTOR, target_vec=govec, res=res))

            #get yaw of wall tangent (average_wall_vec_median)
            relative_angle = utils.angle_between(Vector2(0, -1), average_wall_vec_median)
            target_yaw = utils.normalize_angle(self.yaw - relative_angle)

            self.enqueue(Command(CommandType.TURN, target_yaw=target_yaw))
            #enqueue update local

            #enqueue rescan
            self.enqueue(Command(CommandType.SCAN))
        
        mx = int(round(target_point.x))
        my = int(round(target_point.y))

        width = msg.info.width
        height = msg.info.height

        grid = np.array(msg.data, dtype=np.int8).reshape((height, width))
        grid[my, mx] = 2  # set cell

        # flatten back to msg.data
        msg.data = grid.flatten().tolist()

        self.display_debug_map(msg)
        pass

    def state_move_by_vector(self, target_vec, res):
        """
        this is always with respect to North -Y.
        """
        dist = target_vec.mag()
        relative_angle = utils.angle_between(Vector2(0,-1), target_vec.normalize()) #relative to north
        target_yaw = utils.normalize_angle(self.yaw - relative_angle)
        

        print("Distance: ", dist, "Vector: ", target_vec, "Resolution: ", res)

        self.cutqueue(Command(CommandType.MOVE, magnitude=dist*res))
        self.cutqueue(Command(CommandType.TURN, target_yaw=target_yaw))

    def get_local_route(self, samples=5):
        """
        gets the local route from the local occupancy movement module
        """
        rate = rospy.Rate(3)  # 5 Hz

        average_inlier_vec = []
        inlier_list = []

        for i in range(0,samples):
            msg, avg_inlier, inlier = self.local_occupancy_movement.trigger(self.local_map_msg)
            average_inlier_vec.append(avg_inlier)
            inlier_list.append(inlier)
            rate.sleep()

        # Take up to 5 samples (or fewer if not enough)

        inlier_point_list_x = []
        inlier_point_list_y = []

        for sample in inlier_list:
            #get first n
            inlier_len = len(sample)
            get_count = min(3, inlier_len)
            for i in range(0,get_count):
                inlier_point_list_x.append(sample[i].x)
                inlier_point_list_y.append(sample[i].y)
                
        inlier_point_list_x.sort()        
        inlier_point_list_y.sort()

        if inlier_point_list_x is not None:
            median_inlier = Vector2(
                inlier_point_list_x[len(inlier_point_list_x)//2],
                inlier_point_list_y[len(inlier_point_list_y)//2]
            )

        #compute average normal vector
        wall_vec_sum = Vector2(0,0)
        for vec in average_inlier_vec:
            wall_vec_sum.add(vec)
            average_wall_vec_median = wall_vec_sum.normalize()

        normal_vec_median = average_wall_vec_median.normal()
        # print("average_inliner_vec", average_inlier_vec)

        res = self.local_map_msg.info.resolution
        # print("Median inlier", median_inlier)
        # print("Normal Vector", normal_vec_median)

        #compute target point to hug wall
        target_point = Vector2(median_inlier.x + (normal_vec_median.x * (HUG_DISTANCE) / res), #negative as we want the < direction
                                median_inlier.y + (normal_vec_median.y * (HUG_DISTANCE) / res)) #postitive as we want ^ direction
        
        #move to target point
        cx = self.local_occupancy_movement.map_width // 2
        cy = self.local_occupancy_movement.map_height // 2

        dx = target_point.x - cx
        dy = target_point.y - cy

        mx = int(round(target_point.x))
        my = int(round(target_point.y))

        width = msg.info.width
        height = msg.info.height

        grid = np.array(msg.data, dtype=np.int8).reshape((height, width))
        grid[my, mx] = 2  # set cell

        # flatten back to msg.data
        msg.data = grid.flatten().tolist()

        self.display_debug_map(msg)
        print(dx, dy)

        govec = Vector2(dx, dy).normalize()

        # print("North Vector: ", Vector2(0,-1), "Target Vector: ", govec)

        relative_angle = utils.angle_between(Vector2(0,-1), govec) #relative to north
        target_yaw = utils.normalize_angle(self.yaw - relative_angle)
        
        # print("Current Yaw: ", self.yaw, "Target Yaw: ", target_yaw)
        # print("Target Angle (deg):", math.degrees(relative_angle), "Target Angle (rad):", relative_angle)

        while self.turn_to_target_yaw(target_yaw):
            rospy.sleep(0.02)

        # print("Turn Ended at Yaw: ", self.yaw)

        # while not self.nav_to_vec(govec):
        #     rospy.sleep(0.02)

    def turn_to_face_vec(self, target_yaw):
        """
        Rotate the robot to face the target yaw (radians).
        Returns True if still turning, False if finished.
        """
        ANGLE_THRESH = math.radians(2.5)  # ~2.5 degrees tolerance
        MAX_ANGULAR_SPEED = 0.6           # rad/s

        # Compute smallest angular difference
        angle_diff = utils.normalize_angle(target_yaw - self.yaw)

        cmd = Twist()

        if abs(angle_diff) < ANGLE_THRESH:
            # Finished turning
            self.cmd_pub.publish(Twist())  # stop rotation
            return False

        # Determine direction and speed
        cmd.angular.z = max(-MAX_ANGULAR_SPEED, min(MAX_ANGULAR_SPEED, angle_diff))

        # Optional: scale speed proportionally to angle_diff (smooth approach)
        # cmd.angular.z = max(-MAX_ANGULAR_SPEED, min(MAX_ANGULAR_SPEED, angle_diff))
        #print(cmd)
        self.cmd_pub.publish(cmd)
        return True

    def move_forward_by_magnitude(self, magnitude):
            """
            Moves the robot forward by a specific distance (magnitude) 
            relative to its current orientation.
            """
            # 1. Wait for odom to be available
            while not self.have_odom and not rospy.is_shutdown():
                rospy.sleep(0.1)

            # 2. Record starting position
            start_x = self.x
            start_y = self.y
            
            rate = rospy.Rate(30) # 10Hz control loop
            move_cmd = Twist()
            move_cmd.linear.x = MAX_MOVEMENT_SPEED

            dist_moved = 0.0

            while dist_moved < magnitude and not rospy.is_shutdown():
                # 3. Publish velocity
                self.cmd_pub.publish(move_cmd)
                
                # 4. Calculate Euclidean distance from start
                curr_x = self.x
                curr_y = self.y
                
                # Distance Formula: sqrt((x2-x1)^2 + (y2-y1)^2)
                dist_moved = math.hypot(curr_x - start_x, curr_y - start_y)
                
                

            # 5. Stop the robot
            self.cmd_pub.publish(Twist()) 
            rospy.loginfo("Reached target magnitude: %f" % dist_moved)
    
    def update_global_costmap(self):
        """
        updates the global costmap from the /map topic
        """
        pass

    def check_against_global_map(self):
        """
        checks against the global costmap to see if robot is turning into a spot where it has been before,
        """
        pass

    def state_scan(self):
        self.enqueue(Command(CommandType.LOCAL_SCAN))
        pass
    
    def state_localscan(self):
        self.local_mapping_decision_maker(samples=5)
        pass
    
    def run_once(self):
        """
        Rotates the robot 90 degrees (PI/2) clockwise using odometry feedback.
        """
        self.rate = rospy.Rate(5)
        while not self.have_odom:
            rospy.logwarn("Cannot rotate: No Odom data received yet.")
            self.rate.sleep()

        rospy.loginfo("Starting 90 degree clockwise turn...")

        # 1. Define targets
        target_rad = 90 * (math.pi / 180)  # Convert 90deg to radians (~1.57)
        angular_speed = -0.2              # Negative for clockwise rotation (adjust speed as needed)

        
        
        # 2. Track relative angle
        current_angle_turned = 0.0
        last_yaw = self.yaw

        twist = Twist()
        twist.angular.z = angular_speed

        self.rate = rospy.Rate(60)

        # 3. Loop until we have turned enough
        while current_angle_turned < target_rad and not rospy.is_shutdown():
            self.cmd_pub.publish(twist)
            self.rate.sleep()

            # Calculate the change in angle since the last loop
            current_yaw = self.yaw
            delta_yaw = current_yaw - last_yaw

            # --- HANDLE WRAP AROUND ---
            # If we cross from -PI to +PI or vice versa, delta will be huge (~6.28).
            # We normalize it to be within -PI and +PI.
            if delta_yaw < -math.pi:
                delta_yaw += 2 * math.pi
            elif delta_yaw > math.pi:
                delta_yaw -= 2 * math.pi
            
            # Add the magnitude of the change to our total
            current_angle_turned += abs(delta_yaw)
            
            last_yaw = current_yaw

        # 4. Stop the robot
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)
        rospy.loginfo("Rotation complete.")
        pass

    def run(self):
        rate = rospy.Rate(30)  # 5 Hz
        while not rospy.is_shutdown():
            if self.have_map and self.have_odom:
                try:
                    user_input = raw_input("Press A to run local route: ").strip().lower()
                    if user_input == 'a':
                        rospy.loginfo("Running local route")
                        # self.get_local_route(samples=5)
                        self.local_mapping_decision_maker(samples=5)
                        while len(self._queue) > 0:
                            self.fsm()
                except KeyboardInterrupt:
                    rospy.loginfo("Exiting...")
                    break
            rate.sleep()

    def fsm(self):
        fsm_rate = rospy.Rate(60)
        cmd = self.dequeue()
        cmd_type = cmd.cmd_type
        if cmd_type == CommandType.TURN:
            print("STATE TURN: Turning To", cmd.target_yaw)
            while self.turn_to_face_vec(cmd.target_yaw):
                fsm_rate.sleep()
        elif cmd_type == CommandType.MOVE:
            print("STATE MOVE, DIST: ", cmd.magnitude)
            self.move_forward_by_magnitude(cmd.magnitude)
            fsm_rate.sleep()
        elif cmd_type == CommandType.MOVE_BY_VECTOR:
            print("STATE UNPACK")
            self.state_move_by_vector(cmd.target_vec, cmd.res)
        elif cmd_type == CommandType.SCAN:
            print("SCAN")
            self.state_scan()
        elif cmd_type == CommandType.LOCAL_SCAN:
            print("LOCAL SCAN")
            self.state_localscan()
            pass


if __name__ == "__main__":



    nav = NavigationController()
    nav.run()
