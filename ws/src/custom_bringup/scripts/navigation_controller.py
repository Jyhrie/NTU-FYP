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

import local_occupancy_movement as lom

MAX_MOVEMENT_SPEED = 0.25
MAX_ANGULAR_SPEED = 0.15

HUG_DISTANCE = 0.2  # meters

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

        self.row, self.pitch, self.yaw = 0,0,0
        pass

    def local_costmap_cb(self, msg):
        self.local_map_msg = msg
        self.have_map = True

    def odom_cb(self, msg):
        self.odom = msg

        if msg is not None:
            self.have_odom = True
            q = msg.pose.pose.orientation
            quaternion = [q.x, q.y, q.z, q.w]
            self.roll, self.pitch, self.yaw = tf.transformations.euler_from_quaternion(quaternion) #in rads.


    def display_debug_map(self, msg):
        #msg, normal_vec, inlier = self.local_occupancy_movement.trigger(self.local_map_msg)
        if msg is not None:
            self.debug_pub.publish(msg)
        return
    
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
        print("average_inliner_vec", average_inlier_vec)

        res = self.local_map_msg.info.resolution
        print("Median inlier", median_inlier)
        print("Normal Vector", normal_vec_median)

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

        # govec = Vector2(dx, dy).normalize()

        # print("North Vector: ", Vector2(0,-1), "Target Vector: ", govec)

        # relative_angle = utils.angle_between(Vector2(0,-1), govec) #relative to north
        # target_yaw = utils.normalize_angle(self.yaw + relative_angle)
        
        # print("Current Yaw: ", self.yaw, "Target Yaw: ", target_yaw)
        # print("Target Angle (deg):", math.degrees(relative_angle), "Target Angle (rad):", relative_angle)

        target_yaw = self.yaw - math.pi/2
        target_yaw = (target_yaw + math.pi) % (2*math.pi) - math.pi

        while self.turn_to_face_vec(target_yaw):
            rospy.sleep(0.02)

        print("Turn Ended at Yaw: ", self.yaw)

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

    def nav_to_vec(self, vec):
        """
        Move forward until target relative vector distance is reached
        """
        target_dist = math.hypot(vec.x, vec.y)

        if not hasattr(self, "_nav_start"):
            p = self.odom.pose.pose.position
            self._nav_start = (p.x, p.y)

        p = self.odom.pose.pose.position
        dx = p.x - self._nav_start[0]
        dy = p.y - self._nav_start[1]
        traveled = math.hypot(dx, dy)

        cmd = Twist()

        DIST_THRESH = 0.05
        MAX_LIN = 0.02

        if traveled < target_dist - DIST_THRESH:
            cmd.linear.x = MAX_LIN
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            return False
        else:
            self.cmd_pub.publish(Twist())
            del self._nav_start
            return True

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

    def run_once(self):
        """
        Rotates the robot 90 degrees (PI/2) clockwise using odometry feedback.
        """
        self.rate = 5
        while not self.have_odom:
            rospy.logwarn("Cannot rotate: No Odom data received yet.")
            self.rate.sleep()

        rospy.loginfo("Starting 90 degree clockwise turn...")

        # 1. Define targets
        target_rad = 90 * (math.pi / 180)  # Convert 90deg to radians (~1.57)
        angular_speed = -0.5               # Negative for clockwise rotation (adjust speed as needed)
        
        # 2. Track relative angle
        current_angle_turned = 0.0
        last_yaw = self.yaw

        twist = Twist()
        twist.angular.z = angular_speed

        self.rate = 60

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
            # if self.have_map and self.have_odom:
                # try:
                #     user_input = raw_input("Press A to run local route: ").strip().lower()
                #     if user_input == 'a':
                #         rospy.loginfo("Running local route")
                #         self.get_local_route(samples=5)
                # except KeyboardInterrupt:
                #     rospy.loginfo("Exiting...")
                #     break
            rate.sleep()


if __name__ == "__main__":
    nav = NavigationController()
    nav.run_once()
