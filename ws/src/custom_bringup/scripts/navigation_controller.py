#!/usr/bin/env python3
import rospy
import numpy as np
import math
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist

import local_occupancy_movement as lom


class NavigationController:
    def __init__(self):
        rospy.init_node("navigation_controller")

        rospy.Subscriber("/local_costmap", OccupancyGrid, self.local_occupancy_callback)
        self.debug_pub = rospy.Publisher("/debug_map", OccupancyGrid, queue_size=10)
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        self.local_occupancy_movement = lom.LocalOccupancyNavigator()

        self.local_map_data = None
        self.received_first_map = False

    def run_once(self):
        """Runs ONE cycle: wait for map → trigger → move → rotate → exit."""
        rospy.loginfo("Waiting for first /local_costmap...")
        rate = rospy.Rate(20)

        # Block until callback gives us a map
        while not rospy.is_shutdown() and not self.received_first_map:
            rate.sleep()

        rospy.loginfo("Local costmap received. Running trigger...")

        # Run your algorithm
        msg, origin, end_position, goal_forward_vector = \
            self.local_occupancy_movement.trigger(self.local_map_data)

        # Publish debug map
        if msg:
            self.publish_debug_map(msg)

        # Perform movement
        self.move_to(end_position)

        # Rotate to final facing direction
        self.rotate_to(goal_forward_vector)

        rospy.loginfo("Navigation complete. Exiting.")
        rospy.signal_shutdown("Task complete.")

    # --------------------------------------------------------------
    # Movement and Rotation
    # --------------------------------------------------------------

    def move_to(self, end_position):
        """Move robot straight to the target point (robot-relative)."""
        ex, ey = end_position
        dist = math.sqrt(ex*ex + ey*ey)

        rospy.loginfo(f"Moving {dist:.2f}m toward {end_position}")

        rate = rospy.Rate(30)
        remaining = dist

        while remaining > 0.05 and not rospy.is_shutdown():
            tw = Twist()
            tw.linear.x = 0.2
            self.cmd_pub.publish(tw)

            # approximate travel
            remaining -= 0.2 * (1/30.0)
            rate.sleep()

        self.cmd_pub.publish(Twist())


    def rotate_to(self, forward_vec):
        """Rotate robot until facing the desired forward vector (robot-relative)."""
        fx, fy = forward_vec
        goal_angle = math.atan2(fx, fy)

        rospy.loginfo(f"Rotating to angle {goal_angle:.2f} rad")

        rate = rospy.Rate(30)

        while abs(goal_angle) > 0.05 and not rospy.is_shutdown():
            tw = Twist()
            tw.angular.z = 0.4 * math.copysign(1, goal_angle)
            self.cmd_pub.publish(tw)
            rate.sleep()

        self.cmd_pub.publish(Twist())


    # --------------------------------------------------------------
    # ROS Callbacks
    # --------------------------------------------------------------

    def local_occupancy_callback(self, msg):
        """Receive one map and store it."""
        self.local_map_data = msg
        self.received_first_map = True


    # --------------------------------------------------------------
    # Debug map publishing
    # --------------------------------------------------------------

    def publish_debug_map(self, msg):
        rospy.loginfo("Publishing debug map")
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        self.debug_pub.publish(msg)


# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------

if __name__ == "__main__":
    nav = NavigationController()
    nav.run_once()
