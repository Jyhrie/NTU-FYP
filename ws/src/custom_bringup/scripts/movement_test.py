#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
import time
import sys

# --- Configuration ---
LINEAR_SPEED = 0.5  # meters/second
MOTION_DURATION = 0.5 # seconds for each motion (forward and reverse)
NODE_NAME = 'simple_motion_publisher'
TOPIC_NAME = 'cmd_vel'

def move_robot():
    """
    Initializes a ROS node and publishes Twist messages to
    move the robot forward for a duration, stop, and then reverse for a duration.
    """
    rospy.init_node(NODE_NAME, anonymous=True)
    rospy.loginfo("--- Python Version Check ---")
    rospy.loginfo("Running with Python: {}".format(sys.version.split('\n')[0]))
    rospy.loginfo("----------------------------")
    pub = rospy.Publisher(TOPIC_NAME, Twist, queue_size=1)
    rate = rospy.Rate(10) # 10hz publishing rate (optional, but good practice)

    # *** THE CRITICAL CHANGE: Use wait_for_connection() ***
    rospy.loginfo("Waiting indefinitely for a subscriber to connect to {}...".format(TOPIC_NAME))
    try:
        # This function blocks until a subscriber is connected.
        # This is the most reliable ROS method for this purpose.
        pub.wait_for_connection()
        rospy.loginfo("Subscriber connected! Starting motion sequence.")
    except rospy.ROSInterruptException:
        # Allows the script to exit gracefully if Ctrl+C is pressed while waiting.
        rospy.logwarn("Script interrupted while waiting for connection.")
        return # Exit the function if interrupted

    # 1. Initialize Twist messages
    forward_twist = Twist()
    forward_twist.linear.x = LINEAR_SPEED # Positive linear.x for forward
    forward_twist.angular.z = 0.0

    reverse_twist = Twist()
    reverse_twist.linear.x = -LINEAR_SPEED # Negative linear.x for reverse
    reverse_twist.angular.z = 0.0

    stop_twist = Twist()
    # All linear and angular components are 0 by default, ensuring a stop

    start_time = time.time()

    try:
        # --- 2. Move Forward ---
        #rospy.loginfo(f"Moving Forward at {LINEAR_SPEED} m/s for {MOTION_DURATION} seconds...")
        
        while time.time() - start_time < MOTION_DURATION and not rospy.is_shutdown():
            pub.publish(forward_twist)
            rate.sleep()

        # --- 3. Stop (Briefly) ---
        rospy.loginfo("Stopping...")
        pub.publish(stop_twist) # Send a single stop command
        time.sleep(0.1) # Brief pause before reverse motion

        # --- 4. Move Reverse ---
        #rospy.loginfo(f"Moving Reverse at {-LINEAR_SPEED} m/s for {MOTION_DURATION} seconds...")
        start_time = time.time() # Reset timer for reverse motion

        while time.time() - start_time < MOTION_DURATION and not rospy.is_shutdown():
            pub.publish(reverse_twist)
            rate.sleep()

    except rospy.ROSInterruptException:
        pass

    finally:
        # --- 5. Final Stop and Cleanup ---
        rospy.loginfo("Motion sequence complete. Stopping robot.")
        # Ensure the robot stops before the node shuts down
        pub.publish(stop_twist)

if __name__ == '__main__':
    try:
        move_robot()
    except rospy.ROSInterruptException:
        pass