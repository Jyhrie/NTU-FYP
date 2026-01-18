#!/usr/bin/env python
# encoding: utf-8

import rospy
import time
import sys
# We must import the specific message types your driver checks for
from transbot_msgs.msg import Arm, Joint 

NODE_NAME = 'arm_commander'
TOPIC_NAME = '/TargetAngle' # Must match the subscriber in your driver

def move_arm():
    
    rospy.init_node(NODE_NAME, anonymous=False)
    
    # Publisher for the Arm message
    pub = rospy.Publisher(TOPIC_NAME, Arm, queue_size=1)
    
    # Wait for the driver to connect
    rospy.loginfo("Waiting for driver to connect...")
    while pub.get_num_connections() == 0 and not rospy.is_shutdown():
        rospy.sleep(0.1)
    rospy.loginfo("Driver connected. Moving arm...")
    
    try:
        # --- SCENARIO 1: Open the Claw (ID 9) ---
        arm_msg = Arm() # Create the container message
        
        joint_claw = Joint() 
        joint_claw.id = 9       # ID 9 is usually the claw
        joint_claw.angle = 30   # 30 degrees (Open)
        joint_claw.run_time = 1000 # Take 1000ms (1 second) to move
        
        # Add the joint to the message list
        arm_msg.joint.append(joint_claw)
        
        rospy.loginfo("Opening Claw...")
        pub.publish(arm_msg)
        rospy.sleep(2) # Wait for motion to finish

        # --- SCENARIO 2: Move Arm Up and Close Claw ---
        # We can send multiple joints at once
        multi_msg = Arm()
        
        # Joint 1: Lift the main arm (ID 7)
        j7 = Joint()
        j7.id = 7
        j7.angle = 100 # Middle position
        j7.run_time = 1000 

        # Joint 1: Lift the main arm (ID 7)
        j8 = Joint()
        j8.id = 8
        j8.angle = 60 # Middle position
        j8.run_time = 1000 
        
        # Joint 2: Close the claw (ID 9)
        j9 = Joint()
        j9.id = 9
        j9.angle = 180 # 180 degrees (Closed tight)
        j9.run_time = 1000

        multi_msg.joint = [j7, j8, j9] # Add both to the list

        rospy.loginfo("Moving Arm Up and Closing Claw...")
        pub.publish(multi_msg)
        rospy.sleep(2)

    except rospy.ROSInterruptException:
        pass
    finally:
        rospy.loginfo("Arm test complete.")

if __name__ == '__main__':
    move_arm()