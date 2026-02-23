#!/usr/bin/env python
# encoding: utf-8

import rospy
from transbot_msgs.msg import Arm, Joint 

def move_arm_staggered():
    rospy.init_node('arm_staggered_test')
    pub = rospy.Publisher('/TargetAngle', Arm, queue_size=1)
    
    rospy.loginfo("Waiting for driver...")
    while pub.get_num_connections() == 0 and not rospy.is_shutdown():
        rospy.sleep(0.1)

    # Helper to send a single joint command
    def send_single_joint(jid, angle, runtime):
        msg = Arm()
        j = Joint()
        j.id = jid
        j.angle = angle
        j.run_time = runtime
        msg.joint.append(j)
        pub.publish(msg)
        rospy.loginfo("Published Joint %d to Angle %d", jid, angle)
        # CRITICAL: Wait long enough for the serial bus to clear 
        # and the motor to start moving.
        rospy.sleep(0.2) 

    try:
        # --- STEP 1: EXTEND ---
        # Stagger the start of each motor
        send_single_joint(7, 100, 1500)
        send_single_joint(8, 180, 1500)
        send_single_joint(9, 30, 1000)
        rospy.sleep(2.0) # Wait for full extension

        # --- STEP 2: GRIP ---
        send_single_joint(9, 70, 800)
        rospy.sleep(1.0)

        # --- STEP 3: TUCK ---
        # Stagger the return
        send_single_joint(7, 220, 2000)
        send_single_joint(8, 30, 2000)
        send_single_joint(9, 70, 1000)
        rospy.sleep(2.5)

    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    move_arm_staggered()