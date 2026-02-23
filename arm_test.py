#!/usr/bin/env python
# encoding: utf-8

import rospy
from transbot_msgs.msg import Arm, Joint 

NODE_NAME = 'arm_commander_test'
TOPIC_NAME = '/TargetAngle'

def move_arm():
    rospy.init_node(NODE_NAME, anonymous=False)
    pub = rospy.Publisher(TOPIC_NAME, Arm, queue_size=1)
    
    rospy.loginfo("Waiting for driver...")
    while pub.get_num_connections() == 0 and not rospy.is_shutdown():
        rospy.sleep(0.1)
    rospy.loginfo("Connected. Starting Sequence.")

    try:
        # --- SCENARIO 1: GO TO EXTENDED ---
        # TUCKED: J7: 225, J8: 30, J9: 30
        # EXTENDED: J7: 100, J8: 180, J9: 30
        arm_msg = Arm()
        
        j7 = Joint()
        j7.id = 7; j7.angle = 100; j7.run_time = 1500
        
        j8 = Joint()
        j8.id = 8; j8.angle = 180; j8.run_time = 1500
        
        j9 = Joint()
        j9.id = 9; j9.angle = 30; j9.run_time = 1000

        arm_msg.joint = [j7, j8, j9]
        
        rospy.loginfo("Action: Extending Arm...")
        pub.publish(arm_msg)
        rospy.sleep(2.5) # Allow time to reach extension

        # --- SCENARIO 2: GRIP (J9 to 70) ---
        arm_msg = Arm()
        
        j9_grip = Joint()
        j9_grip.id = 9; j9_grip.angle = 70; j9_grip.run_time = 800
        
        arm_msg.joint.append(j9_grip)
        
        rospy.loginfo("Action: Gripping (70 degrees)...")
        pub.publish(arm_msg)
        rospy.sleep(1.5)

        # --- SCENARIO 3: RETURN TO TUCKED ---
        # Holding the item (J9 stays at 70)
        arm_msg = Arm()
        
        j7_tuck = Joint()
        j7_tuck.id = 7; j7_tuck.angle = 220; j7_tuck.run_time = 2000
        
        j8_tuck = Joint()
        j8_tuck.id = 8; j8_tuck.angle = 30; j8_tuck.run_time = 2000
        
        j9_hold = Joint()
        j9_hold.id = 9; j9_hold.angle = 70; j9_hold.run_time = 1000

        arm_msg.joint = [j7_tuck, j8_tuck, j9_hold]
        
        rospy.loginfo("Action: Tucking Arm...")
        pub.publish(arm_msg)
        rospy.sleep(3.0)

    except rospy.ROSInterruptException:
        pass
    finally:
        rospy.loginfo("Test sequence finished.")

if __name__ == '__main__':
    move_arm()