#!/usr/bin/env python

import rospy
import json
from std_msgs.msg import String

class ArmControlNode:
    def __init__(self):
        rospy.init_node("arm_control_node")

        # Publishers
        # Sends {"data": "success"} or {"data": "failed"}
        self.status_pub = rospy.Publisher("/robot/reply", String, queue_size=1)
        
        # Sends 'done' to let the Controller know the physical movement finished
        self.movement_msg_pub = rospy.Publisher("/movement_controller_message", String, queue_size=1)

        # Subscriber
        self.cmd_sub = rospy.Subscriber("/controller/global", String, self.cmd_cb)

        rospy.loginfo("Arm Control Node: Listening for Grip/Release commands...")

    def cmd_cb(self, msg):
        """
        Routes the 'execute_pickup' (grip) and 'execute_release' commands.
        """
        # We assume the Controller sends these specific strings
        if msg.data == "execute_pickup":
            self.handle_grip()
        elif msg.data == "execute_release":
            self.handle_release()

    def handle_grip(self):
        rospy.loginfo("Mechanical Action: GRIP")
        
        # --- INSERT HARDWARE CODE HERE ---
        # Example: self.arm.close_gripper()
        rospy.sleep(2.0) # Simulate time taken to close
        # ---------------------------------

        # Check sensors to see if we actually caught something
        success = True # e.g., if gripper_width > threshold
        
        if success:
            self.status_pub.publish(json.dumps({"data": "success"}))
            self.movement_msg_pub.publish("done")
        else:
            self.status_pub.publish(json.dumps({"data": "failed"}))
            # We still publish 'done' so the state machine moves to the REVERSE state
            self.movement_msg_pub.publish("done")

    def handle_release(self):
        rospy.loginfo("Mechanical Action: RELEASE")
        
        # --- INSERT HARDWARE CODE HERE ---
        # Example: self.arm.open_gripper()
        rospy.sleep(1.0)
        # ---------------------------------

        # Usually release is always considered successful
        self.status_pub.publish(json.dumps({"data": "success"}))
        self.movement_msg_pub.publish("done")

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    node = ArmControlNode()
    node.run()