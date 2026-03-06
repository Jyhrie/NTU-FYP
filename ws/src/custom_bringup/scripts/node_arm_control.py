#!/usr/bin/env python
# encoding: utf-8

import rospy
import json
from std_msgs.msg import String
from transbot_msgs.msg import Arm, Joint

class ArmControlNode:
    def __init__(self):
        rospy.init_node("arm_control_node")

        # Publishers
        self.status_pub = rospy.Publisher("/robot/reply", String, queue_size=1)
        self.movement_msg_pub = rospy.Publisher("/movement_controller_message", String, queue_size=1)
        
        # Hardware Publisher
        self.arm_pub = rospy.Publisher('/TargetAngle', Arm, queue_size=1)

        # Subscriber
        self.cmd_sub = rospy.Subscriber("/controller/global", String, self.cmd_cb)

        rospy.loginfo("Arm Control Node: Ready for Grab/Release sequences.")

    def send_single_joint(self, jid, angle, runtime):
        """
        Staggers joint commands to prevent UART serial buffer overflow.
        """
        msg = Arm()
        j = Joint()
        j.id = jid
        j.angle = angle
        j.run_time = runtime
        msg.joint.append(j)
        self.arm_pub.publish(msg)
        # 0.3s is the 'sweet spot' for Transbot serial stability
        rospy.sleep(0.3)

    def cmd_cb(self, msg):
        try:
            # Check if it's a JSON command or a raw string
            if msg.data.startswith('{'):
                cmd_dict = json.loads(msg.data)
                if cmd_dict.get("header") == "arm":
                    command = cmd_dict.get("command")
                else:
                    return
            else:
                command = msg.data

            # Route commands
            if command == "grab":
                self.handle_grab_sequence()
            elif command == "release":
                self.handle_release_sequence()
        except Exception as e:
            rospy.logerr("Error parsing arm command: %s", e)

    def handle_grab_sequence(self):
        rospy.loginfo("Mechanical Action: START GRAB SEQUENCE")
        
        try:
            # 1. EXTEND
            self.send_single_joint(7, 100, 1500)
            self.send_single_joint(8, 180, 1100)
            self.send_single_joint(9, 30, 1000)
            rospy.sleep(2.0) # Wait for extension to finish

            # 2. CLOSE GRIPPER
            self.send_single_joint(9, 85, 800)
            rospy.sleep(1.0)

            # 3. HALF TUCK (holding item)
            self.send_single_joint(7, 220, 2000)
            self.send_single_joint(8, 50, 2000)
            # Re-affirm grip just in case
            self.send_single_joint(9, 85, 1000)
            rospy.sleep(2.5)

            # --- Success Check ---
            # If your robot has gripper sensors, implement check here. 
            # Otherwise, we assume success for the state machine.
            self.status_pub.publish(json.dumps({"data": "success"}))
            self.movement_msg_pub.publish("done")
            rospy.loginfo("Grab Sequence Complete.")

        except Exception as e:
            rospy.logerr("Grab sequence failed: %s", e)
            self.status_pub.publish(json.dumps({"data": "failed"}))
            self.movement_msg_pub.publish("done")

    def handle_release_sequence(self):
        rospy.loginfo("Mechanical Action: START RELEASE SEQUENCE")
        
        try:
            # 1. RELEASE item
            self.send_single_joint(9, 30, 800)
            rospy.sleep(1.0)

            # 2. FULL TUCK
            self.send_single_joint(7, 220, 2000)
            self.send_single_joint(8, 30, 2000)
            # Close claw in tuck position for safety
            self.send_single_joint(9, 85, 1000)
            rospy.sleep(2.5)

            self.status_pub.publish(json.dumps({"data": "success"}))
            self.movement_msg_pub.publish("done")
            rospy.loginfo("Release Sequence Complete.")

        except Exception as e:
            rospy.logerr("Release sequence failed: %s", e)
            self.movement_msg_pub.publish("done")

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    node = ArmControlNode()
    node.run()