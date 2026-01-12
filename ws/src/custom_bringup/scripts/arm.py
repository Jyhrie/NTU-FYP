#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import sys
import moveit_commander
from geometry_msgs.msg import Pose

def main():
    # Initialize MoveIt
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('transbot_pick_place', anonymous=True)

    # Note: Yahboom Transbot usually uses these specific group names
    arm = moveit_commander.MoveGroupCommander("arm_group")
    gripper = moveit_commander.MoveGroupCommander("gripper_group")

    # 1. Open Gripper
    print("Opening gripper...")
    gripper.set_named_target("open")
    gripper.go(wait=True)

    # 2. Move to Pick Position (Example coordinates in meters)
    # You should adjust these based on where your object is
    pick_pose = Pose()
    pick_pose.position.x = 0.15   # 15cm forward
    pick_pose.position.y = 0.0    # centered
    pick_pose.position.z = 0.12   # height
    pick_pose.orientation.w = 1.0 # simple orientation

    print("Moving to object...")
    arm.set_pose_target(pick_pose)
    arm.go(wait=True)

    # 3. Close Gripper
    print("Closing gripper...")
    gripper.set_named_target("close")
    gripper.go(wait=True)

    # 4. Lift and Put Down
    print("Lifting...")
    pick_pose.position.z += 0.05
    arm.set_pose_target(pick_pose)
    arm.go(wait=True)

    rospy.sleep(1)
    
    print("Putting down...")
    pick_pose.position.z -= 0.05
    arm.set_pose_target(pick_pose)
    arm.go(wait=True)
    
    gripper.set_named_target("open")
    gripper.go(wait=True)

if __name__ == '__main__':
    main()