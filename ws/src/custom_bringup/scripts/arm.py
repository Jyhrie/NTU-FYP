#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import sys
import moveit_commander

def main():
    # 1. Initialize
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('transbot_blind_pick', anonymous=True)

    arm = moveit_commander.MoveGroupCommander("arm")
    gripper = moveit_commander.MoveGroupCommander("Jaws")

    # Set speed (Safety first! 20% speed)
    arm.set_max_velocity_scaling_factor(0.2)

    # 2. Open Gripper
    print("Step 1: Opening Jaws...")
    # 'open' is a standard named pose, if it fails, we use joint values
    try:
        gripper.set_named_target("open")
    except:
        gripper.set_joint_value_target([0.0]) # Adjust based on your gripper
    gripper.go(wait=True)

    # 3. Move Arm to Pick Position
    # joint_goals: [arm_joint1, arm_joint2] in Radians
    # CHANGE THESE NUMBERS to your measured 'Blind' spot
    print("Step 2: Moving to Pick Position...")
    pick_joint_goal = [0.0, -0.6] 
    arm.set_joint_value_target(pick_joint_goal)
    arm.go(wait=True)
    rospy.sleep(1)

    # 4. Close Gripper
    print("Step 3: Closing Jaws...")
    try:
        gripper.set_named_target("close")
    except:
        gripper.set_joint_value_target([-0.5]) # Adjust to grip tightness
    gripper.go(wait=True)

    # 5. Lift Object
    print("Step 4: Lifting...")
    lift_joint_goal = [0.0, 0.0] # Return to 'up' position
    arm.set_joint_value_target(lift_joint_goal)
    arm.go(wait=True)

    print("Task Complete!")
    moveit_commander.roscpp_shutdown()

if __name__ == '__main__':
    main()