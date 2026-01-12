import sys
import moveit_commander
import geometry_msgs.msg

# Initialize MoveIt
moveit_commander.roscpp_initialize(sys.argv)
arm = moveit_commander.MoveGroupCommander("arm")
gripper = moveit_commander.MoveGroupCommander("hand")

# 1. Move to a "Pre-grasp" position above the object
pose_target = geometry_msgs.msg.Pose()
pose_target.position.x = 0.5
pose_target.position.y = 0.0
pose_target.position.z = 0.3  # Stay slightly above
arm.set_pose_target(pose_target)
arm.go(wait=True)

# 2. Open Gripper
gripper.set_named_target("open")
gripper.go(wait=True)

# 3. Descend and Grasp
pose_target.position.z = 0.2 # Lower to object level
arm.set_pose_target(pose_target)
arm.go(wait=True)
gripper.set_named_target("closed")
gripper.go(wait=True)

# 4. Lift and Move to Placement
pose_target.position.z = 0.4
arm.set_pose_target(pose_target)
arm.go(wait=True)