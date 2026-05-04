#!/bin/bash

echo "Starting ROS & Mapping..."
roslaunch custom_bringup master.launch &
sleep 3

echo "Starting Control Nodes..."
rosrun custom_bringup movement_controller.py &
rosrun custom_bringup map_manager.py &
rosrun custom_bringup node_computer_vision.py &
rosrun custom_bringup node_depth.py &
rosrun custom_bringup node_arm_control.py & 
sleep 25

echo "Starting Controller..."
rosrun custom_bringup robot_controller.py


