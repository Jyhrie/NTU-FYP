
# SETUP
roslaunch custom_bringup setup.launch

# ONNX -> .engine
/usr/src/tensorrt/bin/trtexec --onnx=best.onnx --saveEngine=best.engine --fp16 --exportTimes=timing.json

/usr/src/tensorrt/bin/trtexec \
  --onnx=best.onnx \
  --saveEngine=best.engine \
  --workspace=2048 \
  --explicitBatch \
  --avgTiming=1 \
  --minTiming=1 \
  --buildOnly \
  --fp16

# Startup Sequence
roslaunch custom_bringup custom_bringup.launch
roslaunch rosbridge_server rosbridge_websocket.launch
roslaunch custom_bringup gmapping.launch
# OR
roslaunch custom_bringup master.launch

# Custom Nodes
rosrun custom_bringup movement_controller.py
rosrun custom_bringup map_manager.py
rosrun custom_bringup node_computer_vision.py
rosrun custom_bringup node_depth.py
rosrun custom_bringup node_arm_control.py

rosrun custom_bringup robot_controller.py  <-- Wait for other nodes to initalize, then launch this

# RESET THE ARM
cd fyp
python arm_test.py

# Teleoperation
rosrun teleop_twist_keyboard teleop_twist_keyboard.py 

