# Credentials
```bash
Login ID: Jetson@IP
Password: yahboom
```

# Zombie Master Issue
Assuming you are using the given usb likely that ROS is initailized on launch, and u need to kill the roscore so that the nodes can talk to each other. 

# Killing Zombie Master
```bash
killall -9 roscore rosmaster python python3 roslaunch
```

# Installation Guide
Pull the entire directory into both your jetson nano, as well as ur high-performance pc.
Then in the jetson nano, navigate to the workspace, and initialize the ros environment such that it identifies custom_bringup as a ROS workspace.

copy the contents of the NTU_FYP into a folder called ~/fyp.

then, 
```bash
cd ~/fyp/ws
catkin make
source devel/setup.bash
```
Refer to commands.md for all relevant files.

# Setup Guide
launch the setup launch file. then run pc_nodes/setup_pc.py, change the IP in setup.py to your pc's ip.
For all files, env params should just belong in the file itself, no .env file required.

You will need to download the SAM1 model.

To download the SAM1 model, run this in /pc_nodes in powershell, in the computer.
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```
The file will automatically be transferred to fyp/ws/scripts/models in the nano via scp

then run the trtexec command (located in commands.md) to convert from onnx to engine.

# Runtime Guide
In runtime guide, launch the startup sequence, then launch the custom nodes. Upon robot_controller.py finish initializing, the robot will run.

Refer to the report 

# Foxglove Studio
https://app.foxglove.dev/
launch this on ur pc for visualization. much more lightweight than vncing and using rviz.
you can also launch a vm with ros, and hook view from there, but i find that foxglove is easier

# Relevant Resources
https://www.yahboom.net/study/Transbot-jetson_nano
https://wiki.purduesigbots.com/software/control-algorithms/basic-pure-pursuit
https://www.youtube.com/watch?v=xqjVTE7QvOg

# Summary
For the ros launch files, you only need to care about
master.launch
setup.launch
custom_bringup.launch
gmapping.launch

all other files are irrelevant archived files

For python scripts, everything in ws/scripts is used.

Frontier Node is for frontier search
Map Manager is for costmap generation & pathfinding
Movement Controller is the motion control for the robot
Node Arm Control controls the arm
Node Computer Vision runs the YOLO model
Node Depth is for the Depth Camera

Robot Contoller is the FSM




