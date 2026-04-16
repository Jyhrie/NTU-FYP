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

# Setup Guide
launch the setup launch file. then run pc_nodes/setup_pc.py, change the IP in setup.py to your pc's ip.

# Runtime Guide
In runtime guide, launch the startup sequence, then launch the custom nodes. Upon robot_controller.py finish initializing, the robot will run.

# Foxglove Studio
https://app.foxglove.dev/

launch this on ur pc for visualization. much more lightweight than vncing and using rviz.

