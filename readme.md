# Credentials

# Login ID: Jetson@IP
# Password: yahboom

# Killing Zombie Master
killall -9 roscore rosmaster python python3 roslaunch

# Tell ROS to use .18 network (my home network), then start the master

export ROS_IP=192.168.18.86
roscore

