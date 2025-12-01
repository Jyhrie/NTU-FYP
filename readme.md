# Credentials
```bash
Login ID: Jetson@IP
Password: yahboom
```

# Killing Zombie Master
```bash
killall -9 roscore rosmaster python python3 roslaunch
```

# Tell ROS to use .18 network (my home network), then start the master
```bash
export ROS_IP=192.168.18.86
roscore
```

