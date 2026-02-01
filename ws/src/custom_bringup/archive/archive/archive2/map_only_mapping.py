#!/usr/bin/env python3

import rospy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
import tf
import math

class MapOnlyNavigator:
    def __init__(self):
        rospy.init_node("map_only_mapping")

        # Subscribers
        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        self.tf_listener = tf.TransformListener()

        # Publisher
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        # Robot pose
        self.x = 0
        self.y = 0
        self.yaw = 0

        # Map
        self.map = None

        # Target on map (example: center)
        self.x_goal = 2.0
        self.y_goal = 2.0

        # Control gains
        self.Kp_ang = 1.0
        self.Kp_lin = 0.2
        self.max_ang_speed = 0.3
        self.max_lin_speed = 0.2

        rospy.Timer(rospy.Duration(0.1), self.tick)  # 10 Hz

    def map_callback(self, msg):
        self.map = msg  # store latest map