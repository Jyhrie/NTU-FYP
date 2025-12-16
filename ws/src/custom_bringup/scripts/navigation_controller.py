#!/usr/bin/env python3
import rospy
import numpy as np
import math
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

import local_occupancy_movement as lom

MAX_MOVEMENT_SPEED = 0.25
MAX_ANGULAR_SPEED = 0.35

class NavigationController:
    def __init__(self):
        pass

    

    def update_global_costmap(self):
        """
        updates the global costmap from the /map topic
        """
        pass

    def check_against_global_map(self):
        """
        checks against the global costmap to see if robot is turning into a spot where it has been before,
        """
        pass


if __name__ == "__main__":
    nav = NavigationController()
    nav.run_once()
