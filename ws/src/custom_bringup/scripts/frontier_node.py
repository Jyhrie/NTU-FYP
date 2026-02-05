#!/usr/bin/env python

from enum import Enum
import rospy
from std_msgs.msg import Empty, String
from nav_msgs.msg import Path
import math

import tf2_ros
import tf
import numpy as np
from nav_msgs.msg import OccupancyGrid
from dependencies.frontier_detector import FrontierDetector
from dependencies.astar_planner import a_star_exploration

class FrontierNode:

    def __init__(self):
        
        rospy.init_node("frontier_node")

        #vars
        self.map = None
        self.global_costmap = None

        #classes
        self.detector = None

        #in
        self.global_request_topic = rospy.Subscriber("/controller/global", String, self.controller_cb)

        self.map_topic = rospy.Subscriber("/map", OccupancyGrid, self.map_cb)
        self.costmap_global_topic = rospy.Subscriber("/move_base/global_costmap/costmap", OccupancyGrid, self.global_costmap_cb)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        #out
        self.frontier_node_pub = rospy.Publisher("/frontier_node_message", String, queue_size=1)

        self.is_active = False
        self.last_trigger_time = rospy.Time(0)
        self.cooldown_duration = rospy.Duration(2.5) # 2 seconds
        print("Initialization Complete, Node is Ready!")


    def controller_cb(self, msg):
        current_time = rospy.Time.now()

        if msg.data == "request_frontier":
            if self.is_active and (current_time - self.last_trigger_time) < self.cooldown_duration:
                rospy.loginfo("Trigger ignored: Cooldown in progress.")
                return

            # If we passed the check, trigger the logic
            print("Request Received! Fetching Frontiers")
            self.is_active = True
            self.last_trigger_time = current_time
            self.trigger()
    
    def map_cb(self, msg):
        print("Map Instance Received.")
        self.map = msg
        if self.detector is None:
            self.detector = FrontierDetector(
                map_width=msg.info.width,
                map_height=msg.info.height,
                resolution=msg.info.resolution,
                origin_x=msg.info.origin.position.x,
                origin_y=msg.info.origin.position.y
            )
        pass

    def global_costmap_cb(self, msg):
        self.global_costmap = msg.data
        pass

    def trigger(self):
        if self.map == None:
            print("Early Return due to No Map")
            return
        
        x,y,_ = self.get_robot_pose()
        start = self.pose_to_cell(x,y, self.map)
        x_start, y_start = start
        frontiers = self.detector.get_frontiers(x_start, y_start, self.map.data)
        print(frontiers)
        paths = []
        if frontiers:
            for frontier in frontiers:
                path = a_star_exploration(self.map.data, self.global_costmap, start, frontier)
                paths.append(path)

            sel_path = self.get_shortest_path(paths)
            #print(sel_path)
            print("Found a Selected Path")
            self.frontier_node_pub.publish(sel_path)
        return

    def get_robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                "map", "base_link", rospy.Time(0), rospy.Duration(0.1)
            )
            x = t.transform.translation.x
            y = t.transform.translation.y

            q = t.transform.rotation
            (_, _, yaw) = tf.transformations.euler_from_quaternion(
                [q.x, q.y, q.z, q.w]
            )
            print("Robot Pose: ", x, y, yaw)
            return x, y, yaw

        except:
            return None
        
    def pose_to_cell(self, x, y, map):
        origin_x = map.info.origin.position.x
        origin_y = map.info.origin.position.y
        resolution = map.info.resolution

        grid_x = int((x - origin_x) / resolution)
        grid_y = int((y - origin_y) / resolution)

        width = map.info.width
        height = map.info.height

        grid_x = max(0, min(grid_x, width - 1))
        grid_y = max(0, min(grid_y, height - 1))

        return grid_x, grid_y

    def get_shortest_path(self, paths):
        if not paths:
            return None

        best_path = None
        min_length = float('inf')

        for path in paths:
            # Calculate the actual cumulative length of this specific path
            current_length = self.calculate_path_length(path)
            
            if current_length < min_length:
                min_length = current_length
                best_path = path

        return best_path

    def calculate_path_length(self, path):
        length = 0.0
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i+1]
            # Distance between consecutive waypoints
            length += math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        return length


if __name__ == "__main__":
    node = FrontierNode()
    rospy.spin()