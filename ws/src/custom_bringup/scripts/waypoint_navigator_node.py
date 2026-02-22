#!/usr/bin/env python

import rospy
import json
import math
from std_msgs.msg import String
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped
import tf2_ros
import tf
from dependencies.astar_planner import a_star_exploration

class WaypointNavigatorNode:

    def __init__(self):
        rospy.init_node("waypoint_navigator_node")

        # --- State & Data ---
        self.map = None
        self.global_costmap = None
        self.home_pose = (0.0, 0.0)  # Default home coordinates

        # --- Subscribers ---
        self.global_request_sub = rospy.Subscriber(
            "/controller/global", String, self.controller_cb
        )
        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.map_cb)
        self.costmap_sub = rospy.Subscriber(
            "/map/costmap_global", OccupancyGrid, self.global_costmap_cb
        )

        # --- TF Buffer for Localization ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # --- Publishers ---
        self.global_reply_pub = rospy.Publisher("/robot/reply", String, queue_size=1)
        self.global_path_pub = rospy.Publisher("/robot/path_reply", Path, queue_size=1)

        rospy.loginfo("Waypoint Navigator Node Initialized")

    def map_cb(self, msg):
        self.map = msg

    def global_costmap_cb(self, msg):
        self.global_costmap = msg.data

    def controller_cb(self, msg):
        if self.map is None or self.global_costmap is None:
            rospy.logwarn("Navigator: Waiting for map/costmap...")
            return

        target_x, target_y = None, None

        # 1. Handle "Home" Request
        if msg.data == "request_home":
            target_x, target_y = self.home_pose

        # 2. Handle JSON Waypoint Request
        else:
            try:
                data = json.loads(msg.data)
                if data.get("cmd") == "request_waypoint":
                    target_x = data["x"]
                    target_y = data["y"]
                    rospy.loginfo(f"📍 Navigating to Waypoint: ({target_x}, {target_y})")
            except ValueError:
                return # Not a JSON command for this node

        if target_x is not None:
            self.plan_and_publish(target_x, target_y)

    def plan_and_publish(self, tx, ty):
        # 1. Get current robot position
        robot_pose = self.get_robot_pose()
        if not robot_pose:
            return
        rx, ry, _ = robot_pose

        # 2. Convert World (m) to Grid (cells)
        start_cell = self.pose_to_cell(rx, ry)
        goal_cell = self.pose_to_cell(tx, ty)

        # 3. Use A* to find path (Note: goal must be wrapped in a list for your planner)
        # Assuming a_star_exploration(map_data, costmap, start, goal_list)
        path, success = a_star_exploration(
            self.map.data, self.global_costmap, start_cell, [goal_cell]
        )

        if path and len(path) > 0:
            self.publish_path(path)
            # Notify controller that we found a path
            self.reply_pub.publish(json.dumps({"cmd": "path"}))
        else:
            rospy.logerr("Navigator: Pathfinding failed!")

    def get_robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform("map", "base_link", rospy.Time(0), rospy.Duration(0.1))
            q = t.transform.rotation
            (_, _, yaw) = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
            return t.transform.translation.x, t.transform.translation.y, yaw
        except:
            return None

    def pose_to_cell(self, x, y):
        res = self.map.info.resolution
        origin_x = self.map.info.origin.position.x
        origin_y = self.map.info.origin.position.y
        grid_x = int((x - origin_x) / res)
        grid_y = int((y - origin_y) / res)
        return (grid_x, grid_y)

    def grid_to_world(self, gx, gy):
        res = self.map.info.resolution
        wx = gx * res + self.map.info.origin.position.x
        wy = gy * res + self.map.info.origin.position.y
        return wx, wy

    def publish_path(self, grid_path):
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = rospy.Time.now()

        for gx, gy in grid_path:
            wx, wy = self.grid_to_world(gx, gy)
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)

if __name__ == "__main__":
    node = WaypointNavigatorNode()
    rospy.spin()