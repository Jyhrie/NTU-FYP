#!/usr/bin/env python

from enum import Enum
import json
import rospy
from std_msgs.msg import Empty, String
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import math

import tf2_ros
import tf
import numpy as np
from nav_msgs.msg import OccupancyGrid
from dependencies.frontier_detector import FrontierDetector
from dependencies.astar_planner import a_star_exploration
from dependencies.astar import PathPlanner


class FrontierNode:

    def __init__(self):

        rospy.init_node("frontier_node")

        # vars
        self.map = None
        self.global_costmap = None

        # classes
        self.detector = None

        # in
        self.global_request_topic = rospy.Subscriber(
            "/controller/global", String, self.controller_cb
        )

        self.map_topic = rospy.Subscriber("/map", OccupancyGrid, self.map_cb)
        self.costmap_global_topic = rospy.Subscriber(
            "/move_base/global_costmap/costmap", OccupancyGrid, self.global_costmap_cb
        )
        self.marker_pub = rospy.Publisher("/detected_frontiers", Marker, queue_size=10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # out
        self.global_reply_pub = rospy.Publisher("/robot/reply", String, queue_size=1)
        self.global_path_pub = rospy.Publisher("/robot/path_reply", Path, queue_size=1)

        # self.frontier_node_pub = rospy.Publisher(
        #     "/frontier_node_reply", String, queue_size=1
        # )
        # self.frontier_node_path_pub = rospy.Publisher(
        #     "/frontier_node_path", Path, queue_size=1
        # )

        self.is_active = False
        self.last_trigger_time = rospy.Time(0)
        self.cooldown_duration = rospy.Duration(2.5)  # 2 seconds

        self.blacklist = []  # List of (cx, cy) tuples
        self.blacklist_threshold = 5.0 # Distance in pixels to consider a match

        self.debug = 1
        print("Initialization Complete, Node is Ready!")

    def controller_cb(self, msg):
        current_time = rospy.Time.now()

        if msg.data == "request_frontier":
            if (
                self.is_active
                and (current_time - self.last_trigger_time) < self.cooldown_duration
            ):
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
                origin_y=msg.info.origin.position.y,
            )
        pass

    def global_costmap_cb(self, msg):
        self.global_costmap = msg.data
        pass

    def trigger(self):
        if self.map == None:
            print("Early Return due to No Map")
            return

        pose = self.get_robot_pose()
        if pose is None:
            rospy.logwarn("No TF pose available")
            return

        x, y, yaw = pose

        start = self.pose_to_cell(x, y, self.map)
        x_start, y_start = start
        frontiers = self.detector.get_frontiers(x_start, y_start, self.map.data)
        if self.debug:
            self.publish_frontier_markers(frontiers)

        paths = []

        for frontier in frontiers:
            path, success = PathPlanner.astar(self.map.data, self.global_costmap, start, frontier)
        for frontier in frontiers:
            path, success = a_star_exploration(
                self.map.data, self.global_costmap, start, frontier
            )

            if success and path:
                if len(path) >= 2:
                    # Look ahead several steps for a more stable angle estimate
                    lookahead = min(5, len(path) - 1)
                    first_dx = path[lookahead][0] - path[0][0]
                    first_dy = path[lookahead][1] - path[0][1]
                    first_step_angle = math.atan2(first_dy, first_dx)
                    angle_diff = math.atan2(
                        math.sin(first_step_angle - yaw),
                        math.cos(first_step_angle - yaw)
                    )
                    print("Path initial angle: {}deg, robot yaw: {}deg, diff: {}deg".format(
                        round(math.degrees(first_step_angle), 1),
                        round(math.degrees(yaw), 1),
                        round(math.degrees(angle_diff), 1)
                    ))
                    if abs(angle_diff) > math.radians(90):
                        print("First move requires {}deg turn, rotating first.".format(
                            round(math.degrees(angle_diff), 1)))
                        self.publish_rotate_command()
                        return

                print("Found a valid path, publishing.")
                self.publish_visual_path(path)
                return

            if path:
                print("appending path")
                paths.append(path)

        # No successful path, fall back to best partial
        print(paths)
        if len(paths) > 0:
            sel_path = self.get_shortest_path(paths)
            if len(sel_path) > 3:
                print("No complete path, sending closest attempt.")
                self.publish_visual_path(sel_path)
                return

        print("No Valid Path Detected.")

    def get_robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                "map", "base_link", rospy.Time(0), rospy.Duration(0.1)
            )
            x = t.transform.translation.x
            y = t.transform.translation.y

            q = t.transform.rotation
            (_, _, yaw) = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
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
        min_length = float("inf")

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
            p2 = path[i + 1]
            # Distance between consecutive waypoints
            length += math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        return length

    def grid_to_world(self, grid_point):
        """Converts (x, y) grid indices to (x, y) meters"""
        info = self.map.info
        wx = grid_point[0] * info.resolution + info.origin.position.x
        wy = grid_point[1] * info.resolution + info.origin.position.y
        return (wx, wy)

    def publish_visual_path(self, grid_path):
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = rospy.Time.now()

        data = {
            "header": 'frontier',
            "data": "path"
        }
        cmd_msg = String()
        cmd_msg.data = json.dumps(data)

        for grid_point in grid_path:
            # Convert each grid point back to meters (world coords)
            world_x, world_y = self.grid_to_world(grid_point)

            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = world_x
            pose.pose.position.y = world_y
            pose.pose.orientation.w = 1.0  # Default orientation
            path_msg.poses.append(pose)

        self.global_reply_pub.publish(cmd_msg)
        self.global_path_pub.publish(path_msg)

    def publish_rotate_command(self):
        data = {
            "header": 'frontier',
            "data": "rotate"
        }
        cmd_msg = String()
        cmd_msg.data = json.dumps(data)

        self.global_reply_pub.publish(cmd_msg)


    def publish_frontier_markers(self, frontiers):
        if not frontiers:
            return

        marker = Marker()
        # 1. Basic Header Info
        marker.header.frame_id = "map"  # Must match your global frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "frontiers"
        marker.id = 0
        marker.type = Marker.SPHERE_LIST  # Efficiently renders many points
        marker.action = Marker.ADD

        # 2. Set the Size (0.1m spheres)
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1

        # 3. Set the Color (Electric Blue for high visibility)
        marker.color.r = 0.0
        marker.color.g = 0.5
        marker.color.b = 1.0
        marker.color.a = 1.0  # Alpha (transparency)

        # 4. Map Metadata for coordinate conversion
        res = self.map.info.resolution
        origin_x = self.map.info.origin.position.x
        origin_y = self.map.info.origin.position.y

        # 5. Populate Points
        for f in frontiers:
            p = Point()
            # Convert grid index to world coordinates
            # We add half a resolution to center the point in the cell
            p.x = (f[0] * res) + origin_x + (res / 2.0)
            p.y = (f[1] * res) + origin_y + (res / 2.0)
            p.z = 0.2  # Lift slightly above the map to prevent "z-fighting"
            marker.points.append(p)

        self.marker_pub.publish(marker)


if __name__ == "__main__":
    node = FrontierNode()
    rospy.spin()
