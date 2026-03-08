#!/usr/bin/env python

import rospy
import json
import math
import numpy as np
from enum import Enum

from std_msgs.msg import String
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker

import tf2_ros
import tf

from dependencies.frontier_detector import FrontierDetector
from dependencies.astar_planner import a_star_exploration

TRUNCATION_SIZE = 7


class PathingNode:

    def __init__(self):
        rospy.init_node("pathing_node")

        # --- State & Data ---
        self.map = None
        self.global_costmap = None
        self.home_pose = (0.0, 0.0)

        # --- Frontier State ---
        self.detector = None
        self.is_active = False
        self.last_trigger_time = rospy.Time(0)
        self.cooldown_duration = rospy.Duration(2.5)
        self.blacklist = []
        self.blacklist_threshold = 5.0
        self.debug = 1

        # --- TF ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # --- Subscribers ---
        rospy.Subscriber("/controller/global", String, self.controller_cb)
        rospy.Subscriber("/map", OccupancyGrid, self.map_cb)
        rospy.Subscriber("/map/costmap_global", OccupancyGrid, self.global_costmap_cb)

        # --- Publishers ---
        self.reply_pub = rospy.Publisher("/robot/reply", String, queue_size=1)
        self.path_pub = rospy.Publisher("/robot/path_reply", Path, queue_size=1)
        self.marker_pub = rospy.Publisher("/detected_frontiers", Marker, queue_size=10)

        rospy.loginfo("Pathing Node Initialized and Ready.")

    # -------------------------------------------------------------------------
    # Map Callbacks
    # -------------------------------------------------------------------------

    def map_cb(self, msg):
        self.map = msg
        if self.detector is None:
            self.detector = FrontierDetector(
                map_width=msg.info.width,
                map_height=msg.info.height,
                resolution=msg.info.resolution,
                origin_x=msg.info.origin.position.x,
                origin_y=msg.info.origin.position.y,
            )

    def global_costmap_cb(self, msg):
        self.global_costmap = np.array(msg.data).reshape(
            (msg.info.height, msg.info.width)
        )

    def controller_cb(self, msg):
        try:
            data = json.loads(msg.data)
        except ValueError:
            rospy.logwarn("Pathing Node: Received non-JSON message, ignoring.")
            return

        header = data.get("header", "")

        if header == "pathing":
            command = data.get("command", "")
            if command == "waypoint":
                self._handle_waypoint(data)
            elif command == "frontier":
                self._handle_frontier()
            elif command == "object":
                self._handle_object(data)
            else:
                rospy.logwarn("Pathing Node: Unknown command type '{}'".format(command))

    # -------------------------------------------------------------------------
    # Waypoint Handling
    # -------------------------------------------------------------------------

    def _handle_waypoint(self, data):
        if not self._maps_ready():
            return

        target_x = data.get("x")
        target_y = data.get("y")
        if target_x is None or target_y is None:
            rospy.logerr("Waypoint command missing 'x' or 'y'.")
            return

        self._plan_and_publish_waypoint(target_x, target_y)

    def _plan_and_publish_waypoint(self, tx, ty):
        robot_pose = self.get_robot_pose()
        if not robot_pose:
            return
        rx, ry, _ = robot_pose

        start_cell = self.pose_to_cell(rx, ry)
        goal_cell = self.pose_to_cell(tx, ty)

        path, success = a_star_exploration(
            self.map.data, self.global_costmap, start_cell, goal_cell
        )

        if path and len(path) > 0:
            self._publish_path(path)
            self.reply_pub.publish(json.dumps({"cmd": "path"}))
        else:
            rospy.logerr("Pathing Node: Waypoint pathfinding failed.")

    # -------------------------------------------------------------------------
    # Frontier Handling
    # -------------------------------------------------------------------------

    def _handle_frontier(self):
        current_time = rospy.Time.now()
        if (
            self.is_active
            and (current_time - self.last_trigger_time) < self.cooldown_duration
        ):
            rospy.loginfo("Frontier trigger ignored: cooldown in progress.")
            return

        if not self._maps_ready():
            return

        rospy.loginfo("Frontier request received, detecting frontiers...")
        self.is_active = True
        self.last_trigger_time = current_time
        self._run_frontier()

    def _run_frontier(self):
        pose = self.get_robot_pose()
        if pose is None:
            rospy.logwarn("No TF pose available for frontier exploration.")
            return

        x, y, yaw = pose
        start = self.pose_to_cell(x, y)
        sx, sy = start

        frontiers = self.detector.get_frontiers(sx, sy, self.map.data)
        if self.debug:
            self._publish_frontier_markers(frontiers)

        paths = []

        for frontier in frontiers:
            path, success = a_star_exploration(
                self.map.data, self.global_costmap, start, frontier
            )

            if len(path) < TRUNCATION_SIZE + 5:
                rospy.loginfo(
                    "Discarding short path to frontier {}: len={}".format(
                        frontier, len(path)
                    )
                )
                continue

            path = path[:-TRUNCATION_SIZE]

            if success and path:
                if len(path) >= 3:
                    lookahead = min(5, len(path) - 1)
                    first_dx = path[lookahead][0] - path[0][0]
                    first_dy = path[lookahead][1] - path[0][1]
                    first_step_angle = math.atan2(first_dy, first_dx)
                    angle_diff = math.atan2(
                        math.sin(first_step_angle - yaw),
                        math.cos(first_step_angle - yaw),
                    )
                    rospy.loginfo(
                        "Path angle: {}deg | Yaw: {}deg | Diff: {}deg".format(
                            round(math.degrees(first_step_angle), 1),
                            round(math.degrees(yaw), 1),
                            round(math.degrees(angle_diff), 1),
                        )
                    )
                    if abs(angle_diff) > math.radians(90):
                        rospy.loginfo(
                            "Large angle diff ({}deg), rotating first.".format(
                                round(math.degrees(angle_diff), 1)
                            )
                        )
                        self._publish_frontier_reply("rotate")
                        return

                rospy.loginfo("Valid frontier path found, publishing.")
                self._publish_frontier_path(path)
                return

            if path:
                paths.append(path)

        # Fallback to best partial path
        if paths:
            sel_path = self._get_shortest_path(paths)
            if sel_path and len(sel_path) > 3:
                rospy.loginfo("No complete path; sending best partial.")
                self._publish_frontier_path(sel_path)
                return

        rospy.logwarn("No valid frontier path found.")

    # -------------------------------------------------------------------------
    # Object Handling
    # -------------------------------------------------------------------------

    def _handle_object(self, data):
            """
            Calculates a path to the safest spot (lowest costmap value) 
            in a ring around a detected object.
            """
            if not self._maps_ready():
                return

            # 1. Extract object coordinates from message
            obj_x = data.get("x")
            obj_y = data.get("y")
            
            if obj_x is None or obj_y is None:
                rospy.logerr("Object command missing 'x' or 'y'.")
                return

            # 2. Find the safest spot near the object
            # Convert world coordinates of object to grid cells
            obj_cx, obj_cy = self.pose_to_cell(obj_x, obj_y)

            # Define ring parameters (in cells)
            # Assuming resolution is 0.05m, radius=10 is 0.5m away
            sampling_radius = 18
            sampling_thickness = 4

            candidates = self.get_lowest_cost_in_ring(
                obj_cx, obj_cy, radius=sampling_radius, thickness=sampling_thickness, n_best=1
            )

            if not candidates:
                rospy.logwarn("Pathing Node: No safe spot found around object at ({}, {})".format(obj_x, obj_y))
                return

            # candidates[0] is (cost, gx, gy)
            _, safe_gx, safe_gy = candidates[0]
            
            # 3. Plan path from robot to the safe spot
            robot_pose = self.get_robot_pose()
            if not robot_pose:
                return
            
            rx, ry, _ = robot_pose
            start_cell = self.pose_to_cell(rx, ry)
            goal_cell = (safe_gx, safe_gy)

            rospy.loginfo("Planning path to safe spot near object: grid({}, {})".format(safe_gx, safe_gy))
            
            path, success = a_star_exploration(
                self.map.data, self.global_costmap, start_cell, goal_cell
            )

            # 4. Publish results
            if success and path:
                # Reusing your waypoint publishing logic
                self._publish_path(path)
                # Notify the controller that an 'object' path is ready
                self.reply_pub.publish(json.dumps({"header": "object", "data": "path_ready"}))
            else:
                rospy.logerr("Pathing Node: Failed to find A* path to the safe spot.")

    # -------------------------------------------------------------------------
    # Shared Helpers
    # -------------------------------------------------------------------------

    def _maps_ready(self):
        if self.map is None or self.global_costmap is None:
            rospy.logwarn("Pathing Node: Waiting for map/costmap...")
            return False
        return True

    def get_robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                "map", "base_link", rospy.Time(0), rospy.Duration(0.1)
            )
            q = t.transform.rotation
            (_, _, yaw) = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
            x = t.transform.translation.x
            y = t.transform.translation.y
            rospy.logdebug("Robot Pose: x={} y={} yaw={}".format(x, y, yaw))
            return x, y, yaw
        except Exception as e:
            rospy.logwarn("TF lookup failed: {}".format(e))
            return None

    def pose_to_cell(self, x, y):
        """Convert world (meters) to grid (cell indices), clamped to map bounds."""
        res = self.map.info.resolution
        origin_x = self.map.info.origin.position.x
        origin_y = self.map.info.origin.position.y

        grid_x = int((x - origin_x) / res)
        grid_y = int((y - origin_y) / res)

        grid_x = max(0, min(grid_x, self.map.info.width - 1))
        grid_y = max(0, min(grid_y, self.map.info.height - 1))

        return grid_x, grid_y

    def grid_to_world(self, gx, gy):
        """Convert grid (cell indices) to world (meters)."""
        res = self.map.info.resolution
        wx = gx * res + self.map.info.origin.position.x
        wy = gy * res + self.map.info.origin.position.y
        return wx, wy

    def get_lowest_cost_in_ring(self, cx, cy, radius, thickness, n_best=1):
        """
        Samples the costmap in a ring around a centre cell (cx, cy) and
        returns the lowest-cost reachable cell(s).

        Args:
            cx, cy    : Centre in grid cells (can be world-derived via pose_to_cell).
            radius    : Outer radius of the ring in cells.
            thickness : Radial width of the ring in cells (inner radius = radius - thickness).
            n_best    : Number of lowest-cost candidates to return (default 1).

        Returns:
            List of (cost, gx, gy) tuples sorted ascending by cost, length <= n_best.
            Returns an empty list if the costmap isn't ready or no valid cells exist.
        """
        if self.global_costmap is None or self.map is None:
            rospy.logwarn("get_lowest_cost_in_ring: costmap not ready.")
            return []

        map_h, map_w = self.global_costmap.shape
        inner_radius = max(0, radius - thickness)

        # Bounding box to avoid iterating the whole map
        x_min = max(0, cx - radius)
        x_max = min(map_w - 1, cx + radius)
        y_min = max(0, cy - radius)
        y_max = min(map_h - 1, cy + radius)

        # Build coordinate grids for the bounding box
        gx = np.arange(x_min, x_max + 1)
        gy = np.arange(y_min, y_max + 1)
        GX, GY = np.meshgrid(gx, gy)  # shape: (rows, cols)

        dist_sq = (GX - cx) ** 2 + (GY - cy) ** 2
        inner_sq = inner_radius**2
        outer_sq = radius**2

        # Ring mask: within outer circle, outside inner circle
        ring_mask = (dist_sq <= outer_sq) & (dist_sq >= inner_sq)

        costs = self.global_costmap[y_min : y_max + 1, x_min : x_max + 1]

        # Exclude unknown (-1) and lethal (>=100) cells
        valid_mask = ring_mask & (costs >= 0) & (costs < 100)

        if not np.any(valid_mask):
            rospy.logwarn(
                "get_lowest_cost_in_ring: no valid cells in ring (r={}, t={}).".format(
                    radius, thickness
                )
            )
            return []

        valid_costs = costs[valid_mask]
        valid_GX = GX[valid_mask]
        valid_GY = GY[valid_mask]

        # Partial sort only need the n_best lowest
        k = min(n_best, len(valid_costs))
        idx = np.argpartition(valid_costs, k - 1)[:k]
        idx_sorted = idx[np.argsort(valid_costs[idx])]  # sort that small slice

        results = [
            (int(valid_costs[i]), int(valid_GX[i]), int(valid_GY[i]))
            for i in idx_sorted
        ]

        rospy.logdebug(
            "Ring sample (cx={}, cy={}, r={}, t={}): {}".format(
                cx, cy, radius, thickness, results
            )
        )
        return results

    def _get_shortest_path(self, paths):
        best_path, min_length = None, float("inf")
        for path in paths:
            length = sum(
                math.sqrt(
                    (path[i][0] - path[i + 1][0]) ** 2
                    + (path[i][1] - path[i + 1][1]) ** 2
                )
                for i in range(len(path) - 1)
            )
            if length < min_length:
                min_length = length
                best_path = path
        return best_path

    # -------------------------------------------------------------------------
    # Publishing
    # -------------------------------------------------------------------------

    def _build_path_msg(self, grid_path):
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
        return path_msg

    def _publish_path(self, grid_path):
        """Used by waypoint handler."""
        self.path_pub.publish(self._build_path_msg(grid_path))

    def _publish_frontier_path(self, grid_path):
        """Publishes a frontier path with its reply header."""
        self._publish_frontier_reply("path")
        self.path_pub.publish(self._build_path_msg(grid_path))

    def _publish_frontier_reply(self, data_value):
        msg = String()
        msg.data = json.dumps({"header": "frontier", "data": data_value})
        self.reply_pub.publish(msg)

    def _publish_frontier_markers(self, frontiers):
        if not frontiers:
            return

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "frontiers"
        marker.id = 0
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.scale.x = marker.scale.y = marker.scale.z = 0.1
        marker.color.r = 0.0
        marker.color.g = 0.5
        marker.color.b = 1.0
        marker.color.a = 1.0

        res = self.map.info.resolution
        origin_x = self.map.info.origin.position.x
        origin_y = self.map.info.origin.position.y

        for f in frontiers:
            p = Point()
            p.x = (f[0] * res) + origin_x + (res / 2.0)
            p.y = (f[1] * res) + origin_y + (res / 2.0)
            p.z = 0.2
            marker.points.append(p)

        self.marker_pub.publish(marker)


if __name__ == "__main__":
    node = PathingNode()
    rospy.spin()
