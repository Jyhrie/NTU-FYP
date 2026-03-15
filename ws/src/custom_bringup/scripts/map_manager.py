#!/usr/bin/env python

import rospy
import json
import math
import numpy as np
from enum import Enum
import copy

from std_msgs.msg import String
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker

import tf2_ros
import tf

from dependencies.frontier_detector import FrontierDetector
from dependencies.astar_planner import a_star_exploration
from dependencies.costmap_wall import calc_cost_map

TRUNCATION_SIZE = 5


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
        self.cooldown_duration = rospy.Duration(0.3)
        self.blacklist = []
        self.blacklist_threshold = 5.0
        self.debug = 1

        # --- TF ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # --- Subscribers ---
        rospy.Subscriber("/controller/global", String, self.controller_cb)
        rospy.Subscriber("/map", OccupancyGrid, self.map_cb)
        #rospy.Subscriber("/map/costmap_global", OccupancyGrid, self.global_costmap_cb)

        self.marker_pub = rospy.Publisher('/detected_object_marker', Marker, queue_size=10)
        self.costmap_pub = rospy.Publisher('/costmap/global', OccupancyGrid, queue_size=1)
        # --- Publishers ---
        self.reply_pub = rospy.Publisher("/robot/reply", String, queue_size=1)
        self.path_pub = rospy.Publisher("/robot/path_reply", Path, queue_size=1)
        # self.marker_pub = rospy.Publisher("/detected_frontiers", Marker, queue_size=10)

        rospy.loginfo("Pathing Node Initialized and Ready.")
    

    def publish_costmap(self):
        if self.global_costmap is None:
            rospy.logwarn("Costmap is empty, skipping publish.")
            return

        msg = self.map
        # 2. Prepare the OccupancyGrid message
        out_msg = OccupancyGrid()
        out_msg.header = msg.header
        out_msg.header.stamp = rospy.Time.now()
        out_msg.info = msg.info

        # 3. Normalize to 0-100 range for Rviz visualization
        max_val = np.max(self.global_costmap)
        if max_val > 0:
            # High cost (walls) = 100, Low cost (centers) = 0
            normalized = (self.global_costmap.astype(float) / max_val * 100)
            # OccupancyGrid data must be a list of int8
            out_msg.data = normalized.flatten().astype(np.int8).tolist()
        else:
            out_msg.data = self.global_costmap.flatten().astype(np.int8).tolist()

        self.costmap_pub.publish(out_msg)

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

        # self.global_costmap = calc_cost_map(msg)
        # # 2. Prepare the OccupancyGrid message
        # out_msg = OccupancyGrid()
        # out_msg.header = msg.header
        # out_msg.header.stamp = rospy.Time.now()
        # out_msg.info = msg.info

        # # 3. Normalize to 0-100 range for Rviz visualization
        # max_val = np.max(self.global_costmap)
        # if max_val > 0:
        #     # High cost (walls) = 100, Low cost (centers) = 0
        #     normalized = (self.global_costmap.astype(float) / max_val * 100)
        #     # OccupancyGrid data must be a list of int8
        #     out_msg.data = normalized.flatten().astype(np.int8).tolist()
        # else:
        #     out_msg.data = self.global_costmap.flatten().astype(np.int8).tolist()

        # self.costmap_pub.publish(out_msg)

    # def global_costmap_cb(self, msg):
    #     self.global_costmap = np.array(msg.data).reshape(
    #         (msg.info.height, msg.info.width)
    #     )

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

        self.global_costmap = calc_cost_map(self.map)
        self.publish_costmap()

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

            if len(path) < TRUNCATION_SIZE + 3:
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
                        reply_msg = String()
                        reply_msg.data = json.dumps({
                            "header": "map",
                            "command": "rotate"
                        })
                        self.reply_pub.publish(reply_msg)
                        return
                reply_msg = String()
                reply_msg.data = json.dumps({
                    "header": "map",
                    "command": "path"
                })
                self.reply_pub.publish(reply_msg)
                rospy.loginfo("Valid frontier path found, publishing.")
                self._publish_path(path)
                return

            if path:
                paths.append(path)

        # Fallback to best partial path
        if paths:
            sel_path = self._get_shortest_path(paths)
            if sel_path and len(sel_path) > 3:
                reply_msg = String()
                rospy.loginfo("No complete path; sending best partial.")
                reply_msg.data = json.dumps({
                    "header": "map",
                    "command": "path",
                    "extra": "incomplete"
                })
                self.reply_pub.publish(reply_msg)
                self._publish_path(path)
                return
            
        reply_msg = String()
        reply_msg.data = json.dumps({
            "header": "map",
            "command": "complete"
        })
        self.reply_pub.publish(reply_msg) 

            
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

            # out_msg = OccupancyGrid()
            # out_msg.header = msg.header
            # out_msg.header.stamp = rospy.Time.now()
            # out_msg.info = msg.info

            # # 3. Normalize to 0-100 range for Rviz visualization
            # max_val = np.max(self.global_costmap)
            # if max_val > 0:
            #     # High cost (walls) = 100, Low cost (centers) = 0
            #     normalized = (self.global_costmap.astype(float) / max_val * 100)
            #     # OccupancyGrid data must be a list of int8
            #     out_msg.data = normalized.flatten().astype(np.int8).tolist()
            # else:
            #     out_msg.data = self.global_costmap.flatten().astype(np.int8).tolist()

            # self.costmap_pub.publish(out_msg)

            # 1. Extract object coordinates from message
            obj_x = data.get("x")
            obj_y = data.get("y")
            
            if obj_x is None or obj_y is None:
                rospy.logerr("Object command missing 'x' or 'y'.")
                return

            local_map = copy.deepcopy(self.map)

            # Convert world coordinates of object to grid cells
            obj_cx, obj_cy = self.pose_to_cell(obj_x, obj_y)

            
            local_map = copy.deepcopy(self.map)
            temp_data = list(local_map.data)
            width = local_map.info.width
            height = local_map.info.height

            # 2. Inject a 3x3 block instead of 1 pixel
            # This ensures the dilation logic has a stronger starting point
            thickness = 1 # radius of 1 = 3x3 block
            for dx in range(-thickness, thickness + 1):
                for dy in range(-thickness, thickness + 1):
                    nx, ny = obj_cx + dx, obj_cy + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        idx = ny * width + nx
                        temp_data[idx] = 100 
            
            local_map.data = temp_data
            rospy.loginfo("Injected 3x3 obstacle at: {}, {}".format(obj_cx, obj_cy))

            # 3. Calculate costmap with the thicker obstacle
            self.global_costmap = calc_cost_map(local_map)
            self.publish_costmap()

            # 2. Prepare the OccupancyGrid message
            out_msg = OccupancyGrid()
            out_msg.header = self.map.header
            out_msg.header.stamp = rospy.Time.now()
            out_msg.info = self.map.info

            # 3. Normalize to 0-100 range for Rviz visualization
            max_val = np.max(self.global_costmap)
            if max_val > 0:
                # High cost (walls) = 100, Low cost (centers) = 0
                normalized = (self.global_costmap.astype(float) / max_val * 100)
                # OccupancyGrid data must be a list of int8
                out_msg.data = normalized.flatten().astype(np.int8).tolist()
            else:
                out_msg.data = self.global_costmap.flatten().astype(np.int8).tolist()

            self.costmap_pub.publish(out_msg)


            # Define ring parameters (in cells)
            # Assuming resolution is 0.05m, radius=10 is 0.5m away
            sampling_radius = 10
            sampling_thickness = 4

            candidates = self.get_lowest_cost_in_ring(
                obj_cx, obj_cy, radius=sampling_radius, thickness=sampling_thickness, n_best=1
            )

            if not candidates:
                rospy.logwarn("Pathing Node: No safe spot found around object at ({}, {})".format(obj_x, obj_y))
                return

            # candidates[0] is (cost, gx, gy)
            print()
            _, safe_gx, safe_gy = candidates[0]
            
            # 3. Plan path from robot to the safe spot
            robot_pose = self.get_robot_pose()
            if not robot_pose:
                return
            
            rx, ry, _ = robot_pose
            start_cell = self.pose_to_cell(rx, ry)
            goal_cell = (safe_gx, safe_gy)

            rospy.loginfo("Planning path to safe spot near object: grid({}, {})".format(safe_gx, safe_gy))
            
            wx, wy = self.grid_to_world(safe_gx, safe_gy)
            self.publish_marker(obj_x, obj_y, marker_id=2, color="blue")
            self.publish_marker(rx, ry, marker_id=1, color="red")
            self.publish_marker(wx, wy)
            
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

    def publish_marker(self, x, y, marker_id=0, color="green"):
        print("publishing marker")
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "localization_debug"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.05 
        
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        
        marker.color.a = 1.0
        if color == "green": # Target
            marker.color.g = 1.0
        elif color == "blue": # Robot Position
            marker.color.b = 1.0
        elif color == "red":
            marker.color.r = 1.0
            
        self.marker_pub.publish(marker)

    def _maps_ready(self):
        if self.map is None:
            rospy.logwarn("Pathing Node: Waiting for map")
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
        Finds the safest cells in a ring by cross-referencing the static Map 
        and the Global Costmap.
        """
        if self.global_costmap is None or self.map is None:
            rospy.logwarn("get_lowest_cost_in_ring: Map or Costmap not ready.")
            return []

        # 1. Setup search window
        map_h, map_w = self.global_costmap.shape
        x_min, x_max = max(0, cx - radius), min(map_w - 1, cx + radius)
        y_min, y_max = max(0, cy - radius), min(map_h - 1, cy + radius)

        # 2. Create local grids
        xs = np.arange(x_min, x_max + 1)
        ys = np.arange(y_min, y_max + 1)
        GX, GY = np.meshgrid(xs, ys)

        # 3. Calculate Ring Mask
        dist_sq = (GX - cx)**2 + (GY - cy)**2
        inner_r_sq = max(0, radius - thickness)**2
        outer_r_sq = radius**2
        ring_mask = (dist_sq >= inner_r_sq) & (dist_sq <= outer_r_sq)

        # 4. Extract Costmap and Map data for the window
        # Note: OccupancyGrid data is a flat list, we need to reshape it or index it correctly
        window_costs = self.global_costmap[y_min:y_max+1, x_min:x_max+1]
        
        # Reshape map data to match costmap dimensions for indexing
        map_array = np.array(self.map.data).reshape((self.map.info.height, self.map.info.width))
        window_map = map_array[y_min:y_max+1, x_min:x_max+1]

        # 5. Logical Intersection (The Multi-Layer Check)
        # Map Check: 0 = Free, 100 = Occupied, -1 = Unknown
        map_free = (window_map == 0) 
        
        # Costmap Check: Avoid Lethal (100) and Inscribed (99)
        cost_safe = (window_costs >= 0) & (window_costs < 99)

        # Combine all constraints
        final_mask = ring_mask & map_free & cost_safe

        if not np.any(final_mask):
            rospy.logwarn("No cells found that are FREE in the map AND safe in the costmap.")
            return []

        # 6. Extract and Sort
        valid_x = GX[final_mask]
        valid_y = GY[final_mask]
        valid_costs = window_costs[final_mask]

        # Return as (x, y, cost) sorted by cost
        results = sorted(zip(valid_costs, valid_x, valid_y), key=lambda x: x[0])
        
        return results[:n_best]

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

    # def _publish_frontier_path(self, grid_path):
    #     """Publishes a frontier path with its reply header."""
    #     self.path_pub.publish(self._build_path_msg(grid_path))

    # def _publish_frontier_reply(self, data_value):
    #     msg = String()
    #     msg.data = json.dumps({"header": "frontier", "data": data_value})
    #     self.reply_pub.publish(msg)

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
