#!/usr/bin/env python

import rospy
import tf
import numpy as np
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, Empty, String
from move_base_msgs.msg import MoveBaseActionResult
from frontier_finder import FrontierDetector 
from frontier_selector import FrontierSelector

class FrontierNode:
    def __init__(self):
        rospy.init_node('frontier_explorer_node')

        # 1. State Variables
        self.latest_map = None
        self.latest_costmap = None
        self.detector = None
        self.current_goal = None
        
        self.listener = tf.TransformListener()
        self.selector = FrontierSelector()

        # 2. Subscribers
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        self.costmap_sub = rospy.Subscriber("/move_base/global_costmap/costmap", OccupancyGrid, self.costmap_callback)
        self.result_sub = rospy.Subscriber("/move_base/result", MoveBaseActionResult, self.result_callback)
        self.controller_sub = rospy.Subscriber("controller_main", String, self.controller_cb)

        # 3. Publishers
        self.frontier_map_pub = rospy.Publisher('/detected_frontiers', OccupancyGrid, queue_size=1)
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
        self.path_pub = rospy.Publisher('/global_exploration_path', Path, queue_size=1)

        # 4. Loop Timer //enable this to make this loop
        self.timer = rospy.Timer(rospy.Duration(2.0), self.process_frontiers)

    def map_callback(self, msg):
        self.latest_map = msg
        if self.detector is None:
            self.detector = FrontierDetector(
                map_width=msg.info.width,
                map_height=msg.info.height,
                resolution=msg.info.resolution,
                origin_x=msg.info.origin.position.x,
                origin_y=msg.info.origin.position.y
            )

    def costmap_callback(self, msg):
        self.latest_costmap = msg

    def controller_cb(self, msg):
        if msg == "process_frontiers":
            print("Processing Frontiers...")
            self.process_frontiers(None)        
        #TODO: do this tmr

    def result_callback(self, msg):
        if msg.status.status == 4 and self.current_goal:
            rospy.logwarn("Goal failed! Blacklisting area.")
            # Ensure your FrontierSelector has an 'add_to_blacklist' or similar
            self.selector.blacklist[self.current_goal] = rospy.get_time()

    def world_to_grid(self, world_point):
        """Converts (x, y) meters to (x, y) grid indices"""
        info = self.latest_map.info
        gx = int((world_point[0] - info.origin.position.x) / info.resolution)
        gy = int((world_point[1] - info.origin.position.y) / info.resolution)
        return (gx, gy)

    def grid_to_world(self, grid_point):
        """Converts (x, y) grid indices to (x, y) meters"""
        info = self.latest_map.info
        wx = grid_point[0] * info.resolution + info.origin.position.x
        wy = grid_point[1] * info.resolution + info.origin.position.y
        return (wx, wy)

    def process_frontiers(self, event):
        if self.latest_map is None or self.latest_costmap is None or self.detector is None:
            return

        # 1. Get Robot Pose in World Coords then convert to Grid
        robot_pose_world = self.get_robot_pose()
        if robot_pose_world is None:
            return
        robot_grid = self.world_to_grid(robot_pose_world)

        # 2. Reshape 1D data to 2D numpy arrays for the A* Planner
        width = self.latest_map.info.width
        height = self.latest_map.info.height
        
        # Static map: -1, 0, 100
        static_2d = np.array(self.latest_map.data).reshape((height, width))
        # Costmap: 0-100
        cost_2d = np.array(self.latest_costmap.data).reshape((height, width))

        # 3. Detect Frontiers
        # Assuming detector returns centroids in GRID coordinates
        frontiers_metadata, frontier_map_data = self.detector.detect_frontiers(
            self.latest_map.data, 
            self.latest_costmap.data
        )

        # 4. Select Frontier using our cost-aware A* logic
        # best_f_dict now contains 'path' and 'centroid' (in grid units)
        best_f_dict = self.selector.select_frontier(
            robot_grid, 
            frontiers_metadata, 
            cost_2d, 
            static_2d
        )

        print("Best Frontier", best_f_dict)

        # 5. Visualization
        self.publish_frontier_map(self.latest_map, frontier_map_data)

        if best_f_dict:
            self.publish_visual_path(best_f_dict['path'])
            grid_goal = best_f_dict['centroid']
            
            # Convert back to World Coordinates for move_base
            world_goal = self.grid_to_world(grid_goal)
            
            rospy.loginfo("TARGET SELECTED -> ID: {} | World Coords: {}".format(
                best_f_dict.get('id', 'N/A'), world_goal))

            self.current_goal = grid_goal

    def get_robot_pose(self):
        try:
            (trans, rot) = self.listener.lookupTransform('/map', '/base_link', rospy.Time(0))
            return (trans[0], trans[1])
        except:
            return None

    def publish_goal(self, coords):
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = rospy.Time.now()
        goal.pose.position.x = coords[0]
        goal.pose.position.y = coords[1]
        goal.pose.orientation.w = 1.0
        self.goal_pub.publish(goal)

    def publish_frontier_map(self, original_msg, debug_data):
        f_map = OccupancyGrid()
        f_map.header.stamp = rospy.Time.now()
        f_map.header.frame_id = original_msg.header.frame_id
        f_map.info = original_msg.info 
        f_map.data = debug_data
        self.frontier_map_pub.publish(f_map)

    def publish_visual_path(self, grid_path):
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = rospy.Time.now()

        for grid_point in grid_path:
            # Convert each grid point back to meters (world coords)
            world_x, world_y = self.grid_to_world(grid_point)
            
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = world_x
            pose.pose.position.y = world_y
            pose.pose.orientation.w = 1.0 # Default orientation
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)

if __name__ == '__main__':
    node = FrontierNode()
    rospy.spin()