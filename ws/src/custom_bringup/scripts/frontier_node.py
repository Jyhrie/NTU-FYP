#!/usr/bin/env python

import rospy
import tf
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
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
        
        # Add TF Listener to get the robot's position
        self.listener = tf.TransformListener()
        
        # Initialize Selector in desired mode
        self.selector = FrontierSelector() # Options: "greedy" or "fast"

        # 2. Subscribers
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        self.costmap_sub = rospy.Subscriber("/move_base/global_costmap/costmap", OccupancyGrid, self.costmap_callback)
        
        # NEW: Listen to move_base results to blacklist failed gaps automatically
        self.result_sub = rospy.Subscriber("/move_base/result", MoveBaseActionResult, self.result_callback)

        # 3. Publishers
        self.frontier_map_pub = rospy.Publisher('/detected_frontiers', OccupancyGrid, queue_size=1)
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)

        # 4. Timer - This IS your loop
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

    def result_callback(self, msg):
        # If the goal was aborted (Status 4), tell the selector to blacklist it
        if msg.status.status == 4 and self.current_goal:
            rospy.logwarn("Goal {} failed! Blacklisting area.".format(self.current_goal))
            self.selector.add_to_blacklist(self.current_goal)

    def get_robot_pose(self):
        """Helper to get current robot (x, y) from TF"""
        try:
            (trans, rot) = self.listener.lookupTransform('/map', '/base_link', rospy.Time(0))
            return (trans[0], trans[1])
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return None

    def process_frontiers(self, event):
        """This function acts as the main exploration loop."""
        if self.latest_map is None or self.latest_costmap is None or self.detector is None:
            return

        if len(self.latest_map.data) != len(self.latest_costmap.data):
            return

        # 1. Get Robot Pose
        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            rospy.logwarn_throttle(5, "Waiting for TF transform...")
            return

        # 2. Detect Frontiers (returns metadata list of dicts)
        frontiers_metadata, frontier_map_data = self.detector.detect_frontiers(
            self.latest_map.data, 
            self.latest_costmap.data
        )

        # 3. Select the best goal based on Mode (greedy/fast)
        best_f_dict = self.selector.select_frontier(robot_pose, frontiers_metadata)
        print("Best Frontier: ", best_f_dict)

        # 4. Publish Visualization (RViz)
        self.publish_frontier_map(self.latest_map, frontier_map_data)

        if best_f_dict:
                    # The selector returns the centroid (x,y) tuple based on your current code
                    # To print the ID, we ensure we are looking at the best_f dictionary
                    
                    # Extract data
                    goal_coords = best_f_dict['centroid']
                    goal_id = best_f_dict['id']
                    
                    # --- PRINT THE ID HERE ---
                    rospy.loginfo("TARGET SELECTED -> ID: {} | Coords: {}".format(goal_id, goal_coords))

                    # Update state for blacklisting/hysteresis
                    self.current_goal = goal_coords
                    
                    # Send to MoveBase
                    #self.publish_goal(goal_coords)

    def publish_goal(self, coords):
        """Helper to send a PoseStamped goal to move_base"""
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = rospy.Time.now()
        goal.pose.position.x = coords[0]
        goal.pose.position.y = coords[1]
        goal.pose.orientation.w = 1.0 # Neutral orientation
        self.goal_pub.publish(goal)

    def publish_frontier_map(self, original_msg, debug_data):
        f_map = OccupancyGrid()
        f_map.header.stamp = rospy.Time.now()
        f_map.header.frame_id = original_msg.header.frame_id
        f_map.info = original_msg.info 
        f_map.data = debug_data
        self.frontier_map_pub.publish(f_map)

if __name__ == '__main__':
    try:
        node = FrontierNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass