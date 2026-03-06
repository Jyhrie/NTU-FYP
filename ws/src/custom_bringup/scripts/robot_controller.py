#!/usr/bin/env python

from tkinter import NO

import rospy
from enum import Enum
from std_msgs.msg import Empty, String
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PointStamped
from visualization_msgs.msg import Marker
import json
import tf2_ros
import tf2_geometry_msgs
import tf
import math
import sys
from dependencies import utils


# --- State Definitions ---
class States(Enum):
    IDLE = 0
    MAPPING = 1
    FETCHING = 2
    EXPLORING = 3
    NULL = 4

class SubStates(Enum):
    READY = 0       # Initial entry into a state
    REQUESTING = 1  # Waiting for external node data (paths/commands)
    MOVING = 2      # Actively navigating or rotating
    WAITING = 3     # Short pauses or timeouts

    ITEM_CONFIRMED = 4  # Detected item, preparing to align
    ROTATING_TO_UNOBSCURED_VIEW = 5
    DEPTH_READING_UNAVAILABLE = 6
    DEPTH_READING_AVAILABLE = 7
    MOVING_TO_DEPTH_AVAILABLE = 8
    APPROACH_ITEM = 9
    PICKING_UP = 10
    PICKED_UP = 11
    REQUESTING_HOME_PATH = 12
    COMPLETE = 13

    MOVING_TO_ITEM = 4
    ALIGNING = 5
    PICKING_UP = 6
    PICKED_UP = 7
    RETURNING = 8
    REVERSING = 9        # For retry logic
    REQUESTING_HOME_PATH = 10
    COMPLETE = 11
    APPROACHING = 12

    CONFIRMING_ITEM = 14

class NavStates(Enum):
    NULL = 0
    MOVING = 1
    COMPLETE = 2 # Simplified from your original set

class Controller:
    def __init__(self):
        rospy.init_node("python_controller")
        
        # --- Original Publishers ---
        self.recalib_pub = rospy.Publisher("/recalib_frontiers", Empty, queue_size=1)
        self.state_pub   = rospy.Publisher("/controller_state", String, queue_size=1)
        self.global_request = rospy.Publisher("/controller/global", String, queue_size=1)
        self.rotate_pose_pub = rospy.Publisher("/rotate_target_pose", PoseStamped, queue_size=1)
        self.global_exploration_path = rospy.Publisher("/global_exploration_path", Path, queue_size=1)

        #TO REMOVE
        self.marker_pub = rospy.Publisher('/detected_object_marker', Marker, queue_size=10)

        # --- TF Setup ---
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # --- Original Subscribers ---
        self.fontier_node_sub = rospy.Subscriber("/frontier_node_reply", String, self.frontier_node_cb)
        self.frontier_node_path_sub = rospy.Subscriber("/frontier_node_path", Path, self.frontier_node_path_cb)
        self.navigation_node_sub = rospy.Subscriber("/navigation_node_reply", String, self.navigation_node_cb)
        #self.pc_node_sub = rospy.Subscriber("/pc_node_reply", String, self.pc_node_cb)
        self.movement_controller_sub = rospy.Subscriber("/movement_controller_message", String, self.movement_controller_cb)
        self.reply_sub = rospy.Subscriber("/robot/reply", String, self.global_reply_cb) 
        self.path_sub = rospy.Subscriber("/robot/path_reply", Path, self.path_reply_cb)
        self.cv_sub = rospy.Subscriber("/robot/cv", String, self.cv_cb)

        # --- Internal Variables ---
        self.state = States.IDLE
        self.sub_state = SubStates.READY
        self.nav_state = NavStates.NULL
        
        self.received = None
        self.received_path = None
        self.goal_path = None

        self.last_cv_detection = None

        self.rotate_target_msg = None
        self.pickup_target = None 
        self.pickup_target_angle_relative_to_forward = None
        self.object_box = None

        self.cached_pickup_distance = None
        
        self.request_sent = False
        self.request_timeout = 30
        self.start_time = 0
        self.rate = rospy.Rate(5)

        print("Initialization Complete, Node is Ready!")

    # ====== CALLBACKS (Your original logic preserved) ====== #
    def global_reply_cb(self, msg): self.received = json.loads(msg.data)
    def path_reply_cb(self, msg): self.received_path = msg
    def frontier_node_cb(self, msg): self.received = json.loads(msg.data)
    def frontier_node_path_cb(self, msg): self.received_path = msg

    def cv_cb(self, msg):
        
        if self.state in [States.IDLE, States.MAPPING]: #interruptable
            self.interrupt(clear=True)
            self.state = States.FETCHING

        data = json.loads(msg.data)
        self.last_cv_detection = data
        pass

    def movement_controller_cb(self, msg):
        if msg.data == "done":
            print("Movement Controller reports: Movement Complete")
            if self.state == States.FETCHING:
                if self.sub_state == SubStates.MOVING_TO_ITEM:
                    self.sub_state = SubStates.ALIGNING
                elif self.sub_state == SubStates.APPROACHING:
                    self.sub_state = SubStates.PICKING_UP
                elif self.sub_state == SubStates.ALIGNING:
                    self.cached_pickup_distance = self.calculate_distance(self.object_box[0]) if self.object_box else None
                    self.sub_state = SubStates.APPROACHING
                elif self.sub_state == SubStates.RETURNING:
                    print("Back at origin. Fetch complete.")
                    self.transition(States.NULL)
                return
            self.sub_state = SubStates.COMPLETE

    def navigation_node_cb(self, msg):
        if msg.data == "COMPLETE":
            self.sub_state = SubStates.COMPLETE

    # def pc_node_cb(self, msg):
    #     data = json.loads(msg.data)

    #     # 3. Unpack the specific fields from your log
    #     timestamp = float(data.get('timestamp'))
    #     target_label = data.get('target')        # e.g., "object_name" or "can"
    #     angle_to_target = -float(data.get('angle'))     # e.g., 23.85
    #     confidence = data.get('conf')            # e.g., 0.913
        
    #     width = data.get('w')                    # e.g., 59.6
    #     height = data.get('h')                   # e.g., 125.1

    #     # 4. Use the data (Example: Print and set target)
    #     #print("Robot received target time:", timestamp, " at", angle_to_target, "degrees")

    #     dist_m = self.calculate_distance(width)
    #     #print("Distance = ", dist_m)

    #     get_x, get_y = self.get_relative_pickup_target(timestamp, angle_to_target, dist_m) # Assuming a fixed distance of 1.0m for now

    #     #self.global_request.publish(msg)
    
    #     #load data in
    #     if self.state != States.FETCHING:
    #         self.interrupt() # Stop current action immediately
    #         self.pickup_target = (get_x, get_y)
    #         self.last_pickup_target_time = timestamp
    #         self.transition(States.FETCHING, SubStates.READY)
    #     else:
    #         self.last_pickup_target_time = timestamp
    #         self.pickup_target_angle_relative_to_forward = angle_to_target
    #         self.object_box = (width, height)
        
    # ====== UTILS (Original methods) ====== #
    def get_robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform("map", "base_link", rospy.Time(0), rospy.Duration(0.1))
            x, y = t.transform.translation.x, t.transform.translation.y
            q = t.transform.rotation
            (_, _, yaw) = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
            return x, y, yaw
        except: return None

    def calculate_distance(self, pixel_width):
        # Constants for Starbucks Can Diameter + Astra Pro @ 640px
        FOCAL_LENGTH = 572.5  # Constant for 60 deg HFOV at 640px resolution
        REAL_WIDTH = 0.065    # ~6.5 cm (diameter of the can) in meters
        
        if pixel_width <= 0:
            return 0.0
            
        # Formula: distance = (FocalLength * RealWidth) / PixelWidth
        distance = (FOCAL_LENGTH * REAL_WIDTH) / pixel_width
        return distance

    def get_relative_pickup_target(self, timestamp, angle, distance):
            angle_rad = math.radians(angle)
            
            # Setup the points for the target
            local_pt = PointStamped()
            try:
                if isinstance(timestamp, (str, unicode, float, int)):
                    local_pt.header.stamp = rospy.Time.from_sec(float(timestamp))
                else:
                    local_pt.header.stamp = timestamp
            except:
                local_pt.header.stamp = rospy.Time.now()

            # --- NEW: Get Robot's Global Position at that time ---
            try:
                # Transform the center of the robot (0,0,0) at the timestamp to the map
                robot_origin = PointStamped()
                robot_origin.header.stamp = local_pt.header.stamp
                robot_origin.header.frame_id = "base_link"
                robot_pos = self.tf_buffer.transform(robot_origin, "map", timeout=rospy.Duration(1.0))
                
                # Publish Blue Marker for Robot Position
                self.publish_marker(robot_pos.point.x, robot_pos.point.y, marker_id=1, color="blue")
            except Exception as e:
                rospy.logwarn("Could not find robot past position: %s", e)

            # --- Existing Target Logic ---
            local_pt.header.frame_id = "camera_link" 
            local_pt.point.x = distance * math.cos(angle_rad)
            local_pt.point.y = distance * math.sin(angle_rad)
            
            try:
                map_pt = self.tf_buffer.transform(local_pt, "map", timeout=rospy.Duration(1.0))
                
                # Publish Green Marker for Target Can
                self.publish_marker(map_pt.point.x, map_pt.point.y, marker_id=0, color="green")
                
                return (map_pt.point.x, map_pt.point.y)
            except Exception as e:
                rospy.logwarn("TF Transform failed: %s", str(e))
                return (None, None)

    def publish_marker(self, x, y, marker_id=0, color="green"):
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
                
            self.marker_pub.publish(marker)

    def prepare_flip(self):
        pose = self.get_robot_pose()
        if pose:
            curr_x, curr_y, yaw = pose
            msg = PoseStamped()
            msg.header.frame_id, msg.header.stamp = "map", rospy.Time.now()
            msg.pose.position.x, msg.pose.position.y = curr_x, curr_y
            q = tf.transformations.quaternion_from_euler(0, 0, self.wrap_angle(yaw + math.pi))
            msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = q
            return msg

    def wrap_angle(self, a): return math.atan2(math.sin(a), math.cos(a))

    def interrupt(self, clear=False):
        print("Interrupting Current Action")
        if clear == True:
            self.goal_path = None
            self.rotate_target_msg = None
            self.pickup_target = None

        self.global_request.publish("interrupt")
        self.transition(States.IDLE, SubStates.READY) # Reset to IDLE and clear sub-state to READY for a fresh start

    def transition(self, nxt_state, nxt_sub=SubStates.READY):
        print("Transitioning %s -> %s (%s)" % (self.state.name, nxt_state.name, nxt_sub.name))
        self.state = nxt_state
        self.sub_state = nxt_sub
        self.request_sent = False
        self.received = None

    # ====== MAIN STATE LOGIC ====== #
    def run(self):
        rospy.sleep(1.0)
        # self.state = States.FETCHING
        # self.sub_state = SubStates.REQUESTING_HOME_PATH
        # self.nav_state = NavStates.NULL

        while not rospy.is_shutdown():
            #debug
            status = "\rMain State: %-10s | Sub-State: %-10s\033[K" % (self.state.name, self.sub_state.name)
            sys.stdout.write(status)
            sys.stdout.flush()

            if self.state == States.IDLE:    self.manage_idle()
            elif self.state == States.MAPPING:  self.manage_mapping()
            elif self.state == States.FETCHING: self.manage_fetching()
            self.rate.sleep()

    def manage_idle(self):
        #self.transition(States.MAPPING)
        return

    def manage_mapping(self):
            # Initial entry: move to requesting data
            if self.sub_state == SubStates.READY:
                self.sub_state = SubStates.REQUESTING
                self.goal_path = None
                self.rotate_target_msg = None

            # Logic for requesting and receiving paths/commands
            elif self.sub_state == SubStates.REQUESTING:
                if not self.request_sent:
                    self.global_request.publish("request_frontier")
                    self.request_sent = True
                    self.start_time = rospy.get_time()

                if self.received:
                    # Handle path receipt
                    if self.received.get("data") == "path" and self.received_path:
                        self.goal_path = self.received_path
                        self.sub_state = SubStates.MOVING
                    # Handle rotation command
                    elif self.received.get("data") == "rotate":
                        self.rotate_target_msg = self.prepare_flip()
                        self.sub_state = SubStates.MOVING
                    
                    # Reset communication flags
                    self.received = None
                    self.request_sent = False

            # Logic while the robot is physically in motion
            elif self.sub_state == SubStates.COMPLETE:
                self.goal_path = None
                self.rotate_target_msg = None
                self.transition(States.MAPPING, SubStates.READY)

            elif self.sub_state == SubStates.MOVING:
                # Execute the active task
                if self.goal_path:
                    self.global_request.publish("navigate")
                    self.global_exploration_path.publish(self.goal_path)
                elif self.rotate_target_msg:
                    self.global_request.publish("rotate")
                    self.rotate_pose_pub.publish(self.rotate_target_msg)
            else:
                print("Unknown sub-state in MAPPING: {}".format(self.sub_state))

    def manage_fetching(self):
        #initial entry due to interrupt
        if self.sub_state == SubStates.READY:
            msg = String()
            msg.data = json.dumps({
                "header": "interrupt", 
                "command": "stop_movement"
            })
            self.global_request.publish(msg)
            rospy.sleep(2.0)

            #get ready next set of data received from cv node
            self.sub_state = SubStates.CONFIRMING_ITEM
        
        
        if self.sub_state == SubStates.CONFIRMING_ITEM: #robot will be at rest NOW.
            #get current timestamp
            cached_last_cv_detection = self.last_cv_detection
            rospy.sleep(1.5)
            latest_cv_detection = self.last_cv_detection

            if float(latest_cv_detection['ros_time']) - float(cached_last_cv_detection['ros_time']) > 1.2: #PERSISTENCE CHECK PASS
                bbox_old = (
                    cached_last_cv_detection['x_start'], 
                    cached_last_cv_detection['y_start'], 
                    cached_last_cv_detection['x_len'], 
                    cached_last_cv_detection['y_len']
                )

                bbox_latest = (
                    latest_cv_detection['x_start'], 
                    latest_cv_detection['y_start'], 
                    latest_cv_detection['x_len'], 
                    latest_cv_detection['y_len']
                )
                if utils.compare_bbox_centroid(bbox_old, bbox_latest, radius=12):
                    self.sub_state = SubStates.DEPTH_READING_AVAILABLE
            return
                

        if self.sub_state == SubStates.DEPTH_READING_UNAVAILABLE:
            msg = String()
            msg.data = json.dumps({
                "header": "depth_node",
                "command": "check_depth"
            })
            self.global_request.publish(msg)

        if self.sub_state == SubStates.DEPTH_READING_AVAILABLE:
            msg = String()
            #TODO: use helper function from another node to get map, and find the lowest cost point in a radius.
            msg.data = json.dumps({ 
                "header": "waypoint_node",
                "command": "nav_to_point",
                "x": self.pickup_target[0],
                "y": self.pickup_target[1]
            })
            self.global_request.publish(msg)


        if self.sub_state == SubStates.APPROACH_ITEM:
            msg = String()
            msg.data = json.dumps({
                "header": "movement_controller",
                "command": "approach_item",
                "distance": self.cached_pickup_distance - 0.35, # Stop 35 cm away from the target to prepare for pickup
            })
            self.global_request.publish(msg)

        if self.sub_state == SubStates.PICKING_UP:
            msg = String()
            msg.data = json.dumps({
                "header": "arm",
                "command": "grab"
            })
            self.global_request.publish(msg)
        
        #reverse then go home

            

        #tell the robot to stop until we confirm the pickup target

        #if we get a depth reading right smack in the middle of the pickup target, we can move on to approaching the target
        
        #if we dont get a depth reading, we need to reverse/approach slightly to try to get a depth reading, then we can move on to approaching the target
        #(use waypoint_node to determine position, then tell robot to turn to face object, then offset +- 15deg.)

        #then we rotate to face the target, using center mass of the depth blob as a reference point to align with

        #once we are close enough, we can attempt the pickup sequence

        #then call for return to home

        #then we can loop back to idle or mapping or whatever we want after that


    #     if self.sub_state == SubStates.READY:
    #         if self.pickup_target:
    #             self.sub_state = SubStates.REQUESTING
    #         else:
    #             self.transition(States.IDLE)

    # def manage_fetching(self):
    #         # --- 1. READY: Initial Entry ---
    #         if self.sub_state == SubStates.READY:
    #             if self.pickup_target:
    #                 self.sub_state = SubStates.REQUESTING
    #             else:
    #                 self.transition(States.IDLE)

    #         # --- 2. REQUEST PATH TO ITEM ---
    #         elif self.sub_state == SubStates.REQUESTING:
    #             if not self.request_sent:
    #                 msg = String()
    #                 msg.data = json.dumps({
    #                     "command": "request_waypoint", 
    #                     "x": self.pickup_target[0], 
    #                     "y": self.pickup_target[1],
    #                 })
    #                 self.global_request.publish(msg)
    #                 self.request_sent = True

    #             if self.received_path: # Once path_reply_cb gets the path
    #                 # 1. Create a new Path message container
    #                 path_msg = Path()
    #                 path_msg.header = self.received_path.header
    #                 path_msg.header.stamp = rospy.Time.now()

    #                 # 2. Slice the list of poses from the received path
    #                 # Check length first to avoid empty paths
    #                 if len(self.received_path.poses) > 10:
    #                     path_msg.poses = self.received_path.poses[:-10]
    #                 else:
    #                     self.sub_state = SubStates.ALIGNING

    #                 # 3. Store the reformed message
    #                 self.goal_path = path_msg
    #                 self.request_sent = False
    #                 print("path received")
    #                 self.sub_state = SubStates.MOVING_TO_ITEM
    #         # --- 3. MOVE TO ITEM ---

    #         elif self.sub_state == SubStates.MOVING_TO_ITEM:
    #             if self.goal_path:
    #                 nav_msg = {
    #                 "header": "navigate",
    #                 "end_face_pt_x": self.pickup_target[0],
    #                 "end_face_pt_y": self.pickup_target[1]
    #                 }
    #                 self.global_request.publish(json.dumps(nav_msg))
    #                 self.global_exploration_path.publish(self.goal_path)
                    
    #         # --- 4. ALIGN WITH ITEM ---
    #         elif self.sub_state == SubStates.ALIGNING:
    #             align_msg = {
    #                 "header": "align_with_item",
    #                 "data": {
    #                     "timestamp": self.last_pickup_target_time,
    #                     "relative_angle": self.pickup_target_angle_relative_to_forward
    #                 }
    #             }
    #             self.global_request.publish(json.dumps(align_msg))

    #         elif self.sub_state == SubStates.APPROACHING:
    #             approach_msg = {
    #                 "header": "approach",
    #                 "data": {
    #                     "timestamp": self.last_pickup_target_time,
    #                     "cached_distance": self.cached_pickup_distance - 0.35,
    #                     "relative_angle": self.pickup_target_angle_relative_to_forward,
    #                     "max_distance": self.calculate_distance(self.object_box[0]),
    #                     "linear_speed": 0.07
    #                 }
    #             }
    #             self.global_request.publish(json.dumps(approach_msg))

    #             #perform a check to see if we are close enough to the target to attempt pickup
    #             TARGET_W = 100
                
    #             if self.object_box and self.object_box[0] >= TARGET_W:
    #                 print("Close enough to attempt pickup!")
    #                 self.sub_state = SubStates.PICKING_UP
    #                 msg = {
    #                     "header": "stop_movement",
    #                 }
    #                 self.global_request.publish(json.dumps(msg))
                
    #         # --- 5. PICK UP (With Failure Handling) ---
    #         elif self.sub_state == SubStates.PICKING_UP:
    #             pickup_msg = {
    #                 "header": "arm",
    #                 "command": 'grab'
    #             }
    #             self.global_request.publish(json.dumps(pickup_msg))
    #             rospy.sleep(5)
    #             # After waiting for the arm sequence, we check if the pickup was successful
    #             self.sub_state = SubStates.REQUESTING_HOME_PATH

    #         # --- 6. REVERSE (Retry Loop) ---
    #         # elif self.sub_state == SubStates.REVERSING:
    #         #     if self.movement_complete:
    #         #         self.movement_complete = False
    #         #         # Loop back to alignment to try again
    #         #         self.sub_state = SubStates.ALIGNING
    #         #     else:
    #         #         self.global_request.publish("reverse_slightly")

    #         # --- 7. REQUEST PATH TO ORIGIN ---
    #         elif self.sub_state == SubStates.REQUESTING_HOME_PATH:
    #             if not self.request_sent:
    #                 msg = {
    #                     "header":"waypoint_node",
    #                     "command":"request_home"
    #                 }
    #                 self.global_request.publish(json.dumps(msg))
    #                 self.request_sent = True
    #                 self.received_path = None # Clear any old paths just in case

    #             if self.received_path:
    #                 self.goal_path = self.received_path
    #                 self.request_sent = False
    #                 self.sub_state = SubStates.RETURNING

    #         # --- 8. MOVE TO ORIGIN ---
    #         elif self.sub_state == SubStates.RETURNING:
    #             self.global_request.publish("navigate")
    #             self.global_exploration_path.publish(self.goal_path)
    #             #will get kicked out of this state by the movement controller callback once navigation is complete

if __name__ == "__main__":
    ctrl = Controller()
    ctrl.run()