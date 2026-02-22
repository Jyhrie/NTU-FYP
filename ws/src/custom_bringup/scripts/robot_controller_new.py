#!/usr/bin/env python

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

# --- State Definitions ---
class States(Enum):
    IDLE = 0
    MAPPING = 1
    FETCHING = 2

class SubStates(Enum):
    READY = 0       # Initial entry into a state
    REQUESTING = 1  # Waiting for external node data (paths/commands)
    MOVING = 2      # Actively navigating or rotating
    WAITING = 3     # Short pauses or timeouts
    MOVING_TO_ITEM = 4
    ALIGNING = 5
    PICKING_UP = 6
    PICKED_UP = 7
    RETURNING = 8
    REVERSING = 9        # For retry logic
    REQUESTING_HOME_PATH = 10
    COMPLETE = 11

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
        self.pc_node_sub = rospy.Subscriber("/pc_node_reply", String, self.pc_node_cb)
        self.movement_controller_sub = rospy.Subscriber("/movement_controller_message", String, self.movement_controller_cb)
        self.reply_sub = rospy.Subscriber("/robot/reply", String, self.global_reply_cb) 
        self.path_sub = rospy.Subscriber("/robot/path_reply", Path, self.path_reply_cb)

        # --- Internal Variables ---
        self.state = States.IDLE
        self.sub_state = SubStates.READY
        self.nav_state = NavStates.NULL
        
        self.received = None
        self.received_path = None
        self.goal_path = None
        self.rotate_target_msg = None
        self.pickup_target = None 
        
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
    
    def movement_controller_cb(self, msg):
        if msg.data == "done":
            self.sub_state = SubStates.COMPLETE

    def navigation_node_cb(self, msg):
        if msg.data == "COMPLETE":
            self.sub_state = SubStates.COMPLETE

    def pc_node_cb(self, msg):
        data = json.loads(msg.data)

        # 3. Unpack the specific fields from your log
        timestamp = float(data.get('timestamp'))
        target_label = data.get('target')        # e.g., "object_name" or "can"
        angle_to_target = data.get('angle')      # e.g., 23.85
        confidence = data.get('conf')            # e.g., 0.913
        
        width = data.get('w')                    # e.g., 59.6
        height = data.get('h')                   # e.g., 125.1

        dist_m = self.calculate_distance(width)
        print("Distance = ", dist_m)

        self.get_relative_pickup_target(timestamp, angle_to_target, dist_m) # Assuming a fixed distance of 1.0m for now

        # 4. Use the data (Example: Print and set target)
        print("Robot received target time:", timestamp, " at", angle_to_target, "degrees")
        self.interrupt() # Stop current action immediately

        return 
        if data.get("header") == "interrupt":
            timestamp = data.get("timestamp")
            angle = data.get("angle")
            distance = data.get("distance")

            if None not in (timestamp, angle, distance):
                coords = self.get_relative_pickup_target(timestamp, angle, distance)
                if coords[0] is not None:
                    self.pickup_target = coords
                    self.interrupt() # Stops current motion
                    self.transition(States.FETCHING, SubStates.READY)

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
        print("Distance calculated:", distance)
        return distance

    def get_relative_pickup_target(self, timestamp, angle, distance):
            angle = -angle
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
        self.substate = SubStates.READY

    def transition(self, nxt_state, nxt_sub=SubStates.READY):
        print("Transitioning %s -> %s (%s)" % (self.state.name, nxt_state.name, nxt_sub.name))
        self.state = nxt_state
        self.sub_state = nxt_sub
        self.request_sent = False
        self.received = None

    # ====== MAIN STATE LOGIC ====== #
    def run(self):
        rospy.sleep(1.0)
        while not rospy.is_shutdown():
            if self.state == States.IDLE:    self.manage_idle()
            elif self.state == States.MAPPING:  self.manage_mapping()
            elif self.state == States.FETCHING: self.manage_fetching()
            self.rate.sleep()

    def manage_idle(self):
        pass
        #self.transition(States.MAPPING)

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
            # --- 1. READY: Initial Entry ---
            if self.sub_state == SubStates.READY:
                if self.pickup_target:
                    self.sub_state = SubStates.REQUESTING
                else:
                    self.transition(States.IDLE)

            # --- 2. REQUEST PATH TO ITEM ---
            elif self.sub_state == SubStates.REQUESTING:
                if not self.request_sent:
                    msg = String()
                    msg.data = json.dumps({
                        "cmd": "request_path", 
                        "x": self.pickup_target[0], 
                        "y": self.pickup_target[1]
                    })
                    self.global_request.publish(msg)
                    self.request_sent = True

                if self.received_path: # Once path_reply_cb gets the path
                    self.goal_path = self.received_path
                    self.request_sent = False
                    self.sub_state = SubStates.MOVING_TO_ITEM

            # --- 3. MOVE TO ITEM ---
            elif self.sub_state == SubStates.MOVING_TO_ITEM:
                if self.movement_complete:
                    self.movement_complete = False
                    self.sub_state = SubStates.ALIGNING
                else:
                    self.global_request.publish("navigate")
                    self.global_exploration_path.publish(self.goal_path)

            # --- 4. ALIGN WITH ITEM ---
            elif self.sub_state == SubStates.ALIGNING:
                if self.movement_complete:
                    self.movement_complete = False
                    self.sub_state = SubStates.PICKING_UP
                else:
                    self.global_request.publish("align_with_item")

            # --- 5. PICK UP (With Failure Handling) ---
            elif self.sub_state == SubStates.PICKING_UP:
                if self.received: # Wait for a reply from the hardware node
                    if self.received.get("data") == "success":
                        print("Pick up successful!")
                        self.received = None
                        self.sub_state = SubStates.REQUESTING_HOME_PATH
                    elif self.received.get("data") == "failed":
                        print("Pick up failed. Reversing to retry...")
                        self.received = None
                        self.sub_state = SubStates.REVERSING
                else:
                    self.global_request.publish("execute_pickup")

            # --- 6. REVERSE (Retry Loop) ---
            elif self.sub_state == SubStates.REVERSING:
                if self.movement_complete:
                    self.movement_complete = False
                    # Loop back to alignment to try again
                    self.sub_state = SubStates.ALIGNING
                else:
                    self.global_request.publish("reverse_slightly")

            # --- 7. REQUEST PATH TO ORIGIN ---
            elif self.sub_state == SubStates.REQUESTING_HOME_PATH:
                if not self.request_sent:
                    self.global_request.publish("request_home")
                    self.request_sent = True

                if self.received_path:
                    self.goal_path = self.received_path
                    self.request_sent = False
                    self.sub_state = SubStates.RETURNING

            # --- 8. MOVE TO ORIGIN ---
            elif self.sub_state == SubStates.RETURNING:
                if self.movement_complete:
                    self.movement_complete = False
                    print("Back at origin. Fetch complete.")
                    self.transition(States.IDLE)
                else:
                    self.global_request.publish("navigate")
                    self.global_exploration_path.publish(self.goal_path)

if __name__ == "__main__":
    ctrl = Controller()
    ctrl.run()