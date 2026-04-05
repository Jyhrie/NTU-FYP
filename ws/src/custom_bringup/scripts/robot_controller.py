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
    PATH_RECEIVED = 4
    ITEM_CONFIRMED = 5  # Detected item, preparing to align
    ROTATING_TO_UNOBSCURED_VIEW = 6
    READING_DEPTH = 7
    DEPTH_READING_UNAVAILABLE = 8
    DEPTH_READING_AVAILABLE = 9
    MOVING_TO_DEPTH_AVAILABLE = 10
    APPROACH_ITEM = 11
    PICKING_UP = 12
    PICKED_UP = 13
    REQUESTING_HOME_PATH = 14
    COMPLETE = 15
    CONFIRMING_ITEM = 16
    WAITING_PATH_RESPONSE = 17
    WAITING_HOME_PATH_RESPONSE = 18
    MOVING_HOME = 19
    DROPPING_ITEM = 20
    REALIGNMENT_OUT = 21
    REALIGNMENT_OUT_MOVING = 22
    REALIGNMENT_WAITING_ITEM = 23
    REALIGNMENT_WAITING_DEPTH = 24
    REALIGNMENT_IN = 25
    REALIGNMENT_IN_MOVING = 26

class NavStates(Enum):
    NULL = 0
    MOVING = 1
    COMPLETE = 2

class Controller:
    def __init__(self):
        rospy.init_node("python_controller")
        
        # --- Original Publishers ---
        self.recalib_pub = rospy.Publisher("/recalib_frontiers", Empty, queue_size=1)
        self.state_pub   = rospy.Publisher("/controller_state", String, queue_size=1)
        self.global_request = rospy.Publisher("/controller/global", String, queue_size=1)
        self.rotate_pose_pub = rospy.Publisher("/rotate_target_pose", PoseStamped, queue_size=1)
        self.global_path = rospy.Publisher("/global_path", Path, queue_size=1)

        #TO REMOVE
        self.marker_pub = rospy.Publisher('/self_marker', Marker, queue_size=10)

        # --- TF Setup ---
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # --- Original Subscribers ---
        # self.fontier_node_sub = rospy.Subscriber("/frontier_node_reply", String, self.frontier_node_cb)
        # self.frontier_node_path_sub = rospy.Subscriber("/frontier_node_path", Path, self.frontier_node_path_cb)
        # self.navigation_node_sub = rospy.Subscriber("/navigation_node_reply", String, self.navigation_node_cb)
        self.depth_node_sub = rospy.Subscriber("/robot/depth", String, self.depth_cb)
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

        self.detected_distance = None
        self.detected_angle = None
        self.target_object_transform = None

        self.f_mapping_complete = False
        
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
        data = json.loads(msg.data)
        if data['header'] == "done":
            print("Movement Controller reports: Movement Complete")
            if self.state == States.FETCHING:
                if self.sub_state == SubStates.MOVING:
                    self.sub_state = SubStates.REALIGNMENT_OUT
                
                if self.sub_state == SubStates.REALIGNMENT_OUT_MOVING:
                    self.sub_state = SubStates.REALIGNMENT_WAITING_ITEM
                
                if self.sub_state == SubStates.REALIGNMENT_IN_MOVING:
                    if data['extra'] == "rotate":
                        self.sub_state = SubStates.APPROACH_ITEM

                elif self.sub_state == SubStates.APPROACH_ITEM:
                    self.sub_state = SubStates.PICKING_UP
                elif self.sub_state == SubStates.REQUESTING_HOME_PATH:
                    print("Back at origin. Fetch complete.")
                    self.transition(States.NULL)
                return
            if self.state == States.MAPPING:
                if self.sub_state == SubStates.MOVING:
                    self.sub_state = SubStates.COMPLETE

    def navigation_node_cb(self, msg):
        if msg.data == "COMPLETE":
            self.sub_state = SubStates.COMPLETE
    
    def depth_cb(self, msg): #mayhaps consolidate all these under 1 topic to prevent clutter.
        data = json.loads(msg.data)
        if data['header'] == "depth_reading":
            self.detected_angle = data.get('angle_deg')
            self.detected_distance = data.get('dist_m')

    # ====== UTILS (Original methods) ====== #
    def get_robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform("map", "base_link", rospy.Time(0), rospy.Duration(0.1))
            x, y = t.transform.translation.x, t.transform.translation.y
            q = t.transform.rotation
            (_, _, yaw) = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
            return x, y, yaw
        except: return None

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
            status = "\rMain State: %-10s | Sub-State: %-10s\033[K" % (self.state.name, self.sub_state.name)
            sys.stdout.write(status)
            sys.stdout.flush()
            x, y, _ = self.get_robot_pose()
            self.publish_marker(x,y)

            if self.state == States.IDLE:    self.manage_idle()
            elif self.state == States.MAPPING:  self.manage_mapping()
            elif self.state == States.FETCHING: self.manage_fetching()
            self.rate.sleep()

    def manage_idle(self):
        if self.f_mapping_complete != True:
            self.transition(States.MAPPING)
        return

    def manage_mapping(self):
            # Initial entry: move to requesting data
            if self.sub_state == SubStates.READY:
                self.sub_state = SubStates.REQUESTING

                #clear goal_path for next request.
                self.goal_path = None
                self.rotate_target_msg = None

            # Logic for requesting and receiving paths/commands
            elif self.sub_state == SubStates.REQUESTING:
                if not self.request_sent:
                    msg = String()
                    msg.data = json.dumps({
                        "header": "pathing",
                        "command": "frontier",
                    })
                    self.global_request.publish(msg)
                    self.request_sent = True
                    self.start_time = rospy.get_time()
                    self.sub_state = SubStates.WAITING_PATH_RESPONSE

            if self.sub_state == SubStates.WAITING_PATH_RESPONSE:
                if self.received and self.received.get("header") == "map": #NOTE: additionally perform a timestamp check in case its an old piece of data.
                    #TODO: add a timeout
                    recv_cmd = self.received.get("command")
                    if self.received_path and recv_cmd == "path":
                        self.goal_path = self.received_path
                        self.received = None
                        self.request_sent = False
                        self.sub_state = SubStates.MOVING
                    # Handle rotation command   
                    if recv_cmd == "rotate":
                        self.rotate_target_msg = self.prepare_flip()
                        self.received = None
                        self.request_sent = False
                        self.sub_state = SubStates.MOVING
                    if recv_cmd == "complete":
                        self.f_mapping_complete = True
                        self.received = None
                        self.request_sent = False
                        self.sub_state = SubStates.COMPLETE
                pass

            elif self.sub_state == SubStates.MOVING:
                # Execute the active task
                if self.goal_path:
                    msg = String()
                    msg.data = json.dumps({
                        "header": "movement",
                        "command": "follow_path",
                    })
                    self.global_request.publish(msg)
                    self.global_path.publish(self.goal_path)
                elif self.rotate_target_msg:
                    msg = String()
                    msg.data = json.dumps({
                        "header": "movement",
                        "command": "rotate",
                        "angle": 180
                    })
                    self.global_request.publish(msg)

            # Logic while the robot is physically in motion
            elif self.sub_state == SubStates.COMPLETE:
                self.goal_path = None
                self.rotate_target_msg = None
                self.transition(States.IDLE, SubStates.READY)

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
                    self.sub_state = SubStates.READING_DEPTH
            return
        

        if self.sub_state == SubStates.READING_DEPTH:
            msg = String()
            msg.data = json.dumps({
                "header": "depth_node",
                "command": "check_depth",
                "x_start": self.last_cv_detection['x_start'],
                "x_len": self.last_cv_detection['x_len'],
                "y_start": self.last_cv_detection['y_start'],
                "y_len": self.last_cv_detection['y_len'],
            })
            self.global_request.publish(msg)

            if self.detected_distance is None:
                return
            elif self.detected_distance < -1:
                return
            else:
                self.sub_state = SubStates.DEPTH_READING_AVAILABLE
                                    
        # if self.sub_state == SubStates.DEPTH_READING_UNAVAILABLE:
        #     msg = String()
        #     msg.data = json.dumps({
        #         "header": "depth_node",
        #         "command": "check_depth",
        #         "x_start": self.last_cv_detection['x_start'],
        #         "x_len": self.last_cv_detection['x_len'],
        #         "y_start": self.last_cv_detection['y_start'],
        #         "y_len": self.last_cv_detection['y_len'],
        #     })
        #     self.global_request.publish(msg)

        #     #TODO: potentially need to back up then request for depth reading again.

        if self.sub_state == SubStates.DEPTH_READING_AVAILABLE:
            msg = String()
        
            dist = self.detected_distance
            pose = self.get_robot_pose()
            angle    = self.detected_angle
            #so now we have all the information required to get the object's position in the world
            print(angle, dist, pose)

            #ok we know the definite position here, update it in the map.
            self.target_object_transform = utils.project_local_to_world(pose, angle, dist)#robot forward, object angle, depth distance
            
            if self.target_object_transform is None:
                return

            obj_x, obj_y = self.target_object_transform
            self.publish_marker(obj_x, obj_y, 1 , color="blue")

            #wipe detected distance for next detection.
            self.detected_distance = None

            #in case somewhere got a leftover path.
            self.received_path = None

            #now given the map, determine a safe spot (Lowest Cost) to position within the radius of target_object_transform (use waypoint navigator and rename the node to something else).
            msg.data = json.dumps({
                "header": "pathing",
                "command": "object",
                "x": obj_x,
                "y": obj_y
            })
            self.global_request.publish(msg)

            self.sub_state = SubStates.WAITING_PATH_RESPONSE

        if self.sub_state == SubStates.WAITING_PATH_RESPONSE:
            if self.received_path is not None:
                obj_x, obj_y = self.target_object_transform
                msg = String()
                msg.data = json.dumps({
                    "header": "movement",
                    "command": "follow_path",
                    "extra": "face_coordinates",
                    "x": obj_x,
                    "y": obj_y
                })
                self.global_request.publish(msg)
                self.global_path.publish(self.received_path)
                self.sub_state = SubStates.MOVING

        
        if self.sub_state == SubStates.MOVING:
            pass

        if self.sub_state == SubStates.REALIGNMENT_OUT:
            rospy.sleep(1.5)
            msg = String()
            msg.data = json.dumps({
                "header": "movement",
                "command": "rotate",
                "angle": 12
            })
            self.global_request.publish(msg)
            self.sub_state = SubStates.REALIGNMENT_OUT_MOVING

        if self.sub_state == SubStates.REALIGNMENT_OUT_MOVING: #waiting phase
            pass

        if self.sub_state == SubStates.REALIGNMENT_WAITING_ITEM:
            # msg = String()
            # msg.data = json.dumps({
            #     "header": "arm",
            #     "command": "extend"
            # })
            # self.global_request.publish(msg)
            rospy.sleep(2.5)
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
                    rospy.sleep(0.5)
                    self.detected_distance = None
                    self.sub_state = SubStates.REALIGNMENT_WAITING_DEPTH
                    return
                
            self.transition(States.MAPPING)   
            pass

        if self.sub_state == SubStates.REALIGNMENT_WAITING_DEPTH:
            msg = String()  
            msg.data = json.dumps({
                "header": "depth_node",
                "command": "check_depth",
                "x_start": self.last_cv_detection['x_start'],
                "x_len": self.last_cv_detection['x_len'],
                "y_start": self.last_cv_detection['y_start'],
                "y_len": self.last_cv_detection['y_len'],
            })
            self.global_request.publish(msg)
            print("Detected Distance:", self.detected_distance)

            if self.detected_distance is None:
                return
            elif self.detected_distance < -1:
                return
            else:
                #update target object pose.
                dist = self.detected_distance
                pose = self.get_robot_pose()
                angle    = self.detected_angle
                #so now we have all the information required to get the object's position in the world
                print(angle, dist, pose)

                #ok we know the definite position here, update it in the map.
                self.target_object_transform = utils.project_local_to_world(pose, angle, dist)#robot forward, object angle, depth distance
                tx, ty = self.target_object_transform
                self.publish_marker(tx, ty, 1 , color="blue")
                print("Target Object Transform: %s" % str(self.target_object_transform))
                self.sub_state = SubStates.REALIGNMENT_IN

        if self.sub_state == SubStates.REALIGNMENT_IN:
            # msg = String()
            # msg.data = json.dumps({
            #     "header": "arm",
            #     "command": "tuck"
            # })
            # self.global_request.publish(msg)
            rospy.sleep(2.5)
            #calculate rotation from current rotation to face the object, then just call a naive rotate in place command, then transition to next sub-state to move forward a bit to get into the ideal position for pickup.
            rx, ry, ryaw = self.get_robot_pose()
            tx, ty = self.target_object_transform
            dx = tx - rx
            dy = ty - ry
            target_yaw = math.atan2(dy, dx)
            yaw_error = target_yaw - ryaw
            yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi
            yaw_error_deg = math.degrees(yaw_error)
            msg = String()
            msg.data = json.dumps({
                "header": "movement",
                "command": "rotate",
                "angle": yaw_error_deg
            })
            self.global_request.publish(msg)
            self.sub_state = SubStates.REALIGNMENT_IN_MOVING
            rospy.sleep(1.5)
            pass

        if self.sub_state == SubStates.REALIGNMENT_IN_MOVING: #waiting phase
            pass

            
        if self.sub_state == SubStates.APPROACH_ITEM:
            print("Approaching item...")
            rospy.sleep(5)
            obj_x, obj_y = self.target_object_transform
            msg = String()
            msg.data = json.dumps({
                "header": "movement",
                "command": "approach",
                "extra": "face_coordinates",
                "stopping_distance": 0.395,
                "x": obj_x,
                "y": obj_y
            })
            #NOTE: potentially just pass in the coords of the object and let the movement controller handle it due to lower latency, but state transitions might be abit more annoying and i cba rn.
            self.global_request.publish(msg)



        if self.sub_state == SubStates.PICKING_UP:
            rospy.sleep(1.5)
            #object should be perfectly positioned in front of the robot now, so just perform standard FK based grab command
            msg = String()
            msg.data = json.dumps({
                "header": "arm",
                "command": "grab"
            })
            self.global_request.publish(msg)
            rospy.sleep(3.0)
            self.sub_state = SubStates.REQUESTING_HOME_PATH

        #TIME TO HUI JIA LIAO
        if self.sub_state == SubStates.REQUESTING_HOME_PATH:
            msg = String()
            msg.data = json.dumps({
                "header": "pathing",
                "command": "waypoint",
                "x": 0,
                "y": 0
            })
            self.received = None
            self.global_request.publish(msg)
            self.sub_state = SubStates.WAITING_HOME_PATH_RESPONSE
            
            
        if self.sub_state == SubStates.WAITING_HOME_PATH_RESPONSE:
            if self.received and self.received.get("header") == "map": #NOTE: additionally perform a timestamp check in case its an old piece of data.
                self.goal_path = self.received_path
                self.received = None
                self.request_sent = False
                self.sub_state = SubStates.MOVING_HOME
        
        if self.sub_state == SubStates.MOVING_HOME:
            obj_x, obj_y = self.target_object_transform
            msg = String()
            msg.data = json.dumps({
                #TODO
            })
            self.global_request.publish(msg)
            self.global_path.publish(self.received_path)
            pass

        if self.sub_state == SubStates.DROPPING_ITEM:
            #object should be perfectly positioned in front of the robot now, so just perform standard FK based grab command
            msg = String()
            msg.data = json.dumps({
                "header": "arm",
                "command": "drop"
            })
            self.global_request.publish(msg)
            rospy.sleep(3.0)
            self.sub_state = SubStates.COMPLETE
        
        if self.sub_state == SubStates.COMPLETE:
            self.transition(States.IDLE, SubStates.READY)

        


        
        

if __name__ == "__main__":
    ctrl = Controller()
    ctrl.run()