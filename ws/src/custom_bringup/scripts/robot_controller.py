#!/usr/bin/env python

import rospy
from enum import Enum
from std_msgs.msg import Empty, String
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import json
import tf2_ros
import tf
import math

class NavStates(Enum):
    NULL = 0
    MOVING = 1
    INTERRUPTED = 2
    COMPLETE = 3
    FAILED = 4


class States(Enum):
    NULL = 0
    INIT = 1
    IDLE = 2
    REQUEST_FRONTIER_PATH = 3
    NAVIGATE = 4
    ROTATE = 5


class State:
    def __init__(self, state = States.NULL, info = ""):
        self.state = state
        self.info = info
        pass

class Controller:
    def __init__(self):

        rospy.init_node("python_controller")
        self.recalib_pub = rospy.Publisher("/recalib_frontiers", Empty, queue_size=1)
        self.state_pub   = rospy.Publisher("/controller_state", String, queue_size=1)
        self.global_request = rospy.Publisher("/controller/global", String, queue_size=1)
        self.rotate_pose_pub = rospy.Publisher("/rotate_target_pose", PoseStamped, queue_size=1)
        self.global_exploration_path = rospy.Publisher("/global_exploration_path", Path, queue_size=1)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        self.fontier_node_sub = rospy.Subscriber("/frontier_node_reply", String, self.frontier_node_cb)
        self.frontier_node_path_sub = rospy.Subscriber("/frontier_node_path", Path, self.frontier_node_path_cb)
        self.navigation_node_sub = rospy.Subscriber("/navigation_node_reply", String, self.navigation_node_cb)
        self.pc_node_sub = rospy.Subscriber("/pc_node_reply", String, self.pc_node_cb)
        #self.pure_pursuit_sub = rospy.Subscriber("/pure_pursuit_message", String, self.pure_pursuit_cb)

        self.request_sent = False
        self.received = False
        self.received_path = False
        self.nav_state = NavStates.NULL
        self.request_timeout = 10

        self.rotate_target_yaw = None

        self.prev_state = None
        self.init_complete = False
        
        self.state = State(state=States.INIT)
        self.rate = rospy.Rate(5)

        print("Initialization Complete, Node is Ready!")
        pass

    def frontier_node_cb(self, msg):
        self.received = json.loads(msg.data)
        pass

    def frontier_node_path_cb(self, msg):
        self.received_path = msg
        pass

    def navigation_node_cb(self, msg):
        if msg == "COMPLETE":
            self.nav_state = NavStates.COMPLETE
        pass

    def pc_node_cb(self, msg):
        pass

    # ====== UTIL ====== # 
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
    
    def wrap_angle(self, a):
        return math.atan2(math.sin(a), math.cos(a))

    # ====== FSM ====== # 

    def interrupt(self, clear = False):
        self.global_request.publish("interrupt")
        pass

    def transition(self, nxt_state):
        if nxt_state != self.state.state:
            print("Transitioning", self.state.state.name, ">", nxt_state.name)
            self.state.state = nxt_state

    def state_init(self):
        self.init_complete = True
        if self.init_complete == True:
            self.transition(States.REQUEST_FRONTIER_PATH)
        pass

    def state_request_frontier_path(self):
        if not self.request_sent:
            msg = String()
            msg.data = "request_frontier"
            print("Trying to Publish: Request Frontier")
            self.global_request.publish(msg)
            self.request_sent = True
            self.start_time = rospy.get_time()

        elif self.received:
            if self.received["cmd"] == "rotate":
                pose = self.get_robot_pose()
                if pose is None: 
                    return
                
                _, _, yaw = pose
                self.rotate_target_yaw = self.wrap_angle(yaw + math.pi)
                self.transition(States.ROTATE)
                self.request_sent = False
                self.received = False
                pass
            elif self.received["cmd"] == "path": 
                print("Path Received")
                if self.received_path and len(self.received_path.poses) > 0:
                    print("Transitioning")
                    self.goal_path = self.received_path
                    self.transition(States.NAVIGATE)
                    self.request_sent = False
                    self.received = False
                pass

        elif (rospy.get_time() - self.start_time) > self.request_timeout:
            print("Request Timeout, Frontier Node Non-Responsive")
            self.request_sent = False
            self.transition(States.IDLE)

    def state_navigate(self):
        if self.nav_state == NavStates.NULL:
            self.nav_state = NavStates.MOVING
        
        elif self.nav_state == NavStates.MOVING:
            self.global_exploration_path.publish(self.goal_path)
            pass

        elif self.nav_state == NavStates.COMPLETE:
            self.nav_state = NavStates.NULL
            self.goal_path = None
            self.transition(States.IDLE)

    def state_rotate(self):
        if self.nav_state == NavStates.NULL:
            self.nav_state = NavStates.MOVING
        
        elif self.nav_state == NavStates.MOVING:
            if self.rotate_target_yaw is not None:
                self.rotate_pose_pub.publish(self.rotate_target_yaw)
            pass

        elif self.nav_state == NavStates.COMPLETE:
            self.nav_state = NavStates.NULL
            self.goal_path = None
            self.transition(States.IDLE)
        pass
        

    def run(self):
        rospy.sleep(1.0)
        while not rospy.is_shutdown():
            print(self.state.state.name)
            if self.state.state == States.INIT:
                self.state_init()
                pass

            elif self.state.state == States.REQUEST_FRONTIER_PATH:
                self.state_request_frontier_path()
                pass
            
            elif self.state.state == States.NAVIGATE:
                self.state_navigate()
                pass

            elif self.state.state == States.ROTATE:
                self.state_rotate()
                pass

            self.rate.sleep()


if __name__ == "__main__":
    ctrl = Controller()
    ctrl.run()

    