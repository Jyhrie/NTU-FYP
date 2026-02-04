#!/usr/bin/env python

import rospy
from enum import Enum
from std_msgs.msg import Empty, String
from nav_msgs.msg import Path

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
        self.global_exploration_path = rospy.Publisher("/global_exploration_path", String, queue_size=1)

        self.fontier_node_sub = rospy.Subscriber("/frontier_node_reply", Path, self.frontier_node_cb)
        self.navigation_node_sub = rospy.Subscriber("/navigation_node_reply", String, self.navigation_node_cb)
        self.pc_node_sub = rospy.Subscriber("/pc_node_reply", String, self.pc_node_cb)
        #self.pure_pursuit_sub = rospy.Subscriber("/pure_pursuit_message", String, self.pure_pursuit_cb)

        self.request_sent = False
        self.received = False
        self.nav_state = NavStates.NULL
        self.request_timeout = 2

        self.prev_state = None
        self.init_complete = False
        
        self.state = State(state=States.INIT)
        self.rate = rospy.Rate(5)

        print("Initialization Complete, Node is Ready!")
        pass

    def frontier_node_cb(self, msg):
        self.received = msg
        pass

    def navigation_node_cb(self, msg):
        if msg == "COMPLETE":
            self.nav_state = NavStates.COMPLETE
        pass

    def pc_node_cb(self, msg):
        pass

    def interrupt(self, clear = False):
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
        print("Trying to Publish!")
        self.global_request.publish("request_frontier")
        if not self.request_sent:
            self.request_sent = True
            self.start_time = rospy.get_time()

        elif self.received:
            self.request_sent = False
            self.received = False
            self.global_exploration_path.pub(self.received)
            self.transition(States.NAVIGATE)

        elif (rospy.get_time() - self.start_time) > self.request_timeout:
            print("Request Timeout, Frontier Node Non-Responsive")
            self.request_sent = False
            self.transition(States.IDLE)

    def state_navigate(self):
        if self.nav_state == NavStates.NULL:
            self.nav_state = NavStates.MOVING
        
        elif self.nav_state == NavStates.MOVING:
            pass

        elif self.nav_state == NavStates.COMPLETE:
            self.transition(States.IDLE)
        

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

            self.rate.sleep()


if __name__ == "__main__":
    ctrl = Controller()
    ctrl.run()

    