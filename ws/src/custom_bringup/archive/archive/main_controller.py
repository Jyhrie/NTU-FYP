#!/usr/bin/env python3
import rospy
from std_msgs.msg import Empty, String
from nav_msgs.msg import Path

class MainControllerFSM:

    def __init__(self):

        rospy.init_node("main_controller")

        # ---- Publishers ----
        self.recalib_pub = rospy.Publisher("/recalib_frontiers", Empty, queue_size=1)
        self.state_pub   = rospy.Publisher("/controller_state", String, queue_size=1)

        # ---- Subscribers ----
        self.fontier_node_sub = rospy.Subscriber("/frontier_node_message", String, self.frontier_node_cb)
        self.pure_pursuit_sub = rospy.Subscriber("/pure_pursuit_message", String, self.pure_pursuit_cb)

        # ---- FSM ----
        self.state = "INIT"
        self.rate = rospy.Rate(5)

    # ---------------- FSM helpers ----------------

    def set_state(self, new_state):
        if new_state != self.state:
            rospy.loginfo(f"[FSM] {self.state} -> {new_state}")
            self.state = new_state
            self.state_pub.publish(String(new_state))

    # ---------------- Callbacks ----------------

    def frontier_node_cb(self, msg):
        if self.state == "WAIT_FOR_PATH":
            
            self.set_state("EXECUTING")

    def pure_pursuit_cb(self, msg):
        if msg == "RESCAN":
            print("Pure Pursuit requested rescan")
        if self.state == "TRAVELLING":
            if msg == "CLOSE_TO_ENDPOINT":
                pass
        pass


    # ---------------- Main loop ----------------

    def run(self):
        rospy.sleep(1.0)
        self.set_state("REQUEST_FRONTIERS")

        while not rospy.is_shutdown():

            if self.state == "INIT":
                pass

            elif self.state == "WAITING_FOR_COMMAND":
                pass

            elif self.state == "TRAVELLING":
                pass

            elif self.state == "WAIT_FOR_PATH":
                pass

            elif self.state == "REQUEST_FRONTIERS":
                rospy.loginfo("[FSM] Requesting frontier recalculation")
                self.recalib_pub.publish(Empty())
                self.set_state("WAIT_FOR_PATH")

            self.rate.sleep()


if __name__ == "__main__":
    MainControllerFSM().run()
