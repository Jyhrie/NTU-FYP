#!/usr/bin/env python3

import rospy
import math
import tf
import tf2_ros

from nav_msgs.msg import Path
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty


class PurePursuitController:

    def __init__(self):

        rospy.init_node("pure_pursuit_controller")

        # ---------------- Params ----------------
        self.lookahead = rospy.get_param("~lookahead", 0.6)   # meters
        self.linear_vel = rospy.get_param("~linear_vel", 0.25)
        self.goal_tol = rospy.get_param("~goal_tolerance", 0.3)

        # ---------------- State ----------------
        self.path = None
        self.current_idx = 0

        # ---------------- TF ----------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # ---------------- Pub/Sub ----------------
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.done_pub = rospy.Publisher("/path_done", Empty, queue_size=1)

        rospy.Subscriber("/global_exploration_path", Path, self.path_cb)

        self.rate = rospy.Rate(20)

    # -------------------------------------------------

    def path_cb(self, msg):
        if len(msg.poses) == 0:
            return

        self.path = msg.poses
        self.current_idx = 0
        rospy.loginfo("[PP] New path received with %d points", len(self.path))

    # -------------------------------------------------

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

            return x, y, yaw

        except:
            return None

    # -------------------------------------------------

    def find_nearest_index(self, x, y):

        best_i = self.current_idx
        best_d = float("inf")

        for i in range(self.current_idx, len(self.path)):
            p = self.path[i].pose.position
            d = (p.x - x)**2 + (p.y - y)**2
            if d < best_d:
                best_d = d
                best_i = i

        self.current_idx = best_i
        return best_i

    # -------------------------------------------------

    def find_lookahead_point(self, x, y, start_idx):

        L2 = self.lookahead * self.lookahead

        for i in range(start_idx, len(self.path)):
            p = self.path[i].pose.position
            d2 = (p.x - x)**2 + (p.y - y)**2
            if d2 >= L2:
                return p.x, p.y

        return None

    # -------------------------------------------------

    def goal_reached(self, x, y):

        goal = self.path[-1].pose.position
        dist = math.hypot(goal.x - x, goal.y - y)
        return dist < self.goal_tol

    # -------------------------------------------------

    def stop_robot(self):
        self.cmd_pub.publish(Twist())

    # -------------------------------------------------

    def run(self):

        rospy.loginfo("[PP] Pure Pursuit controller started")

        while not rospy.is_shutdown():

            if self.path is None:
                self.rate.sleep()
                continue

            pose = self.get_robot_pose()
            if pose is None:
                self.rate.sleep()
                continue

            x, y, yaw = pose

            # ----- Goal check -----
            if self.goal_reached(x, y):
                rospy.loginfo("[PP] Goal reached")
                self.stop_robot()
                self.done_pub.publish(Empty())
                self.path = None
                self.rate.sleep()
                continue

            # ----- Pure Pursuit -----
            nearest = self.find_nearest_index(x, y)
            look = self.find_lookahead_point(x, y, nearest)

            if look is None:
                self.stop_robot()
                self.rate.sleep()
                continue

            lx, ly = look

            # transform to robot frame
            dx = lx - x
            dy = ly - y

            x_r =  math.cos(-yaw)*dx - math.sin(-yaw)*dy
            y_r =  math.sin(-yaw)*dx + math.cos(-yaw)*dy

            # curvature
            kappa = 2.0 * y_r / (self.lookahead**2)

            # velocity commands (diff drive)
            v = self.linear_vel
            omega = v * kappa

            # clamp omega
            omega = max(min(omega, 1.5), -1.5)

            cmd = Twist()
            cmd.linear.x = v
            cmd.angular.z = omega

            self.cmd_pub.publish(cmd)

            self.rate.sleep()


if __name__ == "__main__":
    PurePursuitController().run()
