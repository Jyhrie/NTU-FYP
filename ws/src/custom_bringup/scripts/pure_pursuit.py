#!/usr/bin/env python

import rospy
import math
import tf
import tf2_ros

from nav_msgs.msg import Path
from geometry_msgs.msg import Twist
from std_msgs.msg import String


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
        self.pub = rospy.Publisher("/frontier_node_message", String, queue_size=1)

        rospy.Subscriber("/global_exploration_path", Path, self.path_cb)
        

        self.rate = rospy.Rate(30)

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

            print("Robot Pose: ", x, y, yaw)

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

    def find_lookahead_point(self, x, y, start_idx, lookahead_steps = 7):


        target_idx = min(start_idx + lookahead_steps, len(self.path) - 1)
        p = self.path[target_idx].pose.position
        return p.x, p.y

    # -------------------------------------------------

    def goal_reached(self, x, y, yaw):
        """
        Returns True if the robot is close enough to the goal and facing the goal direction.
        """
        if self.path is None or len(self.path) == 0:
            return False

        # get the final goal position
        goal = self.path[-1].pose.position
        dx = goal.x - x
        dy = goal.y - y

        # straight-line distance to goal
        dist_to_goal = math.hypot(dx, dy)

        # direction from robot to goal
        goal_direction = math.atan2(dy, dx)

        # difference between robot heading and goal direction, wrapped to [-pi, pi]
        heading_error = math.atan2(math.sin(goal_direction - yaw),
                                math.cos(goal_direction - yaw))

        # thresholds
        distance_tol = self.goal_tol       # e.g., 0.3 m
        yaw_tol = 0.1                      # e.g., ~6 degrees

        return dist_to_goal <= distance_tol and abs(heading_error) <= yaw_tol

    # -------------------------------------------------

    def stop_robot(self):
        self.cmd_pub.publish(Twist())

    # -------------------------------------------------

    def run(self):
        rospy.loginfo("[PP] Pure Pursuit controller started")

        while not rospy.is_shutdown():

            # ---------------- Check path ----------------
            if self.path is None or len(self.path) == 0:
                self.rate.sleep()
                continue

            # ---------------- Get robot pose ----------------
            pose = self.get_robot_pose()
            if pose is None:
                self.rate.sleep()
                continue

            x, y, yaw = pose

            # ---------------- Nearest path point ----------------
            nearest = self.find_nearest_index(x, y)

            # ---------------- Lookahead point ----------------
            look = self.find_lookahead_point(x, y, nearest)
            if look is None:
                rospy.loginfo("No lookahead point found")
                self.rate.sleep()
                continue
            lx, ly = look

            # ---------------- Transform to robot frame ----------------
            dx = lx - x
            dy = ly - y
            x_r = math.cos(-yaw)*dx - math.sin(-yaw)*dy
            y_r = math.sin(-yaw)*dx + math.cos(-yaw)*dy

            distance = math.hypot(x_r, y_r)
            lookahead_angle = math.atan2(y_r, x_r)  # relative angle to lookahead

            # ---------------- Sharp turn handling ----------------
            if abs(lookahead_angle) > 2.36:  # 135 deg
                rospy.loginfo("[PP] Sharp turn detected, rotating in place")
                while abs(lookahead_angle) > 0.05 and not rospy.is_shutdown():
                    cmd = Twist()
                    cmd.linear.x = 0  # stop forward motion
                    cmd.angular.z = max(min(lookahead_angle * 1.0, 0.3), -0.3)
                    self.cmd_pub.publish(cmd)
                    self.rate.sleep()

                    # update pose and angle
                    pose = self.get_robot_pose()
                    if pose is None:
                        continue
                    x, y, yaw = pose
                    dx = lx - x
                    dy = ly - y
                    x_r = math.cos(-yaw)*dx - math.sin(-yaw)*dy
                    y_r = math.sin(-yaw)*dx + math.cos(-yaw)*dy
                    lookahead_angle = math.atan2(y_r, x_r)

                rospy.loginfo("[PP] Turn complete")
                self.stop_robot()
                self.path = None
                self.pub.publish("RESCAN")
                continue

            # ---------------- Pure Pursuit motion ----------------
            else:
                # Pure Pursuit curvature
                kappa = 2.0 * y_r / (self.lookahead**2)

                # angular velocity
                omega = max(min(kappa * self.linear_vel, 0.3), -0.3)

                # scale linear speed based on angular velocity
                max_linear = self.linear_vel
                min_linear = 0.05
                scaling = 1.0 - min(abs(omega) / 0.3, 1.0)
                v = min_linear + scaling * (max_linear - min_linear)

                # publish command
                cmd = Twist()
                cmd.linear.x = v
                cmd.angular.z = omega
                self.cmd_pub.publish(cmd)

            # ---------------- Goal check ----------------
            if self.goal_reached(x, y, yaw):
                rospy.loginfo("[PP] Goal reached!")
                self.stop_robot()
                self.path = None
                self.pub.publish("END")

            self.rate.sleep()


if __name__ == "__main__":
    PurePursuitController().run()
