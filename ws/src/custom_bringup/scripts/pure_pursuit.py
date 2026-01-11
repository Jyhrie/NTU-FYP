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

    def find_lookahead_point(self, x, y, start_idx):

        L2 = self.lookahead * self.lookahead

        for i in range(start_idx, len(self.path)):
            p = self.path[i].pose.position
            d2 = (p.x - x)**2 + (p.y - y)**2
            if d2 >= L2:
                return p.x, p.y

        # if no point is far enough, return the last point
        last = self.path[-1].pose.position
        return last.x, last.y

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
            nearest = self.find_nearest_index(x, y)

            look = self.find_lookahead_point(x, y, nearest)
            if look is None:
                rospy.loginfo("No lookahead point found")
            else:
                lx, ly = look
                #rospy.loginfo("Lookahead point: x=%.2f, y=%.2f", lx, ly)
            dx = lx - x
            dy = ly - y

            x_r = math.cos(-yaw)*dx - math.sin(-yaw)*dy
            y_r = math.sin(-yaw)*dx + math.cos(-yaw)*dy

            distance = math.hypot(x_r, y_r)            # straight-line distance
            lookahead_angle = math.atan2(y_r, x_r)     # relative rotation

            #rospy.loginfo("Lookahead relative distance: %.2f m, relative angle: %.2f deg", distance, math.degrees(lookahead_angle))

            if abs(lookahead_angle) >2.36:  # 135
                # too sharp of a turn, turn to said angle, then call main_controller to perform rescan.
                while abs(lookahead_angle) > 0.05 and not rospy.is_shutdown():
                    cmd = Twist()
                    cmd.linear.x = 0  # stop forward motion
                    # rotate toward lookahead
                    cmd.angular.z = max(min(lookahead_angle * 1, 0.3), -0.3)  # proportional controller
                    self.cmd_pub.publish(cmd)
                    
                    self.rate.sleep()
                    
                    # update robot pose and relative angle
                    pose = self.get_robot_pose()
                    if pose is None:
                        continue
                    x, y, yaw = pose
                    dx = lx - x
                    dy = ly - y
                    x_r = math.cos(-yaw)*dx - math.sin(-yaw)*dy
                    y_r = math.sin(-yaw)*dx + math.cos(-yaw)*dy
                    lookahead_angle = math.atan2(y_r, x_r)


                print("Turn Complete")
                self.stop_robot()
                self.path = None
                self.pub.publish("RESCAN") 

            elif abs(lookahead_angle) <= 2.36:  # 135 degrees
                # Pure Pursuit curvature
                kappa = 2.0 * y_r / (self.lookahead**2)

                # compute angular velocity
                omega = max(min(kappa * self.linear_vel, 0.3), -0.3)  # clamp omega

                # scale linear speed: higher omega -> slower linear
                max_linear = self.linear_vel
                min_linear = 0.05  # don't stop completely
                scaling = 1.0 - min(abs(omega) / 0.3, 1.0)  # simple linear scaling
                v = min_linear + scaling * (max_linear - min_linear)

                # create and publish command
                cmd = Twist()
                cmd.linear.x = v
                cmd.angular.z = omega
                self.cmd_pub.publish(cmd)

                
            if self.goal_reached(x, y):
                rospy.loginfo("[PP] Goal reached!")
                self.stop_robot()
                self.path = None
                self.pub.publish("END")

            
            self.rate.sleep()


if __name__ == "__main__":
    PurePursuitController().run()
