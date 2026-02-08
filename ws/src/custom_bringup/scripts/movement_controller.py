#!/usr/bin/env python

import rospy
import math
import tf
import tf2_ros
from enum import Enum

from nav_msgs.msg import Path
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

class MovementState(Enum):
    IDLE = 0
    ROTATE = 1
    MOVE = 2
    COMPLETE = 3 

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
        
        rospy.Subscriber("/controller/global", String, self.controller_cb)
        rospy.Subscriber("/rotate_target_pose", PoseStamped, self.rotate_pose_cb)
        rospy.Subscriber("/global_exploration_path", Path, self.path_cb)

        self.state = MovementState.IDLE
        self.rotate_target_pose = None
        
        self.rate = rospy.Rate(15)

    # -------------------------------------------------

    def controller_cb(self, msg):
        if msg.data == 'move':
            pass
        elif msg.data == 'rotate':
            self.state = MovementState.ROTATE
            pass
        elif msg.data == 'interrupt':
            pass
        pass

    def path_cb(self, msg):
        if len(msg.poses) == 0:
            return

        self.path = msg.poses
        self.current_idx = 0
        rospy.loginfo("[PP] New path received with %d points", len(self.path))

    def rotate_pose_cb(self, msg):
        print("Received Pose")
        self.rotate_target_pose = msg

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

    def find_lookahead_point(self, x, y, start_idx, lookahead_steps = 7):


        target_idx = min(start_idx + lookahead_steps, len(self.path) - 1)
        p = self.path[target_idx].pose.position
        return p.x, p.y

    # -------------------------------------------------

    def goal_reached(self, x, y, yaw):
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

        # heading error, wrapped to [-pi, pi]
        heading_error = math.atan2(math.sin(goal_direction - yaw),
                                math.cos(goal_direction - yaw))

        # thresholds
        dist_tol_far = 0.07    # within 0.2 m -> goal reached regardless of heading
        dist_tol_near = 0.1   # within 0.5 m AND aligned
        yaw_tol = 0.1         # radians (~11 degrees)

        # goal conditions
        if dist_to_goal <= dist_tol_far:
            return True
        elif dist_to_goal <= dist_tol_near and abs(heading_error) <= yaw_tol:
            return True
        else:
            return False

    def normalize_angle(self, a):
        return math.atan2(math.sin(a), math.cos(a))
    
    # -------------------------------------------------

    def stop_robot(self):
        self.cmd_pub.publish(Twist())

    # -------------------------------------------------

    def get_rot(self):
        pose = self.get_robot_pose()
        if pose is None:
            return None
        
        if self.rotate_target_pose is None:
            print("Exit Because No Target Pose")
            return None

        _, _, yaw = pose

        # --- extract target yaw from PoseStamped ---
        q = self.rotate_target_pose.pose.orientation
        (_, _, target_yaw) = tf.transformations.euler_from_quaternion(
            [q.x, q.y, q.z, q.w]
        )

        cmd = Twist()
        
        error = self.normalize_angle(target_yaw - yaw)

        if abs(error) < 0.05:
            rospy.loginfo("[PP] Rotate target reached")
            self.rotate_active = False
            self.rotate_target_pose = None
            return Twist()   # stop
        
        cmd.angular.z = max(min(error * 1.2, 0.4), -0.4)
        cmd.linear.x = 0

        return cmd


    def get_pp(self): #its called pp because pure pursuit.
        pose = self.get_robot_pose()
        if pose is None:
            self.rate.sleep()
            return

        x, y, yaw = pose

        nearest = self.find_nearest_index(x, y)

        look = self.find_lookahead_point(x, y, nearest)
        if look is None:
            rospy.loginfo("No lookahead point found")
            self.rate.sleep()
            return 
        
        lx, ly = look

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
                    return
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
            return

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

        pass

    def run(self):
        rospy.loginfo("[PP] Pure Pursuit controller started")
        while not rospy.is_shutdown():
            if self.state == MovementState.ROTATE:
                pub_res = self.get_rot()
                if pub_res is not None:
                    self.cmd_pub.publish(pub_res)
                pass
            self.rate.sleep()


if __name__ == "__main__":
    PurePursuitController().run()
