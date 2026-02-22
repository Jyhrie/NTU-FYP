#!/usr/bin/env python

import json

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
    ALIGN = 4
    APPROACH = 5

class PurePursuitController:

    def __init__(self):
        print("Initializing Pure Pursuit Controller...")

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
        self.node_topic = rospy.Publisher("/movement_controller_message", String, queue_size=10)
        
        rospy.Subscriber("/controller/global", String, self.controller_cb)
        rospy.Subscriber("/rotate_target_pose", PoseStamped, self.rotate_pose_cb)
        rospy.Subscriber("/global_exploration_path", Path, self.path_cb)

        self.state = MovementState.IDLE
        self.rotate_target_pose = None
        
        self.align_error = None
        self.end_facing_target = None
        self.align_target_reached_time = None
        
        self.rate = rospy.Rate(15)

    # -------------------------------------------------

    def controller_cb(self, msg):
        try:
            # Try to parse as JSON first
            data = json.loads(msg.data)
            header = data.get("header")
            
            if header == 'align_with_item':
                self.state = MovementState.ALIGN
                # Store the relative error in degrees
                self.align_error = data.get("data", {}).get("relative_angle", 0.0)
                rospy.loginfo("[PP] Aligning with error: %.2f", self.align_error)
            elif header == 'navigate':
                self.state = MovementState.MOVE
                self.end_facing_target = (data.get("end_face_pt_x"), data.get("end_face_pt_y"))
            elif header == 'stop_movement':
                self.state = MovementState.IDLE
                self.stop_robot()
            elif header == 'approach':
                # --- APPROACH LOGIC ---
                self.state = MovementState.APPROACH
                # We reuse align_error for steering during the approach
                self.align_error = data.get("data", {}).get("relative_angle", 0.0)
                # Store the speed sent by the high-level controller
                self.approach_speed = data.get("data", {}).get("linear_speed", 0.1)
                rospy.loginfo("[PP] Approaching: Angle %.2f | Speed %.2f", 
                               self.align_error, self.approach_speed)
                
        except ValueError:
            # If not JSON, handle as a plain string
            if msg.data == 'navigate':
                self.state = MovementState.MOVE
            elif msg.data == 'rotate':
                self.state = MovementState.ROTATE
            elif msg.data == 'interrupt':
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

    def find_lookahead_point(self, x, y, start_idx, lookahead_steps = 6):


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
        dist_tol_near = 0.07   # within 0.5 m AND aligned
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

    def get_approach(self):
        # We stop when the object is large enough or close enough 
        # (You can also trigger this exit from your main state machine)
        
        # 1. Steering Logic (Same as ALIGN)
        error_rad = math.radians(self.align_error)
        p_gain = 1.2
        
        cmd = Twist()
        # 2. Linear Movement
        # We use a fixed slow speed to ensure the camera can keep up
        cmd.linear.x = 0.12 
        
        # 3. Angular Correction (to keep it centered)
        angular_z = error_rad * p_gain
        
        # Deadband handling for the Transbot motors
        MIN_ROT_VEL = 0.1
        if abs(angular_z) < MIN_ROT_VEL and abs(error_rad) > math.radians(1.0):
            angular_z = MIN_ROT_VEL if angular_z > 0 else -MIN_ROT_VEL
            
        cmd.angular.z = max(min(angular_z, 0.4), -0.4)
        self.cmd_pub.publish(cmd)

    def get_rot(self):
        pose = self.get_robot_pose()
        if pose is None:
            return None
        
        if self.rotate_target_pose is None:
            rospy.logwarn_throttle(2, "[PP] Rotate state active but no target pose found")
            self.state = MovementState.IDLE # Safety fallback
            return None

        _, _, yaw = pose

        # --- extract target yaw from PoseStamped ---
        q = self.rotate_target_pose.pose.orientation
        (_, _, target_yaw) = tf.transformations.euler_from_quaternion(
            [q.x, q.y, q.z, q.w]
        )

        error = self.normalize_angle(target_yaw - yaw)

        # ---------------- Check Completion ----------------
        if abs(error) < 0.05:
            rospy.loginfo("[PP] Rotate target reached. Returning to IDLE.")
            self.stop_robot()
            
            self.state = MovementState.IDLE
            self.rotate_target_pose = None
            self.node_topic.publish("done")
            return None
        
        # ---------------- Proportional Control ----------------
        cmd = Twist()
        cmd.linear.x = 0
        cmd.angular.z = max(min(error * 1.2, 0.4), -0.4)

        self.cmd_pub.publish(cmd)
        
    def get_align(self):
        # 0.5 degrees in radians
        TOLERANCE = math.radians(0.5)
        HOLD_TIME = 1.0  # seconds
        
        # Current error from the JSON message
        error_rad = math.radians(self.align_error)

        # 1. Check if we are within the center zone
        if abs(error_rad) < TOLERANCE:
            if self.align_target_reached_time is None:
                # Start the clock!
                self.align_target_reached_time = rospy.get_time()
                rospy.loginfo("[PP] Centered. Holding for 1s...")

            # Check if we have held the position long enough
            elapsed = rospy.get_time() - self.align_target_reached_time
            if elapsed >= HOLD_TIME:
                rospy.loginfo("[PP] Alignment Stable. Done.")
                self.stop_robot()
                self.state = MovementState.IDLE
                self.align_target_reached_time = None # Reset for next use
                self.node_topic.publish("done")
                return
            
            # While holding, we stop the motors or apply very small braking
            self.stop_robot()
            return

        else:
            # We are outside the tolerance, reset the timer
            self.align_target_reached_time = None

        # 2. Proportional Control (Only runs if not in the "Hold" phase)
        cmd = Twist()
        p_gain = 1.2
        angular_z = error_rad * p_gain

        # 3. Handle Deadband (Transbot friction)
        MIN_VEL = 0.08
        if abs(angular_z) < MIN_VEL:
            angular_z = MIN_VEL if angular_z > 0 else -MIN_VEL

        # 4. Cap the speed for safety
        cmd.angular.z = max(min(angular_z, 0.25), -0.25)
        
        self.cmd_pub.publish(cmd)

    def get_pp(self): #its called pp because pure pursuit.
        pose = self.get_robot_pose()
        if pose is None or self.path is None:
            return

        x, y, yaw = pose

        if self.goal_reached(x, y, yaw):
            rospy.loginfo("[PP] Goal location reached!")
            # Check if we have a specific point to face
            if self.end_facing_target and all(v is not None for v in self.end_facing_target):
                tx, ty = self.end_facing_target
                
                # Calculate the angle to face the target point from current position
                angle_to_target = math.atan2(ty - y, tx - x)
                
                # Create a temporary PoseStamped to reuse get_rot() logic
                target_pose = PoseStamped()
                target_pose.header.frame_id = "map"
                q = tf.transformations.quaternion_from_euler(0, 0, angle_to_target)
                target_pose.pose.orientation.x = q[0]
                target_pose.pose.orientation.y = q[1]
                target_pose.pose.orientation.z = q[2]
                target_pose.pose.orientation.w = q[3]
                
                self.rotate_target_pose = target_pose
                self.state = MovementState.ROTATE
                rospy.loginfo("[PP] Transitioning to ROTATE to face target point")
            else:
                self.stop_robot()
                self.node_topic.publish("done")
                self.state = MovementState.COMPLETE
            return

        # 2. Find lookahead point
        nearest = self.find_nearest_index(x, y)
        lx, ly = self.find_lookahead_point(x, y, nearest)
        
        # 3. Transform lookahead point to robot frame
        dx = lx - x
        dy = ly - y
        
        # Rotation matrix to local coordinates
        x_r = math.cos(-yaw) * dx - math.sin(-yaw) * dy
        y_r = math.sin(-yaw) * dx + math.cos(-yaw) * dy

        # 4. Heading Check (The Precaution)
        # Calculate angle to the lookahead point relative to robot heading
        lookahead_angle = math.atan2(y_r, x_r)

        # If lookahead is more than 90 degrees away (opposite direction)
        if abs(lookahead_angle) > (math.pi / 2.0):
            rospy.loginfo_throttle(1, "[PP] Lookahead point behind robot - Rotating in place")
            cmd = Twist()
            cmd.linear.x = 0.0
            # Simple proportional control to face the point
            cmd.angular.z = max(min(lookahead_angle * 1.5, 0.5), -0.5)
            self.cmd_pub.publish(cmd)
            return

        # 5. Normal Pure Pursuit (if lookahead is in front)
        # Curvature formula: kappa = 2*y / L^2
        L_sq = x_r**2 + y_r**2
        if L_sq < 0.01: L_sq = 0.01 # Prevent division by zero
        
        kappa = (2.0 * y_r) / L_sq

        # Calculate velocities
        cmd = Twist()
        cmd.angular.z = max(min(kappa * self.linear_vel, 0.4), -0.4)
        
        # Scale linear velocity: slow down if turning sharply
        scaling = 1.0 - min(abs(cmd.angular.z) / 0.4, 0.8)
        cmd.linear.x = self.linear_vel * scaling
        
        self.cmd_pub.publish(cmd)

    def run(self):
        rospy.loginfo("[PP] Pure Pursuit controller started")
        while not rospy.is_shutdown():
            if self.state == MovementState.ROTATE:
                self.get_rot()
                pass
            if self.state == MovementState.MOVE:
                self.get_pp()
            elif self.state == MovementState.ALIGN:
                self.get_align() # Process the alignment P-loop
            elif self.state == MovementState.APPROACH:
                self.get_approach() # Process the approach logic

            self.rate.sleep()


if __name__ == "__main__":
    PurePursuitController().run()
