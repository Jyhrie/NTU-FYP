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
from sensor_msgs.msg import LaserScan # Add this import at the top

class MovementState(Enum):
    IDLE = 0
    ROTATE = 1
    MOVE = 2
    COMPLETE = 3 
    ALIGN = 4
    APPROACH = 5
    PURSUIT = 6

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
        
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.scan_cb)
        rospy.Subscriber("/controller/global", String, self.controller_cb)
        rospy.Subscriber("/rotate_target_pose", PoseStamped, self.rotate_pose_cb)
        rospy.Subscriber("/global_path", Path, self.path_cb)

        self.state = MovementState.IDLE
        self.rotate_target_pose = None
        self.cached_transform = None
        
        self.align_error = None
        self.end_facing_target = None
        self.align_target_reached_time = None
        self.approach_start_pose = None

        self.initial_rotation_yaw = None
        self.face_coordinates = None
        self.rotate_angular = None

        self.side_nudge = 0
        self.safety_threshold = 0.4  # Distance in meters to start nudging
        
        self.rate = rospy.Rate(15)

    # -------------------------------------------------

    def scan_cb(self, msg):
            # Assuming 0 is front, 90 is Left, 270 is Right
            num_readings = len(msg.ranges)
            
            # Define index for 90 degrees (Left) and 270 degrees (Right)
            idx_90 = int(num_readings * 0.25)
            idx_270 = int(num_readings * 0.75)
            
            # 40-degree windows
            window = int(num_readings * (20.0 / 360.0))
            
            left_ranges = msg.ranges[idx_90 - window : idx_90 + window]
            right_ranges = msg.ranges[idx_270 - window : idx_270 + window]

            def get_valid_min(sector):
                # Filtering out 0.0 and infinity
                valid = [r for r in sector if msg.range_min < r < msg.range_max and r > 0.05]
                return min(valid) if valid else float('inf')

            min_l = get_valid_min(left_ranges)
            min_r = get_valid_min(right_ranges)

            nudge_velocity = 0.0
            gain = 1.5  # Strength of the nudge
            
            # --- Obstacle Detection Prints ---
            if min_l < self.safety_threshold:
                rospy.loginfo("[SCAN] OBSTACLE LEFT: dist=%.2f | Nudging Right", min_l)
                nudge_velocity -= (self.safety_threshold - min_l) * gain
                
            if min_r < self.safety_threshold:
                rospy.loginfo("[SCAN] OBSTACLE RIGHT: dist=%.2f | Nudging Left", min_r)
                nudge_velocity += (self.safety_threshold - min_r) * gain

            # Apply low-pass filter
            self.side_nudge = 0.5 * self.side_nudge + 0.5 * nudge_velocity

            # Optional: only print the final nudge if it's significant
            if abs(self.side_nudge) > 0.01:
                print("Total Nudge: {:.3f}".format(self.side_nudge))
        

    def controller_cb(self, msg):
        try:
            # Try to parse as JSON first
            data = json.loads(msg.data)
            # header = data.get(header)
            header = data.get("header")
            command = data.get("command")
            extra = data.get("extra")

            self.face_coordinates = None

            if header == "movement":
                if command == "follow_path":
                    self.state = MovementState.PURSUIT
                    if extra == "face_coordinates":
                        self.face_coordinates = (data.get("x"), data.get("y"))
                if command == "rotate":
                    self.state = MovementState.ROTATE
                    self.rotate_angular = data.get("angle")



                    
                    # self.end_facing_target = (data.get("end_face_pt_x"), data.get("end_face_pt_y"))

            # if header == 'interrupt':
            #     self.stop_robot()
            #     self.state = MovementState.IDLE

            # elif header == 'rotate':
            #     self.state = MovementState.ROTATE
            #     if self.cached_transform is not None:
            #         self.latched_target_yaw = self.cached_transform
            #     relative_angle = data.get("data", {}).get("relative_angle", 0.0)
            
            # elif header == "approach":
            #     self.state = MovementState.APPROACH
            #     self.linear_distance = data.get("data", {}).get("linear_dist", 0.07)

            # elif header == 'align_with_item':
            #     self.state = MovementState.ALIGN
            #     # Store the relative error in degrees
            #     self.align_error = data.get("data", {}).get("relative_angle", 0.0)
            #     rospy.loginfo("[PP] Aligning with error: %.2f", self.align_error)

            # elif header == 'navigate':
            #     self.state = MovementState.MOVE
            #     self.end_facing_target = (data.get("end_face_pt_x"), data.get("end_face_pt_y"))
                
            # elif header == 'stop_movement':
            #     self.state = MovementState.IDLE
            #     self.stop_robot()
            # elif header == 'approach':
            #         self.state = MovementState.APPROACH
            #         self.align_error = data.get("data", {}).get("relative_angle", 0.0)
            #         self.approach_speed = data.get("data", {}).get("linear_speed", 0.07)
                    
            #         # Extract the distance budget
            #         self.max_dist = data.get("data", {}).get("cached_distance", 0.0)
            #         # Reset tracking so get_approach knows to grab a new starting pose
            #         rospy.loginfo("[PP] Approaching blindly for %.2f meters", self.max_dist)
                
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
    def state_rotate(self):
        # 1. Get current pose
        pose = self.get_robot_pose()
        if pose is None:
            return

        _, _, current_yaw = pose

        # 2. THE CACHE CHECK: If no cached pose exists, this is the FIRST frame
        if self.initial_rotation_yaw is None:
            self.initial_rotation_yaw = current_yaw

        # 3. CALCULATE TARGET & ERROR
        # Target is the snapshot + our relative offset
        target_yaw = self.initial_rotation_yaw + math.radians(self.rotate_angular)
        
        # Calculate shortest path error
        error = math.atan2(math.sin(target_yaw - current_yaw), math.cos(target_yaw - current_yaw))

        # 4. TERMINATION CHECK
        if abs(error) < math.radians(1.5): # Within 1.5 degrees
            rospy.loginfo("[ROTATE] Target reached.")
            self.stop_robot()
            
            # RESET EVERYTHING for the next time this node is called
            self.initial_rotation_yaw = None
            self.rotate_angular = None 
            
            self.state = MovementState.COMPLETE
            self.node_topic.publish("done")
            return

        # 5. MOTOR COMMANDS (P-Control)
        cmd = Twist()
        cmd.linear.x = 0.0
        # Clamped angular velocity
        cmd.angular.z = max(min(error * 2.0, 0.6), -0.6)
        self.cmd_pub.publish(cmd)

    def start_relative_rotation(self, degrees):
        pose = self.get_robot_pose()
        if pose is None:
            return
        
        current_x, current_y, current_yaw = pose
        
        # 1. Convert relative degrees to absolute target yaw (radians)
        relative_radians = math.radians(degrees)
        target_yaw = current_yaw + relative_radians
        
        # 2. Wrap angle to keep it within [-pi, pi]
        target_yaw = math.atan2(math.sin(target_yaw), math.cos(target_yaw))
        
        # 3. Create PoseStamped for the ROTATE state to use
        target_pose = PoseStamped()
        target_pose.header.frame_id = "map"
        q = tf.transformations.quaternion_from_euler(0, 0, target_yaw)
        target_pose.pose.orientation.x = q[0]
        target_pose.pose.orientation.y = q[1]
        target_pose.pose.orientation.z = q[2]
        target_pose.pose.orientation.w = q[3]
        
        # 4. Set the state
        self.rotate_target_pose = target_pose
        self.state = MovementState.ROTATE
        
    def get_align(self):
        # 0.5 degrees in radians
        TOLERANCE = math.radians(1.0) # Slightly widened for live camera jitter
        
        if self.align_error is None:
            return

        # Current error from the JSON message (live stream)
        error_rad = math.radians(self.align_error)

        # ---------------- Check Completion ----------------
        if abs(error_rad) < TOLERANCE:
            if self.align_target_reached_time is None:
                self.align_target_reached_time = rospy.get_time()
                rospy.loginfo("[PP] Centered. Holding to confirm stability...")

            # Require the robot to stay centered for 1 second before calling it 'done'
            # This is crucial for the 0.8s camera delay to "catch up"
            elapsed = rospy.get_time() - self.align_target_reached_time
            if elapsed >= 1.2:
                rospy.loginfo("[PP] Alignment Stable. Done.")
                self.stop_robot()
                self.state = MovementState.IDLE
                self.align_target_reached_time = None 
                self.node_topic.publish("done")
                return
            
            self.stop_robot()
            return
        else:
            # We are outside the tolerance, reset the timer
            self.align_target_reached_time = None

        # ---------------- Proportional Control ----------------
        cmd = Twist()
        # LOW GAIN (0.5 - 0.7) is necessary because of the 0.8s delay
        p_gain = 0.6 
        angular_z = error_rad * p_gain

        # Handle Deadband
        MIN_VEL = 0.12
        if abs(angular_z) < MIN_VEL:
            angular_z = MIN_VEL if angular_z > 0 else -MIN_VEL

        # Cap the speed so we don't outrun the camera lag
        cmd.angular.z = max(min(angular_z, 0.25), -0.25)
        self.cmd_pub.publish(cmd)

    def get_approach(self):
            pose = self.get_robot_pose()
            if pose is None:
                return

            curr_x, curr_y, curr_yaw = pose

            # 1. INITIALIZE: Record start position AND starting heading
            if self.approach_start_pose is None:
                self.approach_start_pose = (curr_x, curr_y)
                # This is the "Lock" - we want to stay at this angle
                self.locked_heading = curr_yaw 
                rospy.loginfo("[PP] Approach Lock: Heading %.2f | Dist %.2f", 
                            self.locked_heading, self.max_dist)
                return

            # 2. TRACK DISTANCE
            start_x, start_y = self.approach_start_pose
            dist_traveled = math.hypot(curr_x - start_x, curr_y - start_y)

            # 3. CHECK COMPLETION
            if dist_traveled >= self.max_dist:
                rospy.loginfo("[PP] Distance reached. Final Heading Error: %.2f", 
                            self.normalize_angle(self.locked_heading - curr_yaw))
                self.stop_robot()
                self.state = MovementState.IDLE
                self.approach_start_pose = None 
                self.node_topic.publish("done")
                return

            # 4. HEADING LOCK (Correction Logic)
            # Calculate how much we have drifted from our starting heading
            yaw_error = self.normalize_angle(self.locked_heading - curr_yaw)
            
            cmd = Twist()
            cmd.linear.x = self.approach_speed
            
            # High gain for the gyro-lock because IMU data has zero delay
            # This will keep the robot's "nose" pointed exactly where it started
            cmd.angular.z = yaw_error * 8.0 

            self.cmd_pub.publish(cmd)

    def reset_align_vars(self):
        self.latched_align_target = None
        self.align_target_reached_time = None
        self.is_verifying = False

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

    def state_pursuit(self):
        pose = self.get_robot_pose()
        if pose is None or self.path is None:
            print("no pose or path")
            return

        x, y, yaw = pose

        if self.goal_reached(x, y, yaw):
            rospy.loginfo("[PP] Goal location reached!")
            # Check if we have a specific point to face
            if self.face_coordinates and all(v is not None for v in self.face_coordinates):
                print("face_coordinates exist, rotating")
                tx, ty = self.face_coordinates
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

        base_angular = kappa * self.linear_vel
        
        cmd = Twist()
        # Combine Pure Pursuit steering with the side-avoidance nudge
        cmd.angular.z = base_angular + self.side_nudge
        
        # Optional: Slow down linear speed if we are nudging hard (safety)
        nudge_intensity = abs(self.side_nudge)
        speed_factor = max(0.4, 1.0 - nudge_intensity) 
        cmd.linear.x = self.linear_vel * speed_factor
        
        self.cmd_pub.publish(cmd)
        pass


    def state_approach(self):
        pass


    def run(self):
        rospy.loginfo("[PP] Pure Pursuit controller started")
        while not rospy.is_shutdown():
            # print(self.state.name)
            if self.state == MovementState.PURSUIT:
                self.state_pursuit()
            if self.state == MovementState.ROTATE:
                self.state_rotate()
            #     pass
            # if self.state == MovementState.MOVE:
            #     self.get_pp()
            # elif self.state == MovementState.ALIGN:
            #     self.get_align() # Process the alignment P-loop
            # elif self.state == MovementState.APPROACH:
            #     self.get_approach() # Process the approach logic

            self.rate.sleep()


if __name__ == "__main__":
    PurePursuitController().run()
