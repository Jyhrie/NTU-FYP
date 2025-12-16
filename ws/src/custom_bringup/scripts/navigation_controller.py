#!/usr/bin/env python3
import rospy
import math
import tf2_ros
import tf.transformations

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist


class NavigationController:
    def __init__(self):
        rospy.init_node("tf_wall_follow_controller")

        # -----------------------
        # Map data
        # -----------------------
        self.map_data = None
        self.map_width = 0
        self.map_height = 0
        self.map_res = 0.0
        self.map_origin_x = 0.0
        self.map_origin_y = 0.0

        # -----------------------
        # TF
        # -----------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # -----------------------
        # ROS interfaces
        # -----------------------
        rospy.Subscriber("/map", OccupancyGrid, self.map_cb)
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        rospy.loginfo("TF wall-follow controller initialized")

    # =======================
    # Callbacks
    # =======================
    def map_cb(self, msg):
        self.map_data = msg.data
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_res = msg.info.resolution
        self.map_origin_x = msg.info.origin.position.x
        self.map_origin_y = msg.info.origin.position.y

    # =======================
    # TF Pose
    # =======================
    def get_robot_pose_map(self):
        """
        Returns robot pose (x, y, yaw) in MAP frame
        """
        try:
            trans = self.tf_buffer.lookup_transform(
                "map",
                "base_link",
                rospy.Time(0),
                rospy.Duration(0.1)
            )
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None

        x = trans.transform.translation.x
        y = trans.transform.translation.y

        q = trans.transform.rotation
        yaw = tf.transformations.euler_from_quaternion(
            [q.x, q.y, q.z, q.w]
        )[2]

        return x, y, yaw

    # =======================
    # Coordinate transforms
    # =======================
    def world_to_map(self, x, y):
        mx = int((x - self.map_origin_x) / self.map_res)
        my = int((y - self.map_origin_y) / self.map_res)

        if mx < 0 or my < 0 or mx >= self.map_width or my >= self.map_height:
            return None
        return mx, my

    # =======================
    # Collision checking
    # =======================
    def is_blocked(self, robot_x, robot_y, robot_yaw, angle_offset, dist=0.4):
        """
        Checks if a direction relative to robot is occupied in the map
        """
        angle = robot_yaw + angle_offset
        tx = robot_x + math.cos(angle) * dist
        ty = robot_y + math.sin(angle) * dist

        cell = self.world_to_map(tx, ty)
        if cell is None:
            return True

        mx, my = cell
        idx = my * self.map_width + mx

        occ = self.map_data[idx]
        return occ > 50  # occupied

    # =======================
    # Wall following logic
    # =======================
    def wall_follow_step(self):
        if self.map_data is None:
            return

        pose = self.get_robot_pose_map()
        if pose is None:
            return

        x, y, yaw = pose

        front_blocked = self.is_blocked(x, y, yaw, 0.0)
        right_blocked = self.is_blocked(x, y, yaw, -math.pi / 2)

        cmd = Twist()

        # Right-hand wall following
        if not right_blocked:
            cmd.angular.z = -0.6      # turn right
        elif front_blocked:
            cmd.angular.z = 0.6       # turn left
        else:
            cmd.linear.x = 0.2        # move forward

        self.cmd_pub.publish(cmd)

    # =======================
    # Main loop
    # =======================
    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.wall_follow_step()
            rate.sleep()


if __name__ == "__main__":
    nav = NavigationController()
    nav.run()
