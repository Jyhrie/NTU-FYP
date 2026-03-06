#!/usr/bin/env python3
"""
camera_map_node.py

ROS Node: Camera-Looked Map Tracker
-------------------------------------
Subscribes:
  /controller/global  (std_msgs/String)   - global controller commands
  /map                (nav_msgs/OccupancyGrid) - base occupancy map
  /cmd_vel            (geometry_msgs/Twist)    - velocity commands

Uses:
  TF to determine robot position and orientation in the map frame

Behaviour:
  - Maintains a "camera-looked map" (same dimensions as /map)
  - When cmd_vel is all-zero (robot stopped), marks a forward-facing cone
    in the range [0.2, 1.6] metres as VISITED (value=100) in the camera map
  - Publishes the camera-looked map on /camera_looked_map

Cone parameters (tunable via ROS params):
  ~cone_half_angle_deg  (default 30.0)  - half-angle of the camera cone (degrees)
  ~cone_min_range_m     (default 0.2)   - minimum range of the cone (metres)
  ~cone_max_range_m     (default 1.6)   - maximum range of the cone (metres)
  ~camera_frame         (default "base_link") - frame whose +X is "forward"
  ~map_frame            (default "map")
  ~publish_rate_hz      (default 1.0)   - how often to re-publish camera map
"""

import math
import threading

import rospy
import tf2_ros
import tf2_geometry_msgs  # noqa: F401  (registers transforms)

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist, TransformStamped
from std_msgs.msg import String

# Cell value written for "visited by camera"
VISITED = 100
# Cell value for unknown / not yet seen
UNSEEN = 0


class CameraMapNode:
    def __init__(self):
        rospy.init_node("camera_map_node", anonymous=False)

        # ── Parameters ────────────────────────────────────────────────────────
        self.cone_half_angle = math.radians(
            rospy.get_param("~cone_half_angle_deg", 30.0)
        )
        self.cone_min = rospy.get_param("~cone_min_range_m", 0.2)
        self.cone_max = rospy.get_param("~cone_max_range_m", 1.6)
        self.camera_frame = rospy.get_param("~camera_frame", "base_link")
        self.map_frame = rospy.get_param("~map_frame", "map")
        publish_rate = rospy.get_param("~publish_rate_hz", 1.0)

        # ── Internal state ────────────────────────────────────────────────────
        self._lock = threading.Lock()
        self.map_msg: OccupancyGrid = None        # latest /map message
        self.camera_map: list = None              # flat int8 list (same size)
        self.global_cmd: str = ""                 # last /controller/global string

        # ── TF ───────────────────────────────────────────────────────────────
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # ── Publishers ───────────────────────────────────────────────────────
        self.camera_map_pub = rospy.Publisher(
            "/camera_looked_map", OccupancyGrid, queue_size=1, latch=True
        )

        # ── Subscribers ──────────────────────────────────────────────────────
        rospy.Subscriber("/map", OccupancyGrid, self._map_cb, queue_size=1)
        rospy.Subscriber("/cmd_vel", Twist, self._cmd_vel_cb, queue_size=10)
        rospy.Subscriber(
            "/controller/global", String, self._global_cb, queue_size=10
        )

        # ── Publish timer ────────────────────────────────────────────────────
        rospy.Timer(rospy.Duration(1.0 / publish_rate), self._publish_camera_map)

        rospy.loginfo(
            "[camera_map_node] Started. "
            f"Cone ±{math.degrees(self.cone_half_angle):.1f}° | "
            f"range [{self.cone_min}, {self.cone_max}] m"
        )

    # ── Subscriber callbacks ──────────────────────────────────────────────────

    def _global_cb(self, msg: String):
        with self._lock:
            self.global_cmd = msg.data
        rospy.loginfo(f"[camera_map_node] /controller/global: '{msg.data}'")

    def _map_cb(self, msg: OccupancyGrid):
        with self._lock:
            if self.map_msg is None or self._map_dimensions_changed(msg):
                rospy.loginfo(
                    f"[camera_map_node] Map received: "
                    f"{msg.info.width}x{msg.info.height} cells, "
                    f"res={msg.info.resolution:.3f} m/cell"
                )
                # Reset camera map to match new map dimensions
                num_cells = msg.info.width * msg.info.height
                self.camera_map = [UNSEEN] * num_cells
            self.map_msg = msg

    def _map_dimensions_changed(self, new_msg: OccupancyGrid) -> bool:
        if self.map_msg is None:
            return True
        return (
            new_msg.info.width != self.map_msg.info.width
            or new_msg.info.height != self.map_msg.info.height
        )

    def _cmd_vel_cb(self, msg: Twist):
        """When the robot stops (all velocities == 0), stamp the camera cone."""
        lin = msg.linear
        ang = msg.angular
        is_stopped = (
            lin.x == 0.0
            and lin.y == 0.0
            and lin.z == 0.0
            and ang.x == 0.0
            and ang.y == 0.0
            and ang.z == 0.0
        )
        if is_stopped:
            self._mark_camera_cone()

    # ── Core logic ────────────────────────────────────────────────────────────

    def _mark_camera_cone(self):
        """
        Look up robot pose via TF and mark all cells within the camera cone
        (forward-facing, [cone_min, cone_max] m, ±cone_half_angle) as VISITED.
        """
        with self._lock:
            if self.map_msg is None or self.camera_map is None:
                rospy.logwarn_throttle(
                    5.0, "[camera_map_node] No map yet; skipping cone mark."
                )
                return

            # ── Get robot pose in map frame ───────────────────────────────────
            try:
                transform: TransformStamped = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    self.camera_frame,
                    rospy.Time(0),
                    rospy.Duration(0.3),
                )
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ) as e:
                rospy.logwarn_throttle(
                    2.0, f"[camera_map_node] TF lookup failed: {e}"
                )
                return

            tx = transform.transform.translation.x
            ty = transform.transform.translation.y
            q = transform.transform.rotation
            # yaw from quaternion
            yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z),
            )

            info = self.map_msg.info
            res = info.resolution
            ox = info.origin.position.x
            oy = info.origin.position.y
            w = info.width
            h = info.height

            # ── Iterate over cells in the bounding box of the cone ────────────
            # We scan a square around the robot of radius cone_max, then filter.
            margin_cells = int(math.ceil(self.cone_max / res)) + 1
            robot_col = int((tx - ox) / res)
            robot_row = int((ty - oy) / res)

            col_min = max(0, robot_col - margin_cells)
            col_max = min(w - 1, robot_col + margin_cells)
            row_min = max(0, robot_row - margin_cells)
            row_max = min(h - 1, robot_row + margin_cells)

            marked = 0
            for row in range(row_min, row_max + 1):
                for col in range(col_min, col_max + 1):
                    # world coords of cell centre
                    cx = ox + (col + 0.5) * res
                    cy = oy + (row + 0.5) * res

                    dx = cx - tx
                    dy = cy - ty
                    dist = math.hypot(dx, dy)

                    if dist < self.cone_min or dist > self.cone_max:
                        continue

                    # angle between robot heading and cell direction
                    angle_to_cell = math.atan2(dy, dx)
                    angle_diff = self._angle_diff(angle_to_cell, yaw)

                    if abs(angle_diff) <= self.cone_half_angle:
                        idx = row * w + col
                        self.camera_map[idx] = VISITED
                        marked += 1

            if marked:
                rospy.logdebug(
                    f"[camera_map_node] Marked {marked} cells as visited "
                    f"(robot @ ({tx:.2f},{ty:.2f}), yaw={math.degrees(yaw):.1f}°)"
                )

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """Signed smallest difference between two angles (radians)."""
        d = a - b
        while d > math.pi:
            d -= 2 * math.pi
        while d < -math.pi:
            d += 2 * math.pi
        return d

    # ── Publisher ─────────────────────────────────────────────────────────────

    def _publish_camera_map(self, _event=None):
        with self._lock:
            if self.map_msg is None or self.camera_map is None:
                return

            out = OccupancyGrid()
            out.header.stamp = rospy.Time.now()
            out.header.frame_id = self.map_frame
            out.info = self.map_msg.info          # same metadata as /map
            out.data = self.camera_map

        self.camera_map_pub.publish(out)

    # ── Spin ──────────────────────────────────────────────────────────────────

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    node = CameraMapNode()
    node.spin()