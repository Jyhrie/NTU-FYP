import numpy as np
import os
from math import atan2, sin, pi
from dotenv import load_dotenv

load_dotenv()

class PurePursuitController:
    def __init__(self):
        self.lookahead_dist = float(os.getenv("LOOKAHEAD_DISTANCE", 0.5))
        self.max_v = float(os.getenv("MAX_VELOCITY", 0.5))
        self.max_w = float(os.getenv("MAX_ANGULAR_VEL", 1.0))

    def get_lookahead_point(self, robot_pose, path):
        """
        Finds the point on the path that is exactly 'lookahead_dist' away.
        robot_pose: (x, y, yaw)
        path: List of (x, y) in meters
        """
        rx, ry, _ = robot_pose
        
        for point in reversed(path):
            distance = np.sqrt((point[0] - rx)**2 + (point[1] - ry)**2)
            if distance <= self.lookahead_dist:
                return point
        return path[-1] # Fallback to the last point

    def compute_commands(self, robot_pose, path):
        """
        Calculates v (linear) and w (angular) velocities.
        """
        if not path or len(path) < 2:
            return 0.0, 0.0

        rx, ry, ryaw = robot_pose
        lx, ly = self.get_lookahead_point(robot_pose, path)

        # 1. Transform lookahead point to robot local frame
        dx = lx - rx
        dy = ly - ry
        
        # Angle to lookahead point in robot frame
        alpha = atan2(dy, dx) - ryaw
        
        # Normalize alpha to [-pi, pi]
        alpha = (alpha + pi) % (2 * pi) - pi

        # 2. Pure Pursuit Formula: curvature = 2*sin(alpha) / L
        # We use this to find the angular velocity w
        if abs(alpha) > 1.5: # If goal is behind, rotate in place
            v = 0.0
            w = self.max_w if alpha > 0 else -self.max_w
        else:
            # Curvature-based steering
            v = self.max_v
            w = (2 * self.max_v * sin(alpha)) / self.lookahead_dist
            
        # 3. Limit angular velocity
        w = max(min(w, self.max_w), -self.max_w)
        
        return v, w