import os
import math
from dotenv import load_dotenv

# Load params from .env
load_dotenv()

class FrontierSelector:
    def __init__(self):
        self.gain_w = float(os.getenv("GAIN_WEIGHT", 1.0))
        self.dist_w = float(os.getenv("DISTANCE_WEIGHT", 1.5))
        self.hysteresis_w = float(os.getenv("HYSTERESIS_WEIGHT", 0.2))
        
        self.last_selected_id = None

    def calculate_distance(self, p1, p2):
        """Euclidean distance (or use your A* path length here)"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def select_best_frontier(self, robot_pose, frontiers):
        """
        robot_pose: (x, y) in meters
        frontiers: List of dicts [{'id': 0, 'centroid': (x,y), 'size': n_pixels}, ...]
        """
        if not frontiers:
            return None

        best_score = -float('inf')
        best_frontier = None

        for f in frontiers:
            # 1. Information Gain (Size of the frontier)
            gain = f['size'] * self.gain_w
            
            # 2. Path Cost (Distance)
            # Note: For better results, replace this with your A* path length
            dist = self.calculate_distance(robot_pose, f['centroid'])
            cost = dist * self.dist_w
            
            # 3. Hysteresis (Preference for current target)
            hysteresis = 0
            if f['id'] == self.last_selected_id:
                hysteresis = self.hysteresis_w * gain

            # Final Score Calculation
            # We want high gain and low cost
            score = gain - cost + hysteresis

            if score > best_score:
                best_score = score
                best_frontier = f

        if best_frontier:
            self.last_selected_id = best_frontier['id']
            return best_frontier['centroid']
        
        return None