import math
import time

class FrontierSelector:
    def __init__(self, mode="greedy"):
        self.mode = mode # "greedy" or "fast"
        self.last_selected_id = None
        self.blacklist = {} 
        self.blacklist_dist = 0.5 

    def select_best_frontier(self, robot_pose, frontiers):
        if not frontiers: return None

        # Clean blacklist (30s timeout)
        now = time.time()
        self.blacklist = {p: t for p, t in self.blacklist.items() if now - t < 30}

        # Set weights based on mode
        if self.mode == "greedy":
            # Greedy: Distance is the primary factor
            gain_w = 0.5
            dist_w = 5.0  # High penalty for distance
        else: # "fast"
            # Fast: Size (Information Gain) is the primary factor
            gain_w = 5.0  # High reward for large openings
            dist_w = 1.0

        hysteresis_w = 0.2
        best_score = -float('inf')
        best_f = None

        for f in frontiers:
            # Skip blacklisted failed gaps
            if any(math.dist(f['centroid'], b) < self.blacklist_dist for b in self.blacklist):
                continue

            gain = f['size'] * gain_w
            dist = math.dist(robot_pose, f['centroid'])
            
            # Hysteresis prevents "flickering" between two close points
            hysteresis = (hysteresis_w * gain) if f['id'] == self.last_selected_id else 0
            
            # Final Score
            score = gain - (dist * dist_w) + hysteresis

            if score > best_score:
                best_score = score
                best_f = f

        if best_f:
            self.last_selected_id = best_f['id']
            return best_f['centroid']
        return None

    def add_to_blacklist(self, centroid):
        self.blacklist[centroid] = time.time()