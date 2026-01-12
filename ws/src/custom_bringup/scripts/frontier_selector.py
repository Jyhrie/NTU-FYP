#!/usr/bin/env python

import math
import time
import rospy
from geometry_msgs.msg import PoseStamped
# Import your local A* function
from astar_planner import a_star_exploration

class FrontierSelector:
    def __init__(self):
        # self.blacklist = {} 
        self.blacklist_dist = 0.5 
        # No longer need to wait for move_base/make_plan service

    def get_euclidean(self, p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def sanitize_goal(self, start_idx, goal_idx, static_map, global_costmap):
            """
            Projects a line from start to goal and finds the furthest SAFE point 
            along that line that is NOT unknown and NOT lethal.
            """
            h, w = static_map.shape
            x0, y0 = start_idx
            x1, y1 = int(goal_idx[0]), int(goal_idx[1])

            # Simple Bresenham-like line sampling
            steps = int(math.hypot(x1 - x0, y1 - y0))
            if steps == 0: return (x0, y0)

            last_safe = (x0, y0)

            for i in range(steps + 1):
                t = float(i) / steps
                curr_x = int(x0 + t * (x1 - x0))
                curr_y = int(y0 + t * (y1 - y0))
                
                # Bounds check
                if not (0 <= curr_x < w and 0 <= curr_y < h):
                    break
                
                # If we hit a wall, stop and use the last safe point
                if static_map[curr_y][curr_x] == 100 or global_costmap[curr_y][curr_x] >= 99:
                    break
                    
                # If it is known free space, update our last safe point
                if static_map[curr_y][curr_x] == 0:
                    last_safe = (curr_x, curr_y)
                
                # If we enter unknown space, we stop updating last_safe but keep 
                # looking (in case the line goes unknown -> known again)
                
            return last_safe
    
    def select_frontier(self, start_idx, frontiers, global_costmap, static_map):
        """
        start_idx: (x, y) in grid coordinates
        frontiers: list of dicts with 'centroid' in grid coordinates
        global_costmap: 2D array (0-100)
        static_map: 2D array (-1, 0, 100)
        """
        if not frontiers:
            print("No Frontiers")
            return None

        print("frontiers: ", frontiers)
        # 1. Sort by Euclidean distance
        frontiers.sort(key=lambda f: self.get_euclidean(start_idx, f['centroid']))

        # 2. Validation Loop
        for f in frontiers:
            print("Detecting Frontier f: ", f)
            centroid = f['centroid']
            
            # if centroid in self.blacklist:
            #     continue

            # Ensure the goal is actually reachable (not in -1 space)
            safe_goal = self.sanitize_goal(centroid, static_map, global_costmap)
            
            # Call your local A* implementation
            path = a_star_exploration(static_map, global_costmap, start_idx, safe_goal)
            #print("Path: ", path)
            
            # Check if path is valid and reaches the goal area
            # (Note: a_star_exploration returns best_node if unreachable)
            if path is not None and len(path) > 0:
                dist_to_target = self.get_euclidean(path[-1], safe_goal)
                # 5.0 pixels is usually safe for a 0.05m resolution map (0.25m tolerance)
                if dist_to_target < 5.0:
                    f['path'] = path
                    f['path_length'] = self.calculate_path_length_grid(path)
                    return f
            else:
                # print("Path -1 :", path[-1])
                # print("Safe Goal: ", safe_goal)
                pass

        return None

    def calculate_path_length_grid(self, path):
        """Calculates total distance for a list of (x, y) grid tuples."""
        distance = 0.0
        for i in range(1, len(path)):
            p1 = path[i-1]
            p2 = path[i]
            distance += math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        return distance