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

    def sanitize_goal(self, goal_idx, static_map):
        """
        If the goal is in unknown (-1) or occupied (100) space,
        find the nearest free (0) neighbor.
        """
        # Force coordinates to integers to avoid IndexError
        x, y = int(goal_idx[0]), int(goal_idx[1]) 
        
        if static_map[y][x] == 0:
            return (x, y)

        # Search 8-neighbors for a valid '0' cell
        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
            nx, ny = x + dx, y + dy
            # Ensure neighbors are within map bounds
            if 0 <= nx < static_map.shape[1] and 0 <= ny < static_map.shape[0]:
                if static_map[ny][nx] == 0:
                    return (nx, ny)
        return (x, y)
    
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
            safe_goal = self.sanitize_goal(centroid, static_map)
            
            # Call your local A* implementation
            path = a_star_exploration(static_map, global_costmap, start_idx, safe_goal)
            print("Path: ", path)
            
            # Check if path is valid and reaches the goal area
            # (Note: a_star_exploration returns best_node if unreachable)
            if path and path[-1] == safe_goal:
                f['path'] = path
                f['path_length'] = self.calculate_path_length_grid(path)
                return f 
            else:
                # If path doesn't reach goal or is empty, blacklist it
                self.blacklist[centroid] = time.time()

        return None

    def calculate_path_length_grid(self, path):
        """Calculates total distance for a list of (x, y) grid tuples."""
        distance = 0.0
        for i in range(1, len(path)):
            p1 = path[i-1]
            p2 = path[i]
            distance += math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        return distance