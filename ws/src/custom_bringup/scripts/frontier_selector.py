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

    def sanitize_goal(self, goal_idx, static_map, global_costmap):
            """
            Finds the nearest cell that is both KNOWN FREE (0) and 
            NOT LETHAL (<99) in the costmap.
            """
            h, w = static_map.shape
            x = max(0, min(int(goal_idx[0]), w - 1))
            y = max(0, min(int(goal_idx[1]), h - 1))
            
            # # If already safe, return it
            # if static_map[y][x] == 0 and global_costmap[y][x] < 99:
            #     return (x, y)

            # # BFS-style search in a 10-pixel radius (approx 0.5m)
            # for r in range(1, 11):
            #     for dx in range(-r, r + 1):
            #         for dy in range(-r, r + 1):
            #             nx, ny = x + dx, y + dy
            #             if 0 <= nx < w and 0 <= ny < h:
            #                 if static_map[ny][nx] == 0 and global_costmap[ny][nx] < 99:
            #                     return (nx, ny)
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
            safe_goal = self.sanitize_goal(centroid, static_map, global_costmap)
            
            # Call your local A* implementation
            path = a_star_exploration(static_map, global_costmap, start_idx, safe_goal)
            #print("Path: ", path)
            
            # Check if path is valid and reaches the goal area
            # (Note: a_star_exploration returns best_node if unreachable)
            if path is not None and len(path) > 0:
                dist_to_target = self.get_euclidean(path[-1], safe_goal)
                path_len = self.calculate_path_length_grid(path)
                # 5.0 pixels is usually safe for a 0.05m resolution map (0.25m tolerance)
                if dist_to_target < 5.0 and path_len > 10.0:
                    f['path'] = path
                    f['path_length'] = self.calculate_path_length_grid(path)
                    return f
            elif path is not None:
                print(path)
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