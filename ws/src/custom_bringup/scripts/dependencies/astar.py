#!/usr/bin/env python3

import rospy
import math
import cv2
import numpy as np
from typing import Union
from std_msgs.msg import Header
from nav_msgs.msg import GridCells, OccupancyGrid, Path
from geometry_msgs.msg import Point, Quaternion, Pose, PoseStamped
from priority_queue import PriorityQueue
from tf.transformations import quaternion_from_euler

import numpy as np
import heapq


class PathPlanner:
    @staticmethod
    def is_cell_in_bounds(mapdata: OccupancyGrid, p: "tuple[int, int]") -> bool:
        width = mapdata.info.width
        height = mapdata.info.height
        x = p[0]
        y = p[1]

        if x < 0 or x >= width:
            return False
        if y < 0 or y >= height:
            return False
        return True

    @staticmethod
    def is_cell_walkable(mapdata: OccupancyGrid, p: "tuple[int, int]") -> bool:
        """
        A cell is walkable if all of these conditions are true:
        1. It is within the boundaries of the grid;
        2. It is free (not occupied by an obstacle)
        :param mapdata [OccupancyGrid] The map information.
        :param p       [(int, int)]    The coordinate in the grid.
        :return        [bool]          True if the cell is walkable, False otherwise
        """
        if not PathPlanner.is_cell_in_bounds(mapdata, p):
            return False

        WALKABLE_THRESHOLD = 50
        return PathPlanner.get_cell_value(mapdata, p) < WALKABLE_THRESHOLD
    
    @staticmethod
    def a_star(
        mapdata: OccupancyGrid,
        cost_map: np.ndarray,
        start: "tuple[int, int]",
        goal: "tuple[int, int]"):

        COST_MAP_WEIGHT = 1000
        NEIGHBORS = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),      # Cardinals
            (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414) # Diagonals
        ]

        pq = PriorityQueue()
        pq.put(start, 0)

        cost_so_far = {}
        distance_cost_so_far = {}
        cost_so_far[start] = 0
        distance_cost_so_far[start] = 0
        came_from = {}
        came_from[start] = None

        #DO ASTAR HERE
        found_goal = False
        while not pq.empty():
            current = pq.get()

            if current == goal:
                found_goal = True
                break

            cx, cy = current

            for dx, dy, step_dist in NEIGHBORS:
                nx, ny = cx + dx, cy + dy
                neighbor = (nx, ny)

                # Check bounds and walkability (using his methods)
                if not PathPlanner.is_cell_walkable(mapdata, neighbor):
                    continue

                # TRAVERSAL LOGIC: 
                # Physical distance + (Giant Weight * the "Valley" value)
                # This makes higher costmap areas feel like miles to the robot
                cell_cost = cost_map[ny, nx] # Accessing numpy array directly
                added_cost = step_dist + (COST_MAP_WEIGHT * (cell_cost / 255.0))
                
                new_cost = cost_so_far[current] + added_cost

                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    distance_cost_so_far[neighbor] = distance_cost_so_far[current] + step_dist
                    
                    # Heuristic: Euclidean distance (the 'as the crow flies' guess)
                    h = math.sqrt((nx - goal[0])**2 + (ny - goal[1])**2)
                    priority = new_cost + h
                    
                    pq.put(neighbor, priority)
                    came_from[neighbor] = current
        
        if not found_goal:
            return (None, None, start, goal)

        path = []
        cell = goal

        while cell:
            path.insert(0, cell)

            if cell in came_from:
                cell = came_from[cell]
            else:
                return (None, None, start, goal)

        # Prevent paths that are too short
        MIN_PATH_LENGTH = 12
        if len(path) < MIN_PATH_LENGTH:
            return (None, None, start, goal)

        # Truncate the last few poses of the path
        POSES_TO_TRUNCATE = 8
        path = path[:-POSES_TO_TRUNCATE]

        return (path, distance_cost_so_far[goal], start, goal)