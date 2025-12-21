#!/usr/bin/env python3

import numpy as np
from collections import deque
import os
import scipy.ndimage as ndimage

class FrontierDetector:
    def __init__(self, map_width, map_height, resolution, origin_x, origin_y):
        self.width = map_width
        self.height = map_height
        self.res = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y

        # Parameters for filtering
        self.robot_radius = 0.3
        # Minimum physical width of a frontier to be considered traversable
        self.min_frontier_width = self.robot_radius * 1
        
        # ROS OccupancyGrid Constants
        self.FREE = 0
        self.UNKNOWN = -1
        self.OCCUPIED = 100

    def get_neighbors(self, x, y):
        """Returns 8-connected neighbors."""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0: continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    neighbors.append((nx, ny))
        return neighbors

    def is_frontier_pixel(self, grid, x, y):
        """A pixel is a frontier if it is FREE and has at least one UNKNOWN neighbor."""
        if grid[y, x] != self.FREE:
            return False
        for nx, ny in self.get_neighbors(x, y):
            if grid[ny, nx] == self.UNKNOWN:
                return True
        return False

    def clean_map(self, grid):
        """Removes salt-and-pepper noise from the map using morphological opening."""
        # binary_opening removes small 'islands' of free space
        mask = (grid == self.FREE)
        cleaned_mask = ndimage.binary_opening(mask, structure=np.ones((3,3)))
        
        cleaned_grid = np.copy(grid)
        # Set pixels that were noise to UNKNOWN
        cleaned_grid[(grid == self.FREE) & (~cleaned_mask)] = self.UNKNOWN
        return cleaned_grid

    def detect_frontiers(self, raw_data, costmap_data):
        """
        Processes the map and costmap to find safe, reachable frontiers.
        """
        # 1. Prepare Grids
        # Ensure we use int8 for occupancy data and cast costmap carefully
        raw_grid = np.array(raw_data).reshape((self.height, self.width))
        cost_grid = np.array(costmap_data).reshape((self.height, self.width))

        # 2. Pre-process map to remove noise
        grid = self.clean_map(raw_grid)

        visited = np.zeros_like(grid, dtype=bool)
        frontier_debug_grid = np.zeros_like(grid, dtype=np.int8)
        frontiers = []

        # 3. Scan for frontiers
        for y in range(self.height):
            for x in range(self.width):
                if not visited[y, x] and self.is_frontier_pixel(grid, x, y):
                    # Found a new frontier cluster
                    new_cluster = self.bfs_cluster(grid, x, y, visited)

                    # 4. Filter by Physical Size
                    if self.is_valid_size(new_cluster):
                        # Calculate pixel-space centroid for safety check
                        cx_px = int(sum(p[0] for p in new_cluster) / len(new_cluster))
                        cy_px = int(sum(p[1] for p in new_cluster) / len(new_cluster))

                        # 5. Filter by Costmap Safety (Inflation Check)
                        if self.is_centroid_safe((cx_px, cy_px), cost_grid):
                            # Mark only valid, safe frontier pixels for RViz
                            for px, py in new_cluster:
                                frontier_debug_grid[py, px] = 100 # High visibility
                            
                            # Convert to world coordinates for navigation
                            world_coords = self.calculate_centroid(new_cluster)
                            frontiers.append(world_coords)
        
        # Flatten debug map for ROS OccupancyGrid message
        frontier_map_data = frontier_debug_grid.flatten().tolist()
        
        return frontiers, frontier_map_data
    
    def is_valid_size(self, cluster):
        """Checks if the cluster spans a distance greater than the robot width."""
        if len(cluster) < 5: 
            return False

        xs = [p[0] for p in cluster]
        ys = [p[1] for p in cluster]
        
        width_m = (max(xs) - min(xs)) * self.res
        height_m = (max(ys) - min(ys)) * self.res
        
        # Pythagorean distance of the cluster extent
        span = np.sqrt(width_m**2 + height_m**2)
        return span >= self.min_frontier_width

    def is_centroid_safe(self, centroid_px, costmap_2d):
        """Checks if the navigation goal is inside a wall's inflation zone."""
        cx, cy = centroid_px
        
        # Bounds check
        if not (0 <= cx < self.width and 0 <= cy < self.height):
            return False

        # In costmap_2d: 
        # 0 = Free, 100 = Occupied (standard), 253 = Inscribed, 254 = Lethal
        # If cost > 99, it's generally considered unsafe for the center of the robot
        cost = costmap_2d[cy, cx]
        
        # A common issue is costmap being -1 (unknown). We treat unknown as unsafe.
        if cost > 99 or cost == -1:
            return False
        return True

    def bfs_cluster(self, grid, start_x, start_y, visited):
        """Flood-fill style search to find all connected frontier pixels."""
        cluster = []
        queue = deque([(start_x, start_y)])
        visited[start_y, start_x] = True
        
        while queue:
            curr_x, curr_y = queue.popleft()
            cluster.append((curr_x, curr_y))
            for nx, ny in self.get_neighbors(curr_x, curr_y):
                if not visited[ny, nx] and self.is_frontier_pixel(grid, nx, ny):
                    visited[ny, nx] = True
                    queue.append((nx, ny))
        return cluster

    def calculate_centroid(self, cluster):
        """Converts pixel cluster to world coordinates."""
        avg_x = sum(p[0] for p in cluster) / len(cluster)
        avg_y = sum(p[1] for p in cluster) / len(cluster)
        
        world_x = avg_x * self.res + self.origin_x
        world_y = avg_y * self.res + self.origin_y
        return (world_x, world_y)