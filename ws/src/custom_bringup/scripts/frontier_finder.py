#!/usr/bin/env python3

import numpy as np
from collections import deque

class FrontierDetector:
    def __init__(self, map_width, map_height, resolution, origin_x, origin_y):
        self.width = map_width
        self.height = map_height
        self.res = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y
        
        self.FREE = 0
        self.UNKNOWN = -1
        self.OCCUPIED = 100

    def get_neighbors(self, x, y):
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0: continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    neighbors.append((nx, ny))
        return neighbors

    def is_frontier_pixel(self, grid, x, y):
        if grid[y, x] != self.FREE:
            return False
        for nx, ny in self.get_neighbors(x, y):
            if grid[ny, nx] == self.UNKNOWN:
                return True
        return False

    def detect_frontiers(self, raw_data):
        """
        Returns:
            frontiers: List of centroids (meters)
            frontier_map: 1D array (same size as raw_data) for ROS publishing
        """
        grid = np.array(raw_data).reshape((self.height, self.width))
        visited = np.zeros_like(grid, dtype=bool)
        
        # Initialize a blank debug map (all zeros)
        # We use uint8 or int8 depending on your ROS message type
        frontier_debug_grid = np.zeros_like(grid, dtype=np.int8)
        
        frontiers = []

        for y in range(self.height):
            for x in range(self.width):
                if not visited[y, x] and self.is_frontier_pixel(grid, x, y):
                    new_cluster = self.bfs_cluster(grid, x, y, visited)
                    
                    if len(new_cluster) > 5:
                        # 1. Mark these pixels in our debug map
                        for px, py in new_cluster:
                            frontier_debug_grid[py, px] = 255 # Set your requested value
                        
                        # 2. Add to centroids list
                        frontiers.append(self.calculate_centroid(new_cluster))
        
        # Flatten the debug grid back to a 1D list for ROS
        frontier_map_data = frontier_debug_grid.flatten().tolist()
        
        return frontiers, frontier_map_data

    def bfs_cluster(self, grid, start_x, start_y, visited):
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
        avg_x = sum(p[0] for p in cluster) / len(cluster)
        avg_y = sum(p[1] for p in cluster) / len(cluster)
        world_x = avg_x * self.res + self.origin_x
        world_y = avg_y * self.res + self.origin_y
        return (world_x, world_y)