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

        # Parameters for filtering
        self.robot_radius = 0.15  # 30cm wide robot
        # Filter: Frontier must be at least as wide as the robot
        self.min_frontier_width = self.robot_radius * 2.0 
        
        self.FREE = 0
        self.UNKNOWN = -1

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

    def detect_frontiers(self, raw_data, costmap_data):
        # Reshape the flat data into a 2D grid
        grid = np.array(raw_data).reshape((self.height, self.width))
        visited = np.zeros_like(grid, dtype=bool)
        
        # This will show up in RViz as the /detected_frontiers topic
        frontier_debug_grid = np.zeros_like(grid, dtype=np.int8)
        frontiers_metadata = []

        for y in range(self.height):
            for x in range(self.width):
                if not visited[y, x] and self.is_frontier_pixel(grid, x, y):
                    new_cluster = self.bfs_cluster(grid, x, y, visited)

                    # ONLY SIZE FILTER REMAINS
                    if self.is_valid_size(new_cluster):
                        # Mark pixels for visualization
                        for px, py in new_cluster:
                            frontier_debug_grid[py, px] = 100 
                        
                        # Store world coordinates
                        frontiers_metadata.append({
                            'id': len(frontiers_metadata),
                            'centroid': self.calculate_centroid(new_cluster),
                            'size': len(new_cluster) # Pixel count used for Gain
                        })
        
        return frontiers_metadata, frontier_debug_grid.flatten().tolist()
    
    def is_valid_size(self, cluster):
        if len(cluster) < 3: 
            return False

        xs = [p[0] for p in cluster]
        ys = [p[1] for p in cluster]
        
        # Calculate the physical extent of the cluster
        width_m = (max(xs) - min(xs)) * self.res
        height_m = (max(ys) - min(ys)) * self.res
        
        # The 'span' is the longest dimension of the cluster
        # If this is 20 pixels and your res is 0.05, span = 1.0 meter
        span = np.sqrt(width_m**2 + height_m**2)
        
        return span >= self.min_frontier_width

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
        
        world_x = (avg_x * self.res) + self.origin_x
        world_y = (avg_y * self.res) + self.origin_y
        return (world_x, world_y)