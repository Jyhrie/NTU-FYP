#!/usr/bin/env python
import numpy as np
import math
from collections import deque

MIN_FRONTIER_SIZE = 4

class FrontierDetector:
    def __init__(self, map_width, map_height, resolution, origin_x, origin_y):
        self.width = map_width
        self.height = map_height
        self.res = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y

        self.robot_radius = 0.15 
        self.min_frontier_width = self.robot_radius * 2.0 
        
        self.FREE = 0
        self.UNKNOWN = -1

    def get_frontiers(self, x, y, map):
        unfiltered_frontiers = self.get_frontier_cell_groups_wfd(x, y, map)
        filtered_clusters = self.filter_cluster(unfiltered_frontiers)
        centroids = self.frontier_to_centroid(filtered_clusters)

        final_targets = []
        for i in range(len(centroids)):
            target = self.get_nearest_valid_cell(
                centroids[i], 
                filtered_clusters[i], 
                map
            )
            final_targets.append(target)

        final_targets.sort(key=lambda goal: math.sqrt((x - goal[0])**2 + (y - goal[1])**2))
    
        return 
    
    def frontier_to_centroid(self, clusters): 
        centroids = []
        for cluster in clusters:
            cx = sum(p[0] for p in cluster) / float(len(cluster))
            cy = sum(p[1] for p in cluster) / float(len(cluster))
            centroids.append((cx, cy))
        return centroids
    
    def get_nearest_valid_cell(self, centroid, cluster, map_data):
        # 1. Convert flat map data to 2D for easier indexing
        grid = np.array(map_data).reshape((self.height, self.width))
        cx, cy = int(centroid[0]), int(centroid[1])

        # 2. Check if the centroid itself is already valid
        if grid[cy, cx] == 0:
            return (cx, cy)

        # 3. If not, find the pixel in the cluster that is closest to the centroid
        # AND is free space.
        best_pixel = None
        min_dist = float('inf')

        for px, py in cluster:
            # Check if this specific cluster pixel is FREE space
            if grid[py, px] == 0:
                # Calculate distance to centroid to find the 'closest' valid part
                dist = math.sqrt((px - cx)**2 + (py - cy)**2)
                if dist < min_dist:
                    min_dist = dist
                    best_pixel = (px, py)

        # 4. Fallback: If no pixels in the cluster are FREE, 
        # run a small spiral search around the centroid (max 0.5m)
        if best_pixel is None:
            for r in range(1, 10): 
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            if grid[ny, nx] == 0:
                                return (nx, ny)

        return best_pixel if best_pixel else (cx, cy)

    def filter_cluster(self, clusters):
        return [c for c in clusters if len(c) >= MIN_FRONTIER_SIZE]


    def get_frontier_cell_groups_wfd(self, robot_x, robot_y, raw_data):
        grid = np.array(raw_data).reshape((self.height, self.width))
        
        # Track visited cells across the whole process
        visited_map = np.zeros_like(grid, dtype=bool)
        all_clusters = []
        
        # The primary queue for BFS 1 (starting at the robot)
        queue_map = deque([(int(robot_x), int(robot_y))])
        visited_map[int(robot_y), int(robot_x)] = True

        while queue_map:
            curr_x, curr_y = queue_map.popleft()

            # Check all 8 neighbors
            for nx, ny in self.get_neighbors(curr_x, curr_y):
                if not visited_map[ny, nx]:
                    # 1. If it's a frontier pixel, start BFS 2 to find the whole cluster
                    if self.is_frontier_pixel(grid, nx, ny):
                        # self.bfs_cluster will mark pixels as visited so we don't find them twice
                        new_cluster = self.bfs_cluster(grid, nx, ny, visited_map)
                        if len(new_cluster) >= 3:
                            all_clusters.append(new_cluster)
                    
                    # 2. If it's free space, keep expanding the Map-BFS
                    elif grid[ny, nx] == self.FREE:
                        visited_map[ny, nx] = True
                        queue_map.append((nx, ny))
                        
        return all_clusters
        
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
    
    def is_frontier_pixel(self, grid, x, y):
        if grid[y, x] != self.FREE:
            return False
        for nx, ny in self.get_neighbors(x, y):
            if grid[ny, nx] == self.UNKNOWN:
                return True
        return False
    
    def get_neighbors(self, x, y):
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0: continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    neighbors.append((nx, ny))
        return neighbors

