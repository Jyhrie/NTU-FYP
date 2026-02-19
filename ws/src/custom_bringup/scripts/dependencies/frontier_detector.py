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
        
        # Standard ROS OccupancyGrid values
        self.FREE = 0
        self.OCCUPIED = 100
        self.UNKNOWN = -1

    def get_frontiers(self, x, y, map):
        unfiltered_frontiers = self.get_frontier_cell_groups_wfd(x, y, map) 
        filtered_clusters = self.filter_cluster(unfiltered_frontiers, map)
        centroids = self.frontier_to_centroid(filtered_clusters)

        #print(unfiltered_frontiers)
        #print(filtered_clusters)
        #print(centroids)
        
        final_targets = []
        for i in range(len(centroids)):
            target = self.get_nearest_valid_cell(
                centroids[i], 
                filtered_clusters[i], 
                map
            )
            final_targets.append(target)

        final_targets.sort(key=lambda goal: math.sqrt((x - goal[0])**2 + (y - goal[1])**2))
    
        return final_targets
    
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

    def filter_cluster(self, clusters, map_data):
            # 1. Reshape map for 2D indexing
            grid = np.array(map_data).reshape((self.height, self.width))
            
            filtered = []
            # Define the search window size (e.g., 5x5 or 6x6)
            # A radius of 3 gives a 7x7 box; radius of 2 gives a 5x5 box.
            radius = 3 
            wall_threshold = 10 # Reject if 10 or more cells are occupied

            for cluster in clusters:
                # Basic size check first (efficiency)
                if len(cluster) < MIN_FRONTIER_SIZE:
                    continue

                # 2. Calculate centroid of the cluster
                cx = int(sum(p[0] for p in cluster) / float(len(cluster)))
                cy = int(sum(p[1] for p in cluster) / float(len(cluster)))

                # 3. Define the bounding box around the centroid
                x_min = max(0, cx - radius)
                x_max = min(self.width, cx + radius + 1)
                y_min = max(0, cy - radius)
                y_max = min(self.height, cy + radius + 1)

                # 4. Extract the local patch and count occupied cells
                # Occupied cells in ROS are typically 100
                local_patch = grid[y_min:y_max, x_min:x_max]
                occupied_count = np.sum(local_patch == self.OCCUPIED)

                if occupied_count >= wall_threshold:
                    # Log this for debugging if needed
                    # print("Rejecting frontier at ({},{}): {} walls nearby".format(cx, cy, occupied_count))
                    continue

                filtered.append(cluster)

            return filtered


    def get_frontier_cell_groups_wfd(self, robot_x, robot_y, raw_data):
        if hasattr(raw_data, 'data'):
            raw_data = raw_data.data
        
        actual_size = len(raw_data)
        expected_size = self.height * self.width
        
        if actual_size != expected_size:
            print("!!! SIZE ERROR: Expected {} got {} !!!".format(expected_size, actual_size))
            return []

        # Reshape and check starting cell value
        grid = np.array(raw_data).reshape((self.height, self.width))
        start_val = grid[int(robot_y), int(robot_x)]
        
        print(">>>> STARTING WFD DEBUG <<<<")
        print("Robot Pos: Grid({}, {}) | Map Value at Start: {}".format(int(robot_x), int(robot_y), start_val))
        print("Map Settings: FREE={}, UNKNOWN={}, OCCUPIED={}".format(self.FREE, self.UNKNOWN, self.OCCUPIED))

        visited_map = np.zeros_like(grid, dtype=bool)
        all_clusters = []
        
        queue_map = deque([(int(robot_x), int(robot_y))])
        visited_map[int(robot_y), int(robot_x)] = True

        cells_processed = 0

        while queue_map:
            curr_x, curr_y = queue_map.popleft()
            cells_processed += 1
            
            # Periodic summary so you don't lose track of the flood
            if cells_processed % 100 == 0:
                print("-- Processed {} cells... Queue size: {} --".format(cells_processed, len(queue_map)))

            for nx, ny in self.get_neighbors(curr_x, curr_y):
                val = grid[ny, nx]
                
                # FLOOD PRINT: This will print for every single neighbor check
                # print("Checking Neighbor ({}, {}): Value={}, Visited={}".format(nx, ny, val, visited_map[ny, nx]))

                if not visited_map[ny, nx]:
                    is_frontier = self.is_frontier_pixel(grid, nx, ny)
                    
                    if is_frontier:
                        print("!!! FOUND FRONTIER AT ({}, {}) !!! Starting Cluster BFS...".format(nx, ny))
                        new_cluster = self.bfs_cluster(grid, nx, ny, visited_map)
                        print("Cluster BFS finished. Found {} pixels.".format(len(new_cluster)))
                        
                        if len(new_cluster) >= 3:
                            all_clusters.append(new_cluster)
                            print("Cluster ADDED. Total clusters now: {}".format(len(all_clusters)))
                        else:
                            print("Cluster REJECTED (too small).")
                        
                        # Mark visited to prevent infinite loops
                        visited_map[ny, nx] = True
                        queue_map.append((nx, ny))

                    elif val == self.FREE:
                        # Only expand into FREE space
                        visited_map[ny, nx] = True
                        queue_map.append((nx, ny))
                    else:
                        # It's unknown or occupied, we don't expand but we mark visited
                        visited_map[ny, nx] = True

        print(">>>> WFD COMPLETE <<<<")
        print("Total Cells Searched: {}".format(cells_processed))
        print("Final Cluster Count: {}".format(len(all_clusters)))
        
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

