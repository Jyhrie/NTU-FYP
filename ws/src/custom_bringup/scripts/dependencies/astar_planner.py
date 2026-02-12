#!/usr/bin/env python3
import heapq
import numpy as np

def a_star_exploration(static_map_raw, costmap_raw, start, goal, width=800, height=800, fatal_cost=90):
    # Localize functions for micro-speed boost
    push = heapq.heappush
    pop = heapq.heappop
    
    # 1. Prepare flattened arrays (fastest access on Jetson)
    size = width * height
    costs = np.full(size, 1e8, dtype=np.float32)  # 'Infinity'
    parents = np.full(size, -1, dtype=np.int32)
    
    # Convert raw data to NumPy once
    s_map = np.array(static_map_raw, dtype=np.int8).ravel()
    c_map = np.array(costmap_raw, dtype=np.uint8).ravel()
    
    start_x, start_y = start
    goal_x, goal_y = goal
    start_idx = start_y * width + start_x
    costs[start_idx] = 0.0
    
    # Priority Queue: (priority, x, y, index)
    frontier = [(0.0, start_x, start_y, start_idx)]
    
    # Pre-calculated offsets for 8-way movement: (dx, dy, step_cost)
    neighbors = [
        (0, 1, 1.0), (1, 0, 1.0), (0, -1, 1.0), (-1, 0, 1.0),
        (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)
    ]

    while frontier:
        f_score, cx, cy, c_idx = pop(frontier)

        # 2. Check success (Euclidean distance squared is faster than hypot)
        if (cx - goal_x)**2 + (cy - goal_y)**2 <= 2.25: # 1.5^2
            return reconstruct_path_optimized(parents, start_idx, c_idx, width)

        # Skip stale nodes in the heap
        if f_score > costs[c_idx] + ((cx - goal_x)**2 + (cy - goal_y)**2)**0.5 + 1:
            continue

        for dx, dy, step_cost in neighbors:
            nx, ny = cx + dx, cy + dy
            
            # Boundary check
            if 0 <= nx < width and 0 <= ny < height:
                n_idx = ny * width + nx
                
                # Fatal obstacles check
                if s_map[n_idx] >= 100 or s_map[n_idx] <= -1 or c_map[n_idx] >= fatal_cost:
                    continue

                # Wall avoidance penalty
                penalty = (c_map[n_idx] / 5.0) ** 2
                new_cost = costs[c_idx] + step_cost + penalty

                if new_cost < costs[n_idx]:
                    costs[n_idx] = new_cost
                    # Heuristic: distance to goal + small tie-breaker
                    h = ((goal_x - nx)**2 + (goal_y - ny)**2)**0.5 * 1.001
                    parents[n_idx] = c_idx
                    push(frontier, (new_cost + h, nx, ny, n_idx))

    return None

def reconstruct_path_optimized(parents, start_idx, end_idx, width):
    path = []
    curr = end_idx
    while curr != -1:
        path.append((int(curr % width), int(curr // width)))
        if curr == start_idx: break
        curr = parents[curr]
    path.reverse()
    # Your original trimming logic
    return path[:-3] if len(path) > 6 else []