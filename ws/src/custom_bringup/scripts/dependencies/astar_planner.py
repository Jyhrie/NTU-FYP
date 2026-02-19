#!/usr/bin/env python3
import heapq
import numpy as np

def a_star_exploration(static_map_raw, costmap_raw, start, goal, width=800, height=800, fatal_cost=90):
    push = heapq.heappush
    pop = heapq.heappop
    
    size = width * height
    # Use a large but safe float for infinity
    costs = np.full(size, 1e8, dtype=np.float32)
    parents = np.full(size, -1, dtype=np.int32)
    
    s_map = np.array(static_map_raw, dtype=np.int8).ravel()
    c_map = np.array(costmap_raw, dtype=np.uint8).ravel()
    
    start_x, start_y = start
    goal_x, goal_y = goal
    start_idx = start_y * width + start_x
    costs[start_idx] = 0.0
    
    # Track best attempt for partial paths
    best_idx = start_idx
    min_h = ((start_x - goal_x)**2 + (start_y - goal_y)**2)**0.5
    
    # Priority Queue: (f_score, x, y, index)
    frontier = [(0.0, start_x, start_y, start_idx)]
    
    # Neighbors with pre-calculated costs
    neighbors = [
        (0, 1, 1.0), (1, 0, 1.0), (0, -1, 1.0), (-1, 0, 1.0),       # Cardinal
        (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414) # Diagonal
    ]

    while frontier:
        f_score, cx, cy, c_idx = pop(frontier)

        # Early exit if we already found a cheaper way to this cell
        if f_score > costs[c_idx] + min_h + 10: 
            continue

        # Success Check (within 1.5 cells of goal)
        dist_sq = (cx - goal_x)**2 + (cy - goal_y)**2
        if dist_sq <= 2.25:
            return reconstruct_path_optimized(parents, start_idx, c_idx, width), True

        # Update best attempt for partial path fallback
        current_h = dist_sq**0.5
        if current_h < min_h:
            min_h = current_h
            best_idx = c_idx

        for dx, dy, step_cost in neighbors:
            nx, ny = cx + dx, cy + dy
            
            if 0 <= nx < width and 0 <= ny < height:
                n_idx = ny * width + nx
                
                # 1. Hard Obstacle Check
                if s_map[n_idx] >= 100 or s_map[n_idx] <= -1 or c_map[n_idx] >= fatal_cost:
                    continue

                # 2. Wall Avoidance Penalty Logic
                # Any cost in c_map usually represents proximity to a wall.
                proximity_cost = float(c_map[n_idx])
                
                # Exponential penalty: Small costs (far from walls) stay small.
                # High costs (near walls) explode, making them very unattractive.
                # penalty = (proximity_cost / factor) ^ power
                repulsion_penalty = (proximity_cost / 3.0) ** 2.5
                
                # Add a "buffer bonus": even a small costmap value gets a flat penalty 
                # to discourage brushing against the inflation radius edge.
                if proximity_cost > 10:
                    repulsion_penalty += 20.0

                new_cost = costs[c_idx] + step_cost + repulsion_penalty

                if new_cost < costs[n_idx]:
                    costs[n_idx] = new_cost
                    
                    # 3. Admissible Heuristic (Euclidean)
                    # Use a slight multiplier (1.05) to make search more direct
                    h = ((goal_x - nx)**2 + (goal_y - ny)**2)**0.5
                    parents[n_idx] = c_idx
                    
                    # f = g + h
                    push(frontier, (new_cost + h * 1.05, nx, ny, n_idx))

    # If no complete path, return the path to the closest point reached
    partial_path = reconstruct_path_optimized(parents, start_idx, best_idx, width)
    return partial_path, False

def reconstruct_path_optimized(parents, start_idx, end_idx, width):
    path = []
    curr = end_idx
    while curr != -1:
        path.append((int(curr % width), int(curr // width)))
        if curr == start_idx: 
            break
        curr = parents[curr]
    
    path.reverse()
    
    # Path trimming: remove last few nodes to avoid "overshooting" the target
    # but only if the path is long enough to justify it.
    if len(path) > 10:
        return path[:-3]
    return path