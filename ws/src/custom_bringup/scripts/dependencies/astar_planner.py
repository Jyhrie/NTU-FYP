#!/usr/bin/env python3
import heapq
import numpy as np

def a_star_exploration(static_map_raw, costmap_raw, start, goal, width=800, height=800, fatal_cost=90):
    push = heapq.heappush
    pop = heapq.heappop
    
    size = width * height
    costs = np.full(size, 1e8, dtype=np.float32)
    parents = np.full(size, -1, dtype=np.int32)
    
    s_map = np.array(static_map_raw, dtype=np.int8).ravel()
    c_map = np.array(costmap_raw, dtype=np.uint8).ravel()
    
    start_x, start_y = start
    goal_x, goal_y = goal
    start_idx = start_y * width + start_x
    costs[start_idx] = 0.0
    
    # Track the best attempt in case of failure
    best_idx = start_idx
    min_h = ((start_x - goal_x)**2 + (start_y - goal_y)**2)**0.5
    
    frontier = [(0.0, start_x, start_y, start_idx)]
    
    neighbors = [
        (0, 1, 1.0), (1, 0, 1.0), (0, -1, 1.0), (-1, 0, 1.0),
        (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)
    ]

    while frontier:
        f_score, cx, cy, c_idx = pop(frontier)

        # Success Check
        dist_sq = (cx - goal_x)**2 + (cy - goal_y)**2
        if dist_sq <= 2.25: # 1.5^2
            path = reconstruct_path_optimized(parents, start_idx, c_idx, width)
            return path, True

        # Update best attempt
        current_h = dist_sq**0.5
        if current_h < min_h:
            min_h = current_h
            best_idx = c_idx

        if f_score > costs[c_idx] + current_h + 1:
            continue

        for dx, dy, step_cost in neighbors:
            nx, ny = cx + dx, cy + dy
            
            if 0 <= nx < width and 0 <= ny < height:
                n_idx = ny * width + nx
                
                if s_map[n_idx] >= 100 or s_map[n_idx] <= -1 or c_map[n_idx] >= fatal_cost:
                    continue

                penalty = (c_map[n_idx] / 5.0) ** 2
                new_cost = costs[c_idx] + step_cost + penalty

                if new_cost < costs[n_idx]:
                    costs[n_idx] = new_cost
                    h = ((goal_x - nx)**2 + (goal_y - ny)**2)**0.5
                    parents[n_idx] = c_idx
                    push(frontier, (new_cost + h * 1.001, nx, ny, n_idx))

    partial_path = reconstruct_path_optimized(parents, start_idx, best_idx, width)
    return partial_path, False

def reconstruct_path_optimized(parents, start_idx, end_idx, width):
    path = []
    curr = end_idx
    while curr != -1:
        path.append((int(curr % width), int(curr // width)))
        if curr == start_idx: break
        curr = parents[curr]
    path.reverse()
    return path[:-3] if len(path) > 6 else path