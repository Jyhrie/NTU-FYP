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
    
    # Priority Queue stores: (total_cost, x, y, index)
    # We remove 'f_score' and use only 'accumulated_cost'
    frontier = [(0.0, start_x, start_y, start_idx)]
    
    neighbors = [(0, 1, 1.0), (1, 0, 1.0), (0, -1, 1.0), (-1, 0, 1.0),
                 (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]

    while frontier:
        curr_cost, cx, cy, c_idx = pop(frontier)

        # Success Check: Stop the moment we reach the goal area
        if (cx - goal_x)**2 + (cy - goal_y)**2 <= 2.25:
            return reconstruct_path_optimized(parents, start_idx, c_idx, width), True

        # Optimization: if we found a better way already, skip
        if curr_cost > costs[c_idx]:
            continue

        for dx, dy, step_cost in neighbors:
            nx, ny = cx + dx, cy + dy
            
            if 0 <= nx < width and 0 <= ny < height:
                n_idx = ny * width + nx
                
                # Hard obstacles are strictly forbidden
                if s_map[n_idx] != 0 or c_map[n_idx] >= fatal_cost:
                    continue

                # --- PURE COST PRIORITIZATION ---
                raw_cost = float(c_map[n_idx])
                
                # 1. Traversal cost (1.0 or 1.414)
                # 2. Penalty: We make even a small costmap value 
                # significantly more expensive than a 'clear' cell.
                # If raw_cost is 0, penalty is 0. 
                # If raw_cost is 1, penalty is 50. This creates a hard 'cliff' 
                # around obstacles.
                penalty = 0.0
                if raw_cost > 0:
                    penalty = 50.0 + (raw_cost ** 2) 

                new_total_cost = costs[c_idx] + step_cost + penalty

                if new_total_cost < costs[n_idx]:
                    costs[n_idx] = new_total_cost
                    parents[n_idx] = c_idx
                    
                    # --- NO HEURISTIC ---
                    # By pushing ONLY the accumulated cost, we ensure 
                    # the algorithm finds the absolute cheapest path 
                    # across the entire map before it considers moving 
                    # into a 'high cost' zone.
                    push(frontier, (new_total_cost, nx, ny, n_idx))

    return [], False

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