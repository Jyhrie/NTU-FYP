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
    
    best_idx = start_idx
    min_h = ((start_x - goal_x)**2 + (start_y - goal_y)**2)**0.5
    frontier = [(0.0, start_x, start_y, start_idx)]
    
    # Neighbors
    neighbors = [(0, 1, 1.0), (1, 0, 1.0), (0, -1, 1.0), (-1, 0, 1.0),
                 (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]

    while frontier:
        f_score, cx, cy, c_idx = pop(frontier)

        # Success Check
        dist_sq = (cx - goal_x)**2 + (cy - goal_y)**2
        if dist_sq <= 4.0:
            return reconstruct_path_optimized(parents, start_idx, c_idx, width), True

        for dx, dy, step_cost in neighbors:
            nx, ny = cx + dx, cy + dy
            
            if 0 <= nx < width and 0 <= ny < height:
                n_idx = ny * width + nx
                
                # HARD BLOCKERS (Walls or Unknown)
                if s_map[n_idx] != 0 or c_map[n_idx] >= fatal_cost:
                    continue

                # --- THE CENTER-LINE PREFERENCE LOGIC ---
                
                # Get the costmap value (0-255)
                raw_cost = float(c_map[n_idx])
                
                # 1. Base cost of moving through any cell
                traversal_cost = step_cost
                
                # 2. Add an EXPONENTIAL penalty for wall proximity
                # Even if raw_cost is low (e.g., 10), this makes it much more 
                # expensive than a 0-cost cell.
                # Formula: (cost / scale)^exponent
                wall_repulsion = (raw_cost / 2.0) ** 3.0
                
                # 3. Add a "Comfort Buffer"
                # If the cell isn't perfectly clear (0), add a flat penalty.
                # This makes the "shortest" path cut much more expensive than the "cleanest" path.
                if raw_cost > 0:
                    wall_repulsion += 30.0 

                new_cost = costs[c_idx] + traversal_cost + wall_repulsion

                if new_cost < costs[n_idx]:
                    costs[n_idx] = new_cost
                    # Greedy Heuristic: pushes the path to find the "valley" faster
                    h = ((goal_x - nx)**2 + (goal_y - ny)**2)**0.5
                    parents[n_idx] = c_idx
                    push(frontier, (new_cost + h * 1.5, nx, ny, n_idx))

    return reconstruct_path_optimized(parents, start_idx, best_idx, width), False

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