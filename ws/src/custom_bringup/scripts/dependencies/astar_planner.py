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
    
    frontier = [(0.0, start_x, start_y, start_idx)]
    
    neighbors = [(0, 1, 1.0), (1, 0, 1.0), (0, -1, 1.0), (-1, 0, 1.0),
                 (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]

    while frontier:
        f_score, cx, cy, c_idx = pop(frontier)

        if (cx - goal_x)**2 + (cy - goal_y)**2 <= 4.0:
            return reconstruct_path_optimized(parents, start_idx, c_idx, width), True

        for dx, dy, step_cost in neighbors:
            nx, ny = cx + dx, cy + dy
            
            if 0 <= nx < width and 0 <= ny < height:
                n_idx = ny * width + nx
                
                # Check for walls
                if s_map[n_idx] != 0 or c_map[n_idx] >= fatal_cost:
                    continue

                raw_cost = float(c_map[n_idx])
                
                # --- WIDE ANGLE LOGIC ---
                
                # 1. Exponential Repulsion (The "Wall is Lava" logic)
                # We use a very aggressive power here to make the 'valley' extremely steep.
                repulsion = (raw_cost / 1.2) ** 3.5 
                
                # 2. The "Inertia" Penalty (Prevents tight clipping)
                # If we are in an inflation zone (raw_cost > 0), 
                # we add a penalty that scales with proximity. 
                # This makes "moving away" to a 0-cost cell much cheaper than staying near the wall.
                if raw_cost > 0:
                    repulsion += 100.0  # Massive flat fee to enter any inflation zone
                
                # 3. Path Smoothing Tie-Breaker
                # We penalize the 'traversal' more if it's in a high-cost area.
                weighted_step = step_cost * (1.0 + (raw_cost / 10.0))

                new_cost = costs[c_idx] + weighted_step + repulsion

                if new_cost < costs[n_idx]:
                    costs[n_idx] = new_cost
                    
                    # 4. Modified Heuristic (Admissible but cautious)
                    # We drop the heuristic weight to 1.0. 
                    # High weights (1.5) cause "greedy" wall hugging.
                    # A weight of 1.0 (or even 0.9) makes the robot 'patient' enough
                    # to take the long way around a corner.
                    h = ((goal_x - nx)**2 + (goal_y - ny)**2)**0.5
                    
                    parents[n_idx] = c_idx
                    push(frontier, (new_cost + h * 1.0, nx, ny, n_idx))

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