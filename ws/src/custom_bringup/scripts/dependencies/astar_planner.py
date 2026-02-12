#!/usr/bin/env python3
import heapq
import numpy as np
from numba import njit

@njit
def _core_astar(s_map_flat, c_map_flat, start_tuple, goal_tuple, width, height, fatal_cost):
    """
    JIT-compiled core logic. Operates on flattened arrays for maximum speed.
    """
    size = width * height
    # Use float32 for costs and int32 for parents to save memory/time
    costs = np.full(size, 1e9, dtype=np.float32)
    parents = np.full(size, -1, dtype=np.int32)
    
    start_x, start_y = start_tuple
    goal_x, goal_y = goal_tuple
    start_idx = start_y * width + start_x
    
    costs[start_idx] = 0.0
    # Priority Queue: (priority, x, y, index)
    frontier = [(0.0, start_x, start_y, start_idx)]
    
    # Pre-calculate neighbor offsets for 8-way movement
    # (dx, dy, cost)
    neighbor_offsets = [
        (0, 1, 1.0), (1, 0, 1.0), (0, -1, 1.0), (-1, 0, 1.0),
        (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)
    ]

    while len(frontier) > 0:
        f_score, cx, cy, c_idx = heapq.heappop(frontier)

        # Distance check for success (1.5 pixel tolerance)
        if ((cx - goal_x)**2 + (cy - goal_y)**2)**0.5 <= 1.5:
            return parents, c_idx

        # Standard A* check: if we found a better way to 'current' already, skip
        if f_score > costs[c_idx] + ((cx - goal_x)**2 + (cy - goal_y)**2)**0.5 + 2.0:
            continue

        for dx, dy, step_cost in neighbor_offsets:
            nx, ny = cx + dx, cy + dy
            
            if 0 <= nx < width and 0 <= ny < height:
                n_idx = ny * width + nx
                
                # Fatal/Obstacle Checks
                s_val = s_map_flat[n_idx]
                if s_val >= 100 or s_val <= -1:
                    continue
                
                c_val = c_map_flat[n_idx]
                if c_val >= fatal_cost:
                    continue

                # Penalty logic
                penalty = (c_val / 5.0) ** 2
                new_cost = costs[c_idx] + step_cost + penalty

                if new_cost < costs[n_idx]:
                    costs[n_idx] = new_cost
                    # Tie-breaking heuristic: slight preference for nodes on the direct line
                    h = ((goal_x - nx)**2 + (goal_y - ny)**2)**0.5 * 1.001
                    parents[n_idx] = c_idx
                    heapq.heappush(frontier, (new_cost + h, nx, ny, n_idx))

    return parents, -1

def a_star_exploration(static_map_raw, costmap_raw, start, goal, width=800, height=800, fatal_cost=90):
    """
    Main entry point compatible with your existing calls.
    """
    # Convert inputs to flat numpy arrays for the Numba core
    s_map_flat = np.array(static_map_raw, dtype=np.int8).flatten()
    c_map_flat = np.array(costmap_raw, dtype=np.uint8).flatten()
    
    # Run the compiled core
    parents, end_idx = _core_astar(s_map_flat, c_map_flat, start, goal, width, height, fatal_cost)
    
    if end_idx == -1:
        return None

    # Reconstruct path
    path = []
    curr = end_idx
    start_idx = start[1] * width + start[0]
    
    while curr != -1:
        path.append((int(curr % width), int(curr // width)))
        if curr == start_idx:
            break
        curr = parents[curr]
    
    path.reverse()

    # Apply your specific trimming logic
    if len(path) > 6:
        return path[:-3]
    return []