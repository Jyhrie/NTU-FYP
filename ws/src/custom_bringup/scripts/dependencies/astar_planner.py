import heapq
import math
import numpy as np

def a_star_exploration(static_map_raw, costmap_raw, start, goal, width=800, height=800):
    # Convert to 2D NumPy arrays
    static_map = np.array(static_map_raw, dtype=np.int8).reshape((height, width))
    costmap = np.array(costmap_raw, dtype=np.uint8).reshape((height, width))
    
    rows, cols = height, width
    frontier_queue = []
    heapq.heappush(frontier_queue, (0, start))
    
    came_from = {start: None}
    cost_so_far = {start: 0}
    
    while frontier_queue:
        _, current = heapq.heappop(frontier_queue)

        if current == goal:
            return reconstruct_path(came_from, start, goal)

        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
            nx, ny = current[0] + dx, current[1] + dy
            
            # 1. Boundary Check for the center pixel
            if not (1 <= nx < cols - 1 and 1 <= ny < rows - 1):
                continue

            # ---------------- 3x3 NUMPY SLICE CHECK ----------------
            # Extract the 3x3 area centered at (ny, nx)
            # Slicing is [row_start:row_end, col_start:col_end]
            static_sub = static_map[ny-1:ny+2, nx-1:nx+2]
            cost_sub = costmap[ny-1:ny+2, nx-1:nx+2]

            # Check if any cell in the 3x3 footprint is a wall or high cost
            # Static Map: 100 is wall, -1 is unknown
            # Costmap: 90+ is the "Never Enter" zone
            if np.any(static_sub >= 100) or np.any(static_sub <= -1):
                continue
            
            # Use the maximum cost in the 3x3 area for the "Allergy" penalty
            max_c_val = np.max(cost_sub)
            # -------------------------------------------------------

            # Penalty calculation (Allergy Logic)
            penalty = max_c_val

            dist_step = 1.414 if abs(dx) + abs(dy) == 2 else 1.0
            new_cost = cost_so_far[current] + dist_step + penalty

            neighbor = (nx, ny)
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                h = math.hypot(goal[0]-nx, goal[1]-ny)
                
                # Priority = total cost + heuristic (weight 1.0 for safety)
                priority = new_cost + h
                heapq.heappush(frontier_queue, (priority, neighbor))
                came_from[neighbor] = current

    return None

def reconstruct_path(came_from, start, goal):
    path = []
    curr = goal
    if goal not in came_from: return []
    while curr != start:
        path.append(curr)
        curr = came_from[curr]
    path.append(start)
    path.reverse()

    if len(path) > 6:
        return path[:-3]
    return []