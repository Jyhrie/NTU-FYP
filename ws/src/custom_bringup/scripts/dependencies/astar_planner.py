import heapq
import math
import numpy as np

def a_star_exploration(static_map_raw, costmap_raw, start, goal, width=800, height=800):
    static_map = np.array(static_map_raw, dtype=np.int8).reshape((height, width))
    costmap = np.array(costmap_raw, dtype=np.uint8).reshape((height, width))
    
    rows, cols = height, width
    frontier_queue = []
    heapq.heappush(frontier_queue, (0, start))
    
    came_from = {start: None}
    cost_so_far = {start: 0}
    
    # Goal threshold: if we are within 1 pixel, we call it a success
    goal_threshold = 1.5 

    while frontier_queue:
        _, current = heapq.heappop(frontier_queue)

        # Success check with slight tolerance
        dist_to_final = math.hypot(current[0]-goal[0], current[1]-goal[1])
        if dist_to_final <= goal_threshold:
            # If we didn't land exactly on goal, append it for the reconstruct_path
            if current != goal:
                came_from[goal] = current
            return reconstruct_path(came_from, start, goal)

        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
            nx, ny = current[0] + dx, current[1] + dy
            
            if not (1 <= nx < cols - 1 and 1 <= ny < rows - 1):
                continue

            # ---------------- 3x3 SLICE CHECK ----------------
            static_sub = static_map[ny-1:ny+2, nx-1:nx+2]
            cost_sub = costmap[ny-1:ny+2, nx-1:nx+2]

            # CHECK: Is the center pixel itself valid? 
            # We are more lenient with the neighbors than the center.
            if static_map[ny, nx] >= 100 or static_map[ny, nx] <= -1:
                continue

            # Check footprint: If any part is a hard wall (100)
            # We ignore -1 (unknown) in the footprint check so we can approach frontiers
            if np.any(static_sub >= 100) or np.any(cost_sub >= 95):
                continue
            
            max_c_val = np.max(cost_sub)
            # -------------------------------------------------

            # Allergy penalty: 
            # If max_c_val is high, the cost increases exponentially
            penalty = (max_c_val / 5.0) ** 2 

            dist_step = 1.414 if abs(dx) + abs(dy) == 2 else 1.0
            new_cost = cost_so_far[current] + dist_step + penalty

            neighbor = (nx, ny)
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                h = math.hypot(goal[0]-nx, goal[1]-ny)
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