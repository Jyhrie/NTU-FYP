import heapq
import math
import numpy as np

def a_star_exploration(static_map_raw, costmap_raw, start, goal, width=800, height=800):
    # --- JETSON NANO MEMORY FIX ---
    static_map = np.array(static_map_raw).reshape((height, width))
    costmap = np.array(costmap_raw).reshape((height, width))
    
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
            neighbor = (current[0] + dx, current[1] + dy)
            
            if not (0 <= neighbor[0] < cols and 0 <= neighbor[1] < rows):
                continue
            
            s_val = static_map[neighbor[1], neighbor[0]]
            c_val = costmap[neighbor[1], neighbor[0]]

            # 1. HARD BLOCK: Static Map walls or Unknown space
            if s_val >= 100 or s_val <= -1:
                continue

            penalty = c_val

            # 4. TOTAL COST CALCULATION
            dist_step = 1.414 if abs(dx) + abs(dy) == 2 else 1.0
            new_cost = cost_so_far[current] + dist_step * penalty

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                h = math.hypot(goal[0]-neighbor[0], goal[1]-neighbor[1])
                
                # A* weight 1.1 provides a slightly greedy search
                priority = new_cost + (h * 1.1) 
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