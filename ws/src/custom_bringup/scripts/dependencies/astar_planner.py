import heapq
import math
import numpy as np

def a_star_exploration(static_map_raw, costmap_raw, start, goal, width=800, height=800):
    # --- JETSON NANO MEMORY FIX ---
    # Convert flat tuples to 2D NumPy arrays (O(N) conversion, but allows O(1) indexing)
    static_map = np.array(static_map_raw).reshape((height, width))
    costmap = np.array(costmap_raw).reshape((height, width))
    
    rows, cols = height, width
    
    frontier_queue = []
    # Priority, current_node
    heapq.heappush(frontier_queue, (0, start))
    
    came_from = {start: None}
    cost_so_far = {start: 0}
    
    best_node = start
    min_h = math.hypot(goal[0]-start[0], goal[1]-start[1])

    #print("[A* DEBUG] Starting search from {} to {}".format(start, goal))

    while frontier_queue:
        _, current = heapq.heappop(frontier_queue)

        if current == goal:
            #print("[A* DEBUG] SUCCESS: Goal {} reached.".format(goal))
            return reconstruct_path(came_from, start, goal)

        # Explore 8 neighbors
        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            
            # Boundary check
            if not (0 <= neighbor[0] < cols and 0 <= neighbor[1] < rows):
                continue
            
            # FAST INDEXING with NumPy
            s_val = static_map[neighbor[1], neighbor[0]]
            c_val = costmap[neighbor[1], neighbor[0]]

            # 1. HARD BLOCK: Static Map says it's a wall or unknown
            if s_val >= 100 or s_val <= -1:
                continue

            # 2. CALCULATE DISTANCE TO GOAL (Idea 3)
            # We calculate this once per neighbor to use in both blocking and heuristic
            dist_to_goal = math.hypot(goal[0]-neighbor[0], goal[1]-neighbor[1])

            # 3. FATAL BLOCK: High costmap values
            # Normally, we block if c_val >= 90. 
            # But if we are within 10 pixels of the goal, we allow it (Deflate Inflation).
            if c_val >= 90:
                if dist_to_goal > 10:  # Adjust this '10' based on your inflation radius
                    continue
                else:
                    # Give it a high penalty so it's a "last resort" but NOT a hard block
                    inflation_penalty = c_val * 50 
            else:
                # Standard penalty: your existing formula
                inflation_penalty = (c_val ** 3) / 3

            # 4. TOTAL COST CALCULATION
            dist_step = 1.414 if abs(dx) + abs(dy) == 2 else 1.0
            new_cost = cost_so_far[current] + dist_step + inflation_penalty

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                h = math.hypot(goal[0]-neighbor[0], goal[1]-neighbor[1])
                
                if h < min_h:
                    min_h = h
                    best_node = neighbor
                
                # A* weight: 1.0 is standard. 0.2 makes it behave more like Dijkstra (safer but slower)
                priority = new_cost + (h * 1.1) 
                heapq.heappush(frontier_queue, (priority, neighbor))
                came_from[neighbor] = current

    #print("[A* DEBUG] FAILED: Queue empty. Reached best node {} (Dist: {:.2f})".format(best_node, min_h))
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

    if len(path) > 7:
        return path
    return []