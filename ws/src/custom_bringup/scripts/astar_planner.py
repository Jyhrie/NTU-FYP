import heapq
import math

def a_star_exploration(static_map, costmap, start, goal):
    rows = len(static_map)
    cols = len(static_map[0])
    
    frontier_queue = []
    heapq.heappush(frontier_queue, (0, start))
    
    came_from = {start: None}
    cost_so_far = {start: 0}
    
    # Track the best node reached that is NOT unknown and NOT occupied
    best_node = start
    # Initialize min_h with a very large number if the start is far from goal
    min_h = math.hypot(goal[0]-start[0], goal[1]-start[1])

    while frontier_queue:
        _, current = heapq.heappop(frontier_queue)

        # If we reached the goal and it's a valid cell, return immediately
        if current == goal and static_map[goal[1]][goal[0]] != -1:
            return reconstruct_path(came_from, start, goal)

        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            
            if not (0 <= neighbor[0] < cols and 0 <= neighbor[1] < rows):
                continue
            
            s_val = static_map[neighbor[1]][neighbor[0]]
            c_val = costmap[neighbor[1]][neighbor[0]]

            # 1. HARD BLOCK: Only actual walls or lethal obstacles
            if s_val == 100 or c_val >= 99:
                continue
            
            # 2. COST CALCULATION
            dist = math.sqrt(dx**2 + dy**2)
            move_cost = dist + (c_val * 0.5)
            
            # 3. UNKNOWN PENALTY (Treat as dangerous, but passable)
            if s_val == -1:
                move_cost += 500.0 
            
            new_cost = cost_so_far[current] + move_cost

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                h = math.hypot(goal[0]-neighbor[0], goal[1]-neighbor[1])
                
                # --- FALLBACK LOGIC ---
                # Update best_node ONLY if neighbor is KNOWN and FREE
                # This ensures the path never ends at -1
                if s_val == 0 and h < min_h:
                    min_h = h
                    best_node = neighbor
                
                priority = new_cost + h
                heapq.heappush(frontier_queue, (priority, neighbor))
                came_from[neighbor] = current

    # If goal unreachable or is -1, return path to the closest known free cell
    return reconstruct_path(came_from, start, best_node)

def reconstruct_path(came_from, start, goal):
    path = []
    curr = goal
    if goal not in came_from: return []
    while curr != start:
        path.append(curr)
        curr = came_from[curr]
    path.append(start)
    path.reverse()
    return path