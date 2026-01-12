import heapq
import math

def a_star_exploration(static_map, costmap, start, goal):
    rows = len(static_map)
    cols = len(static_map[0])
    
    # Check if goal is an actual wall immediately
    if static_map[goal[1]][goal[0]] == 100:
        return []

    frontier_queue = []
    heapq.heappush(frontier_queue, (0, start))
    
    came_from = {start: None}
    cost_so_far = {start: 0}
    
    found_goal = False

    while frontier_queue:
        _, current = heapq.heappop(frontier_queue)

        if current == goal:
            found_goal = True
            break

        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
            neighbor = (current[0] + dx, current[1] + dy)

            if not (0 <= neighbor[0] < cols and 0 <= neighbor[1] < rows):
                continue
            
            s_val = static_map[neighbor[1]][neighbor[0]]
            c_val = costmap[neighbor[1]][neighbor[0]]

            # Diagonal wall-cutting prevention
            if dx != 0 and dy != 0:
                if static_map[current[1]][current[0] + dx] == 100 or \
                   static_map[current[1] + dy][current[0]] == 100:
                    continue

            # BLOCK ONLY IF WALL: Treat unknown (-1) as traversable during search
            if s_val == 100 or c_val >= 99:
                continue
            
            dist = math.sqrt(dx**2 + dy**2)
            # Apply a penalty to unknown space so we prefer known paths
            penalty = 10.0 if s_val == -1 else 0.0
            move_cost = dist + (c_val * 0.5) + penalty
            
            new_cost = cost_so_far[current] + move_cost

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                h = math.hypot(goal[0]-neighbor[0], goal[1]-neighbor[1])
                heapq.heappush(frontier_queue, (new_cost + h, neighbor))
                came_from[neighbor] = current
    
    # If the goal is blocked off by walls, return an empty list
    if not found_goal:
        return []

    # --- RECONSTRUCTION & TRUNCATION ---
    path = []
    curr = goal
    # 1. Trace back the full path from goal to start
    while curr is not None:
        path.append(curr)
        curr = came_from.get(curr)
    path.reverse()

    # 2. Iterate forward and stop at the last known 'free' (0) cell
    truncated_path = []
    for node in path:
        # If we hit unknown space, we stop.
        # This returns the path up to the LAST cell before entering -1.
        if static_map[node[1]][node[0]] == -1:
            break
        truncated_path.append(node)
        
    return truncated_path