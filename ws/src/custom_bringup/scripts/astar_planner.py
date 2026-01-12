import heapq
import math

def a_star_exploration(static_map, costmap, start, goal):
    rows = len(static_map)
    cols = len(static_map[0])
    
    # 1. Start/Goal logic
    if static_map[goal[1]][goal[0]] == 100:
        return []

    frontier_queue = []
    heapq.heappush(frontier_queue, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}
    
    found = False
    while frontier_queue:
        _, current = heapq.heappop(frontier_queue)

        # FIX: Just check for goal. Don't care if it's -1.
        if current == goal:
            found = True
            break

        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
            neighbor = (current[0] + dx, current[1] + dy)

            if not (0 <= neighbor[0] < cols and 0 <= neighbor[1] < rows):
                continue
            
            s_val = static_map[neighbor[1]][neighbor[0]]
            c_val = costmap[neighbor[1]][neighbor[0]]

            # Hard Block only on true walls
            if s_val == 100 or c_val >= 100:
                continue
            
            # Distance cost
            dist = math.sqrt(dx**2 + dy**2)
            
            # MINIMAL penalty for unknown. 
            # If this is too high (like 500), A* will refuse to enter unknown space.
            penalty = 2.0 if s_val == -1 else 0.0
            
            new_cost = cost_so_far[current] + dist + (c_val * 0.1) + penalty

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                h = math.hypot(goal[0]-neighbor[0], goal[1]-neighbor[1])
                heapq.heappush(frontier_queue, (new_cost + h, neighbor))
                came_from[neighbor] = current

    if not found:
        return []

    # 2. Reconstruct
    path = []
    curr = goal
    while curr is not None:
        path.append(curr)
        curr = came_from.get(curr)
    path.reverse()

    # 3. Truncate Safe Path
    safe_path = []
    for node in path:
        if static_map[node[1]][node[0]] == -1:
            break
        safe_path.append(node)

    # 4. Final Snip (Your 5-unit requirement)
    SNIP = 5
    if len(safe_path) > SNIP:
        return safe_path[:-SNIP]
    
    return [start] # Return start rather than empty to show we are "at the edge"