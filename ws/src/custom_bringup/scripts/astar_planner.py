import heapq
import math

def a_star_exploration(static_map, costmap, start, goal):
    rows = len(static_map)
    cols = len(static_map[0])
    
    # 1. PHYSICAL WALL CHECK
    # Only return empty if the goal is an actual wall (100).
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

            # Map Bounds
            if not (0 <= neighbor[0] < cols and 0 <= neighbor[1] < rows):
                continue
            
            s_val = static_map[neighbor[1]][neighbor[0]]
            c_val = costmap[neighbor[1]][neighbor[0]]

            # DIAGONAL WALL GUARD
            if dx != 0 and dy != 0:
                if static_map[current[1]][current[0] + dx] == 100 or \
                   static_map[current[1] + dy][current[0]] == 100:
                    continue

            # SEARCH THROUGH UNKNOWN: Only 100 (Wall) or lethal costmap blocks the search.
            # We treat -1 (Unknown) as traversable for the sake of finding the path.
            if s_val == 100 or c_val >= 100:
                continue
            
            dist = math.sqrt(dx**2 + dy**2)
            # Give -1 a slight penalty so the robot prefers known paths if they exist
            unknown_penalty = 2.0 if s_val == -1 else 0.0
            move_cost = dist + (c_val * 0.1) + unknown_penalty
            
            new_cost = cost_so_far[current] + move_cost

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                h = math.hypot(goal[0]-neighbor[0], goal[1]-neighbor[1])
                heapq.heappush(frontier_queue, (new_cost + h, neighbor))
                came_from[neighbor] = current
    
    # If the goal is physically walled off
    if not found_goal:
        return []

    # 2. FULL RECONSTRUCTION
    # Reconstruct the path all the way to the goal first.
    full_path = []
    curr = goal
    while curr is not None:
        full_path.append(curr)
        curr = came_from.get(curr)
    full_path.reverse()

    # 3. FRONTIER CLIPPING
    # We keep points as long as they are KNOWN (0).
    # The moment we hit an UNKNOWN (-1) point, we stop.
    safe_path = []
    for node in full_path:
        val = static_map[node[1]][node[0]]
        if val == -1:
            # We reached the edge of the known world. 
            # We stop here so the robot doesn't drive into the dark.
            break
        safe_path.append(node)
        
    # If safe_path is just the start, the robot is already at the frontier.
    return safe_path