import heapq
import math

def a_star_exploration(static_map, costmap, start, goal):
    rows = len(static_map)
    cols = len(static_map[0])

    SNIP_DISTANCE = 5  # Stop 5 units before unknown/wall
    FATAL_COST = 100   # Hard collision value
    
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

    # 3. IDENTIFY DANGER ZONE
    # Find the first index in the path that is UNKNOWN or FATAL
    first_unsafe_idx = len(full_path)
    for i, node in enumerate(full_path):
        s_val = static_map[node[1]][node[0]]
        c_val = costmap[node[1]][node[0]]
        
        if s_val == -1 or c_val >= FATAL_COST:
            first_unsafe_idx = i
            break
            
    # 4. APPLY THE 5-UNIT SNIP
    # We want to end the path 5 indices BEFORE the first unsafe cell.
    # We use max(1, ...) so it at least returns the start point if we're already close.
    end_idx = max(1, first_unsafe_idx - SNIP_DISTANCE)
    
    # Special Case: If the whole path is shorter than the snip distance,
    # and the goal is already in unknown space, we just stay put (return start).
    if first_unsafe_idx < SNIP_DISTANCE:
        return [start]
        
    return full_path[:end_idx]