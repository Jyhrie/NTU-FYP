import heapq
import math

def a_star_exploration(static_map, costmap, start, goal):
    rows = len(static_map)
    cols = len(static_map[0])
    
    # 1. IMMEDIATE CHECK: Is the goal itself a wall?
    if static_map[goal[1]][goal[0]] == 100:
        return []

    frontier_queue = []
    heapq.heappush(frontier_queue, (0, start))
    
    came_from = {start: None}
    cost_so_far = {start: 0}
    
    goal_reached = False

    while frontier_queue:
        _, current = heapq.heappop(frontier_queue)

        if current == goal:
            goal_reached = True
            break

        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
            neighbor = (current[0] + dx, current[1] + dy)

            # Bounds Check
            if not (0 <= neighbor[0] < cols and 0 <= neighbor[1] < rows):
                continue

            # Diagonal Wall-Cutting Prevention
            if dx != 0 and dy != 0:
                if static_map[current[1]][current[0] + dx] == 100 or \
                   static_map[current[1] + dy][current[0]] == 100:
                    continue
            
            s_val = static_map[neighbor[1]][neighbor[0]]
            c_val = costmap[neighbor[1]][neighbor[0]]

            # HARD BLOCK: Walls or lethal obstacles
            if s_val == 100 or c_val >= 99:
                continue
            
            # COST CALCULATION
            dist = math.sqrt(dx**2 + dy**2)
            # Penalize unknown space so we prefer known paths, but can still enter it
            penalty = 500.0 if s_val == -1 else 0.0
            move_cost = dist + (c_val * 0.5) + penalty
            
            new_cost = cost_so_far[current] + move_cost

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                h = math.hypot(goal[0]-neighbor[0], goal[1]-neighbor[1])
                priority = new_cost + h
                heapq.heappush(frontier_queue, (priority, neighbor))
                came_from[neighbor] = current
    
    # 2. If we never reached the goal, it's blocked
    if not goal_reached:
        return []

    # 3. RECONSTRUCTION LOGIC
    # We trace back from the goal, but we only keep the path up to the 
    # FIRST "known free" cell encountered during the trace-back.
    full_path = []
    curr = goal
    while curr is not None:
        full_path.append(curr)
        curr = came_from[curr]
    full_path.reverse()

    # 4. TRUNCATION: Find the last point before the path enters unknown space
    # If the goal is unknown, we find the last index 'i' where map[i] is known (0).
    refined_path = []
    for node in full_path:
        if static_map[node[1]][node[0]] == 0:
            refined_path.append(node)
        else:
            # We hit an unknown cell (-1). Stop here.
            # The last node added to refined_path is your "boundary" cell.
            break
            
    return refined_path