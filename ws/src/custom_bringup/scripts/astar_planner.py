import heapq
import math

def a_star_exploration(static_map, costmap, start, goal):
    """
    static_map: -1 (Unknown), 0 (Free), 100 (Occupied)
    costmap: 0-100 (0: Free, 1-98: Inflated, 99-100: Fatal)
    """
    rows = len(static_map)
    cols = len(static_map[0])
    
    frontier_queue = []
    heapq.heappush(frontier_queue, (0, start))
    
    came_from = {start: None}
    cost_so_far = {start: 0}
    
    # Track the best node reached in case goal is unreachable
    best_node = start
    min_h = math.hypot(goal[0]-start[0], goal[1]-start[1])

    print(f"[A* DEBUG] Starting search from {start} to {goal}")

    while frontier_queue:
        _, current = heapq.heappop(frontier_queue)

        if current == goal:
            print(f"[A* DEBUG] SUCCESS: Goal {goal} reached.")
            return reconstruct_path(came_from, start, goal)

        # Explore 8 neighbors
        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            
            # Debug: Out of Bounds
            if not (0 <= neighbor[0] < cols and 0 <= neighbor[1] < rows):
                # print(f"[A* DEBUG] Boundary: {neighbor} is outside map.")
                continue
            
            s_val = static_map[neighbor[1]][neighbor[0]]
            c_val = costmap[neighbor[1]][neighbor[0]]

            # --- THE NAVIGATION LOGIC ---
            
            # 1. HARD BLOCK: Static Map says it's a wall or unknown
            if s_val == 100 or s_val == -1:
                # print(f"[A* DEBUG] Static Block: {neighbor} s_val={s_val}")
                continue
            
            # 2. FATAL BLOCK: Costmap says robot will hit something 
            if c_val >= 99:
                # print(f"[A* DEBUG] Costmap Block: {neighbor} c_val={c_val}")
                continue
            
            # 3. COST CALCULATION
            dist = math.sqrt(dx**2 + dy**2)
            
            move_cost = dist + (c_val * 10) 
            
            # This logic below will actually never be triggered because 
            # of the 'continue' on s_val == -1 above.
            if s_val == -1:
                move_cost += 5.0 
            
            new_cost = cost_so_far[current] + move_cost

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                h = math.hypot(goal[0]-neighbor[0], goal[1]-neighbor[1])
                
                # Update fallback point
                if h < min_h:
                    min_h = h
                    best_node = neighbor
                
                priority = new_cost + h
                heapq.heappush(frontier_queue, (priority, neighbor))
                came_from[neighbor] = current

    print(f"[A* DEBUG] FAILED: Queue empty. Reached best node {best_node} (Dist to goal: {min_h:.2f})")
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