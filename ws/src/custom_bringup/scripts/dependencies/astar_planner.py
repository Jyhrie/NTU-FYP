import heapq
import numpy as np
import math


def a_star_exploration(static_map_raw, costmap_raw, start, goal,
                       width=800, height=800, fatal_cost=70):

    # HIS LOGIC: Massive weight to force hallway centering
    # Even a small costmap value will now outweigh a long physical distance.
    COSTMAP_WEIGHT = 1000.0  
    DIAG_COST      = 1.414
    HEURISTIC_WEIGHT = 0
    
    sx, sy = start
    gx, gy = goal

    # 1. Map Preparation
    # We use numpy for the speed you had originally
    cm = np.asarray(costmap_raw, dtype=np.float32).reshape(height, width)
    sm = np.asarray(static_map_raw, dtype=np.float32).reshape(height, width)

    # 2. Wall Recovery (The "Guy's" robustness shim)
    # If we start or end in a wall, find the nearest walkable cell first
    def get_nearest_walkable(x, y):
        if sm[y, x] < fatal_cost and cm[y, x] < fatal_cost:
            return x, y
        # Quick BFS to find the closest safe cell
        q = [(x, y)]
        visited = {(x, y)}
        while q:
            cx, cy = q.pop(0)
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                nx, ny = cx + dx, cy + dy
                if (0 <= nx < width and 0 <= ny < height) and (nx, ny) not in visited:
                    if sm[ny, nx] < fatal_cost and cm[ny, nx] < fatal_cost:
                        return nx, ny
                    visited.add((nx, ny))
                    q.append((nx, ny))
        return x, y

    sx, sy = get_nearest_walkable(sx, sy)
    gx, gy = get_nearest_walkable(gx, gy)

    # 3. Initialization
    INF = np.float32(1e30)
    g_score = np.full((height, width), INF, dtype=np.float32)
    closed  = np.zeros((height, width), dtype=np.bool_)
    from_x  = np.full((height, width), -1, dtype=np.int32)
    from_y  = np.full((height, width), -1, dtype=np.int32)

    g_score[sy, sx] = 0.0

    # Pre-calculated 8-way neighbors
    NEIGHBORS = [
        (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
        (1, 1, DIAG_COST), (-1, -1, DIAG_COST), (1, -1, DIAG_COST), (-1, 1, DIAG_COST)
    ]

    def heuristic(x, y):
        # Euclidean heuristic matches his logic better for smooth valley following
        return math.sqrt((x - gx)**2 + (y - gy)**2)

    heap = []
    # (f_score, g_score, x, y)
    heapq.heappush(heap, (HEURISTIC_WEIGHT * heuristic(sx, sy), 0.0, sx, sy))

    # 4. Main Loop
    while heap:
        f, g, cx, cy = heapq.heappop(heap)

        if closed[cy, cx]: continue
        closed[cy, cx] = True

        if cx == gx and cy == gy:
            # Reconstruction
            path = []
            while cx != -1:
                path.append((cx, cy))
                px, py = from_x[cy, cx], from_y[cy, cx]
                cx, cy = px, py
            return path[::-1], True

        for dx, dy, move_dist in NEIGHBORS:
            nx, ny = cx + dx, cy + dy

            # Bounds and Lethal Check
            if not (0 <= nx < width and 0 <= ny < height): continue
            if closed[ny, nx] or sm[ny, nx] >= fatal_cost or cm[ny, nx] >= fatal_cost:
                continue

            # THE "NICE MOVEMENT" SECRET:
            # We treat the costmap as an additive penalty. 
            # 1000 * cost means the robot would rather walk 50 meters in a 
            # clear hallway than 1 meter near a wall.
            traversal_cost = move_dist + (COSTMAP_WEIGHT * (cm[ny, nx]))
            tg = g + traversal_cost

            if tg < g_score[ny, nx]:
                g_score[ny, nx] = tg
                from_x[ny, nx] = cx
                from_y[ny, nx] = cy
                # We add the heuristic to guide it, but the G-score (valley) dominates
                heapq.heappush(heap, (tg + heuristic(nx, ny), tg, nx, ny))

    return [], False