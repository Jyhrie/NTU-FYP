import heapq
import numpy as np

def a_star_exploration(static_map_raw, costmap_raw, start, goal,
                       width=800, height=800, fatal_cost=90,
                       approach_radius=15):
    """
    approach_radius: cells from goal where we start relaxing wall avoidance
                     and allow cutting in toward the goal.
    """

    # --- Tuning ---
    HEURISTIC_WEIGHT    = 0.6   # lower = less greedy, wider arcs
    COSTMAP_WEIGHT      = 6.0   # high = strongly avoids inflated zones
    STATIC_WEIGHT       = 0.5   # static map matters less, costmap already has it
    DIAG_COST           = 1.414

    # In the final approach, relax wall penalty so it cuts cleanly to goal
    APPROACH_CM_WEIGHT  = 2.0
    APPROACH_H_WEIGHT   = 1.2   # slightly more greedy in final approach

    sx, sy = start
    gx, gy = goal

    if not (0 <= sx < width and 0 <= sy < height):
        return None, False
    if not (0 <= gx < width and 0 <= gy < height):
        return None, False

    # -- Precompute --
    cm = np.asarray(costmap_raw,    dtype=np.float32).reshape(height, width)
    sm = np.asarray(static_map_raw, dtype=np.float32).reshape(height, width)

    blocked = (cm >= fatal_cost) | (sm >= fatal_cost)

    # Two cost grids: one for wide traversal, one for final approach
    cost_wide     = 1.0 + (COSTMAP_WEIGHT   * cm / 100.0) + (STATIC_WEIGHT * sm / 100.0)
    cost_approach = 1.0 + (APPROACH_CM_WEIGHT * cm / 100.0) + (STATIC_WEIGHT * sm / 100.0)

    cost_wide[blocked]     = 0.0
    cost_approach[blocked] = 0.0

    NEIGHBORS = [
        ( 1,  0, 1.0),  (-1,  0, 1.0),
        ( 0,  1, 1.0),  ( 0, -1, 1.0),
        ( 1,  1, DIAG_COST), (-1, -1, DIAG_COST),
        ( 1, -1, DIAG_COST), (-1,  1, DIAG_COST),
    ]

    INF = float('inf')
    g_score = np.full((height, width), INF, dtype=np.float32)
    g_score[sy, sx] = 0.0

    came_from  = {}
    open_heap  = []
    closed_set = set()

    def heuristic(x, y):
        dx = abs(x - gx)
        dy = abs(y - gy)
        return (dx + dy) + (DIAG_COST - 2.0) * min(dx, dy)

    def in_approach(x, y):
        dx = abs(x - gx)
        dy = abs(y - gy)
        # octile, but a simple euclidean-ish check is fine here
        return (dx * dx + dy * dy) <= approach_radius * approach_radius

    heapq.heappush(open_heap, (0.0, 0.0, sx, sy))

    best_node = (sx, sy)
    best_h    = heuristic(sx, sy)

    def reconstruct(end):
        path = []
        node = end
        while node in came_from:
            path.append(node)
            node = came_from[node]
        path.append((sx, sy))
        path.reverse()
        return path

    while open_heap:
        f, g, cx, cy = heapq.heappop(open_heap)

        if (cx, cy) in closed_set:
            continue
        closed_set.add((cx, cy))

        h = heuristic(cx, cy)
        if h < best_h:
            best_h = h
            best_node = (cx, cy)

        if cx == gx and cy == gy:
            return reconstruct((cx, cy)), True

        g = float(g_score[cy, cx])
        approaching = in_approach(cx, cy)

        hw = APPROACH_H_WEIGHT if approaching else HEURISTIC_WEIGHT
        cg = cost_approach     if approaching else cost_wide

        for dx, dy, move_cost in NEIGHBORS:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if (nx, ny) in closed_set:
                continue
            if blocked[ny, nx]:
                continue

            tentative_g = g + move_cost * float(cg[ny, nx])
            if tentative_g < g_score[ny, nx]:
                g_score[ny, nx] = tentative_g
                came_from[(nx, ny)] = (cx, cy)
                f_score = tentative_g + hw * heuristic(nx, ny)
                heapq.heappush(open_heap, (f_score, tentative_g, nx, ny))

    return reconstruct(best_node), False