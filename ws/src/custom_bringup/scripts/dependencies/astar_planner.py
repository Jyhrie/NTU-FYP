import heapq
import numpy as np


def a_star_exploration(static_map_raw, costmap_raw, start, goal,
                       width=800, height=800, fatal_cost=90,
                       approach_radius=15):

    HEURISTIC_WEIGHT   = 0.4    # low = less greedy, wider arcs
    COSTMAP_WEIGHT     = 10.0   # high = strongly avoids walls
    STATIC_WEIGHT      = 0.5
    DIAG_COST          = 1.414

    APPROACH_CM_WEIGHT = 3.0
    APPROACH_H_WEIGHT  = 0.8

    sx, sy = start
    gx, gy = goal

    if not (0 <= sx < width and 0 <= sy < height):
        return None, False
    if not (0 <= gx < width and 0 <= gy < height):
        return None, False

    cm = np.asarray(costmap_raw,    dtype=np.float32).reshape(height, width)
    sm = np.asarray(static_map_raw, dtype=np.float32).reshape(height, width)

    blocked = (cm >= fatal_cost) | (sm >= fatal_cost)

    if blocked[sy, sx] or blocked[gy, gx]:
        return None, False

    # --- Core fix: cost is relative to destination cell only, floored at 1.0 ---
    # This prevents the planner from treating lateral moves through high-cost
    # zones as "free" just because the start is already expensive.
    cost_wide     = np.maximum(1.0 + (COSTMAP_WEIGHT * cm / 100.0) + (STATIC_WEIGHT * sm / 100.0), 1.0)
    cost_approach = np.maximum(1.0 + (APPROACH_CM_WEIGHT * cm / 100.0) + (STATIC_WEIGHT * sm / 100.0), 1.0)

    # Escape bias: add a one-time penalty for staying in high-cost zones.
    # Computed from the start cell so the planner always wants to move to
    # cheaper space rather than laterally through expensive space.
    start_cm_cost = float(cm[sy, sx])
    escape_bias = np.maximum(start_cm_cost - cm, 0.0) * (-0.5)  # reward moving away from start cost
    cost_wide     = cost_wide     + escape_bias
    cost_approach = cost_approach + escape_bias
    cost_wide     = np.maximum(cost_wide,     1.0)
    cost_approach = np.maximum(cost_approach, 1.0)

    INF = np.float32(1e30)
    g_score = np.full((height, width), INF,  dtype=np.float32)
    closed  = np.zeros((height, width),       dtype=np.bool_)
    from_x  = np.full((height, width), -1,    dtype=np.int32)
    from_y  = np.full((height, width), -1,    dtype=np.int32)

    g_score[sy, sx] = 0.0

    NEIGHBORS = [
        ( 1,  0, 1.0),
        (-1,  0, 1.0),
        ( 0,  1, 1.0),
        ( 0, -1, 1.0),
        ( 1,  1, DIAG_COST),
        (-1, -1, DIAG_COST),
        ( 1, -1, DIAG_COST),
        (-1,  1, DIAG_COST),
    ]

    ar2 = approach_radius * approach_radius

    def heuristic(x, y):
        dx = abs(x - gx)
        dy = abs(y - gy)
        return (dx + dy) + (DIAG_COST - 2.0) * min(dx, dy)

    def reconstruct(ex, ey):
        path = []
        cx, cy = ex, ey
        while cx != -1:
            path.append((cx, cy))
            px, py = int(from_x[cy, cx]), int(from_y[cy, cx])
            cx, cy = px, py
        path.reverse()
        return path

    heap = []
    heapq.heappush(heap, (heuristic(sx, sy), 0.0, sx, sy))

    best_x, best_y = sx, sy
    best_h = heuristic(sx, sy)

    while heap:
        f, g, cx, cy = heapq.heappop(heap)

        if closed[cy, cx]:
            continue
        closed[cy, cx] = True

        h = heuristic(cx, cy)
        if h < best_h:
            best_h = h
            best_x, best_y = cx, cy

        if cx == gx and cy == gy:
            return reconstruct(cx, cy), True

        approaching = ((cx - gx) ** 2 + (cy - gy) ** 2) <= ar2
        hw = APPROACH_H_WEIGHT if approaching else HEURISTIC_WEIGHT
        cg = cost_approach     if approaching else cost_wide

        g = float(g_score[cy, cx])

        for dx, dy, move_cost in NEIGHBORS:
            nx, ny = cx + dx, cy + dy

            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if closed[ny, nx] or blocked[ny, nx]:
                continue

            tg = g + move_cost * float(cg[ny, nx])
            if tg < g_score[ny, nx]:
                g_score[ny, nx] = tg
                from_x[ny, nx]  = cx
                from_y[ny, nx]  = cy
                heapq.heappush(heap, (tg + hw * heuristic(nx, ny), tg, nx, ny))

    return reconstruct(best_x, best_y), False