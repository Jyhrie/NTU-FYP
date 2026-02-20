import heapq
import numpy as np


def a_star_exploration(static_map_raw, costmap_raw, start, goal,
                       width=800, height=800, fatal_cost=90,
                       approach_radius=30):

    HEURISTIC_WEIGHT   = 0.05
    COSTMAP_WEIGHT     = 15.0
    STATIC_WEIGHT      = 0.5
    DIAG_COST          = 1.414
    APPROACH_CM_WEIGHT = 2.0
    APPROACH_H_WEIGHT  = 0.05
    LENIENT_COST       = 99

    sx, sy = start
    gx, gy = goal

    if not (0 <= sx < width and 0 <= sy < height):
        return None, False
    if not (0 <= gx < width and 0 <= gy < height):
        return None, False

    cm = np.asarray(costmap_raw,    dtype=np.float32).reshape(height, width)
    sm = np.asarray(static_map_raw, dtype=np.float32).reshape(height, width)

    blocked         = (cm >= fatal_cost)  | (sm >= fatal_cost)
    blocked_lenient = (cm >= LENIENT_COST) | (sm >= fatal_cost)

    cost_wide     = 1.0 + (COSTMAP_WEIGHT    * cm / 100.0) + (STATIC_WEIGHT * sm / 100.0)
    cost_approach = 1.0 + (APPROACH_CM_WEIGHT * cm / 100.0) + (STATIC_WEIGHT * sm / 100.0)

    ys, xs = np.mgrid[0:height, 0:width]
    dist_to_goal = np.sqrt((xs - gx) ** 2 + (ys - gy) ** 2).astype(np.float32)
    proximity_discount = np.ones((height, width), dtype=np.float32)
    within_approach = dist_to_goal <= approach_radius
    proximity_discount[within_approach] = 0.1 + 0.9 * (dist_to_goal[within_approach] / approach_radius)
    cost_approach = cost_approach * proximity_discount

    INF = np.float32(1e30)
    g_score = np.full((height, width), INF,  dtype=np.float32)
    closed  = np.zeros((height, width),       dtype=np.bool_)
    from_x  = np.full((height, width), -1,    dtype=np.int32)
    from_y  = np.full((height, width), -1,    dtype=np.int32)

    g_score[sy, sx] = 0.0

    NEIGHBORS = [
        ( 1,  0, 1.0),       (-1,  0, 1.0),
        ( 0,  1, 1.0),       ( 0, -1, 1.0),
        ( 1,  1, DIAG_COST), (-1, -1, DIAG_COST),
        ( 1, -1, DIAG_COST), (-1,  1, DIAG_COST),
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
        hw  = APPROACH_H_WEIGHT if approaching else HEURISTIC_WEIGHT
        cg  = cost_approach     if approaching else cost_wide
        blk = blocked_lenient   if approaching else blocked

        g = float(g_score[cy, cx])

        for dx, dy, move_cost in NEIGHBORS:
            nx, ny = cx + dx, cy + dy

            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if closed[ny, nx] or blk[ny, nx]:
                continue

            tg = g + move_cost * float(cg[ny, nx])
            if tg < g_score[ny, nx]:
                g_score[ny, nx] = tg
                from_x[ny, nx]  = cx
                from_y[ny, nx]  = cy
                heapq.heappush(heap, (tg + hw * heuristic(nx, ny), tg, nx, ny))

    return reconstruct(best_x, best_y), False