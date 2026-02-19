import heapq
import numpy as np

def a_star_exploration(static_map_raw, costmap_raw, start, goal,
                       width=800, height=800, fatal_cost=90):

    HEURISTIC_WEIGHT = 0.8
    COSTMAP_WEIGHT   = 2.5
    STATIC_WEIGHT    = 1.0
    DIAG_COST        = 1.414

    sx, sy = start
    gx, gy = goal

    # -- Precompute cost grid once up front --
    cm = np.asarray(costmap_raw,    dtype=np.float32).reshape(height, width)
    sm = np.asarray(static_map_raw, dtype=np.float32).reshape(height, width)

    # True where cell is impassable
    blocked = (cm >= fatal_cost) | (sm >= fatal_cost)

    # Traversal cost per cell (valid cells only)
    cost_grid = 1.0 + (COSTMAP_WEIGHT * cm / 100.0) + (STATIC_WEIGHT * sm / 100.0)
    cost_grid[blocked] = 0.0  # sentinel; we check blocked separately

    if not (0 <= sx < width and 0 <= sy < height):
        return None, False
    if not (0 <= gx < width and 0 <= gy < height):
        return None, False

    NEIGHBORS = [
        ( 1,  0, 1.0),  (-1,  0, 1.0),
        ( 0,  1, 1.0),  ( 0, -1, 1.0),
        ( 1,  1, DIAG_COST), (-1, -1, DIAG_COST),
        ( 1, -1, DIAG_COST), (-1,  1, DIAG_COST),
    ]

    INF = float('inf')
    g_score  = np.full((height, width), INF, dtype=np.float32)
    g_score[sy, sx] = 0.0

    came_from = {}
    open_heap = []
    heapq.heappush(open_heap, (0.0, 0.0, sx, sy))
    closed_set = set()

    best_node = (sx, sy)
    best_h    = abs(sx - gx) + abs(sy - gy)  # cheap init, refined in loop

    def heuristic(x, y):
        dx = abs(x - gx)
        dy = abs(y - gy)
        return (dx + dy) + (DIAG_COST - 2) * min(dx, dy)

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

        g = float(g_score[cy, cx])  # use authoritative value

        for dx, dy, move_cost in NEIGHBORS:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if (nx, ny) in closed_set:
                continue
            if blocked[ny, nx]:
                continue

            tentative_g = g + move_cost * float(cost_grid[ny, nx])
            if tentative_g < g_score[ny, nx]:
                g_score[ny, nx] = tentative_g
                came_from[(nx, ny)] = (cx, cy)
                f_score = tentative_g + HEURISTIC_WEIGHT * heuristic(nx, ny)
                heapq.heappush(open_heap, (f_score, tentative_g, nx, ny))

    return reconstruct(best_node), False