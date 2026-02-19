import heapq
import numpy as np


def a_star(static_map_raw,
           costmap_raw,
           start,
           goal,
           width=800,
           height=800,
           fatal_cost=90):

    static_map = np.asarray(static_map_raw, dtype=np.int16).reshape(height, width)
    costmap = np.asarray(costmap_raw, dtype=np.int16).reshape(height, width)

    sx, sy = start
    gx, gy = goal

    # Bounds check
    if not (0 <= sx < width and 0 <= sy < height):
        return np.empty((0, 2), dtype=np.int32), False
    if not (0 <= gx < width and 0 <= gy < height):
        return np.empty((0, 2), dtype=np.int32), False

    # Walkable = free space + not lethal in costmap
    walkable = (static_map == 0) & (costmap < fatal_cost)

    if not walkable[gy, gx]:
        return np.empty((0, 2), dtype=np.int32), False

    # Preallocate
    g_score = np.full((height, width), np.inf, dtype=np.float32)
    closed = np.zeros((height, width), dtype=np.bool_)
    parent = np.full((height, width, 2), -1, dtype=np.int32)

    # Manhattan heuristic
    def heuristic(x, y):
        return abs(x - gx) + abs(y - gy)

    open_heap = []

    g_score[sy, sx] = 0.0
    heapq.heappush(open_heap, (heuristic(sx, sy), sx, sy))

    neighbors = ((1,0), (-1,0), (0,1), (0,-1))

    while open_heap:
        _, x, y = heapq.heappop(open_heap)

        if closed[y, x]:
            continue

        closed[y, x] = True

        # Goal reached
        if x == gx and y == gy:
            path = []
            cx, cy = x, y
            while not (cx == sx and cy == sy):
                path.append((cx, cy))
                cx, cy = parent[cy, cx]
            path.append((sx, sy))
            path.reverse()
            return np.array(path, dtype=np.int32), True

        current_g = g_score[y, x]

        for dx, dy in neighbors:
            nx = x + dx
            ny = y + dy

            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if closed[ny, nx]:
                continue
            if not walkable[ny, nx]:
                continue

            # Costmap-weighted movement
            traversal_cost = 1.0 + (costmap[ny, nx] / 100.0)
            tentative_g = current_g + traversal_cost

            if tentative_g < g_score[ny, nx]:
                g_score[ny, nx] = tentative_g
                parent[ny, nx] = (x, y)
                f_score = tentative_g + heuristic(nx, ny)
                heapq.heappush(open_heap, (f_score, nx, ny))

    return np.empty((0, 2), dtype=np.int32), False
