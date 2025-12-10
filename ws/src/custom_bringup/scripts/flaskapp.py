#!/usr/bin/env python3
from flask import Flask, Response, render_template
import rospy
from nav_msgs.msg import OccupancyGrid
import numpy as np
import cv2
import threading
import time
import signal
import sys
import heapq
import math

app = Flask(__name__)

# -- Globals --
latest_map_msg = None
display_img = None
data_lock = threading.Lock()

# ==========================================
# 1. ROS Callback (Fast & Non-Blocking)
# ==========================================
def map_callback(msg):
    global latest_map_msg
    with data_lock:
        latest_map_msg = msg

# ==========================================
# 2. Main Processing Worker (The Brain)
# ==========================================
def process_map_loop():
    global display_img, latest_map_msg
    
    # Kernel for simple obstacle inflation
    kernel = np.ones((3, 3), np.uint8) 

    while not rospy.is_shutdown():
        # -- A. Get Data Safely --
        msg = None
        with data_lock:
            if latest_map_msg:
                msg = latest_map_msg
        
        if not msg:
            time.sleep(0.1)
            continue

        h, w = msg.info.height, msg.info.width
        raw_grid = np.array(msg.data, dtype=np.int8).reshape((h, w))
        
        # -- B. Create Mini-Map (Performance Optimization) --
        # We process on a smaller map to make A* instant, even on large maps.
        TARGET_WIDTH = 200
        scale = TARGET_WIDTH / float(w)
        small_h = int(h * scale)
        
        # Resize raw grid (Nearest Neighbor preserves sharp walls)
        small_grid_raw = cv2.resize(raw_grid, (TARGET_WIDTH, small_h), interpolation=cv2.INTER_NEAREST)
        
        # Create Binary Obstacle Map (0=Free, 1=Obstacle)
        small_binary = np.zeros_like(small_grid_raw, dtype=np.uint8)
        small_binary[small_grid_raw >= 50] = 1   # Occupied
        small_binary[small_grid_raw == -1] = 1   # Unknown
        
        # Inflate walls slightly for safety
        small_inflated = cv2.dilate(small_binary, kernel, iterations=1)

        # -- C. Generate "Wall Hugging" Cost Map --
        # 1. Calculate distance to nearest wall
        invert_for_dist = 255 - (small_binary * 255)
        dist_map = cv2.distanceTransform(invert_for_dist, cv2.DIST_L2, 3)
        
        # 2. Create "Right Side" Gradient (Left=High Cost, Right=Low Cost)
        col_grid = np.indices((small_h, TARGET_WIDTH))[1]
        left_bias = (1.0 - (col_grid / float(TARGET_WIDTH))) * 100 
        
        # 3. Combine: Penalize open space (distance) and left side
        # Increase '2.0' to hug walls tighter. Increase '1.0' to hate the left side more.
        wall_hug_cost_map = (dist_map * 2.0) + (left_bias * 1.0)

        # -- D. Pathfinding Setup --
        cx, cy = w // 2, h // 2
        
        # Convert Start/Goal to Mini-Map Coordinates
        start_small = (int(cx * scale), int(cy * scale))
        goal_x, goal_y = find_safe_corner(raw_grid, block_size=5) # Find goal on big map
        
        # -- E. Visualization Image --
        display_data = raw_grid.copy()
        display_data[display_data < 0] = 50
        img = 255 - (display_data * 255 / 100).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Draw Robot
        cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)

        # -- F. Run Weighted A* & Smooth --
        if goal_x is not None:
            goal_small = (int(goal_x * scale), int(goal_y * scale))
            
            # 1. Run A* on Mini-Map with Wall Hugging Costs
            path_small = astar_weighted(small_inflated, wall_hug_cost_map, start_small, goal_small)
            
            if path_small:
                # 2. Smooth the path (Line of Sight optimization)
                final_path_small = smooth_path(path_small, small_inflated)
                
                # 3. Scale points back UP to full resolution for drawing
                final_path_big = []
                for (sx, sy) in final_path_small:
                    final_path_big.append((int(sx / scale), int(sy / scale)))
                
                # 4. Draw Path
                pts = np.array(final_path_big, np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [pts], False, (0, 255, 0), 2)
                cv2.circle(img, (goal_x, goal_y), 5, (255, 0, 0), 2)

        # Flip for browser viewing
        display_img = cv2.flip(img, 0)
        time.sleep(0.1) # 10 FPS cap to save CPU

# ==========================================
# 3. Algorithms: A*, Smoothing, Target
# ==========================================

def find_safe_corner(data, block_size=5):
    """ Finds top-right free space on the full-res map. """
    h, w = data.shape
    occ_threshold = 50
    # Scan Top -> Bottom, Right -> Left
    for iy in range(0, h, 2):
        for ix in range(w - block_size, -1, -2):
            if data[iy + block_size//2, ix + block_size//2] < occ_threshold:
                block = data[iy:iy+block_size, ix:ix+block_size]
                if np.all(block < occ_threshold) and np.all(block >= 0):
                    return ix + block_size // 2, iy + block_size // 2
    return None, None

def astar_weighted(binary_grid, cost_map, start, goal):
    """
    Weighted A* with Diagonal Support + Anti-Reverse + Wall Hugging.
    """
    h, w = binary_grid.shape
    
    # Robot starts facing UP (0, -1)
    start_facing = (0, -1) 

    # Base Costs
    COST_STRAIGHT = 1.0
    COST_DIAGONAL = 1.414
    
    # Penalties
    REVERSE_PENALTY = 200 
    TURN_PENALTY = 10     
    SLIGHT_TURN = 2       

    def heuristic(a, b):
        return math.hypot(b[0] - a[0], b[1] - a[1])

    open_set = []
    heapq.heappush(open_set, (0, 0, start, start_facing))
    
    came_from = {}
    g_score = {start: 0}
    
    neighbors = [
        (0, 1), (0, -1), (1, 0), (-1, 0),
        (1, 1), (1, -1), (-1, 1), (-1, -1)
    ]
    
    max_iter = 6000
    iters = 0

    while open_set:
        iters += 1
        if iters > max_iter:
            return None

        f, current_g, current, last_dir = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        x, y = current

        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy

            if 0 <= nx < w and 0 <= ny < h:
                # Collision Check
                if binary_grid[ny, nx] == 1: 
                    continue
                
                # Base Movement Cost
                move_cost = COST_DIAGONAL if (dx != 0 and dy != 0) else COST_STRAIGHT
                
                # Direction Penalties
                dot = (last_dir[0] * dx) + (last_dir[1] * dy)
                if dot < 0: move_cost += REVERSE_PENALTY    # Reverse
                elif dot == 0: move_cost += TURN_PENALTY    # 90 deg turn
                elif (dx, dy) != last_dir: move_cost += SLIGHT_TURN # 45 deg turn

                # Environmental Cost (Wall Hugging)
                environmental_cost = cost_map[ny, nx]
                
                new_g = current_g + move_cost + environmental_cost

                if (nx, ny) not in g_score or new_g < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = new_g
                    f = new_g + heuristic((nx, ny), goal)
                    heapq.heappush(open_set, (f, new_g, (nx, ny), (dx, dy)))
                    came_from[(nx, ny)] = current
    return None

def smooth_path(path, grid):
    """ Removes unnecessary waypoints using Line of Sight. """
    if not path or len(path) < 3:
        return path

    smoothed = [path[0]]
    current_idx = 0
    
    while current_idx < len(path) - 1:
        next_idx = current_idx + 1
        for i in range(len(path) - 1, current_idx, -1):
            if has_line_of_sight(grid, path[current_idx], path[i]):
                next_idx = i
                break
        smoothed.append(path[next_idx])
        current_idx = next_idx

    return smoothed

def has_line_of_sight(grid, p1, p2):
    """ Bresenham's Line Algorithm for collision checking. """
    x1, y1 = p1
    x2, y2 = p2
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    x_step = 1 if x1 < x2 else -1
    y_step = 1 if y1 < y2 else -1
    err = dx - dy
    x, y = x1, y1
    
    while True:
        if not (0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]): return False
        if grid[y, x] == 1: return False
        if x == x2 and y == y2: break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += x_step
        if e2 < dx:
            err += dx
            y += y_step
    return True

# ==========================================
# 4. Flask & Main
# ==========================================

def generate_map_stream():
    while True:
        if display_img is not None:
            ret, jpeg = cv2.imencode('.jpg', display_img)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            time.sleep(0.05)
        else:
            time.sleep(0.1)

@app.route('/map_stream')
def map_stream():
    return Response(generate_map_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('map_viewer.html')

def signal_handler(sig, frame):
    sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    rospy.init_node('map_flask_stream', anonymous=True)
    rospy.Subscriber('/local_costmap', OccupancyGrid, map_callback)

    # Start Worker Threads
    threading.Thread(target=rospy.spin, daemon=True).start()
    threading.Thread(target=process_map_loop, daemon=True).start()

    # Start Flask
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True, use_reloader=False)