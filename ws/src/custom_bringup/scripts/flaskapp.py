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

app = Flask(__name__)

# -- Globals --
latest_map_msg = None
display_img = None
data_lock = threading.Lock()

# ---- 1. Fast Callback (Non-Blocking) ----
def map_callback(msg):
    global latest_map_msg
    # Just store the message and return immediately
    with data_lock:
        latest_map_msg = msg

# ---- 2. Processing Worker (The Brain) ----
def process_map_loop():
    global display_img, latest_map_msg
    
    # Pre-compute the dilation kernel (footprint) once
    # A 5x5 kernel makes obstacles ~2 pixels thicker on all sides
    kernel = np.ones((5, 5), np.uint8) 

    while not rospy.is_shutdown():
        # -- Step A: Get Data Safely --
        msg = None
        with data_lock:
            if latest_map_msg:
                msg = latest_map_msg
        
        if not msg:
            time.sleep(0.1)
            continue

        # -- Step B: Prepare Data --
        w, h = msg.info.width, msg.info.height
        
        # 1. Parse Grid
        raw_grid = np.array(msg.data, dtype=np.int8).reshape((h, w))
        
        # 2. Prepare Visualization Image
        # Convert -1 (unknown) to 50 (gray) for display
        display_data = raw_grid.copy()
        display_data[display_data < 0] = 50
        img = 255 - (display_data * 255 / 100).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # 3. Prepare Logic Map (0=Free, 1=Obstacle)
        # Treat unknowns (-1) and occupied (100) as obstacles
        logic_grid = np.zeros_like(raw_grid, dtype=np.uint8)
        logic_grid[raw_grid >= 50] = 1 # Occupied
        logic_grid[raw_grid == -1] = 1 # Unknown
        
        # ** KEY OPTIMIZATION: INFLATION **
        # "Thicken" walls so we don't need footprint checks in A*
        inflated_grid = cv2.dilate(logic_grid, kernel, iterations=1)

        # -- Step C: Find Points --
        cx = w // 2
        cy = h // 2
        
        # Find Target (using the original visual data for "safe spots")
        goal_x, goal_y = find_safe_corner(raw_grid, block_size=5)

        # -- Step D: Run A* (Fast Version) --
        if goal_x:
            # Draw Target
            cv2.circle(img, (goal_x, goal_y), 5, (255, 0, 0), 2)
            
            # 1. Calculate raw grid path
            raw_path = astar_fast(inflated_grid, (cx, cy), (goal_x, goal_y))
            
            if raw_path:
                # 2. SMOOTH IT
                final_path = smooth_path(raw_path, inflated_grid)
                
                # 3. Draw
                pts = np.array(final_path, np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [pts], False, (0, 255, 0), 2)

        # Robot marker
        cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)

        # -- Step E: Update Display --
        # Flip for browser viewing
        final_img = img
        display_img = final_img
        
        # Cap update rate to save CPU (10 Hz is plenty)
        time.sleep(0.1)

# ---- 3. Optimized Algorithms ----

def find_safe_corner(data, block_size=5):
    """
    Finds top-right free space.
    """
    h, w = data.shape
    occ_threshold = 50
    
    # We can speed this up by skipping pixels (stride=2)
    for iy in range(0, h, 2):
        for ix in range(w - block_size, -1, -2):
            # Quick check center point first
            if data[iy + block_size//2, ix + block_size//2] < occ_threshold:
                # Then check full block
                block = data[iy:iy+block_size, ix:ix+block_size]
                if np.all(block < occ_threshold) and np.all(block >= 0):
                    return ix + block_size // 2, iy + block_size // 2
    return None, None

def astar_fast(grid, start, goal):
    """
    A* that assumes the robot starts facing UP (0, -1).
    Penalizes reversing and encourages straight lines.
    """
    h, w = grid.shape
    
    # 1. HARDCODED START ORIENTATION: UP
    # (dx, dy) = (0, -1)
    start_facing = (0, -1) 

    # Penalties
    REVERSE_PENALTY = 200  # Massive penalty for reversing
    TURN_PENALTY = 10      # Medium penalty for turning
    MOVE_COST = 1          # Base cost

    def h_cost(a, b):
        # Manhattan distance
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # Priority Queue: (f_score, g_score, (x, y), last_move_dir)
    open_set = []
    
    # Initialize with start_facing as the "previous direction"
    heapq.heappush(open_set, (0, 0, start, start_facing))
    
    came_from = {}
    g_score = {start: 0}
    
    # Neighbors: Down, Up, Right, Left
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    max_iter = 5000
    iters = 0

    while open_set:
        iters += 1
        if iters > max_iter:
            return None

        # Pop node
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

            # Bounds Check
            if 0 <= nx < w and 0 <= ny < h:
                # Collision Check (Assuming grid is already inflated/dilated)
                if grid[ny, nx] == 1: 
                    continue
                
                # --- COST LOGIC ---
                extra_cost = 0
                
                # Compare new move (dx, dy) against previous move (last_dir)
                # Dot product:
                #   1  = Continuing Straight
                #   0  = 90 Degree Turn
                #  -1  = 180 Degree Reverse
                alignment = (last_dir[0] * dx) + (last_dir[1] * dy)
                
                if alignment == -1: 
                    extra_cost = REVERSE_PENALTY
                elif alignment == 0:
                    extra_cost = TURN_PENALTY
                
                new_g = current_g + MOVE_COST + extra_cost

                if (nx, ny) not in g_score or new_g < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = new_g
                    f = new_g + h_cost((nx, ny), goal)
                    
                    # Push new state, recording (dx, dy) as the new direction
                    heapq.heappush(open_set, (f, new_g, (nx, ny), (dx, dy)))
                    came_from[(nx, ny)] = current

    return None

def smooth_path(path, grid):
    """
    Simplifies the path by removing unnecessary waypoints.
    If point A and point C have 'Line of Sight' (no obstacles), 
    we skip point B.
    """
    if not path or len(path) < 3:
        return path

    smoothed = [path[0]]
    current_idx = 0
    
    while current_idx < len(path) - 1:
        # Check from the furthest point back towards current
        # We try to connect 'current' to the furthest possible 'next' point
        next_idx = current_idx + 1
        
        for i in range(len(path) - 1, current_idx, -1):
            # Check if we can drive straight from current to i
            start_pt = path[current_idx]
            end_pt = path[i]
            
            if has_line_of_sight(grid, start_pt, end_pt):
                next_idx = i
                break
        
        smoothed.append(path[next_idx])
        current_idx = next_idx

    return smoothed

def has_line_of_sight(grid, p1, p2):
    """
    Checks if a straight line between p1 and p2 hits any obstacles using Bresenham's Line Algorithm.
    Returns True if safe, False if obstacle detected.
    """
    x1, y1 = p1
    x2, y2 = p2
    
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    
    x_step = 1 if x1 < x2 else -1
    y_step = 1 if y1 < y2 else -1
    
    err = dx - dy
    
    x, y = x1, y1
    
    # We check every grid cell the line passes through
    while True:
        # Boundary check (just in case)
        if not (0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]):
            return False
            
        # Collision check (1 = Obstacle)
        if grid[y, x] == 1:
            return False
            
        if x == x2 and y == y2:
            break
            
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += x_step
        if e2 < dx:
            err += dx
            y += y_step
            
    return True


# ---- Boilerplate (Server & Main) ----

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

    # Start threads
    threading.Thread(target=rospy.spin, daemon=True).start()
    threading.Thread(target=process_map_loop, daemon=True).start()

    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True, use_reloader=False)