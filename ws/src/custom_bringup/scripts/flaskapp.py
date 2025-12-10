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
            
            # Run A* on the INFLATED grid
            path = astar_fast(inflated_grid, (cx, cy), (goal_x, goal_y))
            
            if path:
                pts = np.array(path, np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [pts], False, (0, 255, 0), 2)

        # Robot marker
        cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)

        # -- Step E: Update Display --
        # Flip for browser viewing
        final_img = cv2.flip(img, 0)
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
    A* on a binary grid (0=Free, 1=Obstacle).
    No footprint checks needed here because the map is already inflated.
    """
    h, w = grid.shape
    
    # Heuristic: Manhattan distance (Faster than Euclidean)
    def h_cost(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = []
    heapq.heappush(open_set, (0, start))
    
    came_from = {}
    g_score = {start: 0}
    
    # 4-connectivity is much faster than 8-connectivity and usually sufficient
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)] 
    
    # Limit iterations to prevent freeze on unreachable goals
    max_iter = 5000
    iters = 0

    while open_set:
        iters += 1
        if iters > max_iter:
            return None # Path too long or complex

        _, current = heapq.heappop(open_set)

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

            # Bounds check
            if 0 <= nx < w and 0 <= ny < h:
                # Collision Check: Just read the array (O(1))
                if grid[ny, nx] == 1: 
                    continue
                
                new_g = g_score[current] + 1
                
                if (nx, ny) not in g_score or new_g < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = new_g
                    f = new_g + h_cost((nx, ny), goal)
                    heapq.heappush(open_set, (f, (nx, ny)))
                    came_from[(nx, ny)] = current
    return None


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