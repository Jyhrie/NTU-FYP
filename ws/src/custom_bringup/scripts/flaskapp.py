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

# -- Tuning Parameters --
MINI_MAP_WIDTH = 60   # Low res for speed (prevents lag)

def map_callback(msg):
    global latest_map_msg
    with data_lock:
        latest_map_msg = msg

def process_map_loop():
    global display_img, latest_map_msg
    
    # Kernel for obstacle inflation
    kernel = np.ones((3, 3), np.uint8) 
    last_msg_seq = -1

    while not rospy.is_shutdown():
        # -- A. Get Data --
        msg = None
        with data_lock:
            if latest_map_msg:
                msg = latest_map_msg
        
        # Optimization: Skip if map hasn't updated
        if not msg or msg.header.seq == last_msg_seq:
            time.sleep(0.05)
            continue
            
        last_msg_seq = msg.header.seq
        h, w = msg.info.height, msg.info.width
        
        # -- B. Create Mini-Map (Speed Optimization) --
        # We perform all heavy math on this tiny map
        scale = MINI_MAP_WIDTH / float(w)
        small_h = int(h * scale)
        
        raw_grid = np.array(msg.data, dtype=np.int8).reshape((h, w))
        small_grid_raw = cv2.resize(raw_grid, (MINI_MAP_WIDTH, small_h), interpolation=cv2.INTER_NEAREST)
        
        # Binary Map (0=Free, 1=Obstacle)
        small_binary = np.zeros_like(small_grid_raw, dtype=np.uint8)
        small_binary[small_grid_raw >= 50] = 1 
        small_binary[small_grid_raw == -1] = 1
        
        # Inflate obstacles slightly so we don't scrape the wall
        small_inflated = cv2.dilate(small_binary, kernel, iterations=1)

        # -- C. GENERATE WALL-HUGGING COST MAP --
        
        # 1. Distance Cost: "Being far from a wall is expensive"
        # Invert binary so walls are 0 and open space is 255
        invert_for_dist = 255 - (small_binary * 255)
        # Calculate distance to nearest wall for every pixel
        dist_map = cv2.distanceTransform(invert_for_dist, cv2.DIST_L1, 3)
        
        # 2. Left-Side Cost: "Being on the Left is expensive"
        # Creates a gradient from 100 (Left) down to 0 (Right)
        x_indices = np.arange(MINI_MAP_WIDTH)
        row_gradient = (1.0 - (x_indices / float(MINI_MAP_WIDTH))) * 100
        left_bias = np.tile(row_gradient, (small_h, 1))

        # 3. Combine Costs
        # - dist_map * 5.0: Strong pull towards walls
        # - left_bias * 2.0: General push towards the right side
        wall_hug_cost_map = (dist_map * 5.0) + (left_bias * 2.0)

        # -- D. Pathfinding --
        cx, cy = w // 2, h // 2
        start_small = (int(cx * scale), int(cy * scale))
        
        # 1. Find Target: Top Right Corner
        goal_x, goal_y = find_safe_corner(raw_grid, block_size=5)
        
        # Prepare Display Image
        display_data = raw_grid.copy()
        display_data[display_data < 0] = 50
        img = 255 - (display_data * 255 / 100).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1) # Robot

        if goal_x is not None:
            goal_small = (int(goal_x * scale), int(goal_y * scale))
            
            # 2. Run A* with Wall Hugging + No Reversing
            path_small = astar_wall_hugger(small_inflated, wall_hug_cost_map, start_small, goal_small)
            
            if path_small:
                # 3. Smooth Path (Greedy Line of Sight)
                final_path_small = smooth_path_greedy(path_small, small_inflated)
                
                # 4. Scale & Draw
                pts = []
                for (sx, sy) in final_path_small:
                    pts.append([int(sx / scale), int(sy / scale)])
                
                pts = np.array(pts, np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [pts], False, (0, 255, 0), 2)
                cv2.circle(img, (goal_x, goal_y), 5, (255, 0, 0), 2)

        display_img = img
        time.sleep(0.1) 

# -- ALGORITHMS --

def find_safe_corner(data, block_size=5):
    """ Finds top-right free space. """
    h, w = data.shape
    # Optimizing: Stride of 5
    for iy in range(0, h, 5):
        for ix in range(w - block_size, -1, -5):
            if data[iy, ix] < 50: 
                block = data[iy:iy+block_size, ix:ix+block_size]
                if np.all(block < 50) and np.all(block >= 0):
                    return ix + block_size // 2, iy + block_size // 2
    return None, None

def smooth_path_greedy(path, grid):
    """ Fast Line-of-Sight smoothing. """
    if len(path) < 3: return path
    smoothed = [path[0]]
    idx = 0
    while idx < len(path) - 1:
        next_valid = idx + 1
        # Look ahead up to 15 nodes
        for i in range(idx + 2, min(idx + 15, len(path))):
            if has_line_of_sight(grid, path[idx], path[i]):
                next_valid = i
            else:
                break
        smoothed.append(path[next_valid])
        idx = next_valid
    return smoothed

def has_line_of_sight(grid, p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    x_step = 1 if x1 < x2 else -1
    y_step = 1 if y1 < y2 else -1
    err = dx - dy
    x, y = x1, y1
    steps = 0
    # Limit check to 50px to prevent freezing on long lines
    while steps < 50:
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
        steps += 1
    return True

def astar_wall_hugger(binary_grid, cost_map, start, goal):
    """
    A* that:
    1. Bans Reversing (Huge Penalty)
    2. Hugs Walls (Uses cost_map)
    3. Starts Facing UP
    """
    h, w = binary_grid.shape
    start_facing = (0, -1) # Robot is facing UP
    
    max_iter = 4000 
    
    # Costs
    COST_STRAIGHT = 1.0
    COST_DIAGONAL = 1.414
    
    # Penalties
    REVERSE_PENALTY = 1000  # Virtually Impossible to Reverse
    TURN_PENALTY = 5        # Small penalty for turning
    
    def h_cost(a, b): return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = []
    # Queue stores: (f_score, g_score, (x, y), last_move_dir)
    heapq.heappush(open_set, (0, 0, start, start_facing))
    
    came_from = {}
    g_score = {start: 0}
    
    # 8-Way Movement
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    
    iters = 0
    while open_set:
        iters += 1
        if iters > max_iter: return None
        
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
            
            # Boundary & Collision Check
            if 0 <= nx < w and 0 <= ny < h:
                if binary_grid[ny, nx] == 1: continue
                
                # 1. Base Cost
                move_cost = COST_DIAGONAL if (dx and dy) else COST_STRAIGHT
                
                # 2. Directionality Check (Dot Product)
                dot = (last_dir[0] * dx) + (last_dir[1] * dy)
                
                if dot < 0: 
                    # Negative dot product = Reversing (135 or 180 degrees)
                    move_cost += REVERSE_PENALTY
                elif dot == 0:
                    # Zero dot product = 90 degree turn
                    move_cost += TURN_PENALTY
                
                # 3. Environmental Cost (Wall Hugging)
                # This pulls the robot towards the "Cheap" areas (Right Walls)
                move_cost += cost_map[ny, nx]
                
                new_g = current_g + move_cost
                
                if (nx, ny) not in g_score or new_g < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = new_g
                    f = new_g + h_cost((nx, ny), goal)
                    heapq.heappush(open_set, (f, new_g, (nx, ny), (dx, dy)))
                    came_from[(nx, ny)] = current
    return None

# -- Flask & Main --
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

if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    rospy.init_node('map_flask_stream', anonymous=True)
    rospy.Subscriber('/local_costmap', OccupancyGrid, map_callback)
    threading.Thread(target=rospy.spin, daemon=True).start()
    threading.Thread(target=process_map_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True, use_reloader=False)