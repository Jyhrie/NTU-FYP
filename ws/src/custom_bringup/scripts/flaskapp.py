#!/usr/bin/env python3
from flask import Flask, Response, render_template
import rospy
from nav_msgs.msg import OccupancyGrid
import numpy as np
import cv2
from cv_bridge import CvBridge
import threading
import time
import signal
import sys

from scipy.ndimage import maximum_filter
import heapq
import math

app = Flask(__name__)
bridge = CvBridge()
map_img = None

def map_callback(msg):
    global map_img

    # Convert occupancy grid to 0-255 image
    data = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))

    # Unknown (-1) = darker gray
    data[data < 0] = 50

    img = 255 - (data * 255 / 100).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # ---- Mark Robot in the Center ----
    cx = msg.info.width // 2
    cy = msg.info.height // 2

    # A small red circle to mark robot position
    # A small red arrow to show robot facing direction (upwards)
    cv2.line(img, (cx, cy), (cx, cy - 5), (0, 0, 255), 1)

    draw_route(data, img, block_size=5, occ_threshold=50)

    # Flip vertically to match visualization orientation
    map_img = img

def draw_route(data, img, robot_size=(0.3,0.4), resolution=0.02, block_size=5, occ_threshold=50):

    footprint_w = int(robot_size[0]/resolution)
    footprint_h = int(robot_size[1]/resolution)
    inflated = maximum_filter(data, size=(footprint_h, footprint_w))

    # Convert inflated to binary 0=free, 100=occupied for A*
    inflated_bin = np.where(inflated >= occ_threshold, 100, 0)

    h, w = data.shape

    # --- pick target ---
    target = None
    for iy in range(h):
        for ix in range(w-block_size, -1, -1):
            block = data[iy:iy+block_size, ix:ix+block_size]
            if np.all(block < occ_threshold):
                cx = ix + block_size//2
                cy = iy + block_size//2
                target = (cx, cy)
                break
        if target:
            break
    if target is None:
        return None

    start = (w//2, h//2)
    goal = target

    path = astar(inflated_bin, start, goal)
    if path:
        for i in range(len(path)-1):
            x1, y1 = path[i]
            x2, y2 = path[i+1]
            cv2.line(img, (x1, y1), (x2, y2), (0,255,0), 1)

    cv2.circle(img, target, 5, (255,0,0), 2)
    return target, path


# --- A* helper ---
def heuristic(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def astar(grid, start, goal):
    h, w = grid.shape
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start, None))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        f, g, current, parent = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while parent:
                path.append(parent)
                parent = came_from[parent]
            return path[::-1]

        if current in came_from:
            continue
        came_from[current] = parent

        x, y = current
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and grid[ny, nx] < occ_threshold:
                    tentative_g = g + math.hypot(dx, dy)
                    if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                        g_score[(nx, ny)] = tentative_g
                        heapq.heappush(open_set, (tentative_g + heuristic((nx, ny), goal), tentative_g, (nx, ny), current))
    return None

# ---- MJPEG Stream Generator ----
def generate_map_stream():
    global map_img
    while True:
        if map_img is not None:
            ret, jpeg = cv2.imencode('.jpg', map_img)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            time.sleep(0.05)  # ~20 FPS
        else:
            time.sleep(0.05)

# ---- Flask Routes ----
@app.route('/map_stream')
def map_stream():
    return Response(generate_map_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('map_viewer.html')  # Your HTML template in ./templates/

# ---- Ctrl+C Handler ----
def signal_handler(sig, frame):
    print("\nShutting down Flask + ROS map streamer...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# ---- Main ----
if __name__ == '__main__':
    # Initialize ROS node
    rospy.init_node('map_flask_stream', anonymous=True)
    rospy.Subscriber('/local_costmap', OccupancyGrid, map_callback)

    # Start ROS spinning in background thread
    spin_thread = threading.Thread(target=rospy.spin)
    spin_thread.daemon = True
    spin_thread.start()

    # Start Flask app
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True, use_reloader=False)
