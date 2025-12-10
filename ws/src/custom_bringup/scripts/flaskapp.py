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
import heapq

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

    # ---- Find Target ----
    # Note: We pass block_size=5 to match robot footprint
    goal_x, goal_y = mark_top_right_corner(data, img, block_size=5, occ_threshold=50)

    # ---- Draw Route ----
    if goal_x is not None and goal_y is not None:
        draw_route(data, img, (cx, cy), (goal_x, goal_y))

    # Flip vertically to match visualization orientation
    map_img = img

def heuristic(a, b):
    # Euclidean distance
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def draw_route(data, img, start, goal):
    """
    Calculates A* path and draws it on the image.
    start/goal are tuples (x, y).
    """
    path = astar_path(data, start, goal, footprint=5, occ_threshold=50)
    
    if path:
        # Convert path points to numpy array for polylines
        pts = np.array(path, np.int32)
        pts = pts.reshape((-1, 1, 2))
        # Draw the path in Green
        cv2.polylines(img, [pts], False, (0, 255, 0), 2)

def astar_path(data, start, goal, footprint=5, occ_threshold=50):
    """
    A* Pathfinding Algorithm on Occupancy Grid.
    """
    h_map, w_map = data.shape
    
    # Priority Queue: (f_score, g_score, (x, y))
    open_set = []
    heapq.heappush(open_set, (0, 0, start))
    
    came_from = {}
    g_score = {start: 0}
    
    # Offsets for 8-connected neighbors (diagonal movement allowed)
    neighbors = [
        (0, 1), (0, -1), (1, 0), (-1, 0), 
        (1, 1), (1, -1), (-1, 1), (-1, -1)
    ]

    half_fp = footprint // 2

    while open_set:
        _, current_g, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1] # Return reversed path

        x, y = current

        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy

            # 1. Boundary Check
            if 0 <= nx < w_map and 0 <= ny < h_map:
                
                # 2. Footprint/Collision Check
                # We check the 5x5 block around the neighbor node
                y1 = max(0, ny - half_fp)
                y2 = min(h_map, ny + half_fp + 1)
                x1 = max(0, nx - half_fp)
                x2 = min(w_map, nx + half_fp + 1)
                
                # If any pixel in the footprint is occupied, skip this neighbor
                if np.any(data[y1:y2, x1:x2] >= occ_threshold):
                    continue

                # 3. Calculate Cost
                # Cost is 1 for straight, 1.414 for diagonal
                move_cost = 1.414 if dx != 0 and dy != 0 else 1.0
                new_g = current_g + move_cost

                if (nx, ny) not in g_score or new_g < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = new_g
                    f = new_g + heuristic((nx, ny), goal)
                    heapq.heappush(open_set, (f, new_g, (nx, ny)))
                    came_from[(nx, ny)] = current

    return None # No path found

def mark_top_right_corner(data, img, block_size=5, occ_threshold=50):
    """
    Finds the top-right-most block_size x block_size free area
    and marks it on the given OpenCV image.

    Parameters:
        data: 2D numpy array of occupancy grid (0=free, 100=occupied)
        img: BGR image corresponding to the map
        block_size: size of the square to check
        occ_threshold: value >= threshold is considered occupied

    Returns:
        (x, y) center of the selected block in grid coordinates, or None if not found
    """
    h, w = data.shape

    # scan top → bottom, right → left
    for iy in range(h):
        for ix in range(w - block_size, -1, -1):
            block = data[iy:iy+block_size, ix:ix+block_size]
            if np.all(block < occ_threshold):
                # found a free 5x5 block
                cx = ix + block_size // 2
                cy = iy + block_size // 2

                # mark on img (cv2 uses x=cols, y=rows)
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), 2)  # blue circle
                return cx, cy
    return None, None

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
