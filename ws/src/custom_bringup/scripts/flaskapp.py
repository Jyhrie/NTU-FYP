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
    cv2.circle(img, (cx, cy), 5, (0, 0, 255), 2)         # outer ring

    # A small red arrow to show robot facing direction (upwards)
    cv2.line(img, (cx, cy), (cx, cy - 15), (0, 0, 255), 2)

    target_ix, target_iy = pick_navigation_to_position(data)

    img_x = target_ix
    img_y = msg.info.height - target_iy  # flip y to match cv2 image coords

    cv2.circle(img, (img_x, img_y), 5, (0, 255, 0), -1)  # filled green circle

    # Flip vertically to match visualization orientation
    map_img = img

def pick_navigation_to_position(self):
    """
    Pick a valid target position for the robot to navigate to.
    Returns (ix, iy) grid coordinates in self.grid.
    """
    # Scan from top-right to bottom-left
    for iy in range(self.n-1, -1, -1):       # top → bottom
        for ix in range(self.n-1, -1, -1):   # right → left
            if self.grid[iy, ix] == 0:       # free cell
                return (ix, iy)
    
    # fallback if no free cell found
    return (self.c, self.c)  # stay in place

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
