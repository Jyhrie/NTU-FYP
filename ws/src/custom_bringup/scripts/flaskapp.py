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

def rgb(r, g, b):
    return (b, g, r)

def map_callback(msg):
    global map_img

    h = msg.info.height
    w = msg.info.width

    data = np.array(msg.data, dtype=np.int16).reshape((h, w))

    # -----------------------------
    # Mask unknown cells (-1)
    # -----------------------------
    unknown_mask = data < 0

    # Clamp values to [0, 255]
    clamped = np.clip(data, 0, 255).astype(np.uint8)

    unknown_mask   = data == 255
    lethal_mask    = data == 254
    inscribed_mask = data == 253

    # -----------------------------
    # Apply colormap (spectrum)
    # -----------------------------
    # Options: COLORMAP_JET, TURBO, HSV, VIRIDIS
    color = cv2.applyColorMap(clamped, cv2.COLORMAP_TURBO)

    # -----------------------------
    # Force unknowns to gray
    # -----------------------------

    color[inscribed_mask] = (0, 255, 0)    # green
    color[lethal_mask]    = (0, 0, 255)    # red
    color[unknown_mask]   = (60, 60, 60)   # gray

    print("Unique costmap values:", np.unique(data))

    map_img = color


# ---- MJPEG Stream Generator ----
def generate_map_stream():
    global map_img
    while True:
        if map_img is not None:
            ret, jpeg = cv2.imencode('.png', map_img)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n\r\n')
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
    rospy.Subscriber('/move_base/global_costmap/costmap', OccupancyGrid, map_callback)

    # Start ROS spinning in background thread
    spin_thread = threading.Thread(target=rospy.spin)
    spin_thread.daemon = True
    spin_thread.start()

    # Start Flask app
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True, use_reloader=False)