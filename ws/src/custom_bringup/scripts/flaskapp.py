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

    h = msg.info.height
    w = msg.info.width

    # Convert occupancy values to numpy array
    data = np.array(msg.data, dtype=np.int16).reshape((h, w))

    # Create color image
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # ---- Color Mapping ----
    # 0 = free = white
    img[data == 0] = (255, 255, 255)

    # 100 = occupied = black
    img[data == 100] = (0, 0, 0)

    # 1 = blue
    img[data == 1] = (255, 0, 0)

    # 2 = green
    img[data == 2] = (0, 255, 0)

    # 3 = red
    img[data == 3] = (0, 0, 255)

    # 3 = red
    img[data == 5] = (255, 218, 117)

    # 3 = red
    img[data == 6] = (255, 117, 188)

    img[data == 99] = (0, 150, 255)

    # Unknown values (<0) = dark gray
    img[data < 0] = (60, 60, 60)

    map_img = img


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
    rospy.Subscriber('/map', OccupancyGrid, map_callback)

    # Start ROS spinning in background thread
    spin_thread = threading.Thread(target=rospy.spin)
    spin_thread.daemon = True
    spin_thread.start()

    # Start Flask app
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True, use_reloader=False)