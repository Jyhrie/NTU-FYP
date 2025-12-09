from flask import Flask, Response, render_template
import rospy
from nav_msgs.msg import OccupancyGrid
import numpy as np
import cv2
from cv_bridge import CvBridge

app = Flask(__name__)
bridge = CvBridge()

map_img = None

# ROS subscriber callback
def map_callback(msg):
    global map_img
    # Convert occupancy values to 0-255 image
    data = np.array(msg.data, dtype=np.uint8).reshape((msg.info.height, msg.info.width))
    img = 255 - (data * 255 / 100).astype(np.uint8)  # occupied=black, free=white
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    map_img = cv2.flip(img, 0)  # flip if needed

# Stream as MJPEG
def generate_map_stream():
    global map_img
    while True:
        if map_img is not None:
            ret, jpeg = cv2.imencode('.jpg', map_img)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/map_stream')
def map_stream():
    return Response(generate_map_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('map_viewer.html')

if __name__ == '__main__':
    rospy.init_node('map_flask_stream', anonymous=True)
    rospy.Subscriber('/map', OccupancyGrid, map_callback)
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True, use_reloader=False)
