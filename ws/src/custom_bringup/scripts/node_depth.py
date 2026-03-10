#!/usr/bin/env python3
import rospy
import json
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import String
from scipy import ndimage
import math

INPUT_W         = 640
INPUT_H         = 640
ASTRA_PRO_HFOV  = 58.4
ANGLE_OFFSET = 3
ASSUMED_DEPTH_OFFSET = 0.02
CAMERA_DEPTH_OFFSET = 0.13
class BlobCentroidEstimator: 
    def __init__(self):
        rospy.init_node('blob_distance_node', anonymous=True)
        
        self.image_sub = rospy.Subscriber("camera/depth/image_raw", Image, self.depth_callback)
        self.request_sub = rospy.Subscriber("controller/global", String, self.request_callback)
        self.depth_pub = rospy.Publisher("/robot/depth", String, queue_size=10)

        self.fx = (INPUT_W / 2) / math.tan(math.radians(ASTRA_PRO_HFOV / 2))
        self.cx = INPUT_W / 2

        self.latest_depth_msg = None
        self.bbox = None

        print("Depth Node Ready")

    def request_callback(self, msg):
        data = json.loads(msg.data)

        if data.get("header") != "depth_node" or data.get("command") != "check_depth":
            return

        x_start = data["x_start"]
        x_len   = data["x_len"]
        y_start = data["y_start"]
        y_len   = data["y_len"]

        self.bbox = [x_start, y_start, x_start + x_len, y_start + y_len]
        rospy.loginfo(f"Received depth request with bbox: {self.bbox}")

        # Process immediately if we already have a depth frame
        if self.latest_depth_msg is not None:
            self.process_depth()

    def depth_callback(self, msg):
        self.latest_depth_msg = msg

    def process_depth(self):
        
        depth_data = np.frombuffer(self.latest_depth_msg.data, dtype=np.uint16)
        depth_map = depth_data.reshape((self.latest_depth_msg.height, self.latest_depth_msg.width))
        
        xmin, ymin, xmax, ymax = self.bbox
        roi = depth_map[ymin:ymax, xmin:xmax]

        valid_depths = roi[roi > 0]
        if valid_depths.size == 0:
            result = String()
            result.data = json.dumps({
                "header": "depth_reading",
                "dist_m": -1
            })
            self.depth_pub.publish(result)
            return

        approx_dist = np.median(valid_depths)
        mask = (roi > (approx_dist - 50)) & (roi < (approx_dist + 50))

        if np.any(mask):
            rel_y, rel_x = ndimage.center_of_mass(mask)
            exact_dist_mm = roi[int(rel_y), int(rel_x)]

            if exact_dist_mm == 0:
                exact_dist_mm = np.median(roi[mask])

            global_x = xmin + rel_x
            global_y = ymin + rel_y

            angle_deg = math.degrees(math.atan2(global_x - self.cx, self.fx))

            result = String()
            result.data = json.dumps({
                "header": "depth_reading",
                "x": round(global_x, 1),
                "y": round(global_y, 1),
                "dist_m": round(exact_dist_mm / 1000.0, 3) + ASSUMED_DEPTH_OFFSET + CAMERA_DEPTH_OFFSET,
                "angle_deg": -round(angle_deg, 2)  #- ANGLE_OFFSET
            })
            self.depth_pub.publish(result)
            print("Depth Published!")
        else:
            print("No Depth")

if __name__ == '__main__':
    node = BlobCentroidEstimator()
    rospy.spin()