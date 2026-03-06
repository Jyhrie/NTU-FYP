#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import Image
from scipy import ndimage  # Standard for blob analysis without OpenCV

class BlobCentroidEstimator:
    def __init__(self):
        rospy.init_node('blob_distance_node', anonymous=True)
        self.image_sub = rospy.Subscriber("camera/depth/image_raw", Image, self.depth_callback)
        
        # Your 'bad' bounding box [xmin, ymin, xmax, ymax]
        self.bbox = [150, 100, 450, 400] 

    def depth_callback(self, msg):
        # 1. Byte-to-Array Conversion (No cv_bridge)
        depth_data = np.frombuffer(msg.data, dtype=np.uint16)
        depth_map = depth_data.reshape((msg.height, msg.width))

        # 2. Extract the noisy Bounding Box region
        xmin, ymin, xmax, ymax = self.bbox
        roi = depth_map[ymin:ymax, xmin:xmax]

        # 3. Create a Mask of the actual object
        # We assume the object is the closest thing in the box (common in Search & Retrieve)
        # Filter 0s (noise) and find the approximate object depth
        valid_depths = roi[roi > 0]
        if valid_depths.size == 0:
            return

        approx_dist = np.median(valid_depths)
        
        # Threshold: Only keep pixels within +/- 5cm (50mm) of that median
        # This effectively "cuts out" the background and the box overshoot
        mask = (roi > (approx_dist - 50)) & (roi < (approx_dist + 50))

        if np.any(mask):
            # 4. Find the Center of Mass (Centroid) of the masked blob
            # relative_center is (row, col) inside the ROI
            rel_y, rel_x = ndimage.center_of_mass(mask)
            
            # 5. Get the "Exact" distance at that specific centroid
            exact_dist_mm = roi[int(rel_y), int(rel_x)]
            
            # If the exact center pixel is a hole (0), fallback to the median of the mask
            if exact_dist_mm == 0:
                exact_dist_mm = np.median(roi[mask])

            # 6. Global Coordinates (for your robot's logic)
            global_x = xmin + rel_x
            global_y = ymin + rel_y

            rospy.loginfo(f"Blob Center: ({global_x:.1f}, {global_y:.1f}) | Distance: {exact_dist_mm/1000.0:.3f}m")
        else:
            rospy.logwarn("Could not isolate object blob within bounding box.")

if __name__ == '__main__':
    node = BlobCentroidEstimator()
    rospy.spin()