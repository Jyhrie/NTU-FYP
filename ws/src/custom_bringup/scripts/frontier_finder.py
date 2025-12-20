import rospy
import numpy as np
import cv2
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseArray, Pose

class FrontierGenerator:
    def __init__(self):
        rospy.init_node('frontier_generator')
        # Subscribe to the raw map (better for frontiers than costmap)
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        # Publish coordinates of found frontiers
        self.frontier_pub = rospy.Publisher('/detected_frontiers', PoseArray, queue_size=10)

    def map_callback(self, msg):
        # 1. Convert map to 2D numpy array
        grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        
        # 2. Identify "Known Free" (0) and "Unknown" (-1)
        unknown = np.uint8(grid == -1) * 255
        free = np.uint8(grid == 0) * 255

        # 3. Find the edge where free meets unknown
        # Dilate free space to overlap into unknown space
        kernel = np.ones((3,3), np.uint8)
        dilated_free = cv2.dilate(free, kernel, iterations=1)
        frontier_mask = cv2.bitwise_and(dilated_free, unknown)

        # 4. Find clusters of frontier pixels
        contours, _ = cv2.findContours(frontier_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 5. Convert pixel centers to real-world Meters
        frontier_poses = PoseArray()
        frontier_poses.header.frame_id = "map"
        
        for cnt in contours:
            if cv2.contourArea(cnt) < 5: # Filter out tiny noise
                continue
                
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                # Pixel coordinates
                px = int(M["m10"] / M["m00"])
                py = int(M["m01"] / M["m00"])
                
                # Convert to meters
                pose = Pose()
                pose.position.x = px * msg.info.resolution + msg.info.origin.position.x
                pose.position.y = py * msg.info.resolution + msg.info.origin.position.y
                frontier_poses.poses.append(pose)

        self.frontier_pub.publish(frontier_poses)

if __name__ == '__main__':
    fg = FrontierGenerator()
    rospy.spin()