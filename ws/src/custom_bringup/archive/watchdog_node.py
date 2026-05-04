#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

class OptimizedWatchdog:
    def __init__(self):
        rospy.init_node('laser_watchdog', anonymous=True)
        
        self.max_range = rospy.get_param('~max_range', 3.0)
        self.min_range = 0.15  # Ignore Transbot chassis/arm hits
        
        # Pre-calculate indices to save CPU cycles in the callback
        self.indices_calculated = False
        self.front_idx = []
        self.left_idx = []
        self.right_idx = []

        self.sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback, queue_size=1)
        self.pub = rospy.Publisher('/watchdog_status', String, queue_size=1)

    def calculate_indices(self, msg):
        # Determine how many indices correspond to our degree windows
        # Angle = angle_min + (index * angle_increment)
        angles = np.arange(len(msg.ranges)) * msg.angle_increment + msg.angle_min
        
        self.front_idx = np.where((angles >= np.radians(-15)) & (angles <= np.radians(15)))[0]
        self.left_idx = np.where((angles >= np.radians(15)) & (angles <= np.radians(90)))[0]
        self.right_idx = np.where((angles >= np.radians(-90)) & (angles <= np.radians(-15)))[0]
        
        self.indices_calculated = True

    def scan_callback(self, msg):
        if not self.indices_calculated:
            self.calculate_indices(msg)

        # Convert to numpy array for lightning-fast processing
        ranges = np.array(msg.ranges)

        # Replace 0.0 or Inf with a high value so they don't count as 'nearest'
        ranges[ranges < self.min_range] = self.max_range
        ranges[np.isnan(ranges)] = self.max_range
        ranges[np.isinf(ranges)] = self.max_range

        # Get minimums using sliced indexing
        f_dist = np.min(ranges[self.front_idx]) if len(self.front_idx) > 0 else self.max_range
        l_dist = np.min(ranges[self.left_idx]) if len(self.left_idx) > 0 else self.max_range
        r_dist = np.min(ranges[self.right_idx]) if len(self.right_idx) > 0 else self.max_range

        # Calculate influence
        angular_influence = round(r_dist - l_dist, 2)
        
        # Minimalist string formatting
        output = "header: watchdog\ninfo: {{\n  angular_influence: {}\n  forward obstacle distance: {}\n}}".format(
            angular_influence, round(f_dist, 2)
        )

        self.pub.publish(output)

if __name__ == '__main__':
    try:
        OptimizedWatchdog()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass