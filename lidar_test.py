#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import LaserScan
import math

def callback(scan):
    angle = scan.angle_min  # start angle in radians
    print("Angle (deg) | Distance (m)")
    print("---------------------------")
    for r in scan.ranges:
        if math.isinf(r):
            dist = "inf"
        else:
            dist = f"{r:.2f}"
        print(f"{math.degrees(angle):6.1f}° | {dist}")
        angle += scan.angle_increment
    print("\n" + "="*30 + "\n")

def listener():
    rospy.init_node('lidar_echo', anonymous=True)
    rospy.Subscriber("/scan", LaserScan, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
