
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
import math

LINEAR_SPEED = 0.15      # m/s
ANGULAR_SPEED = 0.05      # rad/s
DESIRED_DISTANCE = 0.5   # meters from wall
RATE_HZ = 10

NODE_NAME = 'mapper_node'
CMD_TOPIC = '/cmd_vel'
SCAN_TOPIC = '/scan'
MAP_TOPIC = '/map'

class Mapper:

    def __init__(self):
        rospy.init_node(NODE_NAME, anonymous=False)
        rospy.loginfo("--- Custom Mapping Algoritm ---")

        self.cmd_pub = rospy.Publisher(CMD_TOPIC, Twist, queue_size=1)
        rospy.Subscriber(SCAN_TOPIC, LaserScan, self.scan_callback)
        rospy.Subscriber(MAP_TOPIC, OccupancyGrid, self.map_callback)

    def scan_callback(self, msg):
        self.scan = msg

    def map_callback(self, msg):
        self.map_data = msg

    def set_start_position(self, x, y, theta):
        pass

    def publish_move_command(self, linear, angular):
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        self.cmd_pub.publish(twist)

    def get_current_transform(self):
        pass

    def run(self):
        while not rospy.is_shutdown():
            if self.scan is None or self.map_data is None:
                rospy.loginfo("Waiting for SCAN and MAP data...")
                continue
            pass

        scan_len = len(self.scan.ranges)
        left_indices = range(0, int(scan_len * 0.25))
        right_indices = range(int(scan_len * 0.75), scan_len)
        front_indices = range(int(scan_len * 0.45), int(scan_len * 0.55))

        #for now we only care about right indices


#first thing, establish start position

#find right wall

#if wall right is no longer detected, turn in and try to hug will at x-distance away

#continue forward until wall in front is detected

#stabilize at 1m away from wall,

#turn left until wall angle

def start_mapping():
    pass

def start_coroutine():
    pass


if __name__ == '__main__':
    try:
        start_mapping()
    except rospy.ROSInterruptException:
        pass
