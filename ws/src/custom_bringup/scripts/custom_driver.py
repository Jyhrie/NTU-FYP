#!/usr/bin/env python
# encoding: utf-8

import sys
sys.path.append("/home/jetson/Transbot/transbot")
import rospy
import random
import threading
from math import pi
from time import sleep
from transbot_msgs.msg import *
from transbot_msgs.srv import *
from sensor_msgs.msg import Imu
from Transbot_Lib import Transbot
from geometry_msgs.msg import Twist
from arm_transbot import Transbot_ARM
from dynamic_reconfigure.server import Server
from transbot_bringup.cfg import PIDparamConfig

class transbot_driver:
    def __init__(self):
        rospy.on_shutdown(self.cancel)
        self.bot_arm = Transbot_ARM()
        bot_arm_offset = self.bot_arm.get_arm_offset()
        self.bot = Transbot(arm_offset=bot_arm_offset)
        # 弧度转角度
	    # Radians turn angle
        self.RA2DE = 180 / pi
        imu = rospy.get_param("imu", "/transbot/imu")
        vel = rospy.get_param("vel", "/transbot/get_vel")

        self.CameraDevice = rospy.get_param("CameraDevice", "astra")
        self.linear_max = rospy.get_param('~linear_speed_limit', 0.4)
        self.linear_min = rospy.get_param('~linear_speed_limit', 0.0)
        self.angular_max = rospy.get_param('~angular_speed_limit', 2.0)
        self.angular_min = rospy.get_param('~angular_speed_limit', 0.0)
        self.sub_cmd_vel = rospy.Subscriber("/cmd_vel", Twist, self.cmd_vel_callback, queue_size=10)
        self.velPublisher = rospy.Publisher(vel, Twist, queue_size=10)
        self.imuPublisher = rospy.Publisher(imu, Imu, queue_size=10)
        self.dyn_server = Server(PIDparamConfig, self.dynamic_reconfigure_callback)
        self.bot.create_receive_threading()
        self.bot.set_uart_servo_angle(9, 90)

    def cancel(self):
        # self.srv_CurrentAngle.shutdown()
        # self.srv_RGBLight.shutdown()
        # self.srv_Buzzer.shutdown()
        # self.srv_Headlight.shutdown()
        self.velPublisher.unregister()
        self.imuPublisher.unregister()
        # self.volPublisher.unregister()
        self.sub_cmd_vel.unregister()
        # self.sub_TargetAngle.unregister()
        # self.sub_PWMServo.unregister()
        # Always stop the robot when shutting down the node
        rospy.loginfo("Close the robot...")
        rospy.sleep(1)

    def cmd_vel_callback(self, msg):
        # 小车运动控制，订阅者回调函数
	    # Car motion control, subscriber callback function
        if not isinstance(msg, Twist): return
        # 下发线速度和角速度
	    # Issue linear velocity and angular velocity
        velocity = msg.linear.x
        angular = msg.angular.z
        # 小车运动控制,velocity=[-0.45, 0.45], angular=[2, 2]
	    # Trolley motion control,velocity=[-0.45, 0.45], angular=[2, 2]
        if velocity > self.linear_max:
            velocity = self.linear_max
        elif velocity < -self.linear_max:
            velocity = -self.linear_max
        elif -self.linear_min < velocity < 0:
            velocity = -self.linear_min
        elif 0 < velocity < self.linear_min:
            velocity = self.linear_min
        if angular > self.angular_max:
            angular = self.angular_max
        elif angular < -self.angular_max:
            angular = -self.angular_max
        elif -self.angular_min < angular < 0:
            angular = -self.angular_min
        elif 0 < angular < self.angular_min:
            angular = self.angular_min
        # rospy.loginfo("cmd_vel: {}, cmd_ang: {}".format(velocity, angular))
        self.bot.set_car_motion(velocity, angular)

if __name__ == '__main__':
    rospy.init_node("driver_node", anonymous=False)
    try:
        driver = transbot_driver()
        driver.pub_data()
        rospy.spin()
    except:
        rospy.loginfo("Final!!!")