#!/usr/bin/env python
import rospy
import time
from sensor_msgs.msg import Imu
from Transbot_Lib import *
import math

bot = Transbot()
for i in range(5):
    try:
        bot.create_receive_threading()
        break
    except Exception as e:
        print("Serial receive failed, retrying...", e)
        time.sleep(1)
time.sleep(0.5)
bot.set_auto_report_state(True, forever=False)
bot.clear_auto_report_data()
enable = True
bot.set_imu_adjust(enable, forever=False)
time.sleep(0.5)
state = bot.get_imu_state()
print("IMU state:", state)

try:
    while True:
        # Read raw accel & gyro
        a_x, a_y, a_z = bot.get_accelerometer_data()
        g_x, g_y, g_z = bot.get_gyroscope_data()

        # Roll & Pitch from accelerometer
        roll  = math.atan2(a_y, a_z) * 180 / math.pi
        pitch = math.atan2(-a_x, math.sqrt(a_y*a_y + a_z*a_z)) * 180 / math.pi

        print("Roll: {:.2f}, Pitch: {:.2f}".format(roll, pitch))
        print("Gyro: x={:.3f}, y={:.3f}, z={:.3f}".format(g_x, g_y, g_z))
        print("Accel: x={:.3f}, y={:.3f}, z={:.3f}".format(a_x, a_y, a_z))
        print("-----")

        time.sleep(0.1)
except KeyboardInterrupt:
    pass



