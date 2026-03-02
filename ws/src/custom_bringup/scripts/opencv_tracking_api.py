#!/usr/bin/env python3
import rospy
import cv2
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class TrackerTestModule:
    def __init__(self):
        rospy.init_node('tracker_test_module')
        self.bridge = CvBridge()
        
        # State: 0=Waiting, 1=Tracking
        self.state = 0 
        self.tracker = None
        self.bbox = None
        
        # Performance metrics
        self.prev_time = 0
        
        self.img_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)
        rospy.loginfo("Testing Module Ready. Press 's' to select the object.")

    def callback(self, msg):
        # Calculate Input FPS (Camera side)
        current_time = time.time()
        input_fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time

        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        start_tick = cv2.getTickCount()

        if self.state == 0:
            cv2.putText(frame, "READY - Press 's' to select", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Tracking Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                self.bbox = cv2.selectROI("Tracking Test", frame, False)
                # You can change 'CSRT' to 'KCF' here to compare speed
                self.tracker = cv2.TrackerCSRT_create() 
                self.tracker.init(frame, self.bbox)
                self.state = 1

        elif self.state == 1:
            success, box = self.tracker.update(frame)
            
            # Calculate Processing Time (Nano Latency)
            end_tick = cv2.getTickCount()
            exec_time = (end_tick - start_tick) / cv2.getTickFrequency()
            fps = 1.0 / exec_time

            if success:
                x, y, w, h = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                status = "LOCKED"
                color = (0, 255, 0)
            else:
                status = "LOST"
                color = (0, 0, 255)

            # Performance Overlay
            cv2.putText(frame, f"Status: {status}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"Proc FPS: {fps:.1f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Latency: {exec_time*1000:.1f}ms", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow("Tracking Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to reset
                self.state = 0

if __name__ == '__main__':
    TrackerTestModule()
    rospy.spin()