#!/usr/bin/env python3
import rospy
import cv2
import time
import numpy as np
from sensor_msgs.msg import Image

class TrackerTestModule:
    def __init__(self):
        rospy.init_node('tracker_test_module')
        
        # Removed CvBridge entirely
        
        # State: 0=Waiting, 1=Tracking
        self.state = 0 
        self.tracker = None
        self.bbox = None
        
        # Performance metrics
        self.prev_time = time.time()
        
        # Subscribers
        self.img_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)
        rospy.loginfo("Testing Module Ready (Bypassing CvBridge). Press 's' to select.")

    def callback(self, msg):
        # Calculate Input FPS (Camera side)
        current_time = time.time()
        dt = current_time - self.prev_time
        input_fps = 1.0 / dt if dt > 0 else 0
        self.prev_time = current_time

        # --- BYPASSING CV_BRIDGE ---
        # Convert raw byte string to numpy array
        try:
            # Astra Pro data is usually uint8. Reshape to (Height, Width, Channels)
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            
            # ROS usually sends RGB8, but OpenCV uses BGR. Swap them:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except Exception as e:
            rospy.logerr(f"Failed to convert image: {e}")
            return

        start_tick = cv2.getTickCount()

        if self.state == 0:
            cv2.putText(frame, "READY - Press 's' to select", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Tracking Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                self.bbox = cv2.selectROI("Tracking Test", frame, False)
                if self.bbox[2] > 0 and self.bbox[3] > 0:
                    # CSRT is accurate but heavier on Nano
                    self.tracker = cv2.TrackerCSRT_create() 
                    self.tracker.init(frame, self.bbox)
                    self.state = 1
                else:
                    rospy.logwarn("Invalid selection. Try again.")

        elif self.state == 1:
            success, box = self.tracker.update(frame)
            
            # Calculate Processing Time (Nano Latency)
            end_tick = cv2.getTickCount()
            exec_time = (end_tick - start_tick) / cv2.getTickFrequency()
            fps = 1.0 / exec_time if exec_time > 0 else 0

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
            cv2.putText(frame, f"Input FPS: {input_fps:.1f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("Tracking Test", frame)
            
            # Press 'q' to reset the tracker
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.state = 0
                self.tracker = None

if __name__ == '__main__':
    try:
        TrackerTestModule()
        # Keep the window alive
        while not rospy.is_shutdown():
            rospy.sleep(0.1)
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()