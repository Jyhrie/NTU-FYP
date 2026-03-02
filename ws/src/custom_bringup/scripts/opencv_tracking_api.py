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
            # Calculate Input FPS
            current_time = time.time()
            dt = current_time - self.prev_time
            input_fps = 1.0 / dt if dt > 0 else 0
            self.prev_time = current_time

            try:
                # Convert raw bytes to numpy
                raw_frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
                
                # --- SCALE FOR VNC DISPLAY ---
                # Set this to 0.5 to make the window half-size, or 0.75 for 3/4 size
                scale_percent = 0.5 
                width = int(frame.shape[1] * scale_percent)
                height = int(frame.shape[0] * scale_percent)
                dim = (width, height)
                
                # We resize the frame ONLY for display purposes
                display_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
            except Exception as e:
                rospy.logerr(f"Failed to convert image: {e}")
                return

            start_tick = cv2.getTickCount()

            if self.state == 0:
                cv2.putText(display_frame, "READY - Press 's'", (10, 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("Tracking Test", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    # IMPORTANT: selectROI must use the display_frame
                    self.bbox = cv2.selectROI("Tracking Test", display_frame, False)
                    
                    # We must scale the bbox BACK UP to the original frame size 
                    # so the tracker works on the full resolution image
                    orig_bbox = (
                        int(self.bbox[0] / scale_percent),
                        int(self.bbox[1] / scale_percent),
                        int(self.bbox[2] / scale_percent),
                        int(self.bbox[3] / scale_percent)
                    )
                    
                    self.tracker = cv2.TrackerCSRT_create() 
                    self.tracker.init(frame, orig_bbox)
                    self.state = 1

            elif self.state == 1:
                # Tracker still runs on the ORIGINAL high-res frame for accuracy
                success, box = self.tracker.update(frame)
                
                end_tick = cv2.getTickCount()
                exec_time = (end_tick - start_tick) / cv2.getTickFrequency()
                fps = 1.0 / exec_time if exec_time > 0 else 0

                if success:
                    # Scale the box DOWN for drawing on the display_frame
                    x = int(box[0] * scale_percent)
                    y = int(box[1] * scale_percent)
                    w = int(box[2] * scale_percent)
                    h = int(box[3] * scale_percent)
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    status = "LOCKED"
                    color = (0, 255, 0)
                else:
                    status = "LOST"
                    color = (0, 0, 255)

                # Performance Overlay (adjusted font size for smaller window)
                cv2.putText(display_frame, f"Status: {status}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(display_frame, f"Proc FPS: {fps:.1f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(display_frame, f"Latency: {exec_time*1000:.1f}ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                cv2.imshow("Tracking Test", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.state = 0

if __name__ == '__main__':
    try:
        TrackerTestModule()
        # Keep the window alive
        while not rospy.is_shutdown():
            rospy.sleep(0.1)
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()