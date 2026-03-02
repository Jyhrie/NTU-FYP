#!/usr/bin/python3
# ^ This shebang points to the system Python 3.6 where jetson-inference is installed

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image

# NVIDIA Jetson Libraries
import jetson.inference
import jetson.utils

class TransbotVision:
    def __init__(self):
        rospy.init_node('yolo_vision_node', anonymous=True)

        # 1. Load the Engine you just built
        # We point to your .engine file. 
        # 'input_0' is the default input name for YOLOv8 ONNX exports.
        model_path = "/home/jetson/fyp/ws/src/custom_bringup/scripts/models/best.engine"
        
        self.net = jetson.inference.detectNet(argv=[
            '--model=' + model_path, 
            '--input-blob=input_0', 
            '--output-cvg=scores', 
            '--output-bbox=boxes'
        ])

        # 2. Setup ROS Subscriber (Astra Pro topic)
        self.img_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)
        
        # 3. Setup VNC Window
        # This creates a high-performance OpenGL window inside your VNC desktop
        self.display = jetson.utils.videoOutput("display://0")

        rospy.loginfo("YOLO Vision Node Initialized. Engine loaded at 20 FPS.")

    def callback(self, data):
        # Convert ROS Image to Open CV (Manual conversion to avoid cv_bridge issues)
        # We convert the raw buffer to a numpy array, then to a CUDA image
        frame = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        
        # Convert BGR (OpenCV) to RGBA (Jetson Utils expects RGBA)
        frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        
        # Move image to GPU memory
        cuda_mem = jetson.utils.cudaFromNumpy(frame_rgba)

        # 4. Run Inference
        # This draws the boxes and labels directly onto the 'cuda_mem'
        detections = self.net.Detect(cuda_mem)

        # 5. Process Detections for Logic (optional)
        for detection in detections:
            # detection.ClassID, detection.Confidence, detection.Center
            if detection.ClassID == 0: # Assuming your 100plus can is class 0
                rospy.loginfo(f"Target Found! Center X: {detection.Center[0]}")

        # 6. Render to VNC Window
        self.display.Render(cuda_mem)
        self.display.SetStatus(f"YOLOv8 | {self.net.GetNetworkFPS():.0f} FPS")

if __name__ == '__main__':
    try:
        node = TransbotVision()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass