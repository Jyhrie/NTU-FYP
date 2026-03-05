#!/usr/bin/python3
import rospy
import cv2
import numpy as np
import onnxruntime as ort
import json
import time
from sensor_msgs.msg import Image

# --- CONFIGURATION ---
ONNX_PATH = "/home/jetson/fyp/ws/src/custom_bringup/scripts/models/best.onnx"
IMAGE_TOPIC = "/camera/rgb/image_raw"
INPUT_W, INPUT_H = 640, 640
ASTRA_PRO_HFOV = 58.4 
CONF_THRESHOLD = 0.85  

class YOLOv8ONNXNode:
    def __init__(self):
        rospy.init_node('yolo_onnx_monitor', anonymous=True)
        
        # 1. Load ONNX Session (Uses CUDAExecutionProvider for Jetson GPU)
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(ONNX_PATH, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        # 2. State Variables
        self.detection_counter = 0
        self.interrupt_sent = False 
        
        self.sub = rospy.Subscriber(IMAGE_TOPIC, Image, self.image_callback, queue_size=1)
        rospy.loginfo("✅ YOLOv8 ONNX Node Active. Running on GPU...")

    def preprocess(self, img):
        h, w = img.shape[:2]
        r = min(INPUT_H / h, INPUT_W / w)
        new_unpad = (int(round(w * r)), int(round(h * r)))
        dw, dh = (INPUT_W - new_unpad[0]) / 2, (INPUT_H - new_unpad[1]) / 2
        
        resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        # Convert to float and NCHW format
        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0) # Add batch dimension: [1, 3, 640, 640]
        return blob, r, (dw, dh)

    def image_callback(self, msg):
        try:
            # 1. Image Conversion
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            if msg.encoding == 'rgb8':
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img_h, img_w = frame.shape[:2]

            # 2. Preprocess
            blob, ratio, (dw, dh) = self.preprocess(frame)

            # 3. Inference
            outputs = self.session.run(None, {self.input_name: blob})
            output = np.squeeze(outputs[0]) # Shape: [84, 8400]

            # 4. Post-processing
            # YOLOv8: first 4 rows are cx, cy, w, h
            boxes = output[:4, :].T 
            scores = np.max(output[4:, :], axis=0)
            
            mask = scores > 0.4
            valid_boxes = boxes[mask]
            valid_scores = scores[mask]

            if len(valid_boxes) > 0:
                rects = []
                for i in range(len(valid_boxes)):
                    cx, cy, bw, bh = valid_boxes[i]
                    
                    # Convert from 640 model space to real pixel space
                    # ONNX usually gives raw pixels in 640 scale
                    real_x = (cx - dw) / ratio
                    real_y = (cy - dh) / ratio
                    real_w = bw / ratio
                    real_h = bh / ratio
                    
                    l = int(real_x - (real_w / 2))
                    t = int(real_y - (real_h / 2))
                    rects.append([l, t, int(real_w), int(real_h)])

                indices = cv2.dnn.NMSBoxes(rects, valid_scores.tolist(), 0.4, 0.4)

                if len(indices) > 0:
                    for i in indices.flatten():
                        conf = valid_scores[i]
                        if conf > CONF_THRESHOLD:
                            rx, ry, rw, rh = rects[i]
                            
                            # Drawing the box
                            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
                            cv2.putText(frame, f"can {conf:.2f}", (rx, max(20, ry - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("ONNX Detection", frame)
            cv2.waitKey(1)

        except Exception as e:
            rospy.logerr(f"ONNX Error: {e}")

if __name__ == '__main__':
    node = YOLOv8ONNXNode()
    rospy.spin()