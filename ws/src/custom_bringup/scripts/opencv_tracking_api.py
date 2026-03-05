#!/usr/bin/python3
import rospy
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import json
import time
from sensor_msgs.msg import Image

# --- CONFIGURATION ---
ENGINE_PATH = "/home/jetson/fyp/ws/src/custom_bringup/scripts/models/best.engine"
IMAGE_TOPIC = "/camera/rgb/image_raw"
INPUT_W, INPUT_H = 640, 640
ASTRA_PRO_HFOV = 58.4 
CONF_THRESHOLD = 0.85  
FRAME_THRESHOLD = 5    
PRINT_INTERVAL = 3.0

class YOLOv8TRTNode:
    def __init__(self):
        rospy.init_node('yolo_trt_monitor', anonymous=True)
        
        # 1. CUDA Context for Threaded Environment
        self.ctx = cuda.Device(0).make_context()
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # 2. Load Engine
        with open(ENGINE_PATH, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()

        # 3. State Variables
        self.detection_counter = 0
        self.last_print_time = 0
        self.interrupt_sent = False 
        
        # 4. Subscriber
        self.sub = rospy.Subscriber(IMAGE_TOPIC, Image, self.image_callback, queue_size=1, buff_size=2**24)
        rospy.loginfo("🚀 YOLOv8 TensorRT Node Active. Monitoring for Target...")

    def allocate_buffers(self):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        for i in range(self.engine.num_bindings):
            size = trt.volume(self.engine.get_binding_shape(i))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self.engine.binding_is_input(i):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
        return inputs, outputs, bindings, stream

    def preprocess(self, img):
        """Letterbox resize with padding to 640x640."""
        h, w = img.shape[:2]
        r = min(INPUT_H / h, INPUT_W / w)
        new_unpad = (int(round(w * r)), int(round(h * r)))
        dw = (INPUT_W - new_unpad[0]) / 2
        dh = (INPUT_H - new_unpad[1]) / 2
        
        resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        # NCHW format
        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1)).ravel()
        return blob, r, (dw, dh)

    def image_callback(self, msg):
        self.ctx.push()
        try:
            # 1. Image Conversion
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            if msg.encoding == 'rgb8':
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img_h, img_w = frame.shape[:2]

            # 2. Preprocess
            blob, ratio, (dw, dh) = self.preprocess(frame)

            # 3. Inference
            np.copyto(self.inputs[0]['host'], blob)
            cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
            self.stream.synchronize()

            # 4. Corrected Post-processing
            # YOLOv8 typically outputs [1, 4+C, 8400]
            output = self.outputs[0]['host'].reshape(-1, 8400) 
            if output.shape[0] > 100: # Safety check for transpose
                output = output.T 
            
            # Now output is [4 + classes, 8400]
            boxes = output[:4, :].T  # [8400, 4]
            scores = np.max(output[4:, :], axis=0) # [8400]
            
            # Filtering
            mask = scores > 0.4
            valid_boxes = boxes[mask]
            valid_scores = scores[mask]

            found_target = False
            if len(valid_boxes) > 0:
                rects = []
                for i in range(len(valid_boxes)):
                    cx, cy, bw, bh = valid_boxes[i]
                    # Map from 640x640 to Original Image
                    real_x = (cx - dw) / ratio
                    real_y = (cy - dh) / ratio
                    real_w = bw / ratio
                    real_h = bh / ratio
                    
                    # Convert to Top-Left corner (Integers for OpenCV)
                    l = int(real_x - (real_w / 2))
                    t = int(real_y - (real_h / 2))
                    rects.append([l, t, int(real_w), int(real_h)])

                indices = cv2.dnn.NMSBoxes(rects, valid_scores.tolist(), 0.4, 0.4)

                if len(indices) > 0:
                    for i in indices.flatten():
                        conf = valid_scores[i]
                        if conf > CONF_THRESHOLD:
                            found_target = True
                            rx, ry, rw, rh = rects[i]
                            
                            self.detection_counter += 1
                            angle = (((rx + rw/2) - (img_w/2)) / img_w) * ASTRA_PRO_HFOV

                            if self.detection_counter >= FRAME_THRESHOLD:
                                if not self.interrupt_sent:
                                    rospy.loginfo(f"TARGET DETECTED: {angle:.2f} deg")
                                    self.interrupt_sent = True
                                
                                if time.time() - self.last_print_time > PRINT_INTERVAL:
                                    print(f"LOG: {{'conf': {conf:.2f}, 'angle': {angle:.2f}}}")
                                    self.last_print_time = time.time()

                            # --- Visualization ---
                            color = (0, 255, 0) if self.interrupt_sent else (0, 255, 255)
                            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), color, 2)
                            cv2.putText(frame, f"can {conf:.2f}", (rx, max(20, ry - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if not found_target:
                self.detection_counter = 0

            cv2.imshow("Detection Feed", frame)
            cv2.waitKey(1)

        finally:
            self.ctx.pop()

    def __del__(self):
        if hasattr(self, 'ctx'):
            self.ctx.detach()

if __name__ == '__main__':
    try:
        node = YOLOv8TRTNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass