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
PRINT_INTERVAL = 3.0  
CONF_THRESHOLD = 0.85  
FRAME_THRESHOLD = 5    

class YOLOv8TRTNode:
    def __init__(self):
        rospy.init_node('yolo_trt_monitor', anonymous=True)
        
        # 1. Setup CUDA Context for Threading (Fixes CUDA Error 400)
        self.ctx = cuda.Device(0).make_context()
        
        # 2. Load TensorRT Engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(ENGINE_PATH, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()

        # 3. Logic Variables
        self.detection_counter = 0
        self.last_print_time = 0
        self.interrupt_sent_sim = False 
        
        # ROS Subscriber
        self.sub = rospy.Subscriber(IMAGE_TOPIC, Image, self.image_callback, queue_size=1, buff_size=2**24)
        
        # Pop context so it's available for the callback thread
        self.ctx.pop()
        rospy.loginfo("Monitoring Started. CUDA Context Initialized. Waiting for can...")

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

    def image_callback(self, msg):
        # CRITICAL: Push context to this thread
        self.ctx.push()
        try:
            current_time = time.time()
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            if msg.encoding == 'rgb8':
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            display_img = frame.copy()
            h_orig, w_orig = frame.shape[:2]

            # 1. Preprocessing (Scaling to 640x640)
            blob = cv2.resize(frame, (INPUT_W, INPUT_H))
            blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            blob = np.transpose(blob, (2, 0, 1)).ravel()

            # 2. Inference
            np.copyto(self.inputs[0]['host'], blob)
            cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
            self.stream.synchronize()

            # 3. Post-processing (Shape: [84, 8400] -> [8400, 84])
            output = self.outputs[0]['host'].reshape(84, -1).T
            scores = np.max(output[:, 4:], axis=1)
            mask = scores > 0.4 
            
            filtered_output = output[mask]
            filtered_scores = scores[mask]

            found_high_conf = False
            rects, confs = [], []

            if len(filtered_output) > 0:
                # Factor to scale back to original resolution
                x_factor = w_orig / INPUT_W
                y_factor = h_orig / INPUT_H

                for i in range(len(filtered_output)):
                    cx, cy, bw, bh = filtered_output[i, :4]

                    # Fix for "Small Dots": Check if coordinates are normalized
                    if cx <= 1.0:
                        cx *= INPUT_W; cy *= INPUT_H; bw *= INPUT_W; bh *= INPUT_H

                    # Map to original frame
                    left = int((cx - 0.5 * bw) * x_factor)
                    top = int((cy - 0.5 * bh) * y_factor)
                    width = int(bw * x_factor)
                    height = int(bh * y_factor)

                    rects.append([left, top, width, height])
                    confs.append(float(filtered_scores[i]))

                indices = cv2.dnn.NMSBoxes(rects, confs, 0.4, 0.4)

                if len(indices) > 0:
                    for i in indices.flatten():
                        conf = confs[i]
                        if conf > CONF_THRESHOLD:
                            found_high_conf = True
                            bx, by, bw_box, bh_box = rects[i]
                            
                            center_x_box = bx + bw_box/2
                            angle = ((center_x_box - (w_orig/2)) / w_orig) * ASTRA_PRO_HFOV

                            self.detection_counter += 1
                            if self.detection_counter >= FRAME_THRESHOLD:
                                if not self.interrupt_sent_sim:
                                    print(f"🎯 TARGET LOCKED: Angle {angle:.2f}")
                                    self.interrupt_sent_sim = True

                                if current_time - self.last_print_time > PRINT_INTERVAL:
                                    log_entry = {
                                        "target": "can",
                                        "conf": round(conf, 3),
                                        "angle": round(angle, 2)
                                    }
                                    print(f"📝 DATA LOG: {json.dumps(log_entry)}")
                                    self.last_print_time = current_time

                            color = (0, 255, 0) if self.interrupt_sent_sim else (0, 255, 255)
                            cv2.rectangle(display_img, (bx, by), (bx + bw_box, by + bh_box), color, 2)
                            cv2.putText(display_img, f"can {conf:.2f}", (bx, max(0, by - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if not found_high_conf:
                self.detection_counter = 0

            cv2.imshow("Astra Pro Vision", display_img)
            cv2.waitKey(1)

        finally:
            # CRITICAL: Always pop context
            self.ctx.pop()

    def __del__(self):
        self.ctx.detach()

if __name__ == '__main__':
    node = YOLOv8TRTNode()
    rospy.spin()