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
CONF_THRESHOLD = 0.85  # Target confidence from your PC script
FRAME_THRESHOLD = 5    # Required consecutive frames

class YOLOv8TRTNode:
    def __init__(self):
        rospy.init_node('yolo_trt_monitor', anonymous=True)
        
        # 1. Load TensorRT Engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(ENGINE_PATH, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()

        # 2. Logic Variables (Matching your PC script)
        self.detection_counter = 0
        self.last_print_time = 0
        self.interrupt_sent_sim = False # Simulated flag
        
        # 3. ROS Subscriber
        self.sub = rospy.Subscriber(IMAGE_TOPIC, Image, self.image_callback, queue_size=1, buff_size=2**24)
        rospy.loginfo("Monitoring Started. Waiting for 100Plus Can...")

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
        current_time = time.time()
        
        # 1. Conversion
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        if msg.encoding == 'rgb8':
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        display_img = frame.copy()
        h, w, _ = frame.shape
        center_x = w / 2

        # 2. Preprocessing & Inference
        blob = cv2.resize(frame, (INPUT_W, INPUT_H))
        blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)
        blob = blob.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1)).ravel()

        np.copyto(self.inputs[0]['host'], blob)
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()

        # 3. Post-processing (NumPy Vectorized)
        output = self.outputs[0]['host'].reshape(84, -1).T
        scores = np.max(output[:, 4:], axis=1)
        mask = scores > 0.1 # Low threshold for NMS, then we filter by CONF_THRESHOLD
        
        filtered_output = output[mask]
        filtered_scores = scores[mask]

        found_high_conf_in_frame = False

        if len(filtered_output) > 0:
            # Scale boxes
            x_factor, y_factor = w / INPUT_W, h / INPUT_H
            boxes = []
            for det in filtered_output:
                cx, cy, bw, bh = det[:4]
                left = int((cx - 0.5 * bw) * x_factor)
                top = int((cy - 0.5 * bh) * y_factor)
                boxes.append([left, top, int(bw * x_factor), int(bh * y_factor)])

            # NMS
            indices = cv2.dnn.NMSBoxes(boxes, filtered_scores.tolist(), 0.1, 0.4)

            if len(indices) > 0:
                for i in indices.flatten():
                    conf = filtered_scores[i]
                    if conf > CONF_THRESHOLD:
                        found_high_conf_in_frame = True
                        bx, by, bw, bh = boxes[i]
                        cx_box = bx + bw/2
                        
                        self.detection_counter += 1
                        angle = ((cx_box - center_x) / w) * ASTRA_PRO_HFOV

                        if self.detection_counter >= FRAME_THRESHOLD:
                            # Simulation of the 'interrupt' logic
                            if not self.interrupt_sent_sim:
                                print(f"🎯 TARGET LOCKED: Angle {angle:.2f}")
                                self.interrupt_sent_sim = True

                            # Throttled JSON Logging
                            if current_time - self.last_print_time > PRINT_INTERVAL:
                                log_entry = {
                                    "ros_time": f"{msg.header.stamp.secs}.{msg.header.stamp.nsecs:09d}",
                                    "target": "can",
                                    "conf": round(float(conf), 3),
                                    "status": "DETECTED",
                                    "angle": round(angle, 2),
                                    "dims": {"w": bw, "h": bh}
                                }
                                print(f"\n📝 DATA LOG:\n{json.dumps(log_entry, indent=2)}")
                                self.last_print_time = current_time

                        # Visualization (Green if locked, Yellow if pending)
                        color = (0, 255, 0) if self.interrupt_sent_sim else (0, 255, 255)
                        cv2.rectangle(display_img, (bx, by), (bx + bw, by + bh), color, 2)
                        cv2.putText(display_img, f"can {conf:.2f}", (bx, by - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if not found_high_conf_in_frame:
            self.detection_counter = 0

        cv2.imshow("Astra Pro Vision", display_img)
        cv2.waitKey(1)

if __name__ == '__main__':
    node = YOLOv8TRTNode()
    rospy.spin()
    cv2.destroyAllWindows()