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
from std_msgs.msg import String

# --- CONFIGURATION ---
ENGINE_PATH     = "/home/jetson/fyp/ws/src/custom_bringup/scripts/models/best.engine"
IMAGE_TOPIC     = "/camera/rgb/image_raw"
SELF_TOPIC      = "/robot/cv"
INPUT_W         = 640
INPUT_H         = 640
ASTRA_PRO_HFOV  = 58.4
PRINT_INTERVAL  = 3.0
CONF_THRESHOLD  = 0.85
FRAME_THRESHOLD = 5
NMS_THRESHOLD   = 0.4  
SKIP_COUNT      = 2    # Number of frames to skip after processing one

class YOLOv8TRTNode:
    def __init__(self):
        rospy.init_node('yolo_trt_monitor', anonymous=True)

        # 1. CUDA Context
        self.ctx = cuda.Device(0).make_context()

        # 2. Load TensorRT Engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(ENGINE_PATH, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()

        # 3. Logic Variables
        self.detection_counter  = 0
        self.last_print_time    = 0
        self.interrupt_sent_sim = False
        self.frame_count        = 0  # <--- Added for frame skipping

        # 4. ROS Subscriber
        self.sub = rospy.Subscriber(IMAGE_TOPIC, Image, self.image_callback,
                                    queue_size=1, buff_size=2**24)
        self.pub = rospy.Publisher(SELF_TOPIC, String, queue_size=1)
        rospy.loginfo("[YOLOv8TRT] Node ready. Skipping %d frames per process.", SKIP_COUNT)

    # -------------------------------------------------------------------------
    def allocate_buffers(self):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        for i in range(self.engine.num_bindings):
            size      = trt.volume(self.engine.get_binding_shape(i))
            dtype     = trt.nptype(self.engine.get_binding_dtype(i))
            host_mem  = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self.engine.binding_is_input(i):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
        return inputs, outputs, bindings, stream

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                        cv2.BORDER_CONSTANT, value=color)
        return img_padded, r, (dw, dh)

    # -------------------------------------------------------------------------
    def image_callback(self, msg):
        # Frame Skipping Logic
        # If SKIP_COUNT = 3: 
        # Frame 0: Process (0 % 4 == 0)
        # Frame 1: Skip
        # Frame 2: Skip
        # Frame 3: Skip
        # Frame 4: Process (4 % 4 == 0)
        if self.frame_count % (SKIP_COUNT + 1) != 0:
            self.frame_count += 1
            return 
        
        self.frame_count += 1
        
        self.ctx.push()
        try:
            current_time = time.time()

            # --- Convert ROS Image to BGR ---
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            if msg.encoding == 'rgb8':
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            display_img = frame.copy()
            orig_h, orig_w = frame.shape[:2]

            # --- Letterbox and normalize ---
            blob_img, ratio, (dw, dh) = self.letterbox(frame, (INPUT_W, INPUT_H))
            blob = cv2.cvtColor(blob_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            blob = np.ascontiguousarray(np.transpose(blob, (2, 0, 1))).ravel()

            # --- Inference ---
            np.copyto(self.inputs[0]['host'], blob)
            cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
            self.stream.synchronize()

            # --- Post-processing ---
            raw = self.outputs[0]['host']
            num_classes = raw.size // 8400 - 4
            num_classes = max(num_classes, 1)
            output = raw.reshape(4 + num_classes, 8400).T

            scores = np.max(output[:, 4:], axis=1)
            mask = scores > CONF_THRESHOLD
            filtered_output = output[mask]
            filtered_scores = scores[mask]

            boxes = []
            if len(filtered_output) > 0:
                for det in filtered_output:
                    cx, cy, bw, bh = det[:4]
                    cx = (cx - dw) / ratio
                    cy = (cy - dh) / ratio
                    bw /= ratio
                    bh /= ratio
                    left = int(cx - bw / 2)
                    top  = int(cy - bh / 2)
                    boxes.append([left, top, int(bw), int(bh)])

                indices = cv2.dnn.NMSBoxes(boxes, filtered_scores.tolist(), CONF_THRESHOLD, NMS_THRESHOLD)

                if len(indices) > 0:
                    for i in indices.flatten():
                        conf = float(filtered_scores[i])
                        left, top, bw_box, bh_box = boxes[i]

                        if conf > CONF_THRESHOLD:
                            self.detection_counter += 1
                            color = (0, 255, 0)
                            if self.detection_counter >= FRAME_THRESHOLD:
                                color = (0, 255, 255)

                                #Sending Data
                                ros_time_str = "{}.{:09d}".format(msg.header.stamp.secs, msg.header.stamp.nsecs)
                                detection_data = {
                                    "x_start": left,
                                    "x_len": bw_box,
                                    "y_start": top,
                                    "y_len": bh_box,
                                    "conf": round(float(conf), 3),
                                    "ros_time": ros_time_str
                                }
                                json_payload = json.dumps(detection_data)
                                self.pub.publish(json_payload)


                            cv2.rectangle(display_img, (left, top), (left + bw_box, top + bh_box), color, 2)
                            cv2.putText(display_img, f"obj {conf:.2f}", (left, top - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                self.detection_counter = 0

            # cv2.imshow("Astra Pro Vision", display_img)
            # cv2.waitKey(1)

        finally:
            self.ctx.pop()

    def __del__(self):
        try:
            self.ctx.detach()
        except Exception:
            pass

if __name__ == '__main__':
    node = YOLOv8TRTNode()
    rospy.spin()