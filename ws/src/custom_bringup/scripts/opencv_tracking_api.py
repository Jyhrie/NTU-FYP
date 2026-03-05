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
ENGINE_PATH     = "/home/jetson/fyp/ws/src/custom_bringup/scripts/models/best.engine"
IMAGE_TOPIC     = "/camera/rgb/image_raw"
INPUT_W         = 640
INPUT_H         = 640
ASTRA_PRO_HFOV  = 58.4
PRINT_INTERVAL  = 3.0
CONF_THRESHOLD  = 0.85
FRAME_THRESHOLD = 5


class YOLOv8TRTNode:
    def __init__(self):
        rospy.init_node('yolo_trt_monitor', anonymous=True)

        # 1. Setup CUDA Context for Threading
        self.ctx = cuda.Device(0).make_context()

        # 2. Load TensorRT Engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(ENGINE_PATH, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()

        # Log output shape so we can verify
        out_shape = self.engine.get_binding_shape(1)
        rospy.loginfo("[YOLOv8TRT] Engine output binding shape: %s", out_shape)

        # 3. Logic Variables
        self.detection_counter  = 0
        self.last_print_time    = 0
        self.interrupt_sent_sim = False
        self.frame_count        = 0  # for frame skipping

        # 4. ROS Subscriber
        self.sub = rospy.Subscriber(IMAGE_TOPIC, Image, self.image_callback,
                                    queue_size=1, buff_size=2**24)
        rospy.loginfo("[YOLOv8TRT] Node ready. Waiting for frames...")

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

    # -------------------------------------------------------------------------

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        shape     = img.shape[:2]  # (h, w)
        r         = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw        = new_shape[1] - new_unpad[0]
        dh        = new_shape[0] - new_unpad[1]
        dw       /= 2
        dh       /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top    = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left   = int(round(dw - 0.1))
        right  = int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=color)
        return img, r, (dw, dh)

    # -------------------------------------------------------------------------

    def image_callback(self, msg):
        # Only run inference every 3rd frame to reduce GPU load and prevent overcurrent
        self.frame_count += 1
        if self.frame_count % 3 != 0:
            return

        self.ctx.push()
        try:
            current_time = time.time()
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            if msg.encoding == 'rgb8':
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            display_img  = frame.copy()
            orig_h, orig_w = frame.shape[:2]

            # 1. Letterbox Preprocessing
            blob_img, ratio, (dw, dh) = self.letterbox(frame, (INPUT_W, INPUT_H))
            blob = cv2.cvtColor(blob_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            blob = np.ascontiguousarray(np.transpose(blob, (2, 0, 1))).ravel()

            # 2. Inference
            np.copyto(self.inputs[0]['host'], blob)
            cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
            self.stream.synchronize()

            # 3. Post-processing
            # Raw TRT output for YOLOv8n 1-class: (1, 5, 8400)
            # After reshape + transpose: (8400, 5) → [cx, cy, bw, bh, conf]
            raw    = self.outputs[0]['host']
            num_classes = raw.size // 8400 - 4   # derive: total / 8400 - 4 coords
            num_classes = max(num_classes, 1)
            output = raw.reshape(4 + num_classes, 8400).T  # (8400, 4+nc)

            # Scores = max class confidence
            scores = np.max(output[:, 4:], axis=1)
            mask   = scores > 0.1
            filtered_output = output[mask]
            filtered_scores = scores[mask]

            found_high_conf_in_frame = False

            if len(filtered_output) > 0:
                boxes = []
                for det in filtered_output:
                    # Coordinates are in letterboxed 640x640 space
                    cx, cy, bw, bh = det[0], det[1], det[2], det[3]

                    # Step 1: remove padding offset
                    cx -= dw
                    cy -= dh

                    # Step 2: scale back to original image size
                    cx /= ratio
                    cy /= ratio
                    bw /= ratio
                    bh /= ratio

                    # Convert centre to top-left corner
                    left = int(cx - bw / 2)
                    top  = int(cy - bh / 2)
                    boxes.append([left, top, int(bw), int(bh)])

                indices = cv2.dnn.NMSBoxes(boxes, filtered_scores.tolist(), 0.1, 0.4)

                if len(indices) > 0:
                    for i in indices.flatten():
                        conf = float(filtered_scores[i])
                        if conf > CONF_THRESHOLD:
                            found_high_conf_in_frame = True
                            left, top, bw_box, bh_box = boxes[i]

                            self.detection_counter += 1
                            center_x_box = left + bw_box / 2
                            angle = ((center_x_box - (orig_w / 2)) / orig_w) * ASTRA_PRO_HFOV

                            if self.detection_counter >= FRAME_THRESHOLD:
                                if not self.interrupt_sent_sim:
                                    rospy.loginfo("[YOLOv8TRT] TARGET LOCKED: Angle %.2f", angle)
                                    self.interrupt_sent_sim = True

                                if current_time - self.last_print_time > PRINT_INTERVAL:
                                    log_entry = {
                                        "ros_time": f"{msg.header.stamp.secs}.{msg.header.stamp.nsecs:09d}",
                                        "target":   "object",
                                        "conf":     round(conf, 3),
                                        "angle":    round(angle, 2)
                                    }
                                    rospy.loginfo("[YOLOv8TRT] DATA LOG: %s", json.dumps(log_entry))
                                    self.last_print_time = current_time

                            color = (0, 255, 0) if self.interrupt_sent_sim else (0, 255, 255)
                            cv2.rectangle(display_img,
                                          (left, top),
                                          (left + bw_box, top + bh_box),
                                          color, 2)
                            cv2.putText(display_img, f"object {conf:.2f}",
                                        (left, top - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if not found_high_conf_in_frame:
                self.detection_counter = 0

            cv2.imshow("Astra Pro Vision", display_img)
            cv2.waitKey(1)

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