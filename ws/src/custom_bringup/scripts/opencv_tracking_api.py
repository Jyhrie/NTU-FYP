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
        
        # 1. Setup CUDA Context
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
        
        # 4. ROS Subscriber
        self.sub = rospy.Subscriber(IMAGE_TOPIC, Image, self.image_callback, queue_size=1, buff_size=2**24)
        rospy.loginfo("Monitoring Started. Waiting for Target...")

    def allocate_buffers(self):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        for i in range(self.engine.num_bindings):
            shape = self.engine.get_binding_shape(i)
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self.engine.binding_is_input(i):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
        return inputs, outputs, bindings, stream

    def letterbox(self, img, new_shape=(640, 640)):
        shape = img.shape[:2]  # [h, w]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
        
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img, r, (dw, dh)

    def image_callback(self, msg):
        self.ctx.push() 
        try:
            current_time = time.time()
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            if msg.encoding == 'rgb8':
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            display_img = frame.copy()
            h, w = frame.shape[:2]

            # --- Preprocessing ---
            blob_img, ratio, (dw, dh) = self.letterbox(frame, (INPUT_W, INPUT_H))
            blob = cv2.cvtColor(blob_img, cv2.COLOR_BGR2RGB)
            blob = blob.astype(np.float32) / 255.0
            blob = np.transpose(blob, (2, 0, 1)).reshape(1, 3, INPUT_H, INPUT_W)
            input_data = np.ascontiguousarray(blob)

            # --- Inference ---
            np.copyto(self.inputs[0]['host'], input_data.ravel())
            cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
            self.stream.synchronize()

            # --- Post-processing ---
            # YOLOv8 output is [1, 4+num_classes, 8400]
            output = self.outputs[0]['host'].reshape(self.engine.get_binding_shape(1)).squeeze()
            if output.shape[0] > output.shape[1]: # Handle potential transpose issues
                output = output.T

            # Extract boxes and scores
            boxes_raw = output[:4, :].T  # [8400, 4] -> cx, cy, bw, bh
            scores_raw = np.max(output[4:, :], axis=0) # [8400]
            
            mask = scores_raw > 0.4 # Preliminary filter to speed up NMS
            boxes_filt = boxes_raw[mask]
            scores_filt = scores_raw[mask]

            found_high_conf_in_frame = False

            if len(boxes_filt) > 0:
                final_boxes = []
                for i in range(len(boxes_filt)):
                    cx, cy, bw_model, bh_model = boxes_filt[i]
                    
                    # Convert from Letterbox 640x640 to Original Frame Pixels
                    # 1. Subtract padding, 2. Divide by scale ratio
                    real_cx = (cx - dw) / ratio
                    real_cy = (cy - dh) / ratio
                    real_w = bw_model / ratio
                    real_h = bh_model / ratio

                    # Convert Center-XYWH to TopLeft-XYWH
                    left = int(real_cx - (real_w / 2))
                    top = int(real_cy - (real_h / 2))
                    final_boxes.append([left, top, int(real_w), int(real_h)])

                indices = cv2.dnn.NMSBoxes(final_boxes, scores_filt.tolist(), 0.4, 0.4)

                if len(indices) > 0:
                    for idx in indices.flatten():
                        conf = scores_filt[idx]
                        if conf > CONF_THRESHOLD:
                            found_high_conf_in_frame = True
                            bx, by, bw_box, bh_box = final_boxes[idx]
                            
                            self.detection_counter += 1
                            center_x_box = bx + (bw_box / 2)
                            angle = ((center_x_box - (w / 2)) / w) * ASTRA_PRO_HFOV

                            if self.detection_counter >= FRAME_THRESHOLD:
                                if not self.interrupt_sent_sim:
                                    rospy.loginfo(f"🎯 TARGET LOCKED: Angle {angle:.2f}")
                                    self.interrupt_sent_sim = True

                                if current_time - self.last_print_time > PRINT_INTERVAL:
                                    log_entry = {
                                        "ros_time": f"{msg.header.stamp.secs}.{msg.header.stamp.nsecs:09d}",
                                        "conf": round(float(conf), 3),
                                        "angle": round(angle, 2)
                                    }
                                    print(f"📝 DATA LOG: {json.dumps(log_entry)}")
                                    self.last_print_time = current_time

                            # --- Drawing ---
                            color = (0, 255, 0) if self.interrupt_sent_sim else (0, 255, 255)
                            cv2.rectangle(display_img, (bx, by), (bx + bw_box, by + bh_box), color, 2)
                            cv2.putText(display_img, f"can {conf:.2f}", (bx, max(0, by - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if not found_high_conf_in_frame:
                self.detection_counter = 0

            cv2.imshow("Astra Pro Vision", display_img)
            cv2.waitKey(1)

        except Exception as e:
            rospy.logerr(f"Inference Error: {e}")
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