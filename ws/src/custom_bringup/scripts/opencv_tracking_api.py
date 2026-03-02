#!/usr/bin/python3
import rospy
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
from sensor_msgs.msg import Image

# Configuration
ENGINE_PATH = "/home/jetson/fyp/ws/src/custom_bringup/scripts/models/best.engine"
IMAGE_TOPIC = "/camera/rgb/image_raw"
INPUT_W, INPUT_H = 640, 640
CONF_THRESHOLD = 0.4
NMS_THRESHOLD = 0.4

class YOLOv8TRTNode:
    def __init__(self):
        rospy.init_node('yolo_trt_detector', anonymous=True)
        
        # 1. Initialize CUDA and Context explicitly
        cuda.init()
        self.device = cuda.Device(0)
        self.ctx = self.device.make_context() # Manual context
        
        self.logger = trt.Logger(trt.Logger.INFO)
        
        # 2. Load TensorRT Engine
        with open(ENGINE_PATH, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        
        # 3. Allocate GPU Buffers
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        
        # 4. ROS Subscriber with large buffer to prevent delay
        self.sub = rospy.Subscriber(IMAGE_TOPIC, Image, self.image_callback, 
                                   queue_size=1, buff_size=2**24)
        
        rospy.loginfo("YOLO TRT Node Fixed. CUDA Context initialized.")

    def allocate_buffers(self):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        for i in range(self.engine.num_bindings):
            shape = self.engine.get_binding_shape(i)
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            # Page-locked memory for faster transfer
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self.engine.binding_is_input(i):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
        return inputs, outputs, bindings, stream

    def postprocess(self, data, orig_w, orig_h):
        # Reshape: 84 rows (4 boxes + 80 classes) x 8400 columns
        data = data.reshape(84, -1).T
        
        # Get max score and class ID for each row
        scores = np.max(data[:, 4:], axis=1)
        mask = scores > CONF_THRESHOLD
        filtered_data = data[mask]
        filtered_scores = scores[mask]
        
        if len(filtered_data) == 0:
            return [], [], []

        class_ids = np.argmax(filtered_data[:, 4:], axis=1)
        boxes = filtered_data[:, :4]
        
        results_boxes = []
        x_factor = orig_w / INPUT_W
        y_factor = orig_h / INPUT_H

        for i in range(len(boxes)):
            cx, cy, w, h = boxes[i]
            left = int((cx - 0.5 * w) * x_factor)
            top = int((cy - 0.5 * h) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            results_boxes.append([left, top, width, height])

        indices = cv2.dnn.NMSBoxes(results_boxes, filtered_scores.tolist(), CONF_THRESHOLD, NMS_THRESHOLD)
        
        final_boxes, final_scores, final_ids = [], [], []
        if len(indices) > 0:
            for i in indices.flatten():
                final_boxes.append(results_boxes[i])
                final_scores.append(filtered_scores[i])
                final_ids.append(class_ids[i])
        
        return final_boxes, final_scores, final_ids

    def image_callback(self, msg):
        # IMPORTANT: Push the CUDA context to the current thread (ROS callback thread)
        self.ctx.push()
        
        try:
            # 1. Conversion
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            if msg.encoding == 'rgb8':
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # 2. Preprocess
            blob = cv2.resize(frame, (INPUT_W, INPUT_H))
            blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)
            blob = blob.astype(np.float32) / 255.0
            blob = np.transpose(blob, (2, 0, 1)).ravel()

            # 3. Inference
            np.copyto(self.inputs[0]['host'], blob)
            cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
            self.stream.synchronize()

            # 4. Post-process
            boxes, scores, ids = self.postprocess(self.outputs[0]['host'], msg.width, msg.height)

            # 5. Visualization
            for box, score, cl_id in zip(boxes, scores, ids):
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"100Plus:{score:.2f}", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("Transbot YOLO View", frame)
            cv2.waitKey(1)
            
        finally:
            # Pop the context to avoid leaks
            self.ctx.pop()

    def __del__(self):
        # Cleanup
        self.ctx.pop()
        self.ctx.detach()

if __name__ == '__main__':
    try:
        node = YOLOv8TRTNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    cv2.destroyAllWindows()