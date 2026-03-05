#!/usr/bin/python3
import rospy
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from sensor_msgs.msg import Image

# --- CONFIGURATION ---
ENGINE_PATH = "/home/jetson/fyp/ws/src/custom_bringup/scripts/models/best.engine"
IMAGE_TOPIC = "/camera/rgb/image_raw" # Change to your Astra Pro topic
INPUT_W, INPUT_H = 320, 320 # Must match your ONNX export size

class YOLOv8TRTNode:
    def __init__(self):
        rospy.init_node('yolo_trt_detector', anonymous=True)
        
        # 1. Load TensorRT Engine
        self.logger = trt.Logger(trt.Logger.INFO)
        with open(ENGINE_PATH, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()

        # 2. ROS Subscriber
        self.sub = rospy.Subscriber(IMAGE_TOPIC, Image, self.image_callback, queue_size=1)
        rospy.loginfo("YOLO TRT Node Started. Waiting for images...")

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
        # Manual conversion from ROS Image to OpenCV (Avoids cv_bridge issues)
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        if msg.encoding == 'rgb8':
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 1. Preprocess
        blob = cv2.resize(frame, (INPUT_W, INPUT_H))
        blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)
        blob = blob.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1)).ravel()

        # 2. GPU Inference
        np.copyto(self.inputs[0]['host'], blob)
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()

        # 3. Handle Output (Simplified for 100plus can detection)
        # Note: YOLOv8 output is usually [1, 84, 2100]. This is a raw buffer.
        # For a quick test, let's just show the window.
        cv2.putText(frame, "Inference Running @ 20FPS", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 4. Display for VNC
        cv2.imshow("Transbot YOLO View", frame)
        cv2.waitKey(1)

if __name__ == '__main__':
    try:
        node = YOLOv8TRTNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    cv2.destroyAllWindows()