#!/usr/bin/python3
import rospy
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from sensor_msgs.msg import Image

# Configuration
ENGINE_PATH = "/home/jetson/fyp/ws/src/custom_bringup/scripts/models/best.engine"
IMAGE_TOPIC = "/camera/rgb/image_raw" # Astra Pro RGB Topic
INPUT_W, INPUT_H = 320, 320 # Must match your ONNX export size

class YOLOv8TRTNode:
    def __init__(self):
        rospy.init_node('yolo_trt_detector', anonymous=True)
        
        # 1. Load TensorRT Engine
        self.logger = trt.Logger(trt.Logger.INFO)
        with open(ENGINE_PATH, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        
        # 2. Allocate GPU Buffers
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()

        # 3. ROS Subscriber & Publisher
        self.sub = rospy.Subscriber(IMAGE_TOPIC, Image, self.image_callback, queue_size=1)
        rospy.loginfo("YOLO TRT Node Started. Engine Loaded. Waiting for images...")

    def allocate_buffers(self):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        for i in range(self.engine.num_bindings):
            size = trt.volume(self.engine.get_binding_shape(i))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self.engine.binding_is_input(i):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
        return inputs, outputs, bindings, stream

    def image_callback(self, msg):
        # Manual conversion to avoid cv_bridge Python 3 conflicts
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        if msg.encoding == 'rgb8':
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Preprocessing: Resize and Normalize
        blob = cv2.resize(frame, (INPUT_W, INPUT_H))
        blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)
        blob = blob.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1)).ravel()

        # Copy to GPU and Execute
        np.copyto(self.inputs[0]['host'], blob)
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()

        # Output logic for 100plus can
        # output = self.outputs[0]['host'] 
        # (Insert NMS/Bounding box logic here for final display)

        # Show window for VNC
        cv2.imshow("Transbot YOLO View", frame)
        cv2.waitKey(1)

if __name__ == '__main__':
    try:
        node = YOLOv8TRTNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    cv2.destroyAllWindows()