#!/usr/bin/env python3
"""
ROS node: TensorRT YOLOv8 object detection — displays bounding boxes in a cv2 window.
Subscribes to IMAGE_TOPIC, runs best.engine inference, shows annotated frames live.
Press 'q' in the window to quit.
"""

import rospy
import numpy as np
import cv2
import time

from sensor_msgs.msg import Image

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401  initialises CUDA context

# ── Config ────────────────────────────────────────────────────────────────────
ENGINE_PATH     = "/home/jetson/fyp/ws/src/custom_bringup/scripts/models/best.engine"
IMAGE_TOPIC     = "/camera/rgb/image_raw"
INPUT_W, INPUT_H = 640, 640
ASTRA_PRO_HFOV  = 58.4          # degrees
PRINT_INTERVAL  = 3.0           # seconds between console log summaries
CONF_THRESHOLD  = 0.85
FRAME_THRESHOLD = 5             # consecutive frames before confirming detection
# ─────────────────────────────────────────────────────────────────────────────

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# ── TensorRT helpers ──────────────────────────────────────────────────────────

def load_engine(engine_path: str) -> trt.ICudaEngine:
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")
    rospy.loginfo(f"[yolo_trt] Engine loaded: {engine_path}")
    return engine


class TRTInference:
    """Allocates CUDA buffers and runs synchronous TensorRT inference."""

    def __init__(self, engine: trt.ICudaEngine):
        self.engine  = engine
        self.context = engine.create_execution_context()

        self.inputs, self.outputs, self.bindings, self.stream = [], [], [], cuda.Stream()

        for binding in engine:
            size  = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem   = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                self.inputs.append({"host": host_mem, "device": device_mem})
            else:
                self.outputs.append({"host": host_mem, "device": device_mem})

    def infer(self, input_array: np.ndarray) -> list:
        np.copyto(self.inputs[0]["host"], input_array.ravel())
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp["device"], inp["host"], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
        self.stream.synchronize()
        return [out["host"] for out in self.outputs]


# ── Pre / post processing ─────────────────────────────────────────────────────

def preprocess(img_bgr: np.ndarray, input_w: int, input_h: int) -> np.ndarray:
    """Letterbox resize → RGB → NCHW float32 normalised to [0, 1]."""
    img = cv2.resize(img_bgr, (input_w, input_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))          # HWC → CHW
    img = np.expand_dims(img, axis=0)            # → NCHW
    return np.ascontiguousarray(img)


def postprocess(output: np.ndarray,
                orig_w: int, orig_h: int,
                input_w: int, input_h: int,
                conf_thresh: float) -> list:
    """
    Parse YOLOv8 output tensor.

    YOLOv8 exports with shape [1, 4+num_classes, num_anchors].
    Returns list of dicts: {x1, y1, x2, y2, conf, class_id}
    scaled to the original image dimensions.
    """
    # output shape: (1, num_det, 4+num_classes)  or transposed – handle both
    preds = np.squeeze(output)          # (num_det, 4+nc) or (4+nc, num_det)

    if preds.ndim == 1:
        preds = preds[np.newaxis, :]    # single detection edge-case

    # YOLOv8 TensorRT export: shape = [4+nc, num_anchors] → transpose
    if preds.shape[0] < preds.shape[1]:
        preds = preds.T                 # → (num_anchors, 4+nc)

    detections = []
    scale_x = orig_w / input_w
    scale_y = orig_h / input_h

    for row in preds:
        cx, cy, w, h = row[:4]
        class_scores  = row[4:]
        class_id      = int(np.argmax(class_scores))
        conf          = float(class_scores[class_id])

        if conf < conf_thresh:
            continue

        # Convert cx,cy,w,h (relative to input size) to pixel coords in orig image
        x1 = int((cx - w / 2) * scale_x)
        y1 = int((cy - h / 2) * scale_y)
        x2 = int((cx + w / 2) * scale_x)
        y2 = int((cy + h / 2) * scale_y)

        detections.append({
            "x1": max(0, x1), "y1": max(0, y1),
            "x2": min(orig_w, x2), "y2": min(orig_h, y2),
            "conf": conf, "class_id": class_id
        })

    return detections


def nms(detections: list, iou_thresh: float = 0.45) -> list:
    """Simple NMS over detections from postprocess()."""
    if not detections:
        return []
    boxes  = np.array([[d["x1"], d["y1"], d["x2"], d["y2"]] for d in detections], dtype=float)
    scores = np.array([d["conf"] for d in detections])
    indices = cv2.dnn.NMSBoxes(
        bboxes  = boxes.tolist(),
        scores  = scores.tolist(),
        score_threshold = 0.0,
        nms_threshold   = iou_thresh
    )
    if len(indices) == 0:
        return []
    return [detections[i] for i in indices.flatten()]


def draw_boxes(img: np.ndarray, detections: list, class_names: list = None) -> np.ndarray:
    """Draw bounding boxes + confidence labels on a copy of img."""
    vis = img.copy()
    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        conf     = det["conf"]
        class_id = det["class_id"]
        label    = f"{class_names[class_id] if class_names else class_id}: {conf:.2f}"

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(vis, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return vis


def bearing_deg(cx_px: float, img_w: int, hfov: float) -> float:
    """Horizontal bearing of a detection centre relative to image centre (degrees)."""
    return ((cx_px - img_w / 2.0) / img_w) * hfov


# ── ROS node ──────────────────────────────────────────────────────────────────

class YoloTRTNode:
    def __init__(self):
        rospy.init_node("yolo_trt_node", anonymous=False)

        engine         = load_engine(ENGINE_PATH)
        self.trt_infer = TRTInference(engine)

        self.frame_counts = {}
        self.last_print_t = time.time()

        cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)

        rospy.Subscriber(IMAGE_TOPIC, Image, self.image_callback, queue_size=1, buff_size=2**24)
        rospy.loginfo("[yolo_trt] Node ready. Subscribing to %s", IMAGE_TOPIC)

    def ros_image_to_bgr(self, msg: Image) -> np.ndarray:
        """Convert sensor_msgs/Image to a BGR numpy array without cv_bridge."""
        dtype  = np.uint8
        arr    = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, -1)
        enc    = msg.encoding.lower()

        if enc in ("bgr8", "bgr"):
            return arr.copy()
        elif enc in ("rgb8", "rgb"):
            return arr[:, :, ::-1].copy()           # RGB → BGR
        elif enc in ("mono8", "8uc1"):
            return cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2BGR)
        elif enc in ("bgra8", "bgra"):
            return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        elif enc in ("rgba8", "rgba"):
            return cv2.cvtColor(arr[:, :, ::-1], cv2.COLOR_RGBA2BGR)
        else:
            rospy.logwarn_once(f"[yolo_trt] Unknown encoding '{msg.encoding}', assuming BGR8")
            return arr[:, :, :3].copy()

    def image_callback(self, msg: Image):
        cv_img = self.ros_image_to_bgr(msg)
        orig_h, orig_w = cv_img.shape[:2]

        # ── Inference ──
        inp  = preprocess(cv_img, INPUT_W, INPUT_H)
        raw  = self.trt_infer.infer(inp)
        dets = postprocess(raw[0], orig_w, orig_h, INPUT_W, INPUT_H, CONF_THRESHOLD)
        dets = nms(dets)

        # ── Frame-count gating ──
        seen_ids = {d["class_id"] for d in dets}
        for cid in list(self.frame_counts):
            if cid not in seen_ids:
                self.frame_counts[cid] = 0
        for cid in seen_ids:
            self.frame_counts[cid] = self.frame_counts.get(cid, 0) + 1

        confirmed = [d for d in dets if self.frame_counts.get(d["class_id"], 0) >= FRAME_THRESHOLD]

        # ── Periodic console summary ──
        now = time.time()
        if now - self.last_print_t >= PRINT_INTERVAL:
            if confirmed:
                for d in confirmed:
                    cx = (d["x1"] + d["x2"]) / 2.0
                    bearing = bearing_deg(cx, orig_w, ASTRA_PRO_HFOV)
                    rospy.loginfo(
                        "[yolo_trt] class=%d  conf=%.3f  bbox=(%d,%d,%d,%d)  bearing=%.1f°",
                        d["class_id"], d["conf"], d["x1"], d["y1"], d["x2"], d["y2"], bearing
                    )
            else:
                rospy.loginfo("[yolo_trt] No confirmed detections.")
            self.last_print_t = now

        # ── Display in cv2 window ──
        annotated = draw_boxes(cv_img, confirmed)
        cv2.imshow("YOLO Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            rospy.signal_shutdown("User pressed q")
            cv2.destroyAllWindows()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        node = YoloTRTNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()