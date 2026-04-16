#!/usr/bin/env python3
"""
ROS Bridge Data Collection + YOLOv8 Training Pipeline
- Connects to ROS via roslibpy (rosbridge)
- Collects positive samples via SAM1 + OpenCV tracking
- Collects negative samples (object removed)
- Trains YOLOv8n and exports to ONNX
"""

import cv2
import numpy as np
import base64
import threading
import time
import os
import shutil
import sys
import random
from pathlib import Path
from datetime import datetime

import roslibpy  # pip install roslibpy
import paramiko   # pip install paramiko

# ── Config ────────────────────────────────────────────────────────────────────
ROS_IP       = "192.168.18.86"
ROS_PORT     = 9090
IMAGE_TOPIC  = "/camera/rgb/image_compressed/compressed"

# ── Robot SCP config ──────────────────────────────────────────────────────────
ROBOT_USER       = "jetson"
ROBOT_HOST       = "192.168.18.86"
ROBOT_PASS       = "yahboom"
ROBOT_MODELS_DIR = "/home/jetson/fyp/ws/src/custom_bringup/scripts/models"

SAVE_DIR     = Path("dataset_collection")
POS_DIR      = SAVE_DIR / "positive"
NEG_DIR      = SAVE_DIR / "negative"
YOLO_DIR     = SAVE_DIR / "yolo_dataset"

FRAME_SKIP          = 3      # save every N frames while tracking
NEG_DURATION_SEC    = 3.0    # how long to collect negatives
NEG_TARGET_FRAMES   = 17     # ~15-20 frames of negatives
CLASS_NAME          = "object"

# ── Globals ───────────────────────────────────────────────────────────────────
latest_frame: np.ndarray | None = None
frame_lock = threading.Lock()
ros_client: roslibpy.Ros | None = None


# ══════════════════════════════════════════════════════════════════════════════
#  ROS Bridge connection  (roslibpy)
# ══════════════════════════════════════════════════════════════════════════════

def _image_callback(message):
    """Called by roslibpy on every CompressedImage message."""
    global latest_frame
    try:
        img_data  = message["data"]
        jpg_bytes = base64.b64decode(img_data)
        arr       = np.frombuffer(jpg_bytes, dtype=np.uint8)
        frame     = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is not None:
            with frame_lock:
                latest_frame = frame.copy()
    except Exception:
        pass  # silently drop bad frames


def start_ros_connection():
    """
    Create a roslibpy.Ros client, start it in a daemon thread,
    and subscribe to the compressed-image topic.
    """
    global ros_client

    print(f"[ROS] Connecting to ws://{ROS_IP}:{ROS_PORT} ...")
    ros_client = roslibpy.Ros(host=ROS_IP, port=ROS_PORT)

    # Subscribe once the connection is ready
    def on_ready(event=None):
        print("[ROS] Connected to rosbridge.")
        topic = roslibpy.Topic(
            ros_client,
            IMAGE_TOPIC,
            "sensor_msgs/CompressedImage",
        )
        topic.subscribe(_image_callback)
        print(f"[ROS] Subscribed to {IMAGE_TOPIC}")

    ros_client.on_ready(on_ready, run_in_thread=True)

    # Run the roslibpy event loop in a background daemon thread
    t = threading.Thread(target=ros_client.run_forever, daemon=True)
    t.start()

    # Wait up to 10 s for the first frame
    print("[ROS] Waiting for first image frame...")
    for _ in range(100):
        time.sleep(0.1)
        with frame_lock:
            if latest_frame is not None:
                print("[ROS] Stream active.")
                return

    print("[WARNING] No frame received yet — continuing anyway.")


def get_frame() -> np.ndarray | None:
    with frame_lock:
        return latest_frame.copy() if latest_frame is not None else None


# ══════════════════════════════════════════════════════════════════════════════
#  Bounding box drawing helper
# ══════════════════════════════════════════════════════════════════════════════

class BBoxDrawer:
    def __init__(self, image: np.ndarray):
        self.image    = image.copy()
        self.drawing  = False
        self.start    = (-1, -1)
        self.end      = (-1, -1)
        self.bbox     = None          # (x, y, w, h)

    def mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start   = (x, y)
            self.end     = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end     = (x, y)
            x0 = min(self.start[0], self.end[0])
            y0 = min(self.start[1], self.end[1])
            x1 = max(self.start[0], self.end[0])
            y1 = max(self.start[1], self.end[1])
            self.bbox = (x0, y0, x1 - x0, y1 - y0)

    def run(self) -> tuple | None:
        win = "Draw Bounding Box — press ENTER to confirm"
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, self.mouse_cb)
        while True:
            disp = self.image.copy()
            if self.bbox:
                x, y, w, h = self.bbox
                cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(disp, "ENTER to confirm | drag to redraw",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            elif self.drawing:
                cv2.rectangle(disp, self.start, self.end, (255, 100, 0), 1)
            else:
                cv2.putText(disp, "Click and drag to draw a box",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.imshow(win, disp)
            key = cv2.waitKey(20) & 0xFF
            if key == 13 and self.bbox:   # ENTER
                break
            if key == 27:                 # ESC — cancel
                self.bbox = None
                break
        cv2.destroyWindow(win)
        return self.bbox


# ══════════════════════════════════════════════════════════════════════════════
#  SAM1 mask → bbox refinement  (optional; falls back gracefully)
# ══════════════════════════════════════════════════════════════════════════════

def refine_bbox_with_sam(frame: np.ndarray, bbox: tuple) -> tuple:
    """
    Try to refine the user bbox using SAM1 (segment-anything).
    Falls back to the original bbox if SAM is not installed.
    """
    try:
        import torch
        from segment_anything import sam_model_registry, SamPredictor

        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"[SAM] Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("[SAM] CUDA not available — falling back to CPU.")

        ckpt_candidates = [
            "sam_vit_b_01ec64.pth",
            os.path.expanduser("~/sam_vit_b_01ec64.pth"),
            "/opt/sam/sam_vit_b_01ec64.pth",
        ]
        ckpt = next((c for c in ckpt_candidates if os.path.exists(c)), None)
        if ckpt is None:
            print("[SAM] No checkpoint found — using raw bbox.")
            return bbox

        print("[SAM] Running SAM1 refinement...")
        sam = sam_model_registry["vit_b"](checkpoint=ckpt)
        sam.to(device=device)
        sam.eval()
        pred = SamPredictor(sam)
        pred.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        x, y, w, h = bbox
        input_box   = np.array([x, y, x + w, y + h])
        masks, _, _ = pred.predict(box=input_box[None, :], multimask_output=False)

        mask   = masks[0].astype(np.uint8)
        coords = cv2.findNonZero(mask)
        if coords is not None:
            rx, ry, rw, rh = cv2.boundingRect(coords)
            print(f"[SAM] Refined bbox: ({rx},{ry},{rw},{rh})")
            return (rx, ry, rw, rh)
    except ImportError:
        print("[SAM] segment_anything not installed — using raw bbox.")
    except Exception as e:
        print(f"[SAM] Error: {e} — using raw bbox.")
    return bbox


# ══════════════════════════════════════════════════════════════════════════════
#  Positive sample collection  (tracking loop)
# ══════════════════════════════════════════════════════════════════════════════

def collect_positives(bbox: tuple) -> list[tuple]:
    """
    Track object with CSRT, save every FRAME_SKIP frames.
    Returns list of (frame_path, (cx, cy, nw, nh)) tuples.
    """
    POS_DIR.mkdir(parents=True, exist_ok=True)

    tracker    = cv2.TrackerCSRT_create()
    init_frame = get_frame()
    if init_frame is None:
        print("[ERROR] No frame available for tracking.")
        return []

    tracker.init(init_frame, bbox)
    samples   = []
    frame_idx = 0
    saved     = 0
    win_name  = "Tracking — press ENTER to stop"

    print("[TRACK] Tracking started. Press ENTER to stop.")
    cv2.namedWindow(win_name)

    while True:
        frame = get_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        ok, new_bbox = tracker.update(frame)
        disp = frame.copy()

        if ok:
            x, y, w, h = [int(v) for v in new_bbox]
            cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if frame_idx % FRAME_SKIP == 0:
                fname = POS_DIR / f"pos_{saved:05d}.jpg"
                cv2.imwrite(str(fname), frame)
                ih, iw = frame.shape[:2]
                cx = (x + w / 2) / iw
                cy = (y + h / 2) / ih
                nw = w / iw
                nh = h / ih
                samples.append((str(fname), (cx, cy, nw, nh)))
                saved += 1
                cv2.putText(disp, f"Saved: {saved}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(disp, "Tracking lost!", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(disp, "ENTER to stop", (10, disp.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.imshow(win_name, disp)

        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # ENTER
            break
        frame_idx += 1

    cv2.destroyWindow(win_name)
    print(f"[TRACK] Collected {saved} positive frames.")
    return samples


# ══════════════════════════════════════════════════════════════════════════════
#  Negative sample collection
# ══════════════════════════════════════════════════════════════════════════════

def collect_negatives() -> list[str]:
    NEG_DIR.mkdir(parents=True, exist_ok=True)
    win_name = "Remove the object — collecting negatives..."
    cv2.namedWindow(win_name)

    print("[NEG] Please REMOVE the object from view.")
    print(f"[NEG] Collecting for {NEG_DURATION_SEC}s...")

    for countdown in range(3, 0, -1):
        frame = get_frame()
        if frame is None:
            time.sleep(1)
            continue
        disp = frame.copy()
        cv2.putText(disp, f"Remove object! Starting in {countdown}...",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.imshow(win_name, disp)
        cv2.waitKey(1000)

    saved     = 0
    t_start   = time.time()
    neg_paths = []

    while saved < NEG_TARGET_FRAMES or (time.time() - t_start) < NEG_DURATION_SEC:
        frame = get_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        fname = NEG_DIR / f"neg_{saved:05d}.jpg"
        cv2.imwrite(str(fname), frame)
        neg_paths.append(str(fname))
        saved += 1

        disp = frame.copy()
        cv2.putText(disp, f"Collecting negatives... {saved}/{NEG_TARGET_FRAMES}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
        cv2.imshow(win_name, disp)
        cv2.waitKey(1)
        time.sleep(NEG_DURATION_SEC / NEG_TARGET_FRAMES)

    cv2.destroyWindow(win_name)
    print(f"[NEG] Collected {saved} negative frames.")
    return neg_paths


# ══════════════════════════════════════════════════════════════════════════════
#  Redo / Complete prompt
# ══════════════════════════════════════════════════════════════════════════════

def prompt_redo_or_complete(pos_samples, neg_paths) -> bool:
    """Returns True → complete, False → redo."""
    frame = get_frame()
    if frame is None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

    win = "Review — press R to Redo | press C to Complete"
    cv2.namedWindow(win)

    while True:
        disp    = frame.copy()
        overlay = disp.copy()
        cv2.rectangle(overlay, (0, 0), (disp.shape[1], 120), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.6, disp, 0.4, 0, disp)

        cv2.putText(disp, f"Positive samples: {len(pos_samples)}",
                    (15, 35),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2)
        cv2.putText(disp, f"Negative samples: {len(neg_paths)}",
                    (15, 70),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 255), 2)
        cv2.putText(disp, "[ R ] Redo all   |   [ C ] Complete & Train",
                    (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(win, disp)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord('r'), ord('R')):
            cv2.destroyWindow(win)
            return False
        if key in (ord('c'), ord('C')):
            cv2.destroyWindow(win)
            return True


# ══════════════════════════════════════════════════════════════════════════════
#  YOLO dataset builder
# ══════════════════════════════════════════════════════════════════════════════

def build_yolo_dataset(pos_samples: list[tuple], neg_paths: list[str]) -> Path:
    """
    Builds a YOLOv8-compatible directory structure with 75/25 train/val split.
    pos_samples : [(img_path, (cx, cy, nw, nh)), ...]
    neg_paths   : [img_path, ...]   (background — empty label files)
    """
    for split in ("train", "val"):
        (YOLO_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    all_pos = list(pos_samples)
    all_neg = list(neg_paths)
    random.shuffle(all_pos)
    random.shuffle(all_neg)

    split_pos = int(len(all_pos) * 0.75)
    split_neg = int(len(all_neg) * 0.75)

    def copy_sample(img_path, label_line, split):
        dst_img  = YOLO_DIR / "images" / split / Path(img_path).name
        lbl_path = YOLO_DIR / "labels" / split / (Path(img_path).stem + ".txt")
        shutil.copy2(img_path, dst_img)
        lbl_path.write_text(label_line if label_line is not None else "")

    for i, (img_path, (cx, cy, nw, nh)) in enumerate(all_pos):
        split = "train" if i < split_pos else "val"
        copy_sample(img_path, f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}", split)

    for i, img_path in enumerate(all_neg):
        split = "train" if i < split_neg else "val"
        copy_sample(img_path, None, split)

    yaml_content = f"""\
path: {YOLO_DIR.resolve()}
train: images/train
val:   images/val

nc: 1
names: ['{CLASS_NAME}']
"""
    yaml_path = YOLO_DIR / "data.yaml"
    yaml_path.write_text(yaml_content)

    print(f"[DATASET] YOLO dataset written to {YOLO_DIR}")
    print(f"  Train positives : {min(split_pos, len(all_pos))}")
    print(f"  Val   positives : {len(all_pos) - min(split_pos, len(all_pos))}")
    print(f"  Train negatives : {min(split_neg, len(all_neg))}")
    print(f"  Val   negatives : {len(all_neg) - min(split_neg, len(all_neg))}")
    return yaml_path


# ══════════════════════════════════════════════════════════════════════════════
#  SCP transfer to robot
# ══════════════════════════════════════════════════════════════════════════════

def scp_to_robot(onnx_path: Path) -> str | None:
    """
    Transfers the ONNX file to the robot via SFTP.
    Returns the remote path on success, None on failure.
    """
    print(f"\n[SCP] Transferring {onnx_path.name} to {ROBOT_USER}@{ROBOT_HOST}:{ROBOT_MODELS_DIR} ...")
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(ROBOT_HOST, username=ROBOT_USER, password=ROBOT_PASS, timeout=10)

        # Ensure destination directory exists
        ssh.exec_command(f"mkdir -p {ROBOT_MODELS_DIR}")
        time.sleep(0.5)

        # SCP the file
        with ssh.open_sftp() as sftp:
            remote_path = f"{ROBOT_MODELS_DIR}/{onnx_path.name}"
            sftp.put(str(onnx_path), remote_path)

        ssh.close()
        print(f"[SCP] Transfer complete → {ROBOT_HOST}:{remote_path}")
        return remote_path
    except Exception as e:
        print(f"[SCP] Transfer failed: {e}")
        return None


def publish_convert_signal(remote_onnx_path: str):
    """
    Publishes the remote .onnx path to /convert_to_engine on the robot
    so engine_converter_node kicks off trtexec.
    """
    print(f"[ROS] Publishing conversion signal for: {remote_onnx_path}")
    try:
        topic = roslibpy.Topic(
            ros_client,
            "/convert_to_engine",
            "std_msgs/String",
        )
        topic.publish(roslibpy.Message({"data": remote_onnx_path}))
        time.sleep(0.3)  # give rosbridge time to flush
        topic.unadvertise()
        print("[ROS] Signal published → engine_converter_node will begin conversion.")
    except Exception as e:
        print(f"[ROS] Failed to publish signal: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  YOLOv8 training + ONNX export
# ══════════════════════════════════════════════════════════════════════════════

def train_and_export(yaml_path: Path):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    print("\n[TRAIN] Starting YOLOv8n training...")
    model   = YOLO("yolov8n.pt")
    results = model.train(
        data=str(yaml_path),
        epochs=35,
        imgsz=640,
        batch=16,
        name="ros_object_detector",
        project=str(SAVE_DIR / "runs"),
        exist_ok=True,
        verbose=True,
    )

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if not best_pt.exists():
        candidates = list((SAVE_DIR / "runs").rglob("best.pt"))
        if candidates:
            best_pt = candidates[-1]
        else:
            print("[ERROR] Could not find best.pt")
            return

    print(f"\n[EXPORT] Exporting {best_pt} to ONNX...")
    best_model = YOLO(str(best_pt))
    best_model.export(format="onnx", imgsz=640, simplify=True, opset=12)

    onnx_path = best_pt.with_suffix(".onnx")
    if onnx_path.exists():
        print(f"[EXPORT] ONNX saved: {onnx_path}")
        remote_path = scp_to_robot(onnx_path)
        if remote_path:
            publish_convert_signal(remote_path)
        print("[CLEANUP] Deleting local dataset and run files...")
        shutil.rmtree(SAVE_DIR, ignore_errors=True)
        print("[CLEANUP] Done.")
    else:
        print("[EXPORT] ONNX file not found at expected path — check ultralytics export output.")


# ══════════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def main():
    start_ros_connection()

    while True:  # outer redo loop

        # ── Step 1: Wait for S ────────────────────────────────────────────────
        print("\n[MAIN] Press S in the preview window to capture a frame and start.")
        win_preview = "Live Preview — press S to start"
        cv2.namedWindow(win_preview)

        snapshot = None
        while True:
            frame = get_frame()
            if frame is not None:
                disp = frame.copy()
                cv2.putText(disp, "Press  S  to start", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.imshow(win_preview, disp)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord('s'), ord('S')):
                snapshot = get_frame()
                break

        cv2.destroyWindow(win_preview)

        if snapshot is None:
            print("[ERROR] No frame at capture time.")
            continue

        # ── Step 2: Draw bounding box ─────────────────────────────────────────
        print("[MAIN] Draw a bounding box around the object, then press ENTER.")
        raw_bbox = BBoxDrawer(snapshot).run()

        if raw_bbox is None or raw_bbox[2] < 5 or raw_bbox[3] < 5:
            print("[MAIN] No valid bbox — restarting.")
            continue

        # ── Step 3: SAM1 refinement ───────────────────────────────────────────
        bbox = refine_bbox_with_sam(snapshot, raw_bbox)

        # ── Step 4: Tracking / positive collection ────────────────────────────
        print("[MAIN] Tracking object. Press ENTER to stop.")
        pos_samples = collect_positives(bbox)

        # ── Step 5: Negative collection ───────────────────────────────────────
        neg_paths = collect_negatives()

        # ── Step 6: Redo or Complete ──────────────────────────────────────────
        if not prompt_redo_or_complete(pos_samples, neg_paths):
            print("[MAIN] Redo selected — deleting saved files and restarting...")
            shutil.rmtree(POS_DIR,  ignore_errors=True)
            shutil.rmtree(NEG_DIR,  ignore_errors=True)
            shutil.rmtree(YOLO_DIR, ignore_errors=True)
            continue

        # ── Step 7: Build YOLO dataset ────────────────────────────────────────
        print("[MAIN] Building YOLO dataset...")
        yaml_path = build_yolo_dataset(pos_samples, neg_paths)

        # ── Step 8: Train YOLOv8n + export ONNX ──────────────────────────────
        train_and_export(yaml_path)
        break  # done

    # Gracefully shut down the roslibpy client
    try:
        if ros_client and ros_client.is_connected:
            ros_client.terminate()
    except Exception:
        pass  # ignore cleanup errors on disconnect

    cv2.destroyAllWindows()
    print("\n[DONE] Pipeline complete.")


if __name__ == "__main__":
    main()