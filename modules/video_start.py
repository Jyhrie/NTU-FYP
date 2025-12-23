import cv2
import numpy as np
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.image import message_to_cvimage

# --- CONFIGURATION ---
BAG_FILE = 'japan_trip_subset.bag'  # Your bag file name
OUTPUT_VIDEO = 'robot_vision.mp4'   # Desired output name
TOPIC = '/camera/rgb/image_raw'     # The topic we identified earlier
FPS = 30                            # Match your camera's recording speed
# ---------------------

def convert_bag_to_video():
    with AnyReader([Path(BAG_FILE)]) as reader:
        video_writer = None
        
        # Filter for just the RGB topic
        connections = [x for x in reader.connections if x.topic == TOPIC]
        
        print(f"Starting extraction from {TOPIC}...")
        
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            # Deserialize and convert to OpenCV format
            msg = reader.deserialize(rawdata, connection.msgtype)
            cv_img = message_to_cvimage(msg, 'bgr8')
            
            # Initialize VideoWriter on the first frame
            if video_writer is None:
                height, width = cv_img.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (width, height))
            
            video_writer.write(cv_img)
            
        if video_writer:
            video_writer.release()
            print(f"Successfully saved to {OUTPUT_VIDEO}")
        else:
            print("No images found on that topic!")

if __name__ == "__main__":
    convert_bag_to_video()