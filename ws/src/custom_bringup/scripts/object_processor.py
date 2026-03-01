import roslibpy
import time
import json
import math

# --- CONFIGURATION ---
ROBOT_IP = '192.168.18.86'
OBJECTS_TOPIC = '/objects'  # Topic from find_object_2d
INTERRUPT_TOPIC = '/pc_node_reply'
ASTRA_PRO_HFOV = 58.4 
IMAGE_WIDTH = 640  # Match your Astra Pro resolution
FRAME_THRESHOLD = 5 

class FindObjectProcessor:
    def __init__(self):
        self.client = roslibpy.Ros(host=ROBOT_IP, port=9090)
        self.interrupt_sent = False
        self.detection_counter = 0
        
        # Subscriber for find_object_2d data
        # Data format: [id, width, height, homography_matrix...]
        self.sub = roslibpy.Topic(self.client, OBJECTS_TOPIC, 'std_msgs/Float32MultiArray')
        self.pub = roslibpy.Topic(self.client, INTERRUPT_TOPIC, 'std_msgs/String')

    def objects_callback(self, message):
        data = message.get('data', [])
        
        if len(data) > 0:
            # find_object_2d sends 12 values per object
            obj_id = data[0]
            obj_w = data[1]
            obj_h = data[2]
            
            # The center of the object in the image is derived from the homography 
            # but for a watchdog/interrupt, the simple width/height center is often used 
            # if find_object_2d is configured to publish the object's position.
            # In simple mode, we calculate center based on the detected frame.
            
            self.detection_counter += 1
            
            if self.detection_counter >= FRAME_THRESHOLD and not self.interrupt_sent:
                # Assuming the detection is centered in the search frame
                # If find_object_2d provides a full matrix, we'd extract x from there.
                # Here we use a placeholder 'angle' logic consistent with your YOLO node
                angle = 0.0 # find_object_2d requires specific matrix math for exact angle
                
                self.send_interrupt(angle, obj_w, obj_h)
        else:
            self.detection_counter = 0

    def send_interrupt(self, angle, w, h):
        payload = {
            "header": "interrupt",
            "timestamp": str(time.time()),
            "angle": round(angle - 3, 2),
            "w": round(w, 1),
            "h": round(h, 1)
        }
        
        if self.client.is_connected:
            self.pub.publish(roslibpy.Message({'data': json.dumps(payload)}))
            print(f"📡 INTERRUPT SENT -> {payload}")
            self.interrupt_sent = True

    def start(self):
        self.sub.subscribe(self.objects_callback)
        self.client.run_forever()

if __name__ == "__main__":
    processor = FindObjectProcessor()
    try:
        print(f"🟢 Monitoring find_object_2d on {ROBOT_IP}...")
        processor.start()
    except KeyboardInterrupt:
        processor.client.terminate()