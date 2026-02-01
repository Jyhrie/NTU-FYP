
import cv2

# Replace with your robot's actual IP address
ROBOT_IP = "192.168.18.86" 
# The topic name (usually /camera/rgb/image_raw for Astra)
TOPIC = "/camera/rgb/image_raw"

# web_video_server URL format
stream_url = f"http://{ROBOT_IP}:8080/stream?topic={TOPIC}"

def main():
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    print("Streaming started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Display the resulting frame
        cv2.imshow('Astra Camera Stream', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()