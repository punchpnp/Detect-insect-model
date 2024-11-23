from datetime import datetime
import cv2
import warnings
import requests
import platform
import pathlib
from ultralytics import YOLO

# Fix the path issue
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

warnings.filterwarnings("ignore", category=FutureWarning)

# Line Notify
LINE_TOKEN = ""  # Replace with your Line Notify token
url = "https://notify-api.line.me/api/notify"
headers = {"Authorization": f"Bearer {LINE_TOKEN}"}

def line_Notify(text="ตรวจพบแมลง"):
    data = {"message": text}  
    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        print("Notification sent successfully!")
    else:
        print(f"Error sending notification: {response.status_code}")

try:
    model_path = pathlib.Path(r"best_v11.pt").resolve()
    model_path = str(model_path)
    model = YOLO(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

cap = cv2.VideoCapture(0)  # Change 0 or 1 to your correct camera index

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)

    try:
        results = model(frame, conf=0.6)  # Confidence threshold set to 0.6

        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box.cls == 0:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    
                    # Send Line notification
                    line_Notify()
                    current_time = datetime.now().strftime("%H:%M:%S")
                    print(f"Insect detected at {current_time}!")

                    cv2.rectangle(frame, 
                                (int(x1), int(y1)), 
                                (int(x2), int(y2)), 
                                (0, 255, 0), 2)
                    
                    label = f"insect {confidence:.2f}"
                    cv2.putText(frame, 
                              label, 
                              (int(x1), int(y1) - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, 
                              (0, 255, 0), 
                              2)

        cv2.imshow("YOLOv11 Detection", frame)

    except Exception as e:
        print(f"Error during inference: {e}")

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()