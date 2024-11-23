from datetime import datetime
import cv2
import torch
import warnings
import requests
import platform
import pathlib

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

def line_Notify(text="ตรวจพบแมลง" ) :
    data = {"message": text}  
    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        print("Notification sent successfully!")
    else:
        print(f"Error 1  sending notification: {response.status_code}")

try:
    model_path = pathlib.Path(r"best_v5.pt").resolve()
    model_path = str(model_path)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
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
        results = model(frame)

        detected_objects = results.pandas().xyxy[0]

        insects = detected_objects[(detected_objects['name'] == 'insect') & (detected_objects['confidence'] > 0.6)]

        if not insects.empty:
            line_Notify()
            print("Insect detected!")
            current_time = datetime.now().strftime("%H:%M:%S") 
            print(f"Insect detected at {current_time}!")

        for _, row in insects.iterrows():
            x1, y1, x2, y2, confidence, class_name = row['xmin'], row['ymin'], row['xmax'], row['ymax'], row['confidence'], row['name']
            label = f"{class_name} {confidence:.2f}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("YOLOv5 Detection", frame)

    except Exception as e:
        print(f"Error during inference: {e}")

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()