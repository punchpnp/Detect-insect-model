from flask import Flask, Response
import cv2
import torch
import pathlib
import requests
import platform
import warnings
from datetime import datetime

# Fix path issue for Windows
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize Flask app
app = Flask(__name__)

# Initialize YOLOv5 model
model_path = pathlib.Path(r"best_v5.pt").resolve()
model_path = str(model_path)
try:
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Line Notify function (optional for sending alerts)
LINE_TOKEN = "your_line_notify_token"  # Replace with your Line Notify token
url = "https://notify-api.line.me/api/notify"
headers = {"Authorization": f"Bearer {LINE_TOKEN}"}

def line_Notify(text="Insect detected"):
    data = {"message": text}  
    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        print("Notification sent successfully!")
    else:
        print(f"Error sending notification: {response.status_code}")

# Start capturing video from webcam
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Function to generate video frames
def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Object detection with YOLOv5
        results = model(frame)
        detected_objects = results.pandas().xyxy[0]

        # Filter for insect detections (based on class name and confidence)
        insects = detected_objects[(detected_objects['name'] == 'insect') & (detected_objects['confidence'] > 0.6)]

        # Send Line notification if insect detected
        if not insects.empty:
            line_Notify()
            print("Insect detected!")
            current_time = datetime.now().strftime("%H:%M:%S") 
            print(f"Insect detected at {current_time}!")

        # Draw bounding boxes around detected insects
        for _, row in insects.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            label = f"{row['name']} {row['confidence']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert frame to JPEG and stream it to the client
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route for streaming video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Start Flask server
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
