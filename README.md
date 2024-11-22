# YOLOv5 Insect Detection and Notification System

This project uses the YOLOv5 object detection model to detect insects through a webcam feed. When an insect is detected with high confidence, the system sends a notification via Line Notify and displays the detected objects in a video stream with bounding boxes.

## Features
- **Real-time Insect Detection**: Detects insects using a YOLOv5 custom model.
- **Line Notifications**: Sends a notification via Line Notify when an insect is detected.
- **Confidence Filtering**: Only detects insects with a confidence score greater than 0.6.
- **Live Display**: Shows the webcam feed with bounding boxes and labels for detected insects.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your_username/insect-detection.git
   cd insect-detection

2. **Install dependencies: Install the required Python libraries**:
   ```bash
   pip install opencv-python torch requests

3. **Set up Line Notify**:
   - Obtain a Line Notify token from Line Notify.
   - Replace LINE_TOKEN in the script with your token.
  
4. **Run python file**
