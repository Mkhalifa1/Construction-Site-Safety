# Construction Site Safety - PPE Detection Using YOLOv5

## Project Overview
This project uses computer vision to monitor safety compliance at construction sites. It leverages two YOLOv5 models: one for detecting people and another custom model for identifying PPE (helmets, safety vests). The system processes video footage, detects safety violations, and generates a video highlighting workers missing PPE.

## Key Features
- **Real-time Detection**: Identifies workers without helmets and safety vests.
- **Alerts & Notifications**: Notifies safety officers of detected violations.
- **Video Output**: Generates an output video only if violations are detected.

## Tools and Libraries
- **YOLOv5**: Object detection model for PPE and people.
- **OpenCV**: Used for video and image processing.
- **PyTorch**: Framework used for loading and fine-tuning the models.

## Workflow
1. Load the input video and detect people using the COCO-trained YOLOv5 model.
2. Use the custom YOLOv5 model to check if the workers are wearing PPE.
3. Mark violations (missing helmets or vests) with colored boxes.
4. If violations are found, save the processed video with detected violations.

## Code Description
1. **Model Loading**: Two models are loaded—one for detecting people and one custom-trained for detecting PPE (helmets, vests).
2. **Frame Processing**: The input video is processed frame by frame. The first model detects workers, and the second model checks for safety compliance.
3. **Violation Detection**: If a worker is missing a helmet or vest, bounding boxes are drawn on the frame (blue for no helmet, red for no vest, yellow for both missing).
4. **Video Output**: If violations are detected, the frames are compiled into a video. If no violations are found, no video is saved.


