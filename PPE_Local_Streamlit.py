import streamlit as st
import warnings
warnings.filterwarnings('ignore')
import pathlib
from pathlib import Path
temp = pathlib.PosixPath  
pathlib.PosixPath = pathlib.WindowsPath  

import torch
import cv2
import os
import numpy as np
from datetime import datetime
import tempfile 

# Define global parameters
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
FRAME_SIZE = (1020, 600)
CLASSES = ["Helmet", "No Safety Vest", "No Helmet", "Safety Vest", "Shoes"]
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Streamlit setup
st.title('Construction Site Safety Gear Detection')
st.write('Upload a video, and the model will detect any missing safety gear such as helmets or vests.')

# Upload video file
uploaded_file = st.file_uploader("Upload your video", type=["mp4", "avi", "mov"])

def load_model(weights_file, conf=0.5, iou=0.4):
    """Load YOLOv5 model with error handling and custom configuration."""
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_file, force_reload=True)  
        model.conf = conf  # Confidence threshold
        model.iou = iou  # NMS IoU threshold
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

def process_frame(frame, model):
    """Run YOLOv5 inference on a single frame and return predictions."""
    frame_resized = cv2.resize(frame, FRAME_SIZE)
    results = model(frame_resized)
    pred = results.pred[0].cpu().numpy()  # Extract predictions
    return pred, frame_resized

def draw_boxes(frame, predictions, label, color):
    """Draw bounding boxes and labels on a frame."""
    for obj in predictions:
        x1, y1, x2, y2 = map(int, obj[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), FONT, 0.5, color, 2, cv2.LINE_AA)

def check_safety_gear(workers, helmets, vests, frame):
    """Check workers for missing helmets or vests and return the frames with violations."""
    no_safety_frames = []
    violation_detected = False
    
    for worker in workers:
        x1, y1, x2, y2 = map(int, worker[:4])
        has_helmet = any(x1 <= h[2] and x2 >= h[0] and y1 <= h[3] and y2 >= h[1] for h in helmets)
        has_vest = any(x1 <= v[2] and x2 >= v[0] and y1 <= v[3] and y2 >= v[1] for v in vests)

        if not has_helmet and not has_vest:
            draw_boxes(frame, [worker], "No Helmet and Vest", (0, 255, 255))
            violation_detected = True
        elif not has_helmet:
            draw_boxes(frame, [worker], "No Helmet", (255, 0, 0))
            violation_detected = True
        elif not has_vest:
            draw_boxes(frame, [worker], "No Vest", (0, 0, 255))
            violation_detected = True

        if violation_detected:
            no_safety_frames.append(frame.copy())

    return no_safety_frames, violation_detected

def get_output_filename():
    """Generate output filename with current date and time."""
    current_time = datetime.now().strftime("%Y-%m-%d-%I-%M-%p")
    filename = f"Detection_{current_time}.mp4"
    return filename

# Main Streamlit app logic
if uploaded_file is not None:
    # Save uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_video:
        temp_video.write(uploaded_file.read())
        temp_video_path = temp_video.name

    # Display uploaded video
    st.video(uploaded_file)

    # Load YOLO models
    coco_model = load_model('yolov5s.pt', conf=CONFIDENCE_THRESHOLD, iou=NMS_THRESHOLD)  # COCO model
    safety_model = load_model('C:/Users/Mostafa/Desktop/our_Project/Final/PPE_Local_Streamlit/best.pt', conf=CONFIDENCE_THRESHOLD, iou=NMS_THRESHOLD)  

    # Ensure models are loaded
    if coco_model and safety_model:
        cap = cv2.VideoCapture(temp_video_path)

        output_filename = get_output_filename()
        output_path = os.path.join(tempfile.gettempdir(), output_filename)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), FRAME_SIZE)

        violations_detected = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            predictions, frame_resized = process_frame(frame, coco_model)
            persons = predictions[predictions[:, 5] == 0]

            if len(persons) > 0:
                safety_predictions, _ = process_frame(frame, safety_model)
                helmets = safety_predictions[safety_predictions[:, 5] == CLASSES.index('Helmet')]
                vests = safety_predictions[safety_predictions[:, 5] == CLASSES.index('Safety Vest')]
                no_safety_frames, violation_detected = check_safety_gear(persons, helmets, vests, frame_resized)

                if violation_detected:
                    for no_safety_frame in no_safety_frames:
                        out.write(no_safety_frame)
                    violations_detected = True

        cap.release()
        out.release()

        if violations_detected:
            st.success(f"Violations detected. Video saved as {output_filename}")
            with open(output_path, "rb") as processed_video_file:
                st.download_button("Download processed video", processed_video_file, file_name=output_filename)
        else:
            st.warning("No violations detected in the video.")
