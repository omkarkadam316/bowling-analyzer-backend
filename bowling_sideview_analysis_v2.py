# ✅ Fully updated bowling_sideview_analysis_v2.py for Render deployment with relative paths

import cv2
import torch
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import os
import tempfile

def process_video(video_path, bowling_arm):
    # Load both YOLO models from root directory
    person_model = YOLO("yolov8l.pt")
    stumps_model = YOLO("side_stumps.pt")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    if bowling_arm == 'right':
        backfoot_chain = [mp_pose.PoseLandmark.RIGHT_HIP,
                          mp_pose.PoseLandmark.RIGHT_KNEE,
                          mp_pose.PoseLandmark.RIGHT_ANKLE]
        frontfoot_chain = [mp_pose.PoseLandmark.LEFT_HIP,
                           mp_pose.PoseLandmark.LEFT_KNEE,
                           mp_pose.PoseLandmark.LEFT_ANKLE]
        release_wrist = mp_pose.PoseLandmark.RIGHT_WRIST
        foot_colors = {'back': (0, 255, 255), 'front': (0, 165, 255)}
    else:
        backfoot_chain = [mp_pose.PoseLandmark.LEFT_HIP,
                          mp_pose.PoseLandmark.LEFT_KNEE,
                          mp_pose.PoseLandmark.LEFT_ANKLE]
        frontfoot_chain = [mp_pose.PoseLandmark.RIGHT_HIP,
                           mp_pose.PoseLandmark.RIGHT_KNEE,
                           mp_pose.PoseLandmark.RIGHT_ANKLE]
        release_wrist = mp_pose.PoseLandmark.LEFT_WRIST
        foot_colors = {'back': (0, 165, 255), 'front': (0, 255, 255)}

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = video_path.replace(".mp4", "_processed.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Pose estimation
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            for chain, color in [(backfoot_chain, foot_colors['back']), (frontfoot_chain, foot_colors['front'])]:
                for i in range(len(chain)-1):
                    p1 = results.pose_landmarks.landmark[chain[i]]
                    p2 = results.pose_landmarks.landmark[chain[i+1]]
                    x1, y1 = int(p1.x * width), int(p1.y * height)
                    x2, y2 = int(p2.x * width), int(p2.y * height)
                    cv2.line(frame, (x1, y1), (x2, y2), color, 3)

            wrist = results.pose_landmarks.landmark[release_wrist]
            cv2.circle(frame, (int(wrist.x * width), int(wrist.y * height)), 8, (255, 0, 0), -1)

        # YOLO Stump Detection
        detections = stumps_model(frame, verbose=False)[0]
        for det in detections.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = det[:6]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

        out.write(frame)

    cap.release()
    out.release()
    return output_path
