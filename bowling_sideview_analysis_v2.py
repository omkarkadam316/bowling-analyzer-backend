import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import os

def process_video(input_video_path, bowling_arm):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Set keypoints and colors based on arm
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

    # Load YOLO model
    yolo_model = YOLO("side_stumps.pt")

    # Setup video reading and writing
    cap = cv2.VideoCapture(input_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = input_video_path.replace(".mp4", "_processed.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            # Draw back and front foot chains
            for chain, color in [(backfoot_chain, foot_colors['back']), (frontfoot_chain, foot_colors['front'])]:
                for i in range(len(chain)-1):
                    p1 = results.pose_landmarks.landmark[chain[i]]
                    p2 = results.pose_landmarks.landmark[chain[i+1]]
                    x1, y1 = int(p1.x * width), int(p1.y * height)
                    x2, y2 = int(p2.x * width), int(p2.y * height)
                    cv2.line(frame, (x1, y1), (x2, y2), color, 3)

            # Draw release wrist
            wrist = results.pose_landmarks.landmark[release_wrist]
            cv2.circle(frame, (int(wrist.x * width), int(wrist.y * height)), 8, (255, 0, 0), -1)

        # Detect stumps using YOLO
        detections = yolo_model(frame, verbose=False)[0]
        for det in detections.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = map(int, det[:6])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        out.write(frame)

    cap.release()
    out.release()
    return output_path