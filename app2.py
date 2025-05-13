import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Rolling window buffer for stable detection
frame_buffer = deque(maxlen=5)  # Store last 5 posture states

# Function to check standing posture
def check_posture(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y
    nose = landmarks[mp_pose.PoseLandmark.NOSE].y

    # Compute posture checks
    shoulder_diff = abs(left_shoulder - right_shoulder)
    hip_diff = abs(left_hip - right_hip)
    head_above_shoulders = nose < (left_shoulder + right_shoulder) / 2  # Head should be above shoulders
    straight_back = abs(left_shoulder - left_hip) > 0.3 and abs(right_shoulder - right_hip) > 0.3  # Ensures back is straight

    # Thresholds
    SHOULDER_THRESHOLD = 0.02
    HIP_THRESHOLD = 0.02

    # Proper standing posture conditions
    if shoulder_diff < SHOULDER_THRESHOLD and hip_diff < HIP_THRESHOLD and head_above_shoulders and straight_back:
        return "Proper Standing"
    else:
        return "Incorrect Posture"

# Function to get the smoothed posture status
def get_stable_posture_status(current_status):
    frame_buffer.append(current_status)  # Add current prediction to buffer
    if frame_buffer.count("Incorrect Posture") >= 4:  # At least 4 out of last 5 must be incorrect
        return "Incorrect Posture", (0, 0, 255)  # Red
    else:
        return "Proper Standing", (0, 255, 0)  # Green

# Open Webcam and Process Video Feed
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # Convert back to BGR for OpenCV display
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Check and visualize posture if landmarks are detected
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            current_status = check_posture(results.pose_landmarks.landmark)
            stable_status, color = get_stable_posture_status(current_status)

            # Display posture status
            cv2.putText(image, stable_status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Posture Detection", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
