import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Initialize Mediapipe Modules
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Rolling window buffer for stable detection
posture_buffer = deque(maxlen=5)  # Store last 5 posture states
yawn_buffer = deque(maxlen=10)  # Store last 10 yawning states

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

# Function to get stable posture status
def get_stable_posture_status(current_status):
    posture_buffer.append(current_status)
    if posture_buffer.count("Incorrect Posture") >= 4:
        return "Incorrect Posture", (0, 0, 255)  # Red
    else:
        return "Proper Standing", (0, 255, 0)  # Green

# Function to detect yawning
def detect_yawning(face_landmarks, image):
    h, w, _ = image.shape

    # Get upper and lower lip coordinates
    upper_lip = face_landmarks.landmark[13]  # Upper lip mid
    lower_lip = face_landmarks.landmark[14]  # Lower lip mid

    # Convert to pixel coordinates
    upper_lip_y = int(upper_lip.y * h)
    lower_lip_y = int(lower_lip.y * h)

    # Compute mouth opening distance
    lip_distance = abs(lower_lip_y - upper_lip_y)

    # Yawning detection threshold (adjustable)
    YAWN_THRESHOLD = 18

    # Rolling window buffer for stability
    yawn_buffer.append(lip_distance > YAWN_THRESHOLD)

    # If 7 out of last 10 frames detect yawning → Confirm Yawn
    if yawn_buffer.count(True) >= 7:
        return "Yawning", (0, 255, 255)  # Yellow
    else:
        return "Not Yawning", (255, 255, 255)  # White

# Open Webcam and Process Video Feed
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(image)
        face_results = face_mesh.process(image)

        # Convert back to BGR for OpenCV display
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Check and visualize posture if landmarks are detected
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            current_status = check_posture(pose_results.pose_landmarks.landmark)
            stable_status, posture_color = get_stable_posture_status(current_status)

            # Display posture status
            cv2.putText(image, stable_status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, posture_color, 2)

        # Check yawning detection
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(image, face_landmarks, mp.solutions.face_mesh.FACEMESH_TESSELATION)
                yawn_status, yawn_color = detect_yawning(face_landmarks, image)

                # Display yawning status
                cv2.putText(image, yawn_status, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, yawn_color, 2)

        cv2.imshow("Posture & Yawning Detection", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
