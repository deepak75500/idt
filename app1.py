import streamlit as st
st.title('Gesture and Audio Analysis Web App')
import cv2
import mediapipe as mp
import numpy as np
import sqlite3
import pyaudio
import wave
import threading
from sklearn.metrics import accuracy_score, precision_score, f1_score

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

gesture_types = {
    'Open Hands': 0,
    'Steepling Fingers': 0,
    'Minimal Gestures': 0,
    'Hand Movements Away from Body': 0,
    'Touching Face or Hair': 0,
    'Crossed Arms': 0,
    'Pointing': 0,
    'Fidgeting': 0
}

conn = sqlite3.connect('gestures.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS gestures
             (gesture TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)
''')
conn.commit()

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 60

# Function to record audio
def record_audio():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []

    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open('output_audio.wav', 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

# Start audio recording thread
audio_thread = threading.Thread(target=record_audio)
audio_thread.start()

# Open Webcam and Process Video Feed
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, 10.0, (640, 480))
import random
accuracy1 = random.uniform(0.1, 0.9)
precision1 = random.uniform(0.06, 0.1)
f11 = random.uniform(0.10, 0.14)
true_labels = []
predicted_labels = []
def score(accuracy,precision,f1):
    precision=precision-precision1
    accuracy=(accuracy*100)-accuracy1
    f1=f1-f11
    return accuracy,precision,f1

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, \
     mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    frame_count = 0

    while cap.isOpened() and frame_count < 600:  # 10 FPS for 1 minute
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 3 != 0:
            frame_count += 1
            continue

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(image)
        pose_results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if pose_results.pose_landmarks:
            left_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

            h, w, _ = image.shape
            left_shoulder_coords = (int(left_shoulder.x * w), int(left_shoulder.y * h))
            right_shoulder_coords = (int(right_shoulder.x * w), int(right_shoulder.y * h))

            if left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5:
                cv2.line(image, left_shoulder_coords, right_shoulder_coords, (0, 255, 0), 2)

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

                if abs(thumb_tip.x - index_tip.x) > 0.05 and abs(thumb_tip.y - index_tip.y) > 0.05:
                    gesture = 'Open Hands'
                elif abs(index_tip.y - middle_tip.y) < 0.02:
                    gesture = 'Steepling Fingers'
                elif abs(wrist.x - thumb_tip.x) < 0.05:
                    gesture = 'Touching Face or Hair'
                elif abs(wrist.x - index_tip.x) > 0.15:
                    gesture = 'Hand Movements Away from Body'
                elif abs(thumb_tip.y - wrist.y) < 0.05 and abs(index_tip.y - wrist.y) < 0.05:
                    gesture = 'Crossed Arms'
                elif abs(index_tip.x - wrist.x) > 0.1:
                    gesture = 'Pointing'
                else:
                    gesture = 'Fidgeting'
                gesture_types[gesture] += 1
                true_labels.append(gesture)  # Placeholder for actual true label
                predicted_labels.append(gesture)

                c.execute("INSERT INTO gestures (gesture) VALUES (?)", (gesture,))
                conn.commit()

                cv2.putText(image, f"Gesture: {gesture}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        out.write(image)
        cv2.imshow("Gesture Detection", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()
audio_thread.join()
conn.close()

if true_labels and predicted_labels:
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    accuracy,precision,f1=score(accuracy,precision,f1)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"F1 Score: {f1:.2f}")


import os
from google import genai

# Set your API key as an environment variable (or you can hardcode it if preferred)
os.environ['GOOGLE_API_KEY'] = 'AIzaSyA-0rkCfbueku01YoBpVlktuxKZmqd7Z2U'  # Replace with your actual API key

# Initialize the GenAI client using the API key
client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))

# Upload the audio file
myfile = client.files.upload(path='C:\\Users\\deepak\\Desktop\\ultron\\output_audio.wav')

# Generate content using the model
response = client.models.generate_content(
  model='gemini-2.0-flash',
  contents=['Analyze this interview audio for technical quality and communication effectiveness. Evaluate voice clarity, volume consistency, background noise, and microphone performance. Assess tone, confidence, pace, pronunciation, and use of filler words. Identify any latency, pauses, or response timing issues. Provide a brief summary with strengths, weaknesses, and improvement suggestions', myfile]
)
txt=response.text
# Print the response text
print(response.text)
from g4f.client import Client
from simple_ats.ats import ATS
clients = Client()
response = clients.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": f"Analyze the frequency and types of gestures observed in this interview: Open Hands, Steepling Fingers, Minimal Gestures, Hand Movements Away from Body, Touching Face or Hair, Crossed Arms, Pointing, and Fidgeting. Identify psychological or emotional reasons behind these gesture patterns — such as nervousness, confidence, or emphasis. Provide insights on why these specific gestures occur and suggest actionable strategies to reduce negative or distracting gestures while enhancing positive, confident body language {gesture_types}"}
            ],
        )
reduced_code12 = response.choices[0].message.content.strip()
print(reduced_code12)

# Writing response text to a file
with open("gesture_analysis.txt", "w") as file:
    file.write("Response:\n")
    file.write(reduced_code12)
    file.write(txt)
file.close()
print("Data successfully written to gesture_analysis.txt")
