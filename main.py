"""
Real-Time Face Counter with Voice Feedback
Author: Rudra 
Description: Detects faces using Haar Cascade,
counts them in real time, logs results to CSV,
and provides voice feedback.
"""

import cv2
import csv
from datetime import datetime
import os
import pyttsx3


def initialize_speech_engine():
    engine = pyttsx3.init('sapi5')  # Windows
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
    return engine


def initialize_csv(file_path):
    if not os.path.exists(file_path):
        with open(file_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Number_of_Faces"])


def main():
    engine = initialize_speech_engine()

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    if face_cascade.empty():
        print("Haar cascade not loaded")
        return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    os.makedirs("data", exist_ok=True)
    csv_file = "data/face_count.csv"
    initialize_csv(csv_file)

    last_num_faces = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(60, 60)
        )

        num_faces = len(faces)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(frame, f"Faces: {num_faces}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Log to CSV
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), num_faces])

        # Voice feedback only if changed
        if num_faces != last_num_faces:
            if num_faces == 0:
                engine.say("No faces detected")
            elif num_faces == 1:
                engine.say("One face detected")
            else:
                engine.say(f"{num_faces} faces detected")

            engine.runAndWait()
            last_num_faces = num_faces

        cv2.imshow("Real-Time Face Counter", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

