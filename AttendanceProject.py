import cv2
import numpy as np
import face_recognition
import os
from pathlib import Path
from datetime import datetime
import csv

BASE_DIR = Path(__file__).resolve().parent
KNOWN_DIR = BASE_DIR / "data" / "known"
ATTENDANCE_FILE = BASE_DIR / "Attendance.csv"

def ensure_attendance_file():
    if not ATTENDANCE_FILE.exists():
        with open(ATTENDANCE_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "timestamp"])

def mark_attendance(name):
    existing_names = set()

    with open(ATTENDANCE_FILE, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row:
                existing_names.add(row[0])

    if name not in existing_names:
        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        with open(ATTENDANCE_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([name, now])

def load_known_faces():
    known_encodings = []
    known_names = []

    if not KNOWN_DIR.exists():
        raise FileNotFoundError(f"Known faces directory not found: {KNOWN_DIR}")

    for person_name in os.listdir(KNOWN_DIR):
        person_path = KNOWN_DIR / person_name

        if not person_path.is_dir():
            continue

        for file_name in os.listdir(person_path):
            file_path = person_path / file_name

            img = cv2.imread(str(file_path))
            if img is None:
                print(f"Could not read image: {file_path}")
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb)

            if len(encodings) == 1:
                known_encodings.append(encodings[0])
                known_names.append(person_name)
            else:
                print(f"Skipped {file_path} because it has {len(encodings)} faces")
    return known_encodings, known_names

ensure_attendance_file()
known_encodings, known_names = load_known_faces()

print(f"Loaded {len(known_encodings)} face encodings.")
print("Encoding complete!")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

threshold = 0.50

try:
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read frame from webcam")
            break

        img_small = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        img_small_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

        faces_current_frame = face_recognition.face_locations(img_small_rgb)
        encodes_current_frame = face_recognition.face_encodings(img_small_rgb, faces_current_frame)

        for encode_face, face_loc in zip(encodes_current_frame, faces_current_frame):
            face_distances = face_recognition.face_distance(known_encodings, encode_face)

            if len(face_distances) == 0:
                continue

            match_index = np.argmin(face_distances)
            best_distance = face_distances[match_index]

            top, right, bottom, left = face_loc
            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4

            if best_distance < threshold:
                match_name = known_names[match_index].upper()

                cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, match_name, (left + 6, bottom - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                mark_attendance(match_name)
            else:
                cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(img, "UNKNOWN", (left + 6, bottom - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Webcam", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()