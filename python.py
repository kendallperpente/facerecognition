import cv2
import os
import time
import pandas as pd
from datetime import datetime
import face_recognition

# Initialize video capture and face detection
video_capture = cv2.VideoCapture(0)
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Create a directory for saving captured faces
if not os.path.exists("attending_faces"):
    os.makedirs("attending_faces")

# Load pre-trained face encodings (from the 'images' directory)
known_face_encodings = []
known_face_names = []
image_folder = './images/'

# Train face recognizer by loading saved images of known people
for filename in os.listdir(image_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(image_folder, filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(os.path.splitext(filename)[0])  # Using the filename as name

# Initialize attendance dictionary and CSV file
attendance_df = pd.DataFrame(columns=["Name", "Date", "Time"])

last_detection_time = 0
detection_delay = 5  # Delay in seconds between detections

# Function to detect faces
def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    return faces

def mark_attendance(name):
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d")
    time_string = now.strftime("%H:%M:%S")

    # Check if person is already marked as attending today
    if name not in attendance_df["Name"].values:
        attendance_df.loc[len(attendance_df)] = [name, date_string, time_string]
        print(f"Attendance recorded for {name} at {time_string}")

while True:
    result, video_frame = video_capture.read()
    if not result:
        print("Failed to grab frame")
        break

    # Detect faces in the current frame
    faces = detect_bounding_box(video_frame)

    current_time = time.time()

    for (x, y, w, h) in faces:
        cv2.rectangle(video_frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
        face_roi = video_frame[y:y+h, x:x+w]

        # Only save and mark attendance if enough time has passed
        if (current_time - last_detection_time) > detection_delay:
            # Use face_recognition to compare the detected face with known faces
            rgb_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
            face_encoding = face_recognition.face_encodings(rgb_frame, [(y, x + w, y + h, x)])[0]

            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

                # Mark attendance
                mark_attendance(name)

            # Optionally display "Attending" near the face
            cv2.putText(video_frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Update the last detection time to avoid redundant detections
            last_detection_time = current_time

    # Show the video feed with bounding boxes and names
    cv2.imshow("Face Detection & Attendance", video_frame)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Save attendance to CSV file
attendance_df.to_csv("attendance.csv", index=False)
print("\nAttendance has been saved to 'attendance.csv'.")

# Release the video capture and close all windows
video_capture.release()
cv2.destroyAllWindows()
