import cv2
import face_recognition
import os
import time
from datetime import datetime

# Load the known images and encode the faces
obama_image = face_recognition.load_image_file("obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

biden_image = face_recognition.load_image_file("biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding
]
known_face_names = [
    "Barack Obama",
    "Joe Biden"
]

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

# Create a dictionary to track attendance
attendance_dict = {}
last_detection_time = 0
detection_delay = 5  # Delay in seconds between detections

# Function to mark attendance
def mark_attendance(person_name):
    attendance_dict[person_name] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{person_name} marked as attending at {attendance_dict[person_name]}")
    # Optionally write to a text file for attendance log
    with open("attendance_records.txt", "a") as f:
        f.write(f"{person_name}: {attendance_dict[person_name]}\n")

# Main loop to capture frames and recognize faces
while True:
    # Capture a frame from the video feed
    ret, frame = video_capture.read()

    # Resize the frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image to RGB (face_recognition expects RGB images)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # Loop over each face found in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the detected face matches any known face encoding
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # If a match was found, use the known face name
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

            # Mark attendance for the recognized person
            mark_attendance(name)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Label the face with the name
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Display the resulting image
    cv2.imshow('Face Recognition Attendance System', frame)

    # Wait for a key press to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close all windows
video_capture.release()
cv2.destroyAllWindows()

# Print the attendance records
print("\nAttendance Record:")
for person_name, time_recorded in attendance_dict.items():
    print(f"{person_name}: {time_recorded}")
