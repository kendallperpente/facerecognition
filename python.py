import cv2
import os
import time
import face_recognition
from datetime import datetime

# Initialize the face detector and video capture
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
video_capture = cv2.VideoCapture(0)

# Create a directory for saving captured faces
if not os.path.exists("attending_faces"):
    os.makedirs("attending_faces")

# Dictionary to store attendance records
attendance_dict = {}

# Define detection delay
last_detection_time = 0
detection_delay = 5  # Delay in seconds between detections

# Get the desktop path
desktop_path = os.path.expanduser("~/Desktop")

# Path to the images folder on the desktop
images_folder = os.path.join(desktop_path, "images")

# Load known faces (update the file paths)
obama_image = face_recognition.load_image_file(os.path.join(images_folder, "obama.jpg"))
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

biden_image = face_recognition.load_image_file(os.path.join(images_folder, "biden.jpg"))
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

# Function to detect bounding boxes for faces
def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    return faces

# Function to save the detected face as an image
def save_face(face_img):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    face_filename = f"attending_faces/face_{timestamp}.jpg"
    cv2.imwrite(face_filename, face_img)
    return face_filename

# Function to mark attendance
def mark_attendance(face_name):
    attendance_dict[face_name] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{face_name} marked as attending at {attendance_dict[face_name]}")

# Function to recognize faces and compare with known faces
def recognize_faces(frame):
    rgb_frame = frame[:, :, ::-1]  # Convert to RGB (face_recognition uses RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Initialize the list of names of recognized faces
    face_names = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        
        face_names.append(name)

    return face_locations, face_names

# Track attendance for known faces
known_faces_in_frame = []

while True:
    result, video_frame = video_capture.read()
    if not result:
        print("Failed to grab frame")
        break

    # Detect faces in the frame
    faces = detect_bounding_box(video_frame)

    # Recognize known faces and update attendance
    face_locations, face_names = recognize_faces(video_frame)

    current_time = time.time()

    # Loop over all faces detected
    for (x, y, w, h), name in zip(faces, face_names):
        cv2.rectangle(video_frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

        # Only save and mark attendance if enough time has passed
        if (current_time - last_detection_time) > detection_delay:
            face_roi = video_frame[y:y+h, x:x+w]
            face_filename = save_face(face_roi)

            if name != "Unknown":
                mark_attendance(name)
            else:
                print("Unknown face detected.")

            last_detection_time = current_time

            # Display the name of the person (optional)
            cv2.putText(video_frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the video feed
    cv2.imshow("Face Detection & Attendance", video_frame)

    # Break when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Mark absent those who were not detected
all_names = set(known_face_names)
detected_names = set(attendance_dict.keys())

# Determine who is absent
absent_names = all_names - detected_names
for name in absent_names:
    attendance_dict[name] = "Absent"

# Save the attendance record to a text file on the desktop
attendance_file = os.path.join(desktop_path, "attendance_record.txt")

with open(attendance_file, "w") as file:
    for name, time_or_status in attendance_dict.items():
        file.write(f"{name}: {time_or_status}\n")

print("\nAttendance Record saved to Desktop.")
print("Attendance Record:")
for name, time_or_status in attendance_dict.items():
    print(f"{name}: {time_or_status}")

# Release the video capture and close all windows
video_capture.release()
cv2.destroyAllWindows()
