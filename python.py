import cv2
import os
from datetime import datetime
import time

# Initialize the face detector and video capture
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
video_capture = cv2.VideoCapture(0)

# Create a directory for saving captured faces
if not os.path.exists("attending_faces"):
    os.makedirs("attending_faces")

# Dictionary to track attendance (face ID and time of attendance)
attendance_dict = {}
last_detection_time = 0
detection_delay = 5  # Delay in seconds between detections

# Function to detect faces in a frame
def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    return faces

# Function to save the detected face image
def save_face(face_img):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    face_filename = f"attending_faces/face_{timestamp}.jpg"
    cv2.imwrite(face_filename, face_img)
    return face_filename

# Function to mark attendance
def mark_attendance(face_id):
    attendance_dict[face_id] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Face {face_id} marked as attending at {attendance_dict[face_id]}")
    # Write the attendance to the text file
    with open("attendance_records.txt", "a") as f:
        f.write(f"{face_id}: {attendance_dict[face_id]}\n")

# Function to show the attendance on the video feed
def show_attendance_on_frame(video_frame):
    y_offset = 30  # Initial y-offset to start displaying attendance
    for face_id, time_recorded in attendance_dict.items():
        cv2.putText(video_frame, f"{face_id}: {time_recorded}", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        y_offset += 30  # Increase y-offset for the next record

# Main loop to capture video and detect faces
while True:
    result, video_frame = video_capture.read()
    if not result:
        print("Failed to grab frame")
        break

    faces = detect_bounding_box(video_frame)
    current_time = time.time()
    
    for (x, y, w, h) in faces:
        face_roi = video_frame[y:y+h, x:x+w]

        if (current_time - last_detection_time) > detection_delay:
            face_filename = save_face(face_roi)
            face_id = os.path.basename(face_filename).split('.')[0]
            mark_attendance(face_id)
            last_detection_time = current_time

            cv2.putText(video_frame, "Attending", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the live attendance list on the video feed
    show_attendance_on_frame(video_frame)

    # Display the video feed
    cv2.imshow("Face Detection & Attendance", video_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close the OpenCV window
video_capture.release()
cv2.destroyAllWindows()

# Print the final attendance record to console
print("\nAttendance Record:")
for face_id, time_recorded in attendance_dict.items():
    print(f"{face_id}: {time_recorded}")

