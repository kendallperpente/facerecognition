import cv2
import face_recognition

# Load the image of Obama
obama_image = face_recognition.load_image_file("obama.jpg")

# Convert the image from BGR (OpenCV default) to RGB (face_recognition default)
obama_image_rgb = cv2.cvtColor(obama_image, cv2.COLOR_BGR2RGB)

# Get face encodings
obama_face_encoding = face_recognition.face_encodings(obama_image_rgb)[0]

# Alternatively, using OpenCV for detecting faces:
video_capture = cv2.VideoCapture(0)

# Define the face classifier (Haar Cascade for frontal face)
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = video_capture.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame from BGR to RGB
    rgb_frame = frame[:, :, ::-1]  # OpenCV (BGR) to face_recognition (RGB)

    # Find face locations and face encodings
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare face encoding with known encodings (Obama's face encoding here)
        matches = face_recognition.compare_faces([obama_face_encoding], face_encoding)

        name = "Unknown"
        if True in matches:
            name = "Barack Obama"

        # Draw bounding box and name on the frame
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the video feed
    cv2.imshow("Video", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
video_capture.release()
cv2.destroyAllWindows()
