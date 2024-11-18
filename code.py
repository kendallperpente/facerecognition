import cv2

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
video_capture = cv2.VideoCapture(0)

def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces

while True:
    result, video_frame = video_capture.read()  # read frames from the video
    if not result:
        print("Failed to grab frame")
        break  # terminate the loop if the frame is not read successfully

    faces = detect_bounding_box(video_frame)  # apply the function to the video frame

    cv2.imshow("My Face Detection Project", video_frame)  # display the processed frame

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
