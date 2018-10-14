import cv2
import sys

face_cascade = cv2.CascadeClassifier('frontal_face.xml')
video_capture = cv2.VideoCapture(0)

while True:
    retval, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect features specified in Haar Cascade
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(35, 35)
    )

    # Draw a rectangle around recognized face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 200), 2)

    #Display resulting frame
    cv2.imshow('Video', frame)

    # Exit program
    if cv2.waitKey(1) & 0XFF == ord('q'):
        sys.exit()