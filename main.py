import cv2
import sys
import pickle

face_cascade = cv2.CascadeClassifier('cascades/frontal_face.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('lib/training/trainner.yml')

labels = {}

with open('lib/training/labels.pickle', 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

video_capture = cv2.VideoCapture(0)

while True:
    retval, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #** Detect features specified in Haar Cascade
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(35, 35)
    )

    #** Draw a rectangle around recognized face
    for (x, y, w, h) in faces:
        #*? Recognize face ?? Deep learned model predict (keras, tensorflow, pytorch, scikit)
        #* conf = confidence returned by our prediction model
        id_, conf = recognizer.predict(gray)
        #* Condition that if conf is in between certain value then it is correct
        if conf >= 45 and conf <= 85:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 200), 2)

    #** Display resulting frame
    cv2.imshow('Video', frame)

    #** Exit program
    if cv2.waitKey(1) & 0XFF == ord('q'):
        sys.exit()