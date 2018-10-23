import os
import json
import pickle
import cv2
import numpy as np
from PIL import Image

class ImageTrain:

    def __init__(self):
        self.x_train = []
        self.y_labels = []
        self.current_id = 0
        self.label_ids = {}

    def train(self):
        face_cascade = cv2.CascadeClassifier('../../cascades/frontal_face.xml')
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        """
            [#] Verifying the image.
            [#] Turning the Image into GrayScale
            [#] Turning image into a Numpy Array
        """
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        image_dir = os.path.join(BASE_DIR, 'images')

        for root, dirs, files, in os.walk(image_dir):
            for file in files:
                if file.endswith('png') or file.endswith('jpeg') or file.endswith('jpg'):
                    path = os.path.join(root, file)
                    label = os.path.basename(os.path.dirname(path)).replace(' ', '-').lower()
                    pil_image = Image.open(path).convert('L') #* .convert converts the image into GrayScale
                    size = (500, 500)
                    final_image = pil_image.resize(size, Image.ANTIALIAS)
                    image_array = np.array(final_image, 'uint8')
                    # print(image_array)

                    if not label in self.label_ids:
                        self.label_ids[label] = self.current_id
                        self.current_id += 1

                    id_ = self.label_ids[label]
                    # print(self.label_ids)
                    faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.1, minNeighbors=5, minSize=(35, 35))
                    for (x, y, w, h) in faces:
                        roi = image_array[y: y + h, x: x + w]
                        self.x_train.append(roi)
                        self.y_labels.append(id_)

        # print(self.y_labels)
        # print(self.x_train)
        with open('labels.pickle', 'wb') as f:
            pickle.dump(self.label_ids, f)

        recognizer.train(self.x_train, np.array(self.y_labels))
        recognizer.save('trainner.yml')

it = ImageTrain()
it.train()