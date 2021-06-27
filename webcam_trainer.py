import cv2
import os
import numpy as np
import pathlib

people = []

labels = []
features = []

current_dir = pathlib.Path(__file__).absolute()
current_dir = str(current_dir)
current_dir = current_dir.split("w")
current_dir = current_dir[0]

haar = cv2.CascadeClassifier(f"{current_dir}\HAAR_CASCADES\haarcascade_frontalface_default.xml")

path_listed = os.listdir(path=f"{current_dir}\webcam_labels")

DIR = f"{current_dir}\webcam_labels"

for item_person in path_listed:
    people.append(item_person)

def training():
    for person in people:
        path = os.path.join(DIR,person)
        label = people.index(person)
    
        for img in os.listdir(path=path):
            img_path = os.path.join(path,img)

            image = cv2.imread(img_path)
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            faces = haar.detectMultiScale(gray_img,scaleFactor=1.1,minNeighbors=4)

            for (x, y, w, h) in faces:
                faces_roi = gray_img[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

training()

print("Training Done")
print(f"Labels {len(labels)}")


features = np.array(features,dtype="object")
labels = np.array(labels)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

face_recognizer.train(features, labels)

pier = f"{current_dir}\ face_trained_webcam.yml"

face_recognizer.save(pier)

np.save(f"{current_dir}features_webcam.npy",features)
np.save(f"{current_dir}labels_webcam.npy",labels)
