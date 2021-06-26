import cv2
import os
import numpy as np

people = []

labels = []
features = []

haar = cv2.CascadeClassifier("HAAR_CASCADES\haarcascade_frontalface_default.xml")

path_listed = os.listdir(path="Real Time Face Recognition\webcam_labels")

DIR = r"F:\A.I face recognition\Real Time Face Recognition\webcam_labels"

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
            
            faces = haar.detectMultiScale(gray_img,scaleFactor=1.2,minNeighbors=4)

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

face_recognizer.save(r"Real Time Face Recognition\face_trained_webcam.yml")

np.save(r"Real Time Face Recognition\features_webcam.npy",features)
np.save(r"Real Time Face Recognition\labels_webcam.npy",labels)
