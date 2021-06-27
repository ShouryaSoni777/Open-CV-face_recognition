import cv2
import numpy as np
import os
import pathlib

people = []
current_dir = pathlib.Path(__file__).absolute()
current_dir = str(current_dir)
current_dir = current_dir.split("w")
current_dir = current_dir[0]

haar = cv2.CascadeClassifier(f"{current_dir}\HAAR_CASCADES\haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

features = np.load(f"{current_dir}\\features_webcam.npy",allow_pickle=True)
labels = np.load(f"{current_dir}\labels_webcam.npy",allow_pickle=True)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
path = f"{current_dir}\ face_trained_webcam.yml"
face_recognizer.read(path)

path_listed = os.listdir(path=f"{current_dir}\webcam_labels")

for item_person in path_listed:
    people.append(item_person)

while True:
    ret, frames = cap.read()

    gray_img = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    face_rects = haar.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=6)

    for (x, y, w, h) in face_rects:

        face_rectangle = gray_img[y:y+h, x:x+y]

        label, confidence = face_recognizer.predict(face_rectangle)

        if confidence < 100:
            cv2.putText(frames, str(people[label]), (x+w-120,y+h+20), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255,255,0),thickness=1)
            cv2.rectangle(frames, (x,y), (x+w,y+h), (255, 255, 0),thickness=2)
            print(f"{people[label]} with a confidence of {confidence}")


    cv2.imshow("detected faces", frames)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
